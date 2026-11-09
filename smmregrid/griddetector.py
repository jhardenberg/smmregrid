"""
Class to detect which kind of grid is used in a xarray dataset or dataarray
Possible matches are: Regular, GaussianRegular, GaussianReduced, Curvilinear, HEALPix, Unstructured
"""
import numpy as np
import xarray as xr
from .cdogrid import CdoGrid
from .util import find_coord

# Tolerance (degrees) on the coordinates compared with the CDO grids: float32 coordinates near 360 have ~3e-5 resolution
COORD_ATOL = 1e-4

# Define coordinate names for latitude and longitude
LAT_COORDS = ["lat", "latitude", "nav_lat"]
LON_COORDS = ["lon", "longitude", "nav_lon"]


class GridDetector:
    """
    Class to detect the grid type of a given xarray dataset or dataarray.
    
    Args:
        lon (str): The name of the longitude coordinate. Default is "lon". Expand the list of possible names from LON_COORDS.
        lat (str): The name of the latitude coordinate. Default is "lat". Expand the list of possible names from LAT_COORDS.
    """

    def __init__(self, lon="lon", lat="lat"):

        self.lon_coords = set(LON_COORDS + [lon])
        self.lat_coords = set(LAT_COORDS + [lat])

    def detect_grid(self, data):
        """
        Detect the grid type based on the structure of the data.

        Args:
            data (xr.Dataset or xr.DataArray): The input dataset.

        Returns:
            str: The identified grid type:
                    "Regular", "GaussianRegular", "GaussianReduced",
                    "Curvilinear", "HEALPix", "Unstructured".
        """

        # Explicit metadata always wins
        if self.is_healpix_from_attribute(data):
            return "HEALPix"

        # Structural HEALPix check — independent of lat/lon coords
        if self._find_healpix_dim(data) is not None:
            return "HEALPix"

        # Find latitude and longitude coordinates
        lat = find_coord(data, self.lat_coords)
        lon = find_coord(data, self.lon_coords)

        if not lat or not lon:
            return "Unknown"

        # 2D coord-dim dependency
        if data[lat].ndim == 2 and data[lon].ndim == 2:
            return "Curvilinear"

        # 1D coord-dim dependency
        if data[lat].ndim == 1 and data[lon].ndim == 1:

            # Regular: latitude and longitude depend on different coordinates
            if data[lat].dims != data[lon].dims:

                lat_diff = np.diff(data[lat].values)
                lon_diff = np.diff(data[lon].values)

                # Regular: latitude and longitude equidistant
                if np.allclose(lat_diff, lat_diff[0]) and np.allclose(lon_diff, lon_diff[0]):
                    return "Regular"

                # Gaussian: longitude equidistant, latitude not
                if not np.allclose(lat_diff, lat_diff[0]) and np.allclose(lon_diff, lon_diff[0]):
                    return "GaussianRegular"

                return "UndefinedRegular"

            # Guess gaussian reduced: increasing number of latitudes from -90 to 0
            lat_values = data[lat].where(data[lat] < 0).values
            lat_values = lat_values[~np.isnan(lat_values)]
            _, counts = np.unique(lat_values, return_counts=True)
            gaussian_reduced = np.all(np.diff(counts) > 0)
            if gaussian_reduced:
                return "GaussianReduced"

            # None of the above cases
            return "Unstructured"

        return "Unknown"

    def detect_cdo_grid(self, data, kind=None):
        """
        Return the CDO grid name (e.g. 'r360x180', 'F128', 'O96', 'hp32_nested') matching the data.

        The name is returned only when the coordinates are exactly the ones CDO generates for it,
        including the order of the points: lon starting at 0 (or at -180 + dlon/2 for 'global_X'),
        lat south to north for the lonlat grids and north to south for the Gaussian ones.
        Any other layout (shifted longitudes, flipped latitudes, regional grids) returns None,
        since weights built from the CDO name would index the wrong points.

        Args:
            data (xr.Dataset or xr.DataArray): The input dataset.
            kind (str, optional): The grid type from `detect_grid`. Computed if not given.

        Returns:
            str or None: The CDO grid name, or None if there is no exact match.
        """
        kind = kind or self.detect_grid(data)
        finder = {
            "Regular": self._cdo_regular,
            "GaussianRegular": self._cdo_gaussian_regular,
            "GaussianReduced": self._cdo_gaussian_reduced,
            "HEALPix": self._cdo_healpix,
        }.get(kind)
        if finder is None:
            return None
        grid_str = finder(data)
        if grid_str is None or CdoGrid(grid_str).grid_kind is None:
            return None
        return grid_str

    def _latlon(self, data):
        """Latitude and longitude values as float64 arrays, or (None, None)"""
        lat = find_coord(data, self.lat_coords)
        lon = find_coord(data, self.lon_coords)
        if not lat or not lon:
            return None, None
        return data[lat].values.astype(np.float64), data[lon].values.astype(np.float64)

    @staticmethod
    def _match(values, expected):
        return values.shape == expected.shape and np.allclose(values, expected, rtol=0, atol=COORD_ATOL)

    @staticmethod
    def _gaussian_latitudes(nlat):
        """Gaussian latitudes in degrees, north to south as in CDO"""
        roots, _ = np.polynomial.legendre.leggauss(nlat)
        return np.degrees(np.arcsin(roots))[::-1]

    def _cdo_regular(self, data):
        """rNxM: lon from 0, lat south to north, centred on the cells for even M and with the poles for odd M.
        global_X: lon from -180 + X/2, lat centred on the cells."""
        lat, lon = self._latlon(data)
        if lat is None:
            return None
        nlat, nlon = lat.size, lon.size
        if nlat < 2:
            return None
        dlon = 360 / nlon
        centred_lat = -90 + (np.arange(nlat) + 0.5) * 180 / nlat
        if self._match(lon, np.arange(nlon) * dlon):
            exp_lat = centred_lat if nlat % 2 == 0 else -90 + np.arange(nlat) * 180 / (nlat - 1)
            return f"r{nlon}x{nlat}" if self._match(lat, exp_lat) else None
        if self._match(lon, -180 + (np.arange(nlon) + 0.5) * dlon) and np.isclose(dlon, 180 / nlat) \
                and self._match(lat, centred_lat):
            return f"global_{dlon:g}"
        return None

    def _cdo_gaussian_regular(self, data):
        """FN: 2N Gaussian latitudes north to south, 4N longitudes from 0"""
        lat, lon = self._latlon(data)
        if lat is None or lat.size % 2 or lon.size != 2 * lat.size:
            return None
        if self._match(lat, self._gaussian_latitudes(lat.size)) and \
                self._match(lon, np.arange(lon.size) * 360 / lon.size):
            return f"F{lat.size // 2}"
        return None

    def _cdo_gaussian_reduced(self, data):
        """ON: 2N Gaussian rings north to south with 20 + 4i points from each pole, longitudes from 0.
        Classic reduced grids (NN) use ECMWF tables of points per ring and are not mapped."""
        lat, lon = self._latlon(data)
        if lat is None or lat.ndim != 1:
            return None
        rings, start, counts = np.unique(-lat, return_index=True, return_counts=True)
        nlat = rings.size
        if nlat % 2:
            return None
        half = 20 + 4 * np.arange(nlat // 2)
        if not np.array_equal(counts, np.concatenate([half, half[::-1]])):
            return None
        if not self._match(-rings, self._gaussian_latitudes(nlat)) or not np.array_equal(start, np.cumsum(counts) - counts):
            return None
        exp_lon = np.concatenate([np.arange(n) * 360 / n for n in counts])
        if self._match(lon, exp_lon):
            return f"O{nlat // 2}"
        return None

    def _cdo_healpix(self, data):
        """hpN_nested / hpN_ring: nside from the size of the HEALPix dimension, order from the
        metadata (CDO 'indexing_scheme', or 'healpix_order') or else from the latitudes, which
        never increase with the ring ordering. No order information returns None."""
        dim = self._find_healpix_dim(data)
        if dim is None:
            # detected from the attributes: take the only dimension with a HEALPix size
            dims = [d for d, size in data.sizes.items() if self._healpix_nside(size)]
            if len(dims) != 1:
                return None
            dim = dims[0]
        nside = self._healpix_nside(data.sizes[dim])
        order = self._healpix_order(data)
        if order is None:
            lat = find_coord(data, self.lat_coords)
            if lat is None or data[lat].dims != (dim,):
                return None
            order = "ring" if np.all(np.diff(data[lat].values) <= 0) else "nested"
        return f"hp{nside}_{order}"

    @staticmethod
    def _healpix_nside(size):
        """nside for a number of cells 12 * nside**2 with nside a power of two, else None"""
        if size % 12:
            return None
        nside = int(round(np.sqrt(size // 12)))
        if nside > 0 and nside * nside * 12 == size and not nside & (nside - 1):
            return nside
        return None

    @staticmethod
    def _healpix_order(data):
        """'nested' or 'ring' from the grid mapping variable, else None"""
        variables = list(data.variables.values()) if isinstance(data, xr.Dataset) else [data] + list(data.coords.values())
        for var in variables:
            if var.attrs.get("grid_mapping_name") != "healpix":
                continue
            order = str(var.attrs.get("indexing_scheme", var.attrs.get("healpix_order", ""))).lower()
            if order in ("nested", "nest"):
                return "nested"
            if order == "ring":
                return "ring"
        return None

    @staticmethod
    def is_healpix_from_attribute(data):
        """
        Determine if the given xarray Dataset or DataArray uses a HEALPix grid.

        Returns:
            bool: True if HEALPix grid detected, False otherwise.
        """

        # Attribute-based checks
        if isinstance(data, xr.Dataset):
            if "healpix" in data.variables:
                return True
            for var in data.data_vars:
                if data[var].attrs.get('grid_mapping') == 'healpix':
                    return True
        elif isinstance(data, xr.DataArray):
            if data.attrs.get('grid_mapping') == 'healpix':
                return True

        return False

    @staticmethod
    def _find_healpix_dim(data):
        """
        Look for a dimension whose size matches 12 * nside**2 for some
        power-of-two nside, checking dimension names/sizes directly rather
        than relying on lat/lon coordinates being present.
        """
        # Common names used for the HEALPix cell-index dimension
        candidate_names = {"cell", "cells", "pix", "pixel", "healpix", "ncells"}

        dimsdict = data.sizes

        for dim_name, size in dimsdict.items():
            n = size // 12
            is_pow2 = size % 12 == 0 and n > 0 and (n & (n - 1)) == 0
            if is_pow2 and (dim_name.lower() in candidate_names or len(dimsdict) == 1):
                return dim_name

        return None
