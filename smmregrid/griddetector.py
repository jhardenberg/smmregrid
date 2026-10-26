"""
Class to detect which kind of grid is used in a xarray dataset or dataarray
Possible matches are: Regular, GaussianRegular, GaussianReduced, Curvilinear, HEALPix, Unstructured
"""
import math
import numpy as np
import xarray as xr
from .gridcoordinates import GridCoordinates, LAT_COORDS, LON_COORDS  # noqa: F401
from .gridtype import DEFAULT_DIMS

# relative tolerances on the grid spacing: Gaussian latitudes deviate from equidistant by ~1e-2,
# so equidistant coordinates must be closer than that, while Gaussian ones are compared directly
EQUIDISTANT_TOLERANCE = 1e-3
GAUSSIAN_TOLERANCE = 1e-2


class GridDetector:
    """
    Class to detect the grid type of a given xarray dataset or dataarray.
    
    Args:
        lon (str): The name of the longitude coordinate, preferred over the detection done by GridCoordinates
                   (CF attributes, then the names in LON_COORDS).
        lat (str): The name of the latitude coordinate, preferred over the detection done by GridCoordinates
                   (CF attributes, then the names in LAT_COORDS).
    """

    def __init__(self, lon=None, lat=None):

        self.lon = lon
        self.lat = lat

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
        coords = GridCoordinates(data.coords, lat=self.lat, lon=self.lon)
        lat, lon = coords.lat, coords.lon

        if not lat or not lon:
            return "Unknown"

        # 2D coord-dim dependency
        if data[lat].ndim == 2 and data[lon].ndim == 2:
            return "Curvilinear"

        # 1D coord-dim dependency
        if data[lat].ndim == 1 and data[lon].ndim == 1:

            lat_values = self._to_degrees(data[lat])
            lon_values = self._to_degrees(data[lon])

            # Regular: latitude and longitude depend on different coordinates
            if data[lat].dims != data[lon].dims:

                if self._is_equidistant(lon_values, period=360):
                    # Regular: latitude and longitude equidistant
                    if self._is_equidistant(lat_values):
                        return "Regular"
                    # Gaussian: longitude equidistant, latitude on the Gaussian latitudes
                    if self._is_gaussian(lat_values):
                        return "GaussianRegular"

                return "UndefinedRegular"

            # Gaussian reduced: points along the Gaussian latitudes, a few latitudes for many points
            unique_lat = np.unique(lat_values)
            if unique_lat.size ** 2 <= 4 * lat_values.size and self._is_gaussian(unique_lat):
                return "GaussianReduced"

            # None of the above cases
            return "Unstructured"

        return "Unknown"

    @staticmethod
    def _to_degrees(coord):
        """Return the coordinate values in degrees, converting from radians if needed"""
        values = np.asarray(coord.values, dtype=float)
        if coord.attrs.get('units') in ('radian', 'radians', 'rad'):
            return np.rad2deg(values)
        return values

    @staticmethod
    def _is_equidistant(values, period=None):
        """
        Check if 1D coordinate values are equidistant, within EQUIDISTANT_TOLERANCE of the spacing.
        With a period (e.g. 360 for longitude), a jump across the period (e.g. the dateline) is allowed.
        A single value is equidistant.
        """
        if values.size < 2:
            return True
        diffs = np.diff(values)
        if period is not None:
            diffs = np.mod(diffs, period)
        spacing = np.mean(diffs)
        if spacing == 0:
            return False
        return bool(np.all(np.abs(diffs - spacing) <= EQUIDISTANT_TOLERANCE * abs(spacing)))

    @staticmethod
    def _is_gaussian(lat_values):
        """
        Check if latitudes (in degrees) are the Gaussian latitudes of a global grid,
        within GAUSSIAN_TOLERANCE of the mean spacing.
        """
        lat_values = np.sort(lat_values)
        nlat = lat_values.size
        if nlat < 2:
            return False
        gaussian = gaussian_latitudes(nlat)
        return bool(np.max(np.abs(lat_values - gaussian)) <= GAUSSIAN_TOLERANCE * 180. / nlat)

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
        A dimension with another name is accepted only if it is the only one, and not a time dimension.
        """
        # Common names used for the HEALPix cell-index dimension
        candidate_names = {"cell", "cells", "pix", "pixel", "healpix", "ncells"}

        dimsdict = data.sizes

        for dim_name, size in dimsdict.items():
            if not is_healpix_size(size):
                continue
            if dim_name.lower() in candidate_names:
                return dim_name
            if len(dimsdict) == 1 and dim_name not in DEFAULT_DIMS['time']:
                return dim_name

        return None


def is_healpix_size(size):
    """Check if size is 12 * nside**2 with nside a power of two"""
    if size % 12 != 0:
        return False
    npix_face = size // 12
    nside = math.isqrt(npix_face)
    return nside > 0 and nside * nside == npix_face and (nside & (nside - 1)) == 0


def gaussian_latitudes(nlat):
    """
    Gaussian latitudes in degrees, from south to north: the zeros of the Legendre polynomial
    of degree nlat in sin(latitude), found with Newton iterations.
    """
    x = np.cos(np.pi * (np.arange(nlat, 0, -1) - 0.25) / (nlat + 0.5))
    for _ in range(100):
        p_prev, p_cur = np.ones_like(x), x
        for k in range(2, nlat + 1):
            p_prev, p_cur = p_cur, ((2 * k - 1) * x * p_cur - (k - 1) * p_prev) / k
        step = p_cur * (x ** 2 - 1) / (nlat * (x * p_cur - p_prev))
        x = x - step
        if np.max(np.abs(step)) < 1e-14:
            break
    return np.rad2deg(np.arcsin(x))
