"""
Class to detect which kind of grid is used in a xarray dataset or dataarray
Possible matches are: Regular, GaussianRegular, GaussianReduced, Curvilinear, HEALPix, Unstructured
"""
import numpy as np
import xarray as xr
from .util import find_coord

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

        dimsdict = data.dims if isinstance(data, xr.Dataset) else dict(zip(data.dims, data.shape))

        for dim_name, size in dimsdict.items():
            n = size // 12
            is_pow2 = size % 12 == 0 and n > 0 and (n & (n - 1)) == 0
            if is_pow2 and (dim_name.lower() in candidate_names or len(dimsdict) == 1):
                return dim_name

        return None
