"""Utility module with various support function"""

import os
import warnings
import xarray
import numpy as np
from .cdogrid import CdoGrid

# Define coordinate names for latitude and longitude
LAT_COORDS = ["lat", "latitude", "nav_lat"]
LON_COORDS = ["lon", "longitude", "nav_lon"]


def check_gridfile(filename):
    """Check if a grid is a file, a cdo string or xarray object"""

    if filename is None:
        return None
    if isinstance(filename, (xarray.Dataset, xarray.DataArray)):
        return "xarray"
    if isinstance(filename, str):
        if CdoGrid(filename).grid_kind:
            return "grid"
        if os.path.exists(filename):
            return "file"
        raise FileNotFoundError(f"Cannot find {filename} on disk")

    raise TypeError(f'Unsuported format for {filename}')


def deprecated_argument(old, new, oldname='var1', newname='var2'):
    """Utility to provide warning in case of deprecated argument"""

    # Check for deprecated 'old' argument
    if old is not None:
        warnings.warn(
            f"{oldname} is deprecated and will be removed in future versions. "
            f"Please use {newname} instead.",
            DeprecationWarning
        )
        # If new is not provided, use the value from old
        if new is None:
            new = old
    return new

def find_coord(ds, possible_names):
    """Find the first matching coordinate in the dataset."""
    return next((name for name in possible_names if name in ds.coords), None)

def detect_grid(data):
    """Classify the grid type based on coordinate structure."""

    lat = find_coord(data, LAT_COORDS)
    lon = find_coord(data, LON_COORDS)

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
            if np.allclose(lat_diff, lat_diff[0]) and np.allclose(lon_diff, lon_diff[0]):
                return "Regular"

            # Gaussian: second derivative of latitude is positive from -90 to 0
            lat_values = data[lat].where(data[lat]<0).values
            lat_values=lat_values[~np.isnan(lat_values)]
            gaussian = np.all(np.diff(lat_values, n=2) > 0)
            if gaussian:
                return "GaussianRegular"
            
            return "UndefinedRegular"

        # Healpix: number of pixels is a multiple of 12 and log2(pix / 12) is an integer
        pix = data[lat].size
        if pix % 12 == 0 and (pix // 12).bit_length() - 1 == np.log2(pix // 12):
            return "Healpix"
        
        # Guess gaussian reduced: increasing number of latitudes from -90 to 0
        lat_values = data[lat].where(data[lat]<0).values
        lat_values=lat_values[~np.isnan(lat_values)]
        _, counts = np.unique(lat_values, return_counts=True)
        gaussian_reduced = np.all(np.diff(counts)>0)
        if gaussian_reduced:
            return "GaussianReduced"

        # None of the above cases
        return "Unstructured"
