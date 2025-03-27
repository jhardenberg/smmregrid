"""Utility module with various support function"""

import re
import os
import warnings
import xarray
#import numpy as np

# Define CDO regex patterns for each grid type (updated at CDO 2.4.4)
# Define regex patterns for each grid type
CDO_GRID_PATTERNS = {
    "global_regular": re.compile(r"^global_\d+(\.\d+)?$"),                    
    "regional_regular": re.compile(r"^dcw:[A-Z]{2,4}(?:_\d+(\.\d+)?)?$"),     
    "zonal_latitudes": re.compile(r"^zonal_\d+(\.\d+)?$"),                    
    "global_regular_NxM": re.compile(r"^r\d+x\d+$"),                          
    "one_grid_point": re.compile(r"^lon=(-?\d+(\.\d+)?)/lat=(-?\d+(\.\d+)?)$"),
    "gaussian_grid_F": re.compile(r"^F\d+$"),      
    "gaussian_grid_n": re.compile(r"^n\d+$"),                            
    "icosahedral_gme": re.compile(r"^gme\d+$"),                           
    "healpix_grid": re.compile(r"^hp\d+(?:_(nested|ring))?$"),             
    "healpix_zoom": re.compile(r"^hpz\d+$")                       
}

def is_cdo_grid(grid_str):
    """Check if the input string matches any CDO grid type."""

    # Check if the string matches any of the grid patterns
    for grid_type, pattern in CDO_GRID_PATTERNS.items():
        if pattern.match(grid_str):
            return True

    # Return False if no patterns match
    return False

def check_gridfile(filename):
    """Check if a grid is a file, a cdo string or xarray object"""

    if filename is None:
        return None
    if isinstance(filename, (xarray.Dataset, xarray.DataArray)):
        return "xarray"
    if isinstance(filename, str):
        if is_cdo_grid(filename):
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


def is_healpix_grid(ds):
    """Check if the dataset has a HEALPix-compatible number of grid points."""
    n_points = ds.sizes.get("grid", ds.sizes.get("cell", None))  # Adjust based on the dataset structure
    if n_points is None:
        return False  # No appropriate coordinate found
    
    # Solve for nside: nside = sqrt(n_pix / 12)
    nside = np.sqrt(n_points / 12)
    
    # Check if nside is a power of 2
    return nside.is_integer() and (nside & (nside - 1)) == 0

# def detect_grid(data):
#     """Classify the grid type based on coordinate structure."""

#     if 'lat' or 'latitude' in data.coords and 'lon' or 'longitude' in data.coords:
#         if data.lat.ndim == 1 and data.lon.ndim == 1:
#             if data.lat.dims == data.lon.dims:
#                 return "Unstructured Grid"
#             else:
#                 return "Regular Grid"
#         if data.lat.ndim == 2 and data.lon.ndim == 2:
#             return "Curvilinear Grid"