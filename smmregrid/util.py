"""Utility module with various support function"""

import re
import os
import warnings
import xarray

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
