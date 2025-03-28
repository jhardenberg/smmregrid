"""Utility module with various support function"""

import re
import os
import warnings
import xarray
import numpy as np

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

def find_coord(ds, possible_names):
    """Find the first matching coordinate in the dataset."""
    return next((name for name in possible_names if name in ds.coords), None)

def detect_grid(data):
    """Classify the grid type based on coordinate structure."""

    lat = find_coord(data, ["lat", "latitude", "nav_lat"])
    lon = find_coord(data, ["lon", "longitude", "nav_lon"])

    if not lat or not lon:
        return "Unknown"

    # 2D coord-dim dependency
    if data[lat].ndim == 2 and data[lon].ndim == 2:
        return "Curvilinear"

    # 1D coord-dim dependency
    if data[lat].ndim == 1 and data[lon].ndim == 1:

        # Regular: latitude and longitude depend on different coordinates
        if data[lat].dims != data[lon].dims:
            # Gaussian: second derivative of latitude is positive from -90 to 0
            lat_values = data[lat].where(data[lat]<0).values
            lat_values=lat_values[~np.isnan(lat_values)]
            gaussian = np.all(np.diff(lat_values, n=2) > 0)
            if gaussian:
                return "GaussianRegular"
            return "Regular"

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
