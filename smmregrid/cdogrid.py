""""
Class to define CDO objects and their grid types.
This module provides a class to represent CDO grid strings and 
validate them against known CDO grid patterns.
"""

import re

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


class CdoGrid:
    """Class to represent and validate a CDO grid string."""

    def __init__(self, grid_str):
        if not isinstance(grid_str, str):
            raise TypeError("CDOGrid must be initialized with a string.")
        self.grid_kind = self._is_cdo_grid(grid_str)
        if self.grid_kind is None:
            self.grid_str = 'Invalid'
        else:
            self.grid_str = grid_str

    def _is_cdo_grid(self, grid_str):
        """Check if the input string matches any CDO grid type."""

        # Check if the string matches any of the grid patterns
        for grid_type, pattern in CDO_GRID_PATTERNS.items():
            if pattern.match(grid_str):
                return grid_type

        # Return False if no patterns match
        return None

    def __repr__(self):
        return f"CDOGrid(grid_str='{self.grid_str}', grid_kind='{self.grid_kind}')"

