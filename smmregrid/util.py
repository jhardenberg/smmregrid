"""Utils module"""

import xarray

default_vert_coords = ['lev', 'nz1']

def find_vert_coord(xfield):
    """
    Find a vertical coordinate among defaults
    Used to define if we need the 3d interpolation with adaptive mask
    """

    if isinstance(xfield, str):
        xfield = xarray.open_dataset(xfield)
    for coord in default_vert_coords:
        if coord in xfield.coords:
            return coord