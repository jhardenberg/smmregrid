"""Utils module"""

import xarray
from .log import setup_logger

default_vert_coords = ['lev', 'nz1', 'nz', 'depth', 'depth_full', 'depth_half']

# set up logger
loggy = setup_logger(level='WARNING', name=__name__)

def find_vert_coords(xfield):
    """
    Find a vertical coordinate among defaults
    Used to define if we need the 3d interpolation with adaptive mask
    """

    if isinstance(xfield, str):
        xfield = xarray.open_dataset(xfield)

    vcoords = list(set(xfield.coords.keys()).intersection(default_vert_coords))
    if len(vcoords) > 1:
        raise ValueError('Multiple vertical coordinates in the same file are not yet supported')
    if len(vcoords) == 0:
        return None
    return vcoords[0]

    # for coord in default_vert_coords:
    #     if coord in xfield.coords:
    #         return coord
