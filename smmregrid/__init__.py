"""smmregrid module"""

from .regrid import Regridder, regrid
from .cdogenerate import cdo_generate_weights, CdoGenerate
from .gridtype import GridType, DEFAULT_DIMS
from .gridinspector import GridInspector
from .util import detect_grid, is_cdo_grid


__version__ = '0.1.1'

__all__ = ['Regridder', 'regrid', 'cdo_generate_weights',
           'GridType', 'GridInspector', 'CdoGenerate',
           'detect_grid', 'is_cdo_grid']
