"""smmregrid module"""

from .regrid import Regridder, regrid
from .cdogenerate import cdo_generate_weights, CdoGenerate
from .cdogrid import CdoGrid
from .gridtype import GridType
from .gridinspector import GridInspector
from .util import detect_grid


__version__ = '0.1.2'

__all__ = ['Regridder', 'regrid', 'cdo_generate_weights',
           'GridType', 'GridInspector', 'CdoGenerate',
           'CdoGrid', 'detect_grid']
