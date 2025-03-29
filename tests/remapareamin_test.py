"""Tests for smmregrid to verify level selection"""

import os
import pytest
import numpy as np
import xarray as xr
from smmregrid.checker import  Regridder

INDIR = 'tests/data'
ifile = os.path.join(INDIR, 'onlytos-ipsl.nc')
tfile = os.path.join(INDIR, 'r360x180.nc')

@pytest.mark.parametrize("fraction,nan", [
    ("0.0", 11607),
    (0.5, 10731),
    ("0.9", 9866),
])
def test_remap_area_min(fraction, nan):
    """test and assert with CDO multigrids"""

    regridder = Regridder(ifile, 'r180x90', remap_area_min=fraction, loglevel='debug')
    data = xr.open_dataset(ifile)
    regrid = regridder.regrid(data['tos'].isel(time=0))
    nanfound = (~np.isnan(regrid)).sum().values
    assert nanfound == nan
