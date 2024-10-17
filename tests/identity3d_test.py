"""Set of basic tests for smmregrid"""

import os
import pytest
import xarray as xr
from smmregrid.checker import check_cdo_regrid

INDIR = 'tests/data'
tfile = os.path.join(INDIR, 'r360x180.nc')
rfile = os.path.join(INDIR, 'regional.nc')

# test for pressure levels on gaussian grid (3D)
@pytest.mark.parametrize("method", ['con', 'nn'])
def test_nemo_3d(method):
    if method == 'nn':
        xfield = os.path.join(INDIR, 'so3d-nemo.nc')
    else:
        xfield = xr.open_dataset(os.path.join(INDIR, 'so3d-nemo.nc'))
    fff = check_cdo_regrid(xfield, tfile,
                           remap_method=method, init_method='grids')
    assert fff is True

# test for pressure levels on gaussian grid (3D)
@pytest.mark.parametrize("method", ['con', 'nn'])
def test_fesom_3d(method):
    fff = check_cdo_regrid(os.path.join(INDIR, 'temp3d-fesom.nc'), tfile,
                           remap_method=method, init_method='grids')
    assert fff is True