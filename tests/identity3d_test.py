"""Set of basic tests for smmregrid"""

import os
import pytest
import xarray as xr
from smmregrid.checker import check_cdo_regrid

INDIR = 'tests/data'
tfile = os.path.join(INDIR, 'r360x180.nc')
rfile = os.path.join(INDIR, 'regional.nc')

# test for NEMO 3d grid
@pytest.mark.parametrize("method", ['con', 'nn'])
def test_nemo_3d(method):
    if method == 'nn':
        xfield = os.path.join(INDIR, 'so3d-nemo.nc')
    else:
        xfield = xr.open_dataset(os.path.join(INDIR, 'so3d-nemo.nc'))
    fff = check_cdo_regrid(xfield, tfile,
                           remap_method=method, init_method='grids')
    assert fff is True

# test for for FESOM 3d grid
@pytest.mark.parametrize("method", ['con'])
def test_fesom_3d(method):
    fff = check_cdo_regrid(os.path.join(INDIR, 'temp3d-fesom.nc'), tfile,
                           remap_method=method, init_method='grids')
    assert fff is True

# test for pressure levels on gaussian grid (2D, level-by-level), init by weights
@pytest.mark.parametrize("method", ['con'])
def test_levbylev_plev_gaussian(method):
    fff = check_cdo_regrid(os.path.join(INDIR, 'ua-ecearth.nc'), tfile,
                           remap_method=method, init_method='weights',
                           vertical_dim="plev")
    assert fff is True