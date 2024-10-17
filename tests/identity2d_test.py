"""Set of basic tests for smmregrid"""

import os
import pytest
import xarray as xr
from smmregrid.checker import check_cdo_regrid

INDIR = 'tests/data'
tfile = os.path.join(INDIR, 'r360x180.nc')
rfile = os.path.join(INDIR, 'regional.nc')


# test for gaussian reduced grid (only nn)
@pytest.mark.parametrize("method", ['nn', 'con'])
def test_healpix(method):
    fff = check_cdo_regrid(os.path.join(INDIR, 'tas-healpix2.nc'), tfile,
                           remap_method=method, init_method='grids')
    assert fff is True

# test for gaussian reduced grid (only nn)
@pytest.mark.parametrize("method", ['nn'])
def test_gaussian_reduced(method):
    fff = check_cdo_regrid(os.path.join(INDIR, 'lsm-ifs.grb'), tfile,
                           remap_method=method, init_method='grids')
    assert fff is True

# test for gaussian grids as EC-Earth cmor
@pytest.mark.parametrize("method", ['bil', 'con', 'nn'])
def test_gaussian_regular(method):
    xfield = xr.open_dataset(os.path.join(INDIR, 'tas-ecearth.nc'))
    fff = check_cdo_regrid(xfield, tfile,
                           remap_method=method)
    assert fff is True

# test for gaussian grids as EC-Earth cmor
@pytest.mark.parametrize("method", ['bil', 'con', 'nn'])
def test_gaussian_regular_regional(method):
    fff = check_cdo_regrid(os.path.join(INDIR, 'tas-ecearth.nc'), rfile,
                           remap_method=method)
    assert fff is True

# test for lonlt grids, init by weights
@pytest.mark.parametrize("method", ['bil', 'con', 'nn'])
def test_lonlat(method):
    fff = check_cdo_regrid(os.path.join(INDIR, '2t-era5.nc'), tfile,
                           remap_method=method, init_method='weights')
    assert fff is True

# test for unstructured grids as FESOM CMOR (no bilinear)
@pytest.mark.parametrize("method", ['con', 'nn'])
def test_unstructured(method):
    fff = check_cdo_regrid(os.path.join(INDIR, 'tos-fesom.nc'), tfile,
                           remap_method=method, init_method='grids')
    assert fff is True

# test for curvilinear grid
@pytest.mark.parametrize("method", ['con', 'nn', 'bil'])
def test_curivilinear(method):
    fff = check_cdo_regrid(os.path.join(INDIR, 'onlytos-ipsl.nc'), tfile,
                           remap_method=method, init_method='grids')
    assert fff is True

# test for pressure levels on gaussian grid (2D, level-by-level), init by weights
#@pytest.mark.parametrize("method", ['con', 'nn', 'bil'])
#def test_levbylev_plev_gaussian(method):
#    fff = check_cdo_regrid(os.path.join(INDIR, 'ua-ecearth.nc'), tfile,
#                           remap_method=method, init_method='weights',
#                           vertical_dim="plev")
#    assert fff is True

# test for pressure levels on gaussian grid with info logging (3D)
@pytest.mark.parametrize("method", ['con', 'nn', 'bil'])
def test_full_plev_gaussian(method):
    fff = check_cdo_regrid(os.path.join(INDIR, 'ua-ecearth.nc'), tfile,
                           remap_method=method, init_method='grids')
    assert fff is True
