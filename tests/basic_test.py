"""Set of basic tests for smmregrid"""

import os
import pytest
import xarray
import numpy
from smmregrid.checker import check_cdo_regrid, cdo_generate_weights, Regridder
from smmregrid.log import setup_logger

INDIR = 'tests/data'
tfile = os.path.join(INDIR, 'r360x180.nc')
rfile = os.path.join(INDIR, 'regional.nc')


# test to verify that NaN are preserved
@pytest.mark.parametrize("method", ['con', 'nn', 'bil'])
def test_nan_preserve(method): 
    xfield = xarray.open_mfdataset(os.path.join(INDIR, 'tas-ecearth.nc'))
    xfield['tas'][1,:,:] = numpy.nan
    wfield = cdo_generate_weights(xfield, tfile, method = method)
    interpolator = Regridder(weights=wfield)
    rfield = interpolator.regrid(xfield)
    assert numpy.isnan(rfield['tas'][1,:,:]).all().compute()

# test for pressure levels on gaussian grid (3D)
@pytest.mark.parametrize("method", ['con', 'nn'])
def test_nemo_3d(method):
    fff = check_cdo_regrid(os.path.join(INDIR, 'so3d-nemo.nc'), tfile,
                           remap_method=method, init_method='grids')
    assert fff is True

# test for pressure levels on gaussian grid (3D)
@pytest.mark.parametrize("method", ['con', 'nn'])
def test_fesom_3d(method):
    fff = check_cdo_regrid(os.path.join(INDIR, 'temp3d-fesom.nc'), tfile,
                           remap_method=method, init_method='grids')
    assert fff is True

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
    fff = check_cdo_regrid(os.path.join(INDIR, 'tas-ecearth.nc'), tfile,
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
@pytest.mark.parametrize("method", ['con', 'nn', 'bil'])
def test_levbylev_plev_gaussian(method):
    fff = check_cdo_regrid(os.path.join(INDIR, 'ua-ecearth.nc'), tfile,
                           remap_method=method, init_method='weights',
                           vert_coord="plev")
    assert fff is True

# test for pressure levels on gaussian grid with info logging (3D)
@pytest.mark.parametrize("method", ['con', 'nn', 'bil'])
def test_full_plev_gaussian(method):
    _ = setup_logger('INFO')
    fff = check_cdo_regrid(os.path.join(INDIR, 'ua-ecearth.nc'), tfile,
                           remap_method=method, init_method='grids')
    assert fff is True

