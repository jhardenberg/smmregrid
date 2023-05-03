"""Set of basic tests for smmregrid"""

import os
import pytest
from smmregrid.checker import check_cdo_regrid

indir = 'tests/data'
tfile = os.path.join(indir, 'r360x180.nc')

# test for pressure levels on gaussian grid (3D)
@pytest.mark.parametrize("method", ['con', 'nn'])
def test_fesom_3d(method):
    fff = check_cdo_regrid(os.path.join(indir, 'temp3d-fesom.nc'), tfile, method=method, vert_coord="nz1")
    assert fff is True

# test for gaussian reduced grid (only nn)
@pytest.mark.parametrize("method", ['nn', 'con'])
def test_healpix(method):
    fff = check_cdo_regrid(os.path.join(indir, 'tas-healpix2.nc'), tfile, method=method)
    assert fff is True

# test for gaussian reduced grid (only nn)
@pytest.mark.parametrize("method", ['nn'])
def test_gaussian_reduced(method):
    fff = check_cdo_regrid(os.path.join(indir, 'lsm-ifs.grb'), tfile, method=method)
    assert fff is True

# test for gaussian grids as EC-Earth cmor
@pytest.mark.parametrize("method", ['bil', 'con', 'nn'])
def test_gaussian(method):
    fff = check_cdo_regrid(os.path.join(indir, 'tas-ecearth.nc'), tfile, method=method)
    assert fff is True

# test for lonlt grids
@pytest.mark.parametrize("method", ['bil', 'con', 'nn'])
def test_lonlat(method):
    fff = check_cdo_regrid(os.path.join(indir, '2t-era5.nc'), tfile, method=method)
    assert fff is True

# test for unstructured grids as FESOM CMOR (no bilinear)
@pytest.mark.parametrize("method", ['con', 'nn'])
def test_unstructured(method):
    fff = check_cdo_regrid(os.path.join(indir, 'tos-fesom.nc'), tfile, method=method)
    assert fff is True

# test for curvilinear grid
@pytest.mark.parametrize("method", ['con', 'nn', 'bil'])
def test_curivilinear(method):
    fff = check_cdo_regrid(os.path.join(indir, 'onlytos-ipsl.nc'), tfile, method=method)
    assert fff is True

# test for pressure levels on gaussian grid (2D, level-by-level)
@pytest.mark.parametrize("method", ['con', 'nn', 'bil'])
def test_levbylev_plev_gaussian(method):
    fff = check_cdo_regrid(os.path.join(indir, 'ua-ecearth.nc'), tfile, method=method)
    assert fff is True

# test for pressure levels on gaussian grid (3D)
@pytest.mark.parametrize("method", ['con', 'nn', 'bil'])
def test_full_plev_gaussian(method):
    fff = check_cdo_regrid(os.path.join(indir, 'ua-ecearth.nc'), tfile, method=method, vert_coord="plev")
    assert fff is True