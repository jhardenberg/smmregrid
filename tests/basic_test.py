from smmregrid.checker import check_cdo_regrid
import pytest
import os


indir = 'tests/data'
tfile = os.path.join(indir, 'r360x180.nc')

# test for gaussian reduced grid (only nn)


@pytest.mark.parametrize("method", ['nn'])
def test_gaussian_reduced(method):
    f = check_cdo_regrid(os.path.join(indir, 'lsm-ifs.grb'), tfile, method=method)
    assert f is True

# test for gaussian grids as EC-Earth cmor


@pytest.mark.parametrize("method", ['bil', 'con', 'nn'])
def test_gaussian(method):
    f = check_cdo_regrid(os.path.join(indir, 'tas-ecearth.nc'), tfile, method=method)
    assert f is True

# test for lonlt grids


@pytest.mark.parametrize("method", ['bil', 'con', 'nn'])
def test_lonlat(method):
    f = check_cdo_regrid(os.path.join(indir, '2t-era5.nc'), tfile, method=method)
    assert f is True

# test for unstructured grids as FESOM CMOR (no bilinear)


@pytest.mark.parametrize("method", ['con', 'nn'])
def test_unstructured(method):
    f = check_cdo_regrid(os.path.join(indir, 'tos-fesom.nc'), tfile, method=method)
    assert f is True

# test for curvilinear grid


@pytest.mark.parametrize("method", ['con', 'nn', 'bil'])
def test_curivilinear(method):
    f = check_cdo_regrid(os.path.join(indir, 'onlytos-ipsl.nc'), tfile, method=method)
    assert f is True

# test for pressure levels on gaussian grid


@pytest.mark.parametrize("method", ['con', 'nn', 'bil'])
def test_plev_gaussian(method):
    f = check_cdo_regrid(os.path.join(indir, 'ua-ecearth.nc'), tfile, method=method)
    assert f is True
