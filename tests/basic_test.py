from smmregrid.checker import check_cdo_regrid
import pytest
import os


indir = 'tests/data'
tfile = os.path.join(indir, 'r360x180.nc')

# test for gaussian grids as EC-Earth cmor
@pytest.mark.parametrize("method", ['bil', 'con', 'nn'])
def test_gaussian(method):
    f = check_cdo_regrid(os.path.join(indir, 'tas-ecearth.nc'), tfile, method = method)
    assert f == True

# test for lonlt grids
@pytest.mark.parametrize("method", ['bil', 'con', 'nn'])
def test_lonlat(method):
    f = check_cdo_regrid(os.path.join(indir, '2t-era5.nc'), tfile, method = method)
    assert f == True

# test for unstructured grids as FESOM CMOR (no bilinear)
#@pytest.mark.parametrize("method", ['con', 'nn'])
#def test_unstructured(method):
#    f = check_cdo_regrid(os.path.join(indir, 'tos-fesom.nc'), tfile, method = method)
#    assert f == True

