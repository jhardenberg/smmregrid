"""Set of basic tests for smmregrid"""

import os
import pytest
import xarray
import numpy
from smmregrid.checker import cdo_generate_weights, Regridder

INDIR = 'tests/data'
tfile = os.path.join(INDIR, 'r360x180.nc')
rfile = os.path.join(INDIR, 'regional.nc')

@pytest.mark.parametrize("method", ['con', 'nn', 'bil'])
def test_healpix_extra(method):
    """Test for healpix with cdo_extra and cdo_options"""
    wfield = cdo_generate_weights(os.path.join(INDIR, 'healpix_0.nc'), tfile,
                                  method = method, cdo_extra = '-setgrid,hp1_nested',
                                  cdo_options='--force', loglevel='debug')
    interpolator = Regridder(weights=wfield, loglevel='debug')
    xfield = xarray.open_mfdataset(os.path.join(INDIR, 'healpix_0.nc'))
    rfield = interpolator.regrid(xfield)
    assert rfield['tas'].shape == (2, 180, 360)

@pytest.mark.parametrize("method", ['con', 'nn', 'bil'])
def test_nan_preserve(method):
    """Test to verify that NaN are preserved"""
    xfield = xarray.open_mfdataset(os.path.join(INDIR, 'tas-ecearth.nc'))
    xfield['tas'][1,:,:] = numpy.nan
    wfield = cdo_generate_weights(xfield, tfile, method = method, loglevel='debug')
    interpolator = Regridder(weights=wfield, loglevel='debug')
    rfield = interpolator.regrid(xfield)
    assert numpy.isnan(rfield['tas'][1,:,:]).all().compute()
