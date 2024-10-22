"""Set of basic tests for smmregrid"""

import os
import pytest
import xarray
import numpy
from smmregrid import Regridder, cdo_generate_weights


INDIR = 'tests/data'
tfile = os.path.join(INDIR, 'r360x180.nc')
rfile = os.path.join(INDIR, 'regional.nc')

@pytest.mark.parametrize("method", ['con', 'nn', 'bil'])
def test_healpix_extra(method):
    """Test for healpix with cdo_extra and cdo_options"""
    if method == 'con':
        options = ['--force']
    else:
        options = ['--force', '-f', 'nc']
    wfield = cdo_generate_weights(os.path.join(INDIR, 'healpix_0.nc'), tfile,
                                  method = method, cdo_extra = '-setgrid,hp1_nested',
                                  cdo_options=options, loglevel='debug')
    interpolator = Regridder(weights=wfield, loglevel='debug')
    xfield = xarray.open_mfdataset(os.path.join(INDIR, 'healpix_0.nc'))
    rfield = interpolator.regrid(xfield)
    assert rfield['tas'].shape == (2, 180, 360)

@pytest.mark.parametrize("method", ['con', 'nn', 'bic'])
def test_nan_preserve(method):
    """Test to verify that NaN are preserved"""
    xfield = xarray.open_mfdataset(os.path.join(INDIR, 'tas-ecearth.nc'))
    xfield['tas'][1,:,:] = numpy.nan
    wfield = cdo_generate_weights(xfield, tfile, method = method, loglevel='debug')
    interpolator = Regridder(weights=wfield, space_dims='pippo', loglevel='debug')
    rfield = interpolator.regrid(xfield)
    assert numpy.isnan(rfield['tas'][1,:,:]).all().compute()

@pytest.mark.parametrize("method", ['nn'])
def test_datarray(method):
    """"Minimal test to verify regridding from DataArray"""
    xfield = xarray.open_mfdataset(os.path.join(INDIR, 'tas-ecearth.nc'))
    interpolator = Regridder(source_grid=xfield['tas'], target_grid=tfile, loglevel='debug', method = method)
    interp = interpolator.regrid(source_data=xfield)
    assert interp['tas'].shape == (12, 180, 360)

@pytest.mark.parametrize("method", ['dis', 'con'])
def test_horizontal_dims(method):
    """"Minimal test to verify regridding with horizontal_dims coords"""
    xfield = xarray.open_mfdataset(os.path.join(INDIR, 'tas-ecearth.nc'))
    interpolator = Regridder(source_grid=xfield, target_grid=tfile, loglevel='debug',
                             method = method, horizontal_dims=['lon', 'lat'])
    interp = interpolator.regrid(source_data=xfield.isel(time=0))
    assert interp['tas'].shape == (180, 360)

@pytest.mark.parametrize("method", ['bil', 'con'])
def test_toward_healpix(method):
    """"Testing toward healpix string-defined-grid grid"""
    xfield = xarray.open_mfdataset(os.path.join(INDIR, 'tas-ecearth.nc'))
    interpolator = Regridder(source_grid=xfield, target_grid='hp16_nested', loglevel='debug',
                             method = method, cdo_options='--force')
    interp = interpolator.regrid(source_data=xfield)
    assert interp['tas'].shape == (12, 3072)
