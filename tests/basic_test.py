"""Set of basic tests for smmregrid weights"""

import os
import pytest
import xarray
import numpy
from smmregrid import Regridder, CdoGenerate, cdo_generate_weights


INDIR = 'tests/data'
tfile = os.path.join(INDIR, 'r360x180.nc')


@pytest.mark.parametrize("method", ['con', 'nn', 'bil'])
def test_healpix_extra(method):
    """Test for healpix with cdo_extra and cdo_options"""
    if method == 'con':
        options = ['--force']
    else:
        options = ['--force', '-f', 'nc']
    wfield = CdoGenerate(os.path.join(INDIR, 'healpix_0.nc'), tfile,
                         cdo_extra='-setgrid,hp1_nested',
                         cdo_options=options,
                         loglevel='debug').weights(method=method)
    interpolator = Regridder(weights=wfield, loglevel='debug')
    xfield = xarray.open_mfdataset(os.path.join(INDIR, 'healpix_0.nc'))
    rfield = interpolator.regrid(xfield)
    assert rfield['tas'].shape == (2, 180, 360)


@pytest.mark.parametrize("method", ['con', 'nn', 'bic'])
def test_nan_preserve(method):
    """Test to verify that NaN are preserved with deprecated cdo_generate_weights"""
    xfield = xarray.open_mfdataset(os.path.join(INDIR, 'tas-ecearth.nc'))
    xfield['tas'][1, :, :] = numpy.nan
    wfield = cdo_generate_weights(xfield, tfile, loglevel='debug', method=method)
    interpolator = Regridder(weights=wfield, horizontal_dims='pippo', loglevel='debug')
    rfield = interpolator.regrid(xfield)
    assert numpy.isnan(rfield['tas'][1, :, :]).all().compute()


@pytest.mark.parametrize("method", ['nn'])
def test_datarray(method):
    """"Minimal test to verify regridding from DataArray"""
    xfield = xarray.open_mfdataset(os.path.join(INDIR, 'tas-ecearth.nc'))
    interpolator = Regridder(source_grid=xfield['tas'], target_grid=tfile,
                             loglevel='debug', method=method)
    interp = interpolator.regrid(source_data=xfield)
    assert interp['tas'].shape == (12, 180, 360)


@pytest.mark.parametrize("method", ['dis', 'con'])
def test_horizontal_dims(method):
    """"Minimal test to verify regridding with horizontal_dims coords"""
    xfield = xarray.open_mfdataset(os.path.join(INDIR, 'tas-ecearth.nc'))
    interpolator = Regridder(source_grid=xfield, target_grid=tfile, loglevel='debug',
                             method=method, horizontal_dims=['lon', 'lat'])
    interp = interpolator.regrid(source_data=xfield.isel(time=0))
    assert interp['tas'].shape == (180, 360)


@pytest.mark.parametrize("method", ['bil', 'con'])
def test_toward_healpix(method):
    """"Testing toward healpix string-defined-grid grid"""
    xfield = xarray.open_mfdataset(os.path.join(INDIR, 'tas-ecearth.nc'))
    interpolator = Regridder(source_grid=xfield, target_grid='hp16_nested', loglevel='debug',
                             method=method, cdo_options='--force')
    interp = interpolator.regrid(source_data=xfield)
    assert interp['tas'].shape == (12, 3072)


@pytest.mark.parametrize("source_grid,target_grid,src_grid_size,dst_grid_size", [
    ('r180x90', 'r360x180', 180 * 90, 360 * 180),
    ('F128', 'r180x90', 256 * 512, 180 * 90),
    ('hp32', 'r360x180', 12288, 360 * 180),
])
def test_generation_from_cdo(source_grid, target_grid, src_grid_size, dst_grid_size):
    """Test area generation from CDO grid string"""
    generator = CdoGenerate(source_grid=source_grid,
                            target_grid=target_grid,
                            loglevel='debug')
    weights = generator.weights(method="bil")
    assert weights.sizes['src_grid_size'] == src_grid_size
    assert weights.sizes['dst_grid_size'] == dst_grid_size

def test_check_nan_auto():
    """Test to verify that NaN are preserved with automatic check_nan"""
    xfield = xarray.open_dataset(os.path.join(INDIR, 'ua-ipsl.nc'))
    regrid = Regridder(source_grid=xfield, target_grid='r90x45', loglevel='debug', check_nan=True)
    rr = regrid.regrid(xfield.isel(time=0))
    count = rr['ua'].isnull().sum(dim=['lon', 'lat']).compute()
    assert count[-1] == 0, f"NaN values found in the regridded data: {count}"
    assert count[1] == 589, f"NaN values found in the regridded data: {count}"

def test_weights_vertical_coord():
    """Test to verify that weights are created with vertical coordinate"""
    xfield = xarray.open_dataset(os.path.join(INDIR, 'so3d-nemo.nc'))
    xfield = xfield.isel(lev=slice(0, 5))
    generator = CdoGenerate(source_grid=xfield, target_grid='r90x45', loglevel='debug')
    weights = generator.weights(method="nn", vert_coord='lev')
    assert 'lev' in weights.dims, "Vertical coordinate 'lev' not found in weights dimensions"
    assert weights.sizes['lev'] == xfield.sizes['lev'], "Mismatch in vertical levels size"
    assert xfield.lev.equals(weights.lev)
