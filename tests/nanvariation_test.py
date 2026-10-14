import xarray as xr
import numpy as np
import os
import pytest
from smmregrid import Regridder, CdoGenerate
from smmregrid.checker import check_cdo_regrid

def test_time_varying_nan():
    """Test regridding with time-varying NaN values and skipna enabled."""
    # Create a source grid (e.g., 10x10)
    lon = np.linspace(0, 360, 10, endpoint=False)
    lat = np.linspace(-90, 90, 10)
    time = [0, 1, 2]
    
    data = np.ones((3, 10, 10))
    # Time step 0: half the grid is NaNs
    # Time step 1: one column is NaNs
    # Time step 2: no NaNs
    data[0, :, :5] = np.nan
    data[1, :, :1] = np.nan
    
    src = xr.DataArray(
        data,
        coords={'time': time, 'lat': lat, 'lon': lon},
        dims=['time', 'lat', 'lon'],
        name='tas'
    )
    src.lat.attrs = {'units': 'degrees_north', 'standard_name': 'latitude'}
    src.lon.attrs = {'units': 'degrees_east', 'standard_name': 'longitude'}
    src.attrs = {'units': 'K', 'standard_name': 'air_temperature'}
    
    # Create as Dataset
    src_ds = src.to_dataset()
    
    # Denser target grid to hit transitions
    tgt = 'r36x18'
    
    # Regrid without skipna (default behavior)
    regridder_default = Regridder(source_grid=src_ds, target_grid=tgt, method='bil', 
                                  skipna=False, remap_area_min=0.0)
    out_default = regridder_default.regrid(src_ds)
    
    # Regrid with skipna=True (na_thres=1.0)
    regridder_skipna_1 = Regridder(source_grid=src_ds, target_grid=tgt, method='bil', 
                                   skipna=True, na_thres=1.0, remap_area_min=0.0)
    out_skipna_1 = regridder_skipna_1.regrid(src_ds)

    # Regrid with skipna=True and na_thres=0.1
    regridder_skipna_01 = Regridder(source_grid=src_ds, target_grid=tgt, method='bil', 
                                    skipna=True, na_thres=0.1, remap_area_min=0.0)
    out_skipna_01 = regridder_skipna_01.regrid(src_ds)

    non_nan_default = out_default['tas'].isel(time=1).notnull().sum().compute().item()
    non_nan_skipna_1 = out_skipna_1['tas'].isel(time=1).notnull().sum().compute().item()
    non_nan_skipna_01 = out_skipna_01['tas'].isel(time=1).notnull().sum().compute().item()
    
    # Verifications
    assert non_nan_skipna_1 > non_nan_default
    assert non_nan_skipna_1 > non_nan_skipna_01
    # na_thres=0.1 should be similar to default because bilinear uses few neighbors
    # but still might have some differences
    assert non_nan_skipna_01 >= non_nan_default

    non_nan_default = out_default['tas'].isel(time=2).notnull().sum().compute().item()
    non_nan_skipna_1 = out_skipna_1['tas'].isel(time=2).notnull().sum().compute().item()
    non_nan_skipna_01 = out_skipna_01['tas'].isel(time=2).notnull().sum().compute().item()
    assert non_nan_skipna_1 > non_nan_default 
    assert non_nan_skipna_1 == non_nan_skipna_01

    # At time 2, everything should be full
    assert out_skipna_1['tas'].isel(time=2).notnull().all().compute()

@pytest.mark.parametrize("method", ['nn', 'con'])
def test_cdo_regrid_skipna_2d(method):
    """Test regridding of the Berkeley dataset against CDO output with skipna enabled."""

    filein = os.path.join('tests/data/berkley.nc')
    assert check_cdo_regrid(filein, 'r180x90', loglevel='debug', skipna=True, remap_method=method), "Berkeley.nc failed with skipna"
    filein = os.path.join('tests/data/onlytos-ipsl.nc')
    field = xr.open_mfdataset(filein)
    assert check_cdo_regrid(field, 'r180x90', loglevel='debug', skipna=True), "Onlytos-ipsl.nc failed with skipna"

@pytest.mark.parametrize("method", ['nn', 'con'])
def test_cdo_regrid_skipna_3d(method):
    """Test regridding of the Berkeley dataset against CDO output with skipna enabled."""

    filein = os.path.join('tests/data/so3d-nemo.nc')
    field = xr.open_mfdataset(filein)
    assert check_cdo_regrid(field, 'r180x90', loglevel='debug', skipna=True, remap_method=method), "So3d-nemo.nc failed with skipna"

@pytest.mark.parametrize("method", ['nn', 'con'])
def test_cdo_regrid_skipna_full(method):
    """Test regridding of the Berkeley dataset against CDO output with skipna enabled."""

    filein = os.path.join('tests/data/2t-era5.nc')
    field = xr.open_mfdataset(filein)
    assert check_cdo_regrid(field, 'r180x90', loglevel='debug', skipna=True, remap_method=method), "2t-era5.nc failed with skipna"

@pytest.mark.parametrize("method", ['nn', 'con'])
def test_cdo_regrid_skipna_2dmask(method):
    """Test regridding of the Berkeley dataset against CDO output with skipna enabled."""

    filein = os.path.join('tests/data/onlytos-ipsl.nc')
    field = xr.open_mfdataset(filein)
    assert check_cdo_regrid(field, 'r180x90', loglevel='debug', skipna=True, remap_method=method), "Onlytos-ipsl.nc failed with skipna"


@pytest.mark.parametrize("filein", ['tests/data/regional.nc', 'tests/data/onlytos-ipsl.nc'])
def test_cdo_regrid_skipna_destarea(filein):
    """Test conservative remapping with remap_norm='destarea' and skipna against CDO.
    With destarea the weights must not be renormalized by the sum of the valid weights.
    regional.nc has partially covered target cells at its boundaries,
    onlytos-ipsl.nc has a land mask which makes coastal target cells partially covered."""

    assert check_cdo_regrid(filein, 'r100x50', init_method='weights', remap_norm='destarea',
                            skipna=True, loglevel='debug'), f"{filein} failed with skipna and destarea"


def test_skipna_destarea_locality():
    """With skipna and destarea, a single NaN in the source must not change the target cells
    which do not receive any weight from it (see PR #70): this used to change all the
    partially covered target cells at the boundaries of the regional grid."""

    src = xr.open_dataset('tests/data/regional.nc')[['pr']].load()
    weights = CdoGenerate(src, 'r100x50', skipna=True, loglevel='warning').weights(
        method='con', remap_norm='destarea')
    regridder = Regridder(weights=weights, skipna=True, remap_area_min=0.0, loglevel='warning')

    # put a single NaN in the north-east corner of the regional grid
    src_nan = src.copy(deep=True)
    src_nan['pr'][:, -1, -1] = np.nan

    out = regridder.regrid(src)['pr'].compute()
    out_nan = regridder.regrid(src_nan)['pr'].compute()

    # target cells far from the NaN must be unchanged, including the partially covered ones
    far = out.lat < 60
    assert out.where(far).notnull().sum() > 0
    xr.testing.assert_allclose(out.where(far), out_nan.where(far))


@pytest.mark.parametrize("init", ['grids', 'grids_mask_dim', 'weights'])
def test_skipna_3d_as_2d(init):
    """With skipna the weights do not depend on the level, so 3D data are regridded as 2D:
    a single weights matrix and no mask dimension, whatever the initialization (also when
    mask_dim is given, which is ignored, and from 2D weights, which used to fail on 3D data).
    The output must be the same in all cases, with the vertical dimension before the horizontal ones."""

    filein = 'tests/data/so3d-nemo.nc'
    data = xr.open_dataset(filein)
    if init == 'weights':
        weights = CdoGenerate(filein, 'r180x90', skipna=True, loglevel='warning').weights(method='con')
        regridder = Regridder(weights=weights, skipna=True, loglevel='warning')
    else:
        mask_dim = 'lev' if init == 'grids_mask_dim' else None
        regridder = Regridder(source_grid=filein, target_grid='r180x90', mask_dim=mask_dim,
                              skipna=True, loglevel='warning')
        assert regridder.grids[0].mask_dim is None
        assert not isinstance(regridder.grids[0].weights_matrix, list)

    out = regridder.regrid(data)['so']
    assert out.dims == ('time', 'lev', 'lat', 'lon')

    reference = Regridder(source_grid=filein, target_grid='r180x90', skipna=True,
                          loglevel='warning').regrid(data)['so']
    xr.testing.assert_allclose(out.compute(), reference.compute())
