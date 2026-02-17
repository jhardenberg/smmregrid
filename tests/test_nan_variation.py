import xarray as xr
import numpy as np
import pytest
from smmregrid import Regridder

def test_time_varying_nan():
    # Create a source grid (e.g., 10x10)
    lon = np.linspace(0, 360, 10, endpoint=False)
    lat = np.linspace(-90, 90, 10)
    time = [0, 1]
    
    data = np.ones((2, 10, 10))
    # Time step 0: no NaNs
    # Time step 1: half the grid is NaNs
    data[1, :, :5] = np.nan
    
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

    # At time 0, everything should be full
    assert out_skipna_1['tas'].isel(time=0).notnull().all().compute()
