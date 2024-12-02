"""Tests for smmregrid to verify level selection"""

import os
import xarray as xr
from smmregrid.checker import check_cdo_regrid, Regridder

INDIR = 'tests/data'
tfile = os.path.join(INDIR, 'r360x180.nc')

def test_multi_grid_inspection():
    """test and assert with CDO multigrids"""

    filename = os.path.join(INDIR, 'ua-so_mix_ecearth.nc')
    regrid = Regridder(filename, 'r180x90', loglevel='debug')
    data = xr.open_dataset(filename)
    regridded = regrid.regrid(data)
    assert len(regrid.grids) == 2
    assert vars(regrid.grids[1])['dims'] == ('time', 'lev', 'j', 'i')
    assert vars(regrid.grids[0])['dims'] == ('time', 'plev', 'lat', 'lon')
    assert regridded['ua'].shape == (2, 3, 90, 180)
    assert regridded['so'].shape == (2, 3, 90, 180)


def test_multi_grid_cdo():
    """test and assert with CDO multigrids"""

    assert check_cdo_regrid(os.path.join(INDIR, 'ua-so_mix_ecearth.nc'), tfile, init_method='grids')