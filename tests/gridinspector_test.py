"""Set of basic tests for smmregrid gridinspector"""

import os
import pytest
import xarray
from smmregrid import GridInspector


INDIR = 'tests/data'

@pytest.mark.parametrize("file,ngrids,firstdims, variables", [
    ("2t-era5.nc", 1, ["lon", "lat"], ['2t']),
    ("mix-cesm.nc", 1, ["lon", "lat"], ['hfss', 'hfls', 'rlds', 'rlus']),
    ("so3d-nemo.nc", 1, ["i", "j", "lev"], ['so']),
    ("ua-so_mix_ecearth.nc", 2, ["lon", "lat"], ['ua']),
    ("temp3d-fesom.nc", 1, ['nod2', 'nz1'], ['temp']),
    ])
def test_basic_gridinspector(file, ngrids, firstdims, variables):
    """test for GridInspector"""
    filename = os.path.join('tests/data', file)
    xfield = xarray.open_mfdataset(filename)
    grids = GridInspector(xfield, loglevel='debug').get_grid_info()
    assert len(grids) == ngrids
    assert sorted(grids[0].dims) == sorted(firstdims)
    assert list(grids[0].variables.keys()) == variables