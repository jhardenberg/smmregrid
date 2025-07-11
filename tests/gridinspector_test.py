"""Set of basic tests for smmregrid gridinspector"""

import os
import pytest
import xarray
from smmregrid import GridInspector


INDIR = 'tests/data'

@pytest.mark.parametrize("file,ngrids,firstdims, variables,kind", [
    ("2t-era5.nc", 1, ["lon", "lat"], ['2t'], "Regular"),
    ("mix-cesm.nc", 1, ["lon", "lat"], ['hfss', 'hfls', 'rlds', 'rlus'], "Regular"),
    ("so3d-nemo.nc", 1, ["i", "j", "lev"], ['so'], "Curvilinear"),
    ("ua-so_mix_ecearth.nc", 2, ["lon", "lat"], ['ua'], "GaussianRegular"),
    ("temp3d-fesom.nc", 1, ['nod2', 'nz1'], ['temp'], "Unstructured"),
    ])
def test_basic_gridinspector(file, ngrids, firstdims, variables, kind):
    """test for GridInspector"""
    xfield = os.path.join('tests/data', file)
    xfield = xarray.open_dataset(xfield)
    grids = GridInspector(xfield, loglevel='debug').get_gridtype()
    assert len(grids) == ngrids
    assert set(grids[0].dims) == set(firstdims)
    assert set(grids[0].variables.keys()) == set(variables)
    assert grids[0].kind == kind

def test_basic_gridinspector_raise():
    """test for GridInspector raise"""
    with pytest.raises(TypeError):
        GridInspector(24)
    with pytest.raises(FileNotFoundError):
        GridInspector('not_a_file.nc')

def test_basic_gridinspector_dataarray():
    """test for GridInspector with a DataArray"""
    xfield = os.path.join('tests/data', '2t-era5.nc')
    xfield = xarray.open_dataset(xfield)['2t']
    grids = GridInspector(xfield, loglevel='debug').get_gridtype()
    assert len(grids) == 1
    assert set(grids[0].dims) == set(['lon', 'lat'])
    assert set(grids[0].variables.keys()) == set(['2t'])
    assert grids[0].kind == 'Regular'

def test_get_gridtype_attr():
    """test for GridInspector get_gridtype_attr"""
    xfield = os.path.join('tests/data', '2t-era5.nc')
    gridinspect = GridInspector(xfield, loglevel='debug')
    grids = gridinspect.get_gridtype()
    assert set(gridinspect.get_gridtype_attr(grids, 'dims')) == set(['lon', 'lat'])
    assert gridinspect.get_gridtype_attr(grids, 'variables') == ['2t']
    assert gridinspect.get_gridtype_attr(grids, 'kind') == ['Regular']