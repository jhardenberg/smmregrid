"""Set of basic tests for smmregrid weights"""

import pytest
from smmregrid import GridType


@pytest.mark.parametrize("definition,dims,other", [
    (['lon', 'lat'], ['lon', 'lat'], []),
    (['lon', 'lat', 'lev', 'time'], ['lon', 'lat', 'lev'], []),
    (['i', 'k'], ['i'], ['k']),
    (['pix', 'time', 'papera'], ['pix'], ['papera']),
    (['time', 'papera'], [], ['papera']),
    (['time', 'lat'], ['lat'], []),  # partial family, e.g. zonal mean
    (['time', 'lat', 'lon', 'x'], ['lat', 'lon'], ['x']),  # only the first family
    (['lat', 'lon', 'nlat', 'nlon'], ['lat', 'lon'], ['nlat', 'nlon']),
    (['nlat', 'nlon', 'lat'], ['nlat', 'nlon'], ['lat']),  # complete family wins over partial
    (['time', 'yq', 'xh'], ['yq', 'xh'], []),  # MOM6 v points
    (['yq', 'xq'], ['yq', 'xq'], []),  # MOM6 corner points
    (['ncells', 'values'], ['ncells'], ['values']),  # unstructured: one dimension only
])
def test_gridtype(definition, dims, other):
    """Basic gridtype investigation"""
    grid = GridType(dims=definition)
    assert set(grid.dims) == set(dims)
    assert set(grid.other_dims) == set(other)
    assert 'GridType' in repr(grid)


def test_difference():
    """Test equality for gridtypes"""
    grid1 = GridType(dims=['lon', 'lat'])
    grid2 = GridType(dims=['i', 'j'])
    grid3 = GridType(dims=['lon', 'lat', 'time', 'plev'])
    assert grid1 != grid2
    assert grid1 == grid3


def test_multiple_vertical():
    """Test multiple vertical"""
    with pytest.raises(ValueError):
        GridType(dims=['lon', 'lat', 'lev', 'nz1'])


def test_gridtype_extradims():
    """Test for extradimensions"""
    with pytest.raises(TypeError):
        GridType(dims=["lon", "lat", "ciccio"],
                 extra_dims=['horizontal'])

    grid = GridType(dims=["lon", "lat", "ciccio"],
                    extra_dims={'mask': ["ciccio"]})
    assert grid.mask_dim == "ciccio"


def test_gridtype_extradims_priority():
    """Declared horizontal dimensions come before the default families"""
    grid = GridType(dims=["lat", "lon", "foo", "bar"], extra_dims={'horizontal': ["foo", "bar"]})
    assert grid.horizontal_dims == ["foo", "bar"]
    assert set(grid.other_dims) == {"lat", "lon"}


def test_gridtype_extradims_override():
    """Test for extradimensions"""

    grid = GridType(dims=["alfa", "beta", "ciccio"],
                    extra_dims={
                        'horizontal': ["alfa", "beta"],
                        'mask': ["ciccio"]},
                    override=True)
    assert grid.mask_dim == "ciccio"
    assert set(grid.horizontal_dims) == set(["alfa", "beta"])
    assert grid.time_dims is None
