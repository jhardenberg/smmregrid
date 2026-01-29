"""Set of basic tests for smmregrid weights"""

import pytest
from smmregrid import GridType


@pytest.mark.parametrize("definition,dims,other", [
    (['lon', 'lat'], ['lon', 'lat'], []),
    (['lon', 'lat', 'lev', 'time'], ['lon', 'lat', 'lev'], []),
    (['i', 'k'], ['i'], ['k']),
    (['pix', 'time', 'papera'], ['pix'], ['papera']),
    (['time', 'papera'], [], ['papera'])

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
