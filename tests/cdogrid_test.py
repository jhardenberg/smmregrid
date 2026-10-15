import os
import pytest
from cdo import Cdo
from smmregrid import CdoGrid

# grids that CDO cannot generate offline (dcw needs downloaded coastline data)
NO_CDO_CONST = ('gme', 'dcw')


@pytest.fixture(scope="module")
def cdo():
    """python-cdo instance, created once per module"""
    return Cdo()


@pytest.mark.parametrize("grid_str,expected", [
    ("global_1.0", True),        # Global regular grid
    ("dcw:US", True),            # Regional regular grid
    ("zonal_2.5", True),         # Zonal latitudes
    ("r360x180", True),          # Global regular NxM grid
    ("lon=-75.0/lat=40.0", True),  # One grid point
    ("F64", True),               # Full regular Gaussian grid
    ("N200", True),              # Gaussian reduced grid
    ("f64", True),               # Full regular Gaussian grid
    ("n200", True),              # Gaussian reduced grid
    ("O32", True),              # Octahedral grid
    ("gme10", True),             # Global icosahedral-hexagonal GME grid
    ("hp256", True),            # HEALPix grid
    ("hp32_ring", True),         # HEALPix grid
    ("hpz4", True),              # HEALPix zoom grid
    ("hpr6", True),             # HEALPix refinement grid
    ("random_string", False),    # Invalid CDO grid
    ("/path/to/file.nc", False)  # Invalid CDO grid (file path)
])
def test_is_cdo_grid_and_recognition(grid_str, expected, tmp_path, cdo):
    """Grid detection by smmregrid, and grid generation by CDO"""
    assert bool(CdoGrid(grid_str).grid_kind) == expected
    if expected and not grid_str.startswith(NO_CDO_CONST):
        output = str(tmp_path / "output.nc")
        cdo.const(f'1,{grid_str}', output=output, options="-f nc4")
        assert os.path.exists(output)


def test_healpix_zoom_deprecated():
    """The hpzN format is still recognised, but deprecated in favour of hprN"""
    with pytest.warns(DeprecationWarning, match="hprN"):
        assert CdoGrid("hpz4").grid_kind == "healpix_zoom"


def test_invalid_grid_type():
    """Test for invalid grid type"""
    with pytest.raises(TypeError):
        CdoGrid(12345)  # Passing an integer instead of a string


def test_print_repr():
    """Test the __repr__ method"""
    grid = CdoGrid("global_1.0")
    expected_repr = "CDOGrid(grid_str='global_1.0', grid_kind='global_regular')"
    assert repr(grid) == expected_repr
    assert grid.grid_kind == 'global_regular'
