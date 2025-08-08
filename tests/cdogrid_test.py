import pytest
from smmregrid import CdoGrid


@pytest.mark.parametrize("grid_str,expected", [
    ("global_1.0", True),        # Global regular grid
    ("dcw:US", True),            # Regional regular grid
    ("zonal_2.5", True),         # Zonal latitudes
    ("r360x180", True),          # Global regular NxM grid
    ("lon=-75.0/lat=40.0", True),  # One grid point
    ("F64", True),               # Full regular Gaussian grid
    ("n400", True),              # Full regular Gaussian grid
    ("gme10", True),             # Global icosahedral-hexagonal GME grid
    ("hp1024", True),            # HEALPix grid
    ("hp32_ring", True),         # HEALPix grid
    ("hpz4", True),              # HEALPix zoom grid
    ("random_string", False),    # Invalid CDO grid
    ("/path/to/file.nc", False)  # Invalid CDO grid (file path)
])
def test_is_cdo_grid(grid_str, expected):
    """Simple test for cdo grid detection"""
    assert bool(CdoGrid(grid_str).grid_kind) == expected


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
