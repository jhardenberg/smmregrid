"""Tests for util module"""

import os
import pytest
import xarray
from smmregrid.util import is_cdo_grid, check_gridfile, detect_grid

@pytest.mark.parametrize("grid_str,expected", [
    ("global_1.0", True),        # Global regular grid
    ("dcw:US", True),            # Regional regular grid
    ("zonal_2.5", True),         # Zonal latitudes
    ("r360x180", True),          # Global regular NxM grid
    ("lon=-75.0/lat=40.0", True),# One grid point
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
    assert is_cdo_grid(grid_str) == expected

# Mock function to avoid actual file system checks for testing purposes
def mock_exists(path):
    return path == "/valid/file/path.nc"

# Patch os.path.exists to use the mock function
@pytest.fixture(autouse=True)
def mock_os_path_exists(monkeypatch):
    monkeypatch.setattr(os.path, "exists", mock_exists)

@pytest.mark.parametrize("filename,expected,raises", [
    (None, None, None),                                  # None input
    (xarray.Dataset(), "xarray", None),                      # xarray.Dataset input
    (xarray.DataArray(), "xarray", None),                    # xarray.DataArray input
    ("global_1.0", "grid", None),                        # CDO grid string
    ("r360x180", "grid", None),                          # CDO grid string
    ("/valid/file/path.nc", "file", None),               # Valid file path
    ("/invalid/file/path.nc", None, FileNotFoundError),  # Invalid file path
    (123, None, TypeError),                              # Invalid type (not supported)
])
def test_check_gridfile(filename, expected, raises):
    """simple test for check grid file"""
    if raises:
        with pytest.raises(raises):
            check_gridfile(filename)
    else:
        assert check_gridfile(filename) == expected

# Define test cases
TEST_FILES = [
    ("2t-era5.nc", "Regular"),
    ("mix-cesm.nc", "Regular"),
    ("r360x180.nc", "Regular"),
    ("so3d-nemo.nc", "Curvilinear"),
    ("tas-healpix2.nc", "Healpix"),
    ("tos-fesom.nc", "Unstructured"),
    ("ua-so_mix_ecearth.nc", "GaussianRegular"),
    ("healpix_0.nc", "Unknown"),
    ("onlytos-ipsl.nc", "Curvilinear"),
    ("regional.nc", "Regular"),
    ("tas-ecearth.nc", "GaussianRegular"),
    ("temp3d-fesom.nc", "Unstructured"),
    ("ua-ecearth.nc", "GaussianRegular"),
    ("lsm-ifs.grb", "GaussianReduced"),
]
@pytest.mark.parametrize("file_name, expected_grid", TEST_FILES)
def test_detect_grid(file_name, expected_grid):
    """Test for grid format detection"""
    filename = os.path.join('tests/data', file_name)
    xfield = xarray.open_mfdataset(filename)
    detected_grid = detect_grid(xfield)
    assert detected_grid == expected_grid, f"File {file_name}: expected {expected_grid}, got {detected_grid}"
