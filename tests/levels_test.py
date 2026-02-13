"""Tests for smmregrid to verify level selection"""

import os
import pytest
from smmregrid.checker import check_cdo_regrid_levels

INDIR = 'tests/data'
tfile = os.path.join(INDIR, 'r360x180.nc')

@pytest.mark.parametrize("method", ['bil'])
def test_plev_gaussian_levels(method):
    """Pressure levels on gaussian grid test with level selection"""
    fff = check_cdo_regrid_levels(os.path.join(INDIR, 'ua-ecearth.nc'), tfile,
                                  "plev", [14, 15, 17], remap_method=method)
    assert fff is True


@pytest.mark.parametrize("method", ['con'])
def test_nemo_3d_single_level(method):
    """NEMO 3D levels test with level selection"""
    fff = check_cdo_regrid_levels(os.path.join(INDIR, 'so3d-nemo.nc'), tfile,
                                  "lev", 16, remap_method=method)  # single level
    assert fff is True, "Single selected 3D level test failed"

@pytest.mark.parametrize("method", ['con'])
def test_nemo_3d_multiple_levels(method):
    """NEMO 3D levels test with level selection"""
    fff = check_cdo_regrid_levels(os.path.join(INDIR, 'so3d-nemo.nc'), tfile,
                                  "lev", [14, 17], remap_method=method)
    assert fff is True, "Multiple levels test failed"

@pytest.mark.parametrize("method", ['con'])
def test_nemo_3d_single_level_extrapolate(method):
    """NEMO 3D levels test with level selection and extrapolation"""
    fff = check_cdo_regrid_levels(os.path.join(INDIR, 'so3d-nemo.nc'), tfile,
                                  "lev", [15], remap_method=method)  # single level with slice
    assert fff is True, "Single sliced 3D level test failed"
