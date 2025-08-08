"""Tests for smmregrid to verify level selection"""

import os
import pytest
from smmregrid.checker import check_cdo_regrid_levels

INDIR = 'tests/data'
tfile = os.path.join(INDIR, 'r360x180.nc')

# test for pressure levels on gaussian grid (3D)


@pytest.mark.parametrize("method", ['con'])
def test_plev_gaussian_levels(method):
    fff = check_cdo_regrid_levels(os.path.join(INDIR, 'ua-ecearth.nc'), tfile,
                                  "plev", [14, 15, 17], remap_method=method)
    assert fff is True

# 3D NEMO grid test


@pytest.mark.parametrize("method", ['nn'])
def test_nemo_3d_levels(method):
    fff = check_cdo_regrid_levels(os.path.join(INDIR, 'so3d-nemo.nc'), tfile,
                                  "lev", [14, 15, 17], remap_method=method)
    assert fff is True, "Multiple levels test failed"

    fff = check_cdo_regrid_levels(os.path.join(INDIR, 'so3d-nemo.nc'), tfile,
                                  "lev", [15], remap_method=method)  # single level
    assert fff is True, "Single 3D level test failed"
