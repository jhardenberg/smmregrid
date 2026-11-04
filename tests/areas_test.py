"""Set of basic tests for smmregrid"""

import os
import numpy as np
import pytest
import xarray as xr
from smmregrid import CdoGenerate, Regridder


INDIR = 'tests/data'
tfile = os.path.join(INDIR, 'r360x180.nc')
EARTH_SURFACE = 5.101 * 1e8  # km2
OCEAN_SURFACE = 3.6 * 1e8
RELTOL = 0.02  # %2 percent margin
TOLERANCE = EARTH_SURFACE * RELTOL


@pytest.mark.parametrize("filename,shape,kind", [
    ('2t-era5.nc', (73, 144), "global"),
    ('onlytos-ipsl.nc', (332, 362), "global"),
    ('tos-fesom.nc', (126859,), "ocean"),
    ('tas-healpix2.nc', (12288,), "global"),
    ('so3d-nemo.nc', (292, 362), "global")
])
def test_basic_areas_source(filename, shape, kind):
    """Basic test for area generation from source file"""
    generator = CdoGenerate(os.path.join(INDIR, filename), tfile, loglevel='debug')
    area = generator.areas()
    compare_surface = EARTH_SURFACE if kind == "global" else OCEAN_SURFACE
    earth_surface = area.cell_area.values.sum() / 1e6
    assert area.cell_area.shape == shape
    assert earth_surface == pytest.approx(compare_surface, abs=TOLERANCE)


@pytest.mark.parametrize("filename,shape", [
    (os.path.join(INDIR, 'r360x180.nc'), (180, 360)),
    ('hp32', (12288,)),
    ('r180x90', (90, 180,)),
    (os.path.join(INDIR, 'tas-healpix2.nc'), (12288,))
])
def test_basic_areas_target(filename, shape):
    """Basic test for area generation from source file"""
    infile = os.path.join(INDIR, 'r360x180.nc')
    generator = CdoGenerate(infile, filename, loglevel='debug')
    area = generator.areas(target=True)
    earth_surface = area.cell_area.values.sum() / 1e6
    assert area.cell_area.shape == shape
    assert earth_surface == pytest.approx(EARTH_SURFACE, abs=TOLERANCE)


def test_nosource_areas_target():
    """Basic test for area generation from source file"""
    generator = CdoGenerate(source_grid=None, target_grid='r180x91', loglevel='debug')
    area = generator.areas(target=True)
    earth_surface = area.cell_area.values.sum() / 1e6
    assert area.cell_area.shape == (91, 180)
    assert earth_surface == pytest.approx(EARTH_SURFACE, abs=TOLERANCE)


@pytest.mark.parametrize("target", ['F32', 'F128', 'r180x91'])
def test_areas_coords_aligned_with_regrid(target):
    """Lat/lon of target areas and of regridded data are identical (#51)"""
    generator = CdoGenerate(tfile, target, loglevel='debug')
    weights = generator.weights(method='bil')
    area = generator.areas(target=True)
    out = Regridder(weights=weights).regrid(xr.open_dataset(tfile))
    for coord in ['lat', 'lon']:
        assert np.array_equal(out[coord].values, area[coord].values)
