"""Tests for GridDetector.detect_cdo_grid: CDO grid names matched exactly from the coordinates"""

import os
import subprocess
import numpy
import pytest
import xarray
from smmregrid import GridDetector, CdoGrid
from smmregrid.cdobinary import resolve_cdo


def cdo_grid(tmp_path, grid, fmt='nc4'):
    """Write a constant field on a CDO named grid and open it"""
    suffix = 'grb' if fmt.startswith('grb') else 'nc'
    filename = os.path.join(tmp_path, f'{grid}.{suffix}')
    subprocess.run([resolve_cdo(), '-s', '-f', fmt, '-setname,t', f'-const,1,{grid}', filename], check=True)
    if suffix == 'grb':
        return xarray.open_dataset(filename, engine='cfgrib', backend_kwargs={'indexpath': ''})
    return xarray.open_dataset(filename)


@pytest.mark.parametrize("grid", [
    'r360x180', 'r144x73', 'r10x5', 'r10x6',  # even/odd number of latitudes: cell centres/poles
    'global_2', 'global_36',
    'F4', 'F128',
    'hp2_nested', 'hp2_ring', 'hp32_ring',
])
def test_cdo_roundtrip(tmp_path, grid):
    """A field on a CDO grid gives back the name of that grid"""
    data = cdo_grid(tmp_path, grid)
    grid_str = GridDetector().detect_cdo_grid(data)
    assert grid_str == grid
    assert CdoGrid(grid_str).grid_kind is not None


def test_cdo_roundtrip_octahedral(tmp_path):
    """Octahedral reduced Gaussian grid, with one lat/lon per point as read from GRIB"""
    pytest.importorskip('cfgrib')
    data = cdo_grid(tmp_path, 'O4', fmt='grb2')
    assert GridDetector().detect_grid(data) == 'GaussianReduced'
    assert GridDetector().detect_cdo_grid(data) == 'O4'


@pytest.mark.parametrize("grid, transform", [
    ('r360x180', lambda ds: ds.isel(lat=slice(None, None, -1))),  # lat north to south
    ('r360x180', lambda ds: ds.roll(lon=180, roll_coords=True).assign_coords(
        lon=((ds.lon.roll(lon=180) + 180) % 360) - 180)),  # lon -180..180
    ('r360x180', lambda ds: ds.isel(lon=slice(0, 90))),  # regional
    ('F128', lambda ds: ds.isel(lat=slice(None, None, -1))),  # Gaussian south to north
    ('hp2_ring', lambda ds: ds.drop_vars('healpix')),  # no indexing scheme, no latitudes
])
def test_no_exact_match(tmp_path, grid, transform):
    """Same geometry with a different layout of the points has no CDO name"""
    data = transform(cdo_grid(tmp_path, grid))
    assert GridDetector().detect_cdo_grid(data) is None


def test_float32_coords(tmp_path):
    """Coordinates stored as float32 still match"""
    data = cdo_grid(tmp_path, 'r360x180')
    data = data.assign_coords(lat=data.lat.astype('float32'), lon=data.lon.astype('float32'))
    assert GridDetector().detect_cdo_grid(data) == 'r360x180'


def test_healpix_order_from_latitudes():
    """Without metadata, the HEALPix order follows the latitudes: never increasing for ring"""
    lat = numpy.linspace(80, -80, 48)
    ring = xarray.Dataset({'t': ('cell', numpy.zeros(48))}, coords={'lat': ('cell', lat), 'lon': ('cell', lat)})
    assert GridDetector().detect_cdo_grid(ring) == 'hp2_ring'
    nested = ring.assign_coords(lat=('cell', numpy.roll(lat, 5)))
    assert GridDetector().detect_cdo_grid(nested) == 'hp2_nested'


def test_healpix_order_attribute():
    """healpix_order ('nest'/'ring') of a grid mapping variable is read as well"""
    data = xarray.Dataset({'t': ('cell', numpy.zeros(48), {'grid_mapping': 'crs'}),
                           'crs': ((), 0, {'grid_mapping_name': 'healpix', 'healpix_order': 'nest'})})
    assert GridDetector().detect_cdo_grid(data) == 'hp2_nested'


@pytest.mark.parametrize("file_name, expected", [
    ("r360x180.nc", "r360x180"),
    ("2t-era5.nc", "r144x73"),
    ("ua-ipsl.nc", "r144x143"),
    ("tas-healpix2.nc", "hp32_nested"),
    ("tas-ecearth.nc", None),  # Gaussian south to north: CDO F128 is north to south
    ("regional.nc", None),
    ("onlytos-ipsl.nc", None),  # curvilinear
    ("tos-fesom.nc", None),  # unstructured
])
def test_detect_cdo_grid_files(file_name, expected):
    """CDO grid names of the test files"""
    data = xarray.open_dataset(os.path.join('tests/data', file_name))
    assert GridDetector().detect_cdo_grid(data) == expected


def test_kind_argument(tmp_path):
    """An explicit kind skips detect_grid; kinds without a CDO name give None"""
    data = cdo_grid(tmp_path, 'r360x180')
    assert GridDetector().detect_cdo_grid(data, kind='Regular') == 'r360x180'
    assert GridDetector().detect_cdo_grid(data, kind='Curvilinear') is None
