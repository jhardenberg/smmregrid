"""Tests for the GridCoordinates class and its use in GridDetector"""

import dask.array
import numpy as np
import pytest
import xarray
from smmregrid import GridDetector
from smmregrid.gridcoordinates import GridCoordinates


def make_coords(lat_attrs=None, lon_attrs=None):
    """Dataset with only clat/clon coordinates on an unstructured ncells dimension"""
    data = np.arange(6.) - 3  # 6 cells: not a HEALPix size
    return xarray.Dataset(coords={'clat': ('ncells', data, lat_attrs or {}), 'clon': ('ncells', data, lon_attrs or {})})


@pytest.mark.parametrize("file,var,horizontal_dims", [
    ("2t-era5.nc", None, ["lat", "lon"]),
    ("so3d-nemo.nc", None, ["j", "i"]),  # i/j have axis X/Y but are not lat/lon
    ("temp3d-fesom.nc", None, ["nod2"]),
    ("ua-so_mix_ecearth.nc", "so", ["j", "i"]),
    ("healpix_0.nc", None, None),
])
def test_gridcoordinates_files(file, var, horizontal_dims):
    """Horizontal dimensions are found from the coordinates of the test files"""
    xfield = xarray.open_dataset(f'tests/data/{file}')
    assert GridCoordinates((xfield[var] if var else xfield).coords).horizontal_dims == horizontal_dims


@pytest.mark.parametrize("lat_attrs,lon_attrs,source", [
    ({'standard_name': 'latitude'}, {'standard_name': 'longitude'}, 'standard_name'),
    ({'units': 'degrees_north'}, {'units': 'degree_E'}, 'units'),
    ({'_CoordinateAxisType': 'Lat'}, {'_CoordinateAxisType': 'Lon'}, 'axis_type'),
])
def test_gridcoordinates_attributes(lat_attrs, lon_attrs, source):
    """CF attributes identify coordinates with unknown names, e.g. ICON clat/clon"""
    coords = GridCoordinates(make_coords(lat_attrs, lon_attrs))
    assert (coords.lat, coords.lon, coords.horizontal_dims) == ('clat', 'clon', ['ncells'])
    assert coords.source == {'lat': source, 'lon': source}


def test_gridcoordinates_no_match():
    """Unknown, scalar and incompatible coordinates give no latitude/longitude"""
    assert GridCoordinates(make_coords()).lat is None
    assert GridCoordinates(xarray.Dataset(coords={'lat': 10., 'lon': 20.})).lat is None
    mixed = xarray.Dataset(coords={'lat': ('y', np.zeros(3)), 'lon': (('y', 'x'), np.zeros((3, 4)))})
    assert GridCoordinates(mixed).horizontal_dims is None


def test_gridcoordinates_lazy():
    """Dask-backed coordinates are never computed"""
    def fail(block):
        raise RuntimeError('coordinate values should not be computed')
    lazy = dask.array.zeros((3, 4), chunks=-1).map_blocks(fail, dtype=float)
    ds = xarray.Dataset(coords={'lat': (('j', 'i'), lazy), 'lon': (('j', 'i'), lazy)})
    assert GridCoordinates(ds).horizontal_dims == ['j', 'i']


def test_griddetector_coordinates():
    """GridDetector finds latitude/longitude through GridCoordinates, user names first"""
    ds = make_coords({'standard_name': 'latitude'}, {'standard_name': 'longitude'})
    assert GridDetector().detect_grid(ds) == 'Unstructured'
    assert GridDetector().detect_grid(make_coords()) == 'Unknown'
    assert GridDetector(lat='clat', lon='clon').detect_grid(make_coords()) == 'Unstructured'


ERA5 = xarray.open_dataset('tests/data/2t-era5.nc')['2t']
LAT01 = np.arange(300, 400, dtype='float32') / 10


@pytest.mark.parametrize("data,kind", [
    (ERA5.isel(lat=[10]), 'Regular'),  # single latitude
    (ERA5.roll(lon=10, roll_coords=True).isel(lon=slice(0, 20)), 'Regular'),  # across the dateline
    (xarray.DataArray(np.zeros((100, 100)), dims=('lat', 'lon'), coords={'lat': LAT01, 'lon': LAT01}), 'Regular'),
    (ERA5.assign_coords(lat=ERA5.lat ** 3 / 8100), 'UndefinedRegular'),  # irregular, not Gaussian
    (xarray.DataArray(np.zeros(12), dims='time'), 'Unknown'),  # 12 is a HEALPix size, but time is not
    (xarray.DataArray(np.zeros(24), dims='ncells'), 'Unknown'),  # 24 is not a HEALPix size
    (xarray.DataArray(np.zeros(48), dims='ncells'), 'HEALPix'),
])
def test_griddetector_kind(data, kind):
    """Grid kind on edge cases of regular, Gaussian and HEALPix detection"""
    assert GridDetector().detect_grid(data) == kind
