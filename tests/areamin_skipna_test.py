"""remap_area_min must act in the same way with and without skipna.

CDO masks the target cells whose area covered by valid source points is below remap_area_min, using the
mask of the data. Without skipna the weights are generated with that mask, so dst_grid_frac already
accounts for it; with skipna the weights come from a grid without mask, and the covered fraction
has to be computed at each call from the valid source points.
"""

import pytest
import xarray as xr
from smmregrid import Regridder
from checker import check_cdo_regrid

INDIR = 'tests/data'


# 0.5 is not used on onlytos-ipsl and berkley: some target cells are covered by exactly 50%, and the
# result for them depends on floating point rounding
@pytest.mark.parametrize("filein,remap_area_min", [
    ('so3d-nemo.nc', 0.5), ('so3d-nemo.nc', 0.9),
    ('onlytos-ipsl.nc', 0.75), ('onlytos-ipsl.nc', 0.9),
    ('berkley.nc', 0.6), ('berkley.nc', 0.9),
])
def test_skipna_area_min_cdo(filein, remap_area_min):
    """Conservative remapping with skipna and remap_area_min against CDO (2D and 3D, masked data)"""
    assert check_cdo_regrid(f'{INDIR}/{filein}', 'r180x90', skipna=True, remap_area_min=remap_area_min)


@pytest.mark.parametrize("filein", ['regional.nc', 'onlytos-ipsl.nc'])
def test_skipna_area_min_destarea_cdo(filein):
    """With destarea the covered fraction is the sum of the valid weights"""
    assert check_cdo_regrid(f'{INDIR}/{filein}', 'r100x50', init_method='weights', remap_norm='destarea',
                            skipna=True, remap_area_min=0.5)


@pytest.mark.parametrize("method", ['nn', 'bil'])
def test_skipna_area_min_not_conservative(method):
    """remap_area_min has no effect on the other methods, as in CDO"""
    filein = f'{INDIR}/onlytos-ipsl.nc'
    data = xr.open_dataset(filein)
    out = [Regridder(source_grid=filein, target_grid='r180x90', method=method, skipna=True,
                     remap_area_min=remap_area_min, loglevel='warning').regrid(data)['tos'].compute()
           for remap_area_min in (0.0, 0.9)]
    xr.testing.assert_equal(out[0], out[1])


@pytest.mark.parametrize("remap_area_min", [0.5, 0.9])
def test_skipna_3d_same_as_noskipna(remap_area_min):
    """3D data with skipna (a single weights matrix) as without skipna (weights of each level)"""
    filein = f'{INDIR}/so3d-nemo.nc'
    data = xr.open_dataset(filein)
    out = Regridder(source_grid=filein, target_grid='r180x90', skipna=True, remap_area_min=remap_area_min,
                    loglevel='warning').regrid(data)['so']
    reference = Regridder(source_grid=filein, target_grid='r180x90', skipna=False, remap_area_min=remap_area_min,
                          loglevel='warning').regrid(data)['so']
    xr.testing.assert_allclose(out.compute(), reference.compute())
