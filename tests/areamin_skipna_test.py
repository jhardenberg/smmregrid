"""remap_area_min must act in the same way with and without skipna.

CDO masks the target cells whose area covered by valid source points is below remap_area_min, using the
mask of the data. Without skipna the weights are generated with that mask, so dst_grid_frac already
accounts for it; with skipna the weights come from a grid without mask, and the covered fraction
has to be computed at each call from the valid source points.
"""

import dask.array
import numpy
import pytest
import xarray as xr
from smmregrid import Regridder
from smmregrid.weights import mask_na_thres, mask_area_min
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


def test_mask_helpers():
    """The two masks of Regridder: na_thres on the missing weights, remap_area_min on the covered area"""
    target = dask.array.from_array(numpy.array([1.0, 2.0, 3.0, 4.0]), chunks=2)
    valid = dask.array.from_array(numpy.array([1.0, 0.6, 0.5, 0.2]), chunks=2)

    # more than 40% of the weights missing: masked (strictly larger)
    out = mask_na_thres(target, valid, 0.4).compute()
    assert numpy.array_equal(numpy.isnan(out), [False, False, True, True])
    # na_thres=1: nothing is masked
    assert not numpy.isnan(mask_na_thres(target, valid, 1.0).compute()).any()
    # as a 0/1 mask, as used for the target mask without skipna
    assert mask_na_thres(1, valid, 0.4, fill_value=0).compute().tolist() == [1, 1, 0, 0]

    # covered area below 0.5: masked (strictly smaller)
    out = mask_area_min(target, valid, 0.5).compute()
    assert numpy.array_equal(numpy.isnan(out), [False, False, False, True])
    # remap_area_min=0: untouched, and covered_fraction is not needed
    assert mask_area_min(target, None, 0.0) is target


def _skipna_regrid(filein, var, method, **kwargs):
    """Regrid the first time step of a variable with skipna, from the grid of the same file"""
    data = xr.open_dataset(f'{INDIR}/{filein}')[var].isel(time=0)
    regridder = Regridder(source_grid=f'{INDIR}/{filein}', target_grid='r180x90', method=method, skipna=True,
                          loglevel='ERROR', **kwargs)
    return regridder.regrid(data)


# thresholds other than 0.5, where the result of some cells depends on floating point rounding
@pytest.mark.parametrize("filein,var", [('so3d-nemo.nc', 'so'), ('onlytos-ipsl.nc', 'tos')])
@pytest.mark.parametrize("threshold", [0.3, 0.8])
def test_na_thres_equals_area_min_con(filein, var, threshold):
    """With fracarea, the covered area of a target cell is the share of its weights on valid source points:
    na_thres = 1 - remap_area_min gives the same result"""
    by_na_thres = _skipna_regrid(filein, var, 'con', na_thres=1.0 - threshold, remap_area_min=0.0)
    by_area_min = _skipna_regrid(filein, var, 'con', na_thres=1.0, remap_area_min=threshold)
    unmasked = _skipna_regrid(filein, var, 'con', na_thres=1.0, remap_area_min=0.0)
    assert by_area_min.isnull().sum() > unmasked.isnull().sum()    # the threshold is not irrelevant
    xr.testing.assert_equal(by_na_thres, by_area_min)


def test_na_thres_equals_area_min_nn():
    """With nn the share of valid weights of a target cell is 0 or 1: neither threshold masks anything"""
    unmasked = _skipna_regrid('onlytos-ipsl.nc', 'tos', 'nn', na_thres=1.0, remap_area_min=0.0)
    xr.testing.assert_equal(_skipna_regrid('onlytos-ipsl.nc', 'tos', 'nn', na_thres=0.2, remap_area_min=0.0), unmasked)
    xr.testing.assert_equal(_skipna_regrid('onlytos-ipsl.nc', 'tos', 'nn', na_thres=1.0, remap_area_min=0.8), unmasked)


def test_na_thres_differs_from_area_min_bil():
    """With bil the valid share of the weights is fractional: only na_thres masks target cells,
    remap_area_min is for conservative remapping and has no effect"""
    unmasked = _skipna_regrid('onlytos-ipsl.nc', 'tos', 'bil', na_thres=1.0, remap_area_min=0.0)
    by_na_thres = _skipna_regrid('onlytos-ipsl.nc', 'tos', 'bil', na_thres=0.2, remap_area_min=0.0)
    by_area_min = _skipna_regrid('onlytos-ipsl.nc', 'tos', 'bil', na_thres=1.0, remap_area_min=0.8)
    xr.testing.assert_equal(by_area_min, unmasked)
    assert by_na_thres.isnull().sum() > unmasked.isnull().sum()
