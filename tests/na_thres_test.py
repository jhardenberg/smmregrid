"""na_thres: maximum fraction of missing weights of a target cell, with and without skipna.

For the conservative methods the weights are overlap areas, so one minus the fraction of missing weights is
the fraction of the target cell covered by valid source points, the quantity that CDO compares with
remap_area_min: na_thres = 1 - remap_area_min. Without skipna the weights are generated with the mask of the
data, and that fraction is the dst_grid_frac of the weights; with skipna the weights come from a grid without
mask, and the fraction is computed at each call from the valid source points.
"""

import logging
import dask.array
import numpy
import pytest
import xarray as xr
from smmregrid import Regridder
from smmregrid.util import resolve_na_thres
from smmregrid.weights import mask_na_thres, mask_tensordot
from checker import check_cdo_regrid

INDIR = 'tests/data'


def _regrid(filein, var, method, **kwargs):
    """Regrid the first time step of a variable from the grid of the same file"""
    kwargs.setdefault('loglevel', 'ERROR')
    data = xr.open_dataset(f'{INDIR}/{filein}')[var].isel(time=0)
    regridder = Regridder(source_grid=f'{INDIR}/{filein}', target_grid='r180x90', method=method, **kwargs)
    return regridder.regrid(data)


# 0.5 is not used: some target cells are covered by exactly 50%, and the result for them depends on
# floating point rounding (the covered fraction is 0.5 +- 1e-16, not the same in smmregrid and in CDO)
# berkley has a mask which varies in time: only skipna follows it, CDO recomputes the weights
@pytest.mark.parametrize("filein,remap_area_min,skipna", [
    ('so3d-nemo.nc', 0.6, False), ('so3d-nemo.nc', 0.9, False), ('so3d-nemo.nc', 0.6, True), ('so3d-nemo.nc', 0.9, True),
    ('onlytos-ipsl.nc', 0.75, False), ('onlytos-ipsl.nc', 0.9, False),
    ('onlytos-ipsl.nc', 0.75, True), ('onlytos-ipsl.nc', 0.9, True),
    ('berkley.nc', 0.6, True), ('berkley.nc', 0.9, True),
])
def test_na_thres_cdo_area_min(filein, remap_area_min, skipna):
    """Conservative remapping with na_thres = 1 - remap_area_min against CDO with remap_area_min (2D and 3D, masked data).
    The checker gives smmregrid na_thres = 1 - remap_area_min"""
    assert check_cdo_regrid(f'{INDIR}/{filein}', 'r180x90', skipna=skipna, remap_area_min=remap_area_min)


@pytest.mark.parametrize("filein,skipna", [
    ('so3d-nemo.nc', False), ('so3d-nemo.nc', True), ('onlytos-ipsl.nc', False), ('onlytos-ipsl.nc', True),
    ('berkley.nc', True),
])
def test_na_thres_cdo(filein, skipna):
    """na_thres='cdo' replicates CDO, where remap_area_min is 0: only the cells without any valid source are masked"""
    assert check_cdo_regrid(f'{INDIR}/{filein}', 'r180x90', skipna=skipna, na_thres='cdo')


@pytest.mark.parametrize("remap_area_min", [0.0, 0.5])
@pytest.mark.parametrize("filein", ['regional.nc', 'onlytos-ipsl.nc'])
def test_skipna_destarea_cdo(filein, remap_area_min):
    """With destarea the covered fraction is the sum of the valid weights (absolute area of the target cell)"""
    assert check_cdo_regrid(f'{INDIR}/{filein}', 'r100x50', init_method='weights', remap_norm='destarea',
                            skipna=True, remap_area_min=remap_area_min)


@pytest.mark.parametrize("na_thres", [0.4, 0.1])
def test_skipna_3d_same_as_noskipna(na_thres):
    """3D data with skipna (a single weights matrix) as without skipna (weights of each level)"""
    filein = f'{INDIR}/so3d-nemo.nc'
    data = xr.open_dataset(filein)
    out = Regridder(source_grid=filein, target_grid='r180x90', skipna=True, na_thres=na_thres,
                    loglevel='warning').regrid(data)['so']
    reference = Regridder(source_grid=filein, target_grid='r180x90', skipna=False, na_thres=na_thres,
                          loglevel='warning').regrid(data)['so']
    xr.testing.assert_allclose(out.compute(), reference.compute())


@pytest.mark.parametrize("skipna", [False, True])
def test_na_thres_auto_con(skipna):
    """The smmregrid default for the conservative method is na_thres 0.5, i.e. remap_area_min 0.5; it masks
    more than CDO, where remap_area_min is 0"""
    auto = _regrid('so3d-nemo.nc', 'so', 'con', skipna=skipna)
    xr.testing.assert_equal(auto, _regrid('so3d-nemo.nc', 'so', 'con', skipna=skipna, na_thres=0.5))
    assert auto.isnull().sum() > _regrid('so3d-nemo.nc', 'so', 'con', skipna=skipna, na_thres='cdo').isnull().sum()


@pytest.mark.parametrize("method", ['nn', 'bil', 'dis'])
def test_na_thres_no_effect_without_skipna(method, caplog):
    """Without skipna CDO decides which target cells are mapped when it generates the weights (any missing
    neighbour drops the cell for bil, dis renormalizes on the valid neighbours), and the weights hold no
    partial information: na_thres has no effect for the non-conservative methods, and a warning is logged"""
    reference = _regrid('onlytos-ipsl.nc', 'tos', method, skipna=False)
    logger = logging.getLogger('smmregrid.Regrid')
    logger.addHandler(caplog.handler)
    try:
        with caplog.at_level(logging.WARNING, logger='smmregrid.Regrid'):
            out = _regrid('onlytos-ipsl.nc', 'tos', method, skipna=False, na_thres=0.3, loglevel='WARNING')
    finally:
        logger.removeHandler(caplog.handler)
    xr.testing.assert_equal(out, reference)
    assert 'na_thres=0.3 has no effect' in caplog.text


@pytest.mark.parametrize("method", ['nn', 'bil', 'dis'])
def test_na_thres_not_conservative(method):
    """With nn the share of valid weights is 0 or 1, and na_thres does not change the result. With bil and
    dis it is a fraction: na_thres masks the cells with too much weight on missing neighbours (as in CDO
    only with the default for bil)"""
    unmasked = _regrid('onlytos-ipsl.nc', 'tos', method, skipna=True, na_thres=1.0)
    by_na_thres = _regrid('onlytos-ipsl.nc', 'tos', method, skipna=True, na_thres=0.2)
    if method == 'nn':
        xr.testing.assert_equal(by_na_thres, unmasked)
    else:
        assert by_na_thres.isnull().sum() > unmasked.isnull().sum()


def test_dis_without_skipna_same_as_skipna_auto():
    """With dis CDO keeps the valid ones of the nearest neighbours and renormalizes on them when it generates
    the weights with the mask: the same as skipna with na_thres=1.0, the 'auto' value for dis"""
    cdo_mask = _regrid('onlytos-ipsl.nc', 'tos', 'dis', skipna=False)
    renormalized = _regrid('onlytos-ipsl.nc', 'tos', 'dis', skipna=True, na_thres=1.0)
    xr.testing.assert_allclose(renormalized, cdo_mask)
    assert renormalized.isnull().sum() > 0


@pytest.mark.parametrize("skipna", [False, True])
@pytest.mark.parametrize("filein,var", [('so3d-nemo.nc', 'so'), ('onlytos-ipsl.nc', 'tos')])
def test_remap_area_min_deprecated(filein, var, skipna):
    """remap_area_min is deprecated and it is na_thres = 1 - remap_area_min for the conservative method"""
    with pytest.warns(DeprecationWarning, match='remap_area_min'):
        deprecated = _regrid(filein, var, 'con', skipna=skipna, remap_area_min=0.8)
    xr.testing.assert_equal(deprecated, _regrid(filein, var, 'con', skipna=skipna, na_thres=0.2))
    # the same value in the two arguments is fine, a different one is not
    with pytest.warns(DeprecationWarning):
        _regrid(filein, var, 'con', skipna=skipna, remap_area_min=0.8, na_thres=0.2)
    with pytest.warns(DeprecationWarning), pytest.raises(ValueError, match='conflict'):
        _regrid(filein, var, 'con', skipna=skipna, remap_area_min=0.8, na_thres=0.5)


@pytest.mark.parametrize("method", ['nn', 'bil'])
def test_remap_area_min_deprecated_not_conservative(method):
    """remap_area_min has no effect on the other methods, as in CDO"""
    reference = _regrid('onlytos-ipsl.nc', 'tos', method, skipna=True)
    with pytest.warns(DeprecationWarning):
        deprecated = _regrid('onlytos-ipsl.nc', 'tos', method, skipna=True, remap_area_min=0.9)
    xr.testing.assert_equal(deprecated, reference)


@pytest.mark.parametrize("na_thres,method,expected", [
    ('auto', 'con', 0.5), ('auto', 'con2', 0.5), ('auto', 'bil', 1e-6), ('auto', 'bic', 1e-6), ('auto', 'nn', 1.0),
    ('cdo', 'con', 1.0), ('cdo', 'bil', 1e-6), ('cdo', 'nn', 1.0), (0.3, 'bil', 0.3), ('0.3', 'con', 0.3),
])
def test_resolve_na_thres(na_thres, method, expected):
    """'auto' is the smmregrid default of the method, 'cdo' replicates CDO"""
    assert resolve_na_thres(na_thres, method) == expected


@pytest.mark.parametrize("na_thres", [-0.1, 1.5, 'wrong'])
def test_resolve_na_thres_invalid(na_thres):
    with pytest.raises(ValueError):
        resolve_na_thres(na_thres, 'con')


def test_mask_na_thres():
    """Target cells with more than na_thres missing weights are masked (strictly larger), and those without
    any valid weight always, even with na_thres=1"""
    target = dask.array.from_array(numpy.array([1.0, 2.0, 3.0, 4.0, 5.0]), chunks=2)
    valid = dask.array.from_array(numpy.array([1.0, 0.6, 0.5, 0.2, 0.0]), chunks=2)

    out = mask_na_thres(target, valid, 0.4).compute()
    assert numpy.array_equal(numpy.isnan(out), [False, False, True, True, True])
    out = mask_na_thres(target, valid, 1.0).compute()
    assert numpy.array_equal(numpy.isnan(out), [False, False, False, False, True])
    # as a 0/1 mask
    assert mask_na_thres(1, valid, 0.4, fill_value=0).compute().tolist() == [1, 1, 0, 0, 0]


def test_mask_tensordot():
    """The target mask of the weights marks the cells with at least one link from a valid source point,
    without any threshold: the second target cell is linked only to the masked source point"""
    src_mask = numpy.array([1, 0, 1])
    weights_matrix = numpy.array([[0.5, 0.0, 0.0], [0.5, 1.0, 0.0], [0.0, 0.0, 1.0]])
    assert mask_tensordot(src_mask, weights_matrix).compute().tolist() == [1, 0, 1]
