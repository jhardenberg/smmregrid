"""Module for cdo-smmregrid comparison"""
"""Module for cdo-smmregrid comparison"""

import numpy as np
import xarray as xr
from cdo import Cdo
from smmregrid import Regridder, CdoGenerate
from smmregrid.regrid import DEFAULT_AREA_MIN, DEFAULT_NA_THRES

cdo = Cdo()


def find_var(xfield):
    """Find them most likely set of vars that needs to be interpolated"""

    # var as the one which have time and not have bnds (could work)
    myvar = [var for var in xfield.data_vars
             if 'time' in xfield[var].dims and 'bnds' not in xfield[var].dims]

    # if find none otherwise, pick what you have
    if not myvar:
        myvar = list(xfield.data_vars)

    return myvar


def _cdo_remap(remap_method, ftarget, finput, extrapolate=True, remap_area_min=None):
    """Run the reference CDO remapping, returning an xarray Dataset.

    remap_area_min is only passed to CDO's environment when explicitly
    given, since it is not supported by every CDO version.
    """
    cdo_interpolator = getattr(cdo, 'remap' + remap_method)
    env = {'REMAP_EXTRAPOLATE': 'on' if extrapolate else 'off'}
    if remap_area_min is not None:
        env['REMAP_AREA_MIN'] = str(remap_area_min)
    return cdo_interpolator(ftarget, input=finput, returnXDataset=True, env=env)


def _build_interpolator(finput, ftarget, remap_method, init_method, mask_dim,
                        remap_area_min, skipna, loglevel, na_thres):
    """Build a smmregrid Regridder either from grids or from pre-computed weights."""

    if init_method == 'grids':
        return Regridder(source_grid=finput, target_grid=ftarget,
                         remap_area_min=remap_area_min, method=remap_method,
                         mask_dim=mask_dim, skipna=skipna, loglevel=loglevel,
                         na_thres=na_thres)
    if init_method == 'weights':
        wfield = CdoGenerate(finput, ftarget, loglevel=loglevel, skipna=skipna,
                             na_thres=na_thres).weights(method=remap_method, mask_dim=mask_dim)
        return Regridder(weights=wfield, loglevel=loglevel,
                         remap_area_min=remap_area_min, skipna=skipna, na_thres=na_thres)

    raise KeyError('Unsupported init method')


def _report_diagnostics(cdofield, rfield):
    """Print shape/coordinate/NaN diagnostics for a pair of squeezed fields."""

    print('Shape of CDO field and sizes:', cdofield.shape, cdofield.sizes)
    print('CDO coordinates min and max:',
          {dim: (cdofield[dim].min().values, cdofield[dim].max().values) for dim in cdofield.dims})
    print('Shape of regrid field and sizes:', rfield.shape, rfield.sizes)
    print('Regrid coordinates min and max:',
          {dim: (rfield[dim].min().values, rfield[dim].max().values) for dim in rfield.dims})
    print('Max diff', np.max(np.abs(cdofield - rfield)).values)
    print('NaN count in CDO', cdofield.isnull().sum().values)
    print('NaN count in regrid', rfield.isnull().sum().values)


def _compare(xfield, cdofield, interpolator, access='Dataset'):
    """Regrid xfield with the given interpolator, compare against cdofield,
    print diagnostics and return whether the two fields match."""

    smmvar = find_var(xfield)
    cdovar = find_var(cdofield)

    rfield = interpolator.regrid(xfield)

    if access == 'Dataset':
        rfield = rfield[smmvar].to_array()
        cdofield = cdofield[cdovar].to_array()

    checker = np.allclose(cdofield, rfield, equal_nan=True)

    cdofield = cdofield.squeeze(drop=True)
    rfield = rfield.squeeze(drop=True)
    _report_diagnostics(cdofield, rfield)

    return checker


def check_cdo_regrid(finput, ftarget, remap_method='con', access='Dataset',
                     init_method='grids', mask_dim=None, extrapolate=True,
                     remap_area_min=DEFAULT_AREA_MIN,
                     loglevel='INFO', skipna=False, na_thres=DEFAULT_NA_THRES):
    """Given a file to be interpolated finput over the ftarget grid,
    check if the output of the last variable is the same as produced
    by CDO remap command. This function is used for tests."""

    # define files and open input file
    xfield = xr.open_mfdataset(finput) if isinstance(finput, str) else finput

    # interpolation with pure CDO
    # NB: REMAP_AREA_MIN is not passed here, as it does not work with cdo 2.4.4
    cdofield = _cdo_remap(remap_method, ftarget, finput, extrapolate=extrapolate)

    # method with automatic creation of weights (or pre-computed weights)
    interpolator = _build_interpolator(finput, ftarget, remap_method, init_method,
                                       mask_dim, remap_area_min, skipna, loglevel, na_thres)

    return _compare(xfield, cdofield, interpolator, access=access)


def check_cdo_regrid_levels(finput, ftarget, mask_dim, levels, remap_method='con',
                            remap_area_min=DEFAULT_AREA_MIN, access='Dataset',
                            extrapolate=True, loglevel='INFO', skipna=False,
                            na_thres=DEFAULT_NA_THRES):
    """Given a file to be interpolated finput over the ftarget grid,
    check if the output of the last variable is the same as produced
    by CDO remap command. This function is used for tests.
    This is a variant to check the level memory feature (idx_3d)."""

    # define files and open input file
    xfield = xr.open_mfdataset(finput)

    # interpolation with pure CDO, keeping only the selected levels
    cdofield = _cdo_remap(remap_method, ftarget, finput, extrapolate=extrapolate,
                          remap_area_min=remap_area_min)
    cdofield = cdofield.isel(**{mask_dim: levels})

    # weights must be pre-computed (full 3D weights) to exercise level memory
    interpolator = _build_interpolator(finput, ftarget, remap_method, 'weights',
                                       mask_dim, remap_area_min, skipna, loglevel, na_thres)

    # subselect the same levels on the input before regridding
    xfield = xfield.isel(**{mask_dim: levels})

    return _compare(xfield, cdofield, interpolator, access=access)