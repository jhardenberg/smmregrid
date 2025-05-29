"""Module for cdo-smmregrid comparison"""

import numpy as np
import xarray as xr
from cdo import Cdo
from smmregrid import Regridder, CdoGenerate
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


def check_cdo_regrid(finput, ftarget, remap_method='con', access='Dataset',
                     init_method='grids', vertical_dim=None, extrapolate=True,
                     remap_area_min=0.0, loglevel='INFO'):
    """Given a file to be interpolated finput over the ftarget grid,
    check if the output of the last variable is the same as produced
    by CDO remap command. This function is used for tests."""

    # define files and open input file
    if isinstance(finput, str):
        xfield = xr.open_mfdataset(finput)
    else:
        xfield = finput

    # convert extrapolation to cdo!
    cdoextrapolate = 'on' if extrapolate else 'off'

    # interpolation with pure CDO
    cdo_interpolator = getattr(cdo, 'remap' + remap_method)
    cdofield = cdo_interpolator(ftarget, input=finput, returnXDataset=True,
                                env={'REMAP_EXTRAPOLATE': cdoextrapolate})
                                     #'REMAP_AREA_MIN': remap_area_min}) # this is not working with cdo 2.4.4

    # var as the one which have time and not have bnds (could work)
    smmvar = find_var(xfield)
    cdovar = find_var(cdofield)

    # method with automatic creation of weights
    if init_method == 'grids':
        interpolator = Regridder(source_grid=finput, target_grid=ftarget, remap_area_min=remap_area_min,
                                 method=remap_method, vertical_dim=vertical_dim, loglevel=loglevel)
    elif init_method == 'weights':
        wfield = CdoGenerate(finput, ftarget, loglevel=loglevel).weights(method=remap_method, vertical_dim=vertical_dim)
        interpolator = Regridder(weights=wfield, loglevel=loglevel, remap_area_min=remap_area_min)
    else:
        raise KeyError('Unsupported init method')
    rfield = interpolator.regrid(xfield)

    if access == 'Dataset':
        rfield = rfield[smmvar].to_array()
        cdofield = cdofield[cdovar].to_array()

    # check if arrays are equal with numerical tolerance
    checker = np.allclose(cdofield, rfield, equal_nan=True)
    return checker


def check_cdo_regrid_levels(finput, ftarget, vertical_dim, levels, remap_method='con',
                            remap_area_min=0.5, access='Dataset',
                            extrapolate=True, loglevel='INFO'):
    """Given a file to be interpolated finput over the ftarget grid,
    check if the output of the last variable is the same as produced
    by CDO remap command. This function is used for tests.
    This is a variant to check the level memory feature (idx_3d)."""

    # define files and open input file
    xfield = xr.open_mfdataset(finput)

    # convert extrapolate to CDO
    cdoextrapolate = 'on' if extrapolate else 'off'

    # interpolation with pure CDO
    cdo_interpolator = getattr(cdo, 'remap' + remap_method)
    cdofield = cdo_interpolator(ftarget, input=finput, returnXDataset=True,
                                env={'REMAP_EXTRAPOLATE': cdoextrapolate,
                                    'REMAP_AREA_MIN': str(remap_area_min)})

    # Keep only some levels
    cdofield = cdofield.isel(**{vertical_dim: levels})

    # var as the one which have time and not have bnds (could work)
    smmvar = find_var(xfield)
    cdovar = find_var(cdofield)

    # compute weights
    wfield = CdoGenerate(finput, ftarget, loglevel=loglevel).weights(
        method=remap_method, vertical_dim=vertical_dim)

    # Pass full 3D weights
    interpolator = Regridder(weights=wfield, loglevel=loglevel, remap_area_min=remap_area_min)

    # subselect some levels
    xfield = xfield.isel(**{vertical_dim: levels})

    # Regrid level selection
    rfield = interpolator.regrid(xfield)

    if access == 'Dataset':
        rfield = rfield[smmvar].to_array()
        cdofield = cdofield[cdovar].to_array()

    # check if arrays are equal with numerical tolerance
    checker = np.allclose(cdofield, rfield, equal_nan=True)
    return checker
