"""Module for cdo-smmregrid comparison"""

import numpy as np
import xarray as xr
from cdo import Cdo
from smmregrid import Regridder, cdo_generate_weights


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
                     init_method='grids', vert_coord=None):
    """Given a file to be interpolated finput over the ftarget grid,
    check if the output of the last variable is the same as produced
    by CDO remap command. This function is used for tests."""

    # define files and open input file
    xfield = xr.open_mfdataset(finput)

    # var as the last available
    # myvar = list(xfield.data_vars)[-1]

    # interpolation with pure CDO
    cdo_interpolator = getattr(cdo, 'remap' + remap_method)
    cdofield = cdo_interpolator(ftarget, input=finput, returnXDataset=True)
    # print(cdofield)

    # var as the one which have time and not have bnds (could work)
    smmvar = find_var(xfield)
    cdovar = find_var(cdofield)

    if len(smmvar) == 1 and access == 'DataArray':
        xfield = xfield[smmvar[0]]
    if len(cdovar) == 1 and access == 'DataArray':
        cdofield = cdofield[cdovar[0]]

    # interpolation with smmregrid (CDO-based)
    # method with creation of weights
    # wfield = cdo_generate_weights(finput, ftarget, method=method)
    # interpolator = Regridder(weights=wfield)

    # method with automatic creation of weights
    if init_method == 'grids':
        interpolator = Regridder(source_grid=finput, target_grid=ftarget,
                                 method=remap_method, vert_coord=vert_coord)
    if init_method == 'weights':
        wfield = cdo_generate_weights(finput, ftarget,
                                      method=remap_method, vert_coord=vert_coord)
        interpolator = Regridder(weights=wfield, vert_coord=vert_coord)
    rfield = interpolator.regrid(xfield)

    if access == 'Dataset':
        rfield = rfield[smmvar].to_array()
        cdofield = cdofield[cdovar].to_array()

    # check if arrays are equal with numerical tolerance
    checker = np.allclose(cdofield, rfield, equal_nan=True)
    return checker


def check_cdo_regrid_levels(finput, ftarget, vert_coord, levels, remap_method='con', access='Dataset'):
    """Given a file to be interpolated finput over the ftarget grid,
    check if the output of the last variable is the same as produced
    by CDO remap command. This function is used for tests.
    This is a variant to check the level memory feature (idx_3d)."""

    # define files and open input file
    xfield = xr.open_mfdataset(finput)

    # interpolation with pure CDO
    cdo_interpolator = getattr(cdo, 'remap' + remap_method)
    cdofield = cdo_interpolator(ftarget, input=finput, returnXDataset=True)

    # Keep only some levels
    cdofield = cdofield.isel(**{vert_coord: levels})

    # var as the one which have time and not have bnds (could work)
    smmvar = find_var(xfield)
    cdovar = find_var(cdofield)

    if len(smmvar) == 1 and access == 'DataArray':
        xfield = xfield[smmvar[0]]
    if len(cdovar) == 1 and access == 'DataArray':
        cdofield = cdofield[cdovar[0]]

    # compute weights
    wfield = cdo_generate_weights(finput, ftarget,
                                  method=remap_method, vert_coord=vert_coord)
    
    # Pass full 3D weights
    interpolator = Regridder(weights=wfield, vert_coord=vert_coord)

    # Add a helper idx_3d coordinate
    idx = list(range(0, len(xfield.coords[vert_coord])))
    xfield = xfield.assign_coords(idx_3d=(vert_coord, idx))

    # subselect some levels
    xfield = xfield.isel(**{vert_coord: levels})

    # Regrid level selection
    rfield = interpolator.regrid(xfield)

    if access == 'Dataset':
        rfield = rfield[smmvar].to_array()
        cdofield = cdofield[cdovar].to_array()

    # check if arrays are equal with numerical tolerance
    checker = np.allclose(cdofield, rfield, equal_nan=True)
    return checker