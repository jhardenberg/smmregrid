import xarray as xr
import numpy as np
import xarray as xr
from smmregrid import cdo_generate_weights, Regridder

from cdo import Cdo
cdo = Cdo()

def check_cdo_regrid(finput, ftarget, method = 'con'):

    """Given a file to be interpolated finput over the ftarget grid,
    check if the output of the last variable is the same as produced 
    by CDO remap command"""

    # define files and open input file
    xfield = xr.open_mfdataset(finput)

    # var as the last available
    var = list(xfield.data_vars)[-1]

    # interpolation with smmregrid (CDO-based)
    wfield = cdo_generate_weights(finput, ftarget, method = method)
    interpolator = Regridder(weights=wfield)
    rfield = interpolator.regrid(xfield)

    # interpolation with pure CDO
    cdo_interpolator = getattr(cdo,  'remap' + method)
    cdofield = cdo_interpolator(ftarget, input = finput, returnXDataset = True)

    # check if arrays are equal with numerical tolerance
    checker = np.allclose(cdofield[var].data, rfield[var].data, equal_nan=True).compute()
    return checker