"""Tests for smmregrid to verify level selection"""

import os
import pytest
import numpy as np
import xarray as xr
from smmregrid import Regridder

INDIR = 'tests/data'
ifile = os.path.join(INDIR, 'onlytos-ipsl.nc')
tfile = os.path.join(INDIR, 'r360x180.nc')

# numerical noise in the CDO/smmregrid output
RANGE = 10


@pytest.mark.parametrize("fraction,nan", [
    ("0.0", 11607),
    (0.5, 10731),
    ("0.9", 9866),
])
def test_remap_area_min(fraction, nan):
    """test and assert with CDO multigrids"""

    regridder = Regridder(ifile, 'r180x90', remap_area_min=fraction, loglevel='debug')
    data = xr.open_dataset(ifile)
    regrid = regridder.regrid(data['tos'].isel(time=0))
    nanfound = (~np.isnan(regrid)).sum().values
    assert pytest.approx(nanfound, abs=RANGE) == nan


def test_remap_area_min_dask_chunks():
    """Verify that remap_area_min preserves Dask chunking along extra dimensions (e.g. time)
    and does not create a monolithic unchunked broadcast_to task in the Dask graph."""
    dfile = os.path.join(INDIR, '2t-era5.nc')
    data = xr.open_dataset(dfile, chunks={'time': 1})

    regridder = Regridder(dfile, 'r180x90', method='con', remap_area_min=0.5)
    regridded = regridder.regrid(data['2t'])

    # Time chunking must be preserved
    assert regridded.chunks[0] == (1,) * len(data['time'])

    # Ensure no monolithic broadcast_to task across time is added to the dask graph
    for k in regridded.data.dask:
        if isinstance(k, tuple):
            assert not k[0].startswith("broadcast_to"), f"Found unexpected broadcast_to task: {k}"

    # Ensure the result computes cleanly
    res = regridded.compute()
    assert res.shape == (len(data['time']), 90, 180)
    assert not np.isnan(res.values).all()

