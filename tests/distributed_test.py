"""Tests for smmregrid running on a dask distributed cluster.

The regridded output computed on a LocalCluster with several worker processes
must be the same as the one computed serially, both for 2D and 3D grids.
This checks that the dask graph (including the sparse weights) can be serialized
and shipped to the workers, and that chunking does not affect the results.
The Regridder is initialized both from grids and from a weights file on disk.
"""

import os
import pytest
import dask
import xarray as xr
from smmregrid import Regridder, CdoGenerate

distributed = pytest.importorskip("distributed")

INDIR = 'tests/data'
TARGET = 'r180x90'
NWORKERS = 4
NTILE = 4  # repeat the data along time, to have more chunks than workers


@pytest.fixture(scope="module")
def client():
    """A small local cluster with separate worker processes, one thread each"""
    with distributed.LocalCluster(n_workers=NWORKERS, threads_per_worker=1, processes=True,
                                  dashboard_address=None) as cluster:
        with distributed.Client(cluster) as dask_client:
            yield dask_client


def _load(fname, var, chunks):
    """Open a test file, tile it along time and chunk it, to spread the tasks across workers"""
    data = xr.open_dataset(os.path.join(INDIR, fname), chunks={})[[var]]
    return xr.concat([data] * NTILE, dim='time').chunk(chunks)


def _regridder(fname, init, kwargs, tmpdir, mask_dim=None):
    """Build the Regridder from grids, or from a weights file written to disk and read back"""
    source_grid = os.path.join(INDIR, fname)
    if init == 'grids':
        return Regridder(source_grid=source_grid, target_grid=TARGET, loglevel='WARNING', **kwargs)
    wfile = str(tmpdir / f"weights_{fname}")
    CdoGenerate(source_grid, TARGET, loglevel='WARNING', **kwargs).weights(
        method='con', mask_dim=mask_dim).to_netcdf(wfile)
    return Regridder(weights=wfile, loglevel='WARNING', **kwargs)


# file, variable, chunks, init ('grids' or 'weights' from disk), regridder options, mask_dim
CASES = {
    '2d lonlat': ('2t-era5.nc', '2t', {'time': 1}, 'grids', {}, None),
    '2d curvilinear skipna': ('onlytos-ipsl.nc', 'tos', {'time': 1}, 'grids', {'skipna': True}, None),
    '3d nemo': ('so3d-nemo.nc', 'so', {'time': 1, 'lev': 6}, 'grids', {}, None),
    '2d fesom from weights': ('tos-fesom.nc', 'tos', {'time': 1}, 'weights', {}, None),
    '3d nemo from weights': ('so3d-nemo.nc', 'so', {'time': 1, 'lev': 6}, 'weights', {}, 'lev'),
    '3d nemo skipna': ('so3d-nemo.nc', 'so', {'time': 1, 'lev': 6}, 'grids', {'skipna': True}, None),
    '2d ipsl from weights skipna': ('onlytos-ipsl.nc', 'tos', {'time': 1}, 'weights', {'skipna': True}, None),
}


@pytest.mark.parametrize("case", list(CASES))
def test_distributed_vs_serial(client, case, tmp_path):
    """The output computed on the cluster must match the serial one"""

    fname, var, chunks, init, kwargs, mask_dim = CASES[case]
    source = _load(fname, var, chunks)
    regridder = _regridder(fname, init, kwargs, tmp_path, mask_dim=mask_dim)

    regridded = regridder.regrid(source)
    assert regridded[var].chunks is not None, "regridded output is expected to be lazy"
    nchunks = regridded[var].data.npartitions
    assert nchunks >= NWORKERS, f"only {nchunks} chunks, the cluster would be underused"

    # compute on the cluster, recording which workers executed the tasks
    with distributed.get_task_stream(client) as task_stream:
        distributed_out = client.compute(regridded).result()
    with dask.config.set(scheduler='synchronous'):
        serial_out = regridded.compute()

    workers = {task['worker'] for task in task_stream.data}
    assert len(workers) > 1, f"tasks were executed by {len(workers)} worker only"

    assert distributed_out[var].notnull().any()
    xr.testing.assert_allclose(distributed_out, serial_out)
