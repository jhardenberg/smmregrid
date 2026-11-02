"""Shared pytest fixtures."""

import gc
import pytest
import xarray


@pytest.fixture(autouse=True)
def close_datasets(monkeypatch):
    """Close every dataset opened during a test when the test ends.

    Datasets left open are closed later by the garbage collector, possibly in a dask
    thread while another one reads a netCDF file: netCDF4/HDF5 are not thread-safe and the
    process can crash (SIGBUS) or freeze holding the GIL, so that pytest-timeout cannot fire.
    """
    opened = []

    def tracked(opener):
        def wrapper(*args, **kwargs):
            data = opener(*args, **kwargs)
            opened.append(data)
            return data
        return wrapper

    for name in ('open_dataset', 'open_mfdataset'):
        monkeypatch.setattr(xarray, name, tracked(getattr(xarray, name)))

    yield

    for data in opened:
        data.close()
    opened.clear()
    gc.collect()
