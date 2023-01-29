# smmregrid
A compact regridder using sparse matrix multiplication

This repository represents a modification of the regridding routines in [climtas](https://github.com/ScottWales/climtas) by Scott Wales, which already implements efficiently this idea and has no other significant dependencies (it does not use iris or esmf for regridding).

I only had to change a few lines of code to make it compatible with unstructured grids. The regridder uses efficiently sparse matrix multiplication with dask + some manipulation of the coordinates (which would have to be revised/checked again)

Install with
```
pip install -e .
```
