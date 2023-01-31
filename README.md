# smmregrid
A compact regridder using sparse matrix multiplication

This repository represents a modification of the regridding routines in [climtas](https://github.com/ScottWales/climtas) by Scott Wales, which already implements efficiently this idea and has no other significant dependencies (it does not use iris).

I only had to change a few lines of code to make it compatible with unstructured grids. The regridder uses efficiently sparse matrix multiplication with dask + some manipulation of the coordinates (which would have to be revised/checked again)

Install with
```
pip install -e .
```

Please note that this tool is not thought as "another interpolation tool", but rather a method to apply pre-computed weights (with CDO, which is currently tested, and with ESMF, which is not yet supported completely) within the python environment. 
The speedup is estimated to be about 5 to 10 times. Only 2D grid are currently supported. Masks are treated in a simple way but are kept (still in investigation)
