Introduction
============

A compact regridder using sparse matrix multiplication
------------------------------------------------------

This repository represents an extension and development of the regridding routines in ``climtas`` by Scott Wales, which already implements efficiently this idea and has no other significant dependencies. 
The regridder uses efficiently sparse matrix multiplication with dask plus some manipulation of the coordinates.

Please note that this tool is not thought as "another interpolation tool", but rather a method to apply pre-computed weights (with CDO) within the python environment. 
The speedup is estimated to be about ~1.5 to ~5 times compared to CDO itself, slightly lower if then files are written to the disk. 

2D and 3D data are supported on all the grids supported by CDO, both xarray.Dataset and xarray.DataArray can be used. 
Masks are treated in a simple way but are correctly transfered. 
Attributes are kept.
