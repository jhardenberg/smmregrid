Introduction
============

A compact regridder using sparse matrix multiplication
------------------------------------------------------

The regridder uses efficiently sparse matrix multiplication with dask plus some manipulation of the coordinates.
It provides a way to interpolate data - also starting from precomputed weights - inside python environment

Please note that this tool is not thought as "another interpolation tool", but rather a method to apply pre-computed weights (with CDO) within the python environment. 
The speedup is estimated to be about ~1.5 to ~5 times compared to CDO itself, slightly lower if then files are written to the disk. 

2D and 3D data are supported on all the grids supported by CDO (including gaussian reduced, healpix or unstructured grids with cell corners), both xarray.Dataset and xarray.DataArray can be used. 
Masks are treated in a simple way but correctly transfered. 
3D weights are computed specifically for oceanic grids in order to guarantee precise mask handling
Attributes are kept.

Cautionary notes:
- It does not work (yet) with dataset/files including multiple horizontal grids
- It does not work correctly if the Xarray.Dataset includes fields with time-varying missing points

Acknowledgement
---------------

This repository represents an extension and development of the regridding routines in ``climtas`` by Scott Wales, which already implements efficiently this idea and has no other significant dependencies. 

