Introduction
============

A compact regridder using sparse matrix multiplication
------------------------------------------------------

`smmregrid` is a tool meant to work into python environment to perform efficiently regridding operation of dask-enable xarray.Dataset and xarray.DataArray.
It uses sparse matrix multiplication with basic manipulation of the coordinates and uses CDO as a backend to provide the computation of weights. 
It can work from source and target grids, but most importantly it can also start from precomputed weights.

Please note that `smmregrid` is not meant to be "another interpolation tool", but rather a method to apply pre-computed weights within the python environment. 
The speedup is estimated to be about ~1.5 to ~5 times compared to CDO itself, slightly lower if then files are written to the disk. 

2D and 3D data are supported on all the grids supported by CDO (including gaussian reduced, healpix or unstructured grids with cell corners), both xarray.Dataset and xarray.DataArray can be used. 
It relies on the `GridType()` class which identify the properties of each xarray.DataArray, detecting time, horizontal and vertical coordinates.

Masks are treated in a simple way but correctly transfered from source to target grid. 

A specific treatment for mask changing along the vertical dimension can be applied through the `mask_dim` keyword. 
Indeed, 3D weights are computed specifically for each level and then stored together in specific files so that it guarantees precise mask handling.


.. warning ::

   1. It does not work (yet) with dataset/files including multiple horizontal grids
   2. It does not work correctly if the Xarray.Dataset includes fields with time-varying missing points

Acknowledgement
---------------

This repository represents an extension and development of the regridding routines in ``climtas`` by Scott Wales, 
which already implements efficiently this idea and has no other significant dependencies. 

