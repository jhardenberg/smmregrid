Introduction
============

A compact regridder using sparse matrix multiplication
------------------------------------------------------

`smmregrid` is a tool meant to work within Python to perform efficiently regridding operation of dask-enable xarray.Dataset and xarray.DataArray.
It uses sparse matrix multiplication with basic manipulation of the coordinates and uses CDO as a backend to provide the computation of weights. 
It can be initialized from a pair of source/target files or xarray objects, but most importantly it can also start from precomputed weights.

Please note that `smmregrid` is not meant to be "another interpolation tool", but rather a method to apply pre-computed weights within the python in dask-enabled way. 
When weights are pre-computed, the speedup is estimated to be about ~5 to ~10 times compared to CDO itself. Despite the evident speed advantage, 
the main feature is everything is python-native and dask-enabled, so it can be included in complex pipelines and workflows.

2D and 3D data are supported on all the grids supported by CDO (including gaussian reduced, healpix or unstructured grids with available cell corners). 
Main objects are xarray so that full compatibility is guaranteed with the rest of the scientific python ecosystem.

Internally, it relies on the `GridType()` object which identify the properties of each xarray.DataArray,
detecting time, horizontal and masked coordinates. Indeed, a specific treatment for mask-changing dimensions is available. 

3D weights are computed specifically for each level and then stored together in specific files so that it guarantees precise mask handling.
This case is typical for vertical dimension in oceanic model, and it can be identified through the `mask_dim` keyword. 

.. warning ::

   It does not work with dataset/files including multiple horizontal grids

.. note ::

   The `skipna` option is available since v0.2.0 to allow for regridding of data with time-varying masks (or with vertical masks in 3D data without declaring `mask_dim`).
   It is based on renormalization of weights similarly to what done by ESMF. It is very efficient and might be extended to be default in the future. 
   It replicates CDO behaviour for conservative and nearest neighbour, but it is not guaranteed to be exactly the same for other methods.

Acknowledgement
---------------

This repository represents an extension and development of the regridding routines in ``climtas`` by Scott Wales, 
which already implements efficiently this idea and has no other significant dependencies. 

