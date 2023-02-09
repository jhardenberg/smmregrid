# smmregrid
A compact regridder using sparse matrix multiplication

This repository represents a modification of the regridding routines in [climtas](https://github.com/ScottWales/climtas) by Scott Wales, which already implements efficiently this idea and has no other significant dependencies (it does not use iris).
The regridder uses efficiently sparse matrix multiplication with dask + some manipulation of the coordinates. 

Please note that this tool is not thought as "another interpolation tool", but rather a method to apply pre-computed weights (with CDO, which is currently tested, and with ESMF, which is not yet supported) within the python environment. 
The speedup is estimated to be about ~1.5 to ~5 times, slightly lower if then files are written to the disk. 2D and 3D data are supported on all the grids supported by CDO, both xarray.Dataset and xarray.DataArray can be used. Masks are treated in a simple way but are correctly transfered. Attributes are kept.  

It is safer to run it through conda/mamba. Install with: 

```
conda env create -f environment.yml
```

then activate the environment:

```
conda activate smmregrid
```
and install smmregrid in editable mode:

```
pip install -e .
```

Cautionary notes:
- It does not work correctly if the Xarray.Dataset includes fields with different land-sea masks (e.g. temperature and SST)
- It does not support ESMF weigths.

