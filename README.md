[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/jhardenberg/smmregrid/graphs/commit-activity)
[![PyTest](https://github.com/jhardenberg/smmregrid/actions/workflows/mambatest.yml/badge.svg)](https://github.com/jhardenberg/smmregrid/actions/workflows/mambatest.yml)
[![Coverage Status](https://coveralls.io/repos/github/jhardenberg/smmregrid/badge.svg?branch=main)](https://coveralls.io/github/jhardenberg/smmregrid?branch=main)
[![PyPI version](https://badge.fury.io/py/smmregrid.svg)](https://badge.fury.io/py/smmregrid)

# smmregrid
A compact regridder using sparse matrix multiplication

This repository represents a modification of the regridding routines in [climtas](https://github.com/ScottWales/climtas) by Scott Wales, which already implements efficiently this idea and has no other significant dependencies.
The regridder uses efficiently sparse matrix multiplication with dask + some manipulation of the coordinates. 

Please note that this tool is not thought as "another interpolation tool", but rather a method to apply pre-computed weights (with CDO, which is currently tested) within the python environment. 
The speedup is estimated to be about ~1.5 to ~5 times, slightly lower if then files are written to the disk. 2D and 3D data are supported on all the grids supported by CDO, both xarray.Dataset and xarray.DataArray can be used. Masks are treated in a simple way but are correctly transfered. Attributes are kept.  

The tool works for python versions >=3.8. It is safer to run it through conda/mamba. Install with: 

```
mamba env create -f environment.yml
```

then activate the environment:

```
mamba activate smmregrid
```

and install smmregrid in editable mode:

```
pip install -e .
```

Alternatively - if you have in your environment/machine the required dependencies, mostly CDO - you can install smmregrid directly via pypi with:

```
pip install smmregrid
```

Cautionary notes:
- It does not work correctly if the Xarray.Dataset includes fields with time-varying missing points
- It works only with interpolation methods/grids supported by CDO
- It does not support ESMF weigths.

