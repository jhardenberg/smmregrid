[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/jhardenberg/smmregrid/graphs/commit-activity)
[![PyTest](https://github.com/jhardenberg/smmregrid/actions/workflows/mambatest.yml/badge.svg)](https://github.com/jhardenberg/smmregrid/actions/workflows/mambatest.yml)
[![Coverage Status](https://coveralls.io/repos/github/jhardenberg/smmregrid/badge.svg?branch=main)](https://coveralls.io/github/jhardenberg/smmregrid?branch=main)
[![PyPI](https://badge.fury.io/py/smmregrid.svg)](https://badge.fury.io/py/smmregrid)
[![Conda](https://img.shields.io/conda/vn/conda-forge/smmregrid.svg)](https://anaconda.org/conda-forge/smmregrid)
[![Documentation](https://readthedocs.org/projects/smmregrid/badge/?version=latest)](https://smmregrid.readthedocs.io/en/latest/?badge=latest)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15553576.svg)](https://doi.org/10.5281/zenodo.15553576)



# smmregrid
A compact python regridder using sparse matrix multiplication

The regridder uses CDO as a backend for weights computation, and then uses efficiently sparse matrix multiplication with dask to provide xarray lazy output. It supports all grids supported by CDO. 

Please note that this tool is not thought as "another interpolation tool", but rather a method to apply pre-computed weights  within the python environment. 
The speedup against CDO is estimated to be about ~1.5 to ~5 times, slightly lower if then files are written to the disk. 2D and 3D data are supported on all the grids supported by CDO, and special treatment can be assigned to vertical coordinates with changing mask (e.g. ocean 3D datasets). It works smoothly on both xarray.Dataset and xarray.DataArray. Attributes are kept and target grids can be both file on disk or CDO-compliant grids (e.g. r180x90, n128, etc).  

The tool works for python versions >=3.8. It is safer to run it through conda/mamba. Install with:

```
mamba create -n smmregrid "python>=3.8" cdo eccodes numba
mamba activate smmregrid
pip install smmregrid
``` 

Alternatively, you can clone the repo and use the environment file available

```
git clone https://github.com/jvonhard/smmregrid.git
cd smmregrid
mamba env create -f environment.yml
mamba activate smmregrid
pip install -e .
```

As a disclaimer, this repository represents a modification of the regridding routines developed in [climtas](https://github.com/ScottWales/climtas) by Scott Wales, which already implements efficiently this idea and has no other significant dependencies.

Cautionary notes:
- It does not work correctly if the Xarray.Dataset includes fields with time-varying missing points
- It works only with interpolation methods/grids supported by CDO
- It does not support ESMF weigths.

