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
The speedup against CDO is estimated to be about ~1.5 to ~5 times, slightly lower if then files are written to the disk. 2D and 3D data are supported on all the grids supported by CDO, and special treatment can be assigned to vertical coordinates with changing mask (e.g. ocean 3D datasets). It works smoothly on both xarray.Dataset and xarray.DataArray. Attributes are kept and target grids can be both file on disk or CDO-compliant grids (e.g. r180x90, F128, etc).  

The tool works for python versions >=3.10 and requires CDO >= 2.6.2. It is safer to run it through conda/mamba.

### Linux

The simplest way is to install smmregrid from conda-forge, which brings in all the binary dependencies (including CDO and eccodes):

```
mamba create -n smmregrid -c conda-forge smmregrid
mamba activate smmregrid
```

Or, installing from PyPI:

```
mamba create -n smmregrid "python>=3.10" cdo eccodes numba
mamba activate smmregrid
pip install smmregrid
```

Alternatively, you can clone the repo and use the environment file available

```
git clone https://github.com/jhardenberg/smmregrid.git
cd smmregrid
mamba env create -f environment.yml
mamba activate smmregrid
```

### macOS

Recent CDO versions are not available from conda-forge for macOS, so CDO must be installed with [Homebrew](https://brew.sh), while the python environment is still managed with conda/mamba:

```
brew install cdo
mamba create -n smmregrid -c conda-forge smmregrid
```

The conda-forge package of smmregrid does not include CDO on macOS. Similarly, if you install from PyPI:

```
brew install cdo
mamba create -n smmregrid "python>=3.10" eccodes numba
mamba activate smmregrid
pip install smmregrid
```

or, from the cloned repo, use the dedicated environment file (which does not include CDO):

```
brew install cdo
mamba env create -f environment-macos.yml
mamba activate smmregrid
```

### Which CDO is used

smmregrid looks for the CDO executable in this order:

1. the `cdo` argument of `Regridder` / `CdoGenerate`, if provided;
2. the `CDO` environment variable (the same used by python-cdo);
3. `cdo` in the `PATH`.

On macOS, make sure that no `cdo` from conda-forge is installed in your environment (e.g. pulled in by another package), since it would shadow the Homebrew one in the `PATH`. If you cannot avoid it, point smmregrid to the Homebrew CDO explicitly:

```
export CDO=$(brew --prefix)/bin/cdo
```

The CDO version is checked at runtime, and an error is raised if it is older than the minimum supported. You can check which version is used with `python -c "import smmregrid; print(smmregrid.cdo_version())"`.

As a disclaimer, this repository represents a modification of the regridding routines developed in [climtas](https://github.com/ScottWales/climtas) by Scott Wales, which already implements efficiently this idea and has no other significant dependencies.

Cautionary notes:
- It does not work correctly if the Xarray.Dataset includes fields with time-varying missing points
- It works only with interpolation methods/grids supported by CDO
- It does not support ESMF weigths.

