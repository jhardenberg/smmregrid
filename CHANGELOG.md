# Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)

## [Unreleased]

- Preserve Dask chunking along extra dimensions (e.g. time) in `remap_area_min` masking by broadcasting lazily instead of using `dask.array.broadcast_to`, preventing large memory allocations (#79, #80)

## [v0.2.0]

- `skipna` option: missing source points are skipped and the weights renormalized at each call, supporting masks varying in time or along any dimension without `mask_dim`. Consistent with CDO for conservative and nearest neighbour, and with `remap_norm='destarea'`. 3D data use a single 2D weights matrix (~40x faster init, ~18x fewer dask tasks) (#70, #74)
- `na_thres`: maximum fraction of missing weights of a target cell, with the same meaning with and without `skipna` (as in xESMF); `'auto'` follows the method, also of precomputed weights. Breaking change: for explicit values without `skipna`, previously it was the minimum valid fraction (#70, #76)
- Faster regridding following several optimization: weights matrix and derived quantities built once at init (~35% faster graph construction for 3D data), 15-20% faster tensordot without `skipna` (#70)
- CDO >= 2.6.2 is required and checked at runtime; the executable is taken from the `cdo` argument, the `CDO` environment variable or the PATH. On macOS, CDO has to be installed with Homebrew (`environment-macos.yml`) (#66)
- `python-cdo` is only a test dependency, and `smmregrid.checker` moved to `tests/checker.py` (#66)
- Tests on a dask distributed cluster and on two `pytest-xdist` workers; benchmark tool `benchmarks/benchmark.py` with an on-demand CI job (`benchmark` label) (#70, #74)
- Documentation: conda-forge installation, corrected `Regridder` parameters and examples (#77)

## [v0.1.6]

- Pin for `cdo<2.6.1` to avoid bug in the more recent CDO versions (#68)
- Introduce a `GridDetector` class where grid kind guessing is performed external from `GridInspector` (#68)
- Add `valid_time` and `forecast_time` as time dimension for `Gridtype` guessing (#68)
- Provide a new method for `GridInspector`to extract information on the minimum required data variables from a dataset with different gridtype (#68)

## [v0.1.5]

- Fix bug which was breaking smmregrid for python 3.8, 3.9, 3.10

## [v0.1.4]

- Support for python 3.14 (#57)
- `mask_dim` keep same attributes as original data in 3D weights (#58)
- Vertical coordinate in weights now correspond to the data coordinates, rather than a list (#55)
- Replace `vertical_dim` to more correct `mask_dim` as changing direction for mask (#56)
- Introduction `check_nan` option to detect if a vertical dimension has NaN values (#50)

## [v0.1.3]

- Improve grid kind and bounds detection via `GridInspector` (#48, #49)

## [v0.1.2]

- CDO names can be used to generate remapping weights as source (#46)
- Fix `keep_attrs` behaviour in `GridInspector()` (#47)

## [v0.1.1]

- Support for python==3.13 (#44)
- `GridInspector()` is now able to detect also the `kind` of a grid, i.e. if it is regular, gaussian, unstructured etc. (#44)
- Possibility to replace default dimension in `GridType()` with `override` flag (#44)
- `GridType()` can now be printed (#44)
- `GridInspector` method `get_gridtype()` replaces `get_grid_info()` for cleaner naming structure (#44)
- `remap_area_min` for conservative remapping is now set to 0.5 to avoid coastal erosion (#43)
- Fully dask-array based oriented `regrid2d` method and more efficient `check_mask` (#43)

## [v0.1.0]

- Refactoring of the grid handling to possibly support more complex data structures (#33)
- Introduction of `GridType()` class to handle possible multiple grids in the future (#33)
- Introduction of `GridInspector()` class to investigate properties of a Dataset/DataArray (#33)
- Minimal documentation via Sphinx available through ReadTheDocs (#33)
- Preliminar support to target unstructured grids (#33)
- `CdoGenerate()` class to support weights generator through `weights()` method (#33)
- `CdoGenerate().areas()` method to generate areas based on CDO (#33)

## [v0.0.7]

- Allow for `cdo_options` and move from `extra` to `cdo_extra` (#31)
- Refactor the logging (#31)
- Remove remnants from ESMF support (#30)

## [v0.0.6]

- Preserve NaN after interpolation for unmasked fields

## [v0.0.5]

- Allow also single 3D level
- Prefix used to identify helper index coordinate

## [v0.0.4]

- Additional helper coordinate to select correct levels from full 3D weights

## [v0.0.3]

- Update of the error trap of CDO

## [v0.0.2]

- Linting, logging, autofind of vertical coordinate

## [v0.0.1]

- Add cdo path
- Support for 3D regridding
- Healpix support
- First implementation
