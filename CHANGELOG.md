# Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)

## Unreleased

- Introduction of the `skipna` flag to renormalize weights to support time-varying and space-varying masks without declaring `mask_dim` (#70)
- `na_thres` to control when target cells with missing source points are set to NaN (#70)
- Optimization of tensordot operation to provide a speedup of 15-20% when `skipna=False` (#70)
- The sparse weights matrix is built once at init instead of at each compute, and the total weights and target coordinates are precomputed: ~35% faster graph construction for 3D data (#70)
- `skipna` is consistent with CDO for `remap_norm='destarea'`, and fully lazy (#70)
- Tests on a dask distributed cluster (#70)
- With `skipna`, 3D data are regridded as 2D with a single weights matrix (`mask_dim` is ignored): weights generated once instead of once per level (~40x faster init), ~18x fewer dask tasks, and 3D data can be regridded from 2D weights (#74)
- Benchmark tool `benchmarks/benchmark.py`: init, build, compute, peak memory and dask graph size of 2D/3D cases (with and without mask and skipna) on the serial scheduler and on dask clusters, optionally comparing two versions, and an on-demand CI job triggered by the `benchmark` label (#74)
- Without `skipna`, NaNs which were not in the source when the weights were generated propagate to the target cells again, instead of being treated as zeros; zero-weight links are dropped from the weights matrix (#76)
- `na_thres` has the same meaning with and without `skipna`, as in xESMF: the maximum fraction of missing weights of a target cell. Breaking for explicit values without `skipna` (previously the minimum valid fraction), unchanged for the default (#76)
- `na_thres='auto'` follows the method of the weights (`map_method`) when the `Regridder` is initialized from weights (#76)
- Support and require CDO >= 2.6.2 (2.6.1 has a known bug), checked at runtime with a clear error message (#66)
- macOS users must install CDO with Homebrew: new `environment-macos.yml` without CDO and macOS CI job (#66)
- The CDO executable is taken from the `CDO` environment variable when the `cdo` argument is not provided (#66)
- `python-cdo` is no longer a runtime dependency, only required for tests (#66)
- `smmregrid.checker` moved to `tests/checker.py`, as it is only used for testing (#66)

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
