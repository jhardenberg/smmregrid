# Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)

## [Unreleased]

- `na_thres` is the only threshold for masking target cells, with one rule: a target cell is NaN when the fraction of it covered by valid source points is below `1 - na_thres`, and always when it has no valid source point (also with `na_thres=1.0` without `skipna`, where it used to be 0). For conservative remapping this is the `remap_area_min` of CDO (`na_thres = 1 - remap_area_min`, the weights are overlap areas): the covered fraction is the `dst_grid_frac` of the weights without `skipna`, and with `skipna` the share of the weights on valid points times the `dst_grid_frac` of the weights, computed at each call: the same number as the `dst_grid_frac` of CDO with that mask, for both normalizations (with `destarea` the absolute area of the target cell). For `bil`/`bic`/`nn` without `skipna` CDO already decides which target cells are mapped: `na_thres` has no effect and a warning is logged if a value is given. The target mask stored in the weights (`dst_grid_imask`) now only marks the cells with at least one valid source point, and `mask_weights`/`mask_tensordot` take no `na_thres`. `na_thres='auto'` (default) is the smmregrid default of each method (0.5 for `con`, as the previous default `remap_area_min=0.5`, 1e-6 for `bil`/`bic`, 1.0 otherwise) and `na_thres='cdo'` replicates CDO (1.0 for `con`). `remap_area_min` is deprecated (a `DeprecationWarning`, translated to `na_thres` for `con`, ignored otherwise) (#93)
- Warn when `skipna=True` is used with a source grid that has a mask dimension (e.g. a vertical one) with more than one level: it is ignored, as the weights are generated from the first level only and applied to all levels (#86)
- The source grid used to generate the weights is sliced to the first element along all the non-horizontal dimensions (time and any other), not only time; with `skipna` the mask dimension is sliced too, since the weights are 2D. The grid to be filled and written is smaller and faster (#86)
- Target areas from `CdoGenerate.areas()` have lat/lon rounded to the same number of decimals (`COORD_DECIMALS`) as the regridded data, so that their coordinates are aligned exactly (#89)
- Per-level regridding messages (`Interpolating from CDO ...`, `Regridding from ...`, `Assuming gridtype ...`) moved from `info` to `debug`, and 3D regridding logs a single `info` line per variable instead (#88)
- Horizontal dimensions are grouped by grid in `HORIZONTAL_FAMILIES`, in order of priority: a variable is on the first family whose dimensions it has all, and the other dimensions are left in `other_dims` (e.g. `x` in `(lat, lon, x)`). A partial match is no longer a grid: e.g. a zonal mean `(time, lat)` has no horizontal dimensions (CDO cannot regrid it). A lone `i` or `x` is an unstructured index (ICON in CMIP6, HEALPix), and `ncol` (CESM spectral element) is added. Dimensions declared with `extra_dims`/`horizontal_dims` come first and are used as found. Horizontal dimensions now follow the order of the data (#81)
- `rlon`/`rlat` (rotated pole, e.g. CORDEX) added to the horizontal dimensions. The CMIP6 bounds `vertices_*`/`*_vertices` are recognized as bounds, and no longer taken as a separate grid. `GridInspector` works on a DataArray without a name (#81)
- `nlat`/`nlon` (CESM/POP), `ni`/`nj` (CICE) and `xh`/`yh`/`xq`/`yq` (GFDL/MOM6) added to the default horizontal dimensions, so CMIP6 CESM2 and GFDL-CM4 ocean and sea-ice grids are detected; `CdoGenerate` and `Regridder` raise a clear `ValueError` naming the dimensions found when no horizontal dimension is detected, instead of an `IndexError` or `TypeError` (#81)
- Fix regridding of data with dimensions after the horizontal ones (e.g. `(lat, lon, time)`), which were mixed with the horizontal ones in the output (#81)
- Preserve Dask chunking along extra dimensions (e.g. time) in `remap_area_min` masking by broadcasting lazily instead of using `dask.array.broadcast_to`, preventing large memory allocations (#80)
- `remap_area_min` with `skipna` and conservative remapping now masks the target cells by the area covered by the valid source points, computed at each call, as CDO and as the same regridding without `skipna` do. Previously the geometric coverage of the grid was used and the coastal and masked cells were never masked, e.g. 14778 more valid cells than CDO on a NEMO 3D field with the default `remap_area_min=0.5`. Behaviour change for `skipna=True` users with the default `remap_area_min`. New documentation page on `na_thres` and `remap_area_min` (equivalent for conservative remapping with `fracarea`), with the two masks as functions in `weights.py` (#85)
- Python >= 3.10 is required: support for Python 3.8 and 3.9 is dropped (#82)


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
