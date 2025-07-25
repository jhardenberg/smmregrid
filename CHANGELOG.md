# Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)

## Unreleased

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
