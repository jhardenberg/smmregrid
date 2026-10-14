# Repository guidance

## Project

- `smmregrid` uses CDO to generate remapping weights, then applies them with sparse matrix multiplication and Dask to xarray `Dataset` and `DataArray` inputs. `CdoGenerate` builds weights; `Regridder` applies generated or precomputed weights.
- `smmregrid/` contains the implementation, `tests/` contains `*_test.py` modules and data fixtures, and `docs/sphinx/source/` contains the user documentation. Keep the README, Sphinx docs, and changelog aligned with public behavior.
- Preserve support for Python 3.8 and later, including the package's documented 2D and 3D `mask_dim` workflows.

## Environment and checks

- Create the source environment with `micromamba env create -f environment.yml`, then `micromamba activate smmregrid`. The environment installs the package in editable mode with its extras. The existing local `aqua-dev` environment is also suitable for development checks.
- CDO is required to generate weights and run CDO-dependent tests. The environment pins CDO to `>=2.5.0,<2.6.1` on Linux; on macOS, install the executable separately. Check it with `cdo -V`.
- Run the full suite with `python -m pytest`; for focused work, run `python -m pytest tests/<name>_test.py`. CI runs `coverage run -m pytest` and checks syntax and undefined names with `flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics`.
- Build the Sphinx documentation with `python -m sphinx -b html docs/sphinx/source docs/sphinx/build/html` when changing docs or public APIs.

## Regridding conventions

- Keep regridded results lazy for Dask-backed inputs, with xarray coordinates and attributes preserved. A Dask reduction is lazy until used as a Python condition; avoid whole-input checks such as `if array.all()` in the regridding path.
- Respect the normalization recorded in CDO weights. Conservative `fracarea` weights normalize by covered source area; `destarea` weights normalize by the full destination area and may sum to less than one. Do not accidentally renormalize `destarea` results when handling missing values.
- Test changes to weights or masking against CDO where possible, including partial coverage, missing values, and 2D and 3D grids. Keep regression tests near the related `tests/*_test.py` cases and use fixtures in `tests/data/`.
