# smmregrid Agent Guidelines

## Project

- `smmregrid` regrids xarray `Dataset`/`DataArray` via sparse matrix multiplication with Dask. CDO generates the weights; smmregrid applies them.
- Module map (`smmregrid/`):
  - `cdogenerate.py` — `CdoGenerate`, `cdo_generate_weights`: build 2D/3D weights with CDO (`remap_norm` = `fracarea`|`destarea`).
  - `regrid.py` — `Regridder`, `regrid`: apply generated or precomputed weights (2D, and 3D via `mask_dim`).
  - `gridinspector.py` / `gridtype.py` / `griddetector.py` — detect grid types, horizontal/mask dims and grid kind.
  - `cdogrid.py` — `CdoGrid` target grid description; `weights.py` — weight masking/matrix helpers; `checker.py` — CDO comparison helpers.
  - `log.py` — `setup_logger`; `util.py` — `tolist`, `deprecated_argument`, misc helpers.
- `tests/` holds `*_test.py` modules with fixtures in `tests/data/`; `docs/sphinx/source/` holds the user docs (Read the Docs).
- Downstream: AQUA pins an exact smmregrid version. API or behaviour changes affect AQUA; flag them in the PR.

## Environment and checks

- `mamba env create -f environment.yml && mamba activate smmregrid` (installs the package editable with `[all]` extras).
- CDO is required for weight generation and most tests (`cdo -V`). It is pinned to `>=2.5.0,<2.6.1` because of a bug in later versions (#68): do not relax the pin without re-testing.
- Tests: `python -m pytest` (focused: `python -m pytest tests/<name>_test.py`). CI runs `coverage run -m pytest`.
- Lint: `flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics` must pass; keep lines ≤ 127 chars.
- Docs: `python -m sphinx -b html docs/sphinx/source docs/sphinx/build/html` when changing docs or public APIs.

## Compatibility

- Support Python >= 3.8. PR CI tests only the newest Python; the full 3.8–3.14 matrix runs weekly against the PyPI release, so breakage surfaces late (see v0.1.5). Avoid post-3.8 syntax (`X | Y` annotations, `match`, builtin generics like `list[str]` in annotations).
- Public API is `__all__` in `smmregrid/__init__.py`; `__version__` lives there too.
- New parameters need defaults. Renamed parameters go through `util.deprecated_argument` (see `vertical_dim` → `mask_dim`).
- Every user-visible change gets a `CHANGELOG.md` entry under `## Unreleased` (Keep a Changelog), ending with `(#PR)`.

## Code conventions

- Logging: `self.loggy = setup_logger(level=loglevel, name='smmregrid.<Class>')`; pass `loglevel` through every constructor; %-style logger args; no `print()`.
- Keep results lazy for Dask-backed inputs and preserve xarray coordinates and attributes. Do not call `.compute()`/`.values` or use Dask reductions as Python conditions (`if array.all()`) in the regridding path.
- Respect the normalization recorded in CDO weights: `fracarea` normalizes by covered source area; `destarea` by full destination area and may sum to < 1. Do not renormalize `destarea` results when handling missing values.
- `remap_area_min` (0–1) masks target cells with insufficient source coverage in conservative remapping.

## Testing

- Validate weight/masking changes against CDO output where possible: partial coverage, missing values, 2D and 3D (`mask_dim`) grids.
- Reuse existing files in `tests/data/`; do not add new data files without discussion. Put regression tests next to the related `tests/*_test.py` cases.
