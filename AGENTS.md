# smmregrid Agent Guidelines

## Project

- `smmregrid` regrids xarray `Dataset`/`DataArray` via sparse matrix multiplication with Dask. CDO generates the weights; smmregrid applies them.
- Module map (`smmregrid/`):
  - `cdogenerate.py` — `CdoGenerate`, `cdo_generate_weights`: build 2D/3D weights with CDO (`remap_norm` = `fracarea`|`destarea`).
  - `regrid.py` — `Regridder`, `regrid`: apply generated or precomputed weights (2D, and 3D via `mask_dim`).
  - `gridinspector.py` / `gridtype.py` / `griddetector.py` — detect grid types, horizontal/mask dims and grid kind.
  - `cdogrid.py` — `CdoGrid` target grid description; `weights.py` — weight masking/matrix helpers.
  - `cdobinary.py` — `resolve_cdo` (argument → `CDO` env var → `cdo` on PATH), `check_cdo_version` against `CDO_MIN_VERSION`.
  - `log.py` — `setup_logger`; `util.py` — `tolist`, `deprecated_argument`, misc helpers.
- `tests/` holds `*_test.py` modules with fixtures in `tests/data/`, and `tests/checker.py` with the CDO comparison helpers (`from checker import check_cdo_regrid`); `docs/sphinx/source/` holds the user docs (Read the Docs).
- `benchmarks/benchmark.py` measures init/build/compute, memory and graph size, optionally comparing two versions; the `benchmark` PR label triggers it in CI.
- Downstream: AQUA pins an exact smmregrid version. API or behaviour changes affect AQUA; flag them in the PR.

## Environment and checks

- Linux: `mamba env create -f environment.yml && mamba activate smmregrid` (installs the package editable with `[all]` extras, CDO from conda-forge).
- macOS: `brew install cdo`, then `mamba env create -f environment-macos.yml` (no CDO in the env: conda-forge has no recent macOS builds).
- CDO is required for weight generation and most tests (`cdo -V`). Minimum is `CDO_MIN_VERSION` = 2.6.2, checked at runtime: 2.6.1 has a known bug (#68, #66). `environment.yml` pins the version tested in CI; do not lower the minimum without re-testing.
- Tests: `python -m pytest` (focused: `python -m pytest tests/<name>_test.py`). `pyproject.toml` addopts run two `pytest-xdist` workers with `pytest-timeout`; these come from the `[tests]` extra. CI runs `python -m pytest --cov --cov-report=lcov` once for coverage.
- Lint: `flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics` must pass; keep lines ≤ 127 chars.
- Docs: `python -m sphinx -b html docs/sphinx/source docs/sphinx/build/html` when changing docs or public APIs.

## Compatibility

- Support Python >= 3.10. CI tests Python 3.10, 3.14 and 3.15 on Linux and macOS, on PRs to `main` and weekly. Breakage in versions between them can still surface late (see v0.1.5). Avoid syntax newer than 3.10.
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
