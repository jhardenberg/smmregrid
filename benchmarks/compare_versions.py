#!/usr/bin/env python
"""
Compare smmregrid performance across git refs (branches, commits) and the working tree.

Each version is exported to a temporary directory and run in its own Python process,
so that versions never mix. For each test case it measures:
    - init:    Regridder initialization (dominated by CDO weights generation)
    - build:   the regrid() call, i.e. the dask graph construction
    - compute: the .compute() of the regridded output

and finally checks that the outputs of all versions match the first one.

Usage (from the repository root):
    python benchmarks/compare_versions.py                         # main vs working tree
    python benchmarks/compare_versions.py --refs main HEAD WORKTREE --reps 3
    python benchmarks/compare_versions.py --uncommitted               # last commit vs uncommitted changes
    python benchmarks/compare_versions.py --cases "3D nemo" "fesom x60 time"
    python benchmarks/compare_versions.py --list-cases

WORKTREE is a special ref standing for the current (possibly uncommitted) code.
Requires CDO to be available, as for the test suite.
"""

import argparse
import json
import logging
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import warnings

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(REPO, 'tests', 'data')
TARGET = os.path.join(DATA, 'r360x180.nc')

# name, file, variable (None for all), time tiling factor (to emulate longer datasets)
CASES = [
    ('lonlat era5', '2t-era5.nc', '2t', 1),
    ('gaussian ecearth', 'tas-ecearth.nc', None, 1),
    ('healpix', 'tas-healpix2.nc', None, 1),
    ('unstructured fesom', 'tos-fesom.nc', 'tos', 1),
    ('curvilinear ipsl', 'onlytos-ipsl.nc', None, 1),
    ('plev gaussian', 'ua-ecearth.nc', None, 1),
    ('3D nemo', 'so3d-nemo.nc', 'so', 1),
    ('3D fesom', 'temp3d-fesom.nc', None, 1),
    ('multigrid cesm', 'mix-cesm.nc', None, 1),
    ('fesom x60 time', 'tos-fesom.nc', 'tos', 60),
    ('3D nemo x60 time', 'so3d-nemo.nc', 'so', 60),
]
# extra case: same as 'fesom x60 time', but initializing from a weights file on disk
WEIGHTS_CASE = 'fesom x60 from weights'
ALL_CASES = [c[0] for c in CASES] + [WEIGHTS_CASE]

log = logging.getLogger('benchmark')


def setup_logging(loglevel, name):
    """Log to stderr, so that stdout is kept for results.
    Only the benchmark logger is configured, to keep third-party libraries quiet"""
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(f'%(asctime)s [{name}] %(levelname)s: %(message)s',
                                           datefmt='%H:%M:%S'))
    log.addHandler(handler)
    log.setLevel(loglevel.upper())
    log.propagate = False


# --------------------------------------------------------------------------------------
# Worker: runs inside the process of a single version
# --------------------------------------------------------------------------------------

def _load(xr, fname, var=None, tile=1):
    """Open a test file lazily; when tile > 1 repeat it along time, one chunk per step"""
    if tile > 1:
        ds = xr.open_dataset(os.path.join(DATA, fname))
        if var:
            ds = ds[[var]]
        return xr.concat([ds] * tile, dim='time').chunk({'time': 1})
    ds = xr.open_dataset(os.path.join(DATA, fname), chunks={})
    return ds[[var]] if var else ds


def _bench(tag, case, regridder, ds, tinit, repeat, outdir):
    """Time graph build and compute of a regridder, and store the output"""
    log.debug('%s: init done in %.2fs, warmup', case, tinit)
    regridder.regrid(ds).compute()  # warmup
    tbuild = tcompute = 0.
    for _ in range(repeat):
        t0 = time.perf_counter()
        out = regridder.regrid(ds)
        t1 = time.perf_counter()
        out = out.compute()
        t2 = time.perf_counter()
        tbuild += t1 - t0
        tcompute += t2 - t1
        log.debug('%s: build %.1f ms, compute %.1f ms', case, 1000 * (t1 - t0), 1000 * (t2 - t1))
    print(json.dumps({'tag': tag, 'case': case, 'init': tinit,
                      'build': 1000 * tbuild / repeat,
                      'compute': 1000 * tcompute / repeat}), flush=True)
    log.info('%-24s init %6.2fs  build %8.1f ms  compute %8.1f ms',
             case, tinit, 1000 * tbuild / repeat, 1000 * tcompute / repeat)
    out.to_netcdf(os.path.join(outdir, f"{tag}__{case.replace(' ', '_')}.nc"))


def worker(tag, outdir, repeat, cases):
    """Run all the selected cases for the smmregrid version found in PYTHONPATH"""
    import xarray as xr
    from smmregrid import Regridder, CdoGenerate
    warnings.filterwarnings('ignore')

    for case, fname, var, tile in CASES:
        if case not in cases:
            continue
        log.debug('%s: loading %s (tile x%s)', case, fname, tile)
        try:
            ds = _load(xr, fname, var, tile)
            t0 = time.perf_counter()
            regridder = Regridder(source_grid=os.path.join(DATA, fname),
                                  target_grid=TARGET, loglevel='ERROR')
            _bench(tag, case, regridder, ds, time.perf_counter() - t0, repeat, outdir)
        except Exception as err:  # keep going if a version does not support a case
            log.error('%s failed: %r', case, err)
            print(json.dumps({'tag': tag, 'case': case, 'error': repr(err)[:300]}), flush=True)

    if WEIGHTS_CASE in cases:
        try:
            wfile = os.path.join(outdir, f'{tag}__weights.nc')
            CdoGenerate(os.path.join(DATA, 'tos-fesom.nc'), TARGET,
                        loglevel='ERROR').weights(method='con').to_netcdf(wfile)
            ds = _load(xr, 'tos-fesom.nc', 'tos', 60)
            t0 = time.perf_counter()
            regridder = Regridder(weights=wfile, loglevel='ERROR')
            _bench(tag, WEIGHTS_CASE, regridder, ds, time.perf_counter() - t0, repeat, outdir)
        except Exception as err:
            log.error('%s failed: %r', WEIGHTS_CASE, err)
            print(json.dumps({'tag': tag, 'case': WEIGHTS_CASE, 'error': repr(err)[:300]}), flush=True)


# --------------------------------------------------------------------------------------
# Driver: exports the versions, runs the workers and summarizes
# --------------------------------------------------------------------------------------

def export_version(ref, dest):
    """Export the smmregrid package of a git ref (or of the working tree) into dest"""
    log.info('Exporting %s to %s', ref, dest)
    os.makedirs(dest)
    if ref == 'WORKTREE':
        shutil.copytree(os.path.join(REPO, 'smmregrid'), os.path.join(dest, 'smmregrid'),
                        ignore=shutil.ignore_patterns('__pycache__'))
        return
    archive = subprocess.run(['git', '-C', REPO, 'archive', ref, 'smmregrid'],
                             check=True, capture_output=True).stdout
    subprocess.run(['tar', '-x', '-C', dest], input=archive, check=True)


def run_worker(ref, srcdir, outdir, workdir, repeat, cases, loglevel):
    """Run the worker for one version in a separate process, return parsed results"""
    env = dict(os.environ, PYTHONPATH=srcdir)
    cmd = [sys.executable, os.path.abspath(__file__), '--worker', ref,
           '--outdir', outdir, '--repeat', str(repeat), '--loglevel', loglevel, '--cases', *cases]
    log.debug('Running %s', ' '.join(cmd))
    # cwd outside the repository, so that the installed/editable smmregrid is not picked up.
    # stdout carries the results, stderr (worker logs) goes straight to the terminal
    proc = subprocess.run(cmd, env=env, cwd=workdir, stdout=subprocess.PIPE, text=True)
    results = [json.loads(line) for line in proc.stdout.splitlines() if line.startswith('{')]
    if proc.returncode != 0:
        log.error('Worker for %s exited with code %s', ref, proc.returncode)
        results.append({'tag': ref, 'case': '-', 'error': f'worker exited with code {proc.returncode}'})
    return results


def _median(rows, tag, case, key):
    values = [r[key] for r in rows if r['tag'] == tag and r['case'] == case and key in r]
    return statistics.median(values) if values else float('nan')


def summarize(rows, refs, cases, outdir):
    """Print the comparison table and the output check"""
    base, others = refs[0], refs[1:]
    names = ' / '.join(refs)

    print(f'\nMedian times in ms (build = regrid() call, compute = .compute()), refs: {names}')
    print(f"| case | build ({names}) | compute ({names}) | " +
          ' | '.join(f'total {o} vs {base}' for o in others) + ' |')
    print('|---|---|---|' + '---|' * len(others))
    for case in cases:
        build = [_median(rows, r, case, 'build') for r in refs]
        comp = [_median(rows, r, case, 'compute') for r in refs]
        ratios = [f'{100 * ((build[i] + comp[i]) / (build[0] + comp[0]) - 1):+.0f}%'
                  for i in range(1, len(refs))]
        print(f"| {case} | {' / '.join(f'{b:.1f}' for b in build)} | "
              f"{' / '.join(f'{c:.1f}' for c in comp)} | {' | '.join(ratios)} |")

    print(f'\nMedian init time in s, refs: {names}')
    for case in cases:
        print(f"  {case:24s} " + ' / '.join(f"{_median(rows, r, case, 'init'):.2f}" for r in refs))

    errors = [r for r in rows if 'error' in r]
    print('\nErrors:' + (' none' if not errors else ''))
    for err in errors:
        print(f"  [{err['tag']}] {err['case']}: {err['error']}")

    import xarray as xr
    warnings.filterwarnings('ignore')
    print(f'\nOutput check against {base}:')
    for case in cases:
        fname = case.replace(' ', '_') + '.nc'
        ref_file = os.path.join(outdir, f'{base}__{fname}')
        for other in others:
            other_file = os.path.join(outdir, f'{other}__{fname}')
            if not (os.path.exists(ref_file) and os.path.exists(other_file)):
                print(f'  {case:24s} {other}: missing output')
                continue
            with xr.open_dataset(ref_file) as a, xr.open_dataset(other_file) as b:
                try:
                    xr.testing.assert_identical(a, b)
                    res = 'identical'
                except AssertionError:
                    try:
                        xr.testing.assert_allclose(a, b)
                        res = 'values close, metadata differ'
                    except AssertionError as err:
                        res = 'DIFFERENT: ' + ' '.join(str(err).split())[:200]
            print(f'  {case:24s} {other}: {res}')


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--refs', nargs='+', default=['main', 'WORKTREE'],
                        help='git refs to compare, the first is the reference. '
                             'WORKTREE is the current code (default: main WORKTREE)')
    parser.add_argument('--uncommitted', action='store_true',
                        help='compare the last commit against the uncommitted changes, '
                             'shortcut for --refs HEAD WORKTREE')
    parser.add_argument('--reps', type=int, default=2,
                        help='full runs per version, alternated across versions (default: 2)')
    parser.add_argument('--repeat', type=int, default=8,
                        help='timed regrid+compute calls per case and run (default: 8)')
    parser.add_argument('--cases', nargs='+', default=ALL_CASES, choices=ALL_CASES,
                        metavar='CASE', help='subset of cases to run (default: all)')
    parser.add_argument('--outdir', default=None,
                        help='where to keep outputs and raw results (default: temporary)')
    parser.add_argument('--loglevel', default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], type=str.upper,
                        help='logging level, logs go to stderr (default: INFO)')
    parser.add_argument('--list-cases', action='store_true', help='list the cases and exit')
    parser.add_argument('--worker', default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.list_cases:
        print('\n'.join(ALL_CASES))
        return

    setup_logging(args.loglevel, args.worker or 'driver')

    if args.worker:
        worker(args.worker, args.outdir, args.repeat, args.cases)
        return

    if args.uncommitted:
        args.refs = ['HEAD', 'WORKTREE']
        diff = subprocess.run(['git', '-C', REPO, 'status', '--porcelain', '--', 'smmregrid'],
                              check=True, capture_output=True, text=True).stdout
        if not diff.strip():
            log.warning('No uncommitted changes in smmregrid/, HEAD and WORKTREE are the same code')

    tmp = tempfile.mkdtemp(prefix='smmregrid-bench-')
    outdir = os.path.abspath(args.outdir) if args.outdir else os.path.join(tmp, 'out')
    os.makedirs(outdir, exist_ok=True)
    # refs as safe tags, usable in file names (e.g. origin/main -> origin_main)
    tags = [ref.replace('/', '_') for ref in args.refs]
    log.info('Comparing %s on %d cases, %d runs x %d calls each',
             ' vs '.join(args.refs), len(args.cases), args.reps, args.repeat)
    tstart = time.perf_counter()
    try:
        srcdirs = {}
        for ref, tag in zip(args.refs, tags):
            srcdirs[tag] = os.path.join(tmp, 'src', tag)
            export_version(ref, srcdirs[tag])

        rows = []
        for rep in range(args.reps):
            for tag in tags:  # alternate versions to spread out machine noise
                log.info('Run %d/%d: %s', rep + 1, args.reps, tag)
                rows += run_worker(tag, srcdirs[tag], outdir, tmp, args.repeat, args.cases, args.loglevel)
        log.info('All runs done in %.0fs', time.perf_counter() - tstart)

        with open(os.path.join(outdir, 'results.jsonl'), 'w', encoding='utf-8') as f:
            f.writelines(json.dumps(r) + '\n' for r in rows)

        summarize(rows, tags, args.cases, outdir)
        if args.outdir:
            log.info('Raw results and outputs in %s', outdir)
    finally:
        shutil.rmtree(os.path.join(tmp, 'src'), ignore_errors=True)
        if not args.outdir:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == '__main__':
    main()
