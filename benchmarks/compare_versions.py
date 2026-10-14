#!/usr/bin/env python
"""
Compare smmregrid performance across git refs (branches, commits) and the working tree.

Each version is exported to a temporary directory and run in its own Python process,
so that versions never mix. For each test case it measures:
    - init:    Regridder initialization (dominated by CDO weights generation)
    - build:   the regrid() call, i.e. the dask graph construction
    - compute: the .compute() of the regridded output

and finally checks that the outputs of all versions match the first one.

With --distributed N, each version also runs 2D and 3D cases (from grids and from
weights on disk) on a dask LocalCluster
with N worker processes, timing the compute and checking that the output matches
the serial (synchronous scheduler) one.

With --tile N, every dataset is repeated N times along time (default: 1, no repetition),
to emulate longer datasets.

Usage (from the repository root):
    python benchmarks/compare_versions.py                         # main vs working tree
    python benchmarks/compare_versions.py --refs main HEAD WORKTREE --reps 3
    python benchmarks/compare_versions.py --uncommitted               # last commit vs uncommitted changes
    python benchmarks/compare_versions.py --cases "3D nemo" "unstructured fesom" --tile 60
    python benchmarks/compare_versions.py --list-cases
    python benchmarks/compare_versions.py --distributed 4 --cases "lonlat era5"  # add the cluster check

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

# name, file, variable (None for all). All are repeated --tile times along time
CASES = [
    ('lonlat era5', '2t-era5.nc', '2t'),
    ('gaussian ecearth', 'tas-ecearth.nc', None),
    ('healpix', 'tas-healpix2.nc', None),
    ('unstructured fesom', 'tos-fesom.nc', 'tos'),
    ('curvilinear ipsl', 'onlytos-ipsl.nc', None),
    ('plev gaussian', 'ua-ecearth.nc', None),
    ('3D nemo', 'so3d-nemo.nc', 'so'),
    ('3D fesom', 'temp3d-fesom.nc', None),
    ('multigrid cesm', 'mix-cesm.nc', None),
]
# extra case: same as 'unstructured fesom', but initializing from a weights file on disk
WEIGHTS_CASE = 'fesom from weights'
ALL_CASES = [c[0] for c in CASES] + [WEIGHTS_CASE]
# cases run on a dask LocalCluster with --distributed: 2D and 3D,
# initialized from grids or from a weights file on disk, with or without skipna
# (name, file, variable, mask_dim, init, skipna). All are repeated --tile times along time
DISTRIBUTED_CASES = [
    ('fesom', 'tos-fesom.nc', 'tos', None, 'grids', False),
    ('3D nemo', 'so3d-nemo.nc', 'so', None, 'grids', False),
    ('fesom from weights', 'tos-fesom.nc', 'tos', None, 'weights', False),
    ('3D nemo from weights', 'so3d-nemo.nc', 'so', 'lev', 'weights', False),
    ('ipsl skipna', 'onlytos-ipsl.nc', 'tos', None, 'grids', True),
    ('3D nemo skipna', 'so3d-nemo.nc', 'so', None, 'grids', True),
]

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
    """Open a test file lazily as dask arrays, with the chunking chosen by dask/the backend.
    When tile > 1 repeat it along time, to emulate longer datasets (one chunk per copy, at least).
    The data are never loaded in memory: they would be embedded in the dask graph,
    which is then serialized at each compute (very costly with distributed)"""
    ds = xr.open_dataset(os.path.join(DATA, fname), chunks={})
    if var:
        ds = ds[[var]]
    if tile > 1:
        # only the variables with time are repeated (e.g. not the grid bounds)
        ds = xr.concat([ds] * tile, dim='time', data_vars='minimal', coords='minimal', compat='override')
    return ds


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


def distributed_check(tag, outdir, nworkers, repeat, tile=1):
    """Run DISTRIBUTED_CASES on a LocalCluster with nworkers processes: time the compute
    and check that the output matches the serial one"""
    import inspect
    import dask
    import xarray as xr
    from distributed import Client, LocalCluster
    from smmregrid import Regridder, CdoGenerate

    # older versions (e.g. main before #70) have no skipna: pass it only when used,
    # and skip the skipna cases on versions which do not support it
    has_skipna = 'skipna' in inspect.signature(Regridder.__init__).parameters

    log.info('Starting a LocalCluster with %d workers', nworkers)
    with LocalCluster(n_workers=nworkers, threads_per_worker=1, processes=True,
                      dashboard_address=None) as cluster, Client(cluster) as client:
        for case, fname, var, mask_dim, init, skipna in DISTRIBUTED_CASES:
            name = f'{case} [{nworkers} workers]'
            if skipna and not has_skipna:
                log.warning('%s skipped: skipna not supported by this version', name)
                print(json.dumps({'tag': tag, 'case': name, 'error': 'skipna not supported by this version'}),
                      flush=True)
                continue
            kwargs = {'skipna': True} if skipna else {}
            try:
                ds = _load(xr, fname, var, tile)
                if init == 'weights':
                    wfile = os.path.join(outdir, f'{tag}__dist_weights_{fname}')
                    CdoGenerate(os.path.join(DATA, fname), TARGET, loglevel='ERROR',
                                **kwargs).weights(method='con', mask_dim=mask_dim).to_netcdf(wfile)
                    regridder = Regridder(weights=wfile, loglevel='ERROR', **kwargs)
                else:
                    regridder = Regridder(source_grid=os.path.join(DATA, fname),
                                          target_grid=TARGET, loglevel='ERROR', **kwargs)
                out = regridder.regrid(ds)
                with dask.config.set(scheduler='synchronous'):
                    t0 = time.perf_counter()
                    serial = out.compute()
                    tserial = time.perf_counter() - t0
                client.compute(out).result()  # warmup, also ships the weights to the workers
                tdist = 0.
                for _ in range(repeat):
                    t0 = time.perf_counter()
                    dist = client.compute(out).result()
                    tdist += time.perf_counter() - t0
                try:
                    xr.testing.assert_allclose(dist, serial)
                    check = 'ok'
                except AssertionError as err:
                    check = 'DIFFERENT: ' + ' '.join(str(err).split())[:200]
                print(json.dumps({'tag': tag, 'case': name, 'serial': 1000 * tserial,
                                  'distributed': 1000 * tdist / repeat, 'check': check}), flush=True)
                log.info('%-36s serial %8.1f ms  distributed %8.1f ms  check %s',
                         name, 1000 * tserial, 1000 * tdist / repeat, check)
            except Exception as err:
                log.error('%s failed: %r', name, err)
                print(json.dumps({'tag': tag, 'case': name, 'error': repr(err)[:300]}), flush=True)


def worker(tag, outdir, repeat, cases, tile=1):
    """Run all the selected cases for the smmregrid version found in PYTHONPATH"""
    import xarray as xr
    from smmregrid import Regridder, CdoGenerate
    warnings.filterwarnings('ignore')

    for case, fname, var in CASES:
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
            ds = _load(xr, 'tos-fesom.nc', 'tos', tile)
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


def run_worker(ref, srcdir, outdir, workdir, repeat, cases, loglevel, nworkers=0, tile=1):
    """Run the worker for one version in a separate process, return parsed results"""
    # PYTHONPATH is inherited also by the processes of the dask cluster
    env = dict(os.environ, PYTHONPATH=srcdir)
    cmd = [sys.executable, os.path.abspath(__file__), '--worker', ref,
           '--outdir', outdir, '--repeat', str(repeat), '--loglevel', loglevel,
           '--distributed', str(nworkers), '--tile', str(tile), '--cases', *cases]
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


def _fmt(value, spec='.1f'):
    """Format a number for the tables, with a dash for missing values"""
    return '–' if value != value else format(value, spec)  # NaN check


def _delta(new, ref):
    """Relative change of new vs ref, as a signed percentage"""
    if new != new or ref != ref or ref == 0:
        return '–'
    return f'{100 * (new / ref - 1):+.0f}%'


def _table(header, rows_):
    """Print a markdown table, the first column left aligned and the others right aligned"""
    print('| ' + ' | '.join(header) + ' |')
    print('|:---|' + '---:|' * (len(header) - 1))
    for row in rows_:
        print('| ' + ' | '.join(row) + ' |')


def summarize(rows, refs, cases, outdir):
    """Print the comparison as markdown tables, and the output check"""
    base, others = refs[0], refs[1:]

    print(f'\n### Regrid time (median, ms)\n\nbuild = `regrid()` call, compute = `.compute()`, '
          f'total change is relative to `{base}`\n')
    lines = []
    for case in cases:
        build = [_median(rows, r, case, 'build') for r in refs]
        comp = [_median(rows, r, case, 'compute') for r in refs]
        total = [b + c for b, c in zip(build, comp)]
        lines.append([case] + [_fmt(b) for b in build] + [_fmt(c) for c in comp] +
                     [_delta(t, total[0]) for t in total[1:]])
    _table(['case'] + [f'build {r}' for r in refs] + [f'compute {r}' for r in refs] +
           [f'total {o} vs {base}' for o in others], lines)

    print('\n### Init time (median, s)\n\nRegridder initialization, dominated by CDO weights generation, '
          f'change is relative to `{base}`\n')
    lines = []
    for case in cases:
        init = [_median(rows, r, case, 'init') for r in refs]
        lines.append([case] + [_fmt(i, '.2f') for i in init] + [_delta(i, init[0]) for i in init[1:]])
    _table(['case'] + refs + [f'{o} vs {base}' for o in others], lines)

    # distributed cases are stored as "<case> [N workers]"
    dist_cases = sorted({r['case'] for r in rows if r['case'].endswith('workers]')})
    if dist_cases:
        nworkers = dist_cases[0].rsplit('[', 1)[1].split()[0]
        print(f'\n### Distributed check, {nworkers} workers (median, ms)\n\n'
              'serial = synchronous scheduler (single call), distributed = LocalCluster, '
              'slowdown = distributed / serial, check = distributed output equal to the serial one\n')
        lines = []
        for case in dist_cases:
            serial = [_median(rows, r, case, 'serial') for r in refs]
            dist = [_median(rows, r, case, 'distributed') for r in refs]
            slow = [_fmt(d / s, '.1f') + 'x' if s == s and d == d else '–' for s, d in zip(serial, dist)]
            checks = []
            for r in refs:
                values = {row.get('check', 'error') for row in rows if row['tag'] == r and row['case'] == case}
                if not values:
                    checks.append('–')
                elif values == {'ok'}:
                    checks.append('ok')
                else:  # a crash, or an output different from the serial one
                    checks.append('error' if values == {'error'} else 'DIFFERENT')
            check = 'ok' if set(checks) == {'ok'} else ', '.join(f'{r}: {c}' for r, c in zip(refs, checks))
            lines.append([case.rsplit(' [', 1)[0]] + [_fmt(x) for x in serial] + [_fmt(x) for x in dist] +
                         slow + [_delta(d, dist[0]) for d in dist[1:]] + [check])
        _table(['case'] + [f'serial {r}' for r in refs] + [f'distributed {r}' for r in refs] +
               [f'slowdown {r}' for r in refs] + [f'distributed {o} vs {base}' for o in others] + ['check'],
               lines)

    errors = {(r['tag'], r['case']): r['error'] for r in rows if 'error' in r}
    print('\n### Errors\n')
    if not errors:
        print('none')
    for (tag, case), err in sorted(errors.items()):
        print(f'- **{tag}**, {case}: `{err[:150]}`')

    import xarray as xr
    warnings.filterwarnings('ignore')
    print(f'\n### Output check against {base}\n')
    lines = []
    for case in cases:
        fname = case.replace(' ', '_') + '.nc'
        ref_file = os.path.join(outdir, f'{base}__{fname}')
        line = [case]
        for other in others:
            other_file = os.path.join(outdir, f'{other}__{fname}')
            if not (os.path.exists(ref_file) and os.path.exists(other_file)):
                line.append('missing output')
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
                        res = 'DIFFERENT: ' + ' '.join(str(err).split())[:100]
            line.append(res)
        lines.append(line)
    _table(['case'] + others, lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--refs', nargs='+', default=['main', 'WORKTREE'],
                        help='git refs to compare, the first is the reference. '
                             'WORKTREE is the current code (default: main WORKTREE)')
    parser.add_argument('--uncommitted', action='store_true',
                        help='compare the last commit against the uncommitted changes, '
                             'shortcut for --refs HEAD WORKTREE')
    parser.add_argument('--reps', type=int, default=5,
                        help='full runs per version, alternated across versions (default: 5)')
    parser.add_argument('--repeat', type=int, default=8,
                        help='timed regrid+compute calls per case and run (default: 8)')
    parser.add_argument('--cases', nargs='+', default=ALL_CASES, choices=ALL_CASES,
                        metavar='CASE', help='subset of cases to run (default: all)')
    parser.add_argument('--outdir', default=None,
                        help='where to keep outputs and raw results (default: temporary)')
    parser.add_argument('--loglevel', default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], type=str.upper,
                        help='logging level, logs go to stderr (default: INFO)')
    parser.add_argument('--distributed', type=int, default=0, metavar='N',
                        help='also run %s on a dask LocalCluster with N workers, checking '
                             'the output against the serial one (default: 0, disabled)'
                             % ', '.join(f'"{c[0]}"' for c in DISTRIBUTED_CASES))
    parser.add_argument('--tile', type=int, default=1, metavar='N',
                        help='repeat every dataset N times along time, to emulate longer datasets '
                             '(default: 1, no repetition)')
    parser.add_argument('--list-cases', action='store_true', help='list the cases and exit')
    parser.add_argument('--worker', default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.list_cases:
        print('\n'.join(ALL_CASES))
        return

    setup_logging(args.loglevel, args.worker or 'driver')

    if args.worker:
        worker(args.worker, args.outdir, args.repeat, args.cases, tile=args.tile)
        if args.distributed > 0:
            distributed_check(args.worker, args.outdir, args.distributed, args.repeat, tile=args.tile)
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
    log.info('Comparing %s on %d cases, %d runs x %d calls each, data repeated x%d along time',
             ' vs '.join(args.refs), len(args.cases), args.reps, args.repeat, args.tile)
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
                rows += run_worker(tag, srcdirs[tag], outdir, tmp, args.repeat, args.cases, args.loglevel,
                                   nworkers=args.distributed, tile=args.tile)
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
