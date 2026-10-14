#!/usr/bin/env python
"""
Compare smmregrid performance across git refs (branches, commits) and the working tree.

Each version is exported to a temporary directory and run in its own Python process,
so that versions never mix. For each test case it measures:
    - init:    Regridder initialization (dominated by CDO weights generation)
    - build:   the regrid() call, i.e. the dask graph construction
    - compute: the .compute() of the regridded output

and finally checks that the outputs of all versions match the first one.

With --distributed N, each version also runs a scaling test on 2D and 3D cases (from grids
and from weights on disk, with and without skipna): the regridded output is persisted serially
(synchronous scheduler) and on dask LocalClusters with 1, 2, ..., N worker processes, reporting
the speedup. The output of the N-worker cluster is checked against the serial one (on the
untiled data, to keep memory low). The scaling test runs once per version (first run only),
as it is slow. Memory: each worker gets 1/N of the machine memory, and a worker exceeding it is
killed (seen as CommClosedError on the others): reduce --tile or --chunk-size if it happens.

Data are opened as dask arrays with chunks of about --chunk-size (default 64 MiB) along time,
and whole along the other dimensions: the on-disk chunking of the test files (one time step)
is far too fine for dask, especially with distributed. With --tile N, every dataset is
repeated N times along time to emulate longer datasets (default: 1), written once to a file in
--outdir and read back, so that the repetition adds no tasks to the graph. The test files are small:
the scaling test needs several chunks per case to be meaningful, i.e. --tile of 60-120.

Usage (from the repository root):
    python benchmarks/compare_versions.py                         # main vs working tree
    python benchmarks/compare_versions.py --refs main HEAD WORKTREE --reps 3
    python benchmarks/compare_versions.py --uncommitted               # last commit vs uncommitted changes
    python benchmarks/compare_versions.py --cases "3D nemo" "unstructured fesom" --tile 60
    python benchmarks/compare_versions.py --list-cases
    python benchmarks/compare_versions.py --distributed 4 --tile 120 --cases "lonlat era5"  # scaling test

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
# cases of the scaling test with --distributed: 2D and 3D, initialized from grids or from
# a weights file on disk, with or without skipna. (name, file, variable, mask_dim, init, skipna)
# All are repeated --tile times along time, as the other cases
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

def _load(xr, fname, var=None, tile=1, outdir=None):
    """Open a test file lazily as dask arrays, chunked along time only, with chunks of about
    dask's array.chunk-size (set from --chunk-size). When tile > 1 repeat it along time,
    to emulate longer datasets.
    chunks={} would take the on-disk chunking (one time step, < 1 MB, even split horizontally
    for some files): far too fine for dask, the scheduling overhead would dominate the timings.
    The data are never loaded in memory: they would be embedded in the dask graph,
    which is then serialized at each compute (very costly with distributed)"""
    path = os.path.join(DATA, fname)
    if tile > 1:
        path = _tiled_file(xr, fname, tile, outdir)
    ds = xr.open_dataset(path, chunks='auto')
    if var:
        ds = ds[[var]]
    # whole chunks along all the dimensions but time: the horizontal ones must be whole
    # for the regridding (a split would force a rechunk), vertical levels are handled by smmregrid
    chunks = {dim: -1 for dim in ds.dims if dim != 'time'}
    if 'time' in ds.dims:
        chunks['time'] = 'auto'
    return ds.chunk(chunks)


def _tiled_file(xr, fname, tile, outdir):
    """Write once, and reuse, a copy of a test file repeated tile times along time.
    Repeating it lazily with xr.concat would add tile x (on-disk chunks) tasks to every graph
    (e.g. 4320 tasks for 3D nemo with --tile 60), whose scheduling overhead would dominate the
    timings on a distributed cluster: reading a longer file is closer to a real use case.
    The file is shared by all the versions, as the data do not depend on smmregrid"""
    path = os.path.join(outdir, f'tiled{tile}__{fname}')
    if not os.path.exists(path):
        log.debug('writing %s', path)
        with xr.open_dataset(os.path.join(DATA, fname), chunks={}) as ds:
            # only the variables with time are repeated (e.g. not the grid bounds)
            tiled = xr.concat([ds] * tile, dim='time', data_vars='minimal', coords='minimal', compat='override')
            tmp = f'{path}.{os.getpid()}.tmp'
            tiled.to_netcdf(tmp)
        os.replace(tmp, path)
    return path


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


def _worker_counts(nworkers):
    """1, 2, 4, ... up to nworkers, plus nworkers itself"""
    counts = {1, nworkers}
    counts.update(2 ** k for k in range(1, nworkers.bit_length()) if 2 ** k < nworkers)
    return sorted(counts)


def _trace_start():
    """Run in a process (the workers via client.run): start tracing memory allocations.
    tracemalloc tracks numpy arrays too, and starts with a fresh peak: unlike the RSS of the
    process, which never shrinks back, it isolates the peak memory of a single computation"""
    import gc
    import tracemalloc
    gc.collect()
    tracemalloc.start()


def _trace_peak():
    """Peak traced memory (MB) since _trace_start, and stop tracing"""
    import tracemalloc
    peak = tracemalloc.get_traced_memory()[1] / 2**20
    tracemalloc.stop()
    return peak


def _peak_memory(persist, start=_trace_start, peak=_trace_peak):
    """Peak memory (MB) of one persist call, traced with tracemalloc.
    Done in a call of its own, after the timed ones, as tracing slows the computation down"""
    start()
    result = persist()
    mem = peak()
    del result
    return mem


def _time_persist(persist, repeat):
    """Median time in ms of persisting the output `repeat` times (plus a warmup).
    The result is dropped before the next iteration, which releases its futures/memory
    (client.cancel() must not be used: it would also cancel the next submission of the same keys)"""
    import gc
    result = persist()  # warmup
    times = []
    for _ in range(repeat):
        del result
        gc.collect()
        t0 = time.perf_counter()
        result = persist()
        times.append(time.perf_counter() - t0)
    del result
    gc.collect()
    return 1000 * statistics.median(times)


def distributed_check(tag, outdir, nworkers, repeat, tile=1):
    """Scaling test of DISTRIBUTED_CASES: persist the regridded output serially (synchronous
    scheduler, in process) and on LocalClusters with 1, 2, ... nworkers processes (one thread each).
    Persisting leaves the result on the workers, as writing to disk would: gathering it to the client
    would add a cost proportional to the output size. The output of the nworkers cluster is checked
    against the serial one on the untiled data, so that no large result is ever kept in the client"""
    import inspect
    import dask
    import psutil
    import xarray as xr
    from distributed import Client, LocalCluster, wait
    from smmregrid import Regridder, CdoGenerate

    # older versions (e.g. main before #70) have no skipna: pass it only when used,
    # and skip the skipna cases on versions which do not support it
    has_skipna = 'skipna' in inspect.signature(Regridder.__init__).parameters

    # build all the regridders and graphs first, then time them on each scheduler
    outputs = {}
    for case, fname, var, mask_dim, init, skipna in DISTRIBUTED_CASES:
        if skipna and not has_skipna:
            log.warning('%s skipped: skipna not supported by this version', case)
            print(json.dumps({'tag': tag, 'case': case, 'kind': 'dist',
                              'error': 'skipna not supported by this version'}), flush=True)
            continue
        kwargs = {'skipna': True} if skipna else {}
        try:
            ds = _load(xr, fname, var, tile, outdir)
            if init == 'weights':
                wfile = os.path.join(outdir, f'{tag}__dist_weights_{fname}')
                CdoGenerate(os.path.join(DATA, fname), TARGET, loglevel='ERROR',
                            **kwargs).weights(method='con', mask_dim=mask_dim).to_netcdf(wfile)
                regridder = Regridder(weights=wfile, loglevel='ERROR', **kwargs)
            else:
                regridder = Regridder(source_grid=os.path.join(DATA, fname),
                                      target_grid=TARGET, loglevel='ERROR', **kwargs)
            out = regridder.regrid(ds)
            # small graph on the untiled data, for the correctness check of the cluster
            small = regridder.regrid(_load(xr, fname, var, 1))
            outputs[case] = (var, out, small)
            if ds[var].data.npartitions < nworkers:
                log.warning('%s: only %d chunk(s) for %d workers, increase --tile for a meaningful scaling',
                            case, ds[var].data.npartitions, nworkers)
            print(json.dumps({'tag': tag, 'case': case, 'kind': 'dist',
                              'in_mb': ds[var].nbytes / 2**20, 'out_mb': out[var].nbytes / 2**20,
                              'chunks': ds[var].data.npartitions,
                              'tasks': len(out[var].data.__dask_graph__())}), flush=True)
        except Exception as err:
            log.error('%s failed: %r', case, err)
            print(json.dumps({'tag': tag, 'case': case, 'kind': 'dist', 'error': repr(err)[:300]}), flush=True)

    def report(case, scheduler, count, ms, mem_total):
        print(json.dumps({'tag': tag, 'case': case, 'kind': 'dist', 'scheduler': scheduler,
                          'nworkers': count, 'time': ms, 'mem_total': mem_total}), flush=True)
        log.info('%-24s %-8s x%-2d %8.1f ms  peak memory %6.0f MB', case, scheduler, count, ms, mem_total)

    # serial reference: synchronous scheduler in this process, result dropped right away
    for case, (var, out, small) in outputs.items():
        serial = lambda: dask.compute(out, scheduler='synchronous')  # noqa: E731
        ms = _time_persist(serial, repeat)
        mem = _peak_memory(serial)
        report(case, 'serial', 1, ms, mem)

    for count in _worker_counts(nworkers):
        log.info('Starting a LocalCluster with %d workers, memory limit %.1f GB each',
                 count, psutil.virtual_memory().total / count / 2**30)
        with LocalCluster(n_workers=count, threads_per_worker=1, processes=True,
                          dashboard_address=None) as cluster, Client(cluster) as client:

            def persist(out):
                persisted = client.persist(out)
                wait(persisted)
                return persisted

            for case, (var, out, small) in outputs.items():
                try:
                    ms = _time_persist(lambda: persist(out), repeat)
                    peaks = _peak_memory(lambda: persist(out),
                                         start=lambda: client.run(_trace_start),
                                         peak=lambda: list(client.run(_trace_peak).values()))
                    report(case, 'cluster', count, ms, sum(peaks))
                    if count == nworkers:
                        # correctness: the output of the full cluster must match the serial one
                        try:
                            xr.testing.assert_allclose(client.compute(small).result(),
                                                       dask.compute(small, scheduler='synchronous')[0])
                            check = 'ok'
                        except AssertionError as err:
                            check = 'DIFFERENT: ' + ' '.join(str(err).split())[:200]
                        print(json.dumps({'tag': tag, 'case': case, 'kind': 'dist', 'check': check}), flush=True)
                        log.info('%-24s check on %d workers: %s', case, count, check)
                except Exception as err:
                    log.error('%s failed on %d workers: %r', case, count, err)
                    print(json.dumps({'tag': tag, 'case': case, 'kind': 'dist',
                                      'error': f'{count} workers: {repr(err)[:300]}'}), flush=True)


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
            ds = _load(xr, fname, var, tile, outdir)
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
            ds = _load(xr, 'tos-fesom.nc', 'tos', tile, outdir)
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


def run_worker(ref, srcdir, outdir, workdir, args, cases, nworkers=0):
    """Run the worker for one version in a separate process, return parsed results"""
    # PYTHONPATH is inherited also by the processes of the dask cluster
    env = dict(os.environ, PYTHONPATH=srcdir)
    cmd = [sys.executable, os.path.abspath(__file__), '--worker', ref,
           '--outdir', outdir, '--repeat', str(args.repeat), '--loglevel', args.loglevel,
           '--distributed', str(nworkers), '--dist-repeat', str(args.dist_repeat),
           '--tile', str(args.tile), '--chunk-size', args.chunk_size, '--cases', *cases]
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

    dist = [r for r in rows if r.get('kind') == 'dist']
    dist_cases = [c[0] for c in DISTRIBUTED_CASES if any(r['case'] == c[0] for r in dist)]
    if dist_cases:
        counts = sorted({r['nworkers'] for r in dist if r.get('scheduler') == 'cluster'})
        nworkers = max(counts) if counts else 'N'

        print('\n### Scaling test: data\n\nsizes of the input and of the regridded output, '
              'input dask chunks and tasks of the graph\n')
        lines = []
        for case in dist_cases:
            info = next((r for r in dist if r['case'] == case and 'in_mb' in r), {})
            lines.append([case] + [_fmt(info.get(k, float('nan')), f)
                                   for k, f in (('in_mb', '.0f'), ('out_mb', '.0f'), ('chunks', 'd'), ('tasks', 'd'))])
        _table(['case', 'input MB', 'output MB', 'chunks', 'tasks'], lines)

        print(f'\n### Scaling test: persist time (median, ms)\n\n'
              'serial = synchronous scheduler (no cluster), cluster k = LocalCluster with k worker '
              'processes of one thread each. speedup k = serial / cluster k (above 1x the cluster is '
              f'faster than serial), scaling = cluster 1 / cluster {nworkers} (gain of the extra workers), '
              f'check = output of the {nworkers} workers cluster equal to the serial one\n')
        lines = []
        for case in dist_cases:
            for ref in refs:
                def t(scheduler, count):
                    values = [r['time'] for r in dist if r['tag'] == ref and r['case'] == case
                              and r.get('scheduler') == scheduler and r.get('nworkers') == count]
                    return statistics.median(values) if values else float('nan')
                serial = t('serial', 1)
                clu = [t('cluster', c) for c in counts]

                def ratio(num, den):
                    return _fmt(num / den, '.2f') + 'x' if num == num and den == den and den else '–'
                speedups = [ratio(serial, c) for c in clu]
                scaling = ratio(clu[0], clu[-1]) if clu else '–'
                checks = {r['check'] for r in dist if r['tag'] == ref and r['case'] == case and 'check' in r}
                errors_ = [r for r in dist if r['tag'] == ref and r['case'] == case and 'error' in r]
                check = 'ok' if checks == {'ok'} else ('DIFFERENT' if checks else ('error' if errors_ else '–'))
                lines.append([f'{case} ({ref})', _fmt(serial, '.0f')] + [_fmt(x, '.0f') for x in clu] +
                             speedups + [scaling, check])
        _table(['case (ref)', 'serial'] + [f'cluster {c}' for c in counts] +
               [f'speedup {c}' for c in counts] + ['scaling', 'check'], lines)

        print('\n### Scaling test: peak memory (median, MB)\n\n'
              'peak of the memory allocated by one persist call (tracemalloc, numpy arrays included, '
              'in an extra untimed call), i.e. the intermediate results plus the persisted output: '
              'the memory of the processes at rest is not included. cluster k = sum of the peaks '
              'of the k workers\n')
        lines = []
        for case in dist_cases:
            for ref in refs:
                def m(scheduler, count, key):
                    values = [r[key] for r in dist if r['tag'] == ref and r['case'] == case
                              and r.get('scheduler') == scheduler and r.get('nworkers') == count and key in r]
                    return statistics.median(values) if values else float('nan')
                lines.append([f'{case} ({ref})', _fmt(m('serial', 1, 'mem_total'), '.0f')] +
                             [_fmt(m('cluster', c, 'mem_total'), '.0f') for c in counts])
        _table(['case (ref)', 'serial'] + [f'cluster {c}' for c in counts], lines)

    errors = {(r['tag'], r['case']): r['error'] for r in rows if 'error' in r}
    print('\n### Errors\n')
    if not errors:
        print('none')
    for (tag, case), err in sorted(errors.items()):
        print(f'- **{tag}**, {case}: `{err[:150]}`')

    if not others:  # nothing to compare the outputs with
        return
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
                        help='also run the scaling test (%s) with the threaded scheduler and on '
                             'dask LocalClusters with 1, 2, ... N workers, once per version '
                             '(default: 0, disabled)' % ', '.join(f'"{c[0]}"' for c in DISTRIBUTED_CASES))
    parser.add_argument('--dist-repeat', type=int, default=3,
                        help='timed persist calls per case and scheduler in the scaling test (default: 3)')
    parser.add_argument('--tile', type=int, default=1, metavar='N',
                        help='repeat every dataset N times along time, to emulate longer datasets, '
                             'in all the cases including the scaling test (default: 1, no repetition)')
    parser.add_argument('--chunk-size', default='64MiB',
                        help='target size of the dask chunks along time, as dask array.chunk-size '
                             '(default: 64MiB)')
    parser.add_argument('--list-cases', action='store_true', help='list the cases and exit')
    parser.add_argument('--worker', default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.list_cases:
        print('\n'.join(ALL_CASES))
        return

    setup_logging(args.loglevel, args.worker or 'driver')

    if args.worker:
        import dask
        dask.config.set({'array.chunk-size': args.chunk_size})
        worker(args.worker, args.outdir, args.repeat, args.cases, tile=args.tile)
        if args.distributed > 0:
            distributed_check(args.worker, args.outdir, args.distributed, args.dist_repeat,
                              tile=args.tile)
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
    log.info('Comparing %s on %d cases, %d runs x %d calls each, data repeated x%d along time, '
             'chunks of %s', ' vs '.join(args.refs), len(args.cases), args.reps, args.repeat,
             args.tile, args.chunk_size)
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
                # the scaling test is slow: only in the first run
                rows += run_worker(tag, srcdirs[tag], outdir, tmp, args, args.cases,
                                   nworkers=args.distributed if rep == 0 else 0)
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
