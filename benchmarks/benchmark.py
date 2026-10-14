#!/usr/bin/env python
"""
Benchmark smmregrid: time, memory and dask graph of a fixed matrix of cases, on the serial
scheduler and on dask LocalClusters, optionally comparing two versions of the code.

Cases (see CASES): 2D without NaNs, 2D with a static mask (with and without skipna), 2D with a
time-varying mask (skipna), 3D with a depth-varying mask (with and without skipna), and a 2D case
on a 0.25 deg target grid, whose larger weights stress the size of the dask graph.

For each case it measures:
    - init:    Regridder(source_grid, target_grid), dominated by the CDO weights generation (s)
    - load:    Regridder(weights=...) from the same weights saved to disk, as AQUA does (s)
    - build:   the regrid() call, i.e. the dask graph construction (ms)
    - tasks, graph MB: size of the graph of data read lazily from disk (pickled with cloudpickle)
    - compute: persisting the regridded output on each scheduler (ms). Regridding only: the input
               is loaded in memory (or on the workers) beforehand, so disk I/O is excluded, and the
               output is left where it is computed
    - memory:  peak memory allocated by one compute (tracemalloc, numpy arrays included), traced
               after the input is loaded. On a cluster: the client process, which also runs the
               scheduler, plus the sum of the peaks of the workers, i.e. an upper bound of the total
Times are medians of --repeat calls after a warmup. Schedulers: serial (synchronous scheduler)
and LocalClusters of N worker processes with one thread each ('threads' can be added).

Every case has about --data-mb of data (the larger of input and output), the test file repeated
along time and written once to --cache-dir, in chunks of about --chunk-mb along time.

Versions: the candidate (default WORKTREE, i.e. the current code, uncommitted changes included)
and optionally a reference (--against). Each version is exported and runs in its own process.
The report shows the values of the candidate and, with --against, their change vs the reference
when beyond the noise (≈ otherwise). Correctness: the outputs of the two versions are compared on
the original test files, and the largest cluster is checked against the serial output.
With --fail-on-regression the exit code is 1 on regressions of the deterministic metrics only
(tasks, graph size, serial memory, correctness, new errors): timings are too noisy, e.g. in CI.

Usage (from the repository root, CDO is required as for the tests):
    python benchmarks/benchmark.py                   # current code only
    python benchmarks/benchmark.py --against main    # main vs current code
    python benchmarks/benchmark.py --against HEAD    # last commit vs uncommitted changes
    python benchmarks/benchmark.py --quick --cases "3D mask" "3D mask skipna"
    python benchmarks/benchmark.py --against origin/main --quick --fail-on-regression   # as in CI
"""

import argparse
import json
import logging
import math
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import warnings

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(REPO, 'tests', 'data')

# name: (file, variable, skipna, target grid as CDO name, vertical dimension)
CASES = {
    '2D': ('tas-ecearth.nc', 'tas', False, 'r360x180', None),
    '2D mask': ('onlytos-ipsl.nc', 'tos', False, 'r360x180', None),
    '2D mask skipna': ('onlytos-ipsl.nc', 'tos', True, 'r360x180', None),
    '2D varying mask skipna': ('berkley.nc', 'temperature', True, 'r360x180', None),
    '3D mask': ('so3d-nemo.nc', 'so', False, 'r360x180', 'lev'),
    '3D mask skipna': ('so3d-nemo.nc', 'so', True, 'r360x180', 'lev'),
    '2D mask 0.25deg': ('onlytos-ipsl.nc', 'tos', False, 'r1440x720', None),
}

# schedulers running in the process itself, the others are LocalClusters with N workers
IN_PROCESS = {'serial': 'synchronous', 'threads': 'threads'}

DEFAULTS = {'data_mb': 512, 'chunk_mb': 64, 'repeat': 5, 'schedulers': ['serial', '1', '2', '4']}
QUICK = {'data_mb': 128, 'chunk_mb': 32, 'repeat': 3, 'schedulers': ['serial', '1', '2']}

# relative changes vs the reference below these are shown as ≈. From A/A runs (same code twice):
# times shift by ~6% between the two runs, memory on a cluster changes by up to ~20% with the
# placement of the tasks on the workers, while serial memory is stable
NOISE = {'time': 0.10, 'single': 0.10, 'mem': 0.05, 'mem_cluster': 0.25, 'graph_mb': 0.01}
# absolute changes below these are shown as ≈ too, as tiny values give meaningless percentages
FLOOR = {'init': 0.05, 'load': 0.05, 'build': 1., 'time': 10.}
# relative increases of the deterministic metrics failing --fail-on-regression
FAIL = {'tasks': 0.10, 'graph_mb': 0.10, 'mem': 0.25}

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


def _slug(case):
    """Case name usable in file names"""
    return re.sub(r'[^A-Za-z0-9]+', '_', case).strip('_')


# --------------------------------------------------------------------------------------
# Data: test files repeated along time, shared by all the versions
# --------------------------------------------------------------------------------------

def plan_case(xr, case, data_mb, chunk_mb, cache_dir):
    """Path of the data of a case, written if missing, and the number of time steps per chunk.
    Sized on the larger of input and output per time step, so that a fine target grid (whose
    output is much larger than the input) does not blow up the memory. Repeating the test file
    lazily with xr.concat instead would add tasks to every graph, which would then dominate the
    timings on a cluster: reading a longer file is closer to a real use case"""
    fname, var, _, target, vertical = CASES[case]
    src = os.path.join(DATA, fname)
    with xr.open_dataset(src) as ds:
        in_step = ds[var].isel(time=0).nbytes
        nlon, nlat = map(int, re.fullmatch(r'r(\d+)x(\d+)', target).groups())
        # the output is float64, as the weights
        out_step = nlon * nlat * 8 * (ds.sizes[vertical] if vertical else 1)
        ntime = ds.sizes['time']
    step = max(in_step, out_step)
    nsteps = max(1, round(data_mb * 2**20 / step))
    per_chunk = max(1, int(chunk_mb * 2**20 // step))
    path = os.path.join(cache_dir, f'{fname[:-3]}__{nsteps}steps.nc')
    if not os.path.exists(path):
        log.info('Writing %s', path)
        with xr.open_dataset(src, chunks={}) as ds:
            # only the variables with time are repeated (e.g. not the grid bounds)
            tiled = xr.concat([ds] * math.ceil(nsteps / ntime), dim='time',
                              data_vars='minimal', coords='minimal', compat='override')
            tmp = f'{path}.{os.getpid()}.tmp'
            tiled.isel(time=slice(0, nsteps)).to_netcdf(tmp)
        os.replace(tmp, path)
    return path, per_chunk


def _open(xr, path, var, per_chunk):
    """Open the data lazily, chunked along time only: the horizontal dimensions must be whole
    for the regridding (a split would force a rechunk), vertical levels are handled by smmregrid"""
    with xr.open_dataset(path) as ds:
        dims = ds[var].dims
    chunks = {dim: (per_chunk if dim == 'time' else -1) for dim in dims}
    return xr.open_dataset(path, chunks=chunks)[[var]]


# --------------------------------------------------------------------------------------
# Worker: runs inside the process of a single version
# --------------------------------------------------------------------------------------

def _stats(times):
    """Median in ms and relative interquartile range of a list of times in seconds"""
    median = statistics.median(times)
    if len(times) < 2 or median == 0:
        return 1000 * median, 0.
    q1, _, q3 = statistics.quantiles(times, n=4)
    return 1000 * median, (q3 - q1) / median


def _time_persist(persist, repeat):
    """Times in seconds of persisting the output `repeat` times, after a warmup.
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
    return times


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


def _check(result, reference):
    """'ok' if the two outputs are close, a short description of the difference otherwise"""
    import xarray as xr
    try:
        xr.testing.assert_allclose(result, reference)
        return 'ok'
    except AssertionError as err:
        return 'DIFFERENT: ' + ' '.join(str(err).split())[:200]


def _setup_case(tag, case, args, emit):
    """Init the regridder of a case (from grids and from weights), build its graph and compute the
    reference output on the original test file. Return what the compute measurements need"""
    import cloudpickle
    import dask
    import xarray as xr
    from smmregrid import Regridder

    fname, var, skipna, target, _ = CASES[case]
    kwargs = {'skipna': True} if skipna else {}
    src = os.path.join(DATA, fname)

    t0 = time.perf_counter()
    regridder = Regridder(source_grid=src, target_grid=target, loglevel='ERROR', **kwargs)
    init = time.perf_counter() - t0

    # init from the same weights saved to disk, as AQUA does
    load = float('nan')
    try:
        wfile = os.path.join(args.outdir, f'{tag}__{_slug(case)}__weights.nc')
        regridder.grids[0].weights.to_netcdf(wfile)
        t0 = time.perf_counter()
        Regridder(weights=xr.open_dataset(wfile), loglevel='ERROR', **kwargs)
        load = time.perf_counter() - t0
    except Exception as err:  # keep going: the other measurements do not need it
        log.error('%s: init from weights failed: %r', case, err)
        emit(case=case, kind='error', error=f'init from weights: {repr(err)[:250]}')

    path, per_chunk = plan_case(xr, case, args.data_mb, args.chunk_mb, args.cache_dir)
    data = _open(xr, path, var, per_chunk)
    times = []
    for _ in range(args.repeat):
        t0 = time.perf_counter()
        out = regridder.regrid(data)
        times.append(time.perf_counter() - t0)
    build, build_iqr = _stats(times)
    graph = dict(out.__dask_graph__())

    # reference output on the original test file, for the correctness checks
    small = regridder.regrid(xr.open_dataset(src, chunks={})[[var]])
    small_out = dask.compute(small, scheduler='synchronous')[0]
    small_out.to_netcdf(os.path.join(args.outdir, f'{tag}__{_slug(case)}.nc'))

    emit(case=case, kind='case', init=init, load=load, build=build, build_iqr=build_iqr,
         tasks=len(graph), graph_mb=len(cloudpickle.dumps(graph)) / 2**20, chunks=data[var].data.npartitions,
         in_mb=data[var].nbytes / 2**20, out_mb=out[var].nbytes / 2**20)
    log.info('%-24s init %5.1fs  load %5.2fs  build %6.1f ms  %5d tasks', case, init, load, build, len(graph))
    return regridder, data, small, small_out


def worker(tag, args):
    """Run all the selected cases on all the schedulers, for the smmregrid version in PYTHONPATH.
    Results are printed as JSON lines on stdout"""
    import inspect
    import dask
    from smmregrid import Regridder
    warnings.filterwarnings('ignore')

    def emit(**row):
        print(json.dumps({'tag': tag, **row}), flush=True)

    # older versions (e.g. main before #70) have no skipna: those cases are not available
    has_skipna = 'skipna' in inspect.signature(Regridder.__init__).parameters
    ready = {}
    for case in args.cases:
        if CASES[case][2] and not has_skipna:
            emit(case=case, kind='na')
            continue
        try:
            ready[case] = _setup_case(tag, case, args, emit)
        except Exception as err:
            log.error('%s failed: %r', case, err)
            emit(case=case, kind='error', error=repr(err)[:300])

    def report(case, scheduler, times, mem):
        median, iqr = _stats(times)
        emit(case=case, kind='run', scheduler=scheduler, time=median, iqr=iqr, mem=mem)
        log.info('%-24s %-10s %8.0f ms  %6.0f MB', case, scheduler, median, mem)

    # serial (and threads): input loaded in memory in this process
    in_process = [s for s in args.schedulers if s in IN_PROCESS]
    for case, (regridder, data, _, _) in ready.items():
        if not in_process:
            break
        try:
            out = regridder.regrid(data.persist(scheduler='synchronous'))
            for scheduler in in_process:
                def run(out=out, scheduler=IN_PROCESS[scheduler]):
                    return dask.persist(out, scheduler=scheduler)
                report(case, scheduler, _time_persist(run, args.repeat), _peak_memory(run))
            del out
        except Exception as err:
            log.error('%s failed in process: %r', case, err)
            emit(case=case, kind='error', error=f'in process: {repr(err)[:250]}')

    counts = sorted(int(s) for s in args.schedulers if s not in IN_PROCESS)
    if counts:
        _clusters(ready, counts, args.repeat, emit, report)


def _clusters(ready, counts, repeat, emit, report):
    """Compute on LocalClusters of 1, 2, ... workers: the input is persisted on the workers, so that
    the graph refers to it instead of carrying it, and the largest cluster is checked against serial"""
    from distributed import Client, LocalCluster, wait

    for count in counts:
        scheduler = f'cluster {count}'
        log.info('Starting a LocalCluster with %d workers', count)
        with LocalCluster(n_workers=count, threads_per_worker=1, processes=True,
                          dashboard_address=None) as cluster, Client(cluster) as client:

            def start():
                _trace_start()
                client.run(_trace_start)

            def peak():
                return _trace_peak() + sum(client.run(_trace_peak).values())

            for case, (regridder, data, small, small_out) in ready.items():
                try:
                    data_cluster = client.persist(data)
                    wait(data_cluster)
                    out = regridder.regrid(data_cluster)

                    def run(out=out):
                        persisted = client.persist(out)
                        wait(persisted)
                        return persisted
                    report(case, scheduler, _time_persist(run, repeat), _peak_memory(run, start, peak))
                    if count == counts[-1]:
                        emit(case=case, kind='check', scheduler=scheduler,
                             check=_check(client.compute(small).result(), small_out))
                    del data_cluster, out
                except Exception as err:
                    log.error('%s failed on %d workers: %r', case, count, err)
                    emit(case=case, kind='error', error=f'{scheduler}: {repr(err)[:250]}')


# --------------------------------------------------------------------------------------
# Driver: exports the versions, runs the workers and writes the report
# --------------------------------------------------------------------------------------

def _git(*args):
    return subprocess.run(['git', '-C', REPO, *args], check=True, capture_output=True, text=True).stdout.strip()


def describe(ref):
    """Short commit of a ref, with +changes for a working tree with uncommitted changes in smmregrid/"""
    if ref != 'WORKTREE':
        return _git('rev-parse', '--short', ref)
    dirty = _git('status', '--porcelain', '--', 'smmregrid')
    return _git('rev-parse', '--short', 'HEAD') + ('+changes' if dirty else '')


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


def run_worker(tag, srcdir, workdir, args):
    """Run the worker for one version in a separate process, return parsed results"""
    # PYTHONPATH is inherited also by the processes of the dask cluster
    env = dict(os.environ, PYTHONPATH=srcdir)
    cmd = [sys.executable, os.path.abspath(__file__), '--worker', tag,
           '--outdir', args.outdir, '--cache-dir', args.cache_dir, '--loglevel', args.loglevel,
           '--data-mb', str(args.data_mb), '--chunk-mb', str(args.chunk_mb), '--repeat', str(args.repeat),
           '--schedulers', *args.schedulers, '--cases', *args.cases]
    log.debug('Running %s', ' '.join(cmd))
    # cwd outside the repository, so that the installed/editable smmregrid is not picked up.
    # stdout carries the results, stderr (worker logs) goes straight to the terminal
    proc = subprocess.run(cmd, env=env, cwd=workdir, stdout=subprocess.PIPE, text=True)
    results = [json.loads(line) for line in proc.stdout.splitlines() if line.startswith('{')]
    if proc.returncode != 0:
        log.error('Worker for %s exited with code %s', tag, proc.returncode)
        results.append({'tag': tag, 'case': '-', 'kind': 'error', 'error': f'worker exited with code {proc.returncode}'})
    return results


def environment():
    """One line describing the machine and the main packages"""
    from importlib.metadata import PackageNotFoundError, version

    def pkg(name):
        try:
            return version(name)
        except PackageNotFoundError:
            return '?'
    cpu = platform.processor() or platform.machine()
    try:
        if sys.platform == 'darwin':
            cpu = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], capture_output=True, text=True).stdout.strip()
        elif os.path.exists('/proc/cpuinfo'):
            with open('/proc/cpuinfo', encoding='utf-8') as f:
                cpu = next(line.split(':', 1)[1].strip() for line in f if line.startswith('model name'))
    except (OSError, StopIteration):
        pass
    try:
        import psutil
        memory = f'{psutil.virtual_memory().total / 2**30:.0f} GB'
    except ImportError:
        memory = '? GB'
    try:
        out = subprocess.run(['cdo', '--version'], capture_output=True, text=True)
        cdo = re.search(r'version (\S+)', out.stdout + out.stderr).group(1)
    except (OSError, AttributeError):
        cdo = '?'
    return (f'{cpu} · {os.cpu_count()} cores · {memory} · python {platform.python_version()} · '
            f'dask {pkg("dask")} · distributed {pkg("distributed")} · numpy {pkg("numpy")} · '
            f'xarray {pkg("xarray")} · sparse {pkg("sparse")} · CDO {cdo}')


def _value(rows, tag, case, kind, key, scheduler=None):
    """Median across rounds of a measurement, None if missing"""
    values = [r[key] for r in rows if r['tag'] == tag and r['case'] == case and r.get('kind') == kind
              and r.get('scheduler') == scheduler and r.get(key) is not None and r[key] == r[key]]
    return statistics.median(values) if values else None


def _change(new, ref, noise, floor=0.):
    """Change of new vs ref: ≈ within the relative noise or the absolute floor (noise=0: = only when equal),
    signed percentage otherwise"""
    if ref is None or new is None:
        return ''
    if new == ref:
        return ' (=)' if noise == 0 else ' (≈)'
    if ref == 0:
        return ' (new)'
    rel = new / ref - 1
    if abs(rel) <= noise or abs(new - ref) <= floor:
        return ' (≈)'
    return f' ({100 * rel:+.1f}%)' if abs(rel) < 0.01 else f' ({100 * rel:+.0f}%)'


def _table(header, lines):
    """Markdown table, the first column left aligned and the others right aligned"""
    out = ['| ' + ' | '.join(header) + ' |', '|:---|' + '---:|' * (len(header) - 1)]
    out += ['| ' + ' | '.join(line) + ' |' for line in lines]
    return out


def _compare_outputs(ref_file, new_file):
    """identical / close (values close, metadata differ) / DIFFERENT / missing"""
    import xarray as xr
    if not (os.path.exists(ref_file) and os.path.exists(new_file)):
        return 'missing'
    with xr.open_dataset(ref_file) as a, xr.open_dataset(new_file) as b:
        try:
            xr.testing.assert_identical(a, b)
            return 'identical'
        except AssertionError:
            try:
                xr.testing.assert_allclose(a, b)
                return 'close'
            except AssertionError as err:
                return 'DIFFERENT: ' + ' '.join(str(err).split())[:150]


def make_report(rows, tags, labels, args):
    """Markdown report, and the list of regressions of the deterministic metrics"""
    warnings.filterwarnings('ignore')
    ref, new = (tags[0], tags[1]) if len(tags) == 2 else (None, tags[0])
    schedulers = [s if s in IN_PROCESS else f'cluster {s}' for s in args.schedulers]

    def status(tag, case):
        """'n/a', 'error' or None if the case ran (for the setup of the case)"""
        if any(r['tag'] == tag and r['case'] == case and r.get('kind') == 'na' for r in rows):
            return 'n/a'
        if _value(rows, tag, case, 'case', 'tasks') is None:
            return 'error'
        return None

    def cell(case, kind, key, fmt, noise, scheduler=None, iqr_key=None):
        """Value of the candidate, with its change vs the reference (none if noise is None)"""
        bad = status(new, case)
        new_value = _value(rows, new, case, kind, key, scheduler)
        if bad or new_value is None:
            return bad or 'error'
        text = format(new_value, fmt)
        if ref is None or noise is None or status(ref, case):
            return text
        if iqr_key:  # times: noise from the spread of both measurements
            spread = [_value(rows, t, case, kind, iqr_key, scheduler) or 0 for t in (ref, new)]
            noise = max(noise, 2 * max(spread))
        return text + _change(new_value, _value(rows, ref, case, kind, key, scheduler), noise, FLOOR.get(key, 0.))

    title = f'## smmregrid benchmark: {new} ({labels[new]})'
    if ref:
        title += f' vs {ref} ({labels[ref]})'
    settings = (f'{args.data_mb} MB per case (larger of input and output), chunks of {args.chunk_mb} MB along time · '
                f'median of {args.repeat}' + (f' × {args.rounds} rounds' if args.rounds > 1 else ''))
    if ref:
        settings += ' · change vs reference only beyond noise, ≈ otherwise'
    out = [title, '', environment(), '', settings, '', '### Init and graph', '']
    out += _table(['case', 'init s', 'load s', 'build ms', 'tasks', 'graph MB', 'chunks'], [
        [case,
         cell(case, 'case', 'init', '.1f', NOISE['single']),
         cell(case, 'case', 'load', '.2f', NOISE['single']),
         cell(case, 'case', 'build', '.1f', NOISE['time'], iqr_key='build_iqr'),
         cell(case, 'case', 'tasks', 'd', 0),
         cell(case, 'case', 'graph_mb', '.1f', NOISE['graph_mb']),
         cell(case, 'case', 'chunks', 'd', None)] for case in args.cases])
    out += ['', '### Compute time, ms (regridding only, input in memory)', '']
    out += _table(['case'] + schedulers, [
        [case] + [cell(case, 'run', 'time', '.0f', NOISE['time'], s, iqr_key='iqr') for s in schedulers]
        for case in args.cases])
    out += ['', '### Peak memory, MB (all processes)', '']
    out += _table(['case'] + schedulers, [
        [case] + [cell(case, 'run', 'mem', '.0f', NOISE['mem' if s in IN_PROCESS else 'mem_cluster'], s)
                  for s in schedulers] for case in args.cases])

    # correctness and errors, as single lines unless something is wrong
    problems, regressions = [], []
    checks = {r['case']: r['check'] for r in rows if r['tag'] == new and r.get('kind') == 'check'}
    if checks:
        good = sum(c == 'ok' for c in checks.values())
        problems += [f'- {case}: {schedulers[-1]} vs serial {c}' for case, c in checks.items() if c != 'ok']
        regressions += [f'{case}: {schedulers[-1]} differs from serial' for case, c in checks.items() if c != 'ok']
        line = f'Correctness: {schedulers[-1]} = serial {good}/{len(checks)}'
    else:
        line = 'Correctness: no cluster check'
    if ref:
        outputs = {case: _compare_outputs(os.path.join(args.outdir, f'{ref}__{_slug(case)}.nc'),
                                          os.path.join(args.outdir, f'{new}__{_slug(case)}.nc'))
                   for case in args.cases if not status(ref, case) and not status(new, case)}
        same = sum(o in ('identical', 'close') for o in outputs.values())
        line += f' · outputs equal to {ref} {same}/{len(outputs)}'
        problems += [f'- {case}: output vs {ref} {o}' for case, o in outputs.items() if o not in ('identical', 'close')]
        regressions += [f'{case}: output differs from {ref}' for case, o in outputs.items() if o.startswith('DIFFERENT')]
        na = [case for case in args.cases if status(ref, case) == 'n/a']
        if na:
            line += f' · not available in {ref}: {", ".join(na)}'
    out += ['', line]

    errors = [r for r in rows if r.get('kind') == 'error']
    out += ['Errors: ' + ('none' if not errors else str(len(errors)))]
    out += [f'- {r["tag"]}, {r["case"]}: `{r["error"][:150]}`' for r in errors] + problems

    # deterministic regressions: new errors, graph size and serial memory
    ref_errors = {r['case'] for r in errors if r['tag'] == ref}
    regressions += [f'{r["case"]}: error' for r in errors if r['tag'] == new and r['case'] not in ref_errors]
    if ref:
        for case in args.cases:
            if status(ref, case) or status(new, case):
                continue
            pairs = [('tasks', 'case', 'tasks', None), ('graph_mb', 'case', 'graph MB', None)]
            pairs += [('mem', 'run', 'serial memory', 'serial')] if 'serial' in schedulers else []
            for key, kind, name, scheduler in pairs:
                old, value = (_value(rows, t, case, kind, key, scheduler) for t in (ref, new))
                if old and value and value > old * (1 + FAIL[key]):
                    regressions.append(f'{case}: {name} {old:.0f} → {value:.0f}')
    if args.fail_on_regression:
        out += ['', 'Regressions: ' + ('none' if not regressions else ''), *[f'- {r}' for r in regressions]]
    return '\n'.join(out) + '\n', regressions


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--against', default=None, metavar='REF',
                        help='git ref to compare with, e.g. main or HEAD (default: none, candidate only)')
    parser.add_argument('--candidate', default='WORKTREE', metavar='REF',
                        help='git ref to measure, WORKTREE being the current code (default: WORKTREE)')
    parser.add_argument('--cases', nargs='+', default=list(CASES), choices=list(CASES), metavar='CASE',
                        help='subset of cases (default: all): ' + ', '.join(f'"{c}"' for c in CASES))
    parser.add_argument('--schedulers', nargs='+', default=None, metavar='S',
                        help='serial, threads and/or numbers of cluster workers (default: serial 1 2 4)')
    parser.add_argument('--data-mb', type=int, default=None, help='data per case, in MB (default: 512)')
    parser.add_argument('--chunk-mb', type=int, default=None, help='size of the chunks along time, in MB (default: 64)')
    parser.add_argument('--repeat', type=int, default=None, help='timed calls per measurement (default: 5)')
    parser.add_argument('--rounds', type=int, default=1, help='runs of each version, alternated (default: 1)')
    parser.add_argument('--quick', action='store_true',
                        help='smaller and faster defaults, e.g. for CI: --data-mb 128 --chunk-mb 32 '
                             '--repeat 3 --schedulers serial 1 2')
    parser.add_argument('--output', default=None, help='also write the report to this file')
    parser.add_argument('--keep', default=None, metavar='DIR',
                        help='keep raw results (results.jsonl) and outputs in DIR (default: temporary)')
    parser.add_argument('--cache-dir', default=os.path.join(os.path.expanduser('~'), '.cache', 'smmregrid-benchmark'),
                        help='where the data repeated along time are written once (default: %(default)s)')
    parser.add_argument('--fail-on-regression', action='store_true',
                        help='exit with 1 on regressions of tasks, graph size, serial memory, correctness or errors')
    parser.add_argument('--loglevel', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], type=str.upper,
                        help='logging level, logs go to stderr (default: INFO)')
    parser.add_argument('--worker', default=None, help=argparse.SUPPRESS)
    parser.add_argument('--outdir', default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    for key, value in (QUICK if args.quick else DEFAULTS).items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    for scheduler in args.schedulers:
        if scheduler not in IN_PROCESS and not (scheduler.isdigit() and int(scheduler) > 0):
            parser.error(f'invalid scheduler {scheduler}: use serial, threads or a number of workers')

    setup_logging(args.loglevel, args.worker or 'driver')
    if args.worker:
        worker(args.worker, args)
        return

    refs = [args.against, args.candidate] if args.against else [args.candidate]
    # refs as safe tags, usable in file names (e.g. origin/main -> origin_main)
    tags = [ref.replace('/', '_') for ref in refs]
    if len(set(tags)) != len(tags):
        parser.error('--against and --candidate are the same')
    labels = {tag: describe(ref) for ref, tag in zip(refs, tags)}

    tmp = tempfile.mkdtemp(prefix='smmregrid-bench-')
    args.outdir = os.path.abspath(args.keep) if args.keep else os.path.join(tmp, 'out')
    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)
    log.info('Benchmarking %s on %d cases, schedulers %s', ' vs '.join(refs), len(args.cases), ' '.join(args.schedulers))
    tstart = time.perf_counter()
    try:
        # data written once here, before the versions run
        import xarray as xr
        for case in args.cases:
            plan_case(xr, case, args.data_mb, args.chunk_mb, args.cache_dir)

        srcdirs = {}
        for ref, tag in zip(refs, tags):
            srcdirs[tag] = os.path.join(tmp, 'src', tag)
            export_version(ref, srcdirs[tag])

        rows = []
        for rnd in range(args.rounds):
            for tag in tags:  # alternate versions to spread out machine noise
                log.info('Round %d/%d: %s', rnd + 1, args.rounds, tag)
                rows += run_worker(tag, srcdirs[tag], tmp, args)
        log.info('Done in %.0fs', time.perf_counter() - tstart)

        with open(os.path.join(args.outdir, 'results.jsonl'), 'w', encoding='utf-8') as f:
            f.writelines(json.dumps(r) + '\n' for r in rows)

        report, regressions = make_report(rows, tags, labels, args)
        print(report)
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(report)
        if args.keep:
            log.info('Raw results and outputs in %s', args.outdir)
    finally:
        shutil.rmtree(os.path.join(tmp, 'src'), ignore_errors=True)
        if not args.keep:
            shutil.rmtree(tmp, ignore_errors=True)

    if args.fail_on_regression and regressions:
        sys.exit(1)


if __name__ == '__main__':
    main()
