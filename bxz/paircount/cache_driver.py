"""Bounded, restartable DR1-cache angular counts (P2.3/P2.4).

Each call is one reference tracer/cap and one half-open fine-z slice. Only one
reference storage cell and one photometric neighbour are resident at a time.
Atomic files group neighbour rows by reference cell; zero-neighbour photometric
cells retain their full-population moments for resampling.
This module never allocates compute or changes the selected populations.
"""
from dataclasses import replace
import hashlib
from importlib.metadata import version
import json
from numbers import Integral
import os
from pathlib import Path
import re
import tempfile
import time
from zipfile import ZipFile, BadZipFile

import numpy as np

from bxz.counts_io import CountProductError, array_digest, canonical_json, publish_manifest
from .angular import AngularCatalog, PatchCounts, count_angular
from .driver import CountRun, _software

MANIFEST_BYTES = 32 * 1024**2
SCHEMA = 'bxz.cache-angular-manifest.v1'


class CacheCountIncomplete(RuntimeError):
    """An explicit new-block budget ended; directory is safe to resume."""
    completed = False

    def __init__(self, directory, computed, reused, total_blocks, reason='new-block budget reached'):
        self.directory = Path(directory)
        self.computed = computed
        self.reused = reused
        self.total_blocks = total_blocks
        self.reason = reason
        super().__init__(f'{reason}: {computed} computed, {reused} reused; '
                         f'incomplete run at {self.directory}')


ARRAY_KEYS = {'edges', 'patch_pairs', 'npairs', 'weighted', 'weighted_z', 'weighted_z2',
              'left_moments', 'seconds'}


def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f'{name} must be a positive integer')
    return int(value)


def _manifest(path):
    # Read only a named, bounded metadata file, never discover input paths.
    from bxz.cache_io import _read_manifest
    try:
        return _read_manifest(path, max_bytes=MANIFEST_BYTES)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise CountProductError('unreadable bounded cache/count manifest') from exc


def _group_cells(group, catalogue, dtype):
    if group['dtype'] != [list(item) for item in dtype.descr]:
        raise ValueError('unexpected cache numeric dtype')
    cells = sorted(int(cell) for cell in group['cells'])
    if (len(cells) != len(set(cells)) or any(not 0 <= cell < 3072 for cell in cells)
            or set(group['cells']) != {str(cell) for cell in cells}):
        raise ValueError('unique canonical NESTED16 storage cells required')
    moments = []
    for cell in cells:
        item = group['cells'][str(cell)]
        if (type(item['count']) is not int or item['count'] <= 0
                or not isinstance(item['sha256'], str)
                or re.fullmatch(r'[0-9a-f]{64}', item['sha256']) is None):
            raise ValueError('invalid cached cell identity/count')
        if (item['file'] != f'{catalogue}/{cell:04d}.bin'
                or item['bytes'] != item['count']*dtype.itemsize):
            raise ValueError('unexpected cache shard path or size')
        row = [item['count'], item['sum_w'], item['sum_w2']]
        if not np.isfinite(row).all() or min(row) <= 0:
            raise ValueError('invalid cached cell moments')
        moments.append(row)
    return cells, np.asarray(moments, dtype=float).reshape(-1, 3)


def _load_cell(manifest_path, catalogue, group, cell, dtype, *, max_bytes):
    """Read one prevalidated snapshot shard without reparsing its manifest.

    This retains the immutable full-cache loader's payload checks. Only the
    explicit cell is opened; the caller checks snapshot-manifest fstat as well.
    """
    item = group['cells'][str(cell)]
    if item['bytes'] + 8*item['count'] > min(max_bytes, 8*1024**3):
        raise ValueError('selected cell exceeds explicit memory byte bound')
    values = np.empty(item['count'], dtype=dtype)
    labels = np.full(item['count'], cell, dtype=np.int64)
    digest = hashlib.sha256()
    with open(Path(manifest_path).parent/item['file'], 'rb') as stream:
        before = os.fstat(stream.fileno())
        if before.st_size != item['bytes']:
            raise CountProductError('cache shard byte size changed')
        target = memoryview(values).cast('B')
        for start in range(0, len(target), 1024**2):
            segment = target[start:start+1024**2]
            if stream.readinto(segment) != len(segment):
                raise CountProductError('cache shard truncated')
            digest.update(segment)
        after = os.fstat(stream.fileno())
    if ((before.st_size, before.st_mtime_ns, before.st_ctime_ns) !=
            (after.st_size, after.st_mtime_ns, after.st_ctime_ns)
            or digest.hexdigest() != item['sha256']):
        raise CountProductError('cache shard changed or checksum mismatch')
    return values, labels


def candidate_cells(left_cells, right_cells, theta_max, *, max_tasks):
    """Conservative spherical cell pruning, including wrap and polar neighbours.

    Triangle inequality: a pair within theta_max has centre separation at most
    theta_max + 2*max_pixrad(16). The small outward roundoff allowance only adds
    candidates; exact backend edges still decide whether an object pair counts.
    Only one vector of <=3072 centre distances is formed at a time.
    """
    import healpy as hp
    max_tasks = _positive_int(max_tasks, 'max_tasks')
    for values in (left_cells, right_cells):
        if (len(values) != len(set(values)) or any(isinstance(v, bool)
                or not isinstance(v, Integral) or not 0 <= v < 3072 for v in values)):
            raise ValueError('unique NESTED16 cell labels required')
    if not np.isfinite(theta_max) or not 0 < theta_max <= np.pi:
        raise ValueError('theta_max must be in (0, pi]')
    right_cells = np.asarray(sorted(right_cells), dtype=np.int64)
    vectors = np.asarray(hp.pix2vec(16, right_cells, nest=True)).T
    radius = min(np.pi, theta_max + 2*hp.max_pixrad(16) + 64*np.finfo(float).eps)
    schedule, tasks = {}, 0
    for cell in sorted(left_cells):
        center = np.asarray(hp.pix2vec(16, int(cell), nest=True))
        angle = np.arctan2(np.linalg.norm(np.cross(vectors, center), axis=1), vectors @ center)
        neighbours = right_cells[angle <= radius].tolist()
        tasks += len(neighbours)
        if tasks > max_tasks:
            raise ValueError('candidate pairs exceed explicit max_tasks bound')
        schedule[str(int(cell))] = neighbours
    return schedule


def _request(edges, slice_id, count_type, provenance, context, cache):
    edges = np.asarray(edges, dtype=float)
    if (edges.ndim != 1 or len(edges) < 2 or not np.isfinite(edges).all()
            or edges[0] <= 0 or edges[-1] > np.pi or np.any(np.diff(edges) <= 0)):
        raise ValueError('invalid angular edges')
    if count_type != 'DD' or not isinstance(slice_id, str) or not slice_id.strip():
        raise ValueError('one nonempty slice ID and DD count type required')
    required = {'config_hash', 'config', 'git_commit', 'population_id', 'catalogues', 'input_checksums'}
    if not isinstance(provenance, dict) or not required.issubset(provenance):
        raise ValueError('complete provenance required')
    provenance = json.loads(canonical_json(provenance))
    if (not isinstance(provenance['config'], dict) or not provenance['config']
            or not isinstance(provenance['population_id'], str) or not provenance['population_id'].strip()
            or not isinstance(provenance['catalogues'], dict) or not provenance['catalogues']
            or not isinstance(provenance['input_checksums'], dict) or not provenance['input_checksums']
            or not isinstance(provenance['git_commit'], str)
            or re.fullmatch(r'[0-9a-fA-F]{40}|[0-9a-fA-F]{64}', provenance['git_commit']) is None
            or any(not isinstance(v, str) or re.fullmatch(r'[0-9a-fA-F]{64}', v) is None
                   for v in provenance['input_checksums'].values())):
        raise ValueError('invalid provenance')
    config = json.dumps(provenance['config'], sort_keys=True, separators=(',', ':'),
                        ensure_ascii=False, allow_nan=False)
    if hashlib.sha256(config.encode()).hexdigest() != provenance['config_hash']:
        raise ValueError('configuration snapshot does not match config_hash')
    if not isinstance(context, dict) or not {
            'slice_edges', 'chi_hmpc', 'hard_cut_indices', 'patch_scheme', 'mask'}.issubset(context):
        raise ValueError('complete measurement_context required')
    context = json.loads(canonical_json(context))
    bounds, cuts = np.asarray(context['slice_edges'], dtype=float), context['hard_cut_indices']
    if (bounds.shape != (2,) or not np.isfinite(bounds).all() or bounds[0] < .05
            or bounds[1] <= bounds[0] or not isinstance(cuts, list) or len(cuts) != 2
            or any(type(v) is not int for v in cuts) or not 0 <= cuts[0] < cuts[1] < len(edges)
            or isinstance(context['chi_hmpc'], (str, bool)) or not np.isscalar(context['chi_hmpc'])
            or not np.isfinite(context['chi_hmpc']) or context['chi_hmpc'] <= 0
            or context['patch_scheme'] != cache['storage_scheme']
            or not isinstance(context['mask'], dict)
            or context['mask'].get('sha256') != cache['mask_sha256']):
        raise ValueError('invalid or cache-inconsistent measurement_context')
    return edges.copy(), provenance, context


def _result_bytes(schedule, bins, has_z, nright):
    pairs = sum(len(qs) for qs in schedule.values())
    largest = max((len(qs) for qs in schedule.values()), default=0)
    row_bytes = 16 + bins * (32 if has_z else 16) + 8
    # Final arrays plus one loaded grouped file, its digest copy and zip buffers.
    return 2 * (pairs + largest) * row_bytes + 48*(len(schedule)+nright) + 65536


def _block_metadata(identity, run_hash, cell):
    return {**identity, 'schema': 'bxz.cache-angular-block.v1', 'run_hash': run_hash,
            'left_cell': cell}


def _validate_block(arrays, metadata, edges, neighbours):
    if set(arrays) != ARRAY_KEYS or metadata.get('schema') != 'bxz.cache-angular-block.v1':
        raise CountProductError('unsupported grouped angular block')
    pairs = np.array([(metadata['left_cell'], q) for q in neighbours], dtype=np.int64).reshape(-1, 2)
    if (not np.array_equal(arrays['edges'], edges)
            or arrays['patch_pairs'].dtype.kind not in 'iu'
            or not np.array_equal(arrays['patch_pairs'], pairs)):
        raise CountProductError('grouped block grid or scheduled pairs differ')
    shape = (len(pairs), len(edges)-1)
    for name in ('npairs', 'weighted', 'weighted_z', 'weighted_z2'):
        value = arrays[name]
        expected = shape if name in ('npairs', 'weighted') or metadata['has_redshift_moments'] else (0,)
        if value.shape != expected or not np.isfinite(value).all() or np.any(value < 0):
            raise CountProductError(f'invalid grouped {name}')
    if arrays['npairs'].dtype.kind not in 'iu' or np.any(arrays['npairs'] > np.iinfo(np.int64).max):
        raise CountProductError('grouped npairs must be int64-compatible integers')
    if np.any((arrays['npairs'] == 0) != (arrays['weighted'] == 0)):
        raise CountProductError('grouped weighted and unweighted support differ')
    for key, expected in (('left_moments', (3,)), ('seconds', (len(pairs),))):
        value = arrays[key]
        if value.shape != expected or not np.isfinite(value).all() or np.any(value < 0):
            raise CountProductError(f'invalid grouped {key}')
    n, w, q = arrays['left_moments']
    if n != int(n) or (n == 0) != (w == 0) or (n == 0) != (q == 0):
        raise CountProductError('invalid grouped reference moments')


def _read_block(path, metadata, edges, neighbours, max_bytes):
    try:
        # NPZ headers and payload are bounded before np.load can allocate arrays.
        with ZipFile(path) as zipped:
            if sum(item.file_size for item in zipped.infolist()) > max_bytes:
                raise CountProductError('grouped block exceeds explicit byte bound')
        with np.load(path, allow_pickle=False) as saved:
            actual = json.loads(str(saved['metadata_json'].item()))
            digest = str(saved['sha256'].item())
            arrays = {name: saved[name] for name in saved.files if name not in ('metadata_json', 'sha256')}
        if canonical_json(actual) != canonical_json(metadata):
            raise CountProductError('grouped block task metadata mismatch')
        _validate_block(arrays, actual, edges, neighbours)
        if _block_digest(arrays, actual) != digest:
            raise CountProductError('grouped block checksum mismatch')
    except CountProductError:
        raise
    except (OSError, ValueError, KeyError, TypeError, EOFError, BadZipFile) as exc:
        raise CountProductError(f'unreadable grouped count block: {path}') from exc
    return arrays


def _block_digest(arrays, metadata):
    # Flatten empty 2-D arrays for memoryview while retaining every shape.
    return array_digest({k: v.reshape(-1) for k, v in arrays.items()},
                        dict(metadata=metadata, shapes={k: list(v.shape) for k, v in arrays.items()}))


def _publish_block(path, arrays, metadata):
    digest = _block_digest(arrays, metadata)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.partial-', suffix='.npz', delete=False) as stream:
            temporary = Path(stream.name)
            np.savez(stream, **arrays, metadata_json=canonical_json(metadata), sha256=digest)
            stream.flush()
            os.fsync(stream.fileno())
        # A single worker owns a run. Conflicting writers fail, never clobber.
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def run_cache_angular_counts(manifest_path, reference_group, photo_group, edges, output_dir, *,
                             slice_id, count_type, provenance, measurement_context,
                             max_tasks, max_memory_bytes, max_result_bytes, backend='pycorr',
                             max_new_blocks=None, block_cells=None):
    """Count/resume a bounded fine slice from immutable full-cache cell shards.

    max_new_blocks optionally limits newly computed left-cell blocks. Exhausting
    it raises CacheCountIncomplete before the next missing block is read; the
    same call resumes later. block_cells optionally schedules explicit left
    cells (sorted, unique) without changing run identity or population; finishing
    a subset also raises CacheCountIncomplete until every original block exists.
    A CountRun is returned only after all blocks exist.
    Limits are explicit admission checks, not compute authorization. The working
    array estimate reserves 256 bytes per resident source object for raw shards,
    labels, reference filtering, adapter copies and coordinate/weight temporaries,
    plus four maximum-sized JSON buffers and result buffers. This deliberately
    overcounts known arrays; native allocator/runtime overhead is not a hard RSS
    guarantee and must fit the separately approved compute allocation.

    Cache metadata is parsed once and all named shard metadata is checked before
    counting. No full photometric catalogue is loaded. Unread distant shard digests still
    enter input identity; only candidate shard payloads are read/verified. The
    stored full-domain moments were checked against the audit at cache creation.
    Changing any cache manifest, group, fine slice, edges, source code or supplied
    provenance selects a different output directory. Already published grouped
    files resume without input reads or recounting; corrupt products fail.
    """
    max_tasks = _positive_int(max_tasks, 'max_tasks')
    max_memory_bytes = _positive_int(max_memory_bytes, 'max_memory_bytes')
    max_result_bytes = _positive_int(max_result_bytes, 'max_result_bytes')
    if max_new_blocks is not None:
        max_new_blocks = _positive_int(max_new_blocks, 'max_new_blocks')
    if backend != 'pycorr':
        raise ValueError('cache counts require the float64 pycorr backend')
    def manifest_state():
        stat = Path(manifest_path).stat()
        return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns
    cache_state = manifest_state()
    cache, cache_digest = _manifest(manifest_path)
    if manifest_state() != cache_state:
        raise CountProductError('cache manifest changed while reading')
    if (cache.get('schema') != 'bxz.dr1-full-cache.v1' or cache.get('completed') is not True
            or cache['storage_scheme'] != {'nside': 16, 'frame': 'equatorial', 'ordering': 'NESTED'}
            or cache['units'] != {'angles': 'radian', 'redshift': 'dimensionless'}):
        raise ValueError('completed equatorial NESTED16 radian cache required')
    edges, provenance, context = _request(edges, slice_id, count_type, provenance, measurement_context, cache)
    left, right = cache['catalogues'][reference_group], cache['catalogues'][photo_group]
    tracer, cap = left['source'].get('tracer'), left['source'].get('cap')
    if (tracer not in ('BGS_BRIGHT', 'LRG', 'ELG_LOPnotqso', 'QSO') or cap not in ('NGC', 'SGC')
            or left['source'].get('kind') != 'data' or reference_group != f'{tracer}_{cap}_data'
            or photo_group != 'photo_data'
            or 'tracer' in right['source']):
        raise ValueError('DD requires a named DR1 tracer/cap data group and photometric data')
    allowed = cache['reference_z_ranges'][left['source']['tracer']]
    if context['slice_edges'][0] < allowed[0] or context['slice_edges'][1] > allowed[1]:
        raise ValueError('fine slice lies outside cached reference redshift support')
    from bxz.cache_io import REFERENCE_DTYPE, PHOTO_DTYPE
    lp, _ = _group_cells(left, reference_group, REFERENCE_DTYPE)
    rp, rm = _group_cells(right, photo_group, PHOTO_DTYPE)
    selected_cells = lp if block_cells is None else list(block_cells)
    if (not selected_cells and lp or len(selected_cells) != len(set(selected_cells))
            or any(isinstance(p, bool) or not isinstance(p, Integral) or p not in lp for p in selected_cells)):
        raise ValueError('block_cells must be unique existing left-cell integers')
    selected_cells = sorted(int(p) for p in selected_cells)
    schedule = candidate_cells(lp, rp, edges[-1], max_tasks=max_tasks)
    has_z = False
    result_bytes = _result_bytes(schedule, len(edges)-1, has_z, len(rp))
    if result_bytes > max_result_bytes:
        raise ValueError('scheduled result exceeds explicit max_result_bytes bound')
    # Preflight every task before reading payloads or publishing output.
    for p, neighbours in schedule.items():
        nleft = left['cells'][p]['count']
        nright = max((right['cells'][str(q)]['count'] for q in neighbours), default=0)
        if 256*(nleft+nright) + 4*MANIFEST_BYTES + result_bytes > max_memory_bytes:
            raise ValueError('live cell/adapter temporaries exceed explicit max_memory_bytes bound')
    software = _software(backend)
    software['cache_driver.py_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    import bxz.cache_io as loader
    software['cache_io.py_sha256'] = hashlib.sha256(Path(loader.__file__).read_bytes()).hexdigest()
    software['healpy'] = version('healpy')
    def load(group, cell):
        if manifest_state() != cache_state:
            raise CountProductError('cache manifest changed during counting')
        saved_group = left if group == reference_group else right
        dtype = REFERENCE_DTYPE if group == reference_group else PHOTO_DTYPE
        values = _load_cell(manifest_path, group, saved_group, cell, dtype, max_bytes=max_memory_bytes)
        if manifest_state() != cache_state:
            raise CountProductError('cache manifest changed during counting')
        return values
    identity = dict(schema='bxz.cache-angular-run.v1', slice_id=slice_id, count_type=count_type,
        backend=backend, provenance=provenance, measurement_context=context, software=software,
        has_redshift_moments=has_z,
        pair_convention='ordered_cross; [low, high); scalar_positive_weights', units='radian',
        cache_metadata_sha256=cache_digest, reference_group=reference_group, photo_group=photo_group,
        catalogue_hashes={
            'left': array_digest({}, dict(cache=cache_digest, group=reference_group, slice_edges=context['slice_edges'])),
            'right': array_digest({}, dict(cache=cache_digest, group=photo_group))})
    task_plan = dict(left_cells=lp, right_cells=rp, right_moments=rm.tolist(), schedule=schedule)
    identity['task_plan_sha256'] = hashlib.sha256(canonical_json(task_plan).encode()).hexdigest()
    run_hash = array_digest({'edges': edges}, identity)
    root = Path(output_dir) / run_hash
    root.mkdir(parents=True, exist_ok=True)
    manifest = dict(schema=SCHEMA, identity=identity, run_hash=run_hash, edges=edges.tolist(),
        left_cells=lp, right_cells=rp, right_moments=rm.tolist(), schedule=schedule,
        result_bytes=result_bytes, max_result_bytes=max_result_bytes,
        pruning='centre_separation <= theta_max + 2*healpy.max_pixrad(16) + outward_roundoff',
        block_layout='one atomic NPZ per left cell; all scheduled patch-pair rows retained')
    # Budget choices do not change physics, but a resume uses its original bound.
    publish_manifest(root/'run.json', manifest)
    computed = reused = 0
    for cell in selected_cells:
        neighbours = schedule[str(cell)]
        path = root/f'p{cell}.npz'
        metadata = _block_metadata(identity, run_hash, cell)
        if path.exists():
            _read_block(path, metadata, edges, neighbours, max_result_bytes)
            reused += 1
            continue
        if max_new_blocks is not None and computed >= max_new_blocks:
            raise CacheCountIncomplete(root, computed, reused, len(lp))
        values, labels = load(reference_group, cell)
        low, high = context['slice_edges']
        selected = (values['z'] >= low) & (values['z'] < high)
        values, labels = values[selected], labels[selected]
        a = AngularCatalog(values['ra'], values['dec'], values['weight'], labels)
        shape = (len(neighbours), len(edges)-1)
        arrays = dict(edges=edges, patch_pairs=np.array([(cell, q) for q in neighbours], dtype=np.int64).reshape(-1, 2),
            npairs=np.zeros(shape, dtype=np.int64), weighted=np.zeros(shape),
            weighted_z=np.zeros(shape) if has_z else np.empty(0),
            weighted_z2=np.zeros(shape) if has_z else np.empty(0),
            left_moments=np.array([len(values), values['weight'].sum(), np.square(values['weight']).sum()]),
            seconds=np.zeros(len(neighbours)))
        for k, q in enumerate(neighbours):
            tick = time.perf_counter()
            if len(values):
                other, other_labels = load(photo_group, q)
                b = AngularCatalog(other['ra'], other['dec'], other['weight'], other_labels)
                counted = count_angular(a, b, edges, backend=backend)
                arrays['npairs'][k], arrays['weighted'][k] = counted.total_npairs, counted.total_weighted
                del other, other_labels, b, counted
            arrays['seconds'][k] = time.perf_counter()-tick
        _validate_block(arrays, metadata, edges, neighbours)
        _publish_block(path, arrays, metadata)
        computed += 1
        del values, labels, a, arrays
    if manifest_state() != cache_state:
        raise CountProductError('cache manifest changed during counting')
    if not all((root/f'p{p}.npz').is_file() for p in lp):
        raise CacheCountIncomplete(root, computed, reused, len(lp), reason='selected blocks finished')
    return replace(read_cache_angular_run(root, max_result_bytes=max_result_bytes), computed=computed, reused=reused)


def read_cache_angular_run(directory, *, max_result_bytes=None):
    """Rebuild CountRun without cache payloads; missing/invalid blocks fail.

    The optional caller bound may tighten the recorded reconstruction budget.
    All right-cell N/W/Q rows survive, whether or not any pair was scheduled.
    """
    root = Path(directory)
    saved, _ = _manifest(root/'run.json')
    if saved.get('schema') != SCHEMA:
        raise CountProductError('unsupported cache angular manifest')
    identity, run_hash = saved['identity'], saved['run_hash']
    edges = np.asarray(saved['edges'], dtype=float)
    if array_digest({'edges': edges}, identity) != run_hash:
        raise CountProductError('cache angular run identity mismatch')
    task_plan = {key: saved[key] for key in ('left_cells', 'right_cells', 'right_moments', 'schedule')}
    if hashlib.sha256(canonical_json(task_plan).encode()).hexdigest() != identity['task_plan_sha256']:
        raise CountProductError('sparse task plan identity mismatch')
    lp, rp, schedule = saved['left_cells'], saved['right_cells'], saved['schedule']
    if (lp != sorted(set(lp)) or rp != sorted(set(rp))
            or set(schedule) != {str(p) for p in lp}
            or any(qs != sorted(set(qs)) or not set(qs).issubset(rp) for qs in schedule.values())):
        raise CountProductError('invalid sparse cell schedule')
    has_z = identity['has_redshift_moments']
    needed = _result_bytes(schedule, len(edges)-1, has_z, len(rp))
    limit = _positive_int(saved['max_result_bytes'], 'saved max_result_bytes')
    if max_result_bytes is not None:
        limit = min(limit, _positive_int(max_result_bytes, 'max_result_bytes'))
    if saved['result_bytes'] != needed or needed > limit:
        raise CountProductError('sparse reconstruction exceeds explicit result byte bound')
    rm = np.asarray(saved['right_moments'], dtype=float).reshape(-1, 3)
    if rm.shape != (len(rp), 3) or not np.isfinite(rm).all() or np.any(rm <= 0) or np.any(rm[:, 0] != np.floor(rm[:, 0])):
        raise CountProductError('invalid full-population right moments')
    nrows = sum(len(qs) for qs in schedule.values())
    pairs = np.empty((nrows, 2), dtype=np.int64)
    npairs = np.empty((nrows, len(edges)-1), dtype=np.int64)
    weighted = np.empty_like(npairs, dtype=float)
    z = np.empty_like(weighted) if has_z else None
    z2 = np.empty_like(weighted) if has_z else None
    lm = np.empty((len(lp), 3))
    files, timing = [], []
    start = 0
    for k, p in enumerate(lp):
        neighbours = schedule[str(p)]
        path = root/f'p{p}.npz'
        arrays = _read_block(path, _block_metadata(identity, run_hash, p), edges, neighbours, limit)
        stop = start + len(neighbours)
        pairs[start:stop], npairs[start:stop], weighted[start:stop] = (arrays[name] for name in ('patch_pairs', 'npairs', 'weighted'))
        if has_z:
            z[start:stop], z2[start:stop] = arrays['weighted_z'], arrays['weighted_z2']
        lm[k] = arrays['left_moments']
        timing.extend(arrays['seconds'].tolist())
        files.append(path)
        start = stop
    counts = PatchCounts(edges, pairs, npairs, weighted, dict(zip(lp, lm[:, 1])),
                         dict(zip(rp, rm[:, 1])), identity['backend'], 'float64')
    for value in (edges, pairs, npairs, weighted, lm, rm, z, z2):
        if value is not None:
            value.setflags(write=False)
    return CountRun(counts, z, z2, tuple(files), 0, len(files), run_hash, lm, rm,
                    {**identity, 'task_seconds': timing})
