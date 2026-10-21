"""Bounded reference-random × photometric-data counts (P2/P3, §4.2, V11).

The historical DD/DR driver remains immutable for existing source-pinned runs.
This RD entry point shares its bounded shard, block and reconstruction helpers,
but validates genuine random-reference/data-photo identities and records z/z²
moments for the RD denominator windows. It never requests compute resources.
"""
from dataclasses import replace
import hashlib
from importlib.metadata import version
import json
from numbers import Integral
from pathlib import Path
import re
import time

import numpy as np

from bxz.counts_io import CountProductError, array_digest, canonical_json, publish_manifest
from .angular import AngularCatalog, count_angular
from .driver import _software
from . import cache_driver
from .cache_driver import (MANIFEST_BYTES, SCHEMA, CacheCountIncomplete,
    _positive_int, _manifest, _group_cells, _load_cell, candidate_cells,
    _result_bytes, _block_metadata, _read_block, _validate_block,
    _publish_block, read_cache_angular_run)


def _positive_reference_rows(values, labels):
    """Omit zero-weight centres without changing weighted counts or W/Q.

    Reference N and integer pair counts describe positive-weight centres.
    Zero-only slices retain empty blocks and zero reference moments. Reject
    malformed selected rows before filtering; zero weight must not hide them.
    """
    if (any(not np.isfinite(values[name]).all() for name in ('ra', 'dec', 'weight', 'z'))
            or np.any(np.abs(values['dec']) > np.pi/2)
            or np.any(values['weight'] < 0)):
        raise ValueError('reference rows require finite coordinates/redshifts, valid declinations and nonnegative weights')
    positive = values['weight'] > 0
    return values[positive], labels[positive]


def _request(edges, slice_id, count_type, provenance, context, cache):
    edges = np.asarray(edges, dtype=float)
    if (edges.ndim != 1 or len(edges) < 2 or not np.isfinite(edges).all()
            or edges[0] <= 0 or edges[-1] > np.pi or np.any(np.diff(edges) <= 0)):
        raise ValueError('invalid angular edges')
    if count_type != 'RD' or not isinstance(slice_id, str) or not slice_id.strip():
        raise ValueError('one nonempty slice ID and RD count type required')
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


def _prepare_reference_counts(manifest_path, reference_group, photo_group, edges, *,
                              slice_id, count_type, provenance, measurement_context,
                              max_tasks, max_memory_bytes, max_result_bytes, backend):
    max_tasks = _positive_int(max_tasks, 'max_tasks')
    max_memory_bytes = _positive_int(max_memory_bytes, 'max_memory_bytes')
    max_result_bytes = _positive_int(max_result_bytes, 'max_result_bytes')
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
            or left['source'].get('kind') != 'randoms' or reference_group != f'{tracer}_{cap}_randoms'
            or photo_group != 'photo_data' or right['source'].get('kind') != 'data'
            or 'tracer' in right['source']):
        raise ValueError('RD requires a named DR1 tracer/cap randoms group and photometric data')
    allowed = cache['reference_z_ranges'][left['source']['tracer']]
    if context['slice_edges'][0] < allowed[0] or context['slice_edges'][1] > allowed[1]:
        raise ValueError('fine slice lies outside cached reference redshift support')
    from bxz.cache_io import REFERENCE_DTYPE, PHOTO_DTYPE
    lp, _ = _group_cells(left, reference_group, REFERENCE_DTYPE)
    rp, rm = _group_cells(right, photo_group, PHOTO_DTYPE)
    schedule = candidate_cells(lp, rp, edges[-1], max_tasks=max_tasks)
    has_z = True
    result_bytes = _result_bytes(schedule, len(edges)-1, has_z, len(rp))
    if result_bytes > max_result_bytes:
        raise ValueError('scheduled result exceeds explicit max_result_bytes bound')
    # Preflight every task before reading payloads or publishing output.
    max_live_array_bytes = 0
    for p, neighbours in schedule.items():
        nleft = left['cells'][p]['count']
        nright = max((right['cells'][str(q)]['count'] for q in neighbours), default=0)
        live_bytes = 256*(nleft+nright) + 4*MANIFEST_BYTES + result_bytes
        max_live_array_bytes = max(max_live_array_bytes, live_bytes)
        if live_bytes > max_memory_bytes:
            raise ValueError('live cell/adapter temporaries exceed explicit max_memory_bytes bound')
    software = _software(backend)
    software['reference_cache_driver.py_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    software['cache_driver.py_sha256'] = hashlib.sha256(Path(cache_driver.__file__).read_bytes()).hexdigest()
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
        reference_zero_weight_policy='omit_zero_weight_centres; preserve_weighted_pairs_and_W_Q',
        reference_count_convention='N_and_npairs_use_positive_weight_reference_centres',
        pair_convention='ordered_cross; [low, high); scalar_positive_weights', units='radian',
        cache_metadata_sha256=cache_digest, reference_group=reference_group, photo_group=photo_group,
        catalogue_hashes={
            'left': array_digest({}, dict(cache=cache_digest, group=reference_group, slice_edges=context['slice_edges'])),
            'right': array_digest({}, dict(cache=cache_digest, group=photo_group))})
    task_plan = dict(left_cells=lp, right_cells=rp, right_moments=rm.tolist(), schedule=schedule)
    identity['task_plan_sha256'] = hashlib.sha256(canonical_json(task_plan).encode()).hexdigest()
    run_hash = array_digest({'edges': edges}, identity)
    manifest = dict(schema=SCHEMA, identity=identity, run_hash=run_hash, edges=edges.tolist(),
        left_cells=lp, right_cells=rp, right_moments=rm.tolist(), schedule=schedule,
        result_bytes=result_bytes, max_result_bytes=max_result_bytes,
        pruning='centre_separation <= theta_max + 2*healpy.max_pixrad(16) + outward_roundoff',
        block_layout='one atomic NPZ per left cell; all scheduled patch-pair rows retained')
    return dict(manifest=manifest, cache_state=cache_state, manifest_state=manifest_state,
                load=load, max_live_array_bytes=max_live_array_bytes)


def plan_reference_cache_counts(manifest_path, reference_group, photo_group, edges, *,
                               slice_id, count_type, provenance, measurement_context,
                               max_tasks, max_memory_bytes, max_result_bytes, backend='pycorr'):
    """Return pinned metadata and admission estimates; no shard reads or writes.

    The returned run_manifest is exactly what execution publishes. Source/software
    hashing and bounded cache metadata reads are the only I/O. The memory estimate
    covers known arrays, not a hard RSS guarantee or compute authorization.
    """
    planned = _prepare_reference_counts(manifest_path, reference_group, photo_group, edges,
        slice_id=slice_id, count_type=count_type, provenance=provenance,
        measurement_context=measurement_context, max_tasks=max_tasks,
        max_memory_bytes=max_memory_bytes, max_result_bytes=max_result_bytes, backend=backend)
    return dict(run_manifest=planned['manifest'],
                max_live_array_bytes=planned['max_live_array_bytes'])


def run_reference_cache_counts(manifest_path, reference_group, photo_group, edges, output_dir, *,
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
    if max_new_blocks is not None:
        max_new_blocks = _positive_int(max_new_blocks, 'max_new_blocks')
    planned = _prepare_reference_counts(manifest_path, reference_group, photo_group, edges,
        slice_id=slice_id, count_type=count_type, provenance=provenance,
        measurement_context=measurement_context, max_tasks=max_tasks,
        max_memory_bytes=max_memory_bytes, max_result_bytes=max_result_bytes, backend=backend)
    manifest = planned['manifest']
    identity, run_hash = manifest['identity'], manifest['run_hash']
    edges = np.asarray(manifest['edges'], dtype=float)
    context = identity['measurement_context']
    lp, schedule = manifest['left_cells'], manifest['schedule']
    load, manifest_state = planned['load'], planned['manifest_state']
    cache_state = planned['cache_state']
    has_z = True
    selected_cells = lp if block_cells is None else list(block_cells)
    if (not selected_cells and lp or len(selected_cells) != len(set(selected_cells))
            or any(isinstance(p, bool) or not isinstance(p, Integral) or p not in lp for p in selected_cells)):
        raise ValueError('block_cells must be unique existing left-cell integers')
    selected_cells = sorted(int(p) for p in selected_cells)
    root = Path(output_dir) / run_hash
    root.mkdir(parents=True, exist_ok=True)
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
        values, labels = _positive_reference_rows(values, labels)
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
                if has_z:
                    for power, key in ((1, 'weighted_z'), (2, 'weighted_z2')):
                        weighted = AngularCatalog(a.ra, a.dec, a.weight*values['z']**power, a.patch)
                        arrays[key][k] = count_angular(weighted, b, edges, backend=backend).total_weighted
                    del weighted
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
