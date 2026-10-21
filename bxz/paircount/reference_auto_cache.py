"""Bounded, restartable reference-only DD/DR/RR counts (P2/P3, V3/V11).

Both endpoints use the same half-open redshift slice. DD/RR are ordered
cross-counts of the identical catalogue: the strictly positive first angular
edge excludes self pairs. All selected endpoint N/W/Q and radial histograms
survive spatial pruning, including cells with no selected positive weights.
No compute allocation or catalogue discovery is performed here.
"""
from dataclasses import dataclass, replace
import hashlib
from importlib.metadata import version
import json
from math import prod
import re
from pathlib import Path
from zipfile import ZipFile, BadZipFile

import numpy as np

from bxz.counts_io import CountProductError, array_digest, canonical_json, publish_manifest
from .angular import AngularCatalog, PatchCounts, count_angular
from .driver import CountRun, _software
from . import cache_driver, reference_cache_driver
from .cache_driver import (MANIFEST_BYTES, CacheCountIncomplete, _positive_int,
    _manifest, _group_cells, _load_cell, candidate_cells, _request, _result_bytes,
    _block_metadata, _read_block, _publish_block, _block_digest, _validate_block)
from .reference_cache_driver import _positive_reference_rows

SCHEMA = 'bxz.reference-auto-cache-manifest.v1'


@dataclass(frozen=True)
class ReferenceAutoRun(CountRun):
    """CountRun storage plus per-cell radial N/W/Q.

    PatchCounts.resample is a cross-normalization convenience and must NOT be
    used to normalize DD/RR: the auto estimator must subtract the appropriate
    Q self term, including its declared literal/match pair-weighting rule.
    """
    radial_edges: np.ndarray
    left_radial_moments: np.ndarray
    right_radial_moments: np.ndarray


def _budget(schedule, nr, bins, radial_bins):
    # Final endpoint histograms, one input block plus digest copies, and result.
    return (_result_bytes(schedule, bins, False, nr)
            + 96 * (len(schedule) + nr + 1) * (radial_bins + 1))


def _prepare(manifest_path, left_group, right_group, edges, *, radial_edges,
             slice_id, count_type, provenance, measurement_context, max_tasks,
             max_memory_bytes, max_result_bytes, backend):
    max_tasks = _positive_int(max_tasks, 'max_tasks')
    max_memory_bytes = _positive_int(max_memory_bytes, 'max_memory_bytes')
    max_result_bytes = _positive_int(max_result_bytes, 'max_result_bytes')
    if backend != 'pycorr' or count_type not in ('DD', 'DR', 'RR'):
        raise ValueError('reference auto requires pycorr and DD/DR/RR')
    def state():
        s = Path(manifest_path).stat()
        return s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns
    initial = state()
    cache, digest = _manifest(manifest_path)
    if state() != initial:
        raise CountProductError('cache manifest changed while reading')
    if (cache.get('schema') != 'bxz.dr1-full-cache.v1' or cache.get('completed') is not True
            or cache['storage_scheme'] != dict(nside=16, frame='equatorial', ordering='NESTED')
            or cache['units'] != dict(angles='radian', redshift='dimensionless')):
        raise ValueError('completed equatorial NESTED16 radian cache required')
    edges, provenance, context = _request(edges, slice_id, 'DD', provenance, measurement_context, cache)
    radial_edges = np.asarray(radial_edges, dtype=float)
    if (radial_edges.ndim != 1 or len(radial_edges) < 2
            or not np.isfinite(radial_edges).all() or np.any(np.diff(radial_edges) <= 0)
            or not np.array_equal(radial_edges[[0, -1]], context['slice_edges'])):
        raise ValueError('radial_edges must partition the exact half-open slice')
    left, right = (cache['catalogues'][g] for g in (left_group, right_group))
    tracer, cap = left['source'].get('tracer'), left['source'].get('cap')
    kinds = {'DD': ('data', 'data'), 'DR': ('data', 'randoms'), 'RR': ('randoms', 'randoms')}[count_type]
    if tracer not in ('BGS_BRIGHT', 'LRG', 'ELG_LOPnotqso', 'QSO') or cap not in ('NGC', 'SGC'):
        raise ValueError('named DR1 reference tracer/cap required')
    for group, saved, kind in zip((left_group, right_group), (left, right), kinds):
        if (group != f'{tracer}_{cap}_{kind}' or saved['source'].get('kind') != kind
                or saved['source'].get('tracer') != tracer or saved['source'].get('cap') != cap):
            raise ValueError('DD/DR/RR requires matching reference tracer/cap groups')
    allowed = cache['reference_z_ranges'][tracer]
    if radial_edges[0] < allowed[0] or radial_edges[-1] > allowed[1]:
        raise ValueError('slice lies outside cached reference support')
    from bxz.cache_io import REFERENCE_DTYPE
    import bxz.cache_io as loader
    lp, _ = _group_cells(left, left_group, REFERENCE_DTYPE)
    rp, _ = _group_cells(right, right_group, REFERENCE_DTYPE)
    schedule = candidate_cells(lp, rp, edges[-1], max_tasks=max_tasks)
    needed = _budget(schedule, len(rp), len(edges)-1, len(radial_edges)-1)
    if needed > max_result_bytes:
        raise ValueError('scheduled result exceeds explicit max_result_bytes bound')
    largest_left = max((v['count'] for v in left['cells'].values()), default=0)
    largest_right = max((v['count'] for v in right['cells'].values()), default=0)
    live = 256*(largest_left+largest_right) + 4*MANIFEST_BYTES + needed
    if live > max_memory_bytes:
        raise ValueError('cell/adapter arrays exceed explicit max_memory_bytes bound')
    software = _software(backend)
    for source in (Path(__file__), Path(cache_driver.__file__), Path(reference_cache_driver.__file__), Path(loader.__file__)):
        software[source.name + '_sha256'] = hashlib.sha256(source.read_bytes()).hexdigest()
    software['healpy'] = version('healpy')
    identity = dict(schema='bxz.reference-auto-cache-run.v1', slice_id=slice_id,
        count_type=count_type, backend=backend, provenance=provenance,
        measurement_context=context, software=software, cache_metadata_sha256=digest,
        left_group=left_group, right_group=right_group, has_redshift_moments=False,
        pair_convention='ordered_distinct' if count_type != 'DR' else 'ordered_cross',
        units='radian', reference_count_convention='N_and_npairs_use_positive_weight_reference_centres',
        radial_convention='half_open_z_bins; N_W_Q_per_cell; zero_weights_omitted',
        catalogue_hashes={side: array_digest({}, dict(cache=digest, group=group,
            slice_edges=context['slice_edges'])) for side, group in (('left', left_group), ('right', right_group))})
    task_plan = dict(left_cells=lp, right_cells=rp, schedule=schedule)
    identity['task_plan_sha256'] = hashlib.sha256(canonical_json(task_plan).encode()).hexdigest()
    run_hash = array_digest(dict(edges=edges, radial_edges=radial_edges), identity)
    manifest = dict(schema=SCHEMA, identity=identity, run_hash=run_hash,
        edges=edges.tolist(), radial_edges=radial_edges.tolist(), **task_plan,
        result_bytes=needed, max_result_bytes=max_result_bytes)
    # Conservative payload read bound: all endpoint blocks plus every pair task.
    read_bytes = sum(v['bytes'] for group in (left, right) for v in group['cells'].values())
    read_bytes += sum(left['cells'][p]['bytes'] + sum(right['cells'][str(q)]['bytes'] for q in qs)
                      for p, qs in schedule.items())
    def load(group, cell):
        if state() != initial:
            raise CountProductError('cache manifest changed during counting')
        saved = left if group == left_group else right
        values, labels = _load_cell(manifest_path, group, saved, cell, REFERENCE_DTYPE, max_bytes=max_memory_bytes)
        if state() != initial:
            raise CountProductError('cache manifest changed during counting')
        # Validate before selection; malformed rows must not silently disappear.
        values, labels = _positive_reference_rows(values, labels)
        keep = (values['z'] >= radial_edges[0]) & (values['z'] < radial_edges[-1])
        return values[keep], labels[keep]
    # NPZ Unicode metadata needs four bytes per character. Reserve 16 KiB
    # per file for ZIP/NPY headers; numeric payloads are bounded explicitly.
    disk = len(canonical_json(dict(metadata=manifest, sha256='0'*64)).encode())
    for side, cells in (('left',lp),('right',rp)):
        for cell in cells:
            disk += 24*len(radial_edges) + 4*len(canonical_json(
                _endpoint_metadata(manifest,side,cell))) + 256 + 16384
    for cell in lp:
        nq = len(schedule[str(cell)])
        disk += (edges.nbytes + 24 + nq*(24+16*(len(edges)-1))
                 + 4*len(canonical_json(_block_metadata(identity,run_hash,cell))) + 256 + 16384)
    return dict(run_manifest=manifest, max_live_array_bytes=live,
                logical_read_bytes_upper=read_bytes, block_count=len(lp)*2+len(rp),
                total_blocks=len(lp)*2+len(rp), output_files=len(lp)*2+len(rp)+1,
                result_bytes=needed, run_hash=run_hash,
                count_file_byte_bound=disk,
                load=load)


def plan_reference_auto_cache_counts(manifest_path, left_group, right_group, edges, *,
        radial_edges, slice_id, count_type, provenance, measurement_context,
        max_tasks, max_memory_bytes, max_result_bytes, backend='pycorr'):
    """Metadata-only preflight; no payload reads, writes or allocation.

    Whole-cache cell sizes conservatively bound each selected-slice load.
    Logical read bytes exclude small metadata and describe a fresh execution.
    """
    result = _prepare(manifest_path, left_group, right_group, edges,
        radial_edges=radial_edges, slice_id=slice_id, count_type=count_type,
        provenance=provenance, measurement_context=measurement_context,
        max_tasks=max_tasks, max_memory_bytes=max_memory_bytes,
        max_result_bytes=max_result_bytes, backend=backend)
    return {k: v for k, v in result.items() if k != 'load'}


def _endpoint_metadata(manifest, side, cell):
    return dict(schema='bxz.reference-auto-endpoint.v1', run_hash=manifest['run_hash'], side=side, cell=cell)


def _inspect_npz(path, shapes, limit):
    """Check physical and header-declared sizes before NumPy allocation."""
    if path.stat().st_size > limit:
        raise CountProductError('archive file exceeds explicit byte bound')
    with ZipFile(path) as archive:
        entries = archive.infolist()
        names = [entry.filename for entry in entries]
        if (len(names) != len(set(names))
                or set(names) != {k+'.npy' for k in shapes} | {'metadata_json.npy','sha256.npy'}
                or sum(entry.file_size for entry in entries) > limit):
            raise CountProductError('archive names or uncompressed byte bound invalid')
        for entry in entries:
            with archive.open(entry) as member:
                fmt = np.lib.format.read_magic(member)
                if fmt == (1,0):
                    shape, _, dtype = np.lib.format.read_array_header_1_0(member)
                elif fmt == (2,0):
                    shape, _, dtype = np.lib.format.read_array_header_2_0(member)
                else:
                    raise CountProductError('unsupported NPY version')
                name = entry.filename[:-4]
                text = name in ('metadata_json','sha256')
                if (dtype.hasobject or (text and (shape != () or dtype.kind != 'U'))
                        or (not text and (shape != shapes[name][0] or dtype != shapes[name][1]))
                        or member.tell()+prod(shape)*dtype.itemsize != entry.file_size):
                    raise CountProductError('invalid NPY type/shape/payload size')


def _read_pair(path, metadata, edges, neighbours, limit):
    shape = (len(neighbours),len(edges)-1)
    shapes = dict(edges=(edges.shape,np.dtype('float64')),
        patch_pairs=((len(neighbours),2),np.dtype('int64')),
        npairs=(shape,np.dtype('int64')), weighted=(shape,np.dtype('float64')),
        weighted_z=((0,),np.dtype('float64')), weighted_z2=((0,),np.dtype('float64')),
        left_moments=((3,),np.dtype('float64')),seconds=((len(neighbours),),np.dtype('float64')))
    try:
        _inspect_npz(path, shapes, limit)
        return _read_block(path, metadata, edges, neighbours, limit)
    except CountProductError:
        raise
    except (OSError, ValueError, KeyError, TypeError, EOFError, BadZipFile) as exc:
        raise CountProductError('unreadable reference pair block') from exc


def _validate_identity(identity):
    count_type = identity.get('count_type')
    if (identity.get('schema') != 'bxz.reference-auto-cache-run.v1'
            or count_type not in ('DD','DR','RR') or identity.get('backend') != 'pycorr'
            or identity.get('has_redshift_moments') is not False or identity.get('units') != 'radian'
            or identity.get('pair_convention') != ('ordered_cross' if count_type=='DR' else 'ordered_distinct')
            or identity.get('reference_count_convention') != 'N_and_npairs_use_positive_weight_reference_centres'
            or identity.get('radial_convention') != 'half_open_z_bins; N_W_Q_per_cell; zero_weights_omitted'):
        raise CountProductError('unsupported reference auto identity semantics')
    pattern = r'(BGS_BRIGHT|LRG|ELG_LOPnotqso|QSO)_(NGC|SGC)_(data|randoms)'
    left = re.fullmatch(pattern, identity.get('left_group',''))
    right = re.fullmatch(pattern, identity.get('right_group',''))
    kinds = {'DD':('data','data'),'DR':('data','randoms'),'RR':('randoms','randoms')}[count_type]
    if (left is None or right is None or left.groups()[:2] != right.groups()[:2]
            or (left.group(3),right.group(3)) != kinds):
        raise CountProductError('reference auto groups differ from count type')
    digest = identity.get('cache_metadata_sha256','')
    if re.fullmatch('[0-9a-f]{64}',digest) is None:
        raise CountProductError('invalid cache identity')
    for side in ('left','right'):
        expected = array_digest({},dict(cache=digest,group=identity[side+'_group'],
            slice_edges=identity['measurement_context']['slice_edges']))
        if identity.get('catalogue_hashes',{}).get(side) != expected:
            raise CountProductError('reference catalogue identity mismatch')


def _read_endpoint(path, metadata, nr, limit):
    try:
        _inspect_npz(path, dict(moments=((3,),np.dtype('float64')),
            radial_moments=((nr,3),np.dtype('float64'))),limit)
        with np.load(path, allow_pickle=False) as saved:
            actual = json.loads(str(saved['metadata_json'].item()))
            digest = str(saved['sha256'].item())
            arrays = {k: saved[k] for k in saved.files if k not in ('metadata_json', 'sha256')}
        if canonical_json(actual) != canonical_json(metadata) or _block_digest(arrays, actual) != digest:
            raise CountProductError('endpoint identity/checksum mismatch')
        if set(arrays) != {'moments', 'radial_moments'}:
            raise CountProductError('invalid endpoint arrays')
        for key, shape in (('moments', (3,)), ('radial_moments', (nr, 3))):
            a = arrays[key]
            if a.shape != shape or a.dtype != np.dtype('float64') or not np.isfinite(a).all() or np.any(a < 0):
                raise CountProductError('invalid endpoint moments')
            if (np.any(a[..., 0] != np.floor(a[..., 0]))
                    or np.any((a[..., 0] == 0) != (a[..., 1] == 0))
                    or np.any((a[..., 0] == 0) != (a[..., 2] == 0))):
                raise CountProductError('inconsistent endpoint N/W/Q')
        # Totals are defined by summing histogram bins, enabling exact replay.
        if not np.array_equal(arrays['moments'], arrays['radial_moments'].sum(axis=0)):
            raise CountProductError('endpoint histogram totals differ')
        return arrays
    except CountProductError:
        raise
    except (OSError, ValueError, KeyError, TypeError, EOFError, BadZipFile) as exc:
        raise CountProductError('unreadable endpoint block') from exc


def run_reference_auto_cache_counts(manifest_path, left_group, right_group, edges, output_dir, *,
        radial_edges, slice_id, count_type, provenance, measurement_context,
        max_tasks, max_memory_bytes, max_result_bytes, backend='pycorr', max_new_blocks=None):
    """Execute/resume atomic endpoint and left-cell pair blocks.

    max_new_blocks counts newly published files, including endpoint blocks.
    A complete replay validates existing blocks without touching cache payloads.
    One worker owns a run; concurrent writers fail rather than overwrite.
    """
    if max_new_blocks is not None:
        max_new_blocks = _positive_int(max_new_blocks, 'max_new_blocks')
    plan = _prepare(manifest_path, left_group, right_group, edges,
        radial_edges=radial_edges, slice_id=slice_id, count_type=count_type,
        provenance=provenance, measurement_context=measurement_context,
        max_tasks=max_tasks, max_memory_bytes=max_memory_bytes,
        max_result_bytes=max_result_bytes, backend=backend)
    m, load = plan['run_manifest'], plan['load']
    edges, radial_edges = np.asarray(m['edges']), np.asarray(m['radial_edges'])
    root = Path(output_dir)/m['run_hash']
    root.mkdir(parents=True, exist_ok=True)
    publish_manifest(root/'run.json', m)
    computed = reused = 0
    def admit():
        if max_new_blocks is not None and computed >= max_new_blocks:
            raise CacheCountIncomplete(root, computed, reused, plan['block_count'])
    for side, group in (('left', left_group), ('right', right_group)):
        for cell in m[side+'_cells']:
            path = root/f'{side}-{cell}.npz'
            meta = _endpoint_metadata(m, side, cell)
            if path.exists():
                _read_endpoint(path, meta, len(radial_edges)-1, max_result_bytes)
                reused += 1
                continue
            admit()
            values, labels = load(group, cell)
            index = np.searchsorted(radial_edges, values['z'], side='right')-1
            hist = np.stack([np.bincount(index, weights=weight, minlength=len(radial_edges)-1)
                for weight in (None, values['weight'], values['weight']**2)], axis=-1).astype(float)
            _publish_block(path, dict(moments=hist.sum(axis=0), radial_moments=hist), meta)
            computed += 1
            del values, labels, hist
    for cell in m['left_cells']:
        path = root/f'p{cell}.npz'
        meta = _block_metadata(m['identity'], m['run_hash'], cell)
        neighbours = m['schedule'][str(cell)]
        if path.exists():
            _read_pair(path, meta, edges, neighbours, max_result_bytes)
            reused += 1
            continue
        admit()
        values, labels = load(left_group, cell)
        a = AngularCatalog(values['ra'], values['dec'], values['weight'], labels)
        shape = (len(neighbours), len(edges)-1)
        endpoint = _read_endpoint(root/f'left-{cell}.npz', _endpoint_metadata(m, 'left', cell), len(radial_edges)-1, max_result_bytes)
        arrays = dict(edges=edges, patch_pairs=np.array([(cell, q) for q in neighbours], dtype=np.int64).reshape(-1, 2),
            npairs=np.zeros(shape, dtype=np.int64), weighted=np.zeros(shape),
            weighted_z=np.empty(0), weighted_z2=np.empty(0),
            left_moments=endpoint['moments'], seconds=np.zeros(len(neighbours)))
        import time
        for k, q in enumerate(neighbours):
            tick = time.perf_counter()
            if len(values):
                other, other_labels = load(right_group, q)
                b = AngularCatalog(other['ra'], other['dec'], other['weight'], other_labels)
                counted = count_angular(a, b, edges, backend=backend)
                arrays['npairs'][k], arrays['weighted'][k] = counted.total_npairs, counted.total_weighted
                del other, other_labels, b, counted
            arrays['seconds'][k] = time.perf_counter()-tick
        _validate_block(arrays, meta, edges, neighbours)
        _publish_block(path, arrays, meta)
        computed += 1
        del values, labels, a, arrays
    return replace(read_reference_auto_cache_run(root, max_result_bytes=max_result_bytes), computed=computed, reused=reused)


def read_reference_auto_cache_run(directory, *, max_result_bytes=None):
    """Strict bounded reconstruction without reading any catalogue payloads."""
    root = Path(directory)
    m, _ = _manifest(root/'run.json')
    if m.get('schema') != SCHEMA:
        raise CountProductError('unsupported reference auto manifest')
    identity, run_hash = m['identity'], m['run_hash']
    _validate_identity(identity)
    edges, radial_edges = np.asarray(m['edges'], dtype=float), np.asarray(m['radial_edges'], dtype=float)
    if (edges.ndim != 1 or len(edges)<2 or edges[0]<=0 or edges[-1]>np.pi
            or not np.isfinite(edges).all() or np.any(np.diff(edges)<=0)
            or radial_edges.ndim != 1 or len(radial_edges)<2
            or not np.isfinite(radial_edges).all() or np.any(np.diff(radial_edges)<=0)
            or not np.array_equal(radial_edges[[0,-1]], identity['measurement_context']['slice_edges'])):
        raise CountProductError('invalid angular/radial grid')
    if array_digest(dict(edges=edges, radial_edges=radial_edges), identity) != run_hash:
        raise CountProductError('reference auto identity mismatch')
    plan = {k: m[k] for k in ('left_cells', 'right_cells', 'schedule')}
    if hashlib.sha256(canonical_json(plan).encode()).hexdigest() != identity['task_plan_sha256']:
        raise CountProductError('reference auto task plan mismatch')
    lp, rp, schedule = m['left_cells'], m['right_cells'], m['schedule']
    for cells in (lp, rp):
        if cells != sorted(set(cells)) or any(type(p) is not int or not 0<=p<3072 for p in cells):
            raise CountProductError('invalid storage cells')
    if (set(schedule) != {str(p) for p in lp}
            or any(qs != sorted(set(qs)) or not set(qs).issubset(rp) for qs in schedule.values())):
        raise CountProductError('invalid pair schedule')
    needed = _budget(schedule, len(rp), len(edges)-1, len(radial_edges)-1)
    limit = _positive_int(m['max_result_bytes'], 'saved max_result_bytes')
    if max_result_bytes is not None:
        limit = min(limit, _positive_int(max_result_bytes, 'max_result_bytes'))
    if needed != m['result_bytes'] or needed > limit:
        raise CountProductError('reference auto result exceeds byte bound')
    endpoint, files = {}, []
    for side, cells in (('left',lp),('right',rp)):
        hist = np.empty((len(cells), len(radial_edges)-1, 3))
        for j, cell in enumerate(cells):
            path = root/f'{side}-{cell}.npz'
            hist[j] = _read_endpoint(path, _endpoint_metadata(m, side, cell), len(radial_edges)-1, limit)['radial_moments']
            files.append(path)
        endpoint[side] = hist
    lm, rm = endpoint['left'].sum(axis=1), endpoint['right'].sum(axis=1)
    if identity['count_type'] in ('DD','RR'):
        if lp != rp or not np.array_equal(endpoint['left'], endpoint['right']):
            raise CountProductError('same-catalogue endpoint moments differ')
    n = sum(map(len,schedule.values()))
    pairs = np.empty((n,2),dtype=np.int64)
    npairs = np.empty((n,len(edges)-1),dtype=np.int64)
    weighted = np.empty_like(npairs,dtype=float)
    start, timing = 0, []
    for j,p in enumerate(lp):
        qs = schedule[str(p)]
        path = root/f'p{p}.npz'
        a = _read_pair(path,_block_metadata(identity,run_hash,p),edges,qs,limit)
        if not np.array_equal(a['left_moments'],lm[j]):
            raise CountProductError('pair block endpoint moments differ')
        stop = start+len(qs)
        pairs[start:stop],npairs[start:stop],weighted[start:stop] = a['patch_pairs'],a['npairs'],a['weighted']
        timing.extend(a['seconds'].tolist())
        files.append(path)
        start = stop
    counts = PatchCounts(edges,pairs,npairs,weighted,dict(zip(lp,lm[:,1])),dict(zip(rp,rm[:,1])),identity['backend'],'float64')
    for a in (edges,radial_edges,pairs,npairs,weighted,lm,rm,*endpoint.values()):
        a.setflags(write=False)
    return ReferenceAutoRun(counts,None,None,tuple(files),0,len(files),run_hash,lm,rm,
        {**identity,'task_seconds':timing},radial_edges,endpoint['left'],endpoint['right'])
