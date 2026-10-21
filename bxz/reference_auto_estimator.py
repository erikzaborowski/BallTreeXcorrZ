"""Same-cap reference angular LS jackknife (P3 §4.6/§4.9, V3).

No counting, cap pooling, covariance inversion, bias fitting or theory occurs.
DD/RR contain ordered distinct-object pairs; DR contains ordered cross pairs.
Match deletions use the same pair factor in angular counts and all-pair
normalizations, subtracting Q only for retained auto self terms.
"""
from dataclasses import dataclass
from numbers import Integral
import re

import numpy as np

from .counts_io import CountProductError, canonical_json
from .paircount.reference_auto_cache import ReferenceAutoRun, _validate_identity


COUNT_TYPES = ('DD', 'DR', 'RR')


@dataclass(frozen=True)
class ReferenceAutoJackknife:
    """Full result first, then deletions in sorted common-region order.

    The three original runs are retained by reference for later radial-window
    modelling, not copied or transformed into a retained-endpoint p(z) model.
    The caller-owned input arrays are excluded from max_result_bytes.
    """
    inputs: tuple
    theta_edges: np.ndarray
    radial_edges: np.ndarray
    region_ids: tuple
    omitted_regions: tuple
    cell_regions: tuple
    weighted_counts: np.ndarray  # realization, (DD, DR, RR), angle
    normalizations: np.ndarray  # realization, (DD, DR, RR)
    normalized_counts: np.ndarray
    value: np.ndarray  # realization, angle; NaN iff unsupported
    supported: np.ndarray
    pair_weighting: str
    cross_region_weight: float | None
    run_hashes: tuple
    result_bytes_bound: int
    count_types: tuple = COUNT_TYPES
    estimator: str = 'reference_angular_Landy_Szalay'


def _label(value):
    return (isinstance(value, Integral) and not isinstance(value, (bool, np.bool_))
            and 0 <= value <= np.iinfo(np.int64).max)


def _array(a, shape, *, integer=False):
    if (not isinstance(a, np.ndarray) or a.shape != shape or a.flags.writeable
            or (a.dtype.kind not in 'iu' if integer else a.dtype != np.dtype('float64'))
            or not np.isfinite(a).all() or np.any(a < 0)):
        raise CountProductError('invalid or mutable count/moment array')


def _endpoint(run, side):
    weights = getattr(run.counts, 'weight_sums_'+side)
    cells = tuple(weights)
    if (cells != tuple(sorted(set(cells)))
            or any(not _label(c) or c >= 3072 for c in cells)):
        raise CountProductError('endpoint cells must be sorted unique NESTED16 labels')
    moments = getattr(run, side+'_moments')
    radial = getattr(run, side+'_radial_moments')
    _array(moments, (len(cells), 3))
    _array(radial, (len(cells), len(run.radial_edges)-1, 3))
    for a in (moments, radial):
        if (np.any(a[..., 0] != np.floor(a[..., 0]))
                or np.any((a[..., 0] == 0) != (a[..., 1] == 0))
                or np.any((a[..., 0] == 0) != (a[..., 2] == 0))):
            raise CountProductError('endpoint N/W/Q support differs')
    if (not np.array_equal(moments, radial.sum(axis=1))
            or not np.array_equal(moments[:, 1], list(weights.values()))):
        raise CountProductError('endpoint totals, radial histogram and weight sums differ')
    return cells, moments, radial, run.metadata['catalogue_hashes'][side]


def _same_endpoint(a, b):
    if (a[0] != b[0] or a[3] != b[3]
            or not np.array_equal(a[1], b[1]) or not np.array_equal(a[2], b[2])):
        raise CountProductError('shared reference endpoint identity or moments differ')


def _validate_runs(runs):
    endpoints = []
    common = None
    theta = radial_edges = None
    for run, kind in zip(runs, COUNT_TYPES):
        if not isinstance(run, ReferenceAutoRun):
            raise TypeError('three validated ReferenceAutoRun products are required')
        m, counts = run.metadata, run.counts
        try:
            _validate_identity(m)
            if m['count_type'] != kind:
                raise CountProductError('expected DD, DR, RR in that order')
            shared = canonical_json({key: m[key] for key in (
                'slice_id', 'measurement_context', 'software', 'provenance',
                'cache_metadata_sha256', 'backend', 'units',
                'reference_count_convention', 'radial_convention')})
        except (KeyError, TypeError) as exc:
            raise CountProductError('missing reference compatibility metadata') from exc
        if not isinstance(run.run_hash, str) or re.fullmatch('[0-9a-f]{64}', run.run_hash) is None:
            raise CountProductError('a validated source run hash is required')
        if common is None:
            common, theta, radial_edges = shared, counts.edges, run.radial_edges
        if (common != shared or not np.array_equal(theta, counts.edges)
                or not np.array_equal(radial_edges, run.radial_edges)):
            raise CountProductError('reference science context, source or grids differ')
        for edges in (counts.edges, run.radial_edges):
            if (not isinstance(edges, np.ndarray) or edges.ndim != 1 or len(edges) < 2
                    or edges.flags.writeable or edges.dtype != np.dtype('float64')
                    or not np.isfinite(edges).all() or np.any(np.diff(edges) <= 0)):
                raise CountProductError('invalid immutable angular/radial edges')
        if (theta[0] <= 0 or theta[-1] > np.pi
                or not np.array_equal(radial_edges[[0,-1]], m['measurement_context']['slice_edges'])
                or counts.backend != m['backend'] or counts.units != 'radian'
                or counts.weight_precision != 'float64' or counts.weight_rounding_allowed
                or counts.pair_convention != 'ordered_cross; [low, high); scalar_positive_weights'
                or run.weighted_z is not None or run.weighted_z2 is not None):
            raise CountProductError('reference count conventions or slice edges differ')
        left, right = _endpoint(run, 'left'), _endpoint(run, 'right')
        pairs = counts.patch_pairs
        if not isinstance(pairs, np.ndarray) or pairs.ndim != 2 or pairs.shape[1] != 2:
            raise CountProductError('invalid pair-row shape')
        _array(pairs, (len(pairs),2), integer=True)
        _array(counts.npairs, (len(pairs),len(theta)-1), integer=True)
        _array(counts.weighted, (len(pairs),len(theta)-1))
        if (any(int(p) not in counts.weight_sums_left or int(q) not in counts.weight_sums_right for p,q in pairs)
                or (len(pairs)>1 and np.any((pairs[1:,0]<pairs[:-1,0])
                    | ((pairs[1:,0]==pairs[:-1,0]) & (pairs[1:,1]<=pairs[:-1,1]))))
                or np.any((counts.npairs==0)!=(counts.weighted==0))):
            raise CountProductError('pair rows are unsorted, duplicated, unknown or inconsistent')
        if kind in ('DD','RR'):
            _same_endpoint(left,right)
        endpoints.append((left,right))
    _same_endpoint(endpoints[0][0],endpoints[1][0])
    _same_endpoint(endpoints[2][0],endpoints[1][1])
    return endpoints


def _moments_by_region(endpoint, regions):
    cells, moments, _, _ = endpoint
    return np.asarray([regions[int(c)] for c in cells],dtype=np.int64), moments


def _totals(endpoint, omitted):
    labels, moments = endpoint
    keep = np.ones(len(labels),dtype=bool) if omitted is None else labels != omitted
    kept = moments[keep].sum(axis=0)
    removed = moments[~keep].sum(axis=0)
    return kept[1], removed[1], kept[2]


def reconstruct_reference_auto_jackknife(dd, dr, rr, *, cell_regions,
                                         pair_weighting='match', max_result_bytes):
    """Reconstruct one tracer/cap/slice, full plus every common-region deletion.

    Durable readers remain responsible for source checksum verification. This
    pure helper validates live product semantics, compatibility and immutable
    numerical arrays; a run_hash alone is not a checksum proof for live arrays.

    max_result_bytes bounds owned numerical outputs plus conservative live
    contraction/validation arrays. It excludes caller-owned input arrays and
    Python metadata/runtime overhead. No region-by-pair-by-angle cube is built.
    Extra map regions empty in this slice are preserved and enter match gamma.
    Unsupported normalization or zero RR yields NaN, with no clipping/floor.
    """
    if pair_weighting not in ('literal','match'):
        raise ValueError('pair_weighting must be literal or match')
    if not _label(max_result_bytes) or max_result_bytes == 0:
        raise ValueError('max_result_bytes must be an explicit positive integer')
    regions = {}
    for cell, region in cell_regions.items():
        if not _label(cell) or cell >= 3072 or not _label(region):
            raise ValueError('map requires NESTED16 cells and nonnegative integer regions')
        regions[int(cell)] = int(region)
    region_ids = tuple(sorted(set(regions.values())))
    if len(region_ids)<2:
        raise ValueError('jackknife requires at least two common regions')
    runs = (dd,dr,rr)
    if any(not isinstance(run,ReferenceAutoRun) for run in runs):
        raise TypeError('three validated ReferenceAutoRun products are required')
    nreal, bins = len(region_ids)+1, len(dd.counts.edges)-1
    if bins <= 0:
        raise CountProductError('nonempty angular bins required')
    maxrows = max(len(run.counts.patch_pairs) for run in runs)
    cells = sum(len(getattr(run.counts,'weight_sums_'+side)) for run in runs for side in ('left','right'))
    # Two count tensors (raw and normalized), three norms, value/support;
    # region-pair labels, one scale vector, transient validation masks, totals.
    needed = (nreal*(57*bins+32) + (3*bins+64)*maxrows
              + (3*(len(dd.radial_edges)-1)+64)*cells + 64*bins
              + 16*(len(dd.counts.edges)+len(dd.radial_edges)) + 65536)
    if needed > max_result_bytes:
        raise ValueError('reference LS result exceeds explicit max_result_bytes bound')
    endpoints = _validate_runs(runs)
    occupied = {int(c) for pair in endpoints for endpoint in pair for c in endpoint[0]}
    if not occupied.issubset(regions):
        raise ValueError('common region map is missing an occupied cell')
    theta = dd.counts.edges.copy()
    radial = dd.radial_edges.copy()
    omitted_regions = (None,*region_ids)
    gamma = 1-len(region_ids)/(2+np.sqrt(2)*(len(region_ids)-1)) if pair_weighting=='match' else 0.
    if not np.isfinite(gamma) or not 0<=gamma<=1:
        raise ValueError('invalid match pair factor')
    data = _moments_by_region(endpoints[0][0],regions)
    random = _moments_by_region(endpoints[2][0],regions)
    raw = np.empty((nreal,3,bins))
    norms = np.empty((nreal,3))
    normalized = np.full_like(raw,np.nan)
    value = np.full((nreal,bins),np.nan)
    supported = np.zeros((nreal,bins),dtype=bool)
    for j,omitted in enumerate(omitted_regions):
        dk,do,qd = _totals(data,omitted)
        rk,ro,qr = _totals(random,omitted)
        norms[j] = (dk*dk+2*gamma*dk*do-qd,
                    dk*rk+gamma*(do*rk+dk*ro),
                    rk*rk+2*gamma*rk*ro-qr)
    if not np.isfinite(norms).all():
        raise ArithmeticError('reference all-pair normalization overflowed')
    for k,run in enumerate(runs):
        labels = np.asarray([(regions[int(p)],regions[int(q)]) for p,q in run.counts.patch_pairs],
                            dtype=np.int64).reshape(-1,2)
        for j,omitted in enumerate(omitted_regions):
            if omitted is None:
                raw[j,k] = run.counts.weighted.sum(axis=0)
            else:
                n_omitted = (labels[:,0]==omitted).astype(np.int8)+(labels[:,1]==omitted)
                scale = np.choose(n_omitted,(1.,gamma,0.))
                # Contract rows directly; never allocate pair-by-angle products.
                raw[j,k] = np.einsum('i,ij->j',scale,run.counts.weighted,optimize=False)
            if norms[j,k]>0:
                normalized[j,k] = raw[j,k]/norms[j,k]
    for j in range(nreal):
        if not np.isfinite(raw[j]).all():
            raise ArithmeticError('reference angular count contraction overflowed')
        supported[j] = ((norms[j]>0).all() & (raw[j,2]>0)
                        & np.isfinite(normalized[j]).all(axis=0) & (normalized[j,2]>0))
        numerator = normalized[j,0]-2*normalized[j,1]+normalized[j,2]
        np.divide(numerator,normalized[j,2],out=value[j],where=supported[j])
        if np.any(supported[j] & ~np.isfinite(value[j])):
            raise ArithmeticError('supported reference LS value overflowed')
    for a in (theta,radial,raw,norms,normalized,value,supported):
        a.setflags(write=False)
    return ReferenceAutoJackknife(runs,theta,radial,region_ids,omitted_regions,
        tuple(sorted(regions.items())),raw,norms,normalized,value,supported,pair_weighting,
        gamma if pair_weighting=='match' else None,tuple(run.run_hash for run in runs),needed)
