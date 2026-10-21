"""Diagnostic DP-reference replay (P3, plan sections 4.2 and 4.3).

This explicitly chosen cap/slice combination is not the DP-global baseline:
U[k,i] = sum_c W_Ds[c,k]/W_Rs[c,k] * RD[c,k,i], then omega-weighted
DD/U - 1. Its windows and redshift moments derive from U, never DR. No data
loading, counting, covariance estimation, theory conversion or fitting occurs.
"""
from dataclasses import dataclass
from numbers import Integral
import re

import numpy as np

from .counts_io import CountProductError, canonical_json
from .estimators import Estimate, _ratio
from .measurement import _catalogue, _same
from .operator import BandEstimate, MeasurementOperator
from .paircount.driver import CountRun
from .resample import resample_cross, resample_cross_match


COMBINATION = "diagnostic_cap_slice_reference_scaled_RD"
CAPS = ("NGC", "SGC")


@dataclass(frozen=True)
class ReferenceMeasurement:
    operator: MeasurementOperator
    bands: BandEstimate
    integrated: Estimate
    native_dd_by_cap: np.ndarray  # (slice, cap, angle)
    native_rd_by_cap: np.ndarray
    native_rd_z_by_cap: np.ndarray
    native_rd_z2_by_cap: np.ndarray
    weight_data_s: np.ndarray  # (slice, cap), retained endpoint totals
    weight_random_s: np.ndarray
    z_eff: np.ndarray
    z2_eff: np.ndarray
    run_hashes: tuple  # slice, cap, (DD, RD)
    pair_weighting: str
    omitted_region: int | None
    cell_regions: tuple
    cross_region_weight: float | None
    combination: str = COMBINATION
    normalization: str = "retained_reference_endpoint_totals_per_cap_per_native_slice"
    cap_order: tuple = CAPS


@dataclass(frozen=True)
class ReferenceJackknife:
    full: ReferenceMeasurement
    deletions: tuple[ReferenceMeasurement, ...]
    region_ids: tuple
    cell_regions: tuple
    pair_weighting: str
    cross_region_weight: float | None
    combination: str = COMBINATION


def _science(run):
    """Ignore only declared implementation/estimator labels, not science inputs."""
    m = run.metadata
    p = m["provenance"]
    config = {k: v for k, v in p["config"].items() if k not in ("estimator", "name")}
    context = {k: v for k, v in m["measurement_context"].items()
               if k not in ("slice_edges", "chi_hmpc")}
    return {"cache": m["cache_metadata_sha256"], "photo_group": m["photo_group"],
            "context": context, "config": config, "population_id": p["population_id"],
            "catalogues": p["catalogues"], "input_checksums": p["input_checksums"],
            "backend": m["backend"], "units": m["units"],
            "pair_convention": m["pair_convention"], "photo": _catalogue(run, "right"),
            "count_convention": {key: getattr(run.counts, key) for key in
                ("backend", "units", "pair_convention", "weight_precision", "weight_rounding_allowed")}}


def _pairs(runs):
    """Validate one-pass slice-major NGC/SGC pairs; retain metadata only."""
    baseline = theta = previous_hi = slice_context = slice_id = tracer = None
    seen = set()
    index = 0
    for index, pair in enumerate(runs, 1):
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise CountProductError("each input must be a (DD, RD) pair")
        dd, rd = pair
        if any(not isinstance(run, CountRun) for run in pair):
            raise TypeError("expected CountRun inputs")
        cap = CAPS[(index - 1) % 2]
        for run, typ, kind in ((dd, "DD", "data"), (rd, "RD", "randoms")):
            m = run.metadata
            group = m.get("reference_group", "")
            match = re.fullmatch(r"(BGS_BRIGHT|LRG|ELG_LOPnotqso|QSO)_(NGC|SGC)_(data|randoms)", group)
            if not match or (match[2], match[3]) != (cap, kind):
                raise CountProductError("expected ordered NGC/SGC reference data/random groups")
            if tracer is None:
                tracer = match[1]
            if match[1] != tracer or m.get("count_type") != typ:
                raise CountProductError("tracer or count type differs")
            if m.get("photo_group") != "photo_data":
                raise CountProductError("both pair types require photometric data")
            if not re.fullmatch(r"[0-9a-f]{64}", m.get("cache_metadata_sha256", "")):
                raise CountProductError("cache manifest digest is required")
            try:
                science = canonical_json(_science(run))
            except KeyError as exc:
                raise CountProductError(f"missing compatibility field {exc}") from exc
            if baseline is None:
                baseline, theta = science, run.counts.edges.copy()
            if science != baseline:
                raise CountProductError("shared science context or photometric catalogue differs")
            if not np.array_equal(run.counts.edges, theta):
                raise CountProductError("native angular grids differ")
            for key in ("backend", "units", "pair_convention"):
                _same(m[key], getattr(run.counts, key), f"count metadata {key} differs")
            _catalogue(run, "left")
            for side in ("left", "right"):
                weights = getattr(run.counts, "weight_sums_" + side)
                moments = getattr(run, side + "_moments")
                if (moments.shape != (len(weights), 3) or not np.isfinite(moments).all()
                        or np.any(moments < 0)
                        or not np.array_equal(moments[:, 1], list(weights.values()))):
                    raise CountProductError("endpoint moments disagree with saved weight sums")
        for key in ("slice_id", "measurement_context"):
            _same(dd.metadata[key], rd.metadata[key], f"DD/RD {key} differs")
        for key in ("backend", "weight_precision", "weight_rounding_allowed", "units", "pair_convention"):
            _same(getattr(dd.counts, key), getattr(rd.counts, key), f"count {key} differs")
        current = dd.metadata["measurement_context"]
        if cap == "NGC":
            low, high = current["slice_edges"]
            if not np.isfinite([low, high]).all() or not 0 <= low < high:
                raise CountProductError("invalid native slice edges")
            if previous_hi is not None and low < previous_hi:
                raise CountProductError("fine slices overlap or are out of order")
            slice_id = dd.metadata["slice_id"]
            if slice_id in seen:
                raise CountProductError("duplicate native slice ID")
            seen.add(slice_id)
            previous_hi, slice_context = high, canonical_json(current)
        elif (dd.metadata["slice_id"] != slice_id or canonical_json(current) != slice_context):
            raise CountProductError("NGC/SGC slice geometry differs")
        if rd.weighted_z is None or rd.weighted_z2 is None:
            raise CountProductError("RD redshift moments are required")
        yield (index - 1) // 2, (index - 1) % 2, dd, rd
    if index == 0 or index % 2:
        raise CountProductError("both NGC and SGC are required for every slice")


def _assemble(op, arrays, weights, hashes, regions, omitted, weighting, cross_weight):
    dd, rd, rd_z, rd_z2 = arrays
    wd, wr = weights
    if not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("reference endpoint totals must be finite and positive in every cap/slice")
    alpha = wd / wr
    # No DP-global photometric normalizer or nonfactorizing pair normalization.
    u, uz, uz2 = [(x * alpha[:, :, None]).sum(axis=1) for x in (rd, rd_z, rd_z2)]
    numerator, _ = op._aggregate(op._native(dd.sum(axis=1)))
    denominator, window = op._random_window(op._native(u))
    ratio = _ratio(numerator, denominator)
    bands = BandEstimate(ratio.value - 1, ratio.supported, denominator, window)
    integrated = op.integrate(bands)
    summaries = []
    for moment in (uz, uz2):
        total, _ = op._aggregate(op._native(moment))
        mean = np.full(total.shape, np.nan)
        np.divide(total, denominator, out=mean, where=bands.supported)
        summaries.append(mean)
    for value in (*arrays, *weights, *summaries, bands.value, bands.supported,
                  bands.random_support, bands.slice_window, integrated.value, integrated.supported):
        value.setflags(write=False)
    return ReferenceMeasurement(op, bands, integrated, *arrays, *weights, *summaries,
        hashes, weighting, omitted, regions, cross_weight)


def reconstruct_dp_reference_jackknife(run_pairs, *, r_edges, output_bin,
        n_output_bins, cell_regions, pair_weighting, max_realizations, max_result_bytes):
    """Full + all deletions from one-pass (DD,RD) pairs, slice then NGC/SGC.

    Each CountRun has reference objects on the left and the identical complete
    photometric data catalogue on the right. Input readers remain responsible
    for durable payload verification. At most one pair is retained internally;
    all numerical outputs are reduced native arrays, not count rows. Inputs
    must include exactly two caps for every output_bin entry, for one tracer.

    The documented cap/slice combination is a diagnostic estimator choice;
    native one-cap values match plan section 4.2. Literal and match deletions
    both use ordinary retained reference endpoint totals per cap/slice. Match
    pair_normalization is deliberately unused. Zero reference totals reject;
    zero angular random support is NaN. Recompute responses using these U
    windows before kernel conversion; existing DR-based responses do not apply.

    max_result_bytes bounds retained numerical arrays plus shared geometry,
    excluding caller-owned CountRuns, metadata/Python overhead and per-pair
    resampling temporaries. No complete native aggregation copy is retained.
    No covariance coverage or physical-systematics claim is made.
    """
    for name, value in (("max_realizations", max_realizations),
            ("max_result_bytes", max_result_bytes), ("n_output_bins", n_output_bins)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if pair_weighting not in ("literal", "match"):
        raise ValueError("pair_weighting must be literal or match")
    regions = dict(cell_regions)
    for cell, region in regions.items():
        if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, Integral) or v < 0 for v in (cell, region)):
            raise ValueError("cell and region labels must be nonnegative integers")
    region_ids = tuple(sorted(set(regions.values())))
    if len(region_ids) < 2:
        raise ValueError("jackknife requires at least two common regions")
    nreal = len(region_ids) + 1
    if nreal > max_realizations:
        raise ValueError("jackknife exceeds max_realizations")
    labels = np.asarray(output_bin)
    if (labels.ndim != 1 or not len(labels) or labels.dtype.kind not in "iu"
            or np.any(labels < 0) or np.any(labels >= n_output_bins)):
        raise ValueError("output_bin must be nonempty valid integer labels")
    r = np.asarray(r_edges, dtype=float)
    if r.ndim != 1 or len(r) < 2 or not np.isfinite(r).all() or np.any(r <= 0) or np.any(np.diff(r) <= 0):
        raise ValueError("r_edges must be finite positive increasing edges")
    nslice, nband = len(labels), len(r) - 1
    omissions = (None,) + region_ids
    cross_weight = 1 - len(region_ids) / (2 + np.sqrt(2) * (len(region_ids)-1)) if pair_weighting == "match" else None
    native = weights = None
    chi, redshift, hashes = [], [], []
    for k, c, dd, rd in _pairs(run_pairs):
        if k >= nslice:
            raise CountProductError("run slices must align with output_bin")
        context = dd.metadata["measurement_context"]
        if native is None:
            theta = dd.counts.edges.copy()
            cuts = context["hard_cut_indices"]
            nangle = len(theta) - 1
            # Per-cap four arrays, two reference totals, per-realization bands
            # and windows; conservative shared geometry bound.
            bound = (nreal * (64 * nslice * nangle + 32 * nslice
                + 33 * n_output_bins * nband + 8 * n_output_bins * nband * nslice
                + 9 * n_output_bins) + 8 * nslice * nangle * nband
                + 128 * (nslice + nangle + nband + 1))
            if bound > max_result_bytes:
                raise ValueError("jackknife exceeds max_result_bytes")
            native = np.empty((nreal, 4, nslice, 2, nangle))
            weights = np.empty((nreal, 2, nslice, 2))
        if c == 0:
            chi.append(context["chi_hmpc"])
            redshift.append(context["slice_edges"][0])
            hashes.append([])
        hashes[k].append((dd.run_hash, rd.run_hash))
        for j, omitted in enumerate(omissions):
            kwargs = dict(weighted_z=rd.weighted_z, weighted_z2=rd.weighted_z2)
            if omitted is None or pair_weighting == "literal":
                selection = {cell: float(omitted is None or region != omitted) for cell, region in regions.items()}
                d = resample_cross(dd.counts, selection)
                rr = resample_cross(rd.counts, selection, **kwargs)
                totals = d.weight_sum_left, rr.weight_sum_left
            else:
                d = resample_cross_match(dd.counts, cell_regions=regions, omitted_region=omitted)
                rr = resample_cross_match(rd.counts, cell_regions=regions, omitted_region=omitted, **kwargs)
                totals = d.retained_weight_sum_left, rr.retained_weight_sum_left
            native[j, :, k, c] = d.weighted, rr.weighted, rr.weighted_z, rr.weighted_z2
            weights[j, :, k, c] = totals
    if len(chi) != nslice:
        raise CountProductError("run slices must align with output_bin")
    if not np.isfinite(native).all() or np.any(native < 0):
        raise ValueError("reduced counts must be finite and nonnegative")
    op = MeasurementOperator(theta, r, chi, redshift, labels, n_output_bins=n_output_bins,
        lower_edge_index=cuts[0], upper_edge_index=cuts[1])
    native.setflags(write=False)
    weights.setflags(write=False)
    hashes, region_items = tuple(tuple(row) for row in hashes), tuple(sorted(regions.items()))
    results = [_assemble(op, native[j], weights[j], hashes, region_items, omitted,
        "literal" if omitted is None else pair_weighting,
        cross_weight if omitted is not None else None) for j, omitted in enumerate(omissions)]
    return ReferenceJackknife(results[0], tuple(results[1:]), region_ids,
        region_items, pair_weighting, cross_weight)
