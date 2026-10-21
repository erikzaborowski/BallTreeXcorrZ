"""Exact angular cross counts with explicit patch partitions.

Coordinates and edges are radians. Pairs are ordered between catalogues: passing
the same catalogue on both sides counts both orientations, and a positive lower
edge excludes self pairs. Bins are [low, high), including the final bin. This
differs from numpy.histogram only for pairs exactly on the final upper edge.
This spike accepts positive scalar weights; assignment/PIP weights are absent.
"""
from dataclasses import dataclass
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class AngularCatalog:
    ra: np.ndarray
    dec: np.ndarray
    weight: np.ndarray
    patch: np.ndarray

    def __post_init__(self):
        for name in ("ra", "dec", "weight"):
            value = np.asarray(getattr(self, name), dtype=np.float64)
            if value.ndim != 1 or not np.isfinite(value).all():
                raise ValueError(f"{name} must be a finite one-dimensional array")
            object.__setattr__(self, name, value)
        patch = np.asarray(self.patch)
        if patch.ndim != 1 or patch.dtype.kind not in "iu":
            raise ValueError("patch must be a one-dimensional integer array")
        object.__setattr__(self, "patch", patch)
        if len({len(self.ra), len(self.dec), len(self.weight), len(patch)}) != 1:
            raise ValueError("catalogue columns must have equal lengths")
        if np.any(abs(self.dec) > np.pi / 2) or np.any(self.weight <= 0):
            raise ValueError("declination must be in [-pi/2, pi/2]; weights must be positive")

    def subset(self, selection):
        return AngularCatalog(*(getattr(self, name)[selection]
                                for name in ("ra", "dec", "weight", "patch")))


@dataclass(frozen=True)
class PatchCounts:
    edges: np.ndarray
    patch_pairs: np.ndarray
    npairs: np.ndarray
    weighted: np.ndarray
    weight_sums_left: Mapping[int, float]
    weight_sums_right: Mapping[int, float]
    backend: str
    weight_precision: str
    weight_rounding_allowed: bool = False
    units: str = "radian"
    pair_convention: str = "ordered_cross; [low, high); scalar_positive_weights"

    @property
    def total_npairs(self):
        return self.npairs.sum(axis=0)

    @property
    def total_weighted(self):
        return self.weighted.sum(axis=0)

    def resample(self, left=None, right=None):
        """Recombine cross counts and recompute W_left * W_right.

        Multiplicities are keyed by storage-cell label; omitted labels are zero.
        None retains every label once. Common realizations must be supplied on
        both sides by the caller. These are cross, never auto/PIP normalizations.
        """
        def factors(weights, requested):
            result = {p: 1. if requested is None else float(requested.get(p, 0.))
                      for p in weights}
            if any(not np.isfinite(x) or x < 0 for x in result.values()):
                raise ValueError("resampling multiplicities must be finite and nonnegative")
            if requested is not None and set(requested) - set(weights):
                raise ValueError("unknown patch label in resampling multiplicities")
            return result
        ml = factors(self.weight_sums_left, left)
        mr = factors(self.weight_sums_right, right)
        scale = np.asarray([ml[int(p)] * mr[int(q)] for p, q in self.patch_pairs])
        weighted = (self.weighted * scale[:, None]).sum(axis=0)
        wl = sum(self.weight_sums_left[p] * ml[p] for p in ml)
        wr = sum(self.weight_sums_right[p] * mr[p] for p in mr)
        return weighted, wl * wr


def _pycorr(left, right, edges):
    """Scalar ordered cross counts through pycorr's Corrfunc engine.

    Always pass both catalogues, even for a same-catalogue call: this interface
    retains cross normalization. Native auto normalization is tested separately.
    Explicit float64, custom bins and standard acos avoid implicit precision or
    binning choices. No bitwise weights or assignment recipes are enabled here.
    """
    from pycorr import TwoPointCounter

    def positions(catalog):
        return [np.rad2deg(catalog.ra) % 360., np.rad2deg(catalog.dec)]

    result = TwoPointCounter(
        "theta", np.rad2deg(edges), positions1=positions(left),
        positions2=positions(right), weights1=left.weight, weights2=right.weight,
        engine="corrfunc", position_type="rd", weight_type="product_individual",
        dtype="f8", bin_type="custom", compute_sepsavg=False, nthreads=1,
        fast_acos=False,
    )
    return result.ncounts.copy(), result.wcounts.copy()


def count_angular(left, right, edges, *, backend="pycorr", allow_weight_rounding=False):
    """Count occupied cell pairs using pycorr/Corrfunc and float64 weights."""
    edges = np.asarray(edges, dtype=np.float64)
    if (edges.ndim != 1 or len(edges) < 2 or not np.isfinite(edges).all()
            or edges[0] <= 0 or edges[-1] > np.pi or np.any(np.diff(edges) <= 0)):
        raise ValueError("edges must increase strictly within (0, pi]")
    if backend != "pycorr" or allow_weight_rounding:
        raise ValueError("production counts require pycorr float64 scalar weights")
    lp, rp = np.unique(left.patch), np.unique(right.patch)
    lc = {int(p): left.subset(left.patch == p) for p in lp}
    rc = {int(p): right.subset(right.patch == p) for p in rp}
    pairs, counts, weighted = [], [], []
    counter = _pycorr
    for p, lcat in lc.items():
        for q, rcat in rc.items():
            n, w = counter(lcat, rcat, edges)
            pairs.append((p, q))
            counts.append(n)
            weighted.append(w)
    shape = (len(pairs), len(edges) - 1)
    return PatchCounts(edges.copy(), np.asarray(pairs, dtype=np.int64).reshape(-1, 2),
                       np.asarray(counts, dtype=np.int64).reshape(shape),
                       np.asarray(weighted, dtype=float).reshape(shape),
                       {p: float(c.weight.sum()) for p, c in lc.items()},
                       {p: float(c.weight.sum()) for p, c in rc.items()}, backend,
                       "float64", False)
