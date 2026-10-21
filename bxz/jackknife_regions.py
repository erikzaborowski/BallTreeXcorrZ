"""Whole-storage-cell, weighted Cartesian k-means (P3, plan §7.4).

The objective uses squared Euclidean distances between unit Cartesian object
positions and unrestricted Cartesian centers, subject to assigning every object
in a storage cell to the same region. For W_c=sum(w), S_c=sum(w*x), Q_c=sum(w*|x|²),
the cell contribution is W_c*|S_c/W_c-center|² + Q_c-|S_c|²/W_c.
Thus weighted k-means of the *unnormalized* cell means is exactly equivalent to
this constrained object objective, up to fixed within-cell scatter. Neither
cell means nor fitted centers are normalized to the sphere. This is not the
unconstrained object k-means problem, and implies no equal areas, contiguity,
pilot suitability, covariance coverage, or globally optimal solution.

Only supplied arrays are consumed; no catalogue or mask files are opened.
"""
from dataclasses import dataclass

import numpy as np

from .patches import CellRegionMapping, _labels, _nside


def _readonly(value):
    result = np.array(value, copy=True)
    result.setflags(write=False)
    return result


def _integer(value, name, *, minimum=1, maximum=None):
    if (isinstance(value, (bool, np.bool_)) or
            not isinstance(value, (int, np.integer)) or value < minimum or
            (maximum is not None and value > maximum)):
        raise ValueError(f"{name} must be an integer in the allowed range")
    return int(value)


def _real(value, shape, name):
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must be real")
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"{name} must have shape {shape} and finite values")
    return result


@dataclass(frozen=True)
class SkyCellStatistics:
    """Immutable sorted sufficient statistics for weighted unit-sky objects.

    ``count`` includes zero-weight objects. ``weighted_norm2_sum`` uses the
    exact unit-vector convention Q=W, rather than numerically squaring the
    trigonometric coordinates. Empty and zero-weight cells may be represented;
    fitting requires their explicit removal by the caller.

    Moment consistency permits only float64 roundoff: gamma_(2*N+20) times
    (Q + W*|S/W|²), with gamma_k=k*eps/(1-k*eps). This conservative absolute
    bound covers up to N contributions accumulated within batches and again
    across batches, plus Cartesian arithmetic and scatter evaluation. It is a
    bound for ordinary float64 sums of supplied weighted unit vectors, not a
    tolerance for physical departures from those moments. k*eps >= 1/2 is
    rejected. Negative scatter within that bound is clipped only when reporting
    scatter; inconsistent moments outside it are rejected.
    """

    cells: np.ndarray
    count: np.ndarray
    weight_sum: np.ndarray
    weighted_position_sum: np.ndarray
    weighted_norm2_sum: np.ndarray
    nside_store: int

    def __post_init__(self):
        nside = _nside(self.nside_store)
        cells = _labels(self.cells, "cells", upper=12*nside**2)
        count = _labels(self.count, "count")
        if count.shape != cells.shape:
            raise ValueError("count must align with cells")
        weight = _real(self.weight_sum, cells.shape, "weight_sum")
        position = _real(self.weighted_position_sum, (len(cells), 3), "weighted_position_sum")
        norm2 = _real(self.weighted_norm2_sum, cells.shape, "weighted_norm2_sum")
        if np.any(weight < 0) or np.any(norm2 != weight):
            raise ValueError("unit-sky moments require nonnegative W and Q=W")
        if np.any((count == 0) & (weight != 0)) or np.any(position[weight == 0] != 0):
            raise ValueError("empty or zero-weight cells must have zero moments")
        order = np.argsort(cells)
        if np.any(np.diff(cells[order]) == 0):
            raise ValueError("duplicate storage cell")
        for name, values in (("cells", cells), ("count", count), ("weight_sum", weight),
                             ("weighted_position_sum", position), ("weighted_norm2_sum", norm2)):
            object.__setattr__(self, name, _readonly(values[order]))
        object.__setattr__(self, "nside_store", nside)
        self.scatter()

    def scatter(self):
        """Return per-cell fixed scatter and its absolute roundoff bound."""
        positive = self.weight_sum > 0
        mean = np.zeros_like(self.weighted_position_sum)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            mean[positive] = self.weighted_position_sum[positive] / self.weight_sum[positive, None]
            term = self.weight_sum * np.sum(mean*mean, axis=1)
            k_eps = (2*self.count.astype(float) + 20) * np.finfo(float).eps
            bound = (k_eps/(1-k_eps)) * (self.weighted_norm2_sum + term)
            scatter = self.weighted_norm2_sum - term
        if (np.any(k_eps >= .5) or not np.isfinite(term).all() or
                not np.isfinite(bound).all() or np.any(scatter < -bound)):
            raise ValueError("physically inconsistent unit-sky moments or unbounded float64 error")
        return _readonly(np.maximum(scatter, 0)), _readonly(bound)


def accumulate_sky_cells(batches, *, units, nside_store, max_cells,
                         max_batch_objects, max_objects):
    """Accumulate supplied (cell, RA, Dec, weight) batches with explicit bounds.

    Coordinates are equatorial. Labels must already use the supplied NESTED
    storage scheme; this routine validates their range, not angular membership.
    Nonempty batches and finite nonnegative weights are required. Memory scales
    with max_cells plus max_batch_objects, never the largest pixel label.
    """
    nside = _nside(nside_store)
    max_cells = _integer(max_cells, "max_cells")
    max_batch_objects = _integer(max_batch_objects, "max_batch_objects")
    max_objects = _integer(max_objects, "max_objects")
    if units not in ("degrees", "radians"):
        raise ValueError("units must be degrees or radians")
    totals, processed = {}, 0
    for cells, ra, dec, weights in batches:
        n = len(cells)
        if n == 0 or n > max_batch_objects or processed + n > max_objects:
            raise ValueError("batch is empty or exceeds explicit object bounds")
        cells = _labels(cells, "cells", upper=12*nside**2)
        ra, dec = _real(ra, (n,), "ra"), _real(dec, (n,), "dec")
        weights = _real(weights, (n,), "weights")
        pole, period = (90., 360.) if units == "degrees" else (np.pi/2, 2*np.pi)
        if np.any(np.abs(dec) > pole) or np.any(weights < 0):
            raise ValueError("declination outside poles or negative weights")
        ra = ra % period
        if units == "degrees":
            ra, dec = np.deg2rad(ra), np.deg2rad(dec)
        xyz = np.column_stack((np.cos(dec)*np.cos(ra), np.cos(dec)*np.sin(ra), np.sin(dec)))
        labels, inverse = np.unique(cells, return_inverse=True)
        if len(totals.keys() | set(map(int, labels))) > max_cells:
            raise ValueError("distinct storage cells exceed max_cells")
        counts = np.bincount(inverse)
        with np.errstate(over="ignore", invalid="ignore"):
            weight_sum = np.bincount(inverse, weights=weights)
            position_sum = np.column_stack([np.bincount(inverse, weights=weights*xyz[:, j]) for j in range(3)])
            for i, label in enumerate(labels):
                previous = totals.get(int(label), (0, 0., np.zeros(3)))
                totals[int(label)] = (previous[0]+int(counts[i]), previous[1]+weight_sum[i],
                                      previous[2]+position_sum[i])
        processed += n
    if not totals:
        raise ValueError("at least one supplied object is required")
    cells = sorted(totals)
    count = [totals[c][0] for c in cells]
    weight = [totals[c][1] for c in cells]
    position = [totals[c][2] for c in cells]
    return SkyCellStatistics(cells, count, weight, position, weight, nside)


@dataclass(frozen=True)
class CellRegionFit:
    mapping: CellRegionMapping
    centers: np.ndarray
    region_weight_sum: np.ndarray
    region_object_count: np.ndarray
    region_cell_count: np.ndarray
    cell_mean_inertia: float
    within_cell_scatter: float
    object_inertia: float
    scatter_roundoff_bound: float
    iterations: int
    iteration_limit_reached: bool
    seed: int
    n_init: int
    max_iter: int
    tol: float
    max_cells: int
    sklearn_version: str


def fit_cell_regions(statistics, *, n_regions, seed, n_init, max_iter, tol, max_cells):
    """Fit weighted Lloyd k-means of cell means, with no production defaults.

    Canonical labels increase with each region's smallest storage-cell ID.
    Centers remain exactly those returned by sklearn, including if iteration
    limits stopped fitting. ``iterations`` and ``iteration_limit_reached`` are
    stopping diagnostics, not certificates of convergence or global optimality.
    ``tol`` is sklearn's relative center-shift tolerance. Input size, number of
    regions, restarts and iterations are all explicitly bounded by the caller.
    Reference-only cells remain unmapped; there is no implicit assignment rule.
    """
    if not isinstance(statistics, SkyCellStatistics):
        raise TypeError("statistics must be SkyCellStatistics")
    max_cells = _integer(max_cells, "max_cells")
    n_regions = _integer(n_regions, "n_regions")
    seed = _integer(seed, "seed", minimum=0, maximum=2**32-1)
    n_init = _integer(n_init, "n_init")
    max_iter = _integer(max_iter, "max_iter")
    if isinstance(tol, (bool, np.bool_)) or np.ndim(tol) != 0 or not np.isfinite(tol) or tol < 0:
        raise ValueError("tol must be finite and nonnegative")
    ncells = len(statistics.cells)
    if not n_regions <= ncells <= max_cells:
        raise ValueError("n_regions <= number of cells <= max_cells is required")
    if np.any(statistics.weight_sum <= 0):
        raise ValueError("every fitted cell must have positive random weight; select support explicitly")
    means = statistics.weighted_position_sum / statistics.weight_sum[:, None]
    if len(np.unique(means, axis=0)) < n_regions:
        raise ValueError("fewer distinct cell means than requested regions")
    if not np.isfinite(np.sum(statistics.weight_sum)):
        raise ValueError("total random weight overflows float64")
    import sklearn
    from sklearn.cluster import KMeans

    model = KMeans(n_clusters=n_regions, init="k-means++", n_init=n_init,
                   max_iter=max_iter, tol=float(tol), random_state=seed, algorithm="lloyd")
    model.fit(means, sample_weight=statistics.weight_sum)
    labels = model.labels_
    if len(np.unique(labels)) != n_regions:
        raise ValueError("k-means did not populate every requested region")
    order = np.argsort([statistics.cells[labels == i].min() for i in range(n_regions)])
    inverse = np.empty(n_regions, dtype=int)
    inverse[order] = np.arange(n_regions)
    labels = inverse[labels]
    centers = model.cluster_centers_[order]
    region_weight = np.bincount(labels, weights=statistics.weight_sum, minlength=n_regions)
    # Preserve integer object counts without converting them to float weights.
    region_count = np.array([sum(map(int, statistics.count[labels == i])) for i in range(n_regions)], dtype=np.int64)
    region_cells = np.bincount(labels, minlength=n_regions)
    scatter, bound = statistics.scatter()
    with np.errstate(over="ignore", invalid="ignore"):
        inertia = float(np.sum(statistics.weight_sum * np.sum((means-centers[labels])**2, axis=1)))
        scatter_total = float(np.sum(scatter))
        total = inertia + scatter_total
        bound_total = float(np.sum(bound))
    if not np.isfinite([inertia, scatter_total, total, bound_total]).all():
        raise ValueError("fitted objective overflows float64")
    return CellRegionFit(CellRegionMapping(statistics.cells, labels, statistics.nside_store),
                         *map(_readonly, (centers, region_weight, region_count, region_cells)),
                         inertia, scatter_total, total, bound_total, int(model.n_iter_),
                         bool(model.n_iter_ >= max_iter), seed, n_init, max_iter, float(tol),
                         max_cells, sklearn.__version__)
