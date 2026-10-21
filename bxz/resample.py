"""Common-cell scalar cross resampling, without choosing a coverage prescription.

The default multiplicity path supports literal bootstrap and delete-one checks.
A separate primitive implements TreeCorr's matched jackknife pair weighting and
normalization. Neither selects covariance coverage or reference-auto/PIP fits.
"""
from dataclasses import dataclass
from numbers import Integral

import numpy as np


@dataclass(frozen=True)
class CrossResample:
    weighted: np.ndarray
    normalization: float
    weight_sum_left: float
    weight_sum_right: float
    weighted_z: np.ndarray | None
    weighted_z2: np.ndarray | None


def resample_cross(counts, cell_multiplicities, *, weighted_z=None, weighted_z2=None):
    """Apply one footprint-wide map, even when occupied cells differ per side.

    Every occupied cell must be named explicitly. Extra unoccupied cells in the
    common footprint are allowed. Call the function with the same map for every
    slice and count type. Returned totals rebuild estimator normalizations.
    Optional redshift moments use exactly the same pair weights as the counts.
    """
    mult = {}
    for cell, value in cell_multiplicities.items():
        if isinstance(cell, (bool, np.bool_)) or not isinstance(cell, Integral) or cell < 0:
            raise ValueError("cell labels must be nonnegative integers")
        if not np.isscalar(value) or not np.isfinite(value) or value < 0:
            raise ValueError("multiplicities must be finite nonnegative scalars")
        mult[int(cell)] = float(value)
    occupied = set(counts.weight_sums_left) | set(counts.weight_sums_right)
    if not occupied.issubset(mult):
        raise ValueError("common map is missing an occupied cell")
    left = {p: mult[p] for p in counts.weight_sums_left}
    right = {p: mult[p] for p in counts.weight_sums_right}
    weighted, norm = counts.resample(left, right)
    sums = [sum(totals[p] * mult[p] for p in totals)
            for totals in (counts.weight_sums_left, counts.weight_sums_right)]
    scale = np.array([mult[int(p)] * mult[int(q)] for p, q in counts.patch_pairs])
    moments = []
    if (weighted_z is None) != (weighted_z2 is None):
        raise ValueError("supply both redshift moments or neither")
    for moment in (weighted_z, weighted_z2):
        if moment is None:
            moments.append(None)
            continue
        array = np.asarray(moment, dtype=float)
        if array.shape != counts.weighted.shape or not np.isfinite(array).all() or np.any(array < 0):
            raise ValueError("redshift moments must match the finite patch-pair count shape")
        moments.append((scale[:, None] * array).sum(axis=0))
    return CrossResample(weighted, norm, *sums, *moments)


@dataclass(frozen=True)
class CrossMatchResample:
    """Matched pairs with a normalization that does not factor into sums.

    ``retained_weight_sum_left/right`` are the ordinary endpoint totals outside
    the omitted region. Their product is NOT ``pair_normalization``: matched
    cross-region pairs still include some endpoints in the omitted region.
    This deliberately differs from CrossResample and is not a drop-in input to
    the existing literal DP-global reconstruction. The separate DP-global
    jackknife path explicitly uses retained photometric endpoint totals.
    """

    weighted: np.ndarray
    pair_normalization: float
    retained_weight_sum_left: float
    retained_weight_sum_right: float
    weighted_z: np.ndarray | None
    weighted_z2: np.ndarray | None
    region_count: int
    omitted_region: int
    cross_region_weight: float


def resample_cross_match(counts, *, cell_regions, omitted_region,
                         weighted_z=None, weighted_z2=None):
    """Apply TreeCorr-style match weighting to exactly one omitted region.

    ``cell_regions`` explicitly maps every storage cell in the common footprint
    to a nonnegative integer region ID. It must include all occupied cells on
    both sides; additional cells/regions empty in this count are allowed and
    contribute to the shared region count N. Use the same complete map and
    omitted region for every slice, tracer and count type. Storage cells sharing
    a region are omitted together, even when their cell labels differ.

    With w = 1 - N / (2 + sqrt(2) * (N - 1)), pairs have weight zero when both
    endpoints are in the omitted region, w when exactly one is there, and one
    otherwise. The identical rule scales both redshift moments. Only delete-one
    jackknife is supported; there is no bootstrap or multiplicity argument.

    The normalization is sum_ij f_ij W_left,i W_right,j over ALL cell pairs,
    including those absent from the angular count rows. This follows TreeCorr
    5.1.4 Corr2.JackknifePairIterator and NNCorrelation._sum/_add_tot:
    https://rmjarvis.github.io/TreeCorr/_build/html/_modules/treecorr/corr2base.html
    https://rmjarvis.github.io/TreeCorr/_build/html/_modules/treecorr/nncorrelation.html
    It is not the product of retained endpoint sums. This scalar primitive
    does not choose a region construction or covariance coverage, and does not
    support auto/PIP normalization. DP-global's separate jackknife path uses
    the retained photometric endpoint totals rather than pair_normalization.
    """
    def label(value):
        return (not isinstance(value, (bool, np.bool_))
                and isinstance(value, Integral) and value >= 0)

    regions = {}
    for cell, region in cell_regions.items():
        if not label(cell) or not label(region):
            raise ValueError("cell and region labels must be nonnegative integers")
        regions[int(cell)] = int(region)
    occupied = set(counts.weight_sums_left) | set(counts.weight_sums_right)
    if not occupied.issubset(regions):
        raise ValueError("common region map is missing an occupied cell")
    region_ids = set(regions.values())
    if len(region_ids) < 2:
        raise ValueError("match jackknife requires at least two common regions")
    if not label(omitted_region) or omitted_region not in region_ids:
        raise ValueError("omitted_region must name exactly one common region")
    omitted_region = int(omitted_region)
    cross_weight = 1 - len(region_ids) / (2 + np.sqrt(2) * (len(region_ids) - 1))
    scale = []
    for left, right in counts.patch_pairs:
        if left not in regions or right not in regions:
            raise ValueError("common region map is missing a pair-row cell")
        n_omitted = int(regions[left] == omitted_region) + int(regions[right] == omitted_region)
        scale.append((1., cross_weight, 0.)[n_omitted])
    scale = np.asarray(scale, dtype=float)
    weighted = (scale[:, None] * counts.weighted).sum(axis=0)

    totals = []
    for weights in (counts.weight_sums_left, counts.weight_sums_right):
        if any(not np.isfinite(value) or value < 0 for value in weights.values()):
            raise ValueError("cell weight sums must be finite and nonnegative")
        kept = sum(value for cell, value in weights.items() if regions[cell] != omitted_region)
        removed = sum(value for cell, value in weights.items() if regions[cell] == omitted_region)
        totals.append((kept, removed))
    (left_kept, left_removed), (right_kept, right_removed) = totals
    norm = (left_kept * right_kept
            + cross_weight * (left_removed * right_kept + left_kept * right_removed))
    if not np.isfinite(norm):
        raise ValueError("matched pair normalization overflowed")
    if (weighted_z is None) != (weighted_z2 is None):
        raise ValueError("supply both redshift moments or neither")
    moments = []
    for moment in (weighted_z, weighted_z2):
        if moment is None:
            moments.append(None)
            continue
        array = np.asarray(moment, dtype=float)
        if array.shape != counts.weighted.shape or not np.isfinite(array).all() or np.any(array < 0):
            raise ValueError("redshift moments must match the finite patch-pair count shape")
        moments.append((scale[:, None] * array).sum(axis=0))
    return CrossMatchResample(weighted, norm, left_kept, right_kept, *moments,
                              len(region_ids), omitted_region, cross_weight)
