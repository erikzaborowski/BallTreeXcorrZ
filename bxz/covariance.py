"""Equal-region delete-one covariance infrastructure (plan §4.9, V11/V12).

Rows are measurements after deleting the indicated common region. Every row
must already include all required estimator recomputation and reference-bias
refitting. This module neither generates resamples nor establishes coverage or
the final patch-weight prescription. No inverse or finite-sample correction is
applied. The measured dimension is the number of columns, independent of any
model parameter count.
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CovarianceDiagnostics:
    dimension: int
    n_regions: int
    rank_bound: int
    numerical_rank: int
    rank_tolerance: float
    condition_number: float


@dataclass(frozen=True)
class JackknifeCovariance:
    mean: np.ndarray
    covariance: np.ndarray
    region_ids: np.ndarray
    diagnostics: CovarianceDiagnostics


def _real_array(value, name, ndim):
    raw = np.asarray(value)
    if np.iscomplexobj(raw):
        raise ValueError(f"{name} must be real")
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != ndim or min(result.shape, default=0) == 0:
        raise ValueError(f"{name} must have {ndim} nonempty axes")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    return result


def _region_ids(value, n):
    ids = np.asarray(value)
    if ids.shape != (n,) or ids.dtype.kind not in "iu":
        raise ValueError("region_ids must be a one-dimensional integer ID per row")
    if np.any(ids < 0) or len(np.unique(ids)) != n:
        raise ValueError("region_ids must be nonnegative and unique")
    return ids


def _readonly(value):
    result = np.array(value, copy=True)
    result.flags.writeable = False
    return result


def _result(mean, covariance, region_ids):
    if not np.isfinite(mean).all() or not np.isfinite(covariance).all():
        raise ValueError("covariance arithmetic exceeded finite float64 range")
    # Roundoff can break symmetry in B C B^T. Preserve the symmetric quadratic
    # form; this does not remove negative eigenvalues or regularize the matrix.
    covariance = covariance / 2 + covariance.T / 2
    eigenvalues = np.linalg.eigvalsh(covariance)
    if not np.isfinite(eigenvalues).all():
        raise ValueError("covariance eigenspectrum exceeded finite float64 range")
    dimension, n_regions = len(mean), len(region_ids)
    largest = max(float(eigenvalues[-1]), 0.)
    # A floating-point diagnostic, not a scientific conditioning threshold.
    tolerance = np.finfo(np.float64).eps * max(dimension, n_regions) * largest
    bound = min(dimension, n_regions - 1)
    rank = min(int(np.count_nonzero(eigenvalues > tolerance)), bound)
    if eigenvalues[0] < -tolerance:
        raise ValueError("covariance arithmetic produced a non-positive-semidefinite matrix")
    condition = float(largest / eigenvalues[0]) if rank == dimension else np.inf
    diagnostics = CovarianceDiagnostics(dimension, n_regions, bound, rank,
                                        float(tolerance), condition)
    return JackknifeCovariance(_readonly(mean), _readonly(covariance),
                               _readonly(region_ids), diagnostics)


def delete_one_covariance(replicates, *, region_ids):
    """Return (N-1)/N sum_i (d_i - mean(d))(d_i - mean(d))^T.

    ``replicates`` has shape (region, measured component), with at least two
    regions. Unsupported components (NaN) are rejected, never dropped. Region
    IDs identify the deleted common regions and must be explicit and unique.
    The mean is the mean of delete-one realizations, not a full-sample estimate.
    ``rank_tolerance`` is eps * max(N, dimension) * largest eigenvalue; rank is
    also bounded by N-1. Condition number is infinite when numerically singular.
    """
    values = _real_array(replicates, "replicates", 2)
    n = len(values)
    if n < 2:
        raise ValueError("at least two delete-one regions are required")
    ids = _region_ids(region_ids, n)
    try:
        with np.errstate(over="raise", invalid="raise"):
            mean = values.mean(axis=0)
            centered = values - mean
            covariance = ((n - 1) / n) * (centered.T @ centered)
    except FloatingPointError as error:
        raise ValueError("covariance arithmetic exceeded finite float64 range") from error
    return _result(mean, covariance, ids)
