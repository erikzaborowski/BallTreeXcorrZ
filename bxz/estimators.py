"""Supported-value arrays and ratio helpers for the reference estimator."""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Estimate:
    """An estimate with NaN wherever random-pair support is absent."""

    value: np.ndarray
    supported: np.ndarray


def _counts(*arrays):
    result = tuple(np.asarray(a, dtype=np.float64) for a in arrays)
    if any(a.ndim != 2 or min(a.shape) == 0 for a in result):
        raise ValueError("counts must have nonempty (slice, angular bin) axes")
    if any(a.shape != result[0].shape for a in result):
        raise ValueError("count shapes must agree exactly")
    if any(not np.isfinite(a).all() or np.any(a < 0) for a in result):
        raise ValueError("counts must be finite and nonnegative")
    return result


def _total(value, name, nslice=None):
    value = np.asarray(value, dtype=np.float64)
    if value.shape != (() if nslice is None else (nslice,)):
        raise ValueError(f"{name} must be a scalar" if nslice is None else
                         f"{name} must have one total per reference slice")
    if not np.isfinite(value).all() or np.any(value <= 0):
        raise ValueError(f"{name} must be finite and positive")
    return value


def _ratio(numerator, denominator):
    supported = denominator > 0
    value = np.full(denominator.shape, np.nan)
    np.divide(numerator, denominator, out=value, where=supported)
    return Estimate(value, supported)
