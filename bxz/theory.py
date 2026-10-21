"""Field-labelled power-spectrum tables with explicit units."""

from dataclasses import dataclass

import numpy as np

from .cosmology import UNITS


_TERM_FIELDS = {"gg": ("cb", "cb"), "gmu": ("cb", "m"), "mumu": ("m", "m")}


def _array(value, name, ndim):
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf" or raw.ndim != ndim:
        raise ValueError(f"{name} must be a real {ndim}-dimensional numeric array")
    result = np.array(raw, dtype=np.float64, copy=True)
    if result.size == 0 or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be nonempty and finite")
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class PowerSpectrumTable:
    """P(z, k) with explicit physical labels and no redshift interpolation.

    ``convention`` must identify the caller's spectrum prescription. Its
    presence records a choice; this class cannot establish its physical
    correctness. Cross spectra may be signed; auto spectra may not.
    Unit strings are the canonical strings in ``bxz.cosmology.UNITS``.
    """

    k: np.ndarray
    redshift: np.ndarray
    power: np.ndarray
    fields: tuple[str, str]
    k_unit: str
    power_unit: str
    convention: str

    def __post_init__(self):
        k = _array(self.k, "k", 1)
        z = _array(self.redshift, "redshift", 1)
        power = _array(self.power, "power", 2)
        if len(k) < 3 or np.any(k <= 0) or np.any(np.diff(k) <= 0):
            raise ValueError("k must contain at least three positive increasing nodes")
        if np.any(z < 0) or np.any(np.diff(z) <= 0):
            raise ValueError("redshift must be nonnegative and strictly increasing")
        if power.shape != (len(z), len(k)):
            raise ValueError("power must have shape (number of redshifts, number of k nodes)")
        if not isinstance(self.fields, (tuple, list)) or tuple(self.fields) not in _TERM_FIELDS.values():
            raise ValueError("fields must be ('cb', 'cb'), ('cb', 'm'), or ('m', 'm')")
        fields = tuple(self.fields)
        if fields[0] == fields[1] and np.any(power < 0):
            raise ValueError("auto power spectra must be nonnegative")
        if self.k_unit != UNITS["k"] or self.power_unit != UNITS["power"]:
            raise ValueError("power spectra require k in h Mpc^-1 and P in (h^-1 Mpc)^3")
        if not isinstance(self.convention, str) or not self.convention.strip():
            raise ValueError("an explicit nonempty spectrum convention is required")
        object.__setattr__(self, "k", k)
        object.__setattr__(self, "redshift", z)
        object.__setattr__(self, "power", power)
        object.__setattr__(self, "fields", fields)
