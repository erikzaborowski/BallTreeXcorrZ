"""Explicit omega-weighted measurement and theory operator (plan §4.3).

One operator represents one complete comoving test window, divided into scale
bands. Geometry exclusion applies to the complete window for each fine slice.
Angles are radians, chi and R use matching distance units (normally h^-1 Mpc).
Soft overlap weights multiply FULL native-bin counts/averages, not clipped
annuli. No physical theory, bin-centre approximation, or fitting is supplied.
"""
from dataclasses import dataclass

import numpy as np

from .estimators import Estimate, _counts, _ratio


def _edges(value, name):
    value = np.asarray(value, dtype=np.float64)
    if (value.ndim != 1 or len(value) < 2 or not np.isfinite(value).all()
            or np.any(value <= 0) or np.any(np.diff(value) <= 0)):
        raise ValueError(f"{name} must be finite positive increasing edges")
    return value


@dataclass(frozen=True)
class Geometry:
    nominal_log_width: float
    retained_log_width: np.ndarray
    accessible_bounds: np.ndarray
    included: np.ndarray
    exclusion_reason: tuple[str, ...]
    band_support: np.ndarray


@dataclass(frozen=True)
class BandEstimate:
    value: np.ndarray
    supported: np.ndarray
    random_support: np.ndarray
    slice_window: np.ndarray


class MeasurementOperator:
    """Build deterministic slice/bin weights before inspecting amplitudes.

    ``output_bin`` is an integer label per fine slice, in [0,n_output_bins).
    Hard-cut indices refer to entries in theta_edges, never rounded angles.
    ``r_edges`` bounds the entire window; separate windows need separate
    operators. Zero random support yields NaN estimates and NaN A_bj(k).
    """

    def __init__(self, theta_edges, r_edges, chi, redshift, output_bin, *,
                 n_output_bins, lower_edge_index, upper_edge_index):
        theta = _edges(theta_edges, "theta_edges")
        r = _edges(r_edges, "r_edges")
        if theta[-1] > np.pi:
            raise ValueError("theta_edges must be in radians and at most pi")
        chi = np.asarray(chi, dtype=np.float64)
        z = np.asarray(redshift, dtype=np.float64)
        output_bin = np.asarray(output_bin)
        if (chi.ndim != 1 or len(chi) == 0 or z.shape != chi.shape
                or not np.isfinite(chi).all() or not np.isfinite(z).all()
                or np.any(chi < 0) or np.any(z < 0)
                or np.any((z >= .05) & (chi <= 0))):
            raise ValueError("chi/redshift must be aligned finite nonnegative slices; chi>0 at z>=.05")
        if (isinstance(n_output_bins, (bool, np.bool_))
                or not isinstance(n_output_bins, (int, np.integer))
                or n_output_bins < 1):
            raise ValueError("n_output_bins must be a positive integer")
        if (output_bin.shape != chi.shape or output_bin.dtype.kind not in "iu"
                or np.any(output_bin < 0) or np.any(output_bin >= n_output_bins)):
            raise ValueError("output_bin must be an aligned array of valid integer labels")
        if any(isinstance(i, (bool, np.bool_)) or not isinstance(i, (int, np.integer))
               for i in (lower_edge_index, upper_edge_index)):
            raise ValueError("hard cuts must be native-grid edge indices")
        if not 0 <= lower_edge_index < upper_edge_index < len(theta):
            raise ValueError("hard-cut edge indices are out of order or outside the grid")

        lo = np.maximum(r[0], chi * theta[lower_edge_index])
        hi = np.minimum(r[-1], chi * theta[upper_edge_index])
        nominal = np.log(r[-1] / r[0])
        width = np.zeros(len(chi))
        nonempty = hi > lo
        width[nonempty] = np.log(hi[nonempty] / lo[nonempty])
        reasons = tuple("redshift_below_0.05" if zz < .05 else
                        "empty_window" if not ne else
                        "less_than_half_nominal_log_width" if ww < .5 * nominal else
                        "" for zz, ne, ww in zip(z, nonempty, width))
        included = np.array([not reason for reason in reasons])

        # Log R boundaries at each slice avoid division by chi=0 in excluded
        # z=0 slices. A harmless positive placeholder is masked out below.
        log_r_native = np.log(np.where(chi > 0, chi, 1))[:, None] + np.log(theta)[None, :]
        overlap = np.maximum(0.,
            np.minimum(log_r_native[:, 1:, None], np.log(r[1:])[None, None, :])
            - np.maximum(log_r_native[:, :-1, None], np.log(r[:-1])[None, None, :]))
        omega = np.clip(overlap / np.diff(np.log(theta))[None, :, None], 0., 1.)
        active = (np.arange(len(theta) - 1) >= lower_edge_index) & (np.arange(len(theta) - 1) < upper_edge_index)
        omega *= active[None, :, None] & included[:, None, None]

        self.theta_edges = theta.copy()
        self.r_edges = r.copy()
        self.output_bin = output_bin.copy()
        self.n_output_bins = int(n_output_bins)
        self.omega = omega
        self.delta_log_r = np.diff(np.log(r))
        self.geometry = Geometry(nominal, width, np.column_stack([lo, hi]),
                                 included, reasons, np.any(omega > 0, axis=1))
        # Keep input mutation from changing a predeclared geometry operator.
        for array in (self.theta_edges, self.r_edges, self.output_bin, self.omega,
                      self.delta_log_r, width, self.geometry.accessible_bounds,
                      included, self.geometry.band_support):
            array.setflags(write=False)

    def _native(self, value):
        (value,) = _counts(value)
        if value.shape != self.omega.shape[:2]:
            raise ValueError("counts must match the operator's (slice, angular bin) axes")
        return value

    def _aggregate(self, native):
        per_slice = np.einsum("kij,ki->kj", self.omega, native)
        bands = np.zeros((self.n_output_bins, len(self.delta_log_r)))
        np.add.at(bands, self.output_bin, per_slice)
        return bands, per_slice

    def _random_window(self, dr):
        denominator, per_slice = self._aggregate(dr)
        # A has explicit (output bin, scale band, fine slice) axes.
        window = np.zeros(denominator.shape + (len(dr),))
        for k, b in enumerate(self.output_bin):
            window[b, :, k] = per_slice[k]
        np.divide(window, denominator[:, :, None], out=window,
                  where=denominator[:, :, None] > 0)
        window[denominator <= 0] = np.nan
        return denominator, window


    def project_theory(self, full_bin_averages, dr):
        """Project externally computed random-weighted FULL native-bin means.

        Means may be signed. The caller must obtain them using the same random
        selection and intra-bin weighting as dr. This method cannot establish
        that provenance from an array alone.
        """
        dr = self._native(dr)
        means = np.asarray(full_bin_averages, dtype=np.float64)
        if means.shape != dr.shape or not np.isfinite(means).all():
            raise ValueError("full_bin_averages must be finite with exact native count shape")
        numerator, _ = self._aggregate(dr * means)
        denominator, window = self._random_window(dr)
        result = _ratio(numerator, denominator)
        return BandEstimate(result.value, result.supported, denominator, window)

    def integrate(self, bands):
        """Sum W_bj delta-lnR_j; unsupported bands propagate NaN explicitly.

        An incomplete scale window is never silently renormalized or summed
        with missing bands set to zero.
        """
        value = np.asarray(bands.value, dtype=np.float64)
        support = np.asarray(bands.supported)
        shape = (self.n_output_bins, len(self.delta_log_r))
        if value.shape != shape or support.shape != shape or support.dtype.kind != "b":
            raise ValueError("bands must match the operator output shape with boolean support")
        if np.any(support & ~np.isfinite(value)):
            raise ValueError("supported bands must be finite")
        supported = np.all(support, axis=1)
        integrated = np.full(self.n_output_bins, np.nan)
        integrated[supported] = value[supported] @ self.delta_log_r
        return Estimate(integrated, supported)
