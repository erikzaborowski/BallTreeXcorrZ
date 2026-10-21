"""Finite-k, uniform-area full-annulus templates (P3, §§4.3–4.4).

The analytic angular average assumes flat-sky uniform area weighting. It is
not a measured survey DR average. No clipping to soft comoving bands, solver,
interpolation, extrapolation or convergence assessment occurs here.
"""
from dataclasses import dataclass
from numbers import Integral

import numpy as np
from scipy.integrate import simpson
from scipy.special import j0, j1, jvp

from .cosmology import UNITS
from .theory import PowerSpectrumTable, _array, _TERM_FIELDS


@dataclass(frozen=True)
class FiniteAnnulusHankelResult:
    """Values have axes (redshift, declared annulus), in the supplied order."""

    annuli: np.ndarray
    redshift: np.ndarray
    value: np.ndarray
    fields: tuple[str, str]
    term: str
    convention: str
    k_bounds: tuple[float, float]
    k_nodes: int
    radius_unit: str = UNITS["distance"]
    value_unit: str = UNITS["w_p"]
    k_unit: str = UNITS["k"]
    quadrature: str = "composite_simpson_in_log_k"
    angular_average: str = "analytic_full_annulus; flat_sky_uniform_area; not_survey_DR_weighted"
    truncation: str = "finite_k_domain; omitted_tails_not_estimated"
    convergence: str = "not_established_by_one_run"


def _annulus_filter(k, low, high):
    """2 [R+ J1(kR+) - R- J1(kR-)] / [k (R+² - R-²)].

    Small-argument and narrow-interval series evaluate the same analytic
    filter without subtractive cancellation. Their thresholds control float64
    evaluation, not a scientific accuracy or angular-bin acceptance rule.
    """
    x, y = k*high, k*low
    ratio = low/high
    area = ((high-low)/high)*(1+ratio)
    result = np.empty_like(k)
    small = x <= .1
    # J1 power series, with (1-t**(2n+2))/(1-t**2) evaluated as a sum.
    # At x<=.1, the first omitted term after n=6 is <3e-26.
    square = -.25*x[small]**2
    coefficient = np.ones_like(square)
    geometric = 1.
    total = coefficient.copy()
    for n in range(1, 7):
        coefficient *= square/(n*(n+1))
        geometric += ratio**(2*n)
        total += coefficient*geometric
    result[small] = total
    middle = x/2+y/2
    half_width = k*((high-low)/2)
    narrow = ~small & (half_width <= .01)
    # For F(x)=x*J1(x), F'=x*J0(x). The symmetric divided difference
    # gives J0 plus even powers of half-width. Here middle>.05, so the
    # first omitted term is bounded by 162*.01**8/9! < 5e-20.
    center, width = middle[narrow], half_width[narrow]
    total = j0(center)
    for order, factorial in ((2, 6), (4, 120), (6, 5040)):
        derivative = jvp(0, center, order) + order*jvp(0, center, order-1)/center
        total += derivative*width**order/factorial
    result[narrow] = total
    direct = ~(small | narrow)
    result[direct] = 2*(j1(x[direct])-ratio*j1(y[direct]))/(x[direct]*area)
    return result


def finite_annulus_hankel(table, annuli, *, radius_unit, term, max_work):
    """Integrate k² P times the full-annulus J1 filter over supplied dln(k).

    ``annuli`` has shape (N,2), containing finite 0<=R_low<R_high in h^-1 Mpc.
    Disks, overlapping annuli and caller-selected ordering are allowed. Each
    entire declared annulus receives uniform flat-sky area weight R dR; these
    are not log-radius averages, clipped bands or measured random-pair means.

    ``max_work`` explicitly bounds Nz*Nk*Nannulus before result allocation.
    Annuli are processed serially; workspace is O(Nz*Nk + Nz*Nannulus + Nk),
    with no Nz*Nk*Nannulus cube. This work bound is not compute authorization.
    Grid and finite-domain convergence, and physical validity of the averaging
    assumption, remain caller responsibilities. Signed cross spectra retain
    their signs. All field/unit conventions match ``finite_hankel``.
    """
    if not isinstance(table, PowerSpectrumTable):
        raise TypeError("table must be an explicit PowerSpectrumTable")
    if not isinstance(term, str) or term not in _TERM_FIELDS:
        raise ValueError("term must be gg, gmu, or mumu")
    if table.fields != _TERM_FIELDS[term]:
        raise ValueError(f"{term} requires fields {_TERM_FIELDS[term]}, got {table.fields}")
    if radius_unit != UNITS["distance"]:
        raise ValueError("annulus radii must be in h^-1 Mpc")
    annuli = _array(annuli, "annuli", 2)
    if (annuli.shape[1] != 2 or np.any(annuli[:, 0] < 0)
            or np.any(annuli[:, 1] <= annuli[:, 0])):
        raise ValueError("annuli must contain pairs with 0 <= lower < upper")
    if (isinstance(max_work, (bool, np.bool_)) or not isinstance(max_work, Integral)
            or max_work <= 0):
        raise ValueError("max_work must be an explicit positive integer")
    work = len(table.redshift)*len(table.k)*len(annuli)
    if work > max_work:
        raise ValueError(f"annulus quadrature work {work} exceeds max_work {max_work}")
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        try:
            log_k = np.log(table.k)
            prefactor = table.power*table.k**2/(2*np.pi)
            value = np.empty((len(table.redshift), len(annuli)))
            for i, (low, high) in enumerate(annuli):
                angular = _annulus_filter(table.k, low, high)
                value[:, i] = simpson(prefactor*angular, x=log_k, axis=1)
        except FloatingPointError as error:
            raise ValueError("nonfinite arithmetic in finite annulus Hankel quadrature") from error
    if not np.isfinite(value).all():
        raise ValueError("nonfinite result in finite annulus Hankel quadrature")
    value.flags.writeable = False
    return FiniteAnnulusHankelResult(
        annuli, table.redshift, value, table.fields, term, table.convention,
        (float(table.k[0]), float(table.k[-1])), len(table.k))
