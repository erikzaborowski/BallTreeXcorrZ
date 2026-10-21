"""Explicit CAMB numerical settings and cosmological parameter construction."""

from dataclasses import dataclass
import math

import numpy as np

from .cosmology import CosmologyParameters


@dataclass(frozen=True)
class LinearCAMBSettings:
    """Caller-selected numerical and physical declarations, without defaults.

    ``kmax_mpc`` is CAMB's *physical* transfer bound in Mpc^-1, whereas the
    returned table uses h Mpc^-1. ``k_per_logint`` controls minimum native
    transfer sampling, independently of the caller's requested output grid.
    Helium and the scalar pivot are explicit to avoid a hidden BBN or pivot
    choice. The settings below do not establish numerical convergence.
    """

    kmax_mpc: float
    k_per_logint: int
    accuracy_boost: float
    pivot_scalar_mpc: float
    helium_mass_fraction: float

    def __post_init__(self):
        for name in ("kmax_mpc", "accuracy_boost", "pivot_scalar_mpc"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be a finite positive number")
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a finite positive number")
        if (isinstance(self.k_per_logint, bool)
                or not isinstance(self.k_per_logint, int) or self.k_per_logint < 1):
            raise ValueError("k_per_logint must be a positive integer")
        y = self.helium_mass_fraction
        if isinstance(y, bool) or not isinstance(y, (int, float)) or not 0 < y < 1:
            raise ValueError("helium_mass_fraction must be strictly between zero and one")


def _grid(values, name, minimum_size, *, positive):
    raw = np.asarray(values)
    if raw.ndim != 1 or raw.dtype.kind not in "iuf" or len(raw) < minimum_size:
        raise ValueError(f"{name} must be a real one-dimensional array with >= {minimum_size} nodes")
    values = np.array(raw, dtype=float, copy=True)
    if (not np.all(np.isfinite(values)) or np.any(np.diff(values) <= 0)
            or np.any(values <= 0 if positive else values < 0)):
        raise ValueError(f"{name} must be finite, strictly increasing, and {'positive' if positive else 'nonnegative'}")
    return values


def _validate(cosmology, k, redshift, settings):
    if not isinstance(cosmology, CosmologyParameters):
        raise TypeError("cosmology must be an explicit CosmologyParameters instance")
    if not isinstance(settings, LinearCAMBSettings):
        raise TypeError("settings must be an explicit LinearCAMBSettings instance")
    if cosmology.As is None or cosmology.sigma8 is not None:
        raise ValueError("linear CAMB provider requires fixed As; sigma8 rescaling is unsupported")
    if cosmology.w0 != -1 or cosmology.wa != 0 or cosmology.omk != 0:
        raise ValueError("linear CAMB provider currently supports flat LambdaCDM only")
    if cosmology.num_massive_neutrinos != 1 or cosmology.mnu <= 0 or cosmology.nnu != 3.046:
        raise ValueError("linear CAMB provider requires one massive species and Neff=3.046")
    k = _grid(k, "k", 3, positive=True)
    redshift = _grid(redshift, "redshift", 1, positive=False)
    if k[-1] * cosmology.h > settings.kmax_mpc:
        raise ValueError("requested k exceeds physical CAMB kmax_mpc after multiplying by h")
    return k, redshift


def _parameters(camb, cosmology, redshift, settings):
    """Construct parameters only; this helper does not execute CAMB."""
    p = cosmology
    pars = camb.CAMBparams()
    pars.set_cosmology(
        H0=100 * p.h, ombh2=p.ombh2, omch2=p.omch2, omk=p.omk,
        mnu=p.mnu, nnu=p.nnu, num_massive_neutrinos=1,
        neutrino_hierarchy="degenerate", standard_neutrino_neff=3.046,
        meffsterile=0, TCMB=p.TCMB, YHe=settings.helium_mass_fraction,
    )
    pars.set_dark_energy(w=-1, wa=0, dark_energy_model="fluid")
    pars.InitPower.set_params(
        As=p.As, ns=p.ns, nrun=0, nrunrun=0, r=0,
        pivot_scalar=settings.pivot_scalar_mpc,
    )
    pars.scalar_initial_condition = "initial_adiabatic"
    pars.WantCls = False
    pars.Want_CMB = False
    pars.Want_CMB_lensing = False
    pars.DoLensing = False
    pars.WantTensors = False
    pars.WantVectors = False
    pars.WantScalars = True
    pars.NonLinear = camb.model.NonLinear_none
    pars.set_accuracy(
        AccuracyBoost=settings.accuracy_boost,
        lAccuracyBoost=settings.accuracy_boost,
        lSampleBoost=settings.accuracy_boost,
        DoLateRadTruncation=False,
    )
    pars.set_matter_power(
        redshifts=redshift, kmax=settings.kmax_mpc,
        k_per_logint=settings.k_per_logint, nonlinear=False,
        accurate_massive_neutrino_transfers=True, silent=True,
    )
    return pars
