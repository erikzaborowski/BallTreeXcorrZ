"""Explicit fiducial parameters and flat-background units (P1, plan §2.4).

The default uses the published *marginal means* of Planck 2018 Table 2,
TT,TE,EE+lowE+lensing, not Astropy's named Planck18 realization:
https://www.aanda.org/articles/aa/abs/2020/09/aa33910-18/aa33910-18.html
Open manuscript: https://arxiv.org/pdf/1807.06209 (Table 2 and section 2.1).
As is exp(3.044) / 1e10. The assumed neutrino sector has one 0.06 eV
species and two massless species, Neff=3.046, and TCMB=2.7255 K.

Background calculations use Astropy's explicit flat w0-wa implementation
and its documented massive-neutrino interpolation. These are not CAMB
transfer functions. Nonlinear/transfer choices must be declared by each run.
"""

from dataclasses import asdict, dataclass
from functools import cached_property
import math
from collections.abc import Mapping

import numpy as np

C_OVER_100 = 2997.92458  # h^-1 Mpc
UNITS = {
    "distance": "h^-1 Mpc", "w_p": "h^-1 Mpc", "k": "h Mpc^-1",
    "power": "(h^-1 Mpc)^3", "H_over_c": "h Mpc^-1",
    "phi_p": "redshift^-1", "Y": "redshift^-1", "separation": "comoving",
}


def _finite(value, name, *, positive=False, nonnegative=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    if not math.isfinite(value) or (positive and value <= 0) or (nonnegative and value < 0):
        raise ValueError(f"invalid {name}: {value!r}")
    return float(value)


@dataclass(frozen=True)
class CosmologyParameters:
    h: float = 0.6736
    ombh2: float = 0.02237
    omch2: float = 0.1200
    ns: float = 0.9649
    As: float | None = math.exp(3.044) * 1e-10
    sigma8: float | None = None
    mnu: float = 0.06
    num_massive_neutrinos: int = 1
    neutrino_masses: tuple[float, ...] = (0.06,)
    nnu: float = 3.046
    w0: float = -1.0
    wa: float = 0.0
    TCMB: float = 2.7255
    omk: float = 0.0

    def __post_init__(self):
        for key in ("h", "ombh2", "omch2", "ns", "nnu", "TCMB"):
            _finite(getattr(self, key), key, positive=True)
        for key in ("w0", "wa", "omk"):
            _finite(getattr(self, key), key)
        if self.omk != 0:
            raise ValueError("only spatially flat cosmologies (omk=0) are supported")
        if (self.As is None) == (self.sigma8 is None):
            raise ValueError("declare exactly one fixed normalization: As or sigma8")
        _finite(self.As if self.As is not None else self.sigma8, "normalization", positive=True)
        _finite(self.mnu, "mnu", nonnegative=True)
        count = self.num_massive_neutrinos
        if isinstance(count, bool) or not isinstance(count, int) or not 0 <= count <= int(self.nnu):
            raise ValueError("invalid number of massive neutrino species")
        masses = tuple(self.neutrino_masses)
        if len(masses) != count:
            raise ValueError("neutrino_masses must specify each massive species")
        for mass in masses:
            _finite(mass, "neutrino mass", positive=True)
        if not math.isclose(sum(masses), self.mnu, rel_tol=1e-12, abs_tol=1e-14):
            raise ValueError("neutrino masses do not sum to mnu")
        if (self.ombh2 + self.omch2) / self.h**2 >= 1:
            raise ValueError("CDM+baryon density must be below the flat critical density")
        object.__setattr__(self, "neutrino_masses", masses)

    @classmethod
    def from_mapping(cls, params: Mapping):
        """Require all numerical declarations; never fill a run from defaults."""
        values = dict(params)
        required = set(cls.__dataclass_fields__) - {"As", "sigma8"}
        missing = required - values.keys()
        unknown = values.keys() - cls.__dataclass_fields__.keys()
        if missing or unknown:
            raise ValueError(f"cosmology parameters missing={sorted(missing)}, unknown={sorted(unknown)}")
        values.setdefault("As", None)
        values.setdefault("sigma8", None)
        return cls(**values)

    def as_dict(self):
        result = asdict(self)
        result["neutrino_masses"] = list(self.neutrino_masses)
        return result


PLANCK18_BASE = CosmologyParameters()


class FlatCosmology:
    """Background with chi in h^-1 Mpc and H/c in h Mpc^-1."""

    def __init__(self, params: CosmologyParameters):
        if not isinstance(params, CosmologyParameters):
            raise TypeError("params must be an explicit CosmologyParameters instance")
        self.params = params

    @cached_property
    def astropy(self):
        from astropy import units as u
        from astropy.cosmology import Flatw0waCDM

        p = self.params
        masses = [0.0] * (int(p.nnu) - p.num_massive_neutrinos) + list(p.neutrino_masses)
        kwargs = dict(H0=100 * p.h, Ob0=p.ombh2 / p.h**2, Tcmb0=p.TCMB,
                      Neff=p.nnu, m_nu=np.asarray(masses) * u.eV, w0=p.w0, wa=p.wa)
        omega_cb = (p.ombh2 + p.omch2) / p.h**2
        # Astropy handles neutrinos separately from Om0, even when massive.
        # Om0 is therefore CDM+baryons here; adding Onu0 would double count it.
        background = Flatw0waCDM(Om0=omega_cb, **kwargs)
        if omega_cb + background.Onu0 + background.Ogamma0 >= 1:
            raise ValueError("flat cosmology has nonpositive dark-energy density")
        return background

    @staticmethod
    def _redshift(z):
        z = np.asarray(z, dtype=float)
        if not np.all(np.isfinite(z)) or np.any(z < 0):
            raise ValueError("redshifts must be finite and nonnegative")
        return z

    def E(self, z):
        return self.astropy.efunc(self._redshift(z))

    def chi(self, z):
        return self.astropy.comoving_distance(self._redshift(z)).value * self.params.h

    def H_over_c(self, z):
        return self.E(z) / C_OVER_100

    def H_over_c_mpc(self, z):
        """Physical Mpc^-1, for explicit conversion checks only."""
        return self.params.h * self.H_over_c(z)


def mpc_to_hmpc(distance, h):
    """Numerical distance in Mpc -> numerical distance in h^-1 Mpc."""
    return np.asarray(distance) * _finite(h, "h", positive=True)


def hmpc_to_mpc(distance, h):
    return np.asarray(distance) / _finite(h, "h", positive=True)


def projected_kernel_factor(H_over_c, w_p, *, hubble_unit, wp_unit):
    """Multiply a projection only with the paired, declared h-units."""
    if hubble_unit != UNITS["H_over_c"] or wp_unit != UNITS["w_p"]:
        raise ValueError("H/c and w_p must use h Mpc^-1 and h^-1 Mpc together")
    return np.asarray(H_over_c) * np.asarray(w_p)
