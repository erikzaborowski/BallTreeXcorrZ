"""Selected nonlinear cb templates for the first apparent kernel (P3 §2.4).

The owner selected Takahashi HALOFIT as default and HMcode 2020 as an
alternative. CAMB 1.6.4 applies its total-matter nonlinear boost to the
requested linear cb spectrum. This is an explicit approximation, not an
independently calibrated nonlinear cb prediction. The source-reviewed path is
Transfer_GetUnsplinedNonlinearPower in
https://github.com/cmbant/CAMB/blob/1.6.4/fortran/results.f90 .

Importing this module does not import CAMB or run a solver. A solver call
requires a workload assessment and any required compute approval. No data,
measured inference, allocation, or package installation is performed here.
"""

from dataclasses import asdict, dataclass
import json
import os

import numpy as np

from .cosmology import UNITS
from .theory import PowerSpectrumTable
from .theory_camb import _parameters, _validate


_REVIEWED_CAMB_VERSION = "1.6.4"
_PRESCRIPTIONS = ("takahashi", "mead2020")
_APPROXIMATION = "P_cbcb_template = P_cbcb_linear * (P_mm_nonlinear / P_mm_linear)"


@dataclass(frozen=True)
class NonlinearCBSpectrum:
    """The gg table and immutable JSON describing its physical convention."""

    gg: PowerSpectrumTable
    provenance_json: str


def _check_backend(camb):
    # The boost field is a process-global CAMB setting, independent of var1/2.
    # Do not silently mutate it or accept another version's unreviewed behavior.
    if camb.__version__ != _REVIEWED_CAMB_VERSION:
        raise RuntimeError("nonlinear cb provider requires source-reviewed CAMB 1.6.4")
    if camb.config.transfer_power_var != camb.model.Transfer_tot:
        raise RuntimeError("CAMB transfer_power_var must be Transfer_tot for the declared total-matter boost")


def _nonlinear_parameters(camb, cosmology, redshift, settings, prescription):
    pars = _parameters(camb, cosmology, redshift, settings)
    pars.NonLinearModel.set_params(halofit_version=prescription)
    pars.NonLinear = camb.model.NonLinear_pk
    if pars.NonLinearModel.halofit_version != prescription:
        raise RuntimeError("CAMB did not retain the requested nonlinear prescription")
    return pars


def nonlinear_cb_spectrum(cosmology, k, redshift, *, settings, prescription="takahashi"):
    """Run one serial CAMB solve with an explicit total-matter-boost template.

    Uses the existing CosmologyParameters and LinearCAMBSettings declarations:
    fixed As, flat LambdaCDM, one massive species, Neff=3.046, explicit helium,
    scalar pivot and numerical settings. The settings type is shared for its
    numerical/parameter declarations; the returned convention is nonlinear.

    Only ``takahashi`` and ``mead2020`` are supported. No feedback prescription
    or independently calibrated cb nonlinear correction is implicit. The
    global CAMB boost field must already be total matter; it is never changed.
    Requested redshifts are native solver nodes. k is in h Mpc^-1 and both
    native interpolation limits are checked; extrapolation is forbidden.

    OMP_NUM_THREADS=1 is required. This check is not compute authorization.
    Numerical convergence and validity on the selected scales remain separate
    validation requirements; no conclusion follows merely from this call.
    """
    if not isinstance(prescription, str) or prescription not in _PRESCRIPTIONS:
        raise ValueError("prescription must be takahashi or mead2020 (without feedback)")
    k, redshift = _validate(cosmology, k, redshift, settings)
    if os.environ.get("OMP_NUM_THREADS") != "1":
        raise ValueError("nonlinear CAMB provider requires OMP_NUM_THREADS=1")
    import camb

    _check_backend(camb)
    if len(redshift) > camb.model.max_transfer_redshifts:
        raise ValueError("too many requested redshifts for installed CAMB")
    pars = _nonlinear_parameters(camb, cosmology, redshift, settings, prescription)
    results = camb.get_results(pars)
    _check_backend(camb)
    interpolator, native_z, native_k = results.get_matter_power_interpolator(
        nonlinear=True, var1="delta_nonu", var2="delta_nonu",
        hubble_units=True, k_hunit=True, return_z_k=True,
        log_interp=True, extrap_kmax=None, silent=True,
    )
    native_k = np.asarray(native_k)
    if (native_k.ndim != 1 or len(native_k) < 2 or native_k.dtype.kind not in "iuf"
            or not np.isfinite(native_k).all() or np.any(native_k <= 0)
            or np.any(np.diff(native_k) <= 0)):
        raise RuntimeError("CAMB returned an invalid native k grid")
    if not np.array_equal(native_z, redshift):
        raise RuntimeError("CAMB redshift nodes differ from the explicitly requested nodes")
    if k[0] < native_k[0] or k[-1] > native_k[-1]:
        raise ValueError("requested k lies outside native CAMB bounds; extrapolation forbidden")
    if not interpolator.islog or interpolator.logsign != 1:
        raise RuntimeError("expected positive cb template for log interpolation")
    power = np.asarray(interpolator.P(redshift, k), dtype=float)
    if np.any(power <= 0):
        raise RuntimeError("CAMB cb template must remain strictly positive")
    table = PowerSpectrumTable(
        k=k, redshift=redshift, power=power, fields=("cb", "cb"),
        k_unit=UNITS["k"], power_unit=UNITS["power"],
        convention=(f"CAMB {camb.__version__}; {prescription}; synchronous; fixed_As; "
                    "adiabatic; delta_nonu x delta_nonu; total_matter_nonlinear_boost; "
                    "not_independently_calibrated_nonlinear_cb; native_logP_logk_interpolation"),
    )
    provenance = {
        "schema": "bxz.camb-nonlinear-cb.v1", "camb_version": camb.__version__,
        "source_review_version": _REVIEWED_CAMB_VERSION,
        "source_review_url": "https://github.com/cmbant/CAMB/blob/1.6.4/fortran/results.f90",
        "prescription": prescription, "baryonic_feedback_model": "not_selected",
        "cosmology": cosmology.as_dict(), "settings": asdict(settings),
        "camb_parameters": str(pars),
        "normalization": "fixed_As_at_explicit_physical_scalar_pivot",
        "neutrino_convention": "one_massive_species; CAMB_mnu_density_proxy; standard_neff=3.046",
        "fields": ["cb", "cb"], "transfer_variables": ["delta_nonu", "delta_nonu"],
        "nonlinear_boost_fields": ["m", "m"],
        "transfer_power_var": int(camb.config.transfer_power_var),
        "nonlinear_approximation": _APPROXIMATION,
        "independently_calibrated_nonlinear_cb": False,
        "k_unit": UNITS["k"], "power_unit": UNITS["power"],
        "requested_k": k.tolist(), "requested_redshift": redshift.tolist(),
        "native_grid": {"kmin": float(native_k[0]), "kmax": float(native_k[-1]),
                        "k_nodes": len(native_k), "redshift": redshift.tolist()},
        "native_transfer_dtype": str(results.get_matter_transfer_data().transfer_data.dtype),
        "solver_calls": 1, "omp_num_threads": 1, "extrapolation": "forbidden",
        "convergence": "not_established_by_one_run",
        "scope": "apparent_kernel_matter_template; no_effective_to_linear_calibration",
    }
    return NonlinearCBSpectrum(table, json.dumps(provenance, sort_keys=True, allow_nan=False))
