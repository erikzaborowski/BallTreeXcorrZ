"""Bounded native annulus responses from saved cb spectra (P3 §§4.3–4.6).

No solver, interpolation, survey-window projection or calibration occurs here.
The caller explicitly supplies the saved redshifts, geometry and k-node subset.
"""
from dataclasses import dataclass
from numbers import Integral

import numpy as np

from .cosmology import UNITS
from .theory import PowerSpectrumTable
from .theory_annulus import finite_annulus_hankel


@dataclass(frozen=True)
class ReferenceMatterGrid:
    value: np.ndarray
    redshift: np.ndarray
    chi_hmpc: np.ndarray
    h_over_c: np.ndarray
    theta_edges: np.ndarray
    active: np.ndarray
    k_indices: np.ndarray
    k: np.ndarray
    spectrum_convention: str
    work: int
    workspace_bytes_bound: int
    fields: tuple = ('cb', 'cb')
    definition: str = 'H_over_c_times_finite_annulus_wp_cbcb'
    angular_average: str = 'full_native_annulus; flat_sky_uniform_area; not_survey_RR_weighted'
    redshift_treatment: str = 'explicit_saved_redshifts; no_interpolation'
    quadrature: str = 'composite_simpson_in_log_k'
    work_definition: str = 'active_native_bins_times_selected_k_nodes'
    truncation: str = 'finite_selected_k_domain; omitted_tails_not_estimated'
    convergence: str = 'not_established_by_one_run'
    inactive: str = 'NaN; not_evaluated'
    distance_unit: str = UNITS['distance']
    h_over_c_unit: str = UNITS['H_over_c']
    angle_unit: str = 'radian'
    value_unit: str = 'dimensionless'


def _limit(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(name+' must be an explicit positive integer')


def _vector(value, name):
    if (not isinstance(value, np.ndarray) or value.ndim != 1
            or value.dtype.kind not in 'fiu' or len(value) == 0):
        raise ValueError(name+' must be an explicit nonempty real array')


def evaluate_reference_matter_grid(table, chi_hmpc, h_over_c, theta_edges, active, *,
                                   k_indices, max_work, max_workspace_bytes,
                                   distance_unit=UNITS['distance'],
                                   h_over_c_unit=UNITS['H_over_c'], angle_unit='radian'):
    """Evaluate G(z,i)=(H/c)wp on complete physical chi(z)*theta annuli.

    ``table`` must be a validated PowerSpectrumTable; caller-owned table arrays
    are excluded from the workspace bound. Other arrays are explicit NumPy
    arrays. k_indices is a strictly increasing integer subset with >=3 nodes.
    Each selected redshift is transformed separately, with no z-by-k-by-angle
    cube. Inactive entries remain NaN, including wholly inactive redshifts.

    Limits are checked before copies, result allocation or quadrature. The
    conservative numerical-array workspace allowance includes copies, one
    row table, annulus/filter/Simpson temporaries and validation masks; it
    excludes caller arrays and Python/SciPy allocator/runtime overhead. Work
    counts quadrature samples, not special-function calls or CPU seconds.
    Neither bound is compute authorization or an accuracy guarantee.
    """
    if not isinstance(table, PowerSpectrumTable):
        raise TypeError('table must be a validated PowerSpectrumTable')
    if (table.fields != ('cb', 'cb') or table.k_unit != UNITS['k']
            or table.power_unit != UNITS['power']):
        raise ValueError('reference matter grid requires cb,cb spectra in canonical units')
    if (distance_unit != UNITS['distance'] or h_over_c_unit != UNITS['H_over_c']
            or angle_unit != 'radian'):
        raise ValueError('geometry requires h^-1 Mpc, h Mpc^-1 and radian units')
    _limit(max_work, 'max_work')
    _limit(max_workspace_bytes, 'max_workspace_bytes')
    for a, name in ((chi_hmpc, 'chi_hmpc'), (h_over_c, 'h_over_c'),
                    (theta_edges, 'theta_edges'), (k_indices, 'k_indices')):
        _vector(a, name)
    nz, nt, nk = len(table.redshift), len(theta_edges)-1, len(k_indices)
    if len(chi_hmpc) != nz or len(h_over_c) != nz or nt < 1:
        raise ValueError('geometry must match saved redshifts and nonempty angular bins')
    if (not isinstance(active, np.ndarray) or active.dtype.kind != 'b'
            or active.shape != (nz, nt)):
        raise ValueError('active must be an explicit boolean (redshift, angular bin) array')
    if k_indices.dtype.kind not in 'iu' or nk < 3:
        raise ValueError('k_indices must contain at least three integer nodes')
    work = int(np.count_nonzero(active))*nk
    needed = 24*nz*nt + 64*nz + 160*(nt+1) + 544*nk + 65536
    if work > max_work:
        raise ValueError('native matter quadrature exceeds max_work')
    if needed > max_workspace_bytes:
        raise ValueError('native matter grid exceeds max_workspace_bytes')
    if (np.any(k_indices >= len(table.k)) or np.any(k_indices < 0)
            or np.any(k_indices[1:] <= k_indices[:-1])):
        raise ValueError('k_indices must be increasing distinct in-range saved nodes')
    if any(not np.isfinite(a).all() or np.any(a <= 0) for a in (chi_hmpc, h_over_c, theta_edges)):
        raise ValueError('geometry arrays must be finite and positive')
    if theta_edges[-1] > np.pi or np.any(theta_edges[1:] <= theta_edges[:-1]):
        raise ValueError('theta_edges must increase in radians up to pi')
    chi, hubble, theta = (np.array(a, dtype=float, copy=True)
                          for a in (chi_hmpc, h_over_c, theta_edges))
    selected = np.array(k_indices, dtype=np.int64, copy=True)
    kvals = table.k[selected].copy()
    z, support = table.redshift.copy(), active.copy()
    value = np.full((nz, nt), np.nan)
    with np.errstate(over='raise', invalid='raise', divide='raise'):
        try:
            for iz in range(nz):
                bins = np.flatnonzero(support[iz])
                if not len(bins):
                    continue
                annuli = chi[iz]*np.column_stack((theta[bins], theta[bins+1]))
                row = PowerSpectrumTable(kvals, z[iz:iz+1], table.power[iz, selected][None, :],
                    table.fields, table.k_unit, table.power_unit, table.convention)
                response = finite_annulus_hankel(row, annuli, radius_unit=UNITS['distance'],
                    term='gg', max_work=len(bins)*nk)
                value[iz, bins] = hubble[iz]*response.value[0]
        except FloatingPointError as exc:
            raise ValueError('nonfinite arithmetic in reference matter grid') from exc
    if not np.isfinite(value[support]).all():
        raise ValueError('nonfinite active reference matter response')
    for a in (value, z, chi, hubble, theta, support, selected, kvals):
        a.setflags(write=False)
    return ReferenceMatterGrid(value, z, chi, hubble, theta, support, selected,
                               kvals, table.convention, work, needed)


def map_radial_bins_to_saved_slices(radial_edges, saved_slice_edges, *, max_result_bytes):
    """Map target-bin interiors to containing saved fine slices, without averaging.

    Both edge arrays are finite, increasing and nonnegative. Each complete
    target interval must lie within one saved interval; crossing or outlying
    intervals are rejected, even if their midpoint lies inside a saved slice.
    Matching edges use exact supplied geometry, without rounding/tolerances.
    Searching the lower edge avoids midpoint rounding onto the upper boundary
    for intervals only one float64 step wide. The result is the same interior
    containment map and is immutable. Bounds cover owned numerical arrays,
    not caller arrays or runtime overhead.
    """
    _limit(max_result_bytes, 'max_result_bytes')
    _vector(radial_edges, 'radial_edges')
    _vector(saved_slice_edges, 'saved_slice_edges')
    if len(radial_edges) < 2 or len(saved_slice_edges) < 2:
        raise ValueError('radial and saved grids require at least one interval')
    needed = 64*(len(radial_edges)+len(saved_slice_edges))+4096
    if needed > max_result_bytes:
        raise ValueError('radial containment exceeds max_result_bytes')
    for edges in (radial_edges, saved_slice_edges):
        if (not np.isfinite(edges).all() or np.any(edges < 0)
                or np.any(edges[1:] <= edges[:-1])):
            raise ValueError('radial and saved edges must be finite, nonnegative and increasing')
    index = np.searchsorted(saved_slice_edges, radial_edges[:-1], side='right')-1
    if (np.any(index < 0) or np.any(index >= len(saved_slice_edges)-1)
            or np.any(radial_edges[1:] > saved_slice_edges[index+1])):
        raise ValueError('each complete radial bin must be contained in one saved slice')
    index.setflags(write=False)
    return index
