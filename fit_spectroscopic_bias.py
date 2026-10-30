"""Fit w_ss(theta) = b_s**2 T(theta), Eq. 5.9 of arXiv:1909.07412.

Standalone addition to the legacy pipeline. Inputs are angular spectroscopic
autocorrelations, their covariance, a radial selection histogram, and a saved
nonlinear total-matter power grid. This does not count pairs or fit w_sp.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.interpolate import interp1d
from scipy.special import j1


def increasing(value, name, positive=False):
    value = np.asarray(value, dtype=float)
    if (value.ndim != 1 or len(value) < 2 or not np.isfinite(value).all()
            or np.any(np.diff(value) <= 0) or (positive and value[0] <= 0)):
        raise ValueError(f'{name} must be a finite increasing vector')
    return value


def annulus_window(xlo, xhi):
    """Area average of J0(k chi theta) over an angular annulus.

    xlo/hi = k chi theta_lo/hi; integral 2 theta dtheta/(hi^2-lo^2).
    The small-argument series avoids subtraction of nearly equal J1 terms.
    """
    xlo, xhi = np.broadcast_arrays(np.asarray(xlo, float), np.asarray(xhi, float))
    if not np.isfinite(xlo).all() or not np.isfinite(xhi).all() or np.any((xlo < 0) | (xhi <= xlo)):
        raise ValueError('Require finite 0 <= xlo < xhi')
    result = np.empty_like(xlo)
    small = xhi < 1e-3
    a, b = xlo[small]**2, xhi[small]**2
    result[small] = 1 - (a+b)/8 + (a*a+a*b+b*b)/192
    a, b = xlo[~small], xhi[~small]
    result[~small] = 2*(b*j1(b)-a*j1(a))/(b*b-a*a)
    return result


def project_auto(k, power, chi, dchi_dz, pz, z_weights, theta_edges):
    """Eq. 5.9 with normalized p(z), annulus averaging, and redshift evolution.

    k: h/Mpc; power: (Mpc/h)^3 with shape (nz,nk); chi and dchi/dz:
    Mpc/h. Computes integral dz p(z)^2/(dchi/dz) integral k dk P W/(2pi).
    z_weights are integration weights, NOT counts or selection weights.
    No RSD, magnification, integral-constraint or survey-window correction.
    """
    k = increasing(k, 'k', positive=True)
    edges = increasing(theta_edges, 'theta_edges', positive=True)
    chi, dchi_dz, pz, z_weights = [np.asarray(x, float) for x in (chi, dchi_dz, pz, z_weights)]
    if (chi.ndim != 1 or any(x.shape != chi.shape for x in (dchi_dz, pz, z_weights))
            or any(not np.isfinite(x).all() for x in (chi, dchi_dz, pz, z_weights))
            or np.any(chi <= 0) or np.any(dchi_dz <= 0) or np.any(pz < 0)
            or np.any(z_weights <= 0)):
        raise ValueError('Invalid radial quadrature or selection')
    if not np.isclose(pz @ z_weights, 1., rtol=1e-8, atol=1e-10):
        raise ValueError('p(z) must integrate to one')
    power = np.asarray(power, float)
    if power.shape != (len(chi), len(k)) or not np.isfinite(power).all() or np.any(power < 0):
        raise ValueError('Power must be finite, nonnegative, and shaped (nz,nk)')
    template = np.zeros(len(edges)-1)
    for distance, derivative, p, weight, pk in zip(chi, dchi_dz, pz, z_weights, power):
        x = distance*k[:, None]*edges[None, :]
        window = annulus_window(x[:, :-1], x[:, 1:])
        response = np.trapz(k[:, None]*pk[:, None]*window, k, axis=0)/(2*np.pi)
        template += weight*p*p/derivative*response
    return template


def make_template(grid, theta_edges, z_edges, nz_counts, radial_order=8):
    """Project a saved Halofit grid through a piecewise-constant p(z).

    nz_counts are histogram counts (or summed selection weights), not dN/dz.
    Interpolate log(P) in z at fixed k. No redshift or k extrapolation.
    Distances match the legacy code's astropy Planck15 cosmology.
    """
    z_edges = increasing(z_edges, 'z_edges')
    counts = np.asarray(nz_counts, float)
    if (z_edges[0] < 0 or counts.shape != (len(z_edges)-1,) or not np.isfinite(counts).all()
            or np.any(counts < 0) or counts.sum() <= 0 or radial_order < 2):
        raise ValueError('Invalid redshift histogram or quadrature order')
    basis = make_template_basis(grid, theta_edges, z_edges, radial_order)
    return templates_from_histograms(basis, counts)


def templates_from_histograms(basis, counts):
    """Reuse radial-cell projections for full and jackknife p(z)^2 windows."""
    counts = np.asarray(counts, float)
    if (counts.ndim not in (1, 2) or counts.shape[-1] != len(basis)
            or not np.isfinite(counts).all() or np.any(counts < 0)
            or np.any(counts.sum(axis=-1) <= 0)):
        raise ValueError('Invalid radial histogram(s)')
    return (counts/counts.sum(axis=-1, keepdims=True))**2 @ basis


def make_template_basis(grid, theta_edges, z_edges, radial_order=8):
    """Each row is the template for a normalized top-hat in one radial cell.

    Squared cell probabilities combine these without reevaluating Bessel
    functions for every jackknife histogram. Geometry/annuli remain fixed.
    """
    from astropy.cosmology import Planck15
    from astropy.constants import c

    z_edges = increasing(z_edges, 'z_edges')
    if z_edges[0] < 0 or radial_order < 2:
        raise ValueError('Invalid radial edges or quadrature order')
    z_grid = increasing(grid['z'], 'power z')
    k = increasing(grid['k_hmpc'], 'power k', positive=True)
    pk = np.asarray(grid['pk_mpc_over_h3'], float)
    if pk.shape != (len(z_grid), len(k)) or not np.isfinite(pk).all() or np.any(pk <= 0):
        raise ValueError('Invalid positive nonlinear power grid')
    if z_edges[0] < z_grid[0] or z_edges[-1] > z_grid[-1]:
        raise ValueError('Power redshift grid must cover the full selection histogram')
    nodes, weights = leggauss(radial_order)
    dz = np.diff(z_edges)
    z = ((z_edges[1:]+z_edges[:-1])[:, None]/2 + dz[:, None]*nodes/2).ravel()
    zw = (dz[:, None]*weights/2).ravel()
    power = np.exp(interp1d(z_grid, np.log(pk), axis=0, bounds_error=True)(z))
    chi = Planck15.comoving_distance(z).value*Planck15.h
    derivative = c.to_value('km/s')/Planck15.H(z).value*Planck15.h
    basis = []
    for i, width in enumerate(dz):
        cell = slice(i*radial_order, (i+1)*radial_order)
        basis.append(project_auto(k, power[cell], chi[cell], derivative[cell],
                                  np.full(radial_order, 1/width), zw[cell], theta_edges))
    return np.asarray(basis)


def fit_bias(values, template, covariance):
    """Generalized least-squares fit of the single amplitude A=b_s^2.

    No covariance regularization, Hartlap factor, positivity clipping, or
    fitted offset. sigma_A is conditional on the supplied covariance/template;
    sigma_b uses linear error propagation. A +/- sigma_A is also transformed
    to an asymmetric physical b interval. This is not a calibrated posterior.
    """
    values, template, covariance = [np.asarray(x, float) for x in (values, template, covariance)]
    if (values.ndim != 1 or len(values) < 2 or template.shape != values.shape
            or covariance.shape != (len(values), len(values))
            or any(not np.isfinite(x).all() for x in (values, template, covariance))):
        raise ValueError('Need >=2 matching finite angular bins and their covariance')
    if not np.allclose(covariance, covariance.T, rtol=1e-10, atol=0):
        raise ValueError('Covariance must be symmetric')
    covariance = (covariance+covariance.T)/2
    try:
        chol = np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError as exc:
        raise ValueError('Covariance must be positive definite; no automatic regularization') from exc
    wt, wy = np.linalg.solve(chol, template), np.linalg.solve(chol, values)
    information = wt @ wt
    if not np.isfinite(information) or information <= 0:
        raise ValueError('Template has no finite amplitude information')
    amplitude = float((wt @ wy)/information)
    sigma_a = float(information**-.5)
    bias = float(np.sqrt(amplitude)) if amplitude > 0 else None
    residual = wy-amplitude*wt
    return dict(bias_squared=amplitude, sigma_bias_squared=sigma_a, bias=bias,
                sigma_bias=sigma_a/(2*bias) if bias is not None else None,
                bias_interval_delta_chi2_1=[float(np.sqrt(max(0, amplitude-sigma_a))),
                                          float(np.sqrt(amplitude+sigma_a))]
                if amplitude+sigma_a > 0 else None,
                usable=amplitude > 0, chi2=float(residual @ residual), ndof=len(values)-1,
                covariance_condition=float(np.linalg.cond(covariance)))


def load_npz(path):
    with np.load(path, allow_pickle=False) as data:
        result = {key: data[key] for key in data.files}
    metadata = json.loads(str(result.pop('metadata_json')))
    return result, metadata


def fingerprint(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return dict(path=str(Path(path).resolve()), sha256=digest.hexdigest())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--measurement', required=True, type=Path, help='Spectroscopic angular AUTO NPZ; see README.md')
    parser.add_argument('--power', required=True, type=Path, help='Saved Planck15 total-matter Halofit NPZ')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--rmin', type=float, default=2.5)
    parser.add_argument('--rmax', type=float, default=10.)
    parser.add_argument('--radial-order', type=int, default=8)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    measured, meta = load_npz(args.measurement)
    grid, power_meta = load_npz(args.power)
    if meta.get('quantity') != 'spectroscopic_angular_autocorrelation':
        raise ValueError('Input must be a spectroscopic AUTO measurement, not the photometric cross-correlation')
    for key in ('tracer', 'sample_id', 'weights', 'randoms', 'geometry_z', 'zmin', 'zmax'):
        if key not in meta:
            raise ValueError(f'Missing measurement metadata: {key}')
    if meta.get('cosmology') != 'Planck15' or power_meta.get('cosmology') != 'Planck15':
        raise ValueError('This legacy adapter requires Planck15 geometry for both inputs')
    if (power_meta.get('model') != 'halofit_takahashi' or power_meta.get('matter_field') != 'total'
            or power_meta.get('k_unit') != 'h/Mpc' or power_meta.get('power_unit') != '(Mpc/h)^3'):
        raise ValueError('Require total-matter Takahashi Halofit with explicit h units')
    edges = increasing(measured['theta_edges_rad'], 'theta_edges_rad', positive=True)
    zedges = increasing(measured['z_edges'], 'z_edges')
    if (not np.allclose(zedges[[0, -1]], [meta['zmin'], meta['zmax']], rtol=0, atol=1e-10)
            or not meta['zmin'] <= meta['geometry_z'] < meta['zmax']):
        raise ValueError('Selection histogram and geometry redshift must match the measurement bin')
    values, covariance = measured['w_auto'], measured['covariance']
    if values.shape != (len(edges)-1,) or covariance.shape != (len(values), len(values)):
        raise ValueError('Autocorrelation/covariance must match the angular annuli')
    if not 0 < args.rmin < args.rmax:
        raise ValueError('Require 0 < rmin < rmax')
    from astropy.cosmology import Planck15
    radii = edges*Planck15.comoving_distance(meta['geometry_z']).value*Planck15.h
    selected = (radii[:-1] >= args.rmin*(1-1e-8)) & (radii[1:] <= args.rmax*(1+1e-8))
    if selected.sum() < 2:
        raise ValueError('Need >=2 complete annuli within the requested physical scale range')
    template = make_template(grid, edges, zedges, measured['nz_counts'], args.radial_order)
    # A radial quadrature check is inexpensive relative to generating the power grid.
    refined = make_template(grid, edges, zedges, measured['nz_counts'], 2*args.radial_order)
    norm = np.max(np.abs(refined[selected]))
    relative_change = float(np.max(np.abs(template[selected]-refined[selected]))/norm) if norm else float('inf')
    if not np.isfinite(relative_change) or relative_change > .01:
        raise ValueError('Radial projection changes by >1% on refinement; increase --radial-order')
    result = fit_bias(values[selected], refined[selected], covariance[np.ix_(selected, selected)])
    result.update(measurement=meta, power=power_meta,
                  inputs=dict(measurement=fingerprint(args.measurement), power=fingerprint(args.power)),
                  fit_indices=np.flatnonzero(selected).tolist(), r_edges_mpc_over_h=radii.tolist(),
                  matter_auto=refined.tolist(), predicted_auto=(result['bias_squared']*refined).tolist(),
                  radial_refinement_relative_change=relative_change,
                  radial_order_used=2*args.radial_order,
                  model='Eq.5.9: scale-independent b_s^2 times annulus-averaged nonlinear total matter',
                  uncertainty='conditional fixed-covariance fit; template/cosmology errors excluded',
                  numerical_validation='radial quadrature checked; k range/resolution require convergence checks',
                  corrections='no RSD, magnification, integral constraint, or explicit survey-window convolution')
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(f"b_s={result['bias']} +/- {result['sigma_bias']}; "
          f"chi2/ndof={result['chi2']:.4g}/{result['ndof']}; usable={result['usable']}")


if __name__ == '__main__':
    main()
