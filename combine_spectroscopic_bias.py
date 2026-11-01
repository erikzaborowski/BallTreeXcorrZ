"""Joint legacy wbar and spectroscopic-bias jackknife, without a b*dN/dz conversion.

Refit b_s for each matching spatial deletion, including its weighted p(z).
Keep the full-sample auto covariance fixed as the GLS fitting metric. No
independent fit-error term is added to the resulting jackknife uncertainty.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from fit_spectroscopic_bias import (fit_bias, load_npz, main as fit_main,
                                    make_template_basis, templates_from_histograms)


def jackknife_covariance(samples, areas):
    samples, areas = np.asarray(samples, float), np.asarray(areas, float)
    if (samples.ndim != 2 or len(samples) < 2 or areas.shape != (len(samples),)
            or not np.isfinite(samples).all() or not np.isfinite(areas).all() or np.any(areas <= 0)):
        raise ValueError('Require finite aligned jackknife vectors and positive region areas')
    delta = samples-samples.mean(axis=0)
    return (delta.T*(1-areas/areas.sum())) @ delta


def joint_statistics(wbar, bias, jack_wbar, jack_bias, areas):
    jack_wbar, jack_bias = np.asarray(jack_wbar), np.asarray(jack_bias)
    if (jack_wbar.shape != jack_bias.shape or jack_wbar.ndim != 1
            or not np.isfinite([wbar, bias]).all() or bias <= 0 or np.any(jack_bias <= 0)):
        raise ValueError('Cross/bias samples must match; all fitted biases must be positive')
    cov = jackknife_covariance(np.column_stack((jack_wbar, jack_bias)), areas)
    quotient = jack_wbar/jack_bias
    qcov = jackknife_covariance(quotient[:, None], areas)
    return dict(wbar=wbar, bias=bias, wbar_over_bias=wbar/bias,
                jackknife_wbar=jack_wbar, jackknife_bias=jack_bias,
                jackknife_wbar_over_bias=quotient, covariance_wbar_bias=cov,
                sigma_wbar=np.sqrt(cov[0, 0]), sigma_bias=np.sqrt(cov[1, 1]),
                sigma_wbar_over_bias=np.sqrt(qcov[0, 0]))


def read_cross(path, auto, meta, phot_weight_column=None, photometry_validation=None):
    header = {}
    with path.open() as stream:
        for line in stream:
            if not line.startswith('#'): break
            key, sep, value = line[2:].strip().partition('=')
            if sep: header[key] = value
    expected = {'SPEC_WEIGHT_COLUMN': meta['weights'], 'PHOT_WEIGHT': phot_weight_column or 'unit',
                'NSIDE': '4', 'COSMO': 'Planck15', 'ERROR': 'jackknife-leave one out'}
    if any(header.get(k) != v for k, v in expected.items()):
        raise ValueError('Cross and auto weight/cosmology/jackknife conventions do not match')
    if (path.parent.parent/header['SPEC']).resolve() != Path(meta['sample_id']).resolve():
        raise ValueError('Cross and auto use different reference catalogs')
    if int(header['N_SPEC']) != meta['n_data'] or meta.get('photometric_weights') != 'unit':
        raise ValueError('Reference counts or photometric weight provenance do not match')
    if phot_weight_column:
        if photometry_validation is None:
            raise ValueError('Weighted photometry requires the coordinate/footprint validation report')
        report = json.loads(photometry_validation.read_text())
        if (report.get('coordinates_and_order_verified') is not True
                or report.get('occupancy_verified') is not True
                or report['photometry']['weight_column'] != phot_weight_column
                or report['original_photometry'] != meta['selection']['photometry']
                or report['randoms'] != meta['selection']['randoms']
                or header.get('PHOT_RANDOM_WEIGHT') != 'unit'
                or int(header['N_PHOT']) != report['photometry']['selected_rows']
                or int(header['N_PHOT_RANDOM']) != report['randoms']['selected_rows']
                or (path.parent.parent/header['PHOTO']).resolve() != Path(report['prepared_photometry']).resolve()
                or (path.parent.parent/header['PHOTO_RANDOM']).resolve() != Path(report['prepared_randoms']).resolve()):
            raise ValueError('Weighted cross does not match the validated original photometric footprint')
    # The original header places z1, z2, zmean on one line.
    redshifts = dict(token.split('=') for token in ('z1='+header['z1']).split())
    if not np.allclose([float(redshifts['z1']), float(redshifts['z2'])],
                       [meta['zmin'], meta['zmax']], rtol=0, atol=1e-10):
        raise ValueError('Cross/auto redshift bins differ')
    table = np.loadtxt(path, ndmin=2)
    nbin, njack = len(table), len(auto['region_area_counts'])
    if table.shape[1] != 13+nbin+3*njack:
        raise ValueError('Unexpected legacy cross table shape')
    if not np.array_equal(table[0, 13+nbin+2*njack:], auto['region_area_counts']):
        raise ValueError('Cross and auto jackknife region areas/order differ')
    # Exactly the ds/s integration from get_b_dndz.py, lower_ind=7.
    selected = (table[:, 4] >= 2.5*(1-1e-8)) & (table[:, 5] <= 10*(1+1e-8))
    if selected.sum() != len(auto['w_auto']):
        raise ValueError('Unexpected cross radial bins')
    from astropy.cosmology import Planck15
    auto_radii = auto['theta_edges_rad']*Planck15.comoving_distance(meta['geometry_z']).value*Planck15.h
    if not np.allclose(np.r_[table[selected, 4], table[selected, 5][-1]], auto_radii, rtol=1e-8):
        raise ValueError('Cross/auto physical scale ranges differ')
    integration = (table[selected, 5]-table[selected, 4])/table[selected, 3]
    return (table[selected, 6] @ integration,
            table[selected, 13+nbin:13+nbin+njack].T @ integration)


def combine_bin(cross, measurement, power, output, phot_weight_column=None, photometry_validation=None):
    output.mkdir(exist_ok=False)
    # Reuse all the existing full-fit validation and convergence checks.
    fit_main(['--measurement', str(measurement), '--power', str(power),
              '--output', str(output/'bias.json')])
    fit = json.loads((output/'bias.json').read_text())
    if not fit['usable']:
        raise ValueError('Full-sample spectroscopic bias is not usable')
    auto, meta = load_npz(measurement)
    grid, _ = load_npz(power)
    wbar, jack_wbar = read_cross(cross, auto, meta, phot_weight_column, photometry_validation)
    basis = make_template_basis(grid, auto['theta_edges_rad'], auto['z_edges'], fit['radial_order_used'])
    templates = templates_from_histograms(basis, auto['jackknife_nz_counts'])
    np.testing.assert_allclose(templates_from_histograms(basis, auto['nz_counts']), fit['matter_auto'], rtol=1e-12)
    selected = np.asarray(fit['fit_indices'])
    covariance = auto['covariance'][np.ix_(selected, selected)]
    if templates.shape != auto['jackknife_w_auto'].shape or len(templates) != len(jack_wbar):
        raise ValueError('Autocorrelation, histogram and cross jackknife samples do not align')
    fits = [fit_bias(values[selected], template[selected], covariance)
            for values, template in zip(auto['jackknife_w_auto'], templates)]
    if not all(item['usable'] for item in fits):
        raise ValueError('Nonpositive bias in a jackknife deletion; no deletion may be dropped')
    result = joint_statistics(wbar, fit['bias'], jack_wbar, [x['bias'] for x in fits], auto['region_area_counts'])
    meta.update(quantity='legacy_wbar_and_spectroscopic_bias',
                photometric_weights=phot_weight_column or 'unit',
                photometric_random_weights='unit',
                photometry_validation=str(photometry_validation.resolve()) if photometry_validation else None,
                cross_file=str(cross.resolve()), auto_file=str(measurement.resolve()),
                fit_metric='full-sample autocorrelation covariance held fixed for all GLS fits',
                uncertainty='paired spatial jackknife, centered on deletion mean; no added independent fit error',
                interpretation='wbar/b_s; NOT b*dN/dz; no matter-response or magnification correction',
                geometry='original cross uses native dz=0.01 scales; auto uses fixed full-bin mean-z angles')
    np.savez(output/'measurement.npz', **result, region_area_counts=auto['region_area_counts'],
             region_map_nside4=auto['region_map_nside4'],
             jackknife_chi2=np.array([x['chi2'] for x in fits]),
             jackknife_bias_squared=np.array([x['bias_squared'] for x in fits]),
             conditional_sigma_bias=fit['sigma_bias'], chi2=fit['chi2'], ndof=fit['ndof'],
             metadata_json=json.dumps(meta))
    print(f"wbar={wbar:.6g}; b_s={fit['bias']:.6g}; wbar/b_s={result['wbar_over_bias']:.6g} "
          f"+/- {result['sigma_wbar_over_bias']:.6g} (joint jackknife)", flush=True)


def collect(root, bins, compare_unit=True):
    products = [load_npz(root/f'bin-{i:03d}'/'joint/measurement.npz') for i in bins]
    first = products[0][0]
    for data, meta in products:
        for key in ('region_map_nside4', 'region_area_counts'):
            if not np.array_equal(data[key], first[key]):
                raise ValueError('Jackknife regions differ across bins')
        if (meta['weights'] != 'WEIGHT' or meta['tracer'] != products[0][1]['tracer']
                or meta['photometric_weights'] != products[0][1]['photometric_weights']):
            raise ValueError('Collector requires a single tracer and consistent weights')
    quantities = ('wbar', 'bias', 'wbar_over_bias')
    arrays = {key: np.array([d[key] for d, m in products]) for key in quantities}
    samples = {key: np.column_stack([d['jackknife_'+key] for d, m in products]) for key in quantities}
    covs = {key: jackknife_covariance(samples[key], first['region_area_counts']) for key in quantities}
    redshifts = np.array([[m['zmin'], m['zmax']] for d, m in products])
    if not np.allclose(redshifts, [[i*.05, (i+1)*.05] for i in bins], rtol=0, atol=1e-10):
        raise ValueError('Collected redshift bins do not match requested bin IDs')
    np.savez(root/'joint_measurements.npz', z_edges=redshifts, **arrays,
             **{'jackknife_'+k: v for k, v in samples.items()},
             **{'covariance_'+k: v for k, v in covs.items()},
             covariance_wbar_bias=jackknife_covariance(np.column_stack((samples['wbar'], samples['bias'])), first['region_area_counts']),
             region_area_counts=first['region_area_counts'], region_map_nside4=first['region_map_nside4'],
             metadata_json=json.dumps(dict(bins=bins, measurements=[m for d, m in products],
                                          covariance_wbar_bias_order='all wbar bins, then all bias bins')))
    with (root/'summary.csv').open('x', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['z_low', 'z_high', 'n_spec', 'wbar', 'sigma_wbar_jk', 'bias', 'sigma_bias_jk',
                         'wbar_over_bias', 'sigma_wbar_over_bias_joint_jk', 'sigma_bias_conditional_fit', 'chi2', 'ndof'])
        for d, m in products:
            writer.writerow([m['zmin'], m['zmax'], m['n_data'], d['wbar'], d['sigma_wbar'], d['bias'],
                             d['sigma_bias'], d['wbar_over_bias'], d['sigma_wbar_over_bias'],
                             d['conditional_sigma_bias'], d['chi2'], d['ndof']])
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
    for ax, key, label in zip(axes, quantities, (r'$\bar w$', r'$b_s$', r'$\bar w/b_s$')):
        ax.errorbar(redshifts.mean(axis=1), arrays[key], yerr=np.sqrt(np.diag(covs[key])), fmt='o', label='DESI WEIGHT; joint jackknife')
        ax.set_ylabel(label)
        ax.grid(alpha=.2)
    if compare_unit:
        unit = [np.loadtxt(Path(m['sample_id']).parent.parent/'wbar.txt', ndmin=2) for d, m in products]
        if any(x.shape != (1, 4) for x in unit):
            raise ValueError('Expected one-bin original wbar tables')
        unit = np.concatenate(unit)
        np.testing.assert_allclose(unit[:, :2], redshifts, rtol=0, atol=1e-10)
        axes[0].errorbar(redshifts.mean(axis=1), unit[:, 2], yerr=unit[:, 3], fmt='s', alpha=.6, label='Previous unit-weight cross')
    axes[0].legend()
    axes[-1].set_xlabel('Reference redshift')
    tracer, phot_weight = products[0][1]['tracer'], products[0][1]['photometric_weights']
    fig.suptitle(f'Photometric ELGs × DESI DR1 {tracer}; photometric weights = {phot_weight}')
    fig.tight_layout()
    fig.savefig(root/f'weighted_{tracer.lower()}.png', dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    one = commands.add_parser('bin')
    for key in ('cross', 'measurement', 'power', 'output'):
        one.add_argument('--'+key, type=Path, required=True)
    one.add_argument('--phot-weight-column')
    one.add_argument('--photometry-validation', type=Path)
    many = commands.add_parser('collect')
    many.add_argument('--root', type=Path, required=True)
    many.add_argument('--bins', type=int, nargs='+', required=True)
    many.add_argument('--no-unit-comparison', action='store_true', help='New footprint with no earlier unit-weight cross')
    args = parser.parse_args()
    if args.command == 'bin':
        combine_bin(args.cross, args.measurement, args.power, args.output,
                    args.phot_weight_column, args.photometry_validation)
    else:
        collect(args.root, args.bins, compare_unit=not args.no_unit_comparison)


if __name__ == '__main__':
    main()
