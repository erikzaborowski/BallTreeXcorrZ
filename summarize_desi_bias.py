"""Summarize saved per-tracer spectroscopic bias fits, including incomplete bins.

Reads small JSON/NPZ results only. Bias errors are conditional fit errors;
this does not recompute cross-correlations or their joint bias uncertainties.
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    rows = []
    for line in (args.root / 'tasks.txt').read_text().splitlines():
        tracer, bin_id, lo, hi, count = line.split()
        directory = args.root / tracer.lower() / f'bin-{int(bin_id):03d}'
        row = dict(tracer=tracer, bin=int(bin_id), z_low=float(lo), z_high=float(hi),
                   n_spec=int(count), status='incomplete', bias=None,
                   sigma_bias_conditional_fit=None, chi2=None, ndof=None,
                   p_value_nominal=None, covariance_condition=None,
                   reused_lrg=tracer == 'LRG')
        path = directory / 'bias.json'
        if path.exists():
            fit = json.loads(path.read_text())
            meta = fit['measurement']
            if (meta['tracer'] != tracer or meta['n_data'] != int(count)
                    or not np.allclose([meta['zmin'], meta['zmax']], [float(lo), float(hi)])
                    or meta['weights'] != 'WEIGHT'):
                raise ValueError(f'Mismatched fit selection: {path}')
            with np.load(directory / 'auto/measurement.npz', allow_pickle=False) as data:
                inds = np.asarray(fit['fit_indices'], dtype=int)
                residual = data['w_auto'][inds] - np.asarray(fit['predicted_auto'])[inds]
                cov = data['covariance'][np.ix_(inds, inds)]
                calculated = float(residual @ np.linalg.solve(cov, residual))
            if not np.isclose(calculated, fit['chi2'], rtol=1e-8, atol=1e-9):
                raise ValueError(f'Saved chi2 mismatch: {path}')
            row.update(status='complete' if fit['usable'] else 'nonpositive_amplitude',
                       bias=fit['bias'], sigma_bias_conditional_fit=fit['sigma_bias'],
                       chi2=fit['chi2'], ndof=fit['ndof'],
                       p_value_nominal=float(chi2.sf(fit['chi2'], fit['ndof'])),
                       covariance_condition=fit['covariance_condition'])
        else:
            status = args.root / 'claims' / f'{tracer.lower()}-bin-{int(bin_id):03d}' / 'exit_code.txt'
            if status.exists() and int(status.read_text()) != 0:
                row['status'] = 'failed'
        rows.append(row)
    with (args.root / 'summary.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for tracer in ('BGS_BRIGHT', 'LRG', 'ELG_LOPnotqso', 'QSO'):
        group = [row for row in rows if row['tracer'] == tracer and row['status'] == 'complete']
        if not group:
            continue
        z = [(row['z_low']+row['z_high'])/2 for row in group]
        axes[0].errorbar(z, [row['bias'] for row in group],
                        yerr=[row['sigma_bias_conditional_fit'] for row in group],
                        fmt='o-', ms=3, capsize=2, label=tracer)
        axes[1].plot(z, [-np.log10(max(row['p_value_nominal'], np.finfo(float).tiny))
                         for row in group], 'o-', ms=3, label=tracer)
    axes[0].set_ylabel(r'$b_s$ (conditional fit error)')
    axes[0].legend(ncol=2)
    axes[1].set_ylabel(r'$-\log_{10}(p_{\rm nominal})$')
    axes[1].axhline(-np.log10(.05), color='gray', ls='--', label='p = 0.05')
    axes[1].set_xlabel('Redshift')
    for ax in axes:
        ax.grid(alpha=.2)
    fig.suptitle('DESI DR1 weighted spectroscopic bias fits: 2.52–10 h⁻¹ Mpc')
    fig.text(.5, .015, 'Full spatial jackknife covariance; nominal χ² p-values. '
             'LRG products reused. Poor fits remain diagnostic.', ha='center', fontsize=9)
    fig.tight_layout(rect=[0, .035, 1, .96])
    fig.savefig(args.root / 'bias_fits.png', dpi=160)
    plt.close(fig)
    for tracer in ('BGS_BRIGHT', 'LRG', 'ELG_LOPnotqso', 'QSO'):
        group = [row for row in rows if row['tracer'] == tracer]
        print(tracer, {status: sum(row['status'] == status for row in group)
                       for status in sorted(set(row['status'] for row in group))})


if __name__ == '__main__':
    main()
