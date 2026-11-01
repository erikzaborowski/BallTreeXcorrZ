"""Plot saved autocorrelation fits; no catalogs or power projections are loaded."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator
import numpy as np
from scipy.stats import chi2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', type=Path)
    parser.add_argument('--bins', type=int, nargs='+', default=[8, 12, 16, 20])
    parser.add_argument('--output-prefix', type=Path, default=Path('weighted_lrg_fit_diagnostics'))
    args = parser.parse_args()
    with (args.run_dir / 'summary.csv').open() as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row['p_value_nominal'] = float(chi2.sf(float(row['chi2']), int(row['ndof'])))
    prefix = args.output_prefix
    prefix.parent.mkdir(parents=True, exist_ok=True)
    with Path(str(prefix) + '.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    fig, axes = plt.subplots(2, len(args.bins), figsize=(3.6*len(args.bins), 6.4),
                             squeeze=False, sharex='col',
                             gridspec_kw={'height_ratios': [2.2, 1]})
    diagnostics = []
    for col, bin_id in enumerate(args.bins):
        directory = args.run_dir / f'bin-{bin_id:03d}'
        fit = json.loads((directory / 'joint/bias.json').read_text())
        with np.load(directory / 'auto/measurement.npz', allow_pickle=False) as data:
            selected = np.asarray(fit['fit_indices'], dtype=int)
            values = data['w_auto'][selected]
            covariance = data['covariance'][np.ix_(selected, selected)]
        predicted = np.asarray(fit['predicted_auto'])[selected]
        residual = values - predicted
        calculated_chi2 = float(residual @ np.linalg.solve(covariance, residual))
        if not np.isclose(calculated_chi2, fit['chi2'], rtol=1e-9, atol=1e-10):
            raise ValueError(f'Bin {bin_id}: saved chi2 does not match data/model/covariance')
        p_value = float(chi2.sf(calculated_chi2, fit['ndof']))
        edges = np.asarray(fit['r_edges_mpc_over_h'])
        centers = np.sqrt(edges[:-1] * edges[1:])[selected]
        sigma = np.sqrt(np.diag(covariance))
        meta = fit['measurement']
        ax, res_ax = axes[:, col]
        ax.errorbar(centers, values, yerr=sigma, fmt='o', color='#14669b',
                    capsize=3, label='Weighted LRG data', zorder=3)
        # Horizontal segments show the annulus averages actually fitted.
        for i, index in enumerate(selected):
            ax.hlines(predicted[i], edges[index], edges[index+1],
                      color='#ce5a21', linewidth=2.3,
                      label=r'Best-fit $b_s^2 T$' if i == 0 else None)
        ax.set_title(f"{meta['zmin']:.2f} ≤ z < {meta['zmax']:.2f}", fontsize=12)
        ax.text(.97, .96,
                f"$b_s$ = {fit['bias']:.3f}\n"
                f"$\\chi^2$/dof = {calculated_chi2:.2f}/{fit['ndof']}\n"
                f"$p_{{\\rm nominal}}$ = {p_value:.3g}",
                transform=ax.transAxes, ha='right', va='top', fontsize=10)
        ax.margins(y=.25)
        res_ax.axhline(0, color='#ce5a21', linewidth=1.5)
        res_ax.errorbar(centers, residual/sigma, yerr=np.ones(len(values)),
                        fmt='o', color='#14669b', capsize=3)
        res_ax.set_ylim(-3.8, 3.8)
        res_ax.set_xlabel(r'$r_\perp$ at slice mean $z$ [$h^{-1}$ Mpc]')
        for panel in (ax, res_ax):
            panel.set_xscale('log')
            panel.set_xticks([3, 5, 8, 10])
            panel.set_xticklabels(['3', '5', '8', '10'])
            panel.xaxis.set_minor_locator(NullLocator())
            panel.grid(alpha=.15)
        diagnostics.append(dict(bin_id=bin_id, z_low=meta['zmin'], z_high=meta['zmax'],
                                bias=fit['bias'], chi2=calculated_chi2, ndof=fit['ndof'],
                                p_value_nominal=p_value, residual_over_sigma=(residual/sigma).tolist()))
    axes[0, 0].set_ylabel(r'$w_{ss}(\theta)$')
    axes[1, 0].set_ylabel(r'$(w_{ss}-b_s^2 T)/\sqrt{C_{ii}}$')
    axes[0, 0].legend(loc='lower left', fontsize=9)
    fig.suptitle('DESI DR1 weighted LRG autocorrelation: saved constant-bias fits', fontsize=15)
    fig.text(.5, .025,
             'Segments: annulus-averaged projected Halofit model (2.52–10 h⁻¹ Mpc). '
             'Error bars: √Cᵢᵢ; residuals are correlated.\n'
             'χ² uses the full 74-region jackknife covariance. Nominal p-values assume χ² with 2 dof; '
             'covariance uncertainty is not calibrated.', ha='center', fontsize=10)
    fig.tight_layout(rect=[0, .095, 1, .94])
    for suffix in ('.png', '.pdf'):
        fig.savefig(str(prefix) + suffix, dpi=180)
    plt.close(fig)
    Path(str(prefix) + '.json').write_text(json.dumps(dict(run_dir=str(args.run_dir.resolve()),
        diagnostics=diagnostics, p_value_definition='scipy.stats.chi2.sf(chi2, ndof); nominal fixed-covariance approximation'), indent=2) + '\n')
    print(json.dumps(diagnostics, indent=2))
    print(f"Bins with nominal p < 0.05: {sum(row['p_value_nominal'] < .05 for row in rows)}/{len(rows)}")


if __name__ == '__main__':
    main()
