"""Per-tracer kernels, full-covariance inverse-variance combination, log-z rebinning."""
from pathlib import Path
import numpy as np
from .cache_io import _read_manifest
from .products import read_arrays

DEFAULT = 'takahashi_refined'
TRACERS = ('BGS_BRIGHT', 'LRG', 'ELG_LOPnotqso', 'QSO')
LABELS = dict(BGS_BRIGHT='BGS', LRG='LRG', ELG_LOPnotqso='ELG', QSO='QSO')

def read_summary(directory, *, case=DEFAULT):
    directory = Path(directory)
    report, _ = _read_manifest(directory/'report.json')
    if not report.get('completed'):
        raise ValueError('completed calibration required')
    record = report['summaries'][case]
    if record['file'] != 'summary-'+case+'.npz':
        raise ValueError('summary filename differs')
    arrays = read_arrays(directory/record['file'], record, 16*1024**2)
    meta = record['metadata']
    value, support = arrays['value'], arrays['supported']
    selected, covariance = arrays['supported_component_indices'], arrays['covariance']
    if (meta['signature'] != report['signature'] or meta['case'] != case
            or value.shape != (len(arrays['region_ids'])+1, len(meta['components']))
            or support.shape != value.shape or support.dtype.kind != 'b'
            or not np.array_equal(selected, np.flatnonzero(np.all(support, axis=0)))
            or covariance.shape != (len(selected), len(selected))
            or not np.isfinite(covariance).all() or not np.isfinite(value[support]).all()
            or not np.isnan(value[~support]).all()):
        raise ValueError('invalid calibration summary')
    return arrays, meta

def combine_supported(values, covariance, components):
    """Combine an already supported vector; return the weights and covariance.

    values has full sample first, followed by deletion realizations. Components
    describe the columns of values AND covariance. Weights are estimated once
    from the supplied covariance and held fixed for every realization.
    """
    values = np.asarray(values, dtype=float)
    covariance = np.asarray(covariance, dtype=float)
    n = len(components)
    if (not 0 < n <= 256 or values.ndim != 2 or values.shape[1] != n
            or len(values) < 1 or covariance.shape != (n, n)
            or not np.isfinite(values).all() or not np.isfinite(covariance).all()
            or not np.array_equal(covariance, covariance.T)):
        raise ValueError('Finite supported values and matching symmetric covariance required')
    variance = np.diag(covariance)
    if np.any(variance <= 0):
        raise ValueError('Inverse-variance weights require strictly positive variances')
    groups = {}
    identities = set()
    for i, component in enumerate(components):
        edges = tuple(component['z_edges'])
        identity = (component['tracer'], edges)
        if (len(edges) != 2 or not np.isfinite(edges).all() or edges[0] >= edges[1]
                or identity in identities):
            raise ValueError('Distinct tracer/bin identities with valid redshift edges required')
        identities.add(identity)
        groups.setdefault(edges, []).append(i)
    edges = np.array(sorted(groups), dtype=float)
    if np.any(edges[1:, 0] < edges[:-1, 1]):
        raise ValueError('Different overlapping bin edges require explicit rebinning first')
    weights = np.zeros((len(edges), n))
    contributors = []
    for row, bounds in enumerate(edges):
        indices = groups[tuple(bounds)]
        # Equivalent to 1/variance, scaled to avoid overflowing its reciprocal.
        inverse_variance = variance[indices].min()/variance[indices]
        weights[row, indices] = inverse_variance/inverse_variance.sum()
        contributors.append(tuple(components[i]['tracer'] for i in indices))
    realizations = values @ weights.T
    combined_covariance = weights @ covariance @ weights.T
    combined_covariance = (combined_covariance + combined_covariance.T)/2
    if (not np.isfinite(combined_covariance).all()
            or np.any(np.diag(combined_covariance) < 0)):
        raise ValueError('Invalid propagated covariance')
    return dict(z=edges.mean(axis=1), z_edges=edges, half_width=np.diff(edges)[:, 0]/2,
        value=realizations[0], sigma=np.sqrt(np.diag(combined_covariance)),
        covariance=combined_covariance, realizations=realizations,
        weights=weights, contributors=contributors, input_components=components)


def log_rebin_supported(values, covariance, components, dlog10_z):
    """Average whole supported bins on a shared approximately logarithmic grid.

    Snap log-spaced edges to existing boundaries and retain each tracer's
    support boundaries, including internal gaps. Average the density b*phi
    with redshift-width weights, preserving its integral over covered bins.
    Return transformed realizations/covariance and the explicit linear map.
    No interpolation or splitting of measured bins is performed.
    """
    if isinstance(dlog10_z, (bool, np.bool_)) or not np.isfinite(dlog10_z) or dlog10_z <= 0:
        raise ValueError('dlog10_z must be positive and finite')
    values = np.asarray(values, dtype=float)
    covariance = np.asarray(covariance, dtype=float)
    n = len(components)
    if (not 0 < n <= 256 or values.ndim != 2 or values.shape[1] != n
            or covariance.shape != (n, n) or not np.isfinite(values).all()
            or not np.isfinite(covariance).all() or not np.array_equal(covariance, covariance.T)):
        raise ValueError('Finite supported inputs with matching covariance required')
    original_edges = np.asarray([c['z_edges'] for c in components], dtype=float)
    if (original_edges.shape != (n, 2) or not np.isfinite(original_edges).all()
            or np.any(original_edges <= 0) or np.any(original_edges[:, 1] <= original_edges[:, 0])):
        raise ValueError('Logarithmic bins require positive, increasing redshift edges')
    available = np.unique(original_edges)
    log_available = np.log10(available)
    span = log_available[-1]-log_available[0]
    if span/dlog10_z > 4096:
        raise ValueError('More than 4096 proposed log bins; use the native grid or a larger dlog10_z')
    targets = log_available[0]+np.arange(int(np.ceil(span/dlog10_z))+1)*dlog10_z
    upper = np.clip(np.searchsorted(log_available, targets), 1, len(available)-1)
    nearest = np.where(abs(targets-log_available[upper-1]) <= abs(targets-log_available[upper]), upper-1, upper)
    boundaries = set(available[nearest]) | {available[0], available[-1]}
    tracer_rows = {}
    for tracer in dict.fromkeys(c['tracer'] for c in components):
        rows = sorted((i for i, c in enumerate(components) if c['tracer'] == tracer),
                      key=lambda i: original_edges[i, 0])
        edges = original_edges[rows]
        if np.any(edges[1:, 0] < edges[:-1, 1]):
            raise ValueError('Overlapping or duplicate bins within a tracer')
        boundaries.update((edges[0, 0], edges[-1, 1]))
        for left, right in zip(edges[:-1], edges[1:]):
            if left[1] != right[0]:
                boundaries.update((left[1], right[0]))
        tracer_rows[tracer] = rows
    target_edges = np.array(sorted(boundaries))
    operators, rebinned_components = [], []
    for tracer, rows in tracer_rows.items():
        output_bin = 0
        for lo, hi in zip(target_edges[:-1], target_edges[1:]):
            inside = [i for i in rows if original_edges[i, 0] >= lo and original_edges[i, 1] <= hi]
            if not inside:
                continue
            edges = original_edges[inside]
            if (edges[0, 0] != lo or edges[-1, 1] != hi
                    or not np.array_equal(edges[:-1, 1], edges[1:, 0])):
                continue  # Never fill a gap or partially covered output bin.
            row = np.zeros(n)
            row[inside] = np.diff(edges)[:, 0]/(hi-lo)
            operators.append(row)
            rebinned_components.append(dict(tracer=tracer, output_bin=output_bin, z_edges=[lo, hi]))
            output_bin += 1
    if not operators:
        raise ValueError('No fully covered logarithmic bins')
    operator = np.stack(operators)
    if np.any(np.count_nonzero(operator, axis=0) != 1):
        raise ValueError('Aligned tracer boundaries required: rebinning would split or discard a supported bin')
    new_covariance = operator @ covariance @ operator.T
    new_covariance = (new_covariance+new_covariance.T)/2
    return values @ operator.T, new_covariance, rebinned_components, dict(
        dlog10_z=float(dlog10_z), target_edges=target_edges, operator=operator,
        input_components=components, convention='bin_width_mean_of_b_phi; snapped_log_edges')


def read_combined_curves(directory, *, case=DEFAULT, tracers=TRACERS, dlog10_z=None):
    """Load the small joint summary and return separate and combined curves.

    Unsupported bins are excluded before weighting. All original signed values
    and normalization are retained. Pass a tracer subset to change the average.
    With dlog10_z, first average each tracer over wider redshift bins using
    bin-width weights, then combine tracers using inverse marginal variances.
    """
    tracers = tuple(tracers)
    if not tracers or len(set(tracers)) != len(tracers) or any(t not in TRACERS for t in tracers):
        raise ValueError('Choose distinct tracer names from TRACERS')
    arrays, metadata = read_summary(directory, case=case)
    selected = arrays['supported_component_indices']
    all_components = metadata['components']
    rows = np.array([j for j, i in enumerate(selected) if all_components[i]['tracer'] in tracers], dtype=int)
    indices = selected[rows]
    components = [all_components[i] for i in indices]
    values = arrays['value'][:, indices]
    covariance = arrays['covariance'][np.ix_(rows, rows)]
    rebinning = None
    if dlog10_z is not None:
        values, covariance, components, rebinning = log_rebin_supported(
            values, covariance, components, dlog10_z)
    curves = {}
    for tracer in tracers:
        positions = np.array([j for j, c in enumerate(components) if c['tracer'] == tracer], dtype=int)
        edges = np.array([components[j]['z_edges'] for j in positions]).reshape(-1, 2)
        subcov = covariance[np.ix_(positions, positions)]
        omitted = [c['z_edges'] for i, c in enumerate(all_components)
                   if c['tracer'] == tracer and i not in selected]
        curves[tracer] = dict(z=edges.mean(axis=1), z_edges=edges,
            half_width=np.diff(edges)[:, 0]/2, value=values[0, positions],
            covariance=subcov, sigma=np.sqrt(np.diag(subcov)), omitted_z_edges=omitted)
    return dict(tracers=curves, combined=combine_supported(values, covariance, components),
                case=case, region_ids=arrays['region_ids'], metadata=metadata, rebinning=rebinning)


def plot_tracers_and_combined(directory, *, case=DEFAULT, tracers=TRACERS, dlog10_z=None):
    """Return (tracer_figure, combined_figure, results); works directly in Jupyter."""
    import matplotlib.pyplot as plt

    result = read_combined_curves(directory, case=case, tracers=tracers, dlog10_z=dlog10_z)
    curves = result['tracers']
    ncols = min(2, len(curves))
    fig, axes = plt.subplots((len(curves)+ncols-1)//ncols, ncols,
        figsize=(6*ncols, 3.6*((len(curves)+ncols-1)//ncols)), squeeze=False, layout='constrained')
    colors = dict(zip(TRACERS, ('#9467bd', '#17669b', '#288342', '#c56b20')))
    def draw(ax, curve, color, title):
        ax.errorbar(curve['z'], curve['value'], yerr=curve['sigma'],
            xerr=curve['half_width'], fmt='o', ms=3, capsize=2, lw=.8, color=color)
        ax.axhline(0, color='0.4', lw=.8)
        ax.set(xlabel='Redshift z', ylabel=r'Apparent $b_p(z)\,\phi_p(z)$', title=title)
        ax.grid(alpha=.15)
    for ax, (tracer, curve) in zip(axes.flat, curves.items()):
        draw(ax, curve, colors[tracer], LABELS[tracer]+' reference')
        if curve['omitted_z_edges']:
            ax.text(.02, .02, f"Bins without support: {len(curve['omitted_z_edges'])}",
                transform=ax.transAxes, fontsize=8, bbox=dict(facecolor='white', edgecolor='none', alpha=.85))
    for ax in axes.flat[len(curves):]:
        ax.set_visible(False)
    bin_label = '' if dlog10_z is None else f' — Δlog₁₀z ≈ {dlog10_z:g} (snapped edges)'
    fig.suptitle('Photometric ELG × DESI DR1 — '+case+bin_label)
    combined_fig, ax = plt.subplots(figsize=(11, 4.5), layout='constrained')
    draw(ax, result['combined'], '#222222', 'Combined apparent kernel: inverse-variance weighting'+bin_label)
    ax.text(.02, .03, 'Errors propagate the full cross-tracer covariance',
            transform=ax.transAxes, va='bottom', fontsize=9,
            bbox=dict(facecolor='white', edgecolor='none', alpha=.85))
    return fig, combined_fig, result
