"""Diagnose shared TARGETIDs in prepared DR1 tracer files; no pair counting.

Run on compute for survey-sized inputs. This reads only the four compact
reference tables, not the photometry, randoms, trees or original LSS tables.
"""
import argparse
import csv
import itertools
import json
from pathlib import Path

import numpy as np
from astropy.io import fits

TRACERS = ('BGS_BRIGHT', 'LRG', 'ELG_LOPnotqso', 'QSO')


def compare(left, right):
    # Preparation already checked uniqueness within each input tracer.
    ids, ia, ib = np.intersect1d(left['TARGETID'], right['TARGETID'],
                                assume_unique=True, return_indices=True)
    za, zb = left['Z'][ia], right['Z'][ib]
    delta = np.abs(za - zb)
    conflict = ~np.isclose(za, zb, rtol=0, atol=1.e-8)
    native_cross = (np.searchsorted(np.arange(401) / 100., za, side='right') !=
                    np.searchsorted(np.arange(401) / 100., zb, side='right'))
    output_cross = (np.searchsorted(np.arange(81) / 20., za, side='right') !=
                    np.searchsorted(np.arange(81) / 20., zb, side='right'))
    report = {'shared_targetids': len(ids), 'redshift_conflicts': int(conflict.sum()),
              'cross_native_dz001': int(native_cross.sum()),
              'cross_output_dz005': int(output_cross.sum())}
    for name in ('RA', 'DEC'):
        report[name.lower() + '_conflicts'] = int(np.count_nonzero(
            ~np.isclose(left[name][ia], right[name][ib], rtol=0, atol=1.e-8)))
    report['abs_delta_z_quantiles'] = dict(zip(('min', 'median', 'p90', 'p99', 'max'),
        np.quantile(delta, [0, .5, .9, .99, 1]).tolist())) if len(ids) else None
    report['conflicting_abs_delta_z_quantiles'] = dict(zip(('min', 'median', 'p90', 'p99', 'max'),
        np.quantile(delta[conflict], [0, .5, .9, .99, 1]).tolist())) if np.any(conflict) else None
    # Bounded examples include ordinary conflicts, largest differences and bin crossings.
    examples = np.unique(np.concatenate((np.flatnonzero(conflict)[:10],
        np.argsort(delta)[-10:] if np.any(conflict) else np.array([], dtype=int),
        np.flatnonzero(output_cross)[:10])))
    rows = [{'targetid': int(ids[i]), 'z_a': float(za[i]), 'z_b': float(zb[i]),
             'abs_delta_z': float(delta[i]), 'cross_native_dz001': bool(native_cross[i]),
             'cross_output_dz005': bool(output_cross[i])} for i in examples if conflict[i]]
    return report, rows, ids[conflict]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inputs', type=Path, help='Prepared inputs from the failed full-range run')
    parser.add_argument('output', type=Path, help='New report directory')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    tables, sources = {}, {}
    for tracer in TRACERS:
        path = args.inputs / ('desi_' + tracer.lower() + '.fits')
        with fits.open(path, memmap=True) as hdus:
            tables[tracer] = np.array(hdus[1].data)
        sources[tracer] = {'path': str(path.resolve()), 'rows': len(tables[tracer])}
    pairs, examples, conflicts = {}, [], []
    for a, b in itertools.combinations(TRACERS, 2):
        report, rows, ids = compare(tables[a], tables[b])
        pairs[a + '__' + b] = report
        examples.extend(dict(tracer_a=a, tracer_b=b, **row) for row in rows)
        conflicts.append(ids)
        print(a, b, json.dumps(report), flush=True)
    result = {'inputs': sources, 'absolute_tolerance': 1.e-8,
              'unique_targetids_with_conflicting_redshifts': len(np.unique(np.concatenate(conflicts))),
              'pairs': pairs}
    (args.output / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    with (args.output / 'examples.tsv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=['tracer_a', 'tracer_b', 'targetid',
            'z_a', 'z_b', 'abs_delta_z', 'cross_native_dz001', 'cross_output_dz005'], delimiter='\t')
        writer.writeheader()
        writer.writerows(examples)
    print('Reports:', args.output, flush=True)


if __name__ == '__main__':
    main()
