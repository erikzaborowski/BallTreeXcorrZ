"""Prepare a new photometric footprint and DESI references using legacy helpers.

Real catalogs and tree construction require an approved compute allocation.
The coordinate-only base is kept separately from the weighted photometry so
the existing exact coordinate/order validation can be used without changes.
"""
import argparse
import json
from pathlib import Path
import pickle

import numpy as np
from astropy.io import fits
from sklearn.neighbors import BallTree

from prepare_elg_sdss import prepare_photometry
from prepare_elg_desi import DR1_ROOT, TRACERS, prepare_reference


def prepare_catalogs(phot, randoms, base, references, catalog_root=DR1_ROOT,
                     tracers=TRACERS, chunk_size=1000000):
    inputs = base / 'inputs'
    inputs.mkdir(parents=True, exist_ok=False)
    references.mkdir(parents=True, exist_ok=False)
    phot_map, phot_info = prepare_photometry(phot, inputs/'elg.fits', 1., 1909, chunk_size)
    rand_map, rand_info = prepare_photometry(randoms, inputs/'randoms.fits', 1., 1909, chunk_size)
    if np.any((phot_map > 0) & (rand_map == 0)):
        raise ValueError('Selected photometric pixels lack random support')
    support = (phot_map > 0) & (rand_map > 0)
    np.savez(inputs/'occupancy_nside8.npz', phot=phot_map, randoms=rand_map)
    common = dict(photometry=phot_info, randoms=rand_info, seed=1909,
                  selection='elg_mask == 0; unit weights; no additional ELG cuts',
                  reference_overlap='occupied photometric AND random RING NSIDE=8 pixels')
    (inputs/'selection.json').write_text(json.dumps(common, indent=2)+'\n')
    tasks = []
    for tracer in tracers:
        tracer_root = references / tracer.lower()
        tracer_root.mkdir()
        source = tracer_root / ('desi_'+tracer.lower()+'.fits')
        parent = prepare_reference(catalog_root, tracer, support, .05, 4., source, allow_empty=True)
        (tracer_root/'selection.json').write_text(json.dumps(dict(common, reference=parent), indent=2)+'\n')
        with fits.open(source, memmap=True) as hdus:
            data = hdus[1].data
            for bin_id in range(1, 80):
                lo, hi = bin_id/20., (bin_id+1)/20.
                keep = (data['Z'] >= lo) & (data['Z'] < hi)
                if not np.any(keep):
                    continue
                selected = data[keep]
                if not np.all(np.isfinite(selected['WEIGHT']) & (selected['WEIGHT'] > 0)):
                    raise ValueError('Reference weights must be finite and positive for the legacy estimator')
                dest = tracer_root / f'bin-{bin_id:03d}' / 'inputs'
                dest.mkdir(parents=True)
                fits.BinTableHDU(selected).writeto(dest/source.name)
                for name in ('elg.fits', 'randoms.fits', 'occupancy_nside8.npz'):
                    (dest/name).symlink_to((inputs/name).resolve())
                reference = dict(release='DR1', tracer=tracer, selected_rows=len(selected),
                    weights_applied='unit', preserved_weight_column='WEIGHT',
                    prepared_reference=str(source.resolve()), parent_selection=str((tracer_root/'selection.json').resolve()),
                    row_order=parent['row_order'], nonpositive_or_nonfinite_weights=0)
                meta = dict(common, reference=reference, zmin=lo, zmax=hi)
                (dest/'selection.json').write_text(json.dumps(meta, indent=2)+'\n')
                tasks.append((tracer, bin_id, lo, hi, len(selected)))
    tasks.sort(key=lambda row: (row[1], tracers.index(row[0])))
    if not tasks:
        raise ValueError('No nonempty DESI reference bins in the new footprint')
    (references/'tasks.txt').write_text(''.join('{}\t{}\t{:.2f}\t{:.2f}\t{}\n'.format(*row) for row in tasks))
    print(f'Prepared {len(tasks)} tracer/bin tasks on the new footprint', flush=True)


def build_trees(base):
    # Identical coordinate order, haversine metric and pickle protocol to cross_correlate.py.
    for catalog, name in (('elg.fits', 'elg-d1tree.p'), ('randoms.fits', 'randoms-r1tree.p')):
        with fits.open(base/'inputs'/catalog, memmap=True) as hdus:
            table = hdus[1].data
            coords = np.empty((len(table), 2))
            coords[:, 0] = table['DEC']*np.pi/180.
            coords[:, 1] = table['RA']*np.pi/180.
        tree = BallTree(coords, metric='haversine')
        with (base/name).open('xb') as stream:
            pickle.dump(tree, stream, protocol=4)
        print(f'Saved {name}: {len(coords):,} coordinates', flush=True)
        del tree, coords


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('catalogs')
    for name in ('phot', 'randoms', 'base', 'references'):
        prep.add_argument('--'+name, type=Path, required=True)
    prep.add_argument('--catalog-root', type=Path, default=DR1_ROOT)
    tree = sub.add_parser('trees')
    tree.add_argument('--base', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'catalogs':
        prepare_catalogs(args.phot, args.randoms, args.base, args.references, args.catalog_root)
    else:
        build_trees(args.base)


if __name__ == '__main__':
    main()
