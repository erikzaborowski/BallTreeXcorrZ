"""Attach photometric weights while verifying exact cached coordinates and footprint.

Run on compute for real catalogs. The existing random catalog and trees are
reusable only after every selected RA/DEC row passes the comparison below.
"""
import argparse
import json
from pathlib import Path

import healpy as hp
import numpy as np
from astropy.io import fits

from prepare_elg_sdss import prepare_photometry


def prepare(source, cached, output, weight_column, chunk_size=1000000):
    previous = json.loads((cached / 'selection.json').read_text())
    if (source.resolve() != Path(previous['photometry']['source']).resolve()
            or previous['selection'] != 'elg_mask == 0; unit weights; no additional ELG cuts'):
        raise ValueError('Expected the original masked, unit-weight photometric preparation')
    output.mkdir(parents=True, exist_ok=False)
    before = source.stat()
    occupancy, info = prepare_photometry(source, output / 'elg.fits', 1., previous['seed'],
                                         chunk_size, weight_column=weight_column)
    if (source.stat().st_size, source.stat().st_mtime_ns) != (before.st_size, before.st_mtime_ns):
        raise ValueError('Source changed during preparation')
    weighted_map = np.zeros(hp.nside2npix(8))
    sumw, sumw2, minimum, maximum, zeros = 0., 0., np.inf, -np.inf, 0
    with fits.open(output / 'elg.fits', memmap=True) as new, fits.open(cached / 'elg.fits', memmap=True) as old:
        data, original = new[1].data, old[1].data
        if len(data) != len(original) or len(data) != previous['photometry']['selected_rows']:
            raise ValueError('Selected row count changed; cached trees/autos cannot be reused')
        for start in range(0, len(data), chunk_size):
            a, b = data[start:start+chunk_size], original[start:start+chunk_size]
            if any(not np.array_equal(a[key], b[key]) for key in ('RA', 'DEC')):
                raise ValueError('Selected coordinates/order changed; cached trees/autos cannot be reused')
            weights = np.asarray(a[weight_column], float)
            sumw += weights.sum()
            sumw2 += weights @ weights
            minimum, maximum = min(minimum, weights.min()), max(maximum, weights.max())
            zeros += np.count_nonzero(weights == 0)
            weighted_map += np.bincount(hp.ang2pix(8, a['RA'], a['DEC'], lonlat=True),
                                       weights=weights, minlength=len(weighted_map))
    with np.load(cached / 'occupancy_nside8.npz') as maps:
        if not np.array_equal(occupancy, maps['phot']):
            raise ValueError('Photometric footprint changed')
        if np.any((occupancy > 0) & (weighted_map <= 0)):
            raise ValueError('An occupied photometric pixel has zero total weight')
        np.savez(output / 'occupancy_nside8.npz', phot=occupancy, randoms=maps['randoms'])
    (output / 'randoms.fits').symlink_to((cached / 'randoms.fits').resolve())
    info.update(weight_column=weight_column, sum_weights=float(sumw), sum_weights_squared=float(sumw2),
                minimum_weight=float(minimum), maximum_weight=float(maximum), zero_weights=int(zeros),
                source_size=before.st_size, source_mtime_ns=before.st_mtime_ns)
    meta = dict(photometry=info, randoms=previous['randoms'], seed=previous['seed'],
                selection=f'elg_mask == 0; photometric weight {weight_column}; unit random weights; no additional ELG cuts',
                cached_inputs=str(cached.resolve()), coordinates_and_order_verified=True,
                occupancy_verified=True, original_photometry=previous['photometry'],
                prepared_photometry=str((output / 'elg.fits').resolve()),
                prepared_randoms=str((output / 'randoms.fits').resolve()))
    (output / 'selection.json').write_text(json.dumps(meta, indent=2)+'\n')
    print(json.dumps(meta, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phot', type=Path, required=True)
    parser.add_argument('--cached-inputs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--weight-column', default='WEIGHT_SN_DEBV0')
    args = parser.parse_args()
    prepare(args.phot, args.cached_inputs, args.output, args.weight_column)


if __name__ == '__main__':
    main()
