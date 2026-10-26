"""Prepare unit-weight FITS inputs for the original clustering-z scripts.

Run on a compute node for the real ELG catalogues. No pair counting here.
"""
import argparse
import json
from pathlib import Path

import healpy as hp
import numpy as np
from astropy.io import fits


def prepare_photometry(source, destination, fraction, seed, chunk_size):
    # Stream selected columns to disk; do not copy the 37 GB input table.
    dtype = np.dtype([('RA', '<f8'), ('DEC', '<f8')])
    raw_path = destination.with_suffix('.raw')
    occupancy = np.zeros(hp.nside2npix(8), dtype=np.int64)
    rng = np.random.RandomState(seed)
    selected = 0
    with fits.open(str(source), memmap=True) as hdus, raw_path.open('wb') as raw:
        data = hdus[1].data
        for start in range(0, len(data), chunk_size):
            chunk = data[start:start + chunk_size]
            keep = chunk['elg_mask'] == 0
            if fraction < 1:
                keep &= rng.random_sample(len(chunk)) < fraction
            rows = np.empty(np.count_nonzero(keep), dtype=dtype)
            for name in dtype.names:
                rows[name] = chunk[name][keep]
            if (not np.all(np.isfinite(rows['RA'])) or
                    not np.all(np.isfinite(rows['DEC'])) or
                    np.any((rows['RA'] < 0) | (rows['RA'] >= 360)) or
                    np.any(np.abs(rows['DEC']) > 90)):
                raise ValueError('Invalid selected coordinates in ' + str(source))
            pixels = hp.ang2pix(8, rows['RA'], rows['DEC'], lonlat=True)
            occupancy += np.bincount(pixels, minlength=len(occupancy))
            rows.tofile(raw)
            selected += len(rows)
        input_rows = len(data)
    if not selected:
        raise ValueError('No selected rows in ' + str(source))
    # FITS conversion can allocate another copy of the compact RA/DEC table.
    rows = np.memmap(str(raw_path), mode='r', dtype=dtype, shape=(selected,))
    fits.BinTableHDU(rows).writeto(str(destination))
    del rows
    raw_path.unlink()
    print('{}: {} -> {} rows'.format(source, input_rows, selected), flush=True)
    return occupancy, {'source': str(source.resolve()), 'input_rows': input_rows,
                       'selected_rows': selected, 'random_fraction': fraction}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phot', type=Path)
    parser.add_argument('randoms', type=Path)
    parser.add_argument('reference', type=Path)
    parser.add_argument('output', type=Path, help='New directory; must not exist')
    parser.add_argument('--zmin', type=float, required=True)
    parser.add_argument('--zmax', type=float, required=True)
    parser.add_argument('--random-fraction', type=float, default=1.)
    parser.add_argument('--seed', type=int, default=1909)
    parser.add_argument('--chunk-size', type=int, default=1000000)
    parser.add_argument('--prepared-inputs', type=Path,
                        help='Reuse elg.fits, randoms.fits, selection.json and occupancy maps from a previous run')
    args = parser.parse_args()
    if not (0 <= args.zmin < args.zmax and 0 < args.random_fraction <= 1
            and args.chunk_size > 0):
        parser.error('Require 0 <= zmin < zmax, 0 < random-fraction <= 1, chunk-size > 0')
    args.output.mkdir(parents=True, exist_ok=False)
    if args.prepared_inputs:
        cached = args.prepared_inputs.resolve()
        previous = json.loads((cached / 'selection.json').read_text())
        phot_info, rand_info = previous['photometry'], previous['randoms']
        if (args.phot.resolve() != Path(phot_info['source']).resolve() or
                args.randoms.resolve() != Path(rand_info['source']).resolve() or
                args.random_fraction != rand_info['random_fraction'] or
                args.seed != previous['seed'] or
                previous['selection'] != 'elg_mask == 0; unit weights; no additional ELG cuts'):
            raise ValueError('Prepared catalogues do not match the requested sample/sampling settings')
        for name, info in [('elg.fits', phot_info), ('randoms.fits', rand_info)]:
            with fits.open(str(cached / name), memmap=True) as hdus:
                if hdus[1].header['NAXIS2'] != info['selected_rows']:
                    raise ValueError('Prepared catalogue row count changed: ' + name)
            (args.output / name).symlink_to(cached / name)
        with np.load(str(cached / 'occupancy_nside8.npz')) as maps:
            phot_map, rand_map = maps['phot'].copy(), maps['randoms'].copy()
    else:
        phot_map, phot_info = prepare_photometry(
            args.phot, args.output / 'elg.fits', 1., args.seed, args.chunk_size)
        rand_map, rand_info = prepare_photometry(
            args.randoms, args.output / 'randoms.fits', args.random_fraction,
            args.seed, args.chunk_size)
    # The legacy estimator indexes both catalogues through the random pixels.
    # Fail instead of silently assigning unsupported ELGs to pixel index zero.
    if np.any((phot_map > 0) & (rand_map == 0)):
        raise ValueError('ELG NSIDE=8 pixels without randoms; check masks or increase random fraction')
    support = (phot_map > 0) & (rand_map > 0)
    with fits.open(str(args.reference), memmap=True) as hdus:
        data = hdus[1].data
        ra, dec, z = (np.asarray(data[name], dtype=float) for name in ('RA', 'DEC', 'Z'))
        valid = (np.isfinite(ra) & np.isfinite(dec) & np.isfinite(z) &
                 (ra >= 0) & (ra < 360) & (np.abs(dec) <= 90))
        keep = valid & (z > 0) & (z >= args.zmin) & (z < args.zmax)
        candidates = np.flatnonzero(keep)
        keep[candidates] &= support[hp.ang2pix(8, ra[candidates], dec[candidates], lonlat=True)]
        if not np.any(keep):
            raise ValueError('No references in the requested redshift range and overlap')
        columns = [fits.Column(name=name, format='D', array=values[keep])
                   for name, values in [('RA', ra), ('DEC', dec), ('Z', z)]]
        if 'Sample_ID' in data.names:
            columns.append(fits.Column(name='Sample_ID', format='I', array=data['Sample_ID'][keep]))
        fits.BinTableHDU.from_columns(columns).writeto(str(args.output / 'sdss.fits'))
        ref_info = {'source': str(args.reference.resolve()), 'input_rows': len(data),
                    'in_redshift_range': len(candidates), 'selected_rows': int(keep.sum())}
    np.savez(str(args.output / 'occupancy_nside8.npz'), phot=phot_map, randoms=rand_map)
    metadata = {'photometry': phot_info, 'randoms': rand_info, 'reference': ref_info,
                'selection': 'elg_mask == 0; unit weights; no additional ELG cuts',
                'reference_overlap': 'occupied photometric AND random RING NSIDE=8 pixels',
                'zmin': args.zmin, 'zmax': args.zmax, 'seed': args.seed}
    (args.output / 'selection.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == '__main__':
    main()
