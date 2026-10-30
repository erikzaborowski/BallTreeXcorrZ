"""Prepare DESI DR1 references for the unit-weight legacy measurement.

Run on a compute node for real catalogues. Reuse the SDSS pilot's selected
photometry; the DESI WEIGHT column is retained for provenance, not applied.
"""
import argparse
import json
from pathlib import Path

import healpy as hp
import numpy as np
from astropy.io import fits

from prepare_elg_sdss import reuse_photometry


DR1_ROOT = Path('/global/cfs/cdirs/desi/public/dr1/survey/catalogs/dr1/LSS/iron/LSScats/v1.5')
TRACERS = ('BGS_BRIGHT', 'LRG', 'ELG_LOPnotqso', 'QSO')


def prepare_reference(root, tracer, support, zmin, zmax, destination, allow_empty=False):
    """Merge NGC then SGC, rejecting repeated TARGETIDs rather than guessing."""
    dtype = np.dtype([('RA', '<f8'), ('DEC', '<f8'), ('Z', '<f8'),
                      ('TARGETID', '<i8'), ('WEIGHT', '<f8')])
    selected, all_ids, sources = [], [], []
    for cap in ('NGC', 'SGC'):
        source = root / (tracer + '_' + cap + '_clustering.dat.fits')
        with fits.open(str(source), memmap=True) as hdus:
            data = hdus[1].data
            missing = set(dtype.names) - set(data.names)
            if missing:
                raise ValueError('Missing columns in {}: {}'.format(source, sorted(missing)))
            if data['TARGETID'].dtype.kind not in 'iu':
                raise ValueError('TARGETID must be an integer column: ' + str(source))
            all_ids.append(np.array(data['TARGETID'], dtype=np.int64))
            ra, dec, z = (np.asarray(data[name], dtype=float) for name in ('RA', 'DEC', 'Z'))
            valid = (np.isfinite(ra) & np.isfinite(dec) & np.isfinite(z) &
                     (ra >= 0) & (ra < 360) & (np.abs(dec) <= 90))
            keep = valid & (z > 0) & (z >= zmin) & (z < zmax)
            candidates = np.flatnonzero(keep)
            keep[candidates] &= support[hp.ang2pix(8, ra[candidates], dec[candidates], lonlat=True)]
            rows = np.empty(np.count_nonzero(keep), dtype=dtype)
            for name in dtype.names:
                rows[name] = data[name][keep]
            selected.append(rows)
            sources.append({'cap': cap, 'source': str(source.resolve()),
                            'size_bytes': source.stat().st_size,
                            'mtime_ns': source.stat().st_mtime_ns,
                            'input_rows': len(data), 'valid_coordinates_redshift': int(valid.sum()),
                            'in_redshift_range': len(candidates), 'selected_rows': len(rows)})
    ids, counts = np.unique(np.concatenate(all_ids), return_counts=True)
    repeated = ids[counts > 1]
    if len(repeated):
        raise ValueError('{} duplicate TARGETIDs within/across caps; examples: {}'.format(
            len(repeated), repeated[:10].tolist()))
    rows = np.concatenate(selected)
    if not len(rows) and not allow_empty:
        raise ValueError('No references in the requested redshift range and overlap')
    fits.BinTableHDU(rows).writeto(str(destination))
    return {'release': 'DR1', 'tracer': tracer, 'sources': sources,
            'input_rows': sum(s['input_rows'] for s in sources),
            'in_redshift_range': sum(s['in_redshift_range'] for s in sources),
            'selected_rows': len(rows), 'duplicate_targetids': 0,
            'row_order': 'NGC then SGC, original order within each cap',
            'weights_applied': 'unit', 'preserved_weight_column': 'WEIGHT',
            'nonpositive_or_nonfinite_weights': int(np.count_nonzero(
                ~np.isfinite(rows['WEIGHT']) | (rows['WEIGHT'] <= 0)))}


def combine_references(catalogues, destination, redshift_conflicts='error'):
    """Unique TARGETID union; retain tracer membership and use unit weights.

    Each input is already unique within its tracer. Coordinates must agree.
    Optionally exclude all copies of IDs with conflicting redshifts, recording
    them in a sidecar FITS file. Original tracer files are never altered.
    """
    parts, memberships = [], []
    for tracer, path in catalogues:
        with fits.open(str(path), memmap=True) as hdus:
            parts.append(np.array(hdus[1].data))
            memberships.append(np.full(len(hdus[1].data), 1 << TRACERS.index(tracer), dtype=np.int16))
    rows = np.concatenate(parts)
    if not len(rows):
        raise ValueError('No references in the requested redshift range and overlap')
    ids, first, inverse, counts = np.unique(rows['TARGETID'], return_index=True,
                                           return_inverse=True, return_counts=True)
    if redshift_conflicts not in ('error', 'exclude'):
        raise ValueError('redshift_conflicts must be error or exclude')
    for name in ('RA', 'DEC'):
        consistent = np.isclose(rows[name], rows[name][first][inverse], rtol=0, atol=1.e-8)
        if not np.all(consistent):
            raise ValueError('Conflicting {} for shared TARGETIDs: {}'.format(
                name, rows['TARGETID'][~consistent][:10].tolist()))
    bits = np.zeros(len(ids), dtype=np.int16)
    np.bitwise_or.at(bits, inverse, np.concatenate(memberships))
    zmin, zmax = np.full(len(ids), np.inf), np.full(len(ids), -np.inf)
    np.minimum.at(zmin, inverse, rows['Z'])
    np.maximum.at(zmax, inverse, rows['Z'])
    conflict = zmax - zmin > 1.e-8
    if np.any(conflict) and redshift_conflicts == 'error':
        raise ValueError('Conflicting Z for shared TARGETIDs: {}'.format(ids[conflict][:10].tolist()))
    keep = ~conflict
    if not np.any(keep):
        raise ValueError('No references remain after redshift-conflict exclusions')
    excluded_name = None
    if redshift_conflicts == 'exclude':
        excluded_name = 'excluded_redshift_targetids.fits'
        fits.BinTableHDU.from_columns([
            fits.Column(name='TARGETID', format='K', array=ids[conflict]),
            fits.Column(name='Z_MIN', format='D', array=zmin[conflict]),
            fits.Column(name='Z_MAX', format='D', array=zmax[conflict]),
            fits.Column(name='TRACER_BITS', format='I', array=bits[conflict]),
        ]).writeto(str(destination.with_name(excluded_name)))
    columns = [fits.Column(name=name, format='D', array=rows[name][first[keep]])
               for name in ('RA', 'DEC', 'Z')]
    columns += [fits.Column(name='TARGETID', format='K', array=ids[keep]),
                fits.Column(name='WEIGHT', format='D', array=np.ones(np.count_nonzero(keep))),
                fits.Column(name='TRACER_BITS', format='I', array=bits[keep])]
    fits.BinTableHDU.from_columns(columns).writeto(str(destination))
    return {'release': 'DR1', 'tracer': 'ALL', 'selected_rows': int(keep.sum()),
            'rows_before_union': len(rows), 'shared_targetids': int(np.count_nonzero(counts > 1)),
            'duplicate_memberships_removed': len(rows) - len(ids),
            'unique_targets_before_exclusion': len(ids),
            'redshift_conflict_policy': redshift_conflicts,
            'excluded_redshift_targetids': int(conflict.sum()), 'excluded_catalog': excluded_name,
            'row_order': 'ascending TARGETID', 'weights_applied': 'unit',
            'weight_column': 'set to unity; original weights retained in component files',
            'tracer_bits': {tracer: 1 << i for i, tracer in enumerate(TRACERS)},
            'shared_id_tolerance': 'absolute 1e-8 in RA/DEC degrees and Z; no relative tolerance'}


def slice_combined(cached, destination, zmin, zmax, phot_info, rand_info):
    """Slice the global union so conflicts/duplicates cannot reappear in other bins."""
    cached = cached.resolve()
    metadata = json.loads((cached / 'selection.json').read_text())
    parent = metadata['reference']
    if (parent['tracer'] != 'ALL' or parent['weights_applied'] != 'unit' or
            metadata['photometry'] != phot_info or metadata['randoms'] != rand_info or
            metadata['selection'] != 'elg_mask == 0; unit weights; no additional ELG cuts' or
            not (metadata['zmin'] <= zmin < zmax <= metadata['zmax'])):
        raise ValueError('Prepared combined catalogue does not match this sample or redshift range')
    source = cached / 'desi_all.fits'
    with fits.open(str(source), memmap=True) as hdus:
        if hdus[1].header['NAXIS2'] != parent['selected_rows']:
            raise ValueError('Prepared combined catalogue row count changed')
        data = hdus[1].data
        keep = (data['Z'] >= zmin) & (data['Z'] < zmax)
        if not np.any(keep):
            raise ValueError('No prepared combined references in this redshift bin')
        fits.BinTableHDU(data[keep]).writeto(str(destination))
    return {'release': 'DR1', 'tracer': 'ALL', 'selected_rows': int(keep.sum()),
            'weights_applied': 'unit', 'prepared_reference': str(source),
            'prepared_selection': str(cached / 'selection.json'),
            'parent_selected_rows': parent['selected_rows'],
            'parent_excluded_redshift_targetids': parent.get('excluded_redshift_targetids', 0),
            'selection_order': 'global union and conflict exclusions before redshift binning'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phot', type=Path)
    parser.add_argument('randoms', type=Path)
    parser.add_argument('output', type=Path, help='New directory; must not exist')
    parser.add_argument('--prepared-inputs', type=Path, required=True)
    parser.add_argument('--catalog-root', type=Path, default=DR1_ROOT)
    parser.add_argument('--tracer', choices=TRACERS + ('ALL',), default='LRG')
    parser.add_argument('--redshift-conflicts', choices=('error', 'exclude'), default='error',
                        help='ALL only: policy for shared IDs with Z span above 1e-8')
    parser.add_argument('--prepared-combined-inputs', type=Path,
                        help='ALL only: slice a global prepared union instead of merging per bin')
    parser.add_argument('--zmin', type=float, required=True)
    parser.add_argument('--zmax', type=float, required=True)
    parser.add_argument('--random-fraction', type=float, default=1.)
    parser.add_argument('--seed', type=int, default=1909)
    args = parser.parse_args()
    if args.tracer != 'ALL' and (args.prepared_combined_inputs or args.redshift_conflicts != 'error'):
        parser.error('Combined-reference options require --tracer ALL')
    if args.prepared_combined_inputs and args.redshift_conflicts != 'error':
        parser.error('Conflict policy is already fixed in the prepared combined catalogue')
    if not (0 <= args.zmin < args.zmax and np.isfinite(args.zmax) and
            0 < args.random_fraction <= 1):
        parser.error('Require finite 0 <= zmin < zmax and 0 < random-fraction <= 1')
    args.output.mkdir(parents=True, exist_ok=False)
    phot_map, rand_map, phot_info, rand_info = reuse_photometry(
        args.phot, args.randoms, args.output, args.prepared_inputs,
        args.random_fraction, args.seed)
    if np.any((phot_map > 0) & (rand_map == 0)):
        raise ValueError('ELG NSIDE=8 pixels without randoms; check cached masks')
    support = (phot_map > 0) & (rand_map > 0)
    name = 'desi_' + args.tracer.lower() + '.fits'
    if args.prepared_combined_inputs:
        reference = slice_combined(args.prepared_combined_inputs, args.output / name,
                                   args.zmin, args.zmax, phot_info, rand_info)
    elif args.tracer == 'ALL':
        catalogues, components = [], {}
        for tracer in TRACERS:
            path = args.output / ('desi_' + tracer.lower() + '.fits')
            components[tracer] = prepare_reference(args.catalog_root, tracer, support,
                args.zmin, args.zmax, path, allow_empty=True)
            catalogues.append((tracer, path))
        reference = combine_references(catalogues, args.output / name, args.redshift_conflicts)
        reference['components'] = components
    else:
        reference = prepare_reference(args.catalog_root, args.tracer, support,
                                      args.zmin, args.zmax, args.output / name)
    np.savez(str(args.output / 'occupancy_nside8.npz'), phot=phot_map, randoms=rand_map)
    metadata = {'photometry': phot_info, 'randoms': rand_info, 'reference': reference,
                'selection': 'elg_mask == 0; unit weights; no additional ELG cuts',
                'reference_overlap': 'occupied photometric AND random RING NSIDE=8 pixels',
                'prepared_inputs': str(args.prepared_inputs.resolve()),
                'zmin': args.zmin, 'zmax': args.zmax, 'seed': args.seed}
    (args.output / 'selection.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == '__main__':
    main()
