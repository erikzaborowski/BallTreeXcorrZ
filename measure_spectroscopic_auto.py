"""Angular reference autocorrelation with optional catalog weights and legacy BallTrees.

Uses the existing prepared DR1 selection and its NSIDE=8 overlap. DD/DR/RR
region matrices give exact region deletions without repeating pair counting.
Run real catalogs on an approved compute node. No photometric FITS are read.
"""
import argparse
import ast
import contextlib
import io
import json
from pathlib import Path
import time
from types import SimpleNamespace

import healpy as hp
import numpy as np
from astropy.cosmology import Planck15
from astropy.io import fits
from sklearn.neighbors import BallTree


def legacy_regions(occupancy, legacy_source):
    """Reuse the original equalize_pixels function, without executing its CLI.

    Aggregating the cached NSIDE=8 photometric-random map to NSIDE=4 exactly
    reproduces the counts used by the legacy cross-correlation jackknife.
    """
    parent = hp.ang2pix(4, *hp.pix2ang(8, np.arange(12*8**2)))
    area = np.bincount(parent, weights=occupancy, minlength=12*4**2).astype(np.int64)
    source = ast.parse(Path(legacy_source).read_text())
    function = next(node for node in source.body if isinstance(node, ast.FunctionDef)
                    and node.name == 'equalize_pixels')
    namespace = dict(np=np, hp=hp, ns=SimpleNamespace(nside=4))
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(legacy_source), 'exec'), namespace)
    with contextlib.redirect_stdout(io.StringIO()):
        counts, groups = namespace['equalize_pixels'](np.flatnonzero(area), area[area > 0])
    mapping = np.full(12*4**2, -1, dtype=int)
    for region, pixels in enumerate(groups):
        mapping[pixels] = region
    return mapping, np.asarray(counts, dtype=float)


def pair_matrix(left, right, left_regions, right_regions, edges, nregions,
                tree=None, batch_size=512, label='pairs', left_weights=None, right_weights=None):
    """Ordered pairs, theta in [low,high); exclude self pairs via low>0.

    As in cross_correlate.py: BallTree with haversine, coordinates (DEC,RA)
    in radians. Here bins are fixed angles across the whole reference slice.
    """
    if edges[0] <= 0 or np.any(np.diff(edges) <= 0):
        raise ValueError('Require increasing positive angular edges')
    if batch_size < 1:
        raise ValueError('batch_size must be positive')
    if tree is None:
        tree = BallTree(right, metric='haversine')
    nbins = len(edges)-1
    weighted = left_weights is not None or right_weights is not None
    lw = checked_weights(left_weights, len(left))
    rw = checked_weights(right_weights, len(right))
    result = np.zeros((nregions, nregions, nbins), dtype=float if weighted else np.int64)
    started = last_report = time.monotonic()
    for start in range(0, len(left), batch_size):
        stop = min(len(left), start+batch_size)
        indices, distances = tree.query_radius(left[start:stop], edges[-1], return_distance=True)
        lengths = np.fromiter((len(x) for x in indices), dtype=np.int64)
        if lengths.sum():
            right_index = np.concatenate(indices)
            angle = np.concatenate(distances)
            left_region = np.repeat(left_regions[start:stop], lengths)
            bins = np.searchsorted(edges, angle, side='right')-1
            keep = (bins >= 0) & (bins < nbins)
            address = ((left_region[keep]*nregions+right_regions[right_index[keep]])*nbins+bins[keep])
            pair_weights = np.repeat(lw[start:stop], lengths)[keep]*rw[right_index[keep]] if weighted else None
            result += np.bincount(address, weights=pair_weights, minlength=result.size).reshape(result.shape)
        now = time.monotonic()
        if now-last_report > 30 or stop == len(left):
            print(f'{label}: {stop:,}/{len(left):,} queries; {now-started:.1f}s', flush=True)
            last_report = now
    return result


def checked_weights(weights, size):
    weights = np.ones(size) if weights is None else np.asarray(weights, float)
    # DESI randoms can have zero completeness weight; they contribute no pairs.
    if weights.shape != (size,) or not np.all(np.isfinite(weights) & (weights >= 0)):
        raise ValueError('Weights must be finite, nonnegative and aligned with catalog rows')
    return weights


def landy_szalay(dd, dr, rr, data_counts, random_counts, data_weight2=None, random_weight2=None):
    """Full sample and region deletions; counts may be sums of object weights.

    Supply per-region sum(w**2) with weighted counts to exclude self pairs.
    """
    nregions = len(data_counts)
    nd, nr = np.r_[data_counts.sum(), data_counts.sum()-data_counts], np.r_[random_counts.sum(), random_counts.sum()-random_counts]
    d2 = data_counts if data_weight2 is None else np.asarray(data_weight2)
    r2 = random_counts if random_weight2 is None else np.asarray(random_weight2)
    ddnorm = nd**2-np.r_[d2.sum(), d2.sum()-d2]
    rrnorm = nr**2-np.r_[r2.sum(), r2.sum()-r2]
    if np.any(ddnorm <= 0) or np.any(rrnorm <= 0) or np.any(nd*nr <= 0):
        raise ValueError('Too few weighted data/random objects in a full or deleted sample')
    pair_vectors = []
    for matrix in (dd, dr, rr):
        total = matrix.sum(axis=(0, 1))
        deleted = total-matrix.sum(axis=0)-matrix.sum(axis=1)+matrix[np.arange(nregions), np.arange(nregions)]
        pair_vectors.append(np.vstack((total, deleted)))
    ddn = pair_vectors[0]/ddnorm[:, None]
    drn = pair_vectors[1]/(nd*nr)[:, None]
    rrn = pair_vectors[2]/rrnorm[:, None]
    if np.any(rrn <= 0):
        raise ValueError('No RR support for an angular bin or jackknife deletion')
    return (ddn-2*drn+rrn)/rrn


def read_randoms(paths, support, lo, hi, weight_column=None):
    parts, provenance = [], []
    for path in paths:
        with fits.open(path, memmap=True) as hdus:
            table = hdus[1].data
            rows = []
            for start in range(0, len(table), 250000):
                chunk = table[start:start+250000]
                ra, dec, z = (np.asarray(chunk[name], float) for name in ('RA', 'DEC', 'Z'))
                good = (np.isfinite(ra) & np.isfinite(dec) & np.isfinite(z)
                        & (ra >= 0) & (ra < 360) & (abs(dec) <= 90) & (z >= lo) & (z < hi))
                ra, dec, z = ra[good], dec[good], z[good]
                keep = support[hp.ang2pix(8, ra, dec, lonlat=True)]
                weight = checked_weights(chunk[weight_column][good][keep] if weight_column else None, keep.sum())
                rows.append(np.column_stack((ra[keep], dec[keep], z[keep], weight)))
            selected = np.concatenate(rows)
            parts.append(selected)
            provenance.append(dict(path=str(path.resolve()), input_rows=len(table), selected_rows=len(selected),
                                   size_bytes=path.stat().st_size, mtime_ns=path.stat().st_mtime_ns))
        print(f'Randoms: {path.name}: {len(selected):,} selected', flush=True)
    return np.concatenate(parts), provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared-run', type=Path, required=True)
    parser.add_argument('--randoms', type=Path, nargs='+', required=True)
    parser.add_argument('--output-dir', type=Path, required=True, help='New directory')
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--weight-column', help='Same column in spectroscopic data and randoms; default unit')
    args = parser.parse_args()
    started = time.monotonic()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    selection = json.loads((args.prepared_run/'inputs/selection.json').read_text())
    ref = selection['reference']
    if ref['weights_applied'] != 'unit':
        raise ValueError('Expected the original prepared reference selection; apply weights here')
    if args.weight_column and ref['tracer'] == 'ALL':
        raise ValueError('The deduplicated ALL catalog has no defined survey weights/randoms')
    lo, hi = selection['zmin'], selection['zmax']
    source = args.prepared_run/'inputs'/('desi_'+ref['tracer'].lower()+'.fits')
    with fits.open(source, memmap=True) as hdus:
        data = np.column_stack([hdus[1].data[key] for key in ('RA', 'DEC', 'Z')])
        dw = checked_weights(hdus[1].data[args.weight_column] if args.weight_column else None, len(data))
    if len(data) != ref['selected_rows'] or not np.all((data[:, 2] >= lo) & (data[:, 2] < hi)):
        raise ValueError('Prepared reference does not match its metadata')
    with np.load(args.prepared_run/'inputs/occupancy_nside8.npz') as maps:
        support = (maps['phot'] > 0) & (maps['randoms'] > 0)
        region_map, areas = legacy_regions(maps['randoms'], Path(__file__).with_name('error_estimator.py'))
    legacy_file = args.prepared_run/'correlations'/f'z{lo:.2f}_{hi:.2f}_jk_loo_equalized.txt'
    legacy = np.loadtxt(legacy_file)
    njack = (legacy.shape[1]-13-len(legacy))//3
    if not np.array_equal(areas, legacy[0, 13+len(legacy)+2*njack:]):
        raise ValueError('Reconstructed regions differ from the saved cross-correlation regions')
    random, random_info = read_randoms(args.randoms, support, lo, hi, args.weight_column)
    rw = random[:, 3]
    dregion, rregion = [region_map[hp.ang2pix(4, x[:, 0], x[:, 1], lonlat=True)] for x in (data, random)]
    if min(dregion.min(), rregion.min()) < 0:
        raise ValueError('Object outside the legacy jackknife regions')
    zmean = float(data[:, 2].mean())
    radii = np.geomspace(.1006799, 10., 11)[7:]
    angles = radii/(Planck15.comoving_distance(zmean).value*Planck15.h)
    drad, rrad = [np.deg2rad(x[:, [1, 0]]) for x in (data, random)]
    nd = np.bincount(dregion, minlength=njack)
    nr = np.bincount(rregion, minlength=njack)
    dsum, rsum = [np.bincount(reg, weights=w, minlength=njack) for reg, w in ((dregion, dw), (rregion, rw))]
    dsum2, rsum2 = [np.bincount(reg, weights=w*w, minlength=njack) for reg, w in ((dregion, dw), (rregion, rw))]
    print(f'Data={len(data):,}; randoms={len(random):,}; regions={njack}; zmean={zmean:.8f}', flush=True)
    dd = pair_matrix(drad, drad, dregion, dregion, angles, njack, batch_size=args.batch_size, label='DD', left_weights=dw, right_weights=dw)
    rtree = BallTree(rrad, metric='haversine')
    dr = pair_matrix(drad, rrad, dregion, rregion, angles, njack, tree=rtree, batch_size=args.batch_size, label='DR', left_weights=dw, right_weights=rw)
    rr = pair_matrix(rrad, rrad, rregion, rregion, angles, njack, tree=rtree, batch_size=args.batch_size, label='RR', left_weights=rw, right_weights=rw)
    values = landy_szalay(dd, dr, rr, dsum, rsum, dsum2, rsum2)
    delta = values[1:]-values[1:].mean(axis=0)
    cov = (delta.T*(1-areas/areas.sum())) @ delta
    zedges = np.linspace(lo, hi, 11)
    nz = np.histogram(random[:, 2], zedges, weights=rw)[0]
    hist_jk = np.stack([np.histogram(random[rregion != k, 2], zedges, weights=rw[rregion != k])[0] for k in range(njack)])
    metadata = dict(quantity='spectroscopic_angular_autocorrelation', cosmology='Planck15',
                    tracer=ref['tracer'], sample_id=str(source.resolve()), weights=args.weight_column or 'unit',
                    randoms=random_info, geometry_z=zmean, zmin=lo, zmax=hi,
                    estimator='Landy-Szalay; ordered DD/RR; (sum w)^2 - sum w^2 normalization',
                    selection=selection, n_data=len(data), n_random=len(random),
                    region_definition='identical to legacy cross-correlation; photometric-random area weights',
                    auto_weight_policy='same specified column in data/randoms; no extra FKP or component multiplication',
                    photometric_weights='unit',
                    angular_definition='fixed angles for the whole slice, using mean data redshift',
                    elapsed_seconds=time.monotonic()-started)
    np.savez(args.output_dir/'measurement.npz', theta_edges_rad=angles, w_auto=values[0], covariance=cov,
             z_edges=zedges, nz_counts=nz, metadata_json=json.dumps(metadata),
             jackknife_w_auto=values[1:], jackknife_nz_counts=hist_jk, region_area_counts=areas,
             region_map_nside4=region_map, data_counts=nd, random_counts=nr,
             data_weight_sums=dsum, random_weight_sums=rsum,
             data_weight2_sums=dsum2, random_weight2_sums=rsum2)
    np.savez(args.output_dir/'pair_counts.npz', DD=dd, DR=dr, RR=rr)
    (args.output_dir/'selection.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print('w_auto:', values[0], 'sigma:', np.sqrt(np.diag(cov)), flush=True)
    print(f'Completed in {time.monotonic()-started:.1f}s; {args.output_dir}', flush=True)


if __name__ == '__main__':
    main()
