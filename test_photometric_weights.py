"""Small synthetic FITS/pair tests; no survey catalogs or compute allocation."""
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import healpy as hp
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.cosmology import Planck15
from astropy.io import fits

from prepare_elg_sdss import prepare_photometry
from prepare_weighted_elg import prepare

REPO = Path(__file__).resolve().parent
WEIGHT = 'WEIGHT_SN_DEBV0'


def write_table(path, **columns):
    fits.BinTableHDU.from_columns([fits.Column(name=k, format='D', array=v)
                                  for k, v in columns.items()]).writeto(path)


class PhotometricWeightsTests(unittest.TestCase):
    def test_preparation_checks_mask_weights_and_cached_order(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cached = root / 'cached'
            cached.mkdir()
            source = root / 'source.fits'
            write_table(source, RA=[1., 2., 3., 4.], DEC=[0., 0., 0., 0.],
                        elg_mask=[0, 1, 0, 0], **{WEIGHT: [2., np.nan, .5, 0.]})
            with contextlib.redirect_stdout(io.StringIO()):
                occupancy, info = prepare_photometry(source, cached/'elg.fits', 1., 1909, 2)
            write_table(cached/'randoms.fits', RA=[1., 3., 4.], DEC=[0., 0., 0.])
            np.savez(cached/'occupancy_nside8.npz', phot=occupancy, randoms=occupancy)
            (cached/'selection.json').write_text(json.dumps(dict(photometry=info,
                randoms=dict(source='synthetic', selected_rows=3, random_fraction=1.), seed=1909,
                selection='elg_mask == 0; unit weights; no additional ELG cuts')))
            with contextlib.redirect_stdout(io.StringIO()):
                prepare(source, cached, root/'weighted', WEIGHT, chunk_size=2)
            with fits.open(root/'weighted/elg.fits') as f:
                np.testing.assert_array_equal(f[1].data[WEIGHT], [2., .5, 0.])
            report = json.loads((root/'weighted/selection.json').read_text())
            self.assertEqual(report['photometry']['sum_weights'], 2.5)
            self.assertTrue(report['coordinates_and_order_verified'])
            with fits.open(cached/'elg.fits', mode='update') as f:
                f[1].data['RA'][0] += .001
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, 'coordinates/order'):
                prepare(source, cached, root/'changed', WEIGHT, chunk_size=2)
            with fits.open(source, mode='update') as f:
                f[1].data[WEIGHT][0] = -1
            with self.assertRaisesRegex(ValueError, 'nonnegative'):
                prepare_photometry(source, root/'bad.fits', 1., 1909, 2, weight_column=WEIGHT)

    def test_pair_counts_and_full_estimator_against_explicit_weighted_pairs(self):
        radii = np.geomspace(.1006799, 10., 11)
        angle_edges = radii/(Planck15.comoving_distance(.825).value*Planck15.h)
        ra, dec = hp.pix2ang(8, np.array([300, 350, 400, 450]), lonlat=True)
        regions = hp.ang2pix(4, ra, dec, lonlat=True)
        self.assertEqual(len(np.unique(regions)), 4)
        photo, random = [], []
        for region in range(4):
            for k in range(10):
                theta = np.rad2deg((angle_edges[k]+angle_edges[k+1])/2)
                for j in range(4+k+region):
                    photo.append((ra[region]+theta/np.cos(np.deg2rad(dec[region])), dec[region],
                                  .3+(region+1)*(k+1)/10, regions[region]))
                for j in range(20+region):
                    random.append((ra[region]+theta/np.cos(np.deg2rad(dec[region])), dec[region], regions[region]))
        photo, random = np.array(photo), np.array(random)
        photo[::17, 2] = 0.
        photo[:4, 2] = 0.  # An annulus with only zero-weight photometric neighbours.
        spec_weights = np.array([.5, 2., 1.3, 3.])
        # Independent spherical separations and explicit sums of pair products.
        spec = SkyCoord(ra=ra, dec=dec, unit='deg')
        dp = spec[:, None].separation(SkyCoord(photo[:, 0], photo[:, 1], unit='deg')[None, :]).rad
        dr = spec[:, None].separation(SkyCoord(random[:, 0], random[:, 1], unit='deg')[None, :]).rad
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                       NUMEXPR_NUM_THREADS='1', MPLCONFIGDIR=str(root/'mpl'), PYTHONDONTWRITEBYTECODE='1')

            def run(name, phot_w, spec_w, weighted):
                directory = root/name
                directory.mkdir()
                (directory/'correlations').mkdir()
                (directory/'plots').mkdir()
                write_table(directory/'elg.fits', RA=photo[:, 0], DEC=photo[:, 1], **{WEIGHT: phot_w})
                write_table(directory/'randoms.fits', RA=random[:, 0], DEC=random[:, 1])
                write_table(directory/'desi.fits', RA=ra, DEC=dec, Z=np.full(4, .825), WEIGHT=spec_w)
                options = ['--phot-weight-column', WEIGHT] if weighted else []
                count = [sys.executable, str(REPO/'cross_correlate.py'), '--no-loadtree',
                         '--zmin', '.82', '--zmax', '.82', '--dz', '.01', '--Smin', '.1006799',
                         '--Smax', '10', '--Nbins', '10', '--query-batch-size', '2'] + options + [
                         'elg.fits', 'randoms.fits', 'desi.fits']
                estimate = [sys.executable, str(REPO/'error_estimator.py'), '--jackknife', '--equalize',
                            '--zmin', '.80', '--zmax', '.85', '--dz', '.05', '--orig_deltaz', '.01',
                            '--Smin', '.1006799', '--Smax', '10', '--nbins', '10', '--nside', '4', '--small_nside', '8']
                if weighted:
                    estimate += options + ['--spec-weight-column', 'WEIGHT']
                estimate += ['elg.fits', 'randoms.fits', 'desi.fits', 'correlations/', 'plots/']
                for stage, command in (('count', count), ('estimate', estimate)):
                    with (directory/(stage+'.log')).open('w') as stream:
                        result = subprocess.run(command, cwd=directory, env=env, stdout=stream,
                                                stderr=subprocess.STDOUT, timeout=60)
                    if result.returncode:
                        self.fail((directory/(stage+'.log')).read_text()[-7000:])
                return np.loadtxt(directory/'correlations/z0.80_0.85_jk_loo_equalized.txt'), estimate, directory

            measured, command, directory = run('weighted', photo[:, 2], spec_weights, True)
            # Equalize with four equal-sized isolated pixels preserves each region;
            # reproduce its actual ordering from the geometry rather than assuming it.
            from measure_spectroscopic_auto import legacy_regions
            occupancy = np.bincount(hp.ang2pix(8, random[:, 0], random[:, 1], lonlat=True), minlength=768)
            mapping, areas = legacy_regions(occupancy, REPO/'error_estimator.py')
            self.assertEqual(len(areas), 4)
            adjusted, global_density = [], []
            for deleted in [None]+list(range(4)):
                spec_keep = np.ones(4, bool) if deleted is None else mapping[regions] != deleted
                phot_keep = np.ones(len(photo), bool) if deleted is None else mapping[photo[:, 3].astype(int)] != deleted
                rand_keep = np.ones(len(random), bool) if deleted is None else mapping[random[:, 2].astype(int)] != deleted
                dd, rr, normalized = np.zeros(10), np.zeros(10), np.zeros(10)
                for i in np.flatnonzero(spec_keep):
                    p_here = photo[:, 3] == regions[i]
                    r_here = random[:, 2] == regions[i]
                    density = photo[p_here, 2].sum()/r_here.sum()
                    for k in range(10):
                        dd[k] += spec_weights[i]*photo[phot_keep & (dp[i] >= angle_edges[k]) & (dp[i] < angle_edges[k+1]), 2].sum()
                        count = spec_weights[i]*np.count_nonzero(rand_keep & (dr[i] >= angle_edges[k]) & (dr[i] < angle_edges[k+1]))
                        rr[k] += count
                        normalized[k] += count*density
                adjusted.append(dd/normalized-1)
                global_density.append(dd/rr*rand_keep.sum()/photo[phot_keep, 2].sum()-1)
            np.testing.assert_allclose(measured[:, 6], adjusted[0], atol=1e-12)
            np.testing.assert_allclose(measured[:, 7], global_density[0], atol=1e-12)
            np.testing.assert_allclose(measured[:, 23:27].T, adjusted[1:], atol=1e-12)
            np.testing.assert_allclose(measured[:, 27:31].T, global_density[1:], atol=1e-12)
            scaled, _, _ = run('scaled', 5*photo[:, 2], 3*spec_weights, True)
            np.testing.assert_allclose(scaled[:, 6:], measured[:, 6:], atol=1e-12, equal_nan=True)
            unit, _, _ = run('unit', np.ones(len(photo)), np.ones(4), False)
            unit_weighted, _, _ = run('unit_weighted', np.ones(len(photo)), np.ones(4), True)
            # Weighted mode intentionally omits the legacy Poisson/MJB diagnostics.
            columns = [6, 7]+list(range(10, measured.shape[1]))
            np.testing.assert_allclose(unit[:, columns], unit_weighted[:, columns], atol=1e-12)
            # Explicitly reject weighted counts being interpreted as unweighted.
            pos = command.index('--phot-weight-column')
            wrong = command[:pos]+command[pos+2:]
            failed = subprocess.run(wrong, cwd=directory, env=env, capture_output=True, timeout=60)
            self.assertNotEqual(failed.returncode, 0)
            self.assertIn(b'Pair-cache weights', failed.stderr)


if __name__ == '__main__':
    unittest.main()
