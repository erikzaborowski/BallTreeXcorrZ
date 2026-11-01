"""Tiny brute-force checks for the pilot's ordered pairs and deletions."""
import contextlib
import io
from pathlib import Path
import tempfile
import unittest

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
import astropy.units as u

from measure_spectroscopic_auto import pair_matrix, landy_szalay, checked_weights, read_randoms


class AutoTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(1909)
        self.d = np.deg2rad(np.column_stack((rng.uniform(-2, 2, 30), rng.uniform(358, 362, 30))))
        self.r = np.deg2rad(np.column_stack((rng.uniform(-2, 2, 90), rng.uniform(358, 362, 90))))
        self.dreg = np.arange(30) % 3
        self.rreg = np.arange(90) % 3
        self.edges = np.deg2rad([.1, 1., 2., 6.])

    def brute(self, a, b):
        ca = SkyCoord(ra=a[:, 1]*u.rad, dec=a[:, 0]*u.rad)
        cb = SkyCoord(ra=b[:, 1]*u.rad, dec=b[:, 0]*u.rad)
        angles = ca[:, None].separation(cb[None, :]).rad
        return np.histogram(angles.ravel(), self.edges)[0]

    def matrices(self):
        with contextlib.redirect_stdout(io.StringIO()):
            dd = pair_matrix(self.d, self.d, self.dreg, self.dreg, self.edges, 3, batch_size=7)
            dr = pair_matrix(self.d, self.r, self.dreg, self.rreg, self.edges, 3, batch_size=7)
            rr = pair_matrix(self.r, self.r, self.rreg, self.rreg, self.edges, 3, batch_size=7)
        return dd, dr, rr

    def test_counts_against_spherical_brute_force(self):
        dd, dr, rr = self.matrices()
        for matrix, a, b in [(dd, self.d, self.d), (dr, self.d, self.r), (rr, self.r, self.r)]:
            np.testing.assert_array_equal(matrix.sum(axis=(0, 1)), self.brute(a, b))
        np.testing.assert_array_equal(dd, dd.transpose(1, 0, 2))
        self.assertTrue(np.all(dd.sum(axis=(0, 1)) % 2 == 0))

    def test_deletions_against_recounting_and_finite_normalization(self):
        dd, dr, rr = self.matrices()
        values = landy_szalay(dd, dr, rr, np.bincount(self.dreg), np.bincount(self.rreg))
        for row, omitted in enumerate([None, 0, 1, 2]):
            d = self.d if omitted is None else self.d[self.dreg != omitted]
            r = self.r if omitted is None else self.r[self.rreg != omitted]
            a = self.brute(d, d)/(len(d)*(len(d)-1))
            b = self.brute(d, r)/(len(d)*len(r))
            c = self.brute(r, r)/(len(r)*(len(r)-1))
            np.testing.assert_allclose(values[row], (a-2*b+c)/c, atol=1e-14)

    def test_weighted_pairs_deletions_and_weight_rescaling(self):
        dw, rw = np.linspace(.2, 2., len(self.d)), np.linspace(.001, .02, len(self.r))
        # Zero completeness weights occur in the actual DESI random catalogs.
        rw[::4] = 0.
        def brute(a, b, wa, wb):
            ca = SkyCoord(ra=a[:, 1]*u.rad, dec=a[:, 0]*u.rad)
            cb = SkyCoord(ra=b[:, 1]*u.rad, dec=b[:, 0]*u.rad)
            return np.histogram(ca[:, None].separation(cb[None, :]).rad.ravel(), self.edges,
                                weights=(wa[:, None]*wb[None, :]).ravel())[0]
        def measured(dw, rw):
            mats = []
            with contextlib.redirect_stdout(io.StringIO()):
                for a, b, ar, br, wa, wb in [(self.d, self.d, self.dreg, self.dreg, dw, dw),
                                            (self.d, self.r, self.dreg, self.rreg, dw, rw),
                                            (self.r, self.r, self.rreg, self.rreg, rw, rw)]:
                    mat = pair_matrix(a, b, ar, br, self.edges, 3, batch_size=7, left_weights=wa, right_weights=wb)
                    np.testing.assert_allclose(mat.sum(axis=(0, 1)), brute(a, b, wa, wb), rtol=1e-13)
                    mats.append(mat)
            return landy_szalay(*mats, np.bincount(self.dreg, weights=dw), np.bincount(self.rreg, weights=rw),
                               np.bincount(self.dreg, weights=dw**2), np.bincount(self.rreg, weights=rw**2))
        values = measured(dw, rw)
        for row, omit in enumerate([None, 0, 1, 2]):
            dmask = np.ones(len(dw), bool) if omit is None else self.dreg != omit
            rmask = np.ones(len(rw), bool) if omit is None else self.rreg != omit
            d, r, a, b = self.d[dmask], self.r[rmask], dw[dmask], rw[rmask]
            dd = brute(d, d, a, a)/(a.sum()**2-(a*a).sum())
            dr = brute(d, r, a, b)/(a.sum()*b.sum())
            rr = brute(r, r, b, b)/(b.sum()**2-(b*b).sum())
            np.testing.assert_allclose(values[row], (dd-2*dr+rr)/rr, atol=1e-13)
        np.testing.assert_allclose(measured(3*dw, 8*rw), values, atol=1e-13)

    def test_random_reader_accepts_zero_weights_and_rejects_invalid_values(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'randoms.fits'
            table = np.array([(147., -7., .42, 0.), (148., -6., .43, 2.), (149., -5., .8, 1.)],
                             dtype=[('RA', 'f8'), ('DEC', 'f8'), ('Z', 'f8'), ('WEIGHT', 'f8')])
            fits.BinTableHDU(table).writeto(path)
            with contextlib.redirect_stdout(io.StringIO()):
                selected, info = read_randoms([path], np.ones(12*8**2, bool), .4, .45, 'WEIGHT')
            np.testing.assert_array_equal(selected[:, 3], [0., 2.])
            self.assertEqual(info[0]['selected_rows'], 2)
            for bad in (-1., np.nan, np.inf):
                table['WEIGHT'][0] = bad
                fits.BinTableHDU(table).writeto(path, overwrite=True)
                with self.assertRaisesRegex(ValueError, 'nonnegative'):
                    read_randoms([path], np.ones(12*8**2, bool), .4, .45, 'WEIGHT')
        with self.assertRaises(ValueError):
            checked_weights([1., 2.], 3)


if __name__ == '__main__':
    unittest.main()
