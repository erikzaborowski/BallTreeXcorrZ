"""Small analytic checks: no survey catalogs, pair counting, or CAMB solve."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.integrate import quad
from scipy.special import j0

from fit_spectroscopic_bias import annulus_window, fit_bias, main, make_template, project_auto


class BiasTests(unittest.TestCase):
    def test_annulus_against_direct_angular_integration(self):
        for low, high in [(1e-8, 2e-8), (.2, .8), (2., 5.), (30., 45.)]:
            expected = quad(lambda x: 2*x*j0(x)/(high**2-low**2), low, high)[0]
            self.assertAlmostEqual(float(annulus_window(low, high)), expected, places=11)

    def test_gaussian_power_projection_and_radial_normalization(self):
        # int k dk exp(-a k^2) J0(k R)/(2pi) = exp(-R^2/(4a))/(4pi a).
        # A uniform normalized radial window of depth L contributes 1/L.
        k = np.geomspace(1e-6, 15., 12001)
        a, depth, chi = .8, 120., 1000.
        edges = np.array([.001, .002, .004])
        pk = np.exp(-a*k*k)[None, :]
        actual = project_auto(k, pk, [chi], [depth/.05], [1/.05], [.05], edges)
        radii = chi*edges
        expected = (np.exp(-radii[:-1]**2/(4*a))-np.exp(-radii[1:]**2/(4*a))) / (
            np.pi*np.diff(radii**2)*depth)
        np.testing.assert_allclose(actual, expected, rtol=5e-7)
        # Twice the radial depth halves the projected auto amplitude.
        wider = project_auto(k, pk, [chi], [2*depth/.05], [1/.05], [.05], edges)
        np.testing.assert_allclose(wider, actual/2)

    def test_correlated_covariance_fit_and_error(self):
        t = np.array([.03, .02, .01])
        c = np.array([[4., 1., .5], [1., 3., .2], [.5, .2, 2.]])*1e-6
        fit = fit_bias(4*t, t, c)
        expected_sigma_a = np.sqrt(1/(t @ np.linalg.solve(c, t)))
        self.assertAlmostEqual(fit['bias'], 2.)
        self.assertAlmostEqual(fit['sigma_bias_squared'], expected_sigma_a)
        self.assertAlmostEqual(fit['sigma_bias'], expected_sigma_a/4)
        self.assertAlmostEqual(fit['chi2'], 0.)
        self.assertEqual(fit['ndof'], 2)
        perturbed = fit_bias(4*t+np.array([.002, -.001, .001]), t, c)
        self.assertGreater(perturbed['chi2'], 0.)

    def test_negative_amplitude_is_not_a_bias(self):
        fit = fit_bias([-2., -4.], [1., 2.], np.eye(2))
        self.assertEqual(fit['bias_squared'], -2.)
        self.assertIsNone(fit['bias'])
        self.assertIsNone(fit['sigma_bias'])
        self.assertFalse(fit['usable'])

    def test_invalid_fit_inputs(self):
        for c in [np.ones((2, 2)), np.diag([1., -1.]), np.array([[1., .1], [.2, 1.]])]:
            with self.assertRaises(ValueError):
                fit_bias([1., 2.], [1., 2.], c)
        with self.assertRaises(ValueError):
            fit_bias([1., 2.], [0., 0.], np.eye(2))
        with self.assertRaises(ValueError):
            fit_bias([1., np.nan], [1., 2.], np.eye(2))

    def test_cli_roundtrip_selection_and_guards(self):
        from astropy.cosmology import Planck15
        distance = Planck15.comoving_distance(.825).value*Planck15.h
        # First annulus is outside the fit interval and must not affect b.
        edges = np.array([1., 2.517, 4., 6.31, 10.])/distance
        k = np.geomspace(1e-5, 10., 4001)
        grid = dict(z=np.linspace(.8, .85, 5), k_hmpc=k,
                    pk_mpc_over_h3=np.tile(500*np.exp(-k*k), (5, 1)))
        zedges, counts = np.array([.8, .82, .85]), np.array([2., 7.])
        template = make_template(grid, edges, zedges, counts)
        np.testing.assert_allclose(make_template(grid, edges, zedges, counts*10), template)
        with self.assertRaises(ValueError):
            make_template(grid, edges, [.79, .85], [1.])
        values = 2.25*template
        values[0] += 100
        covariance = np.eye(4)*1e-6
        meta = dict(quantity='spectroscopic_angular_autocorrelation', cosmology='Planck15',
                    tracer='SYNTHETIC', sample_id='analytic-test', weights='unit',
                    randoms='synthetic', geometry_z=.825, zmin=.8, zmax=.85)
        # Synthetic table deliberately exercises the reader, not a Halofit solve.
        power_meta = dict(cosmology='Planck15', model='halofit_takahashi', matter_field='total',
                          k_unit='h/Mpc', power_unit='(Mpc/h)^3', data_kind='synthetic')
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            np.savez(root/'power.npz', **grid, metadata_json=json.dumps(power_meta))
            def save_measurement():
                np.savez(root/'auto.npz', theta_edges_rad=edges, w_auto=values,
                         covariance=covariance, z_edges=zedges, nz_counts=counts,
                         metadata_json=json.dumps(meta))
            save_measurement()
            args = ['--measurement', str(root/'auto.npz'), '--power', str(root/'power.npz'),
                    '--output', str(root/'bias.json')]
            with contextlib.redirect_stdout(io.StringIO()):
                main(args)
            result = json.loads((root/'bias.json').read_text())
            self.assertAlmostEqual(result['bias'], 1.5, places=7)
            self.assertEqual(result['fit_indices'], [1, 2, 3])
            with self.assertRaises(FileExistsError):
                main(args)
            (root/'bias.json').unlink()
            meta['quantity'] = 'photometric_cross_correlation'
            save_measurement()
            with self.assertRaisesRegex(ValueError, 'AUTO'):
                main(args)


if __name__ == '__main__':
    unittest.main()
