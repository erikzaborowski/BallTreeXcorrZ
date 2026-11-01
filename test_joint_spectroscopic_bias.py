"""Bounded synthetic tests; no survey data, CAMB, or compute allocation."""
import ast
import contextlib
import io
import json
from pathlib import Path
import tempfile
import time
import unittest

import numpy as np
from scipy.sparse import csr_matrix

from combine_spectroscopic_bias import combine_bin, joint_statistics, read_cross, collect
from fit_spectroscopic_bias import make_template, templates_from_histograms


class JointTests(unittest.TestCase):
    def test_legacy_sparse_weights_against_explicit_pairs(self):
        source = Path(__file__).with_name('error_estimator.py')
        function = next(n for n in ast.parse(source.read_text()).body
                        if isinstance(n, ast.FunctionDef) and n.name == 'make_sparse_mat')
        namespace = dict(np=np, csr_matrix=csr_matrix, time=time, t0=time.time())
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), 'exec'), namespace)
        make = namespace['make_sparse_mat']
        entries = np.empty((3, 1), object)
        entries[:, 0] = [(np.array([2]), np.array([0])),
                         (np.array([1, 3]), np.array([1, 2])),
                         (np.array([4, 2, 1]), np.array([0, 2, 3]))]
        weights, pix, downgrade = np.array([.4, 2., 3.]), np.array([0, 1, 0]), np.array([0, 0, 1, 1])
        expected = np.zeros((2, 2))
        for i, (counts, pixels) in enumerate(entries[:, 0]):
            for count, pixel in zip(counts, pixels):
                expected[pix[i], downgrade[pixel]] += count*weights[i]
        with contextlib.redirect_stdout(io.StringIO()):
            actual = make(entries, np.arange(2), pix, 2, 0, downgrade, weights).toarray()
            unit = make(entries, np.arange(2), pix, 2, 0, downgrade).toarray()
            constant = make(entries, np.arange(2), pix, 2, 0, downgrade, np.full(3, 5.)).toarray()
        np.testing.assert_allclose(actual, expected)
        np.testing.assert_allclose(constant, 5*unit)

    def test_joint_uncertainty_retains_cancellation(self):
        bias = np.array([1.9, 2.2, 2., 1.8])
        result = joint_statistics(.06, 2., .03*bias, bias, np.array([8., 9., 10., 11.]))
        self.assertGreater(result['sigma_wbar'], 0)
        self.assertGreater(result['sigma_bias'], 0)
        self.assertLess(result['sigma_wbar_over_bias'], 1e-16)
        with self.assertRaises(ValueError):
            joint_statistics(.06, 2., .03*bias, [1., -1., 1., 1.], np.ones(4))

    def test_radial_basis_uses_squared_normalized_probabilities(self):
        basis = np.array([[1., 2.], [3., 4.]])
        np.testing.assert_allclose(templates_from_histograms(basis, [1, 3]), (basis[0]+9*basis[1])/16)
        np.testing.assert_allclose(templates_from_histograms(basis, [[1, 3], [3, 1]]),
                                   [(basis[0]+9*basis[1])/16, (9*basis[0]+basis[1])/16])

    def test_full_joint_roundtrip_varies_radial_window_and_refits(self):
        from astropy.cosmology import Planck15
        radii = np.geomspace(.1006799, 10., 11)
        edges = radii[7:]/(Planck15.comoving_distance(.825).value*Planck15.h)
        k = np.geomspace(1e-5, 10., 1001)
        grid = dict(z=np.linspace(.8, .85, 5), k_hmpc=k, pk_mpc_over_h3=np.tile(500*np.exp(-k*k), (5, 1)))
        zedges, counts = np.array([.8, .82, .85]), np.array([3., 7.])
        jack_counts = np.array([[2., 7.], [3., 5.], [2., 6.], [3., 6.]])
        true_bias = np.array([1.9, 2.1, 2.05, 1.95])
        auto = 4*make_template(grid, edges, zedges, counts, 16)
        jack_auto = np.stack([b*b*make_template(grid, edges, zedges, n, 16) for b, n in zip(true_bias, jack_counts)])
        power_meta = dict(cosmology='Planck15', model='halofit_takahashi', matter_field='total',
                          k_unit='h/Mpc', power_unit='(Mpc/h)^3', data_kind='synthetic')
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root/'correlations').mkdir()
            meta = dict(quantity='spectroscopic_angular_autocorrelation', cosmology='Planck15', tracer='LRG',
                        sample_id=str(root/'inputs/desi_lrg.fits'), weights='WEIGHT', photometric_weights='unit',
                        randoms='synthetic', geometry_z=.825, zmin=.8, zmax=.85, n_data=12,
                        selection=dict(photometry={'source': 'synthetic', 'selected_rows': 20},
                                       randoms={'source': 'synthetic_randoms', 'selected_rows': 80}))
            np.savez(root/'auto.npz', w_auto=auto, covariance=np.eye(3)*1e-6, theta_edges_rad=edges,
                     z_edges=zedges, nz_counts=counts, jackknife_w_auto=jack_auto, jackknife_nz_counts=jack_counts,
                     region_area_counts=np.ones(4)*10, region_map_nside4=np.arange(4), metadata_json=json.dumps(meta))
            np.savez(root/'power.npz', **grid, metadata_json=json.dumps(power_meta))
            table = np.zeros((10, 13+10+3*4))
            table[:, 3], table[:, 4], table[:, 5] = (radii[:-1]+radii[1:])/2, radii[:-1], radii[1:]
            integral = ((table[7:, 5]-table[7:, 4])/table[7:, 3]).sum()
            table[7:, 6] = .06/integral
            table[7:, 23:27] = .03*true_bias/integral
            table[0, 31:] = 10
            header = 'SPEC=inputs/desi_lrg.fits\nz1=.8 z2=.85 zmean=.825\nSPEC_WEIGHT_COLUMN=WEIGHT\nPHOT_WEIGHT=unit\nNSIDE=4\nCOSMO=Planck15\nN_SPEC=12\nERROR=jackknife-leave one out'
            cross = root/'correlations/cross.txt'
            np.savetxt(cross, table, header=header)
            with contextlib.redirect_stdout(io.StringIO()):
                combine_bin(cross, root/'auto.npz', root/'power.npz', root/'joint')
            with np.load(root/'joint/measurement.npz') as result:
                np.testing.assert_allclose(result['jackknife_bias'], true_bias, rtol=1e-12)
                self.assertAlmostEqual(float(result['wbar_over_bias']), .03)
                self.assertLess(float(result['sigma_wbar_over_bias']), 1e-14)
                self.assertGreater(float(result['sigma_bias']), .01)
            # Reuse exactly the same spectroscopic auto with verified new photometric weights.
            weighted_header = header.replace('PHOT_WEIGHT=unit', 'PHOT_WEIGHT=WEIGHT_SN_DEBV0')
            weighted_header += ('\nPHOT_RANDOM_WEIGHT=unit\nPHOTO=inputs/elg.fits'
                                '\nPHOTO_RANDOM=inputs/randoms.fits\nN_PHOT=20\nN_PHOT_RANDOM=80')
            weighted_cross = root/'correlations/weighted.txt'
            np.savetxt(weighted_cross, table, header=weighted_header)
            report = dict(coordinates_and_order_verified=True, occupancy_verified=True,
                          original_photometry=meta['selection']['photometry'],
                          randoms=meta['selection']['randoms'],
                          photometry=dict(weight_column='WEIGHT_SN_DEBV0', selected_rows=20),
                          prepared_photometry=str(root/'inputs/elg.fits'),
                          prepared_randoms=str(root/'inputs/randoms.fits'))
            validation = root/'validation.json'
            validation.write_text(json.dumps(report))
            with contextlib.redirect_stdout(io.StringIO()):
                combine_bin(weighted_cross, root/'auto.npz', root/'power.npz', root/'joint_weighted',
                            'WEIGHT_SN_DEBV0', validation)
            with np.load(root/'joint_weighted/measurement.npz') as result:
                np.testing.assert_allclose(result['jackknife_bias'], true_bias, rtol=1e-12)
                self.assertLess(float(result['sigma_wbar_over_bias']), 1e-14)
                self.assertEqual(json.loads(str(result['metadata_json']))['photometric_weights'], 'WEIGHT_SN_DEBV0')
            fresh = root/'fresh_collection'
            (fresh/'bin-016').mkdir(parents=True)
            (fresh/'bin-016/joint').symlink_to(root/'joint_weighted', target_is_directory=True)
            collect(fresh, [16], compare_unit=False)
            with np.load(fresh/'joint_measurements.npz') as result:
                self.assertAlmostEqual(float(result['wbar_over_bias'][0]), .03)
            self.assertTrue((fresh/'weighted_lrg.png').is_file())
            report['coordinates_and_order_verified'] = False
            validation.write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, 'validated original photometric footprint'):
                read_cross(weighted_cross, {}, meta, 'WEIGHT_SN_DEBV0', validation)


if __name__ == '__main__':
    unittest.main()
