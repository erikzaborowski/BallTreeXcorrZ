"""Tiny, synthetic tests; no survey catalogue reads or pair counting."""
import json
from pathlib import Path
import tempfile
import unittest

import healpy as hp
import numpy as np
from astropy.io import fits

from prepare_elg_desi import combine_references, prepare_reference, slice_combined
from prepare_elg_sdss import reuse_photometry


class PreparationTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='elg-desi-test-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.support = np.zeros(hp.nside2npix(8), dtype=bool)
        self.support[hp.ang2pix(8, 10., 20., lonlat=True)] = True
        # IDs exceed the exact integer range of float64.
        self.first_id = 2**60 + 1
        self.ngc = np.array([
            (10., 20., .80, self.first_id, 2.),
            (10., 20., .85, self.first_id + 1, 3.),  # upper edge excluded
            (10., 20., .79, self.first_id + 2, 4.),
            (180., -20., .82, self.first_id + 3, 5.),  # outside footprint
            (np.nan, 20., .82, self.first_id + 4, 6.),
            (10., 20., np.nan, self.first_id + 5, 7.),
        ], dtype=[('RA', 'f8'), ('DEC', 'f8'), ('Z', 'f8'),
                  ('TARGETID', 'i8'), ('WEIGHT', 'f8')])
        self.sgc = np.array([(10., 20., .84, self.first_id + 6, 8.)], dtype=self.ngc.dtype)
        self.write_caps()

    def write_caps(self):
        for cap, rows in [('NGC', self.ngc), ('SGC', self.sgc)]:
            fits.BinTableHDU(rows).writeto(
                self.root / ('LRG_' + cap + '_clustering.dat.fits'), overwrite=True)

    def prepare(self):
        return prepare_reference(self.root, 'LRG', self.support, .80, .85,
                                 self.root / 'desi_lrg.fits')

    def test_selection_and_provenance(self):
        metadata = self.prepare()
        with fits.open(self.root / 'desi_lrg.fits') as hdus:
            np.testing.assert_array_equal(hdus[1].data['TARGETID'],
                                          [self.first_id, self.first_id + 6])
            np.testing.assert_array_equal(hdus[1].data['WEIGHT'], [2., 8.])
            np.testing.assert_array_equal(hdus[1].data['Z'], [.80, .84])
        self.assertEqual(metadata['input_rows'], 7)
        self.assertEqual(metadata['in_redshift_range'], 3)
        self.assertEqual(metadata['selected_rows'], 2)
        self.assertEqual(metadata['weights_applied'], 'unit')
        self.assertEqual([s['selected_rows'] for s in metadata['sources']], [1, 1])

    def test_duplicate_across_caps_rejected_even_outside_bin(self):
        self.sgc['TARGETID'][0] = self.ngc['TARGETID'][1]
        self.write_caps()
        with self.assertRaisesRegex(ValueError, 'duplicate TARGETIDs'):
            self.prepare()
        self.assertFalse((self.root / 'desi_lrg.fits').exists())

    def test_duplicate_within_cap_rejected(self):
        self.ngc['TARGETID'][1] = self.ngc['TARGETID'][0]
        self.write_caps()
        with self.assertRaisesRegex(ValueError, 'duplicate TARGETIDs'):
            self.prepare()

    def test_no_overlap_rejected(self):
        self.support[:] = False
        with self.assertRaisesRegex(ValueError, 'No references'):
            self.prepare()

    def test_missing_weight_rejected(self):
        fits.BinTableHDU(self.sgc[['RA', 'DEC', 'Z', 'TARGETID']]).writeto(
            self.root / 'LRG_SGC_clustering.dat.fits', overwrite=True)
        with self.assertRaisesRegex(ValueError, 'Missing columns.*WEIGHT'):
            self.prepare()

    def test_combined_union_and_membership(self):
        self.prepare()
        other = self.sgc.copy()
        other['WEIGHT'] = 123.  # A different tracer weight must not affect this union.
        fits.BinTableHDU(other).writeto(self.root / 'desi_qso.fits')
        info = combine_references([('LRG', self.root / 'desi_lrg.fits'),
                                   ('QSO', self.root / 'desi_qso.fits')], self.root / 'all.fits')
        with fits.open(self.root / 'all.fits') as hdus:
            np.testing.assert_array_equal(hdus[1].data['TARGETID'],
                                          [self.first_id, self.first_id + 6])
            np.testing.assert_array_equal(hdus[1].data['TRACER_BITS'], [2, 2 | 8])
            np.testing.assert_array_equal(hdus[1].data['WEIGHT'], [1., 1.])
        self.assertEqual(info['duplicate_memberships_removed'], 1)
        self.assertEqual(info['shared_targetids'], 1)

    def test_combined_conflicting_redshift_rejected(self):
        self.prepare()
        other = self.sgc.copy()
        other['Z'] = .81
        fits.BinTableHDU(other).writeto(self.root / 'desi_qso.fits')
        with self.assertRaisesRegex(ValueError, 'Conflicting Z'):
            combine_references([('LRG', self.root / 'desi_lrg.fits'),
                                ('QSO', self.root / 'desi_qso.fits')], self.root / 'all.fits')
        self.assertFalse((self.root / 'all.fits').exists())

    def test_empty_component_allowed_in_union(self):
        self.prepare()
        info = prepare_reference(self.root, 'LRG', self.support, 2., 3.,
                                 self.root / 'empty.fits', allow_empty=True)
        self.assertEqual(info['selected_rows'], 0)
        combined = combine_references([('LRG', self.root / 'desi_lrg.fits'),
                                       ('QSO', self.root / 'empty.fits')], self.root / 'all.fits')
        self.assertEqual(combined['selected_rows'], 2)

    def test_excluded_id_cannot_reappear_in_adjacent_bins(self):
        self.prepare()
        other = np.array([(10., 20., .86, self.first_id + 6, 9.),
                          (10., 20., .87, self.first_id + 7, 10.)], dtype=self.ngc.dtype)
        fits.BinTableHDU(other).writeto(self.root / 'desi_qso.fits')
        info = combine_references([('LRG', self.root / 'desi_lrg.fits'),
                                   ('QSO', self.root / 'desi_qso.fits')],
                                  self.root / 'desi_all.fits', redshift_conflicts='exclude')
        self.assertEqual(info['excluded_redshift_targetids'], 1)
        self.assertEqual(info['unique_targets_before_exclusion'], 3)
        self.assertEqual(info['selected_rows'], 2)
        with fits.open(self.root / 'excluded_redshift_targetids.fits') as hdus:
            self.assertEqual(hdus[1].data['TARGETID'].tolist(), [self.first_id + 6])
            np.testing.assert_array_equal(hdus[1].data['Z_MIN'], [.84])
            np.testing.assert_array_equal(hdus[1].data['Z_MAX'], [.86])
        # Both original tracer catalogues still contain the excluded target.
        for tracer in ('lrg', 'qso'):
            with fits.open(self.root / ('desi_' + tracer + '.fits')) as hdus:
                self.assertIn(self.first_id + 6, hdus[1].data['TARGETID'])
        phot, rand = {'source': 'phot'}, {'source': 'rand'}
        metadata = {'photometry': phot, 'randoms': rand, 'reference': info,
                    'selection': 'elg_mask == 0; unit weights; no additional ELG cuts',
                    'zmin': .8, 'zmax': .9}
        (self.root / 'selection.json').write_text(json.dumps(metadata))
        for lo, hi, label, expected in [(.8, .85, '16', self.first_id),
                                         (.85, .9, '17', self.first_id + 7)]:
            path = self.root / ('bin' + label + '.fits')
            sliced = slice_combined(self.root, path, lo, hi, phot, rand)
            self.assertEqual(sliced['parent_excluded_redshift_targetids'], 1)
            with fits.open(path) as hdus:
                self.assertEqual(hdus[1].data['TARGETID'].tolist(), [expected])
        with self.assertRaisesRegex(ValueError, 'does not match'):
            slice_combined(self.root, self.root / 'wrong.fits', .8, .85, {}, rand)

    def test_exclusion_does_not_relax_coordinate_check(self):
        self.prepare()
        other = self.sgc.copy()
        other['RA'] += 1.
        fits.BinTableHDU(other).writeto(self.root / 'desi_qso.fits')
        with self.assertRaisesRegex(ValueError, 'Conflicting RA'):
            combine_references([('LRG', self.root / 'desi_lrg.fits'),
                                ('QSO', self.root / 'desi_qso.fits')],
                               self.root / 'all.fits', redshift_conflicts='exclude')

    def test_cache_reuse_and_mismatched_selection(self):
        cached, output = self.root / 'cached', self.root / 'inputs'
        cached.mkdir()
        output.mkdir()
        metadata = {'seed': 1909,
                    'selection': 'elg_mask == 0; unit weights; no additional ELG cuts'}
        for key, name in [('photometry', 'elg.fits'), ('randoms', 'randoms.fits')]:
            fits.BinTableHDU(self.sgc[['RA', 'DEC']]).writeto(cached / name)
            metadata[key] = {'source': str(self.root / ('raw-' + name)),
                             'selected_rows': 1, 'random_fraction': 1.}
        (cached / 'selection.json').write_text(json.dumps(metadata))
        np.savez(cached / 'occupancy_nside8.npz', phot=self.support, randoms=self.support)
        phot = Path(metadata['photometry']['source'])
        randoms = Path(metadata['randoms']['source'])
        phot_map, rand_map, _, _ = reuse_photometry(phot, randoms, output, cached, 1., 1909)
        np.testing.assert_array_equal(phot_map, self.support)
        np.testing.assert_array_equal(rand_map, self.support)
        self.assertEqual((output / 'elg.fits').resolve(), cached / 'elg.fits')
        with self.assertRaisesRegex(ValueError, 'do not match'):
            reuse_photometry(phot, randoms, output, cached, .5, 1909)


if __name__ == '__main__':
    unittest.main()
