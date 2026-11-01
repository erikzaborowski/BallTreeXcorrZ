"""Tiny fresh-footprint preparation and tree tests; no survey data."""
import contextlib
import io
import json
from pathlib import Path
import pickle
import tempfile
import unittest

import numpy as np
from astropy.io import fits

from prepare_elg_fresh_footprint import prepare_catalogs, build_trees
from prepare_weighted_elg import prepare


class FreshFootprintTests(unittest.TestCase):
    def test_new_selection_reference_slices_weights_and_tree_order(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rows = np.array([(10., 20., 0, 2.), (180., -20., 0, .5), (250., 10., 1, 1.)],
                            dtype=[('RA','f8'), ('DEC','f8'), ('elg_mask','i8'), ('WEIGHT_SN_DEBV0','f8')])
            fits.BinTableHDU(rows).writeto(root/'phot.fits')
            fits.BinTableHDU(rows[['RA','DEC','elg_mask']]).writeto(root/'randoms.fits')
            ref_dtype=[('RA','f8'), ('DEC','f8'), ('Z','f8'), ('TARGETID','i8'), ('WEIGHT','f8')]
            ngc=np.array([(10.,20.,.825,1,3.),(180.,-20.,.875,2,4.),
                          (250.,10.,.825,3,5.)], dtype=ref_dtype)
            sgc=np.array([(10.,20.,.84,4,6.)], dtype=ref_dtype)
            catalogs=root/'desi'
            catalogs.mkdir()
            for cap, data in (('NGC',ngc),('SGC',sgc)):
                fits.BinTableHDU(data).writeto(catalogs/f'LRG_{cap}_clustering.dat.fits')
            base, references = root/'base', root/'refs'
            with contextlib.redirect_stdout(io.StringIO()):
                prepare_catalogs(root/'phot.fits', root/'randoms.fits', base, references,
                                 catalogs, tracers=('LRG',), chunk_size=2)
                prepare(root/'phot.fits', base/'inputs', root/'weighted', 'WEIGHT_SN_DEBV0', chunk_size=2)
                build_trees(base)
            tasks=(references/'tasks.txt').read_text().splitlines()
            self.assertEqual([line.split()[1:] for line in tasks],
                             [['16','0.80','0.85','2'],['17','0.85','0.90','1']])
            with fits.open(references/'lrg/bin-016/inputs/desi_lrg.fits') as f:
                np.testing.assert_array_equal(f[1].data['TARGETID'], [1,4])
                np.testing.assert_array_equal(f[1].data['WEIGHT'], [3.,6.])
            report=json.loads((root/'weighted/selection.json').read_text())
            meta=json.loads((references/'lrg/bin-016/inputs/selection.json').read_text())
            self.assertEqual(report['original_photometry'], meta['photometry'])
            self.assertEqual(report['randoms'], meta['randoms'])
            self.assertEqual(meta['reference']['selected_rows'],2)
            self.assertEqual(meta['reference']['weights_applied'],'unit')
            for name in ('elg-d1tree.p','randoms-r1tree.p'):
                with (base/name).open('rb') as stream:
                    tree=pickle.load(stream)
                coords=np.column_stack((rows['DEC'][:2],rows['RA'][:2]))*np.pi/180.
                distance,index=tree.query(coords,k=1)
                np.testing.assert_array_equal(index[:,0],[0,1])
                np.testing.assert_allclose(distance,0,atol=1e-15)


if __name__ == '__main__':
    unittest.main()
