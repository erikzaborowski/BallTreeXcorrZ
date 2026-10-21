"""Declared photometric population and legacy-compatible mask membership.

Mask lookup uses the catalogue's stored Galactic coordinates. Recomputing them
from RA/DEC can change membership at pixel boundaries. No mask is constructed
or cleaned here; the frozen mask is an input with separately recorded provenance.
"""

import healpy as hp
import numpy as np


def legacy_mask_pixels(l, b, nside):
    """Exact RING assignment used by the legacy Galactic counts-map routine."""
    if isinstance(nside, bool) or not hp.isnsideok(nside, nest=True):
        raise ValueError("nside must be a positive HEALPix power of two")
    lon, lat = np.broadcast_arrays(np.asarray(l, dtype=float), np.asarray(b, dtype=float))
    if not np.isfinite(lon).all() or not np.isfinite(lat).all() or np.any(np.abs(lat) > 90):
        raise ValueError("Galactic coordinates must be finite with latitude in [-90,90]")
    theta = np.radians(90.0 - lat)
    phi = np.radians(lon % 360.0)
    return hp.ang2pix(nside, theta, phi)
