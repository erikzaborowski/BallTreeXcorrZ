"""P2.2 nested HEALPix storage cells and independent resampling regions.

All sky labels use equatorial coordinates and NESTED ordering. Local-density
cells must be unions of storage cells. Jackknife regions instead use an explicit
cell-to-region mapping: the production clustering and region count belong to the
pilot, not to this module. No catalogue or mask files are opened here.
"""
from dataclasses import dataclass

import numpy as np


def _nside(value):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or value < 1 or value > 2**29 or value & (value - 1)):
        raise ValueError("NESTED nside must be a power of two in [1, 2**29]")
    return int(value)


def _labels(value, name, *, upper=None):
    value = np.asarray(value)
    if value.ndim != 1 or (value.size and value.dtype.kind not in "iu"):
        raise ValueError(f"{name} must be a one-dimensional integer array")
    if value.size and (np.any(value < 0) or np.any(value > np.iinfo(np.int64).max)):
        raise ValueError(f"{name} must be nonnegative int64 labels")
    value = np.array(value, dtype=np.int64, copy=True)
    if upper is not None and np.any(value >= upper):
        raise ValueError(f"{name} contains a label outside the HEALPix range")
    return value


def _readonly(value):
    value.setflags(write=False)
    return value


def nested_parent(cells, nside_child, nside_parent):
    """Map NESTED child pixels to parents without angular round trips.

    Equal resolutions are allowed. A finer parent or any non-power-of-two
    resolution is rejected rather than silently changing the partition (D4).
    """
    child, parent = _nside(nside_child), _nside(nside_parent)
    if parent > child:
        raise ValueError("parent nside must not exceed child nside")
    cells = _labels(cells, "cells", upper=12 * child**2)
    return cells // (child // parent)**2


@dataclass(frozen=True)
class NestedPatchScheme:
    """Independent storage and local-density resolutions (plan section 7.4)."""

    nside_store: int = 16
    nside_local: int = 8

    def __post_init__(self):
        object.__setattr__(self, "nside_store", _nside(self.nside_store))
        object.__setattr__(self, "nside_local", _nside(self.nside_local))
        if self.nside_local > self.nside_store:
            raise ValueError("local cells must be unions of storage cells")

    def storage_cells(self, ra, dec, *, units):
        """Label aligned RA/Dec arrays; units must be 'degrees' or 'radians'.

        RA is periodic, including negative values. Declination must lie within
        the poles. Callers must supply equatorial positions, not Galactic l/b.
        """
        import healpy as hp

        if units not in ("degrees", "radians"):
            raise ValueError("units must be 'degrees' or 'radians'")
        ra, dec = np.asarray(ra, dtype=float), np.asarray(dec, dtype=float)
        if ra.ndim != 1 or dec.ndim != 1 or ra.shape != dec.shape:
            raise ValueError("RA and Dec must be aligned one-dimensional arrays")
        if not np.isfinite(ra).all() or not np.isfinite(dec).all():
            raise ValueError("RA and Dec must be finite")
        pole, period = (90., 360.) if units == "degrees" else (np.pi / 2, 2 * np.pi)
        if np.any(abs(dec) > pole):
            raise ValueError("declination is outside the poles")
        phi = ra % period
        latitude = dec
        if units == "degrees":
            phi, latitude = np.deg2rad(phi), np.deg2rad(latitude)
        return np.asarray(hp.ang2pix(self.nside_store, np.pi / 2 - latitude,
                                    phi, nest=True), dtype=np.int64)

    def local_cells(self, storage_cells):
        return nested_parent(storage_cells, self.nside_store, self.nside_local)


@dataclass(frozen=True)
class CellRegionMapping:
    """Explicit whole-storage-cell grouping, independent of local-density cells.

    Repeated region labels are expected; repeated storage labels are an error.
    A partial sky mapping is allowed, but every queried cell must be present.
    Input row order and the numerical order of region IDs have no significance.
    """

    storage_cells: np.ndarray
    regions: np.ndarray
    nside_store: int = 16

    def __post_init__(self):
        nside = _nside(self.nside_store)
        cells = _labels(self.storage_cells, "storage_cells", upper=12 * nside**2)
        regions = _labels(self.regions, "regions")
        if cells.shape != regions.shape:
            raise ValueError("storage_cells and regions must have equal lengths")
        order = np.argsort(cells)
        cells, regions = cells[order], regions[order]
        if np.any(np.diff(cells) == 0):
            raise ValueError("duplicate storage cell in region mapping")
        object.__setattr__(self, "nside_store", nside)
        object.__setattr__(self, "storage_cells", _readonly(cells))
        object.__setattr__(self, "regions", _readonly(regions))

    def assign(self, cells):
        """Return region IDs in object order, failing on any unmapped cell."""
        cells = _labels(cells, "cells", upper=12 * self.nside_store**2)
        index = np.searchsorted(self.storage_cells, cells)
        if np.any(index == len(self.storage_cells)):
            raise ValueError("missing storage cell in region mapping")
        if np.any(self.storage_cells[index] != cells):
            raise ValueError("missing storage cell in region mapping")
        return self.regions[index]
