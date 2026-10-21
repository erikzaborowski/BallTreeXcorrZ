"""Bounded, configuration-driven catalogue input (P1, §4.7 and §5.3).

All reads require a maximum number of *input* rows, before selection. Callers
advance ``start`` by ``metadata['input_rows']``; selection never refills a batch.
The returned weight sums describe this batch, not the complete population.
"""

from dataclasses import dataclass
from pathlib import Path
from collections.abc import Mapping
from numbers import Integral

import fitsio
import numpy as np


DR1_WEIGHT_SOURCES = (
    "https://desidatamodel.readthedocs.io/en/25.3/DESI_ROOT/survey/catalogs/"
    "RELEASE/LSS/SPECPROD/LSScats/VERSION/data_clustering.html",
    "https://lss.fnal.gov/archive/2024/pub/fermilab-pub-24-0850-ppd.pdf#page=41",
    "https://github.com/desihub/LSS/blob/main/py/LSS/common_tools.py",
)
M1_RECIPE = "supplied WEIGHT; w_comp*w_sys*w_zfail/<w_comp>(NTILE); no FKP"


@dataclass(frozen=True)
class CatalogBatch:
    """Aligned selected rows; angles in degrees and weights dimensionless.

    ``region`` retains the declared source column. PHOTSYS is not a resolved
    three-region imaging classification. ``row_index`` indexes the input file.
    """

    ra: np.ndarray
    dec: np.ndarray
    weight: np.ndarray
    row_index: np.ndarray
    metadata: dict
    z: np.ndarray | None = None
    l: np.ndarray | None = None
    b: np.ndarray | None = None
    region: np.ndarray | None = None

    def __len__(self):
        return len(self.ra)

    @property
    def weight_sum(self):
        return float(np.sum(self.weight, dtype=np.float64))

    @property
    def weight_squared_sum(self):
        return float(np.sum(self.weight * self.weight, dtype=np.float64))


def require_weight_mode(mode):
    if mode in ("M2", "M3", "M2_M3", "M2/M3"):
        raise NotImplementedError(
            "M2/M3 disabled: pin catalogue/software versions, probability offsets, "
            "realization count, finite-realization correction, DD/DR angular weights "
            "and normalization, then independently validate a bounded toy case "
            "without double assignment correction (§4.7 E1)."
        )
    if mode != "M1":
        raise ValueError(f"Unknown weight mode: {mode!r}")


def _mapping(config):
    if isinstance(config, Mapping):
        return config
    return config.as_dict()


def _path(value):
    value = str(value)
    if not value or "<" in value or ">" in value:
        raise ValueError("Catalogue path is unresolved")
    return Path(value)


def _bounded_table(path, columns, *, start, max_rows, ext=1):
    # No unbounded default, no directory discovery, no implicit concatenation.
    for name, value, minimum in (("start", start, 0), ("max_rows", max_rows, 1)):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    columns = list(dict.fromkeys(columns))
    with fitsio.FITS(str(path), mode="r") as handle:
        hdu = handle[ext]
        missing = set(columns) - set(hdu.get_colnames())
        if missing:
            raise ValueError(f"Missing catalogue columns: {sorted(missing)}")
        total = hdu.get_nrows()
        stop = min(start + max_rows, total)
        rows = np.arange(min(start, total), stop, dtype=np.int64)
        table = hdu.read(columns=columns, rows=rows)
        header = hdu.read_header()
        units = {}
        for index, column in enumerate(hdu.get_colnames(), 1):
            if column in columns:
                units[column] = header.get(f"TUNIT{index}", "unspecified")
        release = header.get("DESIDR")
    return table, rows, {
        "file": str(path), "total_file_rows": total, "input_rows": len(rows),
        "input_start": start, "input_stop": stop, "requested_max_rows": max_rows,
        "columns_read": columns, "fits_units": units, "header_release": release,
        "complete_file_read": start == 0 and stop == total,
        "coordinate_units": "deg", "weight_units": "dimensionless",
    }


def _finite(values, name):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must be a finite one-dimensional column")
    return values


def _angle_units(metadata, columns):
    for column in columns:
        unit = str(metadata["fits_units"][column]).strip().lower()
        if unit not in ("", "unspecified", "deg", "degree", "degrees"):
            raise ValueError(f"{column} FITS unit {unit!r} is incompatible with degrees")


def _coordinates(table, lon, lat):
    longitude = _finite(table[lon], lon)
    latitude = _finite(table[lat], lat)
    if np.any((longitude < 0) | (longitude >= 360)):
        raise ValueError(f"{lon} must be in [0, 360) degrees")
    if np.any(np.abs(latitude) > 90):
        raise ValueError(f"{lat} must be in [-90, 90] degrees")
    return longitude, latitude


def _weights(table, name):
    weight = _finite(table[name], name)
    if np.any(weight < 0):
        raise ValueError(f"{name} weights must be nonnegative")
    return weight


def _column_weight(value):
    if isinstance(value, (tuple, list)) and len(value) == 1:
        value = value[0]
    if not isinstance(value, str) or not value or "<" in value:
        raise ValueError("Exactly one explicit weight column is required")
    return value


def _kind(kind):
    if kind not in ("data", "randoms"):
        raise ValueError("kind must be 'data' or 'randoms'")


def read_reference(config, tracer, cap, *, kind="data", max_rows, start=0,
                   random_index=0, file_index=0):
    """Read one explicitly configured DR1 file, with a half-open z selection.

    Tracer settings can override ``caps``, ``columns``, and ``data_files`` or
    ``random_files`` (each a cap-to-file-list mapping). Otherwise release-level
    filename patterns are formatted; no glob or directory listing is used.
    """
    config = _mapping(config)
    require_weight_mode(config.get("weight_mode", "M1"))
    _kind(kind)
    reference = config["reference"]
    release = reference.get("release", "dr1").lower()
    version = reference.get("version", "v1.5")
    if release != "dr1":
        raise ValueError("Only public DR1 reference access is enabled; DR2/DA2 is gated")
    if version != "v1.5":
        raise ValueError("M1 recipe is verified for DR1 v1.5 only")
    settings = reference["tracers"][tracer]
    if cap not in settings.get("caps", reference["caps"]):
        raise ValueError(f"Cap {cap!r} is not configured for {tracer}")
    columns = {"ra": "RA", "dec": "DEC", "z": "Z", "region": "PHOTSYS",
               "weight": "WEIGHT"}
    columns.update(reference.get("columns", {}))
    columns.update(settings.get("columns", {}))
    if columns["weight"] != "WEIGHT":
        raise ValueError("M1 requires the supplied combined weight column WEIGHT")
    key = "data_files" if kind == "data" else "random_files"
    files = settings.get(key, reference.get(key))
    if files is not None:
        if isinstance(files, Mapping):
            files = files[cap]
        if isinstance(files, str):
            files = [files]
        path = _path(files[file_index])
    else:
        pattern = reference["data_pattern" if kind == "data" else "random_pattern"]
        path = _path(pattern.format(root=reference["root"], tracer=tracer,
                                    cap=cap, i=random_index))
    # Check path tokens before opening even a FITS header, including symlinks.
    for candidate in (path, path.resolve()):
        if any(token.lower() in {"dr2", "da2", "da02"} for token in candidate.parts):
            raise ValueError("DR2/DA2 spectroscopic paths are prohibited")
    table, rows, metadata = _bounded_table(
        path, [columns[key] for key in ("ra", "dec", "z", "region", "weight")],
        start=start, max_rows=max_rows, ext=reference.get("ext", 1))
    header_release = metadata["header_release"]
    if header_release is not None and str(header_release).lower() != release:
        raise ValueError("FITS release does not match configured DR1 release")
    ra, dec = _coordinates(table, columns["ra"], columns["dec"])
    _angle_units(metadata, [columns["ra"], columns["dec"]])
    z = _finite(table[columns["z"]], columns["z"])
    weight = _weights(table, columns["weight"])
    bounds = np.asarray(settings["z"], dtype=float)
    if bounds.shape != (2,) or not np.all(np.isfinite(bounds)) or bounds[0] >= bounds[1]:
        raise ValueError("Tracer z interval must have two finite increasing endpoints")
    keep = (z >= bounds[0]) & (z < bounds[1])
    metadata.update({"catalogue": "reference", "kind": kind, "release": release,
                     "version": version, "tracer": tracer, "cap": cap,
                     "z_range": bounds.tolist(), "z_interval": "[lower, upper)",
                     "selected_rows": int(keep.sum()), "weight_mode": "M1",
                     "weight_column": columns["weight"], "weight_recipe": M1_RECIPE,
                     "weight_recipe_sources": DR1_WEIGHT_SOURCES,
                     "recipe_checked_on": "2026-10-17", "fkp_applied": False,
                     "region_column": columns["region"],
                     "region_status": "PHOTSYS does not resolve S_DECaLS versus DES"
                     if columns["region"] == "PHOTSYS" else "declared column",
                     "global_validation": "pending; batch validation only"})
    return CatalogBatch(ra[keep], dec[keep], weight[keep], rows[keep], metadata,
                        z=z[keep], region=table[columns["region"]][keep])


def read_photometric(config, *, kind="data", max_rows, start=0, file_index=0):
    """Read positions/weights from the same table; ignore placeholder Z entirely.

    Apply only the declared object cut here. The final-mask lookup is separate,
    and no random subsampling, overlap restriction or region cut is performed.
    """
    config = _mapping(config)
    _kind(kind)
    population = config["population"]
    photometric = config["photometric"]
    settings = photometric[kind]
    cut_kind = "targets" if kind == "data" else "randoms"
    if population["object_cuts"][cut_kind].replace(" ", "") != "elg_mask==0":
        raise ValueError("Only the declared elg_mask == 0 object selection is enabled")
    columns = {"ra": settings.get("ra", "RA"), "dec": settings.get("dec", "DEC"),
               "l": "l", "b": "b", "object_mask": "elg_mask", "region": "PHOTSYS"}
    columns.update(settings.get("columns", {}))
    lookup = population.get("mask_lookup", {})
    columns["l"] = lookup.get("lon", columns["l"])
    columns["b"] = lookup.get("lat", columns["b"])
    declared_region = population.get("regions", {}).get("column", "<ELG_REGION_COL>")
    if "<" not in declared_region:
        columns["region"] = declared_region
    weight_column = _column_weight(settings.get("weight", "WEIGHT_SN" if kind == "data" else "unit"))
    if kind == "randoms" and weight_column != "unit":
        raise ValueError("The fiducial photometric random weights must be unit")
    if kind == "data" and weight_column != "WEIGHT_SN":
        raise ValueError("The fiducial photometric target weight must be WEIGHT_SN")
    requested = list(columns.values())
    if kind == "data":
        requested.append(weight_column)
    if "Z" in requested:
        raise ValueError("Photometric placeholder Z cannot be mapped to any input")
    path = _path(settings["files"][file_index])
    table, rows, metadata = _bounded_table(path, requested, start=start,
                                          max_rows=max_rows, ext=settings.get("ext", 1))
    ra, dec = _coordinates(table, columns["ra"], columns["dec"])
    lon, lat = _coordinates(table, columns["l"], columns["b"])
    _angle_units(metadata, [columns[key] for key in ("ra", "dec", "l", "b")])
    object_mask = table[columns["object_mask"]]
    if object_mask.dtype.kind not in "iu":
        raise ValueError("elg_mask must be an integer bit mask")
    weight = _weights(table, weight_column) if kind == "data" else np.ones(len(table))
    keep = object_mask == 0
    metadata.update({"catalogue": "photometric", "kind": kind,
                     "release": photometric.get("release", "unresolved"),
                     "version": photometric.get("version", "unresolved"),
                     "population": population["name"],
                     "normalization_domain": population["normalization_domain"],
                     "final_mask": dict(population["final_mask"]),
                     "final_mask_applied": False, "mask_lookup": dict(lookup),
                     "object_cut": "elg_mask == 0", "selected_rows": int(keep.sum()),
                     "weight_column": weight_column, "ignored_columns": ["Z", "WEIGHT"],
                     "galactic_coordinates": "stored l/b; degrees",
                     "region_column": columns["region"],
                     "region_status": "unresolved: PHOTSYS merges S_DECaLS and DES"
                     if columns["region"] == "PHOTSYS" else "declared; provenance pending",
                     "global_validation": "pending; batch validation only",
                     "production_compatible": False})
    return CatalogBatch(ra[keep], dec[keep], weight[keep], rows[keep], metadata,
                        l=lon[keep], b=lat[keep], region=table[columns["region"]][keep])
