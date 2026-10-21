"""Bounded, immutable diagnostic DP-reference measurement archives (P3, §7.5).

Archives retain full and common-region deletion measurements, native reduced
counts, windows and their declared identities. They contain no covariance,
bias/kernel inference, physical-validation result or downstream fit. Only the
explicit archive path is accessed; provenance references are never followed.

Native DD/RD/RDz/RDz2 arrays have axes (realization, fine slice, cap, angle),
and reference endpoint totals have axes (realization, fine slice, cap). Caps
are explicitly NGC then SGC; realization zero is the full sample, followed
by region deletions in region_ids order. Source hashes have axes
(fine slice, cap, pair type), with pair types DD then RD. The stored random
support and windows use the declared alpha-scaled RD combination, not raw RD.
"""
from dataclasses import dataclass
import json
from math import prod
from numbers import Integral
import os
from pathlib import Path
import re
import tempfile
from types import MappingProxyType
from zipfile import BadZipFile, ZipFile

import numpy as np

from .counts_io import array_digest, canonical_json
from .reference_measurement import ReferenceJackknife, ReferenceMeasurement, COMBINATION, CAPS


class ReferenceMeasurementProductError(ValueError):
    """Incomplete, corrupt or incompatible cross-correlation product."""


@dataclass(frozen=True)
class ReferenceMeasurementRunArchive:
    """Detached metadata and immutable arrays, with SHA256 content identity."""
    arrays: object
    metadata: dict
    digest: str

    @property
    def sha256(self):
        return self.digest


_REALIZATION_ARRAYS = {
    "native_dd_by_cap": ("native_dd_by_cap",), "native_rd_by_cap": ("native_rd_by_cap",),
    "native_rd_z_by_cap": ("native_rd_z_by_cap",), "native_rd_z2_by_cap": ("native_rd_z2_by_cap",),
    "band_value": ("bands", "value"), "band_supported": ("bands", "supported"),
    "random_support": ("bands", "random_support"), "slice_window": ("bands", "slice_window"),
    "integrated_value": ("integrated", "value"),
    "integrated_supported": ("integrated", "supported"),
    "z_eff": ("z_eff",), "z2_eff": ("z2_eff",),
    "weight_data_s": ("weight_data_s",), "weight_random_s": ("weight_random_s",),
}
_OPERATOR_ARRAYS = {
    "theta_edges": ("theta_edges",), "r_edges": ("r_edges",),
    "output_bin": ("output_bin",), "omega": ("omega",),
    "delta_log_r": ("delta_log_r",),
    "geometry_retained_log_width": ("geometry", "retained_log_width"),
    "geometry_accessible_bounds": ("geometry", "accessible_bounds"),
    "geometry_included": ("geometry", "included"),
    "geometry_band_support": ("geometry", "band_support"),
}
_WEIGHTS = ("weight_data_s", "weight_random_s")
_ARRAY_NAMES = set(_REALIZATION_ARRAYS) | set(_OPERATOR_ARRAYS) | set(_WEIGHTS)
_NORMALIZATION = "retained_reference_endpoint_totals_per_cap_per_native_slice"
_SCOPE = "cross_correlation_only; no_bias_or_kernel_inference; no_physical_validation; no_downstream_fit"


def _digest(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _label(value):
    return type(value) is int and value >= 0


def _provenance(value):
    if not isinstance(value, dict) or value.get("data_kind") not in ("synthetic", "measured"):
        raise ReferenceMeasurementProductError("provenance requires explicit synthetic/measured data_kind")
    if not isinstance(value.get("population_id"), str) or not value["population_id"].strip():
        raise ReferenceMeasurementProductError("provenance requires population_id")
    if not _digest(value.get("config_hash")):
        raise ReferenceMeasurementProductError("provenance requires config_hash")
    for key in ("configuration", "mask", "software"):
        if not isinstance(value.get(key), dict) or not value[key]:
            raise ReferenceMeasurementProductError(f"provenance requires nonempty {key}")
    spec = value.get("input_count_specification", {})
    if (not isinstance(spec, dict) or not isinstance(spec.get("identity"), str)
            or not spec["identity"].strip() or not _digest(spec.get("sha256"))):
        raise ReferenceMeasurementProductError("provenance requires input count specification identity/SHA256")
    units = value.get("units", {})
    if (not isinstance(units, dict) or units.get("theta") != "radian"
            or units.get("correlation") != "dimensionless" or units.get("redshift") != "dimensionless"
            or not isinstance(units.get("separation"), str) or not units["separation"].strip()):
        raise ReferenceMeasurementProductError("provenance requires explicit theta/separation/correlation/redshift units")


def _validate(arrays, metadata):
    if (metadata.get("schema") != "bxz.reference-measurement-run.v1"
            or metadata.get("quantity") != "CROSS_CORRELATION" or metadata.get("stage") != "MEASUREMENT"
            or metadata.get("estimator") != "DP-reference" or metadata.get("covariance") != "not_included"
            or metadata.get("scope") != _SCOPE or metadata.get("normalization") != _NORMALIZATION
            or metadata.get("combination") != COMBINATION or metadata.get("cap_order") != list(CAPS)):
        raise ReferenceMeasurementProductError("unsupported measurement schema or interpretation")
    _provenance(metadata["provenance"])
    if metadata.get("data_kind") != metadata["provenance"]["data_kind"]:
        raise ReferenceMeasurementProductError("data_kind differs from provenance")
    regions, cells = metadata["region_ids"], metadata["cell_regions"]
    if (not isinstance(regions, list) or len(regions) < 2 or not all(map(_label, regions))
            or regions != sorted(set(regions))):
        raise ReferenceMeasurementProductError("region_ids must be sorted distinct nonnegative integers")
    if (not isinstance(cells, list) or not cells
            or any(not isinstance(item, list) or len(item) != 2 or not all(map(_label, item)) for item in cells)
            or [item[0] for item in cells] != sorted({item[0] for item in cells})
            or sorted({item[1] for item in cells}) != regions):
        raise ReferenceMeasurementProductError("invalid complete cell-region map")
    mode = metadata["pair_weighting"]
    if mode not in ("literal", "match"):
        raise ReferenceMeasurementProductError("invalid pair weighting")
    match_weight = 1 - len(regions)/(2 + np.sqrt(2)*(len(regions)-1)) if mode == "match" else None
    if metadata["cross_region_weight"] != match_weight:
        raise ReferenceMeasurementProductError("cross-region weight differs from declared map")
    declarations = metadata["realizations"]
    if not isinstance(declarations, list) or len(declarations) != len(regions)+1:
        raise ReferenceMeasurementProductError("realization identities disagree")
    for j, (declaration, omitted) in enumerate(zip(declarations, [None] + regions)):
        if (not isinstance(declaration, dict) or declaration.get("omitted_region") != omitted
                or (j and not _label(declaration.get("omitted_region")))
                or declaration.get("pair_weighting") != (mode if j else "literal")
                or declaration.get("normalization") != _NORMALIZATION
                or declaration.get("cross_region_weight") != (match_weight if j else None)
                or declaration.get("combination") != COMBINATION
                or declaration.get("cap_order") != list(CAPS)):
            raise ReferenceMeasurementProductError("realization weighting/normalization identity differs")
    if set(arrays) != _ARRAY_NAMES:
        raise ReferenceMeasurementProductError("missing or unexpected measurement arrays")
    theta, r, labels = arrays["theta_edges"], arrays["r_edges"], arrays["output_bin"]
    for name, edges in (("theta", theta), ("r", r)):
        if (edges.ndim != 1 or len(edges) < 2 or edges.dtype.kind not in "iuf"
                or not np.isfinite(edges).all() or np.any(edges <= 0) or np.any(np.diff(edges) <= 0)):
            raise ReferenceMeasurementProductError(f"invalid {name} edges")
    if theta[-1] > np.pi:
        raise ReferenceMeasurementProductError("theta edges must be radians")
    bins = metadata["n_output_bins"]
    if (not _label(bins) or bins == 0 or labels.ndim != 1 or len(labels) == 0
            or labels.dtype.kind not in "iu" or np.any(labels < 0) or np.any(labels >= bins)):
        raise ReferenceMeasurementProductError("invalid output-bin labels")
    nr, ns, nt, nb = len(regions)+1, len(labels), len(theta)-1, len(r)-1
    hashes = metadata["run_hashes"]
    if (not isinstance(hashes, list) or len(hashes) != ns
            or any(not isinstance(caps, list) or len(caps) != 2
                   or any(not isinstance(pair, list) or len(pair) != 2
                          or not all(map(_digest, pair)) for pair in caps) for caps in hashes)):
        raise ReferenceMeasurementProductError("invalid source slice/cap DD/RD run hashes")
    shapes = {name: (nr, ns, 2, nt) for name in ("native_dd_by_cap", "native_rd_by_cap", "native_rd_z_by_cap", "native_rd_z2_by_cap")}
    shapes.update({name: (nr, bins, nb) for name in ("band_value", "band_supported", "random_support", "z_eff", "z2_eff")})
    shapes.update(slice_window=(nr, bins, nb, ns), integrated_value=(nr, bins),
        integrated_supported=(nr, bins), weight_data_s=(nr, ns, 2), weight_random_s=(nr, ns, 2),
        theta_edges=(nt+1,), r_edges=(nb+1,), output_bin=(ns,), omega=(ns, nt, nb),
        delta_log_r=(nb,), geometry_retained_log_width=(ns,), geometry_accessible_bounds=(ns, 2),
        geometry_included=(ns,), geometry_band_support=(ns, nb))
    booleans = {"band_supported", "integrated_supported", "geometry_included", "geometry_band_support"}
    nullable = {"band_value", "integrated_value", "slice_window", "z_eff", "z2_eff"}
    for name, shape in shapes.items():
        value = arrays[name]
        if value.shape != shape or value.dtype.kind not in "biuf":
            raise ReferenceMeasurementProductError(f"invalid measurement array {name}")
        if name in booleans and value.dtype.kind != "b":
            raise ReferenceMeasurementProductError(f"{name} must be Boolean")
        if name not in nullable and not np.isfinite(value).all():
            raise ReferenceMeasurementProductError(f"nonfinite {name}")
        if name not in nullable | {"geometry_accessible_bounds"} and np.any(value < 0):
            raise ReferenceMeasurementProductError(f"negative {name}")
    if any(np.any(arrays[name] <= 0) for name in _WEIGHTS):
        raise ReferenceMeasurementProductError("reference totals must be positive")
    support = arrays["band_supported"]
    if not np.array_equal(support, arrays["random_support"] > 0):
        raise ReferenceMeasurementProductError("random support and band support disagree")
    integrated_support = arrays["integrated_supported"]
    if not np.array_equal(integrated_support, np.all(support, axis=2)):
        raise ReferenceMeasurementProductError("integrated support disagrees with bands")
    for name in nullable:
        value = arrays[name]
        selected = integrated_support if name == "integrated_value" else support
        if not np.isfinite(value[selected]).all() or not np.isnan(value[~selected]).all():
            raise ReferenceMeasurementProductError(f"{name} must be finite on support and NaN elsewhere")
        if name in ("slice_window", "z_eff", "z2_eff") and np.any(value[selected] < 0):
            raise ReferenceMeasurementProductError(f"negative {name}")
    geometry = metadata["geometry"]
    reasons = geometry["exclusion_reason"]
    if (not isinstance(reasons, list) or len(reasons) != ns or not all(isinstance(x, str) for x in reasons)
            or not np.array_equal(arrays["geometry_included"], [not reason for reason in reasons])
            or not isinstance(geometry["nominal_log_width"], (int, float)) or geometry["nominal_log_width"] <= 0
            or np.any(arrays["omega"] > 1) or np.any(arrays["delta_log_r"] <= 0)):
        raise ReferenceMeasurementProductError("invalid operator geometry")
    canonical_json(metadata)


def _get(value, path):
    for name in path:
        value = getattr(value, name)
    if not isinstance(value, np.ndarray):
        raise ReferenceMeasurementProductError("measurement payload must contain NumPy arrays")
    return value


def _payload(result, provenance, max_bytes):
    if not isinstance(result, ReferenceJackknife):
        raise TypeError("result must be a ReferenceJackknife")
    if result.combination != COMBINATION:
        raise ReferenceMeasurementProductError("unsupported reference combination")
    runs = (result.full,) + tuple(result.deletions)
    if any(not isinstance(run, ReferenceMeasurement) for run in runs):
        raise ReferenceMeasurementProductError("invalid realization type")
    op = result.full.operator
    shared = {name: _get(op, path) for name, path in _OPERATOR_ARRAYS.items()}
    geometry = dict(nominal_log_width=op.geometry.nominal_log_width,
                    exclusion_reason=op.geometry.exclusion_reason)
    for run in runs:
        if (canonical_json(run.run_hashes) != canonical_json(result.full.run_hashes)
                or canonical_json(run.cell_regions) != canonical_json(result.cell_regions)
                or run.operator.n_output_bins != op.n_output_bins
                or canonical_json((run.operator.geometry.nominal_log_width, run.operator.geometry.exclusion_reason))
                   != canonical_json((op.geometry.nominal_log_width, op.geometry.exclusion_reason))
                or any(not np.array_equal(_get(run.operator, path), shared[name])
                       for name, path in _OPERATOR_ARRAYS.items())):
            raise ReferenceMeasurementProductError("realizations have different shared operator/map/run identities")
    metadata = dict(schema="bxz.reference-measurement-run.v1", quantity="CROSS_CORRELATION",
        stage="MEASUREMENT", estimator="DP-reference", covariance="not_included", scope=_SCOPE,
        data_kind=provenance.get("data_kind"), provenance=provenance, n_output_bins=op.n_output_bins,
        region_ids=result.region_ids, cell_regions=result.cell_regions, run_hashes=result.full.run_hashes,
        pair_weighting=result.pair_weighting, cross_region_weight=result.cross_region_weight,
        normalization=result.full.normalization, combination=result.combination,
        cap_order=result.full.cap_order, geometry=geometry,
        realizations=[{name: getattr(run, name) for name in
            ("pair_weighting", "omitted_region", "cross_region_weight", "normalization", "combination", "cap_order")}
            for run in runs])
    metadata_json = canonical_json(metadata)
    metadata = json.loads(metadata_json)  # detach caller metadata, normalize tuples
    _provenance(metadata["provenance"])
    groups = {name: [_get(run, path) for run in runs] for name, path in _REALIZATION_ARRAYS.items()}
    group_bytes = 0
    for group in groups.values():
        if any(a.shape != group[0].shape or a.dtype.kind not in "biuf" for a in group):
            raise ReferenceMeasurementProductError("inconsistent realization array shapes/types")
        group_bytes += sum(a.size for a in group)*np.result_type(*[a.dtype for a in group]).itemsize
    # Check before stacking native arrays or creating a serialization buffer.
    # JSON is stored as NPY Unicode (four bytes per code point), not UTF-8.
    size = (sum(a.nbytes for a in shared.values()) + group_bytes
            + 4*len(metadata_json) + 4*64 + 4096*(len(_ARRAY_NAMES)+2))
    if size > max_bytes:
        raise ReferenceMeasurementProductError("measurement payload exceeds max_bytes")
    arrays = {**shared, **{name: np.stack(group) for name, group in groups.items()}}
    _validate(arrays, metadata)
    return arrays, metadata


def _budget(max_bytes):
    if isinstance(max_bytes, (bool, np.bool_)) or not isinstance(max_bytes, Integral) or max_bytes <= 0:
        raise ValueError("max_bytes must be an explicit positive integer")


def _inspect_archive(stream, max_bytes):
    if os.fstat(stream.fileno()).st_size > max_bytes:
        raise ReferenceMeasurementProductError("archive exceeds stored max_bytes")
    with ZipFile(stream) as archive:
        entries = archive.infolist()
        names = [entry.filename for entry in entries]
        if len(set(names)) != len(names):
            raise ReferenceMeasurementProductError("duplicate archive members")
        if sum(entry.file_size for entry in entries) > max_bytes:
            raise ReferenceMeasurementProductError("archive exceeds uncompressed max_bytes")
        if set(names) != {name+".npy" for name in _ARRAY_NAMES | {"metadata_json", "sha256"}}:
            raise ReferenceMeasurementProductError("missing or unexpected archive members")
        # Size checks above happen before decompressing even the NPY headers.
        # Check header-declared allocation sizes before numpy loads payloads.
        for entry in entries:
            with archive.open(entry) as member:
                version = np.lib.format.read_magic(member)
                if version == (1, 0):
                    shape, _, dtype = np.lib.format.read_array_header_1_0(member)
                elif version == (2, 0):
                    shape, _, dtype = np.lib.format.read_array_header_2_0(member)
                else:
                    raise ReferenceMeasurementProductError("unsupported NPY member version")
                text = entry.filename in ("metadata_json.npy", "sha256.npy")
                if (dtype.hasobject or (text and (shape != () or dtype.kind != "U"))
                        or (not text and dtype.kind not in "biuf") or any(x < 0 for x in shape)
                        or member.tell() + prod(shape)*dtype.itemsize != entry.file_size):
                    raise ReferenceMeasurementProductError("invalid NPY type/shape/payload size")


def read_reference_measurement_run(path, *, max_bytes):
    """Read one explicit archive, bounding stored and uncompressed bytes.

    Numeric arrays are read-only and the array mapping is immutable. The JSON
    metadata is a detached snapshot. Input objects, loaded-array overhead and
    temporary validation/serialization memory are not process-memory bounds.
    """
    _budget(max_bytes)
    try:
        with open(path, "rb") as stream:
            _inspect_archive(stream, max_bytes)
            stream.seek(0)
            with np.load(stream, allow_pickle=False) as stored:
                metadata = json.loads(str(stored["metadata_json"].item()))
                digest = str(stored["sha256"].item())
                arrays = {name: stored[name] for name in _ARRAY_NAMES}
        _validate(arrays, metadata)
        if not _digest(digest) or array_digest(arrays, metadata) != digest:
            raise ReferenceMeasurementProductError("measurement checksum mismatch")
    except ReferenceMeasurementProductError:
        raise
    except (OSError, ValueError, KeyError, TypeError, AttributeError, EOFError, BadZipFile) as error:
        raise ReferenceMeasurementProductError("unreadable measurement archive") from error
    for value in arrays.values():
        value.flags.writeable = False
    return ReferenceMeasurementRunArchive(MappingProxyType(arrays), metadata, digest)


def publish_reference_measurement_run(path, result, *, provenance, max_bytes):
    """Publish atomically, reuse identical content, and refuse changed content.

    max_bytes bounds the stored archive and its uncompressed members, including
    a conservative NPY/ZIP header allowance checked before large stacking or
    serialization. It is not a bound on input objects or process peak memory.
    No provenance path is opened or inferred to exist.
    The input_count_specification provenance entry should identify the prepared
    reconstruction specification linking both DD and RD sources; individual
    source count-run hashes are retained separately for every slice and cap.
    """
    _budget(max_bytes)
    _provenance(provenance)
    try:
        arrays, metadata = _payload(result, provenance, max_bytes)
    except ReferenceMeasurementProductError:
        raise
    except (ValueError, KeyError, AttributeError) as error:
        raise ReferenceMeasurementProductError("invalid measurement product") from error
    digest = array_digest(arrays, metadata)
    path = Path(path)
    if path.exists():
        existing = read_reference_measurement_run(path, max_bytes=max_bytes)
        if existing.digest != digest:
            raise ReferenceMeasurementProductError("existing measurement has different content")
        return existing
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".partial-", suffix=".npz", delete=False) as stream:
            temporary = Path(stream.name)
            np.savez(stream, **arrays, metadata_json=canonical_json(metadata), sha256=digest)
            stream.flush()
            os.fsync(stream.fileno())
        read_reference_measurement_run(temporary, max_bytes=max_bytes)
        try:
            os.link(temporary, path)
        except FileExistsError:
            if read_reference_measurement_run(path, max_bytes=max_bytes).digest != digest:
                raise ReferenceMeasurementProductError("concurrent measurement has different content")
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return read_reference_measurement_run(path, max_bytes=max_bytes)
