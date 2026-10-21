"""Shared count result type and backend provenance."""
from dataclasses import dataclass
import hashlib
from importlib.metadata import PackageNotFoundError, version
from importlib.util import find_spec
from pathlib import Path

import numpy as np

from .angular import PatchCounts


@dataclass(frozen=True)
class CountRun:
    counts: PatchCounts
    weighted_z: np.ndarray | None
    weighted_z2: np.ndarray | None
    files: tuple
    computed: int
    reused: int
    run_hash: str
    left_moments: np.ndarray
    right_moments: np.ndarray
    metadata: dict


def _software(backend):
    if backend != "pycorr":
        raise ValueError("production counts require pycorr/Corrfunc")
    packages = ["numpy", "pycorr", "Corrfunc"]
    result = {}
    for package in packages:
        try:
            result[package] = version(package)
        except PackageNotFoundError as exc:
            raise RuntimeError(f"required backend package is missing: {package}") from exc
    # Record our actual adapter/driver bytes, including uncommitted source edits.
    for source in (Path(__file__), Path(__file__).with_name("angular.py"),
                   Path(__file__).parents[1] / "counts_io.py"):
        result[source.name + "_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    if backend == "pycorr":
        for module in ("pycorr.corrfunc", "pycorr.twopoint_counter", "pycorr._utils",
                       "Corrfunc._countpairs_mocks"):
            spec = find_spec(module)
            if spec is None or spec.origin is None:
                raise RuntimeError(f"missing backend module: {module}")
            source = Path(spec.origin)
            result[module + "_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    return result
