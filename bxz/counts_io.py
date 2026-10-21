"""Checksummed numeric metadata and atomic no-clobber publication."""
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np


class CountProductError(ValueError):
    """Incomplete, corrupt, or incompatible count product."""


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def publish_manifest(path, metadata):
    """Atomically publish JSON run metadata, including empty-catalogue totals."""
    path = Path(path)
    payload = canonical_json(metadata)
    envelope = canonical_json({"metadata": metadata,
                               "sha256": hashlib.sha256(payload.encode()).hexdigest()})
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=".partial-", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(envelope)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if canonical_json(read_manifest(path)) != payload:
                raise CountProductError("existing run manifest differs")
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def read_manifest(path):
    try:
        envelope = json.loads(Path(path).read_text(encoding="utf-8"))
        payload = canonical_json(envelope["metadata"])
        if hashlib.sha256(payload.encode()).hexdigest() != envelope["sha256"]:
            raise CountProductError("run manifest checksum mismatch")
        return envelope["metadata"]
    except CountProductError:
        raise
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise CountProductError("unreadable run manifest") from exc


def array_digest(arrays, metadata):
    """Hash bounded supplied arrays; never discover or read catalogue files."""
    digest = hashlib.sha256(canonical_json(metadata).encode())
    for name, array in sorted(arrays.items()):
        value = np.ascontiguousarray(array)
        if value.dtype.kind not in "biuf":
            raise CountProductError("count arrays must be real numeric arrays")
        digest.update(canonical_json([name, value.dtype.str, value.shape]).encode())
        data = memoryview(value).cast("B")
        for start in range(0, len(data), 1024 * 1024):
            digest.update(data[start:start + 1024 * 1024])
    return digest.hexdigest()
