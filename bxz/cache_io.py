"""Shared numeric cache formats and bounded, checksummed metadata reads."""
import hashlib
import json
import numpy as np
from .counts_io import canonical_json

PHOTO_DTYPE = np.dtype([(name, '<f8') for name in ('ra', 'dec', 'weight')] + [('row_index', '<i8')])
REFERENCE_DTYPE = np.dtype(PHOTO_DTYPE.descr + [('z', '<f8')])

def _read_manifest(path, max_bytes=2*1024**2, file_sha256=None):
    with open(path, "rb") as stream:
        raw = stream.read(max_bytes+1)
    if len(raw) > max_bytes:
        raise ValueError("manifest exceeds its explicit byte bound")
    if file_sha256 is not None and hashlib.sha256(raw).hexdigest() != file_sha256:
        raise ValueError("audit JSON differs from the pinned full-file SHA256")
    envelope = json.loads(raw)
    metadata = envelope["metadata"]
    if hashlib.sha256(canonical_json(metadata).encode()).hexdigest() != envelope["sha256"]:
        raise ValueError("manifest checksum mismatch")
    return metadata, envelope["sha256"]
