"""Bounded numerical product IO, extracted from the production stage scripts."""
import hashlib
import os
import re
from pathlib import Path
from math import prod
from zipfile import ZipFile, ZIP_STORED
import numpy as np
from .counts_io import array_digest, canonical_json, publish_manifest

def _hash(value):
    if not isinstance(value, str) or re.fullmatch('[0-9a-f]{64}', value) is None:
        raise ValueError('SHA256 digest required')

def _digest_arrays(arrays, metadata):
    """Existing array_digest byte protocol, also defined for zero-sized axes."""
    if not any(np.asarray(a).size == 0 for a in arrays.values()):
        return array_digest(arrays, metadata)
    digest = hashlib.sha256(canonical_json(metadata).encode())
    for name, array in sorted(arrays.items()):
        value = np.ascontiguousarray(array)
        if value.dtype.kind not in "biuf":
            raise ValueError("pilot arrays must be real numeric arrays")
        digest.update(canonical_json([name, value.dtype.str, value.shape]).encode())
        if value.size:
            data = memoryview(value).cast("B")
            for start in range(0, len(data), 1024**2):
                digest.update(data[start:start+1024**2])
    return digest.hexdigest()

def _save_arrays(path, arrays, metadata):
    digest = _digest_arrays(arrays, metadata)
    with open(path, "xb") as stream:
        np.savez(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    return dict(file=Path(path).name, bytes=Path(path).stat().st_size, array_sha256=digest,
                metadata=metadata)

def read_arrays(path,record,limit,*,expected_shapes=None):
    """Strict stored-NPZ header/size/member/checksum validation before loading."""
    path=Path(path)
    if path.stat().st_size!=record['bytes'] or record['bytes']>limit:
        raise ValueError('array product size differs or exceeds bound')
    _hash(record['array_sha256'])
    with open(path,'rb') as stream:
        with ZipFile(stream) as archive:
            entries=archive.infolist(); names=[x.filename for x in entries]
            if (not entries or len(entries)>128 or len(set(names))!=len(names)
                    or any(e.compress_type!=ZIP_STORED or not e.filename.endswith('.npy')
                           or '/' in e.filename for e in entries)
                    or sum(e.file_size for e in entries)>limit):
                raise ValueError('invalid bounded stored array members')
            if expected_shapes is not None and set(names)!={k+'.npy' for k in expected_shapes}:
                raise ValueError('array product members differ')
            for entry in entries:
                with archive.open(entry) as member:
                    if np.lib.format.read_magic(member)!=(1,0):
                        raise ValueError('version-1 NPY header required')
                    shape,order,dtype=np.lib.format.read_array_header_1_0(member)
                    if (dtype.kind not in 'biuf' or dtype.hasobject or len(shape)>4
                            or member.tell()+prod(shape)*dtype.itemsize!=entry.file_size):
                        raise ValueError('array shape/type/payload differs')
                    if expected_shapes is not None and tuple(shape)!=tuple(expected_shapes[entry.filename[:-4]]):
                        raise ValueError('array shape differs from explicit grid')
        stream.seek(0)
        with np.load(stream,allow_pickle=False) as archive:
            arrays={k:archive[k] for k in archive.files}
    if _digest_arrays(arrays,record['metadata'])!=record['array_sha256']:
        raise ValueError('array product checksum differs')
    return arrays
