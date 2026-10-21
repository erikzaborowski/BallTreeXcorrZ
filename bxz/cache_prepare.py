"""Stream configured DR1/Y3 catalogues into the production cell-cache format."""
from collections import defaultdict
import hashlib
import math
import os
from pathlib import Path
import numpy as np
from .counts_io import canonical_json, publish_manifest
from .cache_io import PHOTO_DTYPE, REFERENCE_DTYPE

BUFFER_BYTES = 512*1024**2
CELL_BUFFER_BYTES = 1024**2

class CellWriter:
    """Bounded buffer pool with largest-buffer eviction and one file handle."""
    def __init__(self, directory, dtype):
        self.directory = Path(directory)
        self.directory.mkdir()
        self.dtype = dtype
        self.pieces, self.buffered = defaultdict(list), defaultdict(int)
        self.total_buffered = self.peak_buffered = self.flushes = 0
        self.cells = {}

    def append(self, values, labels):
        order = np.argsort(labels, kind='stable')
        labels, values = labels[order], values[order]
        starts = np.r_[0, np.flatnonzero(np.diff(labels))+1, len(labels)] if len(labels) else []
        for start, stop in zip(starts[:-1], starts[1:]):
            cell = int(labels[start])
            block = values[start:stop].copy()
            meta = self.cells.setdefault(cell, dict(count=0, sum_w=0., sum_w2=0., digest=hashlib.sha256()))
            meta['count'] += len(block)
            meta['sum_w'] += float(block['weight'].sum())
            meta['sum_w2'] += float(np.square(block['weight']).sum())
            self.pieces[cell].append(block)
            self.buffered[cell] += block.nbytes
            self.total_buffered += block.nbytes
            self.peak_buffered = max(self.peak_buffered, self.total_buffered)
            if self.buffered[cell] >= CELL_BUFFER_BYTES:
                self._flush(cell)
            while self.total_buffered > BUFFER_BYTES:
                self._flush(max(self.buffered, key=self.buffered.get))

    def _flush(self, cell):
        mode = 'ab' if self.cells[cell].get('written') else 'xb'
        with open(self.directory/f'{cell:04d}.bin', mode) as stream:
            for block in self.pieces.pop(cell):
                raw = memoryview(block).cast('B')
                stream.write(raw)
                self.cells[cell]['digest'].update(raw)
        self.cells[cell]['written'] = True
        self.total_buffered -= self.buffered.pop(cell)
        self.flushes += 1

    def finish(self):
        for cell in list(self.pieces):
            self._flush(cell)
        result = {}
        for cell, meta in sorted(self.cells.items()):
            path = self.directory/f'{cell:04d}.bin'
            if path.stat().st_size != meta['count']*self.dtype.itemsize:
                raise ValueError('cell shard size differs from streamed count')
            # Flush each completed shard once before publishing its manifest.
            with open(path, 'rb') as stream:
                os.fsync(stream.fileno())
            result[str(cell)] = dict(file=f'{self.directory.name}/{path.name}', count=meta['count'],
                sum_w=meta['sum_w'], sum_w2=meta['sum_w2'], bytes=path.stat().st_size,
                sha256=meta['digest'].hexdigest())
        return dict(dtype=self.dtype.descr, cells=result,
            moments=dict(count=sum(c['count'] for c in result.values()),
                         sum_w=math.fsum(c['sum_w'] for c in result.values()),
                         sum_w2=math.fsum(c['sum_w2'] for c in result.values())),
            peak_buffer_bytes=self.peak_buffered, flushes=self.flushes)


def prepare_cache(config, output, ranges):
    """Same selections/weights as the full-cache job, without a prior audit job."""
    import fitsio
    import healpy as hp
    from .catalogs import read_photometric, read_reference
    from .patches import NestedPatchScheme
    from .population import legacy_mask_pixels

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    photo = config['photometric']
    population = dict(name=config['population_id'],
        object_cuts=dict(targets='elg_mask == 0', randoms='elg_mask == 0'),
        normalization_domain='final_mask_and_object_cuts',
        mask_lookup=dict(lon='l', lat='b', pixelization='legacy_counts_map'),
        final_mask=dict(file=photo['mask'], nside=2048, coord='G', ordering='RING'))
    reader_config = dict(population=population, photometric={
        'data': dict(files=[photo['data']], weight='WEIGHT_SN'),
        'randoms': dict(files=[photo['randoms']], weight='unit')})
    entries = [dict(name='photo_'+kind, kind=kind, file=photo[kind])
               for kind in ('data', 'randoms')]
    for tracer in ranges:
        for cap in ('NGC', 'SGC'):
            for kind in ('data', 'randoms'):
                suffix = '_0_clustering.ran.fits' if kind == 'randoms' else '_clustering.dat.fits'
                path = Path(config['reference_root'])/(tracer+'_'+cap+suffix)
                if any(p.lower() in {'dr2', 'da2', 'da02'} for p in path.resolve().parts):
                    raise ValueError('only public DR1 spectroscopy is enabled')
                entries.append(dict(name=f'{tracer}_{cap}_{kind}', tracer=tracer,
                                    cap=cap, kind=kind, file=str(path)))
    def identity(path):
        s = Path(path).stat()
        return dict(bytes=s.st_size, mtime_ns=s.st_mtime_ns, ctime_ns=s.st_ctime_ns,
                    inode=s.st_ino, device=s.st_dev)
    paths = {e['name']: e['file'] for e in entries}
    paths['mask'] = photo['mask']
    before = {name: identity(path) for name, path in paths.items()}
    if before['mask']['bytes'] > 512*1024**2:
        raise ValueError('mask exceeds 512 MiB')
    digest = hashlib.sha256()
    with open(photo['mask'], 'rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            digest.update(block)
    if digest.hexdigest() != photo['mask_sha256']:
        raise ValueError('mask checksum differs from configuration')
    mask = hp.read_map(photo['mask'], dtype=np.float64, nest=False)
    if mask.shape != (hp.nside2npix(2048),) or not np.isin(mask, (0, 1)).all():
        raise ValueError('expected a binary Galactic RING NSIDE=2048 mask')
    mask = mask.astype(bool)
    scheme = NestedPatchScheme(16, 1)
    catalogues = {}
    for entry in entries:
        reference = 'tracer' in entry
        dtype = REFERENCE_DTYPE if reference else PHOTO_DTYPE
        with fitsio.FITS(entry['file']) as h:
            nrows = h[1].get_nrows()
        writer = CellWriter(output/entry['name'], dtype)
        if reference:
            ref_config = dict(weight_mode='M1', reference=dict(release='dr1', version='v1.5',
                caps=[entry['cap']], tracers={entry['tracer']: dict(z=ranges[entry['tracer']])},
                **{'data_files' if entry['kind']=='data' else 'random_files':
                   {entry['cap']: [entry['file']]}}))
        for start in range(0, nrows, config['chunk_rows']):
            size = min(config['chunk_rows'], nrows-start)
            if reference:
                batch = read_reference(ref_config, entry['tracer'], entry['cap'],
                    kind=entry['kind'], start=start, max_rows=size)
                keep = np.ones(len(batch), dtype=bool)
            else:
                batch = read_photometric(reader_config, kind=entry['kind'], start=start, max_rows=size)
                keep = mask[legacy_mask_pixels(batch.l, batch.b, 2048)]
            if batch.metadata['input_stop'] != start+size:
                raise ValueError('catalogue changed or input span is incomplete')
            values = np.empty(int(keep.sum()), dtype=dtype)
            for field in dtype.names:
                a = getattr(batch, field)[keep]
                values[field] = np.deg2rad(a) if field in ('ra', 'dec') else a
            writer.append(values, scheme.storage_cells(batch.ra[keep], batch.dec[keep], units='degrees'))
        saved = writer.finish()
        saved.update(source=entry, source_identity=before[entry['name']], input_rows=nrows)
        catalogues[entry['name']] = saved
        print(entry['name'], saved['moments'], flush=True)
    if any(identity(path) != before[name] for name, path in paths.items()):
        raise ValueError('input changed during cache preparation')
    manifest = dict(schema='bxz.dr1-full-cache.v1', completed=True, catalogues=catalogues,
        population=population, input_identity=before, mask_sha256=digest.hexdigest(),
        storage_scheme=dict(nside=16, frame='equatorial', ordering='NESTED'),
        units=dict(angles='radian', redshift='dimensionless'), reference_z_ranges=ranges,
        reference_weight='M1 supplied WEIGHT, no FKP',
        full_domain_moments={k: catalogues['photo_'+k]['moments'] for k in ('data','randoms')})
    publish_manifest(output/'manifest.json', manifest)
