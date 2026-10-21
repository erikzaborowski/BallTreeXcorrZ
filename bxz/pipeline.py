"""Production path: catalogues -> counts -> correlations -> reference bias -> b*phi.

This runner replaces the job-specific pilot/diagnostic orchestration. Numerical
estimators, windows, resampling, bias fits and matter projections are retained
from the production implementation. No scheduler or downstream cosmology fit.
"""
from concurrent.futures import ProcessPoolExecutor
from decimal import Decimal
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import time

import numpy as np

from .cache_io import _read_manifest, PHOTO_DTYPE
from .counts_io import canonical_json, publish_manifest
from .cosmology import PLANCK18_BASE, FlatCosmology, UNITS
from .operator import MeasurementOperator
from .products import _save_arrays, read_arrays

TRACERS = ('BGS_BRIGHT', 'LRG', 'ELG_LOPnotqso', 'QSO')
RANGES = dict(BGS_BRIGHT=(.05,.5), LRG=(.4,1.1), ELG_LOPnotqso=(.8,1.6), QSO=(.8,3.5))
CAPS = ('NGC', 'SGC')
THETA = np.deg2rad(.05*10**(.05*np.arange(-24,36)))
R_EDGES = (2.5,5.,10.)
CUTS = (24,59)
BACKGROUND = FlatCosmology(PLANCK18_BASE)
APPROXIMATIONS = dict(reference_model='Limber; no RSD/FoG/redshift-error/integral-constraint correction',
    angular='flat_sky_uniform_area_full_native_annuli; not_exact_survey_pair_averaging',
    radial='cell_radial_angular_separability; distinct-pair finite-random correction',
    kernel='signed apparent b*phi; magnification uncorrected',
    bias='common NGC/SGC bias per .05 bin; refit each common jackknife deletion')


def digest(value):
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def z_edges(tracer, step):
    low, high = map(lambda x: Decimal(str(x)), RANGES[tracer])
    step = Decimal(str(step))
    return np.array([float(low+i*step) for i in range(int((high-low)/step)+1)])


def layout(tracer):
    edges = z_edges(tracer,.01)
    # Retain the completed run's binary count distances and decimal theory nodes.
    chi = BACKGROUND.chi((edges[:-1]+edges[1:])/2)
    z = np.array([float((Decimal(str(lo))+Decimal(str(hi)))/2)
                  for lo,hi in zip(edges[:-1],edges[1:])])
    labels = np.arange(len(z))//5
    op = MeasurementOperator(THETA, R_EDGES, chi, edges[:-1], labels,
        n_output_bins=len(z)//5, lower_edge_index=CUTS[0], upper_edge_index=CUTS[1])
    return edges,z,chi,op


def auto_geometry(tracer, index):
    coarse = z_edges(tracer,.05)
    low,high = coarse[index:index+2]
    midpoint = float((Decimal(str(low))+Decimal(str(high)))/2)
    upper = min(int(np.searchsorted(THETA, 10/BACKGROUND.chi(low))), CUTS[1])
    theta = THETA[:upper+1]
    chi = float(BACKGROUND.chi(midpoint))
    radial = np.array([float(Decimal(str(low))+i*Decimal('.001')) for i in range(51)])
    op = MeasurementOperator(theta,R_EDGES,[chi],[low],[0],n_output_bins=1,
                             lower_edge_index=CUTS[0],upper_edge_index=upper)
    return (float(low),float(high)),chi,theta,radial,op


def load_config(path):
    path = Path(path)
    if path.stat().st_size > 65536:
        raise ValueError('configuration exceeds 64 KiB')
    cfg = json.loads(path.read_text())
    if not Path(cfg['output']).is_absolute():
        raise ValueError('output must be an absolute scratch path')
    for key in ('chunk_rows','n_regions','region_seed'):
        if type(cfg[key]) is not int or cfg[key] <= 0:
            raise ValueError(key+' must be a positive integer')
    if cfg['chunk_rows']>262144 or not 2<=cfg['n_regions']<=150:
        raise ValueError('chunk_rows <=262144 and 2<=n_regions<=150 required')
    for key in ('max_tasks','max_memory_bytes','max_result_bytes','max_product_bytes','max_matter_work'):
        if type(cfg['limits'][key]) is not int or cfg['limits'][key]<=0:
            raise ValueError('positive explicit resource limits required')
    if (not cfg['prescriptions'] or cfg['prescriptions'][0]!='takahashi'
            or len(set(cfg['prescriptions']))!=len(cfg['prescriptions'])
            or not set(cfg['prescriptions'])<={'takahashi','mead2020'}):
        raise ValueError('Halofit is the default; HMcode 2020 is the supported alternative')
    if not isinstance(cfg['population_id'],str) or not cfg['population_id'].strip():
        raise ValueError('population_id is required')
    return cfg


def initialize(cfg):
    root = Path(cfg['output']); root.mkdir(parents=True,exist_ok=True)
    package = Path(__file__).parent
    # Only the bounded runtime package and its single entrypoint, never data directories.
    sources = sorted(list(package.glob('*.py'))+list((package/'paircount').glob('*.py')))
    sources.append(package.parent/'run_dr1.py')
    hashes = {str(p.relative_to(package.parent)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    record = dict(config=cfg,source_sha256=hashes,cosmology=PLANCK18_BASE.as_dict(),
                  approximations=APPROXIMATIONS)
    signature = digest(record)
    existing = root/'run.json'
    if existing.exists():
        saved,_ = _read_manifest(existing)
        if saved['signature']!=signature:
            raise ValueError('configuration/source changed: use a new output directory')
        return saved
    commit = subprocess.check_output(['git','-C',str(package.parent),'rev-parse','HEAD'],text=True).strip()
    record.update(signature=signature,git_commit=commit)
    publish_manifest(existing,record)
    return record


def cache_input(cfg):
    path = Path(cfg.get('cache_manifest') or Path(cfg['output'])/'cache/manifest.json')
    cache, sha = _read_manifest(path,max_bytes=32*1024**2)
    if (cache.get('schema')!='bxz.dr1-full-cache.v1' or not cache.get('completed')
            or cache['population']['name']!=cfg['population_id']):
        raise ValueError('completed matching-population cache required')
    # Pin the input once; cached stages must not mix catalogue revisions.
    publish_manifest(Path(cfg['output'])/'cache-input.json',dict(path=str(path.resolve()),sha256=sha))
    return path,cache,sha


def provenance(cfg,run,cache,sha):
    science = dict(cosmology=PLANCK18_BASE.as_dict(),dz_fine=.01,dz_out=.05,
        r_edges_hmpc=R_EDGES,hard_cut_indices=CUTS,reference_ranges=RANGES,
        reference_weight='M1 supplied WEIGHT, no FKP',estimator='DP-reference',
        mask_sha256=cache['mask_sha256'],pipeline_signature=run['signature'])
    return dict(config=science,config_hash=digest(science),population_id=cfg['population_id'],
        catalogues=dict(photometric=cache['population'],reference=dict(release='DR1',version='v1.5')),
        input_checksums=dict(cache_manifest=sha),git_commit=run['git_commit'])


def region_map(cfg):
    saved,sha = _read_manifest(Path(cfg['output'])/'regions.json',max_bytes=4*1024**2)
    if saved['cache_metadata_sha256']!=cache_input(cfg)[2]:
        raise ValueError('region map belongs to a different cache')
    return dict(saved['cell_regions']),sha


def make_regions(cfg):
    from .jackknife_regions import accumulate_sky_cells, fit_cell_regions
    from .paircount.cache_driver import _group_cells, _load_cell
    import healpy as hp
    path,cache,sha = cache_input(cfg)
    required = sorted({int(c) for group in cache['catalogues'].values() for c in group['cells']})
    if cfg.get('regions_file'):
        record,_ = _read_manifest(cfg['regions_file'],max_bytes=4*1024**2)
        mapping = dict(record['cell_regions'])
        if not set(required)<=mapping.keys() or len(set(mapping.values()))!=cfg['n_regions']:
            raise ValueError('saved region map does not cover the cache or has the wrong region count')
    else:
        group = cache['catalogues']['photo_randoms']
        cells,moments = _group_cells(group,'photo_randoms',PHOTO_DTYPE)
        def batches():
            for cell in cells:
                a,labels = _load_cell(path,'photo_randoms',group,cell,PHOTO_DTYPE,
                                     max_bytes=cfg['limits']['max_memory_bytes'])
                yield labels,a['ra'],a['dec'],a['weight']
        stats = accumulate_sky_cells(batches(),units='radians',nside_store=16,max_cells=3072,
            max_batch_objects=int(moments[:,0].max()),max_objects=int(moments[:,0].sum()))
        if not np.array_equal(stats.count,moments[:,0]) or not np.array_equal(stats.weight_sum,moments[:,1]):
            raise ValueError('region statistics disagree with cached random moments')
        fit = fit_cell_regions(stats,n_regions=cfg['n_regions'],seed=cfg['region_seed'],
                               n_init=10,max_iter=300,tol=.0001,max_cells=3072)
        mapping = dict(zip(map(int,fit.mapping.storage_cells),map(int,fit.mapping.regions)))
        for cell in set(required)-mapping.keys():
            xyz = np.asarray(hp.pix2vec(16,cell,nest=True))
            mapping[cell] = int(np.argmin(np.sum((fit.centers-xyz)**2,axis=1)))
    publish_manifest(Path(cfg['output'])/'regions.json',dict(cell_regions=sorted(mapping.items()),
        cache_metadata_sha256=sha,source=cfg.get('regions_file') or 'photometric_randoms_weighted_kmeans'))


def count_tasks(auto):
    return [(tracer,i,cap,kind) for tracer in TRACERS
            for i in range(len(z_edges(tracer,.05 if auto else .01))-1)
            for cap in CAPS for kind in (('DD','DR','RR') if auto else ('DD','RD'))]


def task_path(root,auto,task):
    tracer,index,cap,kind = task
    return Path(root)/('auto' if auto else 'cross')/f'{tracer}-{index:03d}-{cap}-{kind}.json'


def count_one(payload):
    cfg,run,auto,task = payload
    from .paircount.cache_driver import run_cache_angular_counts
    from .paircount.reference_cache_driver import run_reference_cache_counts
    from .paircount.reference_auto_cache import run_reference_auto_cache_counts
    path,cache,sha = cache_input(cfg)
    prov = provenance(cfg,run,cache,sha)
    tracer,i,cap,kind = task
    if auto:
        bounds,chi,theta,radial,_ = auto_geometry(tracer,i)
    else:
        edges,z,distances,_ = layout(tracer)
        bounds,chi,theta = edges[i:i+2],float(distances[i]),THETA
    context = dict(slice_edges=list(bounds),chi_hmpc=chi,hard_cut_indices=[CUTS[0],len(theta)-1],
                   mask=dict(sha256=cache['mask_sha256']),patch_scheme=cache['storage_scheme'])
    args = dict(slice_id=f'{tracer}-{bounds[0]:.2f}-{bounds[1]:.2f}',count_type=kind,
        provenance=prov,measurement_context=context,backend='pycorr',
        **{k:cfg['limits'][k] for k in ('max_tasks','max_memory_bytes','max_result_bytes')})
    output = Path(cfg['output'])/('auto' if auto else 'cross')/'counts'
    if auto:
        left,right = dict(DD=('data','data'),DR=('data','randoms'),RR=('randoms','randoms'))[kind]
        result = run_reference_auto_cache_counts(path,f'{tracer}_{cap}_{left}',f'{tracer}_{cap}_{right}',
                                                 theta,output,radial_edges=radial,**args)
    else:
        fun = run_cache_angular_counts if kind=='DD' else run_reference_cache_counts
        group = f'{tracer}_{cap}_'+('data' if kind=='DD' else 'randoms')
        result = fun(path,group,'photo_data',theta,output,**args)
    record = dict(run_hash=result.run_hash,cache_sha256=sha,signature=run['signature'],
                  task=list(task),auto=auto)
    publish_manifest(task_path(cfg['output'],auto,task),record)
    return task


def counts(cfg,run,auto,workers):
    tasks = count_tasks(auto)
    payloads = ((cfg,run,auto,t) for t in tasks)
    if workers==1:
        results = map(count_one,payloads)
        for task in results: print('counted',task,flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
            for task in pool.map(count_one,payloads,chunksize=1): print('counted',task,flush=True)


def read_count(cfg,run,auto,task):
    from .paircount.cache_driver import read_cache_angular_run
    from .paircount.reference_auto_cache import read_reference_auto_cache_run
    record,_ = _read_manifest(task_path(cfg['output'],auto,task))
    _,_,sha = cache_input(cfg)
    if record['task']!=list(task) or record['signature']!=run['signature'] or record['cache_sha256']!=sha:
        raise ValueError('count completion belongs to a different run')
    directory = Path(cfg['output'])/('auto' if auto else 'cross')/'counts'/record['run_hash']
    reader = read_reference_auto_cache_run if auto else read_cache_angular_run
    result = reader(directory,max_result_bytes=cfg['limits']['max_result_bytes'])
    if result.run_hash!=record['run_hash']:
        raise ValueError('count archive identity differs')
    return result


def measure(cfg,run):
    from .reference_measurement import reconstruct_dp_reference_jackknife
    from .reference_measurement_io import publish_reference_measurement_run
    mapping,region_sha = region_map(cfg)
    _,cache,_ = cache_input(cfg)
    dest = Path(cfg['output'])/'measurement'; dest.mkdir(exist_ok=True)
    for tracer in TRACERS:
        edges,z,chi,op = layout(tracer)
        def pairs():
            for i in range(len(z)):
                for cap in CAPS:
                    yield tuple(read_count(cfg,run,False,(tracer,i,cap,kind)) for kind in ('DD','RD'))
        result = reconstruct_dp_reference_jackknife(pairs(),r_edges=R_EDGES,output_bin=op.output_bin,
            n_output_bins=op.n_output_bins,cell_regions=mapping,pair_weighting='match',
            max_realizations=cfg['n_regions']+1,max_result_bytes=cfg['limits']['max_result_bytes'])
        prov = dict(data_kind='measured',population_id=cfg['population_id'],config_hash=run['signature'],
            configuration=cfg,input_count_specification=dict(identity=str(Path(cfg['output'])/'run.json'),sha256=run['signature']),
            common_regions_sha256=region_sha,mask=dict(sha256=cache['mask_sha256']),software=dict(numpy=np.__version__),
            units=dict(theta='radian',separation='h^-1 Mpc',correlation='dimensionless',redshift='dimensionless'))
        publish_reference_measurement_run(dest/(tracer+'.npz'),result,provenance=prov,
                                          max_bytes=cfg['limits']['max_product_bytes'])


def save_product(path,arrays,metadata,limit):
    path = Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    if sum(a.nbytes for a in arrays.values())+4096*(len(arrays)+2)>limit:
        raise ValueError('output exceeds product bound')
    record = _save_arrays(path,arrays,metadata)
    publish_manifest(path.with_suffix('.json'),record)
    return record


def load_product(path,limit):
    path = Path(path)
    record,_ = _read_manifest(path.with_suffix('.json'))
    return read_arrays(path,record,limit),record


def matter(cfg,run):
    from .theory_camb import LinearCAMBSettings
    from .theory_camb_nonlinear import nonlinear_cb_spectrum
    z = np.unique(np.concatenate([layout(t)[1] for t in TRACERS]))
    k = np.exp2(np.linspace(-16,5,43009))
    settings = LinearCAMBSettings(accuracy_boost=2.,k_per_logint=40,kmax_mpc=32.,
                                  helium_mass_fraction=.245,pivot_scalar_mpc=.05)
    for prescription in cfg['prescriptions']:
        tables = []; batches = []
        # The overlapping native-redshift batches retain the first seam copy.
        seam = len(z)//2
        for low,high in ((0,seam+1),(seam,len(z))):
            result = nonlinear_cb_spectrum(PLANCK18_BASE,k,z[low:high],settings=settings,prescription=prescription)
            tables.append(result.gg.power)
            batches.append(dict(index_bounds=[low,high],provider=json.loads(result.provenance_json)))
        power = np.concatenate((tables[0],tables[1][1:]),axis=0)
        save_product(Path(cfg['output'])/'matter'/(prescription+'.npz'),dict(k=k,redshift=z,power=power),
            dict(signature=run['signature'],prescription=prescription,convention=result.gg.convention,
                 batches=batches,seam_copy='first'),cfg['limits']['max_product_bytes'])


def calibrate(cfg,run):
    from .reference_measurement_io import read_reference_measurement_run
    from .reference_auto_estimator import reconstruct_reference_auto_jackknife
    from .reference_matter_grid import evaluate_reference_matter_grid, map_radial_bins_to_saved_slices
    from .reference_calibration import calibrate_reference_bin
    from .reference_apparent_kernel import convert_reference_apparent_kernel
    from .covariance import delete_one_covariance
    from .theory import PowerSpectrumTable
    mapping,region_sha = region_map(cfg)
    root = Path(cfg['output']); lim = cfg['limits']; summaries = {}
    for prescription in cfg['prescriptions']:
        spectra,record = load_product(root/'matter'/(prescription+'.npz'),lim['max_product_bytes'])
        if record['metadata']['signature']!=run['signature']:
            raise ValueError('matter product belongs to a different run')
        values=[]; flags=[]; components=[]; ids=None
        for tracer in TRACERS:
            edges,z,chi,op = layout(tracer)
            archive = read_reference_measurement_run(root/'measurement'/(tracer+'.npz'),max_bytes=lim['max_product_bytes'])
            source = archive.metadata['provenance']
            if (source['config_hash']!=run['signature'] or source['population_id']!=cfg['population_id']
                    or source['common_regions_sha256']!=region_sha):
                raise ValueError('measurement belongs to a different run')
            if archive.metadata['cell_regions']!=[list(v) for v in sorted(mapping.items())]:
                raise ValueError('measurement region map differs')
            ids = np.asarray(archive.metadata['region_ids']); nr=len(ids)+1
            active = np.any(op.omega>0,axis=2)
            geometries = [auto_geometry(tracer,b) for b in range(op.n_output_bins)]
            for b,(_,_,_,_,aop) in enumerate(geometries):
                columns = np.flatnonzero(np.any(aop.omega[0]>0,axis=1))
                active[np.ix_(np.arange(5*b,5*b+5),columns)] = True
            indices = np.searchsorted(spectra['redshift'],z)
            if not np.array_equal(spectra['redshift'][indices],z):
                raise ValueError('native redshifts absent from matter spectra')
            table = PowerSpectrumTable(spectra['k'],z,spectra['power'][indices],('cb','cb'),
                                       UNITS['k'],UNITS['power'],record['metadata']['convention'])
            grid = evaluate_reference_matter_grid(table,chi,BACKGROUND.H_over_c(z),THETA,active,
                k_indices=np.arange(len(table.k)),max_work=lim['max_matter_work'],max_workspace_bytes=lim['max_memory_bytes'])
            biases=np.full((nr,op.n_output_bins),np.nan); support=np.zeros_like(biases,dtype=bool)
            for b,(_,_,_,radial,aop) in enumerate(geometries):
                autos = tuple(reconstruct_reference_auto_jackknife(
                    *(read_count(cfg,run,True,(tracer,b,cap,kind)) for kind in ('DD','DR','RR')),
                    cell_regions=mapping,pair_weighting='match',max_result_bytes=lim['max_result_bytes']) for cap in CAPS)
                columns = np.flatnonzero(np.any(aop.omega[0]>0,axis=1))
                mapped = map_radial_bins_to_saved_slices(radial,edges[5*b:5*b+6],max_result_bytes=lim['max_result_bytes'])+5*b
                model = grid.value[np.ix_(mapped,columns)]
                result = calibrate_reference_bin(autos,aop,(model,model),radial_edges=radial,
                    histogram_policy='distinct_pairs',data_kind='measured',max_result_bytes=lim['max_result_bytes'])
                if result.fit is not None:
                    biases[:,b]=result.fit.bias; support[:,b]=result.fit.supported
            native = np.where(np.any(op.omega>0,axis=2),grid.value,0.)
            prov = dict(measurement_archive_sha256=archive.digest,
                native_bin_averaging=dict(method='full_native_flat_sky_uniform_area_annulus',approximation=APPROXIMATIONS['angular']),
                matter=dict(prescription=prescription,source=record['array_sha256']),
                reference_calibration=dict(source=run['signature'],treatment='refit_each_common_region',
                    cap_policy='common_NGC_SGC',slice_policy='constant_within_output_bin',pair_weighting='match',
                    cell_regions_sha256=digest(archive.metadata['cell_regions'])))
            kernel = convert_reference_apparent_kernel(archive,op,np.broadcast_to(native,(nr,*native.shape)).copy(),
                biases,support,response_region_ids=ids,reference_region_ids=ids,response_provenance=prov,
                max_result_bytes=lim['max_result_bytes'])
            arrays = dict(amplitude=kernel.amplitude,supported=kernel.supported,reference_bias=biases,
                          reference_bias_supported=support,region_ids=ids,output_z_edges=z_edges(tracer,.05))
            save_product(root/'kernels'/tracer/(prescription+'.npz'),arrays,
                dict(signature=run['signature'],provenance_json=kernel.provenance_json),lim['max_product_bytes'])
            values.append(kernel.amplitude);flags.append(kernel.supported)
            coarse=z_edges(tracer,.05)
            components += [dict(tracer=tracer,output_bin=b,z_edges=[lo,hi]) for b,(lo,hi) in enumerate(zip(coarse[:-1],coarse[1:]))]
        value=np.concatenate(values,axis=1);supported=np.concatenate(flags,axis=1)
        selected=np.flatnonzero(np.all(supported,axis=0))
        covariance=delete_one_covariance(value[1:,selected],region_ids=ids).covariance if len(selected) else np.empty((0,0))
        case=prescription+'_refined'
        summaries[case]=save_product(root/('summary-'+case+'.npz'),
            dict(value=value,supported=supported,supported_component_indices=selected,covariance=covariance,region_ids=ids),
            dict(components=components,case=case,signature=run['signature'],approximations=APPROXIMATIONS),lim['max_product_bytes'])
    publish_manifest(root/'report.json',dict(completed=True,summaries=summaries,signature=run['signature']))


def run(config_path,stage,workers=1):
    cfg=load_config(config_path)
    if stage=='plot':
        from .plotting import plot_tracers_and_combined
        figures=plot_tracers_and_combined(cfg['output'])
        for fig,name in zip(figures[:2],('tracers.png','combined.png')):
            fig.savefig(Path(cfg['output'])/name,dpi=160)
        return
    started=time.perf_counter(); record=initialize(cfg)
    if stage=='cache':
        if cfg.get('cache_manifest'):
            raise ValueError('cache_manifest already configured; skip cache stage')
        from .cache_prepare import prepare_cache
        prepare_cache(cfg,Path(cfg['output'])/'cache',RANGES)
    elif stage=='regions': make_regions(cfg)
    elif stage in ('cross-counts','auto-counts'): counts(cfg,record,stage=='auto-counts',workers)
    elif stage=='measure': measure(cfg,record)
    elif stage=='matter': matter(cfg,record)
    elif stage=='calibrate': calibrate(cfg,record)
    else: raise ValueError('unknown stage')
    print(json.dumps(dict(stage=stage,elapsed_seconds=time.perf_counter()-started,
                          slurm_job_id=os.environ.get('SLURM_JOB_ID'))),flush=True)
