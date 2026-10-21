"""Common-cap reference angular bias calibration for one bin (P3 §4.6).

This pure orchestration joins existing LS realizations, radial templates and
fixed-metric scalar bias fits. It does not pool caps, regularize a covariance,
load products, or fit downstream photometric bandpowers.
"""
from dataclasses import dataclass
from numbers import Integral

import numpy as np

from .counts_io import canonical_json
from .covariance import JackknifeCovariance, delete_one_covariance
from .operator import MeasurementOperator
from .reference_auto_estimator import ReferenceAutoJackknife
from .reference_angular_fit import ReferenceAngularBiasFit, fit_reference_angular_bias
from .reference_radial_template import ReferenceRadialTemplate, project_reference_radial_template


@dataclass(frozen=True)
class ReferenceBinCalibration:
    observations: np.ndarray  # realization, cap(NGC,SGC), scale band
    templates: np.ndarray
    supported: np.ndarray
    observation_supported: np.ndarray
    template_supported: np.ndarray
    component_indices: np.ndarray  # shared supported flattened cap-major columns
    covariance: JackknifeCovariance | None
    fit: ReferenceAngularBiasFit | None
    status: str
    radial_templates: tuple[ReferenceRadialTemplate, ReferenceRadialTemplate]
    active_angular_indices: np.ndarray
    radial_edges: np.ndarray
    region_ids: np.ndarray
    cell_regions: tuple
    pair_weighting: str
    source_run_hashes: tuple
    data_kind: str
    result_bytes_bound: int
    cap_order: tuple = ('NGC','SGC')
    bias_policy: str = 'one_common_NGC_SGC_bias_for_this_output_bin'
    metric_policy: str = 'fixed_joint_deletion_covariance; no_regularization_or_component_selection_by_eigenmodes'
    validation: str = 'supplied_model_calibration_only; no_physical_validation_or_coverage_claim'


def _readonly(a):
    a.setflags(write=False)
    return a


def _validate(autos,operator,matter_responses,radial_edges,data_kind):
    if not isinstance(autos,tuple) or len(autos)!=2 or any(not isinstance(a,ReferenceAutoJackknife) for a in autos):
        raise TypeError('explicit (NGC,SGC) tuple of validated reference autos required')
    if not isinstance(operator,MeasurementOperator) or operator.omega.shape[0]!=1 or operator.n_output_bins!=1 or not np.array_equal(operator.output_bin,[0]):
        raise ValueError('one coarse slice and one output-bin operator required')
    if data_kind not in ('synthetic','measured'):
        raise ValueError('explicit synthetic/measured data_kind required')
    if not isinstance(matter_responses,tuple) or len(matter_responses)!=2:
        raise ValueError('explicit two-cap matter response tuple required')
    base=autos[0]
    regions=np.asarray(base.region_ids)
    if (regions.ndim!=1 or len(regions)<2 or regions.dtype.kind not in 'iu'
            or np.any(regions<0) or np.any(regions[1:]<=regions[:-1])
            or base.omitted_regions!=(None,*base.region_ids)):
        raise ValueError('sorted common regions/full-deletion ordering required')
    nr,nt,nj=len(regions)+1,len(operator.theta_edges)-1,len(operator.delta_log_r)
    active=np.flatnonzero(np.any(operator.omega[0]>0,axis=1))
    if len(active)==0:
        raise ValueError('operator must retain at least one native angular column')
    edges=np.asarray(radial_edges)
    if (edges.ndim!=1 or len(edges)<2 or edges.dtype.kind not in 'iuf'
            or not np.isfinite(edges).all() or np.any(np.diff(edges)<=0)
            or not np.array_equal(edges[[0,-1]],base.radial_edges[[0,-1]])):
        raise ValueError('radial coarsening must preserve the exact saved slice')
    edge_index=np.searchsorted(base.radial_edges,edges)
    if np.any(edge_index>=len(base.radial_edges)) or not np.array_equal(base.radial_edges[edge_index],edges):
        raise ValueError('radial coarsening edges must come from the saved grid')
    baseline_science=None;tracer=None
    for cap,auto,model in zip(('NGC','SGC'),autos,matter_responses):
        if (auto.region_ids!=base.region_ids or auto.omitted_regions!=base.omitted_regions
                or auto.cell_regions!=base.cell_regions or auto.pair_weighting!=base.pair_weighting
                or auto.cross_region_weight!=base.cross_region_weight
                or not np.array_equal(auto.theta_edges,operator.theta_edges)
                or not np.array_equal(auto.radial_edges,base.radial_edges)):
            raise ValueError('cap angular/radial grids or common-region map/mode differ')
        if auto.pair_weighting not in ('literal','match'):
            raise ValueError('literal/match auto weighting required')
        gamma=1-len(regions)/(2+np.sqrt(2)*(len(regions)-1)) if auto.pair_weighting=='match' else None
        if auto.cross_region_weight!=gamma:
            raise ValueError('auto match coefficient differs from shared region count')
        if (not isinstance(model,np.ndarray) or model.dtype.kind not in 'iuf'
                or model.shape!=(len(edges)-1,len(active)) or not np.isfinite(model).all()):
            raise ValueError('finite explicit matter response must match radial/active angular grid')
        if (auto.value.shape!=(nr,nt) or auto.supported.shape!=(nr,nt)
                or auto.supported.dtype.kind!='b' or auto.value.dtype.kind!='f'
                or auto.weighted_counts.shape!=(nr,3,nt)
                or not np.isfinite(auto.weighted_counts).all() or np.any(auto.weighted_counts<0)
                or not np.isfinite(auto.value[auto.supported]).all()
                or not np.isnan(auto.value[~auto.supported]).all()):
            raise ValueError('invalid native auto values/support/count arrays')
        if len(auto.inputs)!=3:
            raise ValueError('DD/DR/RR source inputs required')
        for count_type,run in zip(('DD','DR','RR'),auto.inputs):
            meta=run.metadata
            parts=meta['left_group'].rsplit('_',2)
            if len(parts)!=3 or parts[1]!=cap or parts[0] not in ('BGS_BRIGHT','LRG','ELG_LOPnotqso','QSO'):
                raise ValueError('autos must be explicitly ordered NGC then SGC of one tracer')
            if tracer is None:tracer=parts[0]
            kinds={'DD':('data','data'),'DR':('data','randoms'),'RR':('randoms','randoms')}[count_type]
            if (parts[0]!=tracer or meta['count_type']!=count_type
                    or meta['left_group']!=f'{tracer}_{cap}_{kinds[0]}'
                    or meta['right_group']!=f'{tracer}_{cap}_{kinds[1]}'):
                raise ValueError('source count types/tracer/cap groups differ')
            science=canonical_json({key:meta[key] for key in ('slice_id','measurement_context',
                'provenance','cache_metadata_sha256','software','backend','units')})
            if baseline_science is None:baseline_science=science
            if science!=baseline_science:
                raise ValueError('cap/slice population or source provenance differs')
            declared=meta['provenance'].get('config',{}).get('data_kind')
            if declared is not None and declared!=data_kind:
                raise ValueError('data_kind differs from source declaration')
            if not np.array_equal(run.counts.edges,operator.theta_edges) or not np.array_equal(run.radial_edges,base.radial_edges):
                raise ValueError('source native grids differ from reconstructed autos')
    return nr,nt,nj,active,edges


def _project(values,support,rr,operator):
    """Masked omega*RR contraction without filling active missing values."""
    positive=np.any(operator.omega[0]>0,axis=1) & (rr>0)
    weighted=np.zeros_like(rr)
    np.multiply(values,rr,out=weighted,where=positive & support)
    numerator=operator._aggregate(weighted[None,:])[0][0]
    denominator=operator._aggregate(rr[None,:])[0][0]
    missing=np.any((operator.omega[0]>0) & (positive & ~support)[:,None],axis=0)
    keep=(denominator>0) & ~missing
    projected=np.full(len(denominator),np.nan)
    np.divide(numerator,denominator,out=projected,where=keep)
    if not np.isfinite(projected[keep]).all():
        raise ArithmeticError('reference cap/band projection exceeded finite range')
    return projected,keep


def calibrate_reference_bin(autos,operator,matter_responses,*,radial_edges,
        histogram_policy='distinct_pairs',data_kind,max_result_bytes):
    """Project separate caps and fit one common b² with a fixed joint metric.

    Active angular indices are those with positive operator omega. Zero-RR
    columns contribute no weight; a missing value with positive omega AND RR
    makes its affected scale band unsupported. No active missing value is
    replaced by a finite value. Each cap keeps its own actual RR windows.

    The caller must construct and record the operator distance and hard cuts
    from the intended stored geometry. This helper checks native angular grids
    and the one-bin structure; MeasurementOperator does not retain its input
    chi/redshift, so agreement with source context cannot be inferred here.

    Shared components are the cap-major flattened bands supported in full and
    every common deletion. Their deletion covariance is reused in every fit.
    Empty support, numerical rank deficiency or non-positive-definite metric
    returns fit=None with explicit status; no extra components are discarded.

    The conservative bound includes both radial helper budgets, fit/covariance
    workspaces and owned outputs, excluding caller arrays and Python/BLAS
    overhead. Radial outputs are retained directly, without further copies or
    pair cubes. No measured files are loaded by this pure entry point.
    """
    if isinstance(max_result_bytes,(bool,np.bool_)) or not isinstance(max_result_bytes,Integral) or max_result_bytes<=0:
        raise ValueError('explicit positive max_result_bytes required')
    if histogram_policy not in ('distinct_pairs','raw_squared'):
        raise ValueError('explicit supported histogram_policy required')
    nr,nt,nj,active,edges=_validate(autos,operator,matter_responses,radial_edges,data_kind)
    nc=2*nj;nz=len(edges)-1;na=len(active)
    radial_bounds=[]
    for auto in autos:
        rr=auto.inputs[2];npair=len(rr.counts.patch_pairs);ncell=len(rr.counts.weight_sums_left)
        radial_bounds.append(16*npair*na+64*npair+64*ncell*(nz+1)+64*nr*na+32*nz*na+65536)
    fit_bound=80*nr*nc+48*nc*nc+96*nr+16*(nr-1)+65536
    bound=sum(radial_bounds)+fit_bound+128*nr*nc+128*nc*nc+64*nr+64*nt+16*(nz+1)+65536
    if bound>max_result_bytes:
        raise ValueError('reference-bin calibration exceeds max_result_bytes')
    observations=np.full((nr,2,nj),np.nan);templates=np.full_like(observations,np.nan)
    observed_support=np.zeros((nr,2,nj),dtype=bool);model_support=np.zeros_like(observed_support)
    radial=[]
    with np.errstate(over='raise',invalid='raise',divide='raise'):
        for c,(auto,model) in enumerate(zip(autos,matter_responses)):
            projected=project_reference_radial_template(auto,model,angular_indices=active,radial_edges=edges,
                histogram_policy=histogram_policy,max_result_bytes=radial_bounds[c])
            radial.append(projected)
            for r in range(nr):
                rr=auto.weighted_counts[r,2]
                observations[r,c],observed_support[r,c]=_project(auto.value[r],auto.supported[r],rr,operator)
                native=np.zeros(nt);native_support=np.zeros(nt,dtype=bool)
                native[active]=projected.value[r];native_support[active]=projected.supported[r]
                templates[r,c],model_support[r,c]=_project(native,native_support,rr,operator)
    supported=observed_support & model_support
    indices=np.flatnonzero(np.all(supported,axis=0).reshape(-1))
    covariance=fit=None;status='no_joint_support'
    ids=np.asarray(autos[0].region_ids,dtype=np.int64).copy()
    if len(indices):
        values=observations.reshape(nr,nc)[:,indices]
        models=templates.reshape(nr,nc)[:,indices]
        covariance=delete_one_covariance(values[1:],region_ids=ids)
        if covariance.diagnostics.numerical_rank<len(indices):
            status='singular_metric'
        else:
            try:
                np.linalg.cholesky(covariance.covariance)
                positive=np.linalg.eigvalsh(covariance.covariance)[0]>0
            except np.linalg.LinAlgError:
                positive=False
            if not positive:
                status='nonpositive_definite_metric'
            else:
                fit=fit_reference_angular_bias(values,models,covariance.covariance,
                    region_ids=ids,data_kind=data_kind,quantity='reference_angular_correlation',
                    max_result_bytes=fit_bound)
                status='fitted'
    for a in (observations,templates,supported,observed_support,model_support,indices,active,edges,ids):
        # edges may alias caller input: detach instead of changing its flags.
        if a is not edges:a.setflags(write=False)
    saved_edges=_readonly(np.array(edges,dtype=float,copy=True))
    return ReferenceBinCalibration(observations,templates,supported,observed_support,model_support,
        indices,covariance,fit,status,tuple(radial),active,saved_edges,ids,autos[0].cell_regions,
        autos[0].pair_weighting,tuple(auto.run_hashes for auto in autos),data_kind,bound)
