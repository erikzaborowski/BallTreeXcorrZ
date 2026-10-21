"""DP-reference apparent kernels with paired bias realizations (P3).

This integrated ratio consumes explicit unit-bias responses and reference-bias
realizations. It performs no bias fit, physical validation or downstream fit.
"""
from dataclasses import dataclass
import hashlib
from numbers import Integral
import re

import numpy as np

from .counts_io import canonical_json
from .covariance import JackknifeCovariance, delete_one_covariance
from .operator import MeasurementOperator
from .reference_measurement import COMBINATION, CAPS
from .reference_measurement_io import ReferenceMeasurementRunArchive


@dataclass(frozen=True)
class ReferenceApparentKernel:
    """Rows are full sample then the explicitly ordered common deletions.

    Covariance retains columns supported in the full sample and ALL deletions.
    Its mean is the deletion mean. Supplied bias arrays must represent one
    common NGC/SGC bias per output bin, constant over that bin's native slices.
    Refitting and paired provenance are caller declarations, not proven here.
    reference_bias_supported preserves the supplied flags; final support also
    requires finite positive bias and a nonzero projected integrated response.
    """
    amplitude: np.ndarray
    supported: np.ndarray
    native_unit_response: np.ndarray
    reference_bias: np.ndarray
    reference_bias_supported: np.ndarray
    unit_response_band: np.ndarray
    unit_response_integrated: np.ndarray
    response_band: np.ndarray
    response_integrated: np.ndarray
    predicted_band: np.ndarray
    residual_band: np.ndarray
    region_ids: np.ndarray
    covariance_component_indices: np.ndarray
    covariance: JackknifeCovariance | None
    provenance_json: str
    result_bytes_bound: int
    kernel: str = 'EFFECTIVE'
    magnification: str = 'UNCORRECTED'
    calibration: str = 'supplied_common_cap_binwise_reference_bias_realizations'
    uncertainty: str = 'includes paired reference-bias and window variation; conditional on supplied model'
    validation: str = 'no_physical_validation_or_coverage_claim; upstream_refit_is_caller_declaration'


def _freeze(a):
    a.setflags(write=False)
    return a


def _identity(archive, operator, response_region_ids, reference_region_ids):
    if not isinstance(archive, ReferenceMeasurementRunArchive):
        raise TypeError('archive must be a validated ReferenceMeasurementRunArchive')
    if not isinstance(operator, MeasurementOperator):
        raise TypeError('operator must be a MeasurementOperator')
    m, a = archive.metadata, archive.arrays
    if (m.get('schema') != 'bxz.reference-measurement-run.v1'
            or m.get('quantity') != 'CROSS_CORRELATION' or m.get('stage') != 'MEASUREMENT'
            or m.get('estimator') != 'DP-reference' or m.get('combination') != COMBINATION
            or m.get('cap_order') != list(CAPS)
            or m.get('normalization') != 'retained_reference_endpoint_totals_per_cap_per_native_slice'
            or m.get('data_kind') not in ('synthetic','measured')
            or m.get('pair_weighting') not in ('literal','match')
            or not isinstance(archive.digest,str) or re.fullmatch('[0-9a-f]{64}',archive.digest) is None):
        raise ValueError('expected a validated DP-reference archive with declared cap normalization')
    regions = m['region_ids']
    if (not isinstance(regions,list) or len(regions)<2
            or any(type(r) is not int or r<0 or r>np.iinfo(np.int64).max for r in regions)
            or regions != sorted(set(regions))):
        raise ValueError('archive region IDs must be sorted distinct nonnegative integers')
    for supplied in (response_region_ids,reference_region_ids):
        ids = np.asarray(supplied)
        if (ids.shape != (len(regions),) or ids.dtype.kind not in 'iu'
                or not np.array_equal(ids,regions)):
            raise ValueError('response, reference-bias and measurement region ordering must agree')
    if [r.get('omitted_region') for r in m['realizations']] != [None,*regions]:
        raise ValueError('measurement realization ordering differs')
    if m['n_output_bins'] != operator.n_output_bins:
        raise ValueError('operator output-bin count differs from archive')
    for name in ('theta_edges','r_edges','output_bin','omega','delta_log_r'):
        if not np.array_equal(a[name],getattr(operator,name)):
            raise ValueError(f'operator {name} differs from archive')
    for name in ('retained_log_width','accessible_bounds','included','band_support'):
        if not np.array_equal(a['geometry_'+name],getattr(operator.geometry,name)):
            raise ValueError(f'operator geometry {name} differs from archive')
    if (m['geometry']['nominal_log_width'] != operator.geometry.nominal_log_width
            or tuple(m['geometry']['exclusion_reason']) != operator.geometry.exclusion_reason):
        raise ValueError('operator geometry declarations differ from archive')
    nr, (ns,nt,nj), nb = len(regions)+1, operator.omega.shape, operator.n_output_bins
    shapes = dict(native_rd_by_cap=(nr,ns,2,nt),weight_data_s=(nr,ns,2),
        weight_random_s=(nr,ns,2),band_value=(nr,nb,nj),band_supported=(nr,nb,nj),
        integrated_value=(nr,nb),integrated_supported=(nr,nb),random_support=(nr,nb,nj),
        slice_window=(nr,nb,nj,ns))
    for name,shape in shapes.items():
        x = a[name]
        if (not isinstance(x,np.ndarray) or x.shape != shape
                or (x.dtype.kind != 'b' if name.endswith('_supported') else x.dtype.kind not in 'iuf')):
            raise ValueError('measurement shapes/types differ from operator/regions')
    for name in ('native_rd_by_cap','random_support','weight_data_s','weight_random_s'):
        x=a[name]
        if not np.isfinite(x).all() or np.any(x<0) or (name.startswith('weight_') and np.any(x==0)):
            raise ValueError('reference count/support/endpoint arrays must be finite nonnegative with positive totals')
    if not np.array_equal(a['integrated_supported'],np.all(a['band_supported'],axis=2)):
        raise ValueError('measurement band and integrated support differ')
    for prefix in ('band','integrated'):
        values,support=a[prefix+'_value'],a[prefix+'_supported']
        if not np.isfinite(values[support]).all() or not np.isnan(values[~support]).all():
            raise ValueError('measurement values must be finite on support and NaN elsewhere')
    return regions,nr,ns,nt,nb,nj


def _provenance(archive, value):
    if not isinstance(value,dict) or value.get('measurement_archive_sha256') != archive.digest:
        raise ValueError('response provenance must identify the measurement archive digest')
    required = {'native_bin_averaging':('method','approximation'),
                'matter':('prescription','source'),
                'reference_calibration':('source','treatment','cap_policy','slice_policy','pair_weighting','cell_regions_sha256')}
    for section,keys in required.items():
        item=value.get(section)
        if (not isinstance(item,dict)
                or any(not isinstance(item.get(k),str) or not item[k].strip() for k in keys)):
            raise ValueError(f'provenance requires explicit {section} declarations')
    reference=value['reference_calibration']
    expected = dict(treatment='refit_each_common_region',cap_policy='common_NGC_SGC',
        slice_policy='constant_within_output_bin',pair_weighting=archive.metadata['pair_weighting'],
        cell_regions_sha256=hashlib.sha256(canonical_json(archive.metadata['cell_regions']).encode()).hexdigest())
    if any(reference[k] != v for k,v in expected.items()):
        raise ValueError('reference calibration declarations differ from shared-cap/bin/common-region requirements')
    return canonical_json(dict(measurement_archive_sha256=archive.digest,
        data_kind=archive.metadata['data_kind'],response=value,
        response_definition='alpha_scaled_RD_mean_of_unit_reference_bias_H_over_c_times_wp_cbcb',
        bias_assignment='one_common_NGC_SGC_bias_per_output_bin_constant_over_its_native_slices',
        amplitude_unit='redshift^-1',quantity='apparent_effective_kernel',
        method='integrated_ratio; no_likelihood_fit',
        uncertainty='includes paired reference-bias and window variation; conditional on supplied model',
        upstream_refit='caller_declares_refit_each_common_region; not_independently_verified',
        covariance='delete_one_of_paired_cross_and_reference_bias_realizations; no_independent_bias_error_addition',
        provenance_acceptance='declarations_only; no_scientific_acceptance_inferred'))


def convert_reference_apparent_kernel(archive, operator, native_unit_response,
        reference_bias, reference_bias_supported, *, response_region_ids,
        reference_region_ids, response_provenance, max_result_bytes):
    """Project each DP-reference window, then divide by its supplied bias response.

    native_unit_response has exact (realization,native_slice,angular_bin) shape;
    bias and Boolean bias support have exact (realization,output_bin) shape.
    No full-sample row, cap dimension or missing realization is broadcast.
    The native response is the caller's full-bin U-weighted unit-bias mean.
    U=sum_cap[(W_data/W_random)*RD] is rebuilt in every realization. Multiplying
    its projected output-bin response by that bin's bias is valid only under
    the explicitly declared common-cap, piecewise-constant bias assignment.

    The provenance records the archive digest, full-bin averaging and matter
    declarations, and reference_calibration source, treatment, cap/slice policy,
    pair_weighting and SHA256 of canonical_json(archive.metadata['cell_regions']).
    Its declarations do not establish the upstream model or refits physically.

    max_result_bytes bounds returned numeric arrays plus provenance, using the
    full nominal covariance dimension, before owned arrays are allocated. It
    excludes caller arrays, projection temporaries and covariance workspaces;
    it is not a process-memory guarantee. No precision matrix is calculated.
    """
    if (isinstance(max_result_bytes,(bool,np.bool_)) or not isinstance(max_result_bytes,Integral)
            or max_result_bytes<=0):
        raise ValueError('max_result_bytes must be an explicit positive integer')
    regions,nr,ns,nt,nb,nj = _identity(archive,operator,response_region_ids,reference_region_ids)
    for name,x,shape in (('native_unit_response',native_unit_response,(nr,ns,nt)),
                         ('reference_bias',reference_bias,(nr,nb))):
        if not isinstance(x,np.ndarray) or x.dtype.kind not in 'iuf' or x.shape != shape:
            raise ValueError(f'{name} must be a real NumPy array with exact realization shape')
    if (not isinstance(reference_bias_supported,np.ndarray)
            or reference_bias_supported.dtype.kind!='b' or reference_bias_supported.shape!=(nr,nb)):
        raise ValueError('reference_bias_supported must be Boolean with exact realization/output-bin shape')
    if not np.isfinite(native_unit_response).all():
        raise ValueError('native unit response must be finite')
    if (not np.isfinite(reference_bias[reference_bias_supported]).all()
            or np.isinf(reference_bias).any()):
        raise ValueError('bias must be finite on supplied support; unsupported NaN is allowed')
    provenance=_provenance(archive,response_provenance)
    bound=(8*nr*ns*nt+34*nr*nb+32*nr*nb*nj+16*len(regions)+16*nb+8*nb*nb+4*len(provenance))
    if bound>max_result_bytes:
        raise ValueError('reference apparent-kernel result exceeds max_result_bytes')
    a=archive.arrays
    try:
        with np.errstate(over='raise',invalid='raise',divide='raise'):
            native=np.array(native_unit_response,dtype=float,copy=True)
            bias=np.array(reference_bias,dtype=float,copy=True)
            bias_support=np.array(reference_bias_supported,dtype=bool,copy=True)
            amplitude=np.full((nr,nb),np.nan)
            supported=np.zeros((nr,nb),dtype=bool)
            unit_band=np.full((nr,nb,nj),np.nan)
            unit_integrated=np.full((nr,nb),np.nan)
            response_band=np.full_like(unit_band,np.nan)
            response_integrated=np.full_like(unit_integrated,np.nan)
            predicted=np.full_like(unit_band,np.nan)
            residual=np.full_like(unit_band,np.nan)
            for r in range(nr):
                wd,wr=a['weight_data_s'][r],a['weight_random_s'][r]
                if not np.isfinite(wd).all() or not np.isfinite(wr).all() or np.any(wd<=0) or np.any(wr<=0):
                    raise ValueError('reference endpoint totals must be finite positive')
                u=((wd/wr)[:,:,None]*a['native_rd_by_cap'][r]).sum(axis=1)
                bands=operator.project_theory(native[r],u)
                integrated=operator.integrate(bands)
                if (not np.array_equal(bands.supported,a['band_supported'][r])
                        or not np.array_equal(integrated.supported,a['integrated_supported'][r])
                        or not np.array_equal(bands.random_support,a['random_support'][r])
                        or not np.array_equal(bands.slice_window,a['slice_window'][r],equal_nan=True)):
                    raise ValueError('projected U support/windows differ from measurement archive')
                if (not np.isfinite(bands.value[bands.supported]).all()
                        or not np.isfinite(integrated.value[integrated.supported]).all()):
                    raise ValueError('unit response must remain finite on support')
                unit_band[r],unit_integrated[r]=bands.value,integrated.value
                good_bias=bias_support[r] & np.isfinite(bias[r]) & (bias[r]>0)
                response_band[r,good_bias]=bands.value[good_bias]*bias[r,good_bias,None]
                response_integrated[r,good_bias]=integrated.value[good_bias]*bias[r,good_bias]
                keep=integrated.supported & good_bias & np.isfinite(response_integrated[r]) & (response_integrated[r]!=0)
                supported[r]=keep
                amplitude[r,keep]=a['integrated_value'][r,keep]/response_integrated[r,keep]
                predicted[r,keep]=amplitude[r,keep,None]*response_band[r,keep]
                residual[r,keep]=a['band_value'][r,keep]-predicted[r,keep]
                if any(not np.isfinite(x).all() for x in (amplitude[r,keep],predicted[r,keep],residual[r,keep])):
                    raise ValueError('kernel conversion must remain finite on support')
    except FloatingPointError as exc:
        raise ValueError('reference kernel arithmetic exceeded finite float64 range') from exc
    indices=np.flatnonzero(np.all(supported,axis=0))
    ids=np.array(regions,dtype=np.int64)
    covariance=delete_one_covariance(amplitude[1:,indices],region_ids=ids) if len(indices) else None
    frozen=map(_freeze,(amplitude,supported,native,bias,bias_support,unit_band,unit_integrated,
        response_band,response_integrated,predicted,residual,ids,indices))
    return ReferenceApparentKernel(*frozen,covariance,provenance,bound)
