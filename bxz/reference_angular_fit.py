"""Fixed-metric fits of reference angular correlations to b_s² M (P3 §4.6).

This is the paper-style scalar reference calibration only. It does not load
data, construct a matter template/covariance, fit photometric bandpowers, or
change the synthetic-only guard of reference_fit.py. Nonpositive amplitudes
are saved as diagnostics but do not define a usable positive reference bias.
"""
from dataclasses import dataclass
from numbers import Integral

import numpy as np


@dataclass(frozen=True)
class ReferenceAngularBiasFit:
    bias_squared: np.ndarray
    bias: np.ndarray
    supported: np.ndarray
    identifiable: np.ndarray
    prediction: np.ndarray
    residual: np.ndarray
    quadratic_residual: np.ndarray
    template_information: np.ndarray
    region_ids: np.ndarray
    covariance: np.ndarray
    covariance_condition: float
    data_kind: str
    quantity: str = 'reference_angular_correlation'
    model: str = 'b_s_squared_times_supplied_template'
    uncertainty: str = 'refit_each_common_region_realization; no_coverage_claim'
    metric: str = 'same_supplied_covariance_for_every_realization; no_regularization'


def fit_reference_angular_bias(values, templates, covariance, *, region_ids,
                               data_kind, quantity, max_result_bytes):
    """Fit full sample and each deletion with the same positive-definite metric.

    Values and templates have explicit shape (1+nregions, ncomponents); no
    full-sample template is broadcast. Components must already be selected
    consistently and supported in every row. Covariance is supplied for those
    same components and is fixed across all refits. The caller is responsible
    for the reference population, physical model and common-region identity.

    a_r=(M_r^T C^-1 w_r)/(M_r^T C^-1 M_r), b_r=sqrt(a_r) when a_r>0.
    A zero template is unidentifiable. Negative/zero fitted a is preserved;
    bias remains NaN and supported=False. There is no positivity boundary,
    clipping, fallback bias, covariance inverse or precision correction.
    Quadratic residuals are diagnostics, not calibrated chi-square p-values.

    max_result_bytes admits owned numerical results and working copies, not
    caller arrays or Python/BLAS overhead. No measured inputs are accessed by
    this pure helper. Measured use is restricted to reference calibration,
    separately authorized by the owner; it does not authorize downstream fits.
    """
    if quantity != 'reference_angular_correlation' or data_kind not in ('synthetic','measured'):
        raise ValueError('explicit reference angular calibration quantity/data_kind required')
    if isinstance(max_result_bytes,(bool,np.bool_)) or not isinstance(max_result_bytes,Integral) or max_result_bytes<=0:
        raise ValueError('explicit positive max_result_bytes required')
    regions=np.asarray(region_ids)
    if (regions.ndim!=1 or len(regions)<2 or regions.dtype.kind not in 'iu'
            or np.any(regions<0) or np.any(regions[1:]<=regions[:-1])):
        raise ValueError('sorted distinct common region IDs required')
    for value in (values,templates,covariance):
        if not isinstance(value,np.ndarray) or value.dtype.kind!='f':
            raise ValueError('explicit real floating-point arrays required')
    nr=len(regions)+1
    if values.ndim!=2 or values.shape[0]!=nr or values.shape[1]<1 or templates.shape!=values.shape:
        raise ValueError('values/templates must match full plus every region deletion')
    nc=values.shape[1]
    if covariance.shape!=(nc,nc):
        raise ValueError('covariance must match selected reference components')
    bound=80*nr*nc+48*nc*nc+96*nr+16*len(regions)+65536
    if bound>max_result_bytes:
        raise ValueError('reference fit exceeds max_result_bytes')
    if not all(np.isfinite(x).all() for x in (values,templates,covariance)):
        raise ValueError('select jointly supported finite reference components before fitting')
    if not np.array_equal(covariance,covariance.T):
        raise ValueError('covariance must be explicitly symmetric')
    cov=np.array(covariance,dtype=float,copy=True)
    try:
        chol=np.linalg.cholesky(cov)
    except np.linalg.LinAlgError as exc:
        raise ValueError('reference covariance must be positive definite; no automatic regularization') from exc
    eigenvalues=np.linalg.eigvalsh(cov)
    if eigenvalues[0]<=0:
        raise ValueError('reference covariance is not numerically positive definite')
    condition=float(eigenvalues[-1]/eigenvalues[0])
    amplitude=np.full(nr,np.nan);bias=np.full(nr,np.nan)
    information=np.zeros(nr);quadratic=np.full(nr,np.nan)
    identifiable=np.zeros(nr,dtype=bool);supported=identifiable.copy()
    predicted=np.full((nr,nc),np.nan);residual=np.full_like(predicted,np.nan)
    with np.errstate(over='raise',invalid='raise',divide='raise'):
        white_values=np.linalg.solve(chol,np.asarray(values,dtype=float).T).T
        white_templates=np.linalg.solve(chol,np.asarray(templates,dtype=float).T).T
        for row in range(nr):
            model=white_templates[row]
            # Scale the model to avoid squaring a very small/large template.
            scale=np.max(np.abs(model))
            if scale==0:
                continue
            unit=model/scale
            denominator=float(unit@unit)
            amplitude[row]=float(unit@white_values[row])/denominator/scale
            information[row]=denominator*scale*scale
            identifiable[row]=True
            predicted[row]=amplitude[row]*templates[row]
            residual[row]=values[row]-predicted[row]
            white_residual=white_values[row]-amplitude[row]*model
            quadratic[row]=white_residual@white_residual
            if amplitude[row]>0:
                bias[row]=np.sqrt(amplitude[row]);supported[row]=True
    regions=regions.copy()
    for a in (amplitude,bias,information,quadratic,identifiable,supported,predicted,residual,cov,regions):
        a.setflags(write=False)
    return ReferenceAngularBiasFit(amplitude,bias,supported,identifiable,predicted,residual,
        quadratic,information,regions,cov,condition,data_kind)
