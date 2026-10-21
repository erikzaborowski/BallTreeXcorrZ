"""Approximate cell-factorized radial projection for reference autos (P3).

The finite-random histogram overlap excludes identical random objects by
default. This is a declared radial/angular separability approximation within
storage cells, not an exact survey window or a physical validation claim.
"""
from dataclasses import dataclass
from numbers import Integral

import numpy as np

from .reference_auto_estimator import ReferenceAutoJackknife


@dataclass(frozen=True)
class ReferenceRadialTemplate:
    value: np.ndarray
    supported: np.ndarray
    weighted_random_pairs: np.ndarray
    angular_indices: np.ndarray
    radial_edges: np.ndarray
    matter_response: np.ndarray
    region_ids: tuple
    source_rr_hash: str
    histogram_policy: str
    pair_weighting: str
    approximation: str = 'radial_angular_separability_within_storage_cells; finite_random_histogram_noise'
    matter_convention: str = 'supplied_H_over_c_times_projected_matter_response_per_radial_bin_and_native_annulus'


def project_reference_radial_template(auto, matter_response, *, angular_indices,
                                      radial_edges, histogram_policy='distinct_pairs',
                                      max_result_bytes):
    """Project a supplied matter response through each actual RR pair weight.

    ``auto`` is the validated full/common-region LS reconstruction. The supplied
    matter response has shape (radial bins, selected native angular bins).
    Its interpolation, matter field and annulus averaging are caller choices
    that must be declared in the run provenance. radial_edges may coarsen the
    saved grid using exact existing edges, with identical outer boundaries.

    For a!=b use B_ab,l=W_a,l W_b,l/(W_a W_b dz_l). For a=b the default is
    B_aa,l=(W_a,l²-Q_a,l)/((W_a²-Q_a)dz_l). B is already the bin-integrated
    overlap, so T_ab,i=sum_l B_ab,l G_l,i has NO extra dz factor. The result is
    sum_ab f_ab RR_ab,i T_ab,i / sum_ab f_ab RR_ab,i for every realization.
    raw_squared is an explicit no-self-subtraction sensitivity calculation.

    Zero RR rows contribute nothing; positive RR with invalid all-pair
    normalization fails. Sparse overlaps are not repaired with floors. This
    performs no RSD/integral-constraint correction or bias fit. The numerical
    memory bound excludes caller-owned inputs and Python/BLAS overhead.
    """
    if not isinstance(auto,ReferenceAutoJackknife):
        raise TypeError('validated ReferenceAutoJackknife required')
    if histogram_policy not in ('distinct_pairs','raw_squared'):
        raise ValueError('explicit histogram policy required')
    if isinstance(max_result_bytes,(bool,np.bool_)) or not isinstance(max_result_bytes,Integral) or max_result_bytes<=0:
        raise ValueError('explicit positive max_result_bytes required')
    rr=auto.inputs[2]
    columns=np.asarray(angular_indices)
    edges=np.asarray(radial_edges)
    model=np.asarray(matter_response)
    if (columns.ndim!=1 or columns.dtype.kind not in 'iu' or len(columns)==0
            or np.any(columns<0) or np.any(columns>=len(rr.counts.edges)-1)
            or np.any(columns[1:]<=columns[:-1])):
        raise ValueError('sorted unique native angular indices required')
    if (edges.ndim!=1 or len(edges)<2 or edges.dtype.kind not in 'iuf'
            or not np.isfinite(edges).all() or np.any(np.diff(edges)<=0)
            or not np.array_equal(edges[[0,-1]],rr.radial_edges[[0,-1]])):
        raise ValueError('radial coarsening must preserve the exact slice')
    index=np.searchsorted(rr.radial_edges,edges)
    if np.any(index>=len(rr.radial_edges)) or not np.array_equal(rr.radial_edges[index],edges):
        raise ValueError('coarsening edges must come from the saved radial grid')
    if (model.shape!=(len(edges)-1,len(columns)) or model.dtype.kind not in 'iuf'
            or not np.isfinite(model).all()):
        raise ValueError('finite matter response must match explicit radial/angular grid')
    cells=tuple(rr.counts.weight_sums_left)
    npair=len(rr.counts.patch_pairs);nr=len(auto.omitted_regions);nz=len(edges)-1;nt=len(columns)
    bound=16*npair*nt+64*npair+64*len(cells)*(nz+1)+64*nr*nt+32*nz*nt+65536
    if bound>max_result_bytes:
        raise ValueError('reference template exceeds max_result_bytes')
    hist=np.stack([rr.left_radial_moments[:,lo:hi].sum(axis=1) for lo,hi in zip(index[:-1],index[1:])],axis=1)
    weight=hist[:,:,1];squared_weight=hist[:,:,2]
    total=weight.sum(axis=1);total_q=squared_weight.sum(axis=1)
    location={int(c):j for j,c in enumerate(cells)}
    regions=dict(auto.cell_regions)
    labels=np.asarray([(regions[int(a)],regions[int(b)]) for a,b in rr.counts.patch_pairs],dtype=np.int64).reshape(-1,2)
    weighted_model=np.zeros((npair,nt))
    native=np.array(model,dtype=float,copy=True)
    widths=np.diff(edges)
    with np.errstate(over='raise',invalid='raise',divide='raise'):
        for j,(a,b) in enumerate(rr.counts.patch_pairs):
            counts=rr.counts.weighted[j,columns]
            if not np.any(counts>0):
                continue
            ia,ib=location[int(a)],location[int(b)]
            numerator=weight[ia]*weight[ib]
            normalization=total[ia]*total[ib]
            if a==b and histogram_policy=='distinct_pairs':
                numerator=numerator-squared_weight[ia]
                normalization-=total_q[ia]
            if normalization<=0 or np.any(numerator<0):
                raise ValueError('positive RR has invalid radial distinct-pair normalization')
            overlap=numerator/normalization/widths
            weighted_model[j]=counts*(overlap@native)
        denominators=np.array(auto.weighted_counts[:,2,:][:,columns],dtype=float,copy=True)
        supported=denominators>0
        value=np.full((nr,nt),np.nan)
        gamma=auto.cross_region_weight if auto.pair_weighting=='match' else 0.
        for j,omitted in enumerate(auto.omitted_regions):
            if omitted is None:
                summed=weighted_model.sum(axis=0)
            else:
                omitted_count=(labels[:,0]==omitted).astype(np.int8)+(labels[:,1]==omitted)
                scale=np.choose(omitted_count,(1.,gamma,0.))
                summed=np.einsum('i,ij->j',scale,weighted_model,optimize=False)
            np.divide(summed,denominators[j],out=value[j],where=supported[j])
    if not np.isfinite(value[supported]).all():
        raise ArithmeticError('reference template projection exceeded finite range')
    columns=columns.copy();edges=np.array(edges,dtype=float,copy=True)
    for a in (value,supported,denominators,columns,edges,native):
        a.setflags(write=False)
    return ReferenceRadialTemplate(value,supported,denominators,columns,edges,native,
        auto.region_ids,rr.run_hash,histogram_policy,auto.pair_weighting)
