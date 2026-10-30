"""Save total-matter Takahashi Halofit P(k,z) for fit_spectroscopic_bias.py.

Runs CAMB: execute on an approved compute allocation, not the login node.
CAMB is an optional dependency only for this producer, not for fitting.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--zmin', type=float, required=True)
    parser.add_argument('--zmax', type=float, required=True)
    parser.add_argument('--nz', type=int, default=9)
    parser.add_argument('--kmin', type=float, default=1e-4, help='h/Mpc; must lie inside native CAMB coverage')
    parser.add_argument('--kmax', type=float, default=30., help='h/Mpc; test convergence')
    parser.add_argument('--nk', type=int, default=16385, help='Log-spaced output k nodes')
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    if not 0 <= args.zmin < args.zmax or not 4 <= args.nz <= 100:
        raise ValueError('Require 0 <= zmin < zmax and 4 <= nz <= 100')
    if not 0 < args.kmin < args.kmax or args.nk < 32:
        raise ValueError('Invalid power-grid k range or size')
    if os.environ.get('OMP_NUM_THREADS') != '1':
        raise ValueError('Set OMP_NUM_THREADS=1 for this serial producer')
    import camb
    from astropy.cosmology import Planck15

    h = Planck15.h
    pars = camb.CAMBparams()
    # Astropy's Om0 excludes the separately tracked massive-neutrino density.
    # Keep the background of the legacy Planck15 geometry; declare As and ns
    # explicitly because Astropy background cosmologies do not contain them.
    pars.set_cosmology(H0=Planck15.H0.value, ombh2=Planck15.Ob0*h*h,
                       omch2=(Planck15.Om0-Planck15.Ob0)*h*h,
                       omk=0., mnu=.06, num_massive_neutrinos=1,
                       nnu=Planck15.Neff, TCMB=Planck15.Tcmb0.value, tau=.066)
    pars.InitPower.set_params(As=2.142e-9, ns=.9667)
    pars.WantCls = False
    z = np.linspace(args.zmin, args.zmax, args.nz)
    # CAMB's requested kmax is physical 1/Mpc, not h/Mpc.
    pars.set_matter_power(redshifts=z[::-1], kmax=args.kmax*h*1.05)
    pars.NonLinearModel.set_params(halofit_version='takahashi')
    pars.NonLinear = camb.model.NonLinear_pk
    results = camb.get_results(pars)
    interpolator, native_z, native_k = results.get_matter_power_interpolator(
        nonlinear=True, var1='delta_tot', var2='delta_tot',
        hubble_units=True, k_hunit=True, return_z_k=True, extrap_kmax=None)
    k = np.geomspace(args.kmin, args.kmax, args.nk)
    if k[0] < np.min(native_k) or k[-1] > np.max(native_k):
        raise ValueError('Requested k range is outside native CAMB coverage; no extrapolation')
    if z[0] < np.min(native_z) or z[-1] > np.max(native_z):
        raise ValueError('Requested z range is outside native CAMB coverage')
    pk = interpolator.P(z, k)
    if not np.isfinite(pk).all() or np.any(pk <= 0):
        raise ValueError('CAMB returned invalid matter power')
    # Neutrino background conventions can differ slightly between libraries.
    # Record and check geometry consistency rather than silently mixing them.
    positive_z = z[z > 0]
    distance_difference = float(np.max(np.abs(
        results.comoving_radial_distance(positive_z)/Planck15.comoving_distance(positive_z).value-1)))
    if distance_difference > 1e-3:
        raise ValueError('CAMB and legacy Planck15 distances differ by >0.1%')
    metadata = dict(cosmology='Planck15', model='halofit_takahashi', matter_field='total',
                    k_unit='h/Mpc', power_unit='(Mpc/h)^3', camb_version=camb.__version__,
                    As=2.142e-9, ns=.9667, camb_parameters=str(pars),
                    max_relative_distance_difference=distance_difference,
                    validation='Check k range, k resolution and redshift grid convergence before measured use')
    with args.output.open('xb') as stream:
        np.savez(stream, z=z, k_hmpc=k, pk_mpc_over_h3=pk,
                 metadata_json=json.dumps(metadata, allow_nan=False))
    print(f'Saved {len(z)} x {len(k)} Halofit grid to {args.output}')


if __name__ == '__main__':
    main()
