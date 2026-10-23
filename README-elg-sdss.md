# Minimal ELG × SDSS run

`run_elg_sdss.slurm` calls the existing `cross_correlate.py`,
`error_estimator.py`, and `get_b_dndz.py`, in that order, using `elgcmb_legacy`.
It does not use the implementations from the larger DR2 update.

Selection:

- Photometric ELGs and randoms: `elg_mask == 0`, unit weights. No `elglop`,
  `elgvlo`, `PHOTSYS`, photometric-redshift, or additional sky cuts.
- Reference: the local Tomographer `sdss_ref_all.fits`, all `Sample_ID` values,
  finite valid coordinates, requested positive redshifts, and occupied
  ELG/random RING NSIDE=8 pixels. This coarse overlap cut makes the reference
  compatible with the legacy pixel indexing; it is not an exact survey mask.
- Full random catalogue by default. `RANDOM_FRACTION=0.1`, for example, enables
  reproducible uniform thinning with seed 1909. It changes random shot noise;
  assess convergence before using a thinned run for science.

The default first run measures **0.80 ≤ z < 0.85**, combining five Δz=0.01
pair-count bins into one Δz=0.05 result. Radial bins, NSIDE=4 jackknife regions,
NSIDE=8 local density correction, and equalization match `run_test.sh`.
Integration uses the last three radial bins, approximately 2.517–10 h⁻¹ Mpc.
The wrapper compensates for `cross_correlate.py`'s inclusive upper bin start
without changing its code. Redshift settings must lie on the 0.01 grid, with
`DZ` an integer multiple of `COUNT_DZ` and the requested range divisible by `DZ`.

## Submit when ready

From the repository root, supply your NERSC allocation:

```bash
sbatch --account=YOUR_ACCOUNT run_elg_sdss.slurm
```

The script requests one CPU node, one serial process, the node's available
memory, and a 12-hour limit. This follows the
[NERSC Slurm conventions](https://docs.nersc.gov/systems/perlmutter/running-jobs/).
No account or email address is inherited from the old author's scripts.
This is a resource request, not a measured runtime estimate. The full inputs
and all-pairs radius queries require compute-node execution. Low-redshift bins
can be much more expensive; narrow redshift slices do not guarantee that a
full-catalogue run fits in memory. Check the first job's memory and runtime
before expanding the range. The code is serial; additional nodes do not help.

For a subsequent wider run, after reviewing the first result:

```bash
ZMIN=0.80 ZMAX=1.60 sbatch --account=YOUR_ACCOUNT run_elg_sdss.slurm
```

Inputs can be overridden with `PHOT`, `RANDOMS`, and `REFERENCE`; the reference
needs `RA`, `DEC`, `Z` columns. `PYTHON` and `REPO` are also configurable.
By default each job creates `$SCRATCH/elg-sdss-$SLURM_JOB_ID`. Set `WORKDIR` to
choose another **new absolute directory**. Existing directories are rejected
to prevent stale trees or counts from being reused. Reserve space for compact
catalogues, trees, pair lists, and logs, beyond the source catalogue sizes.

## Outputs

- `wbar.txt`: `z_low z_high wbar sigma_jackknife`.
- `correlations/`: legacy radial measurements, covariance, and resamples.
- `integrated/`: original `get_b_dndz.py` outputs.
- `inputs/selection.json`: source paths, selected counts, and sampling settings.
- `inputs/occupancy_nside8.npz`: photometric/random occupancy maps.
- `code/`: copies of the scripts, Git revision/diff, and installed packages.
- `logs/`: preparation, counting, errors, and integration logs.

Here `wbar = sum(Δr/r × w)` uses the locally normalized correlation (column 6
in the legacy table). **No conversion to b dN/dz is performed.** The legacy PDF
still carries its original b dN/dz label; use `wbar.txt` with the interpretation
above. A non-finite final result fails the job and leaves diagnostics intact.
The mixed SDSS reference selection and coarse overlap cut still need scientific
validation before treating this as a calibrated redshift distribution.

## Validation

The complete wrapper was exercised locally on the small bundled CMASS fixture
with a test stub for `srun` (no allocation). All 107 columns match the repository
reference table exactly; the integrated result is 0.2908933000 ± 0.1333327134.
A second fixture verified five-slice aggregation, retention of reference row
zero, masked-row and unsupported-reference rejection, and agreement between the
global correlation and the raw pair counts. The only new estimator edits round
the number of count slices and recognize a nonempty index array containing only
zero.

The first full-catalogue job (60141624) prepared the inputs successfully but
failed before pair counting while saving the random BallTree: Python 3.7's
default pickle protocol cannot serialize a bytes object larger than 4 GiB.
The three `pickle.dump` calls in `cross_correlate.py` now explicitly use
protocol 4. The failed random-tree file is incomplete and must not be loaded;
a fresh submission uses a new output directory and rebuilds the trees.
Full-measurement runtime and peak memory remain unmeasured.
