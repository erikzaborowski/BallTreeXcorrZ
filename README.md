# BallTreeXcorrZ
cross-correlation redshifts using sklearn's BallTree

We implement the clustering redshift method of Menard++13 (see also Newman+08,McQuinn&White+2013, among others) using sklearn's BallTree class. This code can quickly compute angular clustering, b*dN/dz, and their errors, for ~1M photometric and ~100k spectroscopic objects.

Requires numpy, healpy, astropy, scikit-learn.

The code separates pair-counting and error estimation & redshift binning, allowing the user to efficiently test different redshift bin widths and error-estimation schemes.

Pair counting is done in cross_correlate.py, which starts by generating a BallTree of the photometric catalog or photometric randoms. Optionally one can load a pre-existing tree, which allows you to run different deltaz slices at once. See https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.BallTree.html for details on BallTree. I use the default leaf_size as I find tree-generation is <20 minutes for my largest catalogs, but one could imagine optimizing by playing with leaf_size. I use the 'haversine' metric to compute curved-sky distances between points.

After generating (or loading) the tree, I then loop over redshift slices of deltaz=0.01, chosen to be as small or smaller than you would want to use on data. For each redshift slice I return the pair counts around each spectroscopic object and the index of that spectroscopic object in the spectroscopic catalog.

Generating and querying the tree are the most computationally expensive steps of the process. I have not set up cross_correlate.py with MPI, but instead leave it up to the user to choose a zmin and zmax and potentially run several chunks in parallel.

error_estimator.py computes the cross-correlation and is capable of several methods of error estimation.  It supports both jackknife 
and bootstrap error estimation, and always computes Poisson error bars by default.  If the user wants bootstrap error bars, error_estimator.py computes the bootstrap three different ways: the "literal" bootstrap in which certain regions are fully masked and others are counted twice (thus a pair with both objects in a double counted region will be quadruple counted); a "marked" bootstrap (Loh &Stein 2004, Loh 2008) where only the positions of the spectroscopic objects are resampled; and a "sqrt" bootstrap in which I take the square root of the weights from the literal bootstrap.  For the jackknife error bars I compute both leave-one-out and leave-two-out jackknifes.  I also include a method to equalize the areas of the jackknife regions and weight the jackknife covariance estimate by the  (slightly non-equal) areas of the resulting regions (e.g. Myers et al. 2004).

Finally, the b*dN/dz distribution is computed in get_b_dndz.py.

For example, to run the test:

python cross_correlate.py --no-loadtree --dz 0.05 --Smin 0.1006799 --Smax 10 --Nbins 10 test/cmass_masked-r1-v2-flag-wted_30_35_cmass+lowz_mask.fits test/cmass_random0_masked-r1-v2-flag_30_35_cmass+lowz_mask.fits test/cmass_masked-r1-v2-flag-wted_30_35_cmass+lowz_mask.fits

python error_estimator.py --jackknife --equalize --Smin 0.1006799 --Smax 10 --nbins 10 --orig_deltaz 0.05 --nside 4 --nboot 1000 --dz 0.05 --zmin 0.30 --zmax 0.35 --small_nside 8 test/cmass_masked-r1-v2-flag-wted_30_35_cmass+lowz_mask.fits test/cmass_random0_masked-r1-v2-flag_30_35_cmass+lowz_mask.fits test/cmass_masked-r1-v2-flag-wted_30_35_cmass+lowz_mask.fits ./ ./'

This creates 10 logspaced bins between smin = 0.1 and smax = 10 h^-1 Mpc. Angle is converted to distance
in bins given by the --dz argument in cross_correlate.py--these bins can be made finer than the bin
in which you compute the correlation function. I typically use --dz = 0.01 or 0.001. dz=0.05
is only used to construct the test here.
The --loadtree argument will reload the tree, saving time for larger datasets where creating the tree is expensive.

In error_estimator, the --jackknife arguments tells it to compute the jackknife errors, and --equalize
tells it to correct for the slightly different sizes of the jackknife regions.  --orig_deltaz
corresponds to the --dz argument of cross_correlate.  --nside is the nside used to construct
jackknife regions--the code bins all points into healpixels of this nside, and then combines
them together to make regions of roughly the same size.
zmin and zmax give the minimum and maximum of the z range, and --dz gives its spacing.
"small_nside" allows the user to adjust the size of the pixels used to estimate
the local galaxy density--small_nside = 8 means that the code uses nside=8 pixels
to estimate the local density.

To average over the relevant scales and create a b*dndz file:
python get_b_dndz.py --xcorrdir_name path/to/directory --gal_name path/to/output
--error jackknife --dz 0.05 --njack njack, where njack is the number of jackknife
regions used.

run_test.sh runs a test set, which is compared to the correlation function
computed from astropy's SkyCoord module in test_correlation_function.py.
This verifies that we recover the correct correlation function on the curved sky.

## Photometric ELGs with the SDSS reference sample

`run_elg_sdss_full.slurm` drafts the full original redshift range: **0 < z < 4**,
in 80 output bins of width 0.05, using native pair-count bins of width 0.01.
It calls `run_elg_sdss.slurm` once per output bin, retaining the existing
`cross_correlate.py` → `error_estimator.py` → `get_b_dndz.py` workflow.
The sample has `elg_mask == 0` and unit weights. Reference overlap uses occupied
photometric and random NSIDE=8 pixels, as in the pilot.

Submit from this repository on `dr2-update-minimal`:

```bash
sbatch --account=desi run_elg_sdss_full.slurm
```

The array requests one CPU node per task, at most two tasks concurrently, and
48 hours per task. Low-redshift queries cover much larger angles than the pilot;
their runtime has not been benchmarked. Optional `--query-batch-size` limits
the number of references queried together without changing each native slice's
mean redshift, angular bins, or counting calculation. The array uses batches of
1 below z=0.10 and 64 elsewhere; `QUERY_BATCH_SIZE` overrides this. The original
whole-slice query remains the default (0) when calling the counting script or
pilot job directly.

By default, the array reuses the selected ELG/random catalogues and BallTrees
from `/pscratch/sd/e/ezaborow/elg-sdss-60164850`; keep that directory available.
`PREPARED_RUN` can select another compatible run. Each task checks the cached
sample settings and scikit-learn version, then selects its own SDSS references.
It writes into `$SCRATCH/elg-sdss-full-ARRAY_JOB_ID/bin-NNN/`, including
`wbar.txt`, radial correlations, logs, and a code/settings snapshot. Output
directories must be new. `FULL_RUN_ROOT` can override the parent directory.
To check only the pilot's 0.80–0.85 bin through this new wrapper, submit with
`--array=16` instead of the default full array.

### Several bin workers per node

`run_elg_sdss_workers.slurm` is an alternative to the array: it reserves **two
CPU nodes with four single-threaded workers per node**, so eight bins can run
at once. Workers atomically claim the next bin when they finish; each bin still
calls the same full-range wrapper and legacy measurement scripts. This improves
throughput across bins, not the speed of an individual bin. Each process loads
its own catalogues and trees. No shared-memory assumption is made.

```bash
sbatch --account=desi run_elg_sdss_workers.slurm
```

The completed pilot peaked at about 68 GiB; four such processes would use about
272 GiB on a 512 GB CPU node. The low-z timing probe peaked at 46 GiB but did not
finish a native slice. Four workers is a conservative starting configuration,
not a measured full-range memory guarantee or speedup. Concurrent I/O and the
untested lowest bins still need a compute-node benchmark before increasing it.
The draft accepts one or two nodes and one to four workers per node. For example,
`--nodes=1 --ntasks-per-node=2 --export=ALL,BIN_START=16,BIN_STOP=18` selects a
two-bin concurrency check (0.80–0.90); bin index bounds are half-open.

The worker job uses the same `$SCRATCH/elg-sdss-full-JOB_ID/bin-NNN/` layout as
the array. Top-level `logs/bin-NNN.log` and `claims/NNN/` record each bin's owner,
elapsed time and exit code. Failed bins are reported while other workers finish
their work; they are not automatically retried. Each allocation requires a new
output root. The 48-hour limit covers the whole worker allocation, whereas the
array's limit applies separately to each bin. Only the outer worker launch uses
`srun`; the per-bin wrapper executes Python inside that existing step, avoiding
nested launches. The original one-bin/array modes still use their usual steps.

Open `plot_elg_sdss.ipynb` with a Jupyter kernel containing NumPy and matplotlib
(the existing `elgcmb` kernel is suitable). It defaults to the current full DR1
run, showing separate BGS/LRG/ELG/QSO and combined measurements. Set `RUN_ROOT`
to an SDSS run directory to display that run instead, and run all cells. The
notebook reads only small output tables; it reports missing and invalid bins
without replacing them with zeros, so it also works during a partial run. The
orange histogram superimposes the calibrators' `N_SPEC / (z_high - z_low)` from
the saved correlation headers on a separate right axis. It shows the reference
sample used in available bins, not an inferred photometric redshift distribution.

The plotted quantity is the scale-integrated, locally normalized correlation
returned by `get_b_dndz.py`, using the last three radial bins (approximately
2.517–10 h^-1 Mpc), with the existing equalized leave-one-out jackknife errors.
No conversion to b*dN/dz is applied. These outputs contain per-bin errors, not
a cross-redshift covariance matrix.

## Photometric ELGs with DESI DR1 references

`run_elg_desi_dr1.slurm` drafts a single **LRG 0.80 <= z < 0.85** pilot,
using the same selected photometric ELGs, randoms and cached trees as the SDSS
pilot. It requests one CPU node, one counting process and a four-hour limit
(a cap, not a measured DR1 runtime), with `--mail-type=ALL`.
After approval for this compute workload, submit from the repository root:

```bash
sbatch --account=desi run_elg_desi_dr1.slurm
```

The wrapper calls `run_elg_sdss.slurm` with `REFERENCE_FORMAT=desi-dr1`;
the only different pipeline step is `prepare_elg_desi.py`. The counter,
error estimator and integrator receive `inputs/desi_lrg.fits` instead of
`inputs/sdss.fits`. Native redshift slices remain 0.01, the output bin is 0.05,
query batches are 64, and all radial scales and jackknife settings are shared
with the SDSS driver. Photometric and spectroscopic weights are **unity**.

Preparation reads `{TRACER}_{NGC,SGC}_clustering.dat.fits` from
`/global/cfs/cdirs/desi/public/dr1/survey/catalogs/dr1/LSS/iron/LSScats/v1.5`.
It checks TARGETID uniqueness within and across both input caps, then preserves
input order (NGC followed by SGC) for references with valid coordinates and
positive redshifts in the requested half-open range and occupied photometric
AND random NSIDE=8 pixels. This is the same coarse overlap selection as SDSS,
not an exact survey-footprint mask. An empty selection or duplicate TARGETIDs
fails preparation. `TARGETID` stays int64 and `WEIGHT` is retained but **not
applied**; its invalid/nonpositive values are counted in `selection.json`.
The supplied clustering catalogues are used without extra raw-spectrum cuts.

`PREPARED_RUN` defaults to `/pscratch/sd/e/ezaborow/elg-sdss-60164850` and is
required for DR1. Sample settings, FITS row counts and the cached tree's
scikit-learn version are checked before counting. The adapter shares the cache
reuse helper with SDSS; no photometric catalogue or tree is rebuilt.
`DR1_ROOT`, `TRACER`, `ZMIN`, `ZMAX`, `DZ`, `COUNT_DZ`, `QUERY_BATCH_SIZE`,
`PREPARED_RUN`, and `WORKDIR` can be overridden through the environment.
Supported tracers are BGS_BRIGHT, LRG, ELG_LOPnotqso, QSO and ALL (their unique
TARGETID union). The default output is `$SCRATCH/elg-desi-dr1-lrg-JOB_ID/`, with
`wbar.txt`, correlations, logs, code snapshots and input-selection provenance.
Each output directory must be new. The plotting notebook can also read
this single-bin run by setting `RUN_ROOT` to that directory.

The adapter's synthetic checks read no survey catalogues:

```bash
/global/cfs/cdirs/desi/users/ezaborow/conda/envs/elgcmb_legacy/bin/python -m unittest -v test_prepare_elg_desi
```

### Full DR1 range: separate tracers and their union

Set `RUN_ROOT` in `plot_elg_sdss.ipynb` to the full run's parent directory to
overlay all five measurements and show individual panels with their calibrator
histograms. Re-running the cells picks up newly completed bins. The status
table uses `reference_counts.tsv` to distinguish empty bins from unfinished
measurements; the combined curve reads the direct union measurement, rather
than averaging the tracer curves. Optional exports include both figures and
per-sample measurement and calibrator-count tables.

`run_elg_desi_dr1_full.slurm` drafts **0.05 <= z < 4**, in 0.05-wide bins, for
BGS_BRIGHT, LRG, ELG_LOPnotqso, QSO and ALL. It uses the available rows in these
clustering catalogues within that interval, without additional tracer-specific
redshift cuts. The 0--0.05 bin is omitted. Two CPU nodes with four serial workers
per node process the nonempty tracer/bin tasks, with a 48-hour allocation limit
and `--mail-type=ALL`. This limit is not a runtime prediction: low-z bins have
much larger angular searches than the LRG pilot. Batches contain one reference
below z=0.1 and 64 otherwise. All photometric and spectroscopic weights remain
unity. Submission requires approval for this full-range workload:

```bash
sbatch --account=desi run_elg_desi_dr1_full.slurm
```

Before counting, a compute-node preparation pass audits the entire requested
range and writes the separate catalogues and their union under `inputs/`.
`reference_counts.tsv` lists counts including empty bins; `tasks.txt` includes
only nonempty bins, starting at low z. Each task then calls the existing DR1
pilot wrapper, including its ordinary preparation step, so it retains the same
counting, normalization, integration and error calculation. Separate-tracer
tasks use their ordinary preparation. ALL tasks instead slice the global
prepared union using `PREPARED_COMBINED_INPUTS`, so a shared target with different
redshifts cannot reappear in different bins. `BIN_START`
and `BIN_STOP` optionally restrict the half-open interval of 0.05-wide bin
indices, with defaults 1 and 80. Counts with too few references or invalid
jackknife errors fail the task and remain identifiable in its logs; other
tasks continue. There is no automatic retry.

For ALL, shared TARGETIDs occur once. RA and DEC must agree to absolute 1e-8
degrees; coordinate conflicts still stop preparation. The full-range job uses
`--redshift-conflicts exclude`: every ID whose full redshift span exceeds 1e-8
is excluded from the combined catalogue, before binning. Separate-tracer
catalogues are unchanged. `inputs/excluded_redshift_targetids.fits` records
each excluded TARGETID, Z_MIN, Z_MAX and tracer membership; `selection.json`
records the policy and counts. Standalone preparation defaults to the original
strict `--redshift-conflicts error` policy unless explicitly changed.
The diagnostic for failed job 60336994 identified 43,325 such targets (0.453%
of the 9,570,544 unique references); the expected combined sample for the same
inputs and range is 9,527,219 targets.
`TRACER_BITS` records membership (BGS=1, LRG=2, ELG=4, QSO=8). The combined
`WEIGHT` is explicitly unity; original weights remain in the component FITS
files. This is a combined *correlation* measurement, with a redshift-dependent
effective reference bias, just as for the pooled SDSS sample. It is computed
directly from the union, not by averaging the separate curves or assuming their
errors are independent. No spectroscopic-bias correction is applied.

The default root is `$SCRATCH/elg-desi-dr1-full-JOB_ID/`, with subdirectories
`bgs_bright/`, `lrg/`, `elg_lopnotqso/`, `qso/` and `all/`. Each contains
`bin-NNN/wbar.txt`, correlations and logs. Top-level `claims/` records exit
codes and elapsed times, and `logs/` contains task-driver logs. A new output
root is required for every allocation.

## Spectroscopic bias fitting (minimal branch)

`fit_spectroscopic_bias.py` adds the single-bin fit from equation 5.9 of
[Krolewski et al.](https://arxiv.org/pdf/1909.07412):
`w_ss(theta) = b_s^2 T(theta)`. The original master branch has no such fit;
the larger `dr2-update` branch has one inside its separate pipeline. This
standalone implementation leaves the legacy cross-correlation scripts intact.

The template integrates nonlinear total-matter Halofit power through the
**squared, normalized spectroscopic radial selection** and averages the Bessel
kernel over each full angular annulus. The default fit selects complete annuli
between 2.5 and 10 h^-1 Mpc (including the existing 2.517--10 bins). It uses
Planck15 distances to match the legacy runs, rather than changing their
cosmology to the paper's Planck 2018 cosmology. It fits the amplitude `A=b_s^2`
with the supplied full covariance, reporting `A`, `sigma_A`, positive `b_s`,
`sigma_b`, an asymmetric transformed Delta-chi-square=1 interval, chi-square,
degrees of freedom, and covariance condition. Nonpositive amplitudes are
retained as diagnostics with no usable bias. Singular covariances fail instead
of being silently regularized. Errors condition on the supplied covariance
and template; no covariance-precision correction or cosmological/model error
is included. Near zero amplitude, the linear `sigma_b` approximation is poor.

### Inputs and first measured run

**No measured biases have yet been produced on this branch.** The existing
`wbar.txt` files measure photometric ELG x reference cross-correlations and
cannot determine reference bias by themselves. We first need a separate
angular autocorrelation for each reference tracer/bin with matching
**spectroscopic** randoms, selection and footprint. Photometric ELG randoms
cannot stand in for them. The old Landy--Szalay scripts exist but have not been
ported/validated for this stage. The pooled tomographer SDSS catalog also needs
appropriate component identification and selection/random catalogs before
separate-tracer autocorrelations can be measured. Do not substitute the paper's
tabulated biases for these different samples.

The fitter consumes one small NPZ measurement per tracer/bin, written with
`numpy.savez`, containing:

| Key | Contents |
|---|---|
| `theta_edges_rad` | N+1 fixed angular edges in radians, strictly positive |
| `w_auto` | N angular spectroscopic autocorrelation values |
| `covariance` | N x N covariance of those values |
| `z_edges` | Radial selection histogram edges spanning the measurement bin |
| `nz_counts` | Counts or summed selection weights in each radial histogram cell, **not** density |
| `metadata_json` | JSON string, described below |

Required metadata: `quantity="spectroscopic_angular_autocorrelation"`,
`cosmology="Planck15"`, `tracer`, `sample_id`, `weights`, `randoms`, `zmin`,
`zmax`, and `geometry_z` (the redshift used to convert the angular annuli to
the fitted physical scales). The histogram should represent the selected
radial distribution for this same sample, e.g. from suitable spectroscopic
randoms. Its resolution/noise must be checked because the template uses p(z)^2.
No catalog smoothing or assumption of a uniform radial distribution is hidden
in the fitter. The model assumes radial/angular separability and uniform-area
annulus averaging; it does not convolve a detailed survey window or correct
RSD, magnification or integral constraints.

For the first comparison, keep the same unit-weight reference selection as
the existing cross-correlations. The paper applied survey correction weights;
using them here would require consistently updating the corresponding
cross-correlations as well, rather than mixing weighting conventions.

`make_spectroscopic_bias_power.py` can create the other input: a saved
Takahashi-Halofit grid of total-matter P(k,z), with explicit units and
provenance. It uses CAMB, which is **not installed in elgcmb_legacy**. Its
solver path has not yet been exercised here; installation/environment choice
and an approved compute run remain necessary. It fixes As=2.142e-9, ns=0.9667
and the Planck15 background, recording CAMB settings and checking distances
against Astropy. A saved compatible grid can also be supplied directly:
`z`, `k_hmpc`, `pk_mpc_over_h3` of shape (Nz,Nk), and `metadata_json` with
`cosmology="Planck15"`, `model="halofit_takahashi"`, `matter_field="total"`,
`k_unit="h/Mpc"`, `power_unit="(Mpc/h)^3"`.

Example commands **once those inputs and the compute workload are approved**:

```bash
# On a compute node, using a Python environment containing CAMB:
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python make_spectroscopic_bias_power.py \
  --zmin 0.80 --zmax 0.85 --output matter_080_085.npz
# Fit the small measured auto table; no pair counting or CAMB solve here:
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python fit_spectroscopic_bias.py \
  --measurement lrg_auto_080_085.npz --power matter_080_085.npz \
  --output lrg_bias_080_085.json
```

Both commands refuse to overwrite outputs. The fitter checks radial
quadrature refinement, but k-range, k-sampling and power-grid redshift
convergence still need testing on a real Halofit pilot before production.
An initial LRG 0.80--0.85 autocorrelation would match the completed legacy
cross-correlation pilot. The same fit then applies independently to each
SDSS/DESI tracer and bin; the combined unique-target catalog is a different
sample, not an extra independent tracer.

For later combination, remove the fitted spectroscopic bias from each
cross-correlation and propagate its uncertainty. Differences in radial matter
response (equations 5.4--5.5) and magnification must also be accounted for to
interpret a common photometric kernel. The paper's inverse-variance combination
assumes the relevant independence; our overlapping tracers and shared
photometry require their cross-covariance. This fitter does not yet combine
tracers or convert the legacy wbar measurements to b*dN/dz.

Small analytic tests, with no survey inputs or CAMB solve:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /global/cfs/cdirs/desi/users/ezaborow/conda/envs/elgcmb_legacy/bin/python \
  -m unittest -v test_spectroscopic_bias
```
