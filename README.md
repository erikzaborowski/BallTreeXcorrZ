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

The existing
`wbar.txt` files measure photometric ELG x reference cross-correlations and
cannot determine reference bias by themselves. Bias fitting needs a separate
angular autocorrelation for each reference tracer/bin with matching
**spectroscopic** randoms, selection and footprint. Photometric ELG randoms
cannot stand in for them. The old Landy--Szalay scripts remain unchanged;
`measure_spectroscopic_auto.py` supplies a small unit-weight BallTree pilot
with exact finite-N pair normalization and region deletions. The pooled
tomographer SDSS catalog also needs
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
solver path was exercised with CAMB 1.6.4 in the existing `elgcmb` environment
on the interactive pilot. It fixes As=2.142e-9, ns=0.9667
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

### Interactive LRG autocorrelation pilot

`run_spectroscopic_auto_pilot.sh` runs inside an approved interactive CPU
step; it does not allocate resources. It measures the prepared LRG 0.80--0.85
sample from cross-correlation job 60331727, reads the public DR1 NGC/SGC
`LRG_*_0_clustering.ran.fits` catalogs, applies the same NSIDE=8 overlap, and
uses unit weights. The three fixed angular annuli correspond to the legacy
2.517--10 h^-1 Mpc bins at the sample mean redshift. This differs from the
cross-correlation's finer native redshift slicing; the auto model uses these
exact fixed angular edges.

Set `PREPARED_RUN` to another completed LRG bin directory to reuse this
wrapper; its redshift bounds come directly from that bin's `selection.json`.
Set `WORKDIR` to a new output directory. Optional `POWER_GRID` supplies an
absolute path to an existing compatible Halofit grid, which is linked into
the run and checked by the fitter. This permits several bins to share one
power calculation. The wrapper rejects non-LRG prepared samples because its
random-catalog paths are specifically for LRGs.

The pilot reuses the original `equalize_pixels` function by loading its
function definition from `error_estimator.py`, without executing that script's
catalog-reading CLI. Cached photometric-random counts reproduce the same
74 jackknife regions; their area weights are checked against the saved
cross-correlation. DD and RR are ordered pairs normalized by N(N-1), DR by
ND*NR. Removing both ends of pairs gives exact leave-one-region-out counts.
Small tests compare counts and deletions with brute-force SkyCoord separations:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /global/cfs/cdirs/desi/users/ezaborow/conda/envs/elgcmb_legacy/bin/python \
  -m unittest -v test_spectroscopic_auto
```

Interactive pilot 60523395 writes to
`/pscratch/sd/e/ezaborow/lrg-auto-pilot-60523395/`. It contains 233,978 LRGs
and 1,570,017 randoms. Preparation and counting took 51 seconds after Python
startup. The baseline fit is b_s=2.3666 +/- 0.0262, chi-square=7.809 for two
degrees of freedom. This is a diagnostic unit-weight fit: the shape mismatch
should be investigated before using it as a final calibration.

Saved-grid checks in `baseline_validation.json` halve the k sampling, truncate
kmax from 30 to 20 or 10 h/Mpc, and coarsen the radial histogram from ten to
five cells. They shift b_s by at most 0.00020 (under 0.008 of its fitting
error). CAMB/Astropy distances agree to 3.2e-6 fractionally. An optional
wider/finer CAMB solve requested a range outside native coverage; the retry
to diagnose those bounds stalled on shared-filesystem I/O and was stopped.
The requested pilot and saved-grid checks completed, and the allocation was
released. No extrapolated power spectrum was used.

Outputs include `auto/measurement.npz` (auto, covariance, radial histogram,
and all jackknife auto/histogram realizations), `auto/pair_counts.npz`,
`power.npz`, `bias.json`, `baseline_validation.json`, `auto_fit.png`, logs and
code snapshots. The retained region
identity allows later joint propagation with the legacy cross-correlation;
the reported fitting error currently conditions on the full-sample template
and covariance. This pilot does not update any photometric measurements.

Three additional LRG bins ran sequentially in interactive allocation
60524932, with the same settings and a shared Halofit grid over 0.70--1.05:

| Redshift | Spectra | b_s | Conditional fitting error | Chi-square / dof |
|---|---:|---:|---:|---:|
| 0.70--0.75 | 193,529 | 2.3411 | 0.0280 | 5.164 / 2 |
| 0.90--0.95 | 169,092 | 2.4775 | 0.0288 | 2.475 / 2 |
| 1.00--1.05 | 78,659 | 2.5628 | 0.0285 | 1.104 / 2 |

Outputs are under `/pscratch/sd/e/ezaborow/lrg-auto-bins-60524932/`, with
per-bin `auto/measurement.npz`, `bias.json` and logs. Preparation/counting
took 75.9, 29.3 and 12.1 seconds respectively, excluding Python startup and
fitting. The comparison `summary.csv`, `auto_fits.png` and
`bias_vs_redshift.png` include the earlier 0.80--0.85 pilot; diagonal-only
fits are shown for comparison with the full-covariance fits. The saved
`run.sh` and `code/report.py` record this interactive run and its reporting.

### Photometric systematics weights with DESI DR1

`run_elg_desi_dr1_weighted.slurm` applies the photometric catalog's
`WEIGHT_SN_DEBV0` column and each DESI tracer's `WEIGHT`, retaining the
`elg_mask == 0` selection and unit photometric-random weights. It covers the
same 93 populated BGS/LRG/ELG/QSO bins as the bias run below, separately by
tracer. Its draft allocation is one CPU node, four workers, 48 hours, account
`desi`, and mail type `ALL`; submission requires workload approval.

`prepare_weighted_elg.py` streams the selected RA/DEC/weight columns and checks
every coordinate and its row order against the original preparation. It also
requires unchanged NSIDE=8 occupancy and positive total weight in each
occupied pixel. Invalid weights fail instead of being silently replaced or
discarded; zero-weight objects contribute zero. The verified selection keeps
the cached coordinate trees, randoms, and 74 jackknife regions usable.

The original `cross_correlate.py` now accepts `--phot-weight-column`: its
per-reference/per-annulus/pixel DD histograms sum photometric weights, while
DR remains unit photometric-random counts. `error_estimator.py` accepts the
same option, applies the existing spectroscopic row weights to DD/DR, and
uses photometric weight sums in the local-density and global normalizations.
Omitting the option keeps unit photometric weights. Pair-cache metadata guards
against interpreting histograms with the wrong weights, inputs, or geometry.
The old unit-weight pair histograms cannot recover individual photometric
weights, so this job regenerates the pairs in a new output directory.

For each bin, the job reuses the matching weighted spectroscopic autocorrelation
from job 60623365 and refits bias for the matching jackknife deletions with
`combine_spectroscopic_bias.py`. That helper checks the photometric validation
report before accepting the original autocorrelation footprint. Each tracer
gets `summary.csv`, `joint_measurements.npz`, and a `weighted_TRACER.png` plot;
the NPZ retains the paired cross/bias realizations and their covariance.
The reported quotient remains wbar/b_s, not the fully converted b*dN/dz.
The new photometric weights do not alter or remedy the spectroscopic fit-quality
issues already reported for LRGs.

Small tests of preparation, explicit weighted pair products, spatial deletions,
weight-rescaling invariance, and the unit-weight limit:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /global/cfs/cdirs/desi/users/ezaborow/conda/envs/elgcmb_legacy/bin/python \
  -m unittest -v test_photometric_weights test_joint_spectroscopic_bias
```

Restricted-footprint job **60663330** was submitted with the allocation above.
Its output root is `/pscratch/sd/e/ezaborow/elg-desi-dr1-weighted-60663330/`.

#### Additional full-DR9 photometric sample

`run_elg_dr9_desi_dr1_weighted.slurm` is a separate job for
`/global/cfs/cdirs/desicollab/users/ezaborow/projects/elg_cross_cmblens/files/elg_targets_sysnet.fits`
and `elg_randoms_sysnet.fits` in the same directory. The owner confirmed the
same `elg_mask == 0` cut, target `WEIGHT_SN_DEBV0`, and unit random weights.
The unmasked input tables contain 42,193,919 targets and 824,728,828 randoms;
selected counts are determined by preparation on compute.

`prepare_elg_fresh_footprint.py` calls the existing photometric and DESI
preparation functions to create a new coordinate-only base, occupancy map,
and separate reference bins over 0.05 <= z < 4 wherever the DR1 catalogs have
support. It constructs fresh haversine BallTrees in the original coordinate
order. `prepare_weighted_elg.py` then supplies the same checked weighted
photometry used by the restricted-run worker. The preparation retains the
coordinate-only base so validation and auto/cross footprint checks can remain
unchanged; the measurement uses the weighted file.

The full-DR9 wrapper runs that shared worker with `MEASURE_AUTO=1`, measuring
new weighted spectroscopic autocorrelations after each cross-correlation and
checking their new jackknife regions against the cross. Only the existing
cosmology power grids are reused from the restricted analysis, not its trees,
reference selections, bias measurements, or covariance. Its collector uses
`--no-unit-comparison` because there is no earlier unit-weight cross on this
footprint. Output root: `${SCRATCH}/elg-dr9-desi-dr1-weighted-JOBID/`.
Full-DR9 job **60663553** was submitted after separate approval, requesting
one CPU node, four workers, 48 hours, account `desi`, and mail type `ALL`.
Its output root is `/pscratch/sd/e/ezaborow/elg-dr9-desi-dr1-weighted-60663553/`.
Both jobs can occupy one node each, respecting the two-node concurrent limit.

### All-tracer weighted DESI DR1 bias fits

`run_desi_bias_all.sh` runs inside an approved allocation and uses the existing
`measure_spectroscopic_auto.py` and `fit_spectroscopic_bias.py` unchanged.
It reads the completed DR1 task manifest, applies `WEIGHT` to data and matching
spectroscopic randoms, and covers BGS_BRIGHT (0.05--0.50), LRG (0.40--1.10),
ELG_LOPnotqso (0.80--1.60), and QSO (0.80--3.50), all in dz=0.05 bins.
The 14 completed weighted LRG products from job 60530153 are linked into the
new run; four workers count and fit the remaining 79 bins. This run measures
spectroscopic autos only, retaining all jackknife realizations for later
joint propagation. It does not recompute the photometric cross-correlations.
The union `ALL` sample has no defined combined survey weights/randoms and is
not treated as another tracer.

Interactive job 60623365 started with one CPU node, four single-thread workers,
a two-hour limit, account `desi`, and mail type `ALL`. Outputs are under
`/pscratch/sd/e/ezaborow/desi-bias-all-60623365/`. Each tracer/bin writes
`auto/measurement.npz`, `auto/pair_counts.npz`, `bias.json`, and logs; `claims/`
records per-task exit codes and times. Completed bins survive a time limit.
Four 100-redshift Halofit grids cover the new bins, with the existing 16385
k nodes; each fit retains its radial-refinement check and power provenance.
`summarize_desi_bias.py RUN_ROOT` writes `summary.csv` and `bias_fits.png`,
checks the saved chi-square against the actual residual/covariance, and flags
missing or failed bins. Its bias error is the **conditional fit error**, and
its p-value is the nominal chi-square survival probability; neither substitutes
for the joint cross/bias jackknife errors in the weighted LRG products.

Job 60623365 completed successfully (exit 0) in 32m44s. All 79 new bins and
14 reused LRG bins have finite, positive bias fits. Nominal p<0.05 occurs in
0/9 BGS, 12/14 LRG, 0/16 ELG, and 4/54 QSO bins. The lowest-redshift BGS task
took 22m35s. The covariance uncertainty is not calibrated, and the conditional
fit errors should not be interpreted as model-systematic uncertainties.

### Weighted DESI LRG cross-correlation and joint bias jackknife

`run_weighted_lrg.slurm` prepares the next comparison: all 14 bins from
0.40 to 1.10, width 0.05, using the supplied DR1 v1.5 `WEIGHT` on reference
data and spectroscopic randoms. Do not multiply its component weights again.
`WEIGHT_FKP` is not included. Photometric ELGs and their randoms still have
unit weights; new photometric weights are being generated separately.

The job requests one CPU node, two concurrent serial workers and four hours,
with `--mail-type=ALL`. It reads the completed LRG run at
`/pscratch/sd/e/ezaborow/elg-desi-dr1-full-60339137/lrg/`, reuses the saved
per-reference pair lists, and writes a new `${SCRATCH}/lrg-weighted-jk-JOBID`
directory. No previous result is overwritten. It must be submitted only after
approval of this compute workload. A single shared Halofit grid covers
0.40--1.10; the reference-row neighbor searches do not run again.

The first user-approved interactive run was allocation 60528344, writing to
`/pscratch/sd/e/ezaborow/lrg-weighted-jk-60528344/`. It ran the same wrapper
under salloc (one CPU node, two workers, four-hour limit), rather than sbatch.
It failed after 12m26s while reading spectroscopic random weights. The power
grid and weighted cross-correlations for bins 008 and 009 completed; no auto
measurement or bias fit completed. The allocation was released.

The autocorrelation validator incorrectly required strictly positive weights.
A bounded inspection of the first 250,000 NGC random rows found eight zero
weights, including a z=0.40384 row with FRAC_TLOBS_TILES=0. Zero weights now
pass validation and contribute nothing to weighted pairs/histograms; negative
and non-finite values still fail. Regression tests cover the FITS random reader,
weighted pair counts, deletions, and weight rescaling with zero-weight rows.

The corrected run was restarted with user approval in interactive allocation
60530153 on nid200033, with the same one-node/two-worker/four-hour limits.
Its new output directory is `/pscratch/sd/e/ezaborow/lrg-weighted-jk-60530153/`.
Allocation 60530153 completed successfully (exit 0) in 1h30m56s. All 14 bins,
covering 2,138,627 LRG spectra, have cross, auto, bias-fit and joint-jackknife
products; the combined CSV, NPZ covariance and plot were written. The joint
bias propagation changes errors on wbar/b_s by -3.0% to +1.2% relative to
holding b_s fixed. Fit quality remains a concern: 12 of 14 bins have chi-square
above 5.991 for two degrees of freedom; these statistical results do not
establish the adequacy of the scale-independent bias model.

`error_estimator.py --spec-weight-column WEIGHT` multiplies the original
per-reference DD/DR counts during sparse-matrix aggregation, including the
DR-based local photometric-density normalization. Without this option, its
original unit-weight arithmetic is unchanged. In weighted mode the legacy
count-only Poisson/MJB diagnostic columns are NaN; the measurements and
jackknife uncertainties are computed from the weighted pairs. Each worker
still reads the original photometric inputs for the legacy density/region
calculations; this is compute-node work despite reusing pair lists.

`measure_spectroscopic_auto.py --weight-column WEIGHT` uses products of data
and random weights in its ordered DD/DR/RR matrices. The self-pair-excluding
normalizations are `(sum w)^2 - sum(w^2)` for DD/RR and `sum(wD)*sum(wR)`
for DR, recalculated for every region deletion. Both the full radial
histogram and all deleted histograms sum the spectroscopic random weights.
Raw counts and first/second weight sums are retained separately.

`combine_spectroscopic_bias.py bin` refits `b_s` for every matching jackknife
auto-correlation and radial histogram. It keeps the full-sample auto
covariance fixed as the GLS metric, avoiding a nested covariance estimate.
The radial-cell matter integrals are computed once and recombined with each
deletion's squared normalized radial probabilities. The existing fixed-angle
auto convention and native dz=0.01 cross-counting geometry are retained.
Checks require the same reference file, redshift interval, weighting,
physical scale range, and jackknife region areas/order. Nonpositive fitted
biases fail explicitly; no deletion is silently discarded.

The saved quantities are the original integrated `wbar`, fitted `b_s`, and
`wbar/b_s`, with their aligned jackknife vectors. The latter is **not yet
b*dN/dz**: the radial matter response and magnification treatment are separate
steps. Joint errors come from the distribution of `wbar_j/b_s_j`, centered
on its deletion mean with the original area factors `1 - area_j/area_total`.
The conditional fitting error is recorded as a diagnostic and is not added
again in quadrature. Template/cosmology/model uncertainty is not included.

The collector writes `summary.csv`, `weighted_lrg.png` and
`joint_measurements.npz`, including covariance across redshift bins for all
three quantities and the cross-covariance between wbar and bias. Per-bin
`joint/bias.json` retains the full-fit diagnostics and `joint/measurement.npz`
retains the paired resamples. The plot compares the new weighted cross with
the previous unit-weight cross; it labels the current photometric weights.

Future photometric weights require new weighted photometric pair counts and
density normalizations; they cannot be reconstructed from these pixel-summed
unit-weight pair lists. They must then enter the same joint jackknife flow,
with the effective selection/footprint checked again. The deduplicated DESI
ALL catalog has a unity placeholder weight and is deliberately excluded from
this weighted-tracer workflow.

The local Tomographer reference FITS supplies RA, DEC, Z and Sample_ID only.
Its clone also contains precomputed density/activation maps and bias tables,
but no object-level SDSS LSS weight table or matching random catalogs for
this legacy workflow. Its paper (arXiv:2608.03415, section III.2) describes
using the upstream LSS weights/randoms to build those products. Recovering
that treatment here requires the upstream catalogs; the runtime option
`per_source_weight_colname` applies to the user's test catalog and randoms.

Small synthetic validation, with one thread and no survey inputs:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /global/cfs/cdirs/desi/users/ezaborow/conda/envs/elgcmb_legacy/bin/python \
  -m unittest -v test_spectroscopic_auto test_spectroscopic_bias test_joint_spectroscopic_bias
```
