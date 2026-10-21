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

## DESI DR1 runtime

`run_dr1.py` is the new entry point. The original scripts above are unchanged.
This branch starts from `master` at `c945a39`; that is the repository baseline,
not a verified archival checkout of the exact run used in arXiv:1909.07412.
Only the final measurement path is included: no new tests, pilot jobs, scheduler
scripts, validation campaigns, generated products, or downstream cosmology fit.

### What to review

| Step | Original repository | New runtime |
| --- | --- | --- |
| Inputs and counting | `cross_correlate.py`, BallTree | `bxz/catalogs.py`, `cache_prepare.py`, `paircount/`: weighted DR1/Y3 cell caches and pycorr + Corrfunc |
| Cross-correlation | `error_estimator.py` | `bxz/reference_measurement.py`: `DsDp / [(Ws/Wrs) RsDp] - 1`, with normalization separately for each cap and fine redshift slice |
| Scale integration | `get_b_dndz.py`: weighted integral of the cross-correlation | `bxz/operator.py`: explicit angular-to-comoving scale windows and logarithmic scale integration |
| Reference bias | No spectroscopic autocorrelation calibration in `get_b_dndz.py` | `bxz/reference_auto_estimator.py`, `reference_radial_template.py`, `reference_calibration.py`: DD/DR/RR Landy–Szalay autocorrelations and a joint NGC/SGC same-scale bias fit |
| Conversion | The integral is plotted with a `b*dN/dz` label | `bxz/reference_apparent_kernel.py`, `theory_*.py`: divide by reference bias and projected matter response to obtain apparent photometric `b(z)*phi(z)` |
| Errors and combination | Original resampling options | Common 150-region jackknife; refit bias and recompute windows for every deletion; full covariance across tracers; `bxz/plotting.py` combines overlapping bins |

Read `bxz/pipeline.py` for the complete stage wiring. Its numerical routines
are extracted from the completed DR1 implementation; the frontend replaces
job-specific dependencies on earlier pilot and diagnostic runs. File identity
checks prevent combining counts or products from different configurations.

### Inputs and fixed measurement choices

Copy `dr1.example.json` outside the repository and fill in absolute catalogue,
mask, and new scratch-output paths, the mask SHA-256, and a population name.
The photometric selection is `elg_mask == 0` plus the binary Galactic RING
NSIDE=2048 mask using stored `l,b`. Targets use `WEIGHT_SN`; photometric randoms
have unit weight and define the jackknife regions. Photometric redshifts are
not used. Both files need the columns checked by `bxz/catalogs.py`.

The reference inputs are public DR1 LSS v1.5 clustering catalogues and random
index 0, both NGC and SGC, with supplied `WEIGHT` (no FKP): BGS_BRIGHT at
0.05–0.5, LRG at 0.4–1.1, ELG_LOPnotqso at 0.8–1.6, and QSO at 0.8–3.5.
Cross-count slices have width 0.01; output and reference-auto bins have width
0.05. Two comoving scale bins span 2.5–5 and 5–10 h^-1 Mpc, subject to the
native angular cuts. These scientific defaults are explicit in `pipeline.py`.

The conversion uses the fixed Planck-2018 parameters in `bxz/cosmology.py`,
CAMB 1.6.4 with Takahashi Halofit by default, and HMcode 2020 as an alternative.
It applies the total-matter nonlinear boost to the linear CDM+baryon spectrum.
The model uses Limber projection and uniform-area native annuli; RSD/FoG,
redshift errors, integral-constraint and magnification corrections are omitted.
The output is a signed, apparent `b*phi`, with `phi` the normalized photometric
redshift density. It is not a separately measured bias or normalized `phi`.

### Running

Use Python >=3.11 with the dependencies in `pyproject.toml`, plus the
cosmodesi **pycorr** package and **Corrfunc**. Use an existing compatible
scientific environment: the unrelated package named `pycorr` on PyPI is not
this pair counter. No installation or allocation is performed by this code.

Run the following inside an approved Slurm CPU allocation, from this checkout.
Choose `--workers` within that allocation's CPU and memory limits. It controls
counting only; other stages run serially. Production stages must not run on a
login node. Any submitted Slurm job should include `--mail-type=ALL` and
`--mail-user=ezaborowski@berkeley.edu`.

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 NUMBA_NUM_THREADS=1
python run_dr1.py /path/to/config.json cache
python run_dr1.py /path/to/config.json regions
python run_dr1.py /path/to/config.json cross-counts --workers 4
python run_dr1.py /path/to/config.json auto-counts --workers 4
python run_dr1.py /path/to/config.json measure
python run_dr1.py /path/to/config.json matter
python run_dr1.py /path/to/config.json calibrate
python run_dr1.py /path/to/config.json plot
```

The full counting workload is 1,860 cross tasks (DD/RD) and 558 reference-auto
tasks (DD/DR/RR). Cache creation streams catalogues; counting loads bounded
cell pairs. JSON `limits` bound individual arrays/tasks, not total job RSS,
runtime or scratch usage. Estimate those separately for the selected inputs
and workers. The cache buffer pool is 512 MiB plus input/mask overhead.

Set `cache_manifest` to a completed `bxz.dr1-full-cache.v1` manifest to reuse
an existing cache and skip `cache`. Set `regions_file` to a saved region map
to preserve the previous realization definitions. The cache must include
photometric data/randoms and all reference data/random groups. Completed
count blocks are reusable on restart; other stages publish exclusive outputs.
Use a new output directory when changing source or configuration. Existing
job-specific measurement/calibration archives are not imported by this runner.

### Plotting in Jupyter

Start the notebook in this checkout (or put this checkout on `sys.path`):

```python
from bxz.plotting import plot_tracers_and_combined

fig_tracers, fig_combined, result = plot_tracers_and_combined(
    "/path/to/run", case="takahashi_refined", dlog10_z=None,
)
# Optional: dlog10_z=0.08 combines whole existing bins on an approximate log-z grid.
# Alternative matter prescription: case="mead2020_refined".
combined_covariance = result["combined"]["covariance"]
```

The combination uses fixed inverse marginal-variance weights and propagates
all cross-tracer/redshift covariance as `W C W.T`. Unsupported bins are omitted.
Logarithmic rebinning averages densities with redshift-width weights and snaps
to existing bin boundaries; it cannot create resolution finer than 0.05.
