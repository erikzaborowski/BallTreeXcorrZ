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
(the existing `elgcmb` kernel is suitable). It initially displays the completed
pilot. Set `RUN_ROOT` to the full array's parent directory and run all cells to
plot the available measurements and reference counts against redshift. The
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
