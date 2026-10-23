#!/bin/bash
#SBATCH --job-name=elg-sdss-legacy
#SBATCH --constraint=cpu
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=0
#SBATCH --time=12:00:00
#SBATCH --output=elg-sdss-%j.log
#SBATCH --mail-user=ezaborowski@berkeley.edu
#SBATCH --mail-type=ALL

set -euo pipefail
: "${SLURM_JOB_ID:?Submit with sbatch; the full catalogues require a compute node}"
REPO=${REPO:-${SLURM_SUBMIT_DIR:?Submit from the repository root}}
PYTHON=${PYTHON:-/global/cfs/cdirs/desi/users/ezaborow/conda/envs/elgcmb_legacy/bin/python}
PHOT=${PHOT:-/global/cfs/cdirs/desi/users/ezaborow/projects/elg_cross_cmblens/files/elg_ebv_y3_targets_sysnet.fits}
RANDOMS=${RANDOMS:-/global/cfs/cdirs/desi/users/ezaborow/projects/elg_cross_cmblens/files/elg_ebv_y3_randoms_sysnet.fits}
REFERENCE=${REFERENCE:-/global/u1/e/ezaborow/dev/tomographer/tomographer/precalculated_data/references/sdss_ref_all.fits}
WORKDIR=${WORKDIR:-${SCRATCH:?Set SCRATCH or WORKDIR}/elg-sdss-${SLURM_JOB_ID}}
export ZMIN=${ZMIN:-0.80} ZMAX=${ZMAX:-0.85} DZ=${DZ:-0.05}
export COUNT_DZ=${COUNT_DZ:-0.01}
RANDOM_FRACTION=${RANDOM_FRACTION:-1.0}
SEED=${SEED:-1909}
# Match run_test.sh. Integrate its last three radial bins (~2.517--10 Mpc/h).
SMIN=0.1006799
SMAX=10
export NBINS=10 LOWER_IND=7
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1

# Legacy scripts use unquoted shell mkdir internally: restrict only output paths.
[[ "$WORKDIR" = /* && "$WORKDIR" != *[!a-zA-Z0-9_./-]* ]] || {
    echo 'WORKDIR must be absolute and use only letters, digits, _, ., /, -' >&2; exit 1;
}
[[ -x "$PYTHON" && -f "$REPO/cross_correlate.py" ]] || exit 1
for input in "$PHOT" "$RANDOMS" "$REFERENCE"; do [[ -r "$input" ]] || exit 1; done
mkdir -p "$(dirname "$WORKDIR")"
mkdir "$WORKDIR"  # Refuse to mix new results with old trees or pair counts.
cd "$WORKDIR"
mkdir code logs correlations plots integrated
cp "$REPO/prepare_elg_sdss.py" "$REPO/cross_correlate.py" \
   "$REPO/error_estimator.py" "$REPO/get_b_dndz.py" "$REPO/run_elg_sdss.slurm" code/
export MPLCONFIGDIR="$WORKDIR/mplconfig"
git -C "$REPO" rev-parse HEAD > code/git-head.txt
git -C "$REPO" diff -- cross_correlate.py error_estimator.py get_b_dndz.py > code/legacy.diff
"$PYTHON" -m pip freeze > code/packages.txt
declare -p PHOT RANDOMS REFERENCE ZMIN ZMAX DZ COUNT_DZ RANDOM_FRACTION SEED \
    SMIN SMAX NBINS LOWER_IND PYTHON REPO > code/settings.sh

# Generate aligned bins; legacy output filenames have only two decimal places.
"$PYTHON" - <<'PY' > bins.txt
import os
from decimal import Decimal
lo, hi, dz, native = [Decimal(os.environ[k]) for k in ('ZMIN', 'ZMAX', 'DZ', 'COUNT_DZ')]
assert 0 < lo < hi and dz > 0 and native > 0
assert all(v % Decimal('0.01') == 0 for v in (lo, hi, dz, native))
assert lo % native == 0 and (hi - lo) % dz == 0 and dz % native == 0
for i in range(int((hi - lo) / dz)):
    a, b = lo + i * dz, lo + (i + 1) * dz
    # cross_correlate.py includes the bin STARTING at --zmax.
    print('{:.2f} {:.2f} {:.2f}'.format(a, b, b - native))
PY
printf 'Output directory: %s\n' "$WORKDIR"
srun "$PYTHON" -u code/prepare_elg_sdss.py "$PHOT" "$RANDOMS" "$REFERENCE" inputs \
    --zmin "$ZMIN" --zmax "$ZMAX" --random-fraction "$RANDOM_FRACTION" --seed "$SEED" \
    > logs/prepare.log 2>&1

tree_option=--no-loadtree
while read -r zlo zhi last_start; do
    printf 'Counting and estimating z=%s--%s\n' "$zlo" "$zhi"
    srun "$PYTHON" -u code/cross_correlate.py "$tree_option" \
        --zmin "$zlo" --zmax "$last_start" --dz "$COUNT_DZ" \
        --Smin "$SMIN" --Smax "$SMAX" --Nbins "$NBINS" \
        inputs/elg.fits inputs/randoms.fits inputs/sdss.fits > "logs/count-${zlo}.log" 2>&1
    tree_option=--loadtree
    srun "$PYTHON" -u code/error_estimator.py --jackknife --equalize \
        --zmin "$zlo" --zmax "$zhi" --dz "$DZ" --orig_deltaz "$COUNT_DZ" \
        --Smin "$SMIN" --Smax "$SMAX" --nbins "$NBINS" --nside 4 --small_nside 8 \
        inputs/elg.fits inputs/randoms.fits inputs/sdss.fits correlations/ plots/ \
        > "logs/errors-${zlo}.log" 2>&1
done < bins.txt

# Derive the resample count from the actual legacy table, not a CMASS constant.
NJACK=$("$PYTHON" - <<'PY'
import os
import numpy as np
counts = []
for lo, hi, _ in np.loadtxt('bins.txt', ndmin=2):
    data = np.loadtxt('correlations/z%.2f_%.2f_jk_loo_equalized.txt' % (lo, hi), ndmin=2)
    extra = data.shape[1] - 13 - int(os.environ['NBINS'])
    assert data.shape[0] == int(os.environ['NBINS']) and extra > 0 and extra % 3 == 0
    counts.append(extra // 3)
assert len(set(counts)) == 1, 'Jackknife regions changed between bins'
print(counts[0])
PY
)
srun "$PYTHON" -u code/get_b_dndz.py --xcorrdir_name "$WORKDIR" --gal_name correlations \
    --bdndz_name "$WORKDIR/integrated" --error jackknife --jack_name loo_equalized \
    --zmin "$ZMIN" --zmax "$ZMAX" --dz "$DZ" --nbins "$NBINS" \
    --lower_ind "$LOWER_IND" --njack "$NJACK" > logs/integrate.log 2>&1
"$PYTHON" - <<'PY'
import os
import numpy as np
bins = np.loadtxt('bins.txt', ndmin=2)
result = np.loadtxt('integrated/dz%.2f/jackknife/correlations_loo_equalized.txt' %
                    float(os.environ['DZ']), ndmin=2)
np.savetxt('wbar.txt', np.column_stack((bins[:, :2], result)),
           header='z_low z_high wbar sigma_jackknife; unit weights; local-density normalization')
if not np.all(np.isfinite(result)):
    raise SystemExit('Non-finite measurement/error: inspect correlations and logs/integrate.log')
print('Finished: ' + os.path.abspath('wbar.txt'))
PY
