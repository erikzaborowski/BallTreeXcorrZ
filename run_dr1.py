"""Run one DR1 clustering-redshift stage inside an existing compute allocation."""
import argparse
import os


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', help='JSON configuration; see dr1.example.json')
    parser.add_argument('stage', choices=['cache', 'regions', 'cross-counts', 'auto-counts',
                                        'measure', 'matter', 'calibrate', 'plot'])
    parser.add_argument('--workers', type=int, default=1, help='single-thread count workers')
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    if args.stage != 'plot':
        if not os.environ.get('SLURM_JOB_ID'):
            parser.error('requires an existing approved allocation; this program never allocates')
        names = ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
                 'NUMEXPR_NUM_THREADS','NUMBA_NUM_THREADS')
        if any(os.environ.get(n) != '1' for n in names):
            parser.error('set all numerical thread caps listed in README to 1')
    from bxz.pipeline import run
    run(args.config, args.stage, args.workers)


if __name__ == '__main__':
    main()
