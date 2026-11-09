#!/bin/bash
#SBATCH --job-name=gen28
#SBATCH --partition=compute
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:40:00
#SBATCH --output=/scratch/ni2071/Diffusion/Squidiff_git/Squidiff_ms/logger_files/gen28_%A.out
#SBATCH --error=/scratch/ni2071/Diffusion/Squidiff_git/Squidiff_ms/logger_files/gen28_%A.err
# -----------------------------------------------------------------------------
# Purpose: run the Step 28 synthetic test (gen28.py, PREREG_step28_salt_twins.md section 8.2) on a compute node.
# Logic:   the test runs the scorer eight times as a subprocess; each run imports anndata and scipy from the
#          conda environment on the network filesystem, which is slow enough that the login node reaps the process
#          mid-import (seen as exit -2 / KeyboardInterrupt). A compute node has no such reaper. The test touches
#          no real record: every file it reads is generated in a scratch directory and the scorer is pointed at it
#          with S28_TEST_DIR, which also marks its outputs synthetic_test: true.
# -----------------------------------------------------------------------------
set -eo pipefail
source /share/apps/NYUAD5/miniconda/3-4.11.0/etc/profile.d/conda.sh
conda activate /scratch/ni2071/conda_envs/squidiff
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
cd /scratch/ni2071/Diffusion/Squidiff_git/Squidiff_ms
mkdir -p logger_files tahoe_results/step28_logs
echo "[node] $(hostname)  [commit] $(git rev-parse --short HEAD)  [branch] $(git rev-parse --abbrev-ref HEAD)"
python -u gen28.py 2>&1 | tee tahoe_results/step28_logs/gen28_${SLURM_JOB_ID}.log
echo "[$(date)] DONE"
