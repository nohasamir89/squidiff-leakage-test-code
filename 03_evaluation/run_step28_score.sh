#!/bin/bash
#SBATCH --job-name=s28score
#SBATCH --partition=compute
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/ni2071/Diffusion/Squidiff_git/Squidiff_ms/logger_files/s28score_%A.out
#SBATCH --error=/scratch/ni2071/Diffusion/Squidiff_git/Squidiff_ms/logger_files/s28score_%A.err
# -----------------------------------------------------------------------------
# Purpose: score Step 28 (score_step28_salt_twins.py, CPU only) once, on all three splits - the sensitivity of the
#          reported Tahoe-100M results to held-out compounds whose salt or hydrate twin was in training
#          (PREREG_step28_salt_twins.md) - after Step 27 is scored and its records and findings are committed.
# Logic:   same environment and worktree as the other CPU scorers (never the main checkout). The step reads only
#          committed per-condition CSV records: no .h5ad file, no checkpoint, no sampler, so 16 GB and one hour are
#          generous (the work is a few hundred bootstraps of 5,000 replicates over ~3,000 conditions). The scorer
#          refuses to run unless the pre-registration, the scripts and every record it reads are committed and
#          unchanged, its two reproduction gates pass, and no Step 28 output exists; it never overwrites an output.
#          The full log is also written into tahoe_results/step28_logs/ (by tee) for committing.
# -----------------------------------------------------------------------------
set -eo pipefail
source /share/apps/NYUAD5/miniconda/3-4.11.0/etc/profile.d/conda.sh
conda activate /scratch/ni2071/conda_envs/squidiff
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
cd /scratch/ni2071/Diffusion/Squidiff_git/Squidiff_ms
mkdir -p logger_files tahoe_results/step28_logs
echo "[node] $(hostname)  [commit] $(git rev-parse --short HEAD)  [branch] $(git rev-parse --abbrev-ref HEAD)"
python -u score_step28_salt_twins.py 2>&1 | tee tahoe_results/step28_logs/s28score_${SLURM_JOB_ID}.log
echo "[$(date)] DONE"
