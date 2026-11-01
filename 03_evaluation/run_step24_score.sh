#!/bin/bash
#SBATCH --job-name=s24score
#SBATCH --partition=compute
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/ni2071/Diffusion/Squidiff_git/Squidiff_ms/logger_files/s24score_%A.out
#SBATCH --error=/scratch/ni2071/Diffusion/Squidiff_git/Squidiff_ms/logger_files/s24score_%A.err
# -----------------------------------------------------------------------------
# Purpose: score Step 24 (score_step24_tahoe.py, CPU only) once, on the development
#          split, in the manuscript worktree, after the seven evaluation runs.
# Logic:   same environment as the other CPU scorers; the working directory is the
#          step15-tahoe worktree, never the main checkout. The scorer refuses to run
#          unless the pre-registration, the scripts and the records are committed and
#          unchanged, every input matches its provenance hash and the reproduction
#          checks pass; it never overwrites an output. The full log is also written
#          into tahoe_results/step24_logs/ (by tee), so it can be committed with the
#          records.
# -----------------------------------------------------------------------------
set -eo pipefail
source /share/apps/NYUAD5/miniconda/3-4.11.0/etc/profile.d/conda.sh
conda activate /scratch/ni2071/conda_envs/squidiff
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
cd /scratch/ni2071/Diffusion/Squidiff_git/Squidiff_ms
mkdir -p logger_files tahoe_results/step24_logs
echo "[node] $(hostname)  [commit] $(git rev-parse --short HEAD)  [branch] $(git rev-parse --abbrev-ref HEAD)"
python -u score_step24_tahoe.py 2>&1 | tee tahoe_results/step24_logs/s24score_${SLURM_JOB_ID}.log
echo "[$(date)] DONE"
