#!/bin/bash
#SBATCH --job-name=ridge_vs_mlp_v2
#SBATCH --partition=compute
#SBATCH --cpus-per-task=16
#SBATCH --mem=50G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/ni2071/Diffusion/Squidiff_git/Squidiff_ms/logger_files/ridge_vs_mlp_v2_%A.out
#SBATCH --error=/scratch/ni2071/Diffusion/Squidiff_git/Squidiff_ms/logger_files/ridge_vs_mlp_v2_%A.err
# -----------------------------------------------------------------------------
# Purpose: run ridge_vs_mlp_v2.py (CPU only) in the manuscript worktree and
#          leave the new record leakage_results/ridge_vs_mlp_v2.json for commit.
# Logic:   same environment as run_step8_linear_cpu.sh; the working directory is
#          the step15-tahoe worktree, not the main checkout, so a VCC job in the
#          main tree is never touched. The script itself refuses a dirty tree and
#          an existing output, and stops if the ridge does not reproduce Step 7.
# -----------------------------------------------------------------------------
set -eo pipefail
source /share/apps/NYUAD5/miniconda/3-4.11.0/etc/profile.d/conda.sh
conda activate /scratch/ni2071/conda_envs/squidiff
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16
cd /scratch/ni2071/Diffusion/Squidiff_git/Squidiff_ms
mkdir -p logger_files
echo "[node] $(hostname)  [commit] $(git rev-parse --short HEAD)  [branch] $(git rev-parse --abbrev-ref HEAD)"
python ridge_vs_mlp_v2.py
echo "[$(date)] DONE"
