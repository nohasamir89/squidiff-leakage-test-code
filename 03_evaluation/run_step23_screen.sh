#!/bin/bash
#SBATCH --job-name=s23screen
#SBATCH --partition=compute
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/ni2071/Diffusion/Squidiff_git/Squidiff_ms/logger_files/s23screen_%A.out
#SBATCH --error=/scratch/ni2071/Diffusion/Squidiff_git/Squidiff_ms/logger_files/s23screen_%A.err
# -----------------------------------------------------------------------------
# Purpose: run the Step 23 Stage 1 screen (score_step23_screen.py, CPU only) once,
#          on the development split, in the manuscript worktree.
# Logic:   same environment as the other CPU scorers; the working directory is the
#          step15-tahoe worktree, never the main checkout. The scorer refuses to run
#          unless the pre-registration, the scripts, the records and the feature
#          tables are committed and unchanged, and it never overwrites an output.
#          The full log is also written into tahoe_results/step23_logs/ (by tee),
#          so it can be committed with the records.
# -----------------------------------------------------------------------------
set -eo pipefail
source /share/apps/NYUAD5/miniconda/3-4.11.0/etc/profile.d/conda.sh
conda activate /scratch/ni2071/conda_envs/squidiff
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
cd /scratch/ni2071/Diffusion/Squidiff_git/Squidiff_ms
mkdir -p logger_files tahoe_results/step23_logs
echo "[node] $(hostname)  [commit] $(git rev-parse --short HEAD)  [branch] $(git rev-parse --abbrev-ref HEAD)"
python -u score_step23_screen.py 2>&1 | tee tahoe_results/step23_logs/s23screen_${SLURM_JOB_ID}.log
echo "[$(date)] DONE"
