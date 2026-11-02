#!/bin/bash
#SBATCH --job-name=s25score
#SBATCH --partition=compute
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/ni2071/Diffusion/Squidiff_git/Squidiff_ms/logger_files/s25score_%A.out
#SBATCH --error=/scratch/ni2071/Diffusion/Squidiff_git/Squidiff_ms/logger_files/s25score_%A.err
# -----------------------------------------------------------------------------
# Purpose: score Step 25 (score_step25_tahoe.py, CPU only) once, on all three
#          splits and both checkpoint arms, in the manuscript worktree, after
#          the 35 evaluation runs.
# Logic:   same environment as the other CPU scorers; the working directory is the
#          step15-tahoe worktree, never the main checkout. The scorer refuses to run
#          unless the pre-registration, the scripts and the records are committed and
#          unchanged, every input matches its provenance hash, each split's and arm's
#          reproduction run is bitwise identical to Step 15b, and each own arm
#          reproduces the committed per-condition records to 1e-9; it never
#          overwrites an output. Per-cell arrays are released as each run is scored,
#          so peak memory stays near two runs rather than eighteen. The full log is
#          also written into tahoe_results/step25_logs/ (by tee) so it can be
#          committed with the records.
# -----------------------------------------------------------------------------
set -eo pipefail
source /share/apps/NYUAD5/miniconda/3-4.11.0/etc/profile.d/conda.sh
conda activate /scratch/ni2071/conda_envs/squidiff
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
cd /scratch/ni2071/Diffusion/Squidiff_git/Squidiff_ms
mkdir -p logger_files tahoe_results/step25_logs
echo "[node] $(hostname)  [commit] $(git rev-parse --short HEAD)  [branch] $(git rev-parse --abbrev-ref HEAD)"
python -u score_step25_tahoe.py 2>&1 | tee tahoe_results/step25_logs/s25score_${SLURM_JOB_ID}.log
echo "[$(date)] DONE"
