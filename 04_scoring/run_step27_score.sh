#!/bin/bash
#SBATCH --job-name=s27score
#SBATCH --partition=compute
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/ni2071/Diffusion/Squidiff_git/Squidiff_ms/logger_files/s27score_%A.out
#SBATCH --error=/scratch/ni2071/Diffusion/Squidiff_git/Squidiff_ms/logger_files/s27score_%A.err
# -----------------------------------------------------------------------------
# Purpose: score Step 27 (score_step27_tahoe.py, CPU only) once, on all three
#          splits - within-plate PDS with intervals for every Step 26 arm and the
#          plate-residual test (PREREG_step27_within_plate.md) - after Step 26 is
#          scored and its records and findings are committed.
# Logic:   same environment and worktree as the other CPU scorers (never the main
#          checkout). The scorer refuses to run unless the pre-registration, the
#          scripts and every record it reads (including the Step 26 records and
#          findings) are committed and unchanged, the data files hash-match the
#          Step 15 data record, the stored true and C are rebuilt to 1e-4, every
#          stored real cell is recovered bitwise, and every arm's cross-plate PDS
#          reproduces the Step 26 per-condition record to 1e-9; it never overwrites
#          an output. 96 GB covers the densified test and control matrices of one
#          split at a time, as in Step 26 (peak 7.7 GB there). The full log is also
#          written into tahoe_results/step27_logs/ (by tee) for committing.
# -----------------------------------------------------------------------------
set -eo pipefail
source /share/apps/NYUAD5/miniconda/3-4.11.0/etc/profile.d/conda.sh
conda activate /scratch/ni2071/conda_envs/squidiff
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
cd /scratch/ni2071/Diffusion/Squidiff_git/Squidiff_ms
mkdir -p logger_files tahoe_results/step27_logs
echo "[node] $(hostname)  [commit] $(git rev-parse --short HEAD)  [branch] $(git rev-parse --abbrev-ref HEAD)"
python -u score_step27_tahoe.py 2>&1 | tee tahoe_results/step27_logs/s27score_${SLURM_JOB_ID}.log
echo "[$(date)] DONE"
