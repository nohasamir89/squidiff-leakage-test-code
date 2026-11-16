#!/bin/bash
#SBATCH --job-name=step29_tahoe
#SBATCH --partition=compute
#SBATCH --constraint=jubail
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/ni2071/Diffusion/Squidiff_git/Squidiff_ms/logger_files/step29_tahoe_%A.out
#SBATCH --error=/scratch/ni2071/Diffusion/Squidiff_git/Squidiff_ms/logger_files/step29_tahoe_%A.err
# -----------------------------------------------------------------------------
# Purpose: run score_step29_tahoe.py (CPU only) in the manuscript worktree.
#          Step 29 of PREREG_step29_fold_sensitivity.md, Tahoe-100M part: the
#          cross-validated ridge is refitted with the fold assignment of the paper
#          and with 200 random assignments, and every statement of the paper about
#          Tahoe-100M that compares something with it is counted (items B1 to B4
#          and C1 to C3).
# Logic:   The script first checks that nothing is modified and that it gives
#          back the committed values of the ridge to 1e-9 (gene identity, response
#          correlation, population distance, discrimination score, response-
#          magnitude agreement). It stops without writing if a check fails.
#          The folds of the paper are scored in the main process with 8 linear-
#          algebra threads, as the scoring jobs of the paper. The 600 random
#          assignments (200 on each split) are scored in 16 worker processes, one
#          thread each; the workers inherit the data of the main process.
#          --constraint=jubail: the standard compute nodes (AMD EPYC 7742), where
#          the records of the paper were written.
#          Memory: the measured and vehicle cells of three splits and 5,000
#          bootstrap resamples per split are held once and shared with the
#          workers; 100G leaves room for what the workers copy.
# Output:  tahoe_results/step29_fold_sensitivity_tahoe.json
#          tahoe_results/step29_fold_sensitivity_tahoe_per_assignment.csv
#          (written once, at the end; committed after the result has been read)
# -----------------------------------------------------------------------------
set -eo pipefail
source /share/apps/NYUAD5/miniconda/3-4.11.0/etc/profile.d/conda.sh
conda activate /scratch/ni2071/conda_envs/squidiff
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
cd /scratch/ni2071/Diffusion/Squidiff_git/Squidiff_ms
mkdir -p logger_files
echo "[node] $(hostname)  [commit] $(git rev-parse --short HEAD)  [branch] $(git rev-parse --abbrev-ref HEAD)"
python -B -u score_step29_tahoe.py --workers 16
echo "[$(date)] DONE"
