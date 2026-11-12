#!/bin/bash
#SBATCH --job-name=testloss_base
#SBATCH --partition=condo
#SBATCH -q a2s2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/ni2071/Diffusion/Squidiff_git/Squidiff_ms/logger_files/testloss_base_%j.out
#SBATCH --error=/scratch/ni2071/Diffusion/Squidiff_git/Squidiff_ms/logger_files/testloss_base_%j.err
# -----------------------------------------------------------------------------
# Purpose: run compute_test_loss_base.py once on one GPU: the training objective of the ORIGINAL model (Step 4
#          replication checkpoints, sci-Plex splits 0 and 2, five seeds) on held-out and training cells, by the
#          procedure of compute_test_loss.py (f67c8ce). Exploratory; writes leakage_results/test_loss_base.json.
# Logic:   same environment as every sci-Plex evaluation (squidiff conda env, CUDA 12.2 module), run from the
#          Squidiff_ms worktree root so that the committed Squidiff package and sample_squidiff.py are imported.
#          The python script itself stops before writing anything if: the record already exists; it or
#          compute_test_loss.py is uncommitted; a checkpoint's sha256 differs from replication_analysis.json; a
#          checkpoint looks like an adaptive-normalisation model; or the reproduction gate (one committed value of
#          test_loss_v2.json recomputed within 2e-5) fails. Commit and md5 of both scripts are printed first.
# -----------------------------------------------------------------------------
set -eo pipefail
module load cuda/12.2.0
source /share/apps/NYUAD5/miniconda/3-4.11.0/etc/profile.d/conda.sh
conda activate /scratch/ni2071/conda_envs/squidiff
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
cd /scratch/ni2071/Diffusion/Squidiff_git/Squidiff_ms
mkdir -p logger_files
echo "[node] $(hostname)  [commit] $(git rev-parse --short HEAD)  [branch] $(git rev-parse --abbrev-ref HEAD)"
echo "[gpu] $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
md5sum compute_test_loss_base.py compute_test_loss.py
python -u compute_test_loss_base.py
echo "[$(date)] DONE"
