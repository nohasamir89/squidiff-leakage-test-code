#!/bin/bash
#SBATCH --job-name=pvf_v2
#SBATCH --partition=compute
#SBATCH --cpus-per-task=16
#SBATCH --mem=50G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/ni2071/Diffusion/Squidiff_git/Squidiff_ms/logger_files/pathway_vs_fingerprint_v2_%A.out
#SBATCH --error=/scratch/ni2071/Diffusion/Squidiff_git/Squidiff_ms/logger_files/pathway_vs_fingerprint_v2_%A.err
# -----------------------------------------------------------------------------
# Purpose: run pathway_vs_fingerprint_v2.py (CPU only) in the manuscript
#          worktree. It re-creates the record behind Supplementary Note 1 with
#          the folds written down, and measures how much the numbers move over
#          200 random fold assignments.
# Logic:   same environment and thread settings as run_ridge_vs_mlp_v2.sh.
#          The script itself refuses to run if a tracked file is modified or if
#          its output exists, and writes nothing if one of its checks fails.
#          Any compute node will do: the folds no longer depend on the node.
# Output:  leakage_results/pathway_vs_fingerprint_v2.json (to be committed
#          together with the script once the result has been read).
# -----------------------------------------------------------------------------
set -eo pipefail
source /share/apps/NYUAD5/miniconda/3-4.11.0/etc/profile.d/conda.sh
conda activate /scratch/ni2071/conda_envs/squidiff
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16
cd /scratch/ni2071/Diffusion/Squidiff_git/Squidiff_ms
mkdir -p logger_files
echo "[node] $(hostname)  [commit] $(git rev-parse --short HEAD)  [branch] $(git rev-parse --abbrev-ref HEAD)"
python -B pathway_vs_fingerprint_v2.py
echo "[$(date)] DONE"
