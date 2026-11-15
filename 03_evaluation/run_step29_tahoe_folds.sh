#!/bin/bash
#SBATCH --job-name=step29_tfolds
#SBATCH --partition=compute
#SBATCH --constraint=jubail
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/ni2071/Diffusion/Squidiff_git/Squidiff_ms/logger_files/step29_tfolds_%A.out
#SBATCH --error=/scratch/ni2071/Diffusion/Squidiff_git/Squidiff_ms/logger_files/step29_tfolds_%A.err
# -----------------------------------------------------------------------------
# Purpose: run diag_step29_tahoe_folds.py (CPU only, read-only) in the manuscript
#          worktree. Step 29, guard 4 of PREREG_step29_fold_sensitivity.md: find
#          the fold assignment that the cross-validated ridge of the paper used on
#          Tahoe-100M, by trying the known orders of ties and keeping the one that
#          reproduces the committed per-condition record to 1e-9.
# Logic:   The script checks that nothing is modified and that it is committed,
#          then, for each of the three splits, refits the ridge once per candidate
#          partition and compares with the record. It writes
#          tahoe_results/step29_tahoe_folds.json only if every split has a
#          candidate that reproduces the record; otherwise it writes nothing.
#          Which candidates exist depends on the processor: on a node without
#          AVX-512 (the standard compute nodes, --constraint=jubail) two orders are
#          tried; on a node with AVX-512 a third one is added. The file can also
#          be run inside an interactive session on a compute node with
#              bash run_step29_tahoe_folds.sh
#          (the #SBATCH lines are then plain comments; output goes to the screen).
#          8 linear-algebra threads, as the scoring jobs of the paper.
# Output:  tahoe_results/step29_tahoe_folds.json (written once; committed after
#          it has been read)
# -----------------------------------------------------------------------------
set -eo pipefail
source /share/apps/NYUAD5/miniconda/3-4.11.0/etc/profile.d/conda.sh
conda activate /scratch/ni2071/conda_envs/squidiff
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
cd /scratch/ni2071/Diffusion/Squidiff_git/Squidiff_ms
mkdir -p logger_files
echo "[node] $(hostname)  [commit] $(git rev-parse --short HEAD)  [branch] $(git rev-parse --abbrev-ref HEAD)"
python -B -u diag_step29_tahoe_folds.py
echo "[$(date)] DONE"
