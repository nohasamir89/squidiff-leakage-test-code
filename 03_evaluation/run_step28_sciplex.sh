#!/bin/bash
#SBATCH --job-name=s28sciplex
#SBATCH --partition=compute
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/ni2071/Diffusion/Squidiff_git/Squidiff_ms/logger_files/s28sciplex_%A.out
#SBATCH --error=/scratch/ni2071/Diffusion/Squidiff_git/Squidiff_ms/logger_files/s28sciplex_%A.err
# -----------------------------------------------------------------------------
# Purpose: score Step 28 group B (score_step28_sciplex.py, CPU only) once: the sensitivity of the sci-Plex
#          gene-level results to the held-out tofacitinib form whose salt twin was in training
#          (PREREG_step28_salt_twins.md section 4 group B, amendments 15-16).
# Logic:   reads only committed records (step14_per_condition.csv, step14_analysis.json, the overlap record): no
#          .h5ad file, no checkpoint, no sampler. Runs on a compute node because the imported scorers pull in
#          anndata and scipy, which the login node reaps mid-import. The scorer refuses to run unless the
#          pre-registration, scripts and records are committed and clean, gate 1 reproduces every committed
#          sci-Plex value to 1e-9, and each affected split loses exactly 12 conditions; it never overwrites an
#          output. The full log is also written into tahoe_results/step28_logs/ (by tee) for committing.
# -----------------------------------------------------------------------------
set -eo pipefail
source /share/apps/NYUAD5/miniconda/3-4.11.0/etc/profile.d/conda.sh
conda activate /scratch/ni2071/conda_envs/squidiff
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
cd /scratch/ni2071/Diffusion/Squidiff_git/Squidiff_ms
mkdir -p logger_files tahoe_results/step28_logs
echo "[node] $(hostname)  [commit] $(git rev-parse --short HEAD)  [branch] $(git rev-parse --abbrev-ref HEAD)"
python -u score_step28_sciplex.py 2>&1 | tee tahoe_results/step28_logs/s28sciplex_${SLURM_JOB_ID}.log
echo "[$(date)] DONE"
