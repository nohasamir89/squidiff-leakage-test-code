#!/bin/bash
#SBATCH --job-name=s14rescore
#SBATCH --partition=compute
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/ni2071/Diffusion/Squidiff_git/Squidiff_ms/logger_files/s14rescore_%A.out
#SBATCH --error=/scratch/ni2071/Diffusion/Squidiff_git/Squidiff_ms/logger_files/s14rescore_%A.err
# -----------------------------------------------------------------------------
# Purpose: re-run score_step14.py once, after the additive per-condition patch (patch_step14_add_percond.py), to
#          produce leakage_results/step14_per_condition.csv for the Step 28 sci-Plex re-aggregation. CPU only.
# Logic:   same environment the step was first scored in (squidiff). score_step14.py reloads the 614 stored
#          prediction-vector files and the sci-Plex test/control .h5ad files, verifies every provenance hash, and
#          rescores the three splits; it stops if any input is missing or a hash differs. The run overwrites
#          step14_long.csv and step14_analysis.json and newly writes step14_per_condition.csv.
#          GATE (checked after the run, outside this script): step14_long.csv must be BYTE-IDENTICAL to its
#          committed version - it has no commit field, so the additive patch must not change it. step14_analysis.json
#          legitimately differs only in its `commit` field and is NOT committed from this run (git checkout it).
#          The committed md5 of step14_long.csv is printed here before the run so the gate can be checked by eye.
# -----------------------------------------------------------------------------
set -eo pipefail
source /share/apps/NYUAD5/miniconda/3-4.11.0/etc/profile.d/conda.sh
conda activate /scratch/ni2071/conda_envs/squidiff
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
cd /scratch/ni2071/Diffusion/Squidiff_git/Squidiff_ms
mkdir -p logger_files tahoe_results/step28_logs
echo "[node] $(hostname)  [commit] $(git rev-parse --short HEAD)  [branch] $(git rev-parse --abbrev-ref HEAD)"
echo "[gate target] committed step14_long.csv md5: $(git show HEAD:leakage_results/step14_long.csv | md5sum | cut -d' ' -f1)"
python -u score_step14.py 2>&1 | tee tahoe_results/step28_logs/s14rescore_${SLURM_JOB_ID}.log
echo "[gate check]  new       step14_long.csv md5: $(md5sum leakage_results/step14_long.csv | cut -d' ' -f1)"
echo "[per-condition] rows: $(wc -l < leakage_results/step14_per_condition.csv)  file: leakage_results/step14_per_condition.csv"
echo "[$(date)] DONE"
