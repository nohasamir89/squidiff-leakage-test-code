#!/bin/bash
#SBATCH --job-name=s26score
#SBATCH --partition=compute
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/ni2071/Diffusion/Squidiff_git/Squidiff_ms/logger_files/s26score_%A.out
#SBATCH --error=/scratch/ni2071/Diffusion/Squidiff_git/Squidiff_ms/logger_files/s26score_%A.err
# -----------------------------------------------------------------------------
# Purpose: score Step 26 (score_step26_tahoe.py, CPU only) once, on all three
#          splits - the source attribution of the compound-free discrimination
#          (PREREG_step26_control_source.md) - after Step 25 is scored and its
#          records and findings are committed.
# Logic:   same environment as the other CPU scorers; the working directory is the
#          step15-tahoe worktree, never the main checkout. The scorer refuses to run
#          unless the pre-registration, the scripts and every record it reads
#          (including the Step 24/25 swap records and findings) are committed and
#          unchanged, the data files hash-match the Step 15 data record, the stored
#          true and C are rebuilt from those files to 1e-4, every stored real cell
#          is recovered bitwise, and the S-orig rescoring reproduces the committed
#          per-condition records to 1e-9; it never overwrites an output. 96 GB
#          covers the densified test and control matrices of one split at a time
#          (they are released before the next split). The full log is also written
#          into tahoe_results/step26_logs/ (by tee) so it can be committed with the
#          records.
# -----------------------------------------------------------------------------
set -eo pipefail
source /share/apps/NYUAD5/miniconda/3-4.11.0/etc/profile.d/conda.sh
conda activate /scratch/ni2071/conda_envs/squidiff
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
cd /scratch/ni2071/Diffusion/Squidiff_git/Squidiff_ms
mkdir -p logger_files tahoe_results/step26_logs
echo "[node] $(hostname)  [commit] $(git rev-parse --short HEAD)  [branch] $(git rev-parse --abbrev-ref HEAD)"
python -u score_step26_tahoe.py 2>&1 | tee tahoe_results/step26_logs/s26score_${SLURM_JOB_ID}.log
echo "[$(date)] DONE"
