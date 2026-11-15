#!/bin/bash
#SBATCH --job-name=step29_sciplex
#SBATCH --partition=compute
#SBATCH --constraint=jubail
#SBATCH --cpus-per-task=16
#SBATCH --mem=50G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/ni2071/Diffusion/Squidiff_git/Squidiff_ms/logger_files/step29_sciplex_%A.out
#SBATCH --error=/scratch/ni2071/Diffusion/Squidiff_git/Squidiff_ms/logger_files/step29_sciplex_%A.err
# -----------------------------------------------------------------------------
# Purpose: run score_step29_sciplex.py (CPU only) in the manuscript worktree.
#          Step 29 of the pre-registration PREREG_step29_fold_sensitivity.md,
#          sci-Plex part: the cross-validated ridge is refitted with the fold
#          assignment of the paper and with 200 random assignments, and every
#          statement of the paper that compares something with it is counted.
# Logic:   The script first checks that it gives back committed values, the
#          per-condition values of the ridge to 1e-9. Those were written by
#          run_step14_rescore.sh with 8 linear-algebra threads. A last digit of
#          the ridge's arithmetic depends on the number of threads and on the
#          processor type, so this job uses the same: 8 threads, and
#          --constraint=jubail, the standard compute nodes (cn, AMD EPYC 7742,
#          no AVX-512), where the two records whose node is known were written
#          (ridge_vs_mlp_v2.json on cn100, pathway_vs_fingerprint_v2.json on
#          cn068). The 16 CPUs are for the last part: the networks of 50 random
#          assignments, trained side by side, one thread each. The network
#          record itself was written with 16 threads; the script sets the
#          training library to 16 for that check itself (NET_THREADS).
#          Time: all checks first (about ten minutes; the script stops without
#          writing if one fails), then ridge and intervals for the 200
#          assignments (about 20 minutes), then the networks (20 to 40 minutes).
# Output:  leakage_results/step29_fold_sensitivity_sciplex.json
#          leakage_results/step29_fold_sensitivity_sciplex_per_assignment.csv
#          (written once, at the end; committed after the result has been read)
# -----------------------------------------------------------------------------
set -eo pipefail
source /share/apps/NYUAD5/miniconda/3-4.11.0/etc/profile.d/conda.sh
conda activate /scratch/ni2071/conda_envs/squidiff
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
cd /scratch/ni2071/Diffusion/Squidiff_git/Squidiff_ms
mkdir -p logger_files
echo "[node] $(hostname)  [commit] $(git rev-parse --short HEAD)  [branch] $(git rev-parse --abbrev-ref HEAD)"
# for the record only: the node(s) of the job that wrote the per-condition record (its log files, if still there)
echo "[records] step14_per_condition.csv was written by: $(grep -h '^\[node\]' logger_files/s14rescore_*.out 2>/dev/null | tr '\n' ' ')"
python -B -u score_step29_sciplex.py
echo "[$(date)] DONE"
