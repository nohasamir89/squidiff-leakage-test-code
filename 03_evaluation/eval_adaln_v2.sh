#!/bin/bash
#SBATCH --job-name=sciplex_adaln_v2
#SBATCH --partition=nvidia
#SBATCH --gres=gpu:h100:1
#SBATCH -q a2s2
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=08:00:00
#SBATCH --array=0-4
#SBATCH --output=/scratch/ni2071/Diffusion/Squidiff_git/Squidiff/logger_files/adalnv2_%A_%a.out
#SBATCH --error=/scratch/ni2071/Diffusion/Squidiff_git/Squidiff/logger_files/adalnv2_%A_%a.err
# Re-evaluate the five semantic adaLN checkpoints under the PATCHED sampler, so
# amendment 13 compares arms evaluated under one code version. The stored
# pre-patch vectors differ from post-patch by 6.08e-06 on B; three post-patch
# replicates are bitwise identical, so that offset is a code effect, not noise.
set -eo pipefail
module load cuda/12.2.0
source /share/apps/NYUAD5/miniconda/3-4.11.0/etc/profile.d/conda.sh
conda activate /scratch/ni2071/conda_envs/squidiff
W=/scratch/ni2071/Diffusion/Squidiff_git/Squidiff
cd "$W"
[ -n "${SLURM_ARRAY_TASK_ID:-}" ] || { echo "FATAL: needs a SLURM array"; exit 1; }
S=$SLURM_ARRAY_TASK_ID
FROZEN=$(git rev-parse step4b-eval-freeze^{commit})
if ! git diff --quiet "$FROZEN" -- sample_squidiff.py leakage_test_base_sciplex_array.py Squidiff; then
  echo "FATAL: evaluation sources differ from step4b-eval-freeze"; exit 1
fi
echo "[freeze] evaluation sources match step4b-eval-freeze ($FROZEN)"
CKPT=$W/adaln/seed_${S}/model_0.9999.pt
TAG=adaln_v2_seed_${S}
[ -s "$CKPT" ] || { echo "FATAL: checkpoint missing: $CKPT"; exit 1; }
[ -e "$W/leakage_results/vectors_${TAG}.npz" ] && { echo "FATAL: output exists"; exit 1; }
export MASTER_PORT=$((13200 + S))
python eval_timeonly_pre.py "$CKPT" "$TAG" semantic
echo "[$(date)] adaLN v2 seed $S"
python leakage_test_base_sciplex_array.py drug_split_1 "$CKPT" "$TAG"
echo "[$(date)] adaLN v2 seed $S DONE  node=$(hostname)"
