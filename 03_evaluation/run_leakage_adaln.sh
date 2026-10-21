#!/bin/bash
#SBATCH --job-name=sciplex_leak_adaln
#SBATCH --partition=nvidia
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --array=0-4
#SBATCH --output=/scratch/ni2071/Diffusion/Squidiff_git/Squidiff/logger_files/leak_adaln_%A_%a.out
#SBATCH --error=/scratch/ni2071/Diffusion/Squidiff_git/Squidiff/logger_files/leak_adaln_%A_%a.err

# STEP 1 evaluation: A/B/C leakage test on the five seed models.
# Split fixed at drug_split_1; only the checkpoint varies.
# Same harness as the six-split run: seeded decode, no r2 clamps, vectors saved.

set -eo pipefail
module load cuda/12.2.0
source /share/apps/NYUAD5/miniconda/3-4.11.0/etc/profile.d/conda.sh
conda activate /scratch/ni2071/conda_envs/squidiff

export MASTER_PORT=$((12410 + SLURM_ARRAY_TASK_ID))

SEEDS=(0 1 2 3 4)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
SPLIT=drug_split_1

WORKDIR=/scratch/ni2071/Diffusion/Squidiff_git/Squidiff
CKPT=$WORKDIR/adaln/seed_${SEED}/model_0.9999.pt

cd $WORKDIR || { echo "FATAL: cannot cd to $WORKDIR"; exit 1; }

if [ ! -f "$CKPT" ]; then
  echo "FATAL: checkpoint missing: $CKPT"
  exit 1
fi

# guard on training completion, not checkpoint existence.
# save_interval overwrites model_0.9999.pt every 10k steps, so the file is
# full-size and loadable long before training ends. An intermediate checkpoint
# would evaluate cleanly and give a wrong noise floor.
if ! grep -qE "seed=$SEED[[:space:]].*DONE" $WORKDIR/logger_files/adaln_*.out 2>/dev/null; then
  echo "FATAL: no DONE line for seed $SEED - training incomplete"
  exit 1
fi

# Release fix (21 Oct 2026): the harness writes leakage_results/leakage_<TAG>.json and
# vectors_<TAG>.npz unconditionally; refuse to overwrite a finished evaluation.
if [ -e "leakage_results/vectors_adaln_seed_${SEED}.npz" ]; then
    echo "FATAL: leakage_results/vectors_adaln_seed_${SEED}.npz exists; move it aside to re-evaluate" >&2
    exit 1
fi

echo "[$(date)] task $SLURM_ARRAY_TASK_ID  seed=$SEED  split=$SPLIT"
echo "  checkpoint: $CKPT  ($(stat -c%s $CKPT) bytes)"
python -c "from Squidiff import MLPModel; print('[import] ' + MLPModel.__file__)"

python leakage_test_base_sciplex_array.py $SPLIT $CKPT adaln_seed_${SEED}

echo "[$(date)] task $SLURM_ARRAY_TASK_ID  seed=$SEED  done"
