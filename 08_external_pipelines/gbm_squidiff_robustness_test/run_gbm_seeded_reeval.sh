#!/bin/bash
#SBATCH --job-name=gbm_seeded_reeval
#SBATCH --output=/scratch/ni2071/Diffusion/gbm_squidiff_robustness_test/logs/seeded_%A_%a.out
#SBATCH --error=/scratch/ni2071/Diffusion/gbm_squidiff_robustness_test/logs/seeded_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=128G
#SBATCH --gres=gpu:h100:1
#SBATCH --time=01:00:00
#SBATCH -p nvidia
#SBATCH --array=0-9

# ---------------------------------------------------------------------------
# SEEDED re-eval of the GBM gene-count sweep — EVAL ONLY (no retraining).
#
# Purpose: all 10 checkpoints are already trained. This re-runs ONLY the A/B/C
# eval with the decoder seed fixed (decode_mean seeds torch + cuda), producing a
# clean, reproducible sensitivity curve into a FRESH csv (gene_sensitivity_SEEDED.csv),
# leaving the earlier debugging csv untouched.
#
# PREREQUISITES (do these ONCE before submitting):
#   1) In eval_gbm_sensitivity.py, decode_mean is seeded:
#          def decode_mean(z, seed=42):
#              torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
#   2) In eval_gbm_sensitivity.py, RESULTS_CSV points at gene_sensitivity_SEEDED.csv
#   3) All n{N}/run/results/model_0.9999.pt checkpoints exist (they do).
#
# If condo is full / not submittable, fall back to your working partition:
#   change  -p condo  and  --gres=gpu:h100:1  to whatever `sinfo -p nvidia -o "%N %G"`
#   shows you can target (e.g. --gres=gpu:a100:1). The result is identical; only speed differs.
# ---------------------------------------------------------------------------

GENE_COUNTS=(100 200 300 400 500 600 700 800 900 1000)
N=${GENE_COUNTS[$SLURM_ARRAY_TASK_ID]}

echo "=== SEEDED re-eval task ${SLURM_ARRAY_TASK_ID} -> N=${N}  start $(date) ==="
module load cuda/12.2.0
source /share/apps/NYUAD5/miniconda/3-4.11.0/etc/profile.d/conda.sh
conda activate /scratch/ni2071/conda_envs/squidiff

# ---- provenance: record the exact GPU + node this result came from (for methods) ----
echo "--- hardware ---"
echo "node: ${SLURM_NODELIST}   partition: ${SLURM_JOB_PARTITION}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "----------------"

SQUIDIFF_DIR="/scratch/ni2071/Diffusion/Squidiff_gbm"
SCRIPTS="/scratch/ni2071/Diffusion/gbm_squidiff_robustness_test"
BASE="/scratch/ni2071/Diffusion/gbm_squidiff_robustness_test"
DATA_DIR="${BASE}/n${N}"
RUN_DIR="${BASE}/n${N}/run"

# guard: the checkpoint must already exist (this is eval-only)
CKPT="${RUN_DIR}/results/model_0.9999.pt"
if [ ! -f "${CKPT}" ]; then
  echo "MISSING CHECKPOINT for N=${N}: ${CKPT}"
  echo "This script does NOT train. Run training for N=${N} first."
  exit 1
fi

# guard: the per-N data split must already exist (frozen-label prep done earlier)
if [ ! -f "${DATA_DIR}/gbm_train.h5ad" ]; then
  echo "MISSING DATA for N=${N}: ${DATA_DIR}/gbm_train.h5ad"
  echo "Run prep_gbm_sensitivity.py --n_hvg ${N} first (needs the frozen labels)."
  exit 1
fi

echo "--- seeded eval N=${N} ---"
cd ${SQUIDIFF_DIR}
python ${SCRIPTS}/eval_gbm_sensitivity.py \
    --n_hvg ${N} \
    --data_dir ${DATA_DIR} \
    --model ${CKPT} || { echo "EVAL FAILED N=${N}"; exit 1; }

echo "=== DONE N=${N}  $(date) ==="