#!/bin/bash
#SBATCH --job-name=gbm_reeval_v2
#SBATCH --output=/scratch/ni2071/Diffusion/gbm_squidiff_robustness_test/logs/v2_%A_%a.out
#SBATCH --error=/scratch/ni2071/Diffusion/gbm_squidiff_robustness_test/logs/v2_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=128G
#SBATCH --gres=gpu:h100:1
#SBATCH --time=01:00:00
#SBATCH -p nvidia
#SBATCH --array=0-9
# ---------------------------------------------------------------------------
# Purpose: re-evaluate the ten GBM gene-count checkpoints with eval_gbm_sensitivity_v2.py,
#          which compares predictions and truth on the same (log1p) scale. Eval only.
# Logic:   identical to run_gbm_seeded_reeval.sh except the script name and the log names;
#          writes gene_sensitivity_SEEDED_v2.csv; the old csv and logs are untouched.
# ---------------------------------------------------------------------------
set -eo pipefail
GENE_COUNTS=(100 200 300 400 500 600 700 800 900 1000)
N=${GENE_COUNTS[$SLURM_ARRAY_TASK_ID]}
echo "=== v2 re-eval task ${SLURM_ARRAY_TASK_ID} -> N=${N}  start $(date) ==="
module load cuda/12.2.0
source /share/apps/NYUAD5/miniconda/3-4.11.0/etc/profile.d/conda.sh
conda activate /scratch/ni2071/conda_envs/squidiff
echo "node: ${SLURM_NODELIST}  gpu: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
SQUIDIFF_DIR=/scratch/ni2071/Diffusion/Squidiff_gbm
BASE=/scratch/ni2071/Diffusion/gbm_squidiff_robustness_test
CKPT=${BASE}/n${N}/run/results/model_0.9999.pt
[ -f "${CKPT}" ] || { echo "MISSING CHECKPOINT ${CKPT}"; exit 1; }
[ -f "${BASE}/n${N}/gbm_train.h5ad" ] || { echo "MISSING DATA for N=${N}"; exit 1; }
md5sum ${BASE}/eval_gbm_sensitivity_v2.py ${CKPT}
cd ${SQUIDIFF_DIR}
python ${BASE}/eval_gbm_sensitivity_v2.py --n_hvg ${N} --data_dir ${BASE}/n${N} --model ${CKPT}
echo "=== DONE N=${N}  $(date) ==="
