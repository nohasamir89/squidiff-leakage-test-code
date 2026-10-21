#!/bin/bash
#SBATCH --job-name=gbm_gene_sens
#SBATCH --output=/scratch/ni2071/Diffusion/gbm_squidiff_robustness_test/logs/sens_%A_%a.out
#SBATCH --error=/scratch/ni2071/Diffusion/gbm_squidiff_robustness_test/logs/sens_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH -p nvidia
#SBATCH --array=0-9                       # 10 gene counts, run in parallel

# ---------------------------------------------------------------------------
# GBM gene-count sensitivity — array (one gene count per task, parallel).
# PREREQUISITE: run prep_gbm_labels.py ONCE first (freezes labels + ranking).
#   python /scratch/ni2071/Diffusion/gbm_squidiff_robustness_test/prep_gbm_labels.py
# Each task then: per-N prep (loads frozen labels) -> train -> A/B/C eval.
# Tasks are independent; a failure in one N does not affect the others.
# Re-run one N:  sbatch --array=2 run_gbm_sensitivity.sbatch   (index 2 = N=300)
# Resources/flags below match your gbm_run_200 training sbatch exactly.
# ---------------------------------------------------------------------------

GENE_COUNTS=(100 200 300 400 500 600 700 800 900 1000)
N=${GENE_COUNTS[$SLURM_ARRAY_TASK_ID]}

echo "=== task ${SLURM_ARRAY_TASK_ID} -> N=${N}  start $(date) ==="
module load cuda/12.2.0
source /share/apps/NYUAD5/miniconda/3-4.11.0/etc/profile.d/conda.sh
conda activate /scratch/ni2071/conda_envs/squidiff

SQUIDIFF_DIR="/scratch/ni2071/Diffusion/Squidiff_gbm"
SCRIPTS="/scratch/ni2071/Diffusion/gbm_squidiff_robustness_test"
BASE="/scratch/ni2071/Diffusion/gbm_squidiff_robustness_test"
DATA_DIR="${BASE}/n${N}"
RUN_DIR="${BASE}/n${N}/run"
mkdir -p ${BASE}/logs ${RUN_DIR}/logs ${RUN_DIR}/results

# ---- 1) per-N prep: slice first N from the FROZEN labelled object ----
echo "--- prep N=${N} ---"
python ${SCRIPTS}/prep_gbm_sensitivity.py --n_hvg ${N} || { echo "PREP FAILED N=${N}"; exit 1; }

# ---- 2) train from scratch (your exact gbm_run_200 flags; GENE_SIZE=N) ----
# no-fingerprint Delta-z path: use_drug_structure is set False IN the package
# (Squidiff_gbm); do NOT pass it on the CLI (string "False" parses truthy).
echo "--- train N=${N} ---"
export MASTER_PORT=$((12355 + SLURM_ARRAY_TASK_ID))
cd ${SQUIDIFF_DIR}
python train_squidiff.py \
    --data_path ${DATA_DIR}/gbm_train.h5ad \
    --logger_path ${RUN_DIR}/logs \
    --resume_checkpoint ${RUN_DIR}/results \
    --gene_size ${N} \
    --output_dim ${N} \
    --batch_size 64 \
    --lr 0.0001 \
    --lr_anneal_steps 100000 \
    --save_interval 20000 \
    --log_interval 5000 \
    --diffusion_steps 1000 || { echo "TRAIN FAILED N=${N}"; exit 1; }

# ---- 3) A/B/C three-way eval; append one row per cell type to the shared CSV ----
echo "--- eval N=${N} ---"
python ${SCRIPTS}/eval_gbm_sensitivity.py \
    --n_hvg ${N} \
    --data_dir ${DATA_DIR} \
    --model ${RUN_DIR}/results/model_0.9999.pt || { echo "EVAL FAILED N=${N}"; exit 1; }

echo "=== DONE N=${N}  $(date) ==="