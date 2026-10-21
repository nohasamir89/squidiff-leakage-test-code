#!/bin/bash
#SBATCH --job-name=squidiff_gbm
#SBATCH --output=/scratch/ni2071/Diffusion/gbm_run/logs/train_squidiff_%j.out
#SBATCH --error=/scratch/ni2071/Diffusion/gbm_run/logs/train_squidiff_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH -p nvidia
#SBATCH --mail-type=END
echo "========================================"
echo "GBM reproduction (Fig 3f) — Squidiff, Delta-z task, NO fingerprint"
echo "Job ID: $SLURM_JOB_ID  Node: $SLURM_NODELIST  Start: $(date)"
echo "========================================"
module load cuda/12.2.0
source /share/apps/NYUAD5/miniconda/3-4.11.0/etc/profile.d/conda.sh
conda activate /scratch/ni2071/conda_envs/squidiff
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

SQUIDIFF_DIR="/scratch/ni2071/Diffusion/Squidiff_gbm"
DATA_DIR="/scratch/ni2071/Diffusion/datasets_gbm_paper"
RUN_DIR="/scratch/ni2071/Diffusion/gbm_run"
mkdir -p ${RUN_DIR}/logs ${RUN_DIR}/results

# gene_size MUST match the prepared data (2000 HVGs unless N_HVG changed).
# Confirm with: python -c "import scanpy;print(scanpy.read_h5ad('${DATA_DIR}/gbm_train.h5ad').n_vars)"
GENE_SIZE=2000

echo "Training Squidiff on GBM (PW030) — Delta-z task"
echo "  Data:  ${DATA_DIR}/gbm_train.h5ad"
echo "  Genes: ${GENE_SIZE}   drug_structure: OFF   num_classes: None"
echo "========================================"
cd ${SQUIDIFF_DIR}
python train_squidiff.py \
    --data_path ${DATA_DIR}/gbm_train.h5ad \
    --logger_path ${RUN_DIR}/logs \
    --resume_checkpoint ${RUN_DIR}/results \
    --gene_size ${GENE_SIZE} \
    --output_dim ${GENE_SIZE} \
    --batch_size 64 \
    --lr 0.0001 \
    --lr_anneal_steps 100000 \
    --save_interval 20000 \
    --log_interval 5000 \
    --diffusion_steps 1000
echo "========================================"
echo "Finished at: $(date)"