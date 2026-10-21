#!/bin/bash
#SBATCH --job-name=squidiff_gbm200
#SBATCH --output=/scratch/ni2071/Diffusion/gbm_run_200/logs/squidiff_200_%j.out
#SBATCH --error=/scratch/ni2071/Diffusion/gbm_run_200/logs/squidiff_200_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH -p nvidia
echo "GBM 200-gene ablation — Delta-z, no fingerprint"
echo "Job $SLURM_JOB_ID  Node $SLURM_NODELIST  Start $(date)"
module load cuda/12.2.0
source /share/apps/NYUAD5/miniconda/3-4.11.0/etc/profile.d/conda.sh
conda activate /scratch/ni2071/conda_envs/squidiff
SQUIDIFF_DIR="/scratch/ni2071/Diffusion/Squidiff_gbm"
DATA_DIR="/scratch/ni2071/Diffusion/datasets_gbm_paper_200"
RUN_DIR="/scratch/ni2071/Diffusion/gbm_run_200"
mkdir -p ${RUN_DIR}/logs ${RUN_DIR}/results
GENE_SIZE=200
echo "Genes: ${GENE_SIZE}  drug_structure OFF  n_cell_lines 3"
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
echo "Finished $(date)"