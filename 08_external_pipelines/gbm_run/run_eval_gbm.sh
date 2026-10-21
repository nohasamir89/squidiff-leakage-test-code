#!/bin/bash
#SBATCH --job-name=gbm_eval
#SBATCH --output=/scratch/ni2071/Diffusion/gbm_run/logs/eval_%j.out
#SBATCH --error=/scratch/ni2071/Diffusion/gbm_run/logs/eval_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH -p nvidia
#SBATCH --mail-type=END
module load cuda/12.2.0
source /share/apps/NYUAD5/miniconda/3-4.11.0/etc/profile.d/conda.sh
conda activate /scratch/ni2071/conda_envs/squidiff
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start: $(date)"
echo "=== checkpoint check ==="
ls -la /scratch/ni2071/Diffusion/gbm_run/results/model_0.9999.pt
echo "=== running Delta-z eval (Fig 3f reproduction) ==="
python /scratch/ni2071/Diffusion/gbm_run/eval_gbm_deltaz.py
echo "Finished: $(date)"