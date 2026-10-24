#!/bin/bash
#SBATCH --job-name=gbm_eval_v2
#SBATCH --output=/scratch/ni2071/Diffusion/gbm_run/logs/eval_v2_%j.out
#SBATCH --error=/scratch/ni2071/Diffusion/gbm_run/logs/eval_v2_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH -p nvidia
# Purpose: re-run the Fig. 1e reproduction (gbm_run checkpoint) with eval_gbm_deltaz_v2.py,
#          predictions compared on the log1p scale without a second log1p. Eval only.
set -eo pipefail
module load cuda/12.2.0
source /share/apps/NYUAD5/miniconda/3-4.11.0/etc/profile.d/conda.sh
conda activate /scratch/ni2071/conda_envs/squidiff
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)  Start: $(date)"
md5sum /scratch/ni2071/Diffusion/gbm_run/eval_gbm_deltaz_v2.py /scratch/ni2071/Diffusion/gbm_run/results/model_0.9999.pt
python /scratch/ni2071/Diffusion/gbm_run/eval_gbm_deltaz_v2.py
echo "Finished: $(date)"
