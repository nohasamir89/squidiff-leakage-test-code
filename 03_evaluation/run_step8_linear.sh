#!/bin/bash
#SBATCH --job-name=s8_linear
#SBATCH --partition=nvidia
#SBATCH --gres=gpu:h100:1
#SBATCH -q a2s2
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/ni2071/Diffusion/Squidiff_git/Squidiff/logger_files/s8lin_%A.out
#SBATCH --error=/scratch/ni2071/Diffusion/Squidiff_git/Squidiff/logger_files/s8lin_%A.err
set -eo pipefail
module load cuda/12.2.0
source /share/apps/NYUAD5/miniconda/3-4.11.0/etc/profile.d/conda.sh
conda activate /scratch/ni2071/conda_envs/squidiff
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /scratch/ni2071/Diffusion/Squidiff_git/Squidiff
echo "[node] $(hostname)  [commit] $(git rev-parse --short HEAD)"
python step8_linear_check.py
echo "[$(date)] DONE"
