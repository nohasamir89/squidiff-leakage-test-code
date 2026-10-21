#!/bin/bash
#SBATCH --job-name=prep_gbm
#SBATCH --output=/scratch/ni2071/Diffusion/gbm_run/logs/prep_gbm_%j.out
#SBATCH --error=/scratch/ni2071/Diffusion/gbm_run/logs/prep_gbm_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
##SBATCH --partition=bigmem
#SBATCH --mem=64G
#SBATCH --time=01:00:00
##SBATCH -p nvidia
module load cuda/12.2.0
source /share/apps/NYUAD5/miniconda/3-4.11.0/etc/profile.d/conda.sh
conda activate /scratch/ni2071/conda_envs/squidiff
echo "Start: $(date)"
cd /scratch/ni2071/Diffusion/datasets_gbm_paper

echo "=== preprocessing (with HVG subset) ==="
python prep_gbm_paper.py

echo "=== adding obs columns ==="
python add_gbm_obs.py

echo "=== final gene count check ==="
python -c "import scanpy; print('train genes:', scanpy.read_h5ad('gbm_train.h5ad').n_vars)"
echo "Finished: $(date)"