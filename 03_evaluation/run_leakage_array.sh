#!/bin/bash
#SBATCH --job-name=sciplex_leak_arr
#SBATCH --partition=nvidia
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --array=0-5
#SBATCH --output=/scratch/ni2071/Diffusion/Squidiff_git/Squidiff/logger_files/leak_arr_%A_%a.out
#SBATCH --error=/scratch/ni2071/Diffusion/Squidiff_git/Squidiff/logger_files/leak_arr_%A_%a.err
#SBATCH --mail-type=END

module load cuda/12.2.0
source /share/apps/NYUAD5/miniconda/3-4.11.0/etc/profile.d/conda.sh
conda activate /scratch/ni2071/conda_envs/squidiff

export MASTER_PORT=$((12360 + SLURM_ARRAY_TASK_ID))

SPLITS=(drug_split_0 drug_split_1 drug_split_2 random_split_0 random_split_1 random_split_2)
SPLIT=${SPLITS[$SLURM_ARRAY_TASK_ID]}

set -eo pipefail
cd /scratch/ni2071/Diffusion/Squidiff_git/Squidiff

# Release fix (21 Oct 2026): the harness writes leakage_results/leakage_<TAG>.json and
# vectors_<TAG>.npz unconditionally; refuse to overwrite a finished evaluation.
if [ -e "leakage_results/vectors_${SPLIT}.npz" ]; then
    echo "FATAL: leakage_results/vectors_${SPLIT}.npz exists; move it aside to re-evaluate" >&2
    exit 1
fi

echo "[$(date)] task $SLURM_ARRAY_TASK_ID  split=$SPLIT"
python leakage_test_base_sciplex_array.py $SPLIT
echo "[$(date)] task $SLURM_ARRAY_TASK_ID  split=$SPLIT  done"
