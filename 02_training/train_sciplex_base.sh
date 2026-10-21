#!/bin/bash
#SBATCH --job-name=sciplex_base
#SBATCH --partition=nvidia
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --array=0-5
#SBATCH --output=/scratch/ni2071/Diffusion/Squidiff_git/Squidiff/logger_files/base_%A_%a.out
#SBATCH --error=/scratch/ni2071/Diffusion/Squidiff_git/Squidiff/logger_files/base_%A_%a.err

# =============================================================================
# Train the PRISTINE authors' base Squidiff on all six CORRECTED sci-Plex splits.
#   array 0-2 -> drug_split_0/1/2   (unseen-drug)
#   array 3-5 -> random_split_0/1/2 (random)
#
# Base = authors' unmodified code (encoder uses control_feature, NO cell_id
# embedding). Trained on corrected (single-log1p) data. Drug structure ON and
# gene_size 200 are set explicitly because the script defaults are 100 / False.
# =============================================================================

module load cuda/12.2.0
source /share/apps/NYUAD5/miniconda/3-4.11.0/etc/profile.d/conda.sh
conda activate /scratch/ni2071/conda_envs/squidiff

# distinct port per array task (avoids the EADDRINUSE collision seen on GBM)
export MASTER_PORT=$((12355 + SLURM_ARRAY_TASK_ID))

SPLITS=(drug_split_0 drug_split_1 drug_split_2 random_split_0 random_split_1 random_split_2)
SPLIT=${SPLITS[$SLURM_ARRAY_TASK_ID]}

WORKDIR=/scratch/ni2071/Diffusion/Squidiff_git/Squidiff
DATADIR=/scratch/ni2071/Diffusion/sciplex/datasets_corrected
OUTDIR=$WORKDIR/results_base_corrected/base_${SPLIT}

set -eo pipefail                                   # release fix (21 Oct 2026): fail loudly, as every later script does
cd $WORKDIR
# Refuse to overwrite a finished baseline: the trainer saves unconditionally and never resumes,
# so a re-run would silently replace the checkpoints the Step 1 evaluations were made from.
if [ -e "$OUTDIR/model_0.9999.pt" ]; then
    echo "FATAL: $OUTDIR/model_0.9999.pt exists; move it aside before retraining" >&2
    exit 1
fi
mkdir -p $OUTDIR logger_files

echo "[$(date)] task $SLURM_ARRAY_TASK_ID  split=$SPLIT  port=$MASTER_PORT"

python train_squidiff.py \
    --logger_path logger_files/logger_base_${SPLIT} \
    --data_path $DATADIR/sci_plex_train_${SPLIT}.h5ad \
    --control_data_path $DATADIR/sci_plex_train_${SPLIT}_control.h5ad \
    --resume_checkpoint $OUTDIR \
    --use_drug_structure True \
    --gene_size 200 \
    --output_dim 200 \
    --seed 0                                       # explicit: this is the parser default every recorded run used

echo "[$(date)] task $SLURM_ARRAY_TASK_ID  split=$SPLIT  DONE"