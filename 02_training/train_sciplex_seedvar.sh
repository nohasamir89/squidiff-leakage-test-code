#!/bin/bash
#SBATCH --job-name=sciplex_seedvar
#SBATCH --partition=nvidia
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --array=0-4
#SBATCH --output=/scratch/ni2071/Diffusion/Squidiff_git/Squidiff/logger_files/seedvar_%A_%a.out
#SBATCH --error=/scratch/ni2071/Diffusion/Squidiff_git/Squidiff/logger_files/seedvar_%A_%a.err

# =============================================================================
# STEP 1 - training-seed noise floor.
# Five seeds, ONE fixed split (drug_split_1: best SNR 2.12x, 41.0% of conditions
# detectable above the vehicle-only permutation null). Everything except --seed
# is identical to train_sciplex_base.sh.
# GPU pinned to h100 so architecture is not an uncontrolled variable.
# =============================================================================

module load cuda/12.2.0
source /share/apps/NYUAD5/miniconda/3-4.11.0/etc/profile.d/conda.sh
conda activate /scratch/ni2071/conda_envs/squidiff

SEEDS=(0 1 2 3 4)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
SPLIT=drug_split_1

export MASTER_PORT=$((12370 + SLURM_ARRAY_TASK_ID))
export PYTHONHASHSEED=$SEED

WORKDIR=/scratch/ni2071/Diffusion/Squidiff_git/Squidiff
DATADIR=/scratch/ni2071/Diffusion/sciplex/datasets_corrected
OUTDIR=$WORKDIR/seedvar/seed_${SEED}

cd $WORKDIR || { echo "FATAL: cannot cd to $WORKDIR"; exit 1; }
mkdir -p $OUTDIR logger_files seedvar

# --- abort if the output dir already holds a checkpoint ---------------------
# --resume_checkpoint points at OUTDIR, so a stale model_*.pt would make this
# run resume from existing weights instead of training fresh. Five runs would
# then share a starting point and the noise floor would be spuriously tight -
# which looks like an excellent result, not a bug.
if ls $OUTDIR/*.pt >/dev/null 2>&1; then
  echo "FATAL: $OUTDIR already contains checkpoints. Refusing to run."
  exit 1
fi

# --- provenance -------------------------------------------------------------
python -c "from Squidiff import MLPModel; print('[import] ' + MLPModel.__file__)"
python - <<PY > $OUTDIR/provenance.json
import json, subprocess, os
r = lambda c: subprocess.run(c, shell=True, capture_output=True, text=True).stdout.strip()
print(json.dumps({
    'commit': r('git rev-parse HEAD'),
    'branch': r('git rev-parse --abbrev-ref HEAD'),
    'dirty': r('git status --porcelain'),
    'seed': os.environ.get('PYTHONHASHSEED'),
    'split': '${SPLIT}',
    'slurm_job': os.environ.get('SLURM_JOB_ID'),
    'gpu': r('nvidia-smi --query-gpu=name --format=csv,noheader'),
}, indent=2))
PY
cat $OUTDIR/provenance.json

echo "[$(date)] task $SLURM_ARRAY_TASK_ID  seed=$SEED  split=$SPLIT  port=$MASTER_PORT"

python train_squidiff.py \
    --logger_path logger_files/logger_seedvar_seed_${SEED} \
    --data_path $DATADIR/sci_plex_train_${SPLIT}.h5ad \
    --control_data_path $DATADIR/sci_plex_train_${SPLIT}_control.h5ad \
    --resume_checkpoint $OUTDIR \
    --seed $SEED \
    --use_drug_structure True \
    --gene_size 200 \
    --output_dim 200

echo "[$(date)] task $SLURM_ARRAY_TASK_ID  seed=$SEED  DONE"
