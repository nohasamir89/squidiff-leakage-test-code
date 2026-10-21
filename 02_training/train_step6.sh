#!/bin/bash
#SBATCH --job-name=sciplex_step6
#SBATCH --partition=nvidia
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=24:00:00
#SBATCH --array=0-14
#SBATCH --output=/scratch/ni2071/Diffusion/Squidiff_git/Squidiff/logger_files/step6_%A_%a.out
#SBATCH --error=/scratch/ni2071/Diffusion/Squidiff_git/Squidiff/logger_files/step6_%A_%a.err
# Step 6 under PREREG_step6_flow_matching.md. Three arms x five seeds, one commit.
# 0-4 A=ddim | 5-9 B=fm_gaussian | 10-14 C=fm_control_ot
# The three arms are ARCHITECTURALLY IDENTICAL (58,178,180 params, same keys),
# so arm identity is NOT recoverable from the checkpoint. A sidecar records it,
# as in Step 4b.
set -eo pipefail
module load cuda/12.2.0
source /share/apps/NYUAD5/miniconda/3-4.11.0/etc/profile.d/conda.sh
conda activate /scratch/ni2071/conda_envs/squidiff
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
W=/scratch/ni2071/Diffusion/Squidiff_git/Squidiff
D=/scratch/ni2071/Diffusion/sciplex/datasets_corrected
SPLIT=drug_split_1
cd "$W"

[ -n "${SLURM_ARRAY_TASK_ID:-}" ] || { echo "FATAL: needs a SLURM array"; exit 1; }
T=$SLURM_ARRAY_TASK_ID
case "$T" in *[!0-9]*|"") echo "FATAL: non-numeric task id: $T"; exit 1 ;; esac
if [ "$T" -lt 0 ] || [ "$T" -gt 14 ]; then echo "FATAL: task out of range: $T"; exit 1; fi

N=$(grep -c '^## ' PREREG_step6_flow_matching.md)
[ "$N" -eq 9 ] || { echo "FATAL: prereg has $N sections, expected 9"; exit 1; }
git diff --quiet -- PREREG_step6_flow_matching.md || { echo "FATAL: prereg uncommitted"; exit 1; }
[ -n "$(git status --porcelain -- train_squidiff.py Squidiff)" ] && \
  { echo "FATAL: training sources dirty"; git status --porcelain -- train_squidiff.py Squidiff; exit 1; }
COMMIT=$(git rev-parse HEAD)
echo "[prereg] 9 sections committed | [commit] $COMMIT | [branch] $(git rev-parse --abbrev-ref HEAD)"

SEED=$(( T % 5 ))
case $(( T / 5 )) in
  0) ARM=ddim;          FM=none ;;
  1) ARM=fm_gaussian;   FM=gaussian ;;
  2) ARM=fm_control_ot; FM=control_ot ;;
esac
TAG=${SPLIT}_${ARM}_seed_${SEED}
OUTDIR=$W/step6/${ARM}/seed_${SEED}
LOGDIR=logger_files/logger_step6_${TAG}
PROV=$W/provenance/step6_${TAG}.json

if [ -e "$OUTDIR" ] && [ -n "$(find "$OUTDIR" -mindepth 1 -print -quit 2>/dev/null)" ]; then
  echo "FATAL: output dir not empty: $OUTDIR"; exit 1; fi
[ -e "$W/$LOGDIR" ] && { echo "FATAL: logger path exists"; exit 1; }
[ -e "$PROV" ] && { echo "FATAL: provenance exists"; exit 1; }
mkdir -p "$OUTDIR" logger_files provenance
export MASTER_PORT=$(( 15000 + T ))
export PYTHONHASHSEED=$SEED

ARGS="--logger_path $LOGDIR --data_path $D/sci_plex_train_${SPLIT}.h5ad"
ARGS="$ARGS --control_data_path $D/sci_plex_train_${SPLIT}_control.h5ad"
ARGS="$ARGS --resume_checkpoint $OUTDIR --seed $SEED --use_adaln True --fm_source $FM"
ARGS="$ARGS --use_drug_structure True --gene_size 200 --output_dim 200 --lr_anneal_steps 100000"

# resolved flags and constructed model, BEFORE training
python - $ARGS <<'PY'
import sys, torch
argv = sys.argv[1:]; sys.argv = ['train_squidiff.py'] + argv
import train_squidiff as t
from Squidiff.script_util import create_model_and_diffusion, model_and_diffusion_defaults, args_to_dict
a = t.parse_args()
print('[guard1] use_adaln=%r (%s)  fm_source=%r (%s)'
      % (a['use_adaln'], type(a['use_adaln']).__name__,
         a['fm_source'], type(a['fm_source']).__name__))
if a['fm_source'] not in ('none','gaussian','control_ot'):
    sys.exit('FATAL: bad fm_source %r' % a['fm_source'])
m, _ = create_model_and_diffusion(**args_to_dict(a, model_and_diffusion_defaults().keys()))
n = sum(p.numel() for p in m.parameters())
print('[guard2] %d parameters' % n)
if n != 58178180: sys.exit('FATAL: expected 58178180 parameters, got %d' % n)
PY

python -c "from Squidiff import MLPModel; print('[import] ' + MLPModel.__file__)"
echo "[$(date)] task $T  arm=$ARM  fm_source=$FM  seed=$SEED"
echo "[cmd] python train_squidiff.py $ARGS"
python train_squidiff.py $ARGS

python - <<PY
import json, os, hashlib, datetime, subprocess, sys, torch
ck = os.path.join('$OUTDIR', 'model_0.9999.pt')
if not os.path.isfile(ck) or os.path.getsize(ck) == 0:
    sys.exit('FATAL: checkpoint missing or empty')
sd = torch.load(ck, map_location='cpu', weights_only=True)
n = int(sum(v.numel() for v in sd.values() if hasattr(v,'numel')))
if n != 58186374: sys.exit('FATAL: %d state_dict entries, expected 58186374' % n)
# The three arms are architecturally identical, so the objective CANNOT be read
# from the weights. Sidecar, as in Step 4b.
side = ck + '.objective.json'
json.dump({'arm': '$ARM', 'fm_source': '$FM', 'seed': int('$SEED'),
           'split': '$SPLIT', 'commit': '$COMMIT',
           'note': 'objective is not recoverable from the state dict; the three '
                   'Step 6 arms share one architecture'},
          open(side, 'w'), indent=2)
r = lambda c: subprocess.run(c, shell=True, capture_output=True, text=True).stdout.strip()
h = hashlib.sha256()
with open(ck,'rb') as f:
    for c in iter(lambda: f.read(1<<20), b''): h.update(c)
json.dump(dict(status='completed', arm='$ARM', fm_source='$FM', seed='$SEED',
               split='$SPLIT', prereg='PREREG_step6_flow_matching.md', commit='$COMMIT',
               completed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
               checkpoint=ck, checkpoint_sha256=h.hexdigest(), n_state_dict=n,
               objective_sidecar=side, node=r('hostname'),
               gpu=torch.cuda.get_device_name(0), torch=torch.__version__),
          open('$PROV','w'), indent=2)
print('[post] %s, %d entries, sidecar written' % ('$ARM', n))
PY
echo "[$(date)] task $T  arm=$ARM  seed=$SEED  DONE"
