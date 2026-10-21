#!/bin/bash
#SBATCH --job-name=sciplex_s6b
#SBATCH --partition=nvidia
#SBATCH --gres=gpu:h100:1
#SBATCH -q a2s2
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=24:00:00
#SBATCH --array=0-19%2
#SBATCH --output=/scratch/ni2071/Diffusion/Squidiff_git/Squidiff/logger_files/s6b_%A_%a.out
#SBATCH --error=/scratch/ni2071/Diffusion/Squidiff_git/Squidiff/logger_files/s6b_%A_%a.err
# Step 6b under PREREG_step6b_confirmation.md. Two arms x five seeds x two splits.
#  0-4   drug_split_0 ddim        | 5-9   drug_split_0 fm_gaussian
#  10-14 drug_split_2 ddim        | 15-19 drug_split_2 fm_gaussian
# The arms are architecturally identical, so arm identity is NOT recoverable
# from a checkpoint. A sidecar records it.
set -eo pipefail
module load cuda/12.2.0
source /share/apps/NYUAD5/miniconda/3-4.11.0/etc/profile.d/conda.sh
conda activate /scratch/ni2071/conda_envs/squidiff
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
W=/scratch/ni2071/Diffusion/Squidiff_git/Squidiff
D=/scratch/ni2071/Diffusion/sciplex/datasets_corrected
cd "$W"

T=$SLURM_ARRAY_TASK_ID
case "$T" in *[!0-9]*|"") echo "FATAL: non-numeric task id: $T"; exit 1 ;; esac
if [ "$T" -lt 0 ] || [ "$T" -gt 19 ]; then echo "FATAL: task out of range: $T"; exit 1; fi
N=$(grep -c '^## ' PREREG_step6b_confirmation.md)
if [ "$N" -lt 8 ]; then echo "FATAL: prereg has $N sections, expected at least 8"; exit 1; fi
git diff --quiet -- PREREG_step6b_confirmation.md || { echo "FATAL: prereg uncommitted"; exit 1; }
if [ -n "$(git status --porcelain -- train_squidiff.py Squidiff)" ]; then
  echo "FATAL: training sources dirty"; git status --porcelain -- train_squidiff.py Squidiff; exit 1
fi
COMMIT=$(git rev-parse HEAD)
echo "[prereg] $N sections committed | [commit] $COMMIT | [node] $(hostname)"

SEED=$(( T % 5 ))
case $(( T / 5 )) in
  0) SPLIT=drug_split_0; ARM=ddim;        FM=none ;;
  1) SPLIT=drug_split_0; ARM=fm_gaussian; FM=gaussian ;;
  2) SPLIT=drug_split_2; ARM=ddim;        FM=none ;;
  3) SPLIT=drug_split_2; ARM=fm_gaussian; FM=gaussian ;;
esac
TAG=${SPLIT}_${ARM}_seed_${SEED}
OUTDIR=$W/step6b/${SPLIT}_${ARM}/seed_${SEED}
LOGDIR=logger_files/logger_s6b_${TAG}
PROV=$W/provenance/s6b_${TAG}.json

if [ -e "$OUTDIR" ] && [ -n "$(find "$OUTDIR" -mindepth 1 -print -quit 2>/dev/null)" ]; then
  echo "FATAL: output dir not empty: $OUTDIR"; exit 1; fi
if [ -e "$W/$LOGDIR" ]; then echo "FATAL: logger path exists"; exit 1; fi
if [ -e "$PROV" ]; then echo "FATAL: provenance exists"; exit 1; fi
mkdir -p "$OUTDIR" logger_files provenance
export MASTER_PORT=$(( 17000 + T ))
export PYTHONHASHSEED=$SEED
echo "[task] $T  split=$SPLIT arm=$ARM fm_source=$FM seed=$SEED"

ARGS="--logger_path $LOGDIR --data_path $D/sci_plex_train_${SPLIT}.h5ad"
ARGS="$ARGS --control_data_path $D/sci_plex_train_${SPLIT}_control.h5ad"
ARGS="$ARGS --resume_checkpoint $OUTDIR --seed $SEED --use_adaln True --fm_source $FM"
ARGS="$ARGS --use_drug_structure True --gene_size 200 --output_dim 200 --lr_anneal_steps 100000"

python - $ARGS <<'PY'
import sys, torch
sys.argv = ['train_squidiff.py'] + sys.argv[1:]
import train_squidiff as t
from Squidiff.script_util import create_model_and_diffusion, model_and_diffusion_defaults, args_to_dict
a = t.parse_args()
print('[guard1] use_adaln=%r (%s)  fm_source=%r (%s)'
      % (a['use_adaln'], type(a['use_adaln']).__name__,
         a['fm_source'], type(a['fm_source']).__name__))
if a['use_adaln'] is not True: sys.exit('FATAL: use_adaln did not resolve to True')
if a['fm_source'] not in ('none','gaussian'): sys.exit('FATAL: bad fm_source')
m, _ = create_model_and_diffusion(**args_to_dict(a, model_and_diffusion_defaults().keys()))
n = sum(p.numel() for p in m.parameters())
print('[guard2] %d parameters' % n)
if n != 58178180: sys.exit('FATAL: expected 58178180 parameters, got %d' % n)
PY

echo "[$(date)] training $TAG"
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
side = ck + '.objective.json'
json.dump({'arm': '$ARM', 'fm_source': '$FM', 'seed': int('$SEED'), 'split': '$SPLIT',
           'commit': '$COMMIT', 'prereg': 'PREREG_step6b_confirmation.md',
           'note': 'objective not recoverable from the state dict; the arms share '
                   'one architecture'},
          open(side, 'w'), indent=2)
r = lambda c: subprocess.run(c, shell=True, capture_output=True, text=True).stdout.strip()
h = hashlib.sha256()
with open(ck,'rb') as f:
    for c in iter(lambda: f.read(1<<20), b''): h.update(c)
json.dump(dict(status='completed', split='$SPLIT', arm='$ARM', fm_source='$FM',
               seed='$SEED', prereg='PREREG_step6b_confirmation.md', commit='$COMMIT',
               completed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
               checkpoint=ck, checkpoint_sha256=h.hexdigest(), n_state_dict=n,
               objective_sidecar=side, node=r('hostname'),
               gpu=torch.cuda.get_device_name(0)),
          open('$PROV','w'), indent=2)
print('[post] %s %s seed %s: %d entries, sidecar written' % ('$SPLIT','$ARM','$SEED',n))
PY
echo "[$(date)] $TAG DONE"
