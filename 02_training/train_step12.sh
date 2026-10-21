#!/bin/bash
#SBATCH --job-name=sciplex_s12
#SBATCH --partition=condo
#SBATCH -q a2s2
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=24:00:00
#SBATCH --array=0-4
#SBATCH --output=/scratch/ni2071/Diffusion/Squidiff_git/Squidiff/logger_files/s12_%A_%a.out
#SBATCH --error=/scratch/ni2071/Diffusion/Squidiff_git/Squidiff/logger_files/s12_%A_%a.err
# Step 12 under PREREG_step12_guidance.md: ONE arm, five seeds, drug_split_1.
# Identical to train_step10.sh arm 0 (fingerprint_s10: Gaussian-source flow
# matching, semantic adaLN, Morgan-1024 dose-scaled fingerprint, Step 10 data
# construction via the drug_encoding string) plus --cond_drop 0.1. Dropout adds
# no parameters, so the count must equal the arm-0 guard value 58,178,180.
# DRYRUN=1 SLURM_ARRAY_TASK_ID=0 bash train_step12.sh  runs every guard and stops.
set -eo pipefail
module load cuda/12.2.0
source /share/apps/NYUAD5/miniconda/3-4.11.0/etc/profile.d/conda.sh
conda activate /scratch/ni2071/conda_envs/squidiff
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
W=/scratch/ni2071/Diffusion/Squidiff_git/Squidiff
D=/scratch/ni2071/Diffusion/sciplex/datasets_corrected
SPLIT=drug_split_1
PREREG=PREREG_step12_guidance.md
cd "$W"

T=$SLURM_ARRAY_TASK_ID
case "$T" in *[!0-9]*|"") echo "FATAL: non-numeric task id: $T"; exit 1 ;; esac
if [ "$T" -gt 4 ]; then echo "FATAL: task out of range: $T"; exit 1; fi
N=$(grep -c '^## ' $PREREG)
if [ "$N" -lt 9 ]; then echo "FATAL: prereg has $N sections, expected at least 9"; exit 1; fi
git diff --quiet -- $PREREG || { echo "FATAL: prereg uncommitted"; exit 1; }
if [ -n "$(git status --porcelain -- train_squidiff.py Squidiff)" ]; then
  echo "FATAL: training sources dirty"; git status --porcelain -- train_squidiff.py Squidiff; exit 1
fi
COMMIT=$(git rev-parse HEAD)
echo "[prereg] $N sections committed | [commit] $COMMIT | [node] $(hostname)"

SEED=$T
ENC=fingerprint_s10     # drug_encoding string: fires the Step 10 exclusion, then resolves to the fingerprint encoder
ARM=cfg_fingerprint     # Step 12 arm name for directories, tags, sidecars and provenance
CDROP=0.1
EXPECT=58178180
TAG=${SPLIT}_${ARM}_seed_${SEED}
OUTDIR=$W/step12/${ARM}/seed_${SEED}
LOGDIR=logger_files/logger_s12_${TAG}
PROV=$W/provenance/s12_${TAG}.json
if [ -e "$OUTDIR" ] && [ -n "$(find "$OUTDIR" -mindepth 1 -print -quit 2>/dev/null)" ]; then
  echo "FATAL: output dir not empty: $OUTDIR"; exit 1; fi
if [ -e "$W/$LOGDIR" ]; then echo "FATAL: logger path exists"; exit 1; fi
if [ -e "$PROV" ]; then echo "FATAL: provenance exists"; exit 1; fi
export MASTER_PORT=$(( 18100 + T ))
export PYTHONHASHSEED=$SEED
echo "[task] $T  arm=$ARM  drug_encoding=$ENC  cond_drop=$CDROP  seed=$SEED  expected params=$EXPECT"

ARGS="--logger_path $LOGDIR --data_path $D/sci_plex_train_${SPLIT}.h5ad"
ARGS="$ARGS --control_data_path $D/sci_plex_train_${SPLIT}_control.h5ad"
ARGS="$ARGS --resume_checkpoint $OUTDIR --seed $SEED --use_adaln True"
ARGS="$ARGS --fm_source gaussian --drug_encoding $ENC --cond_drop $CDROP"
ARGS="$ARGS --use_drug_structure True --gene_size 200 --output_dim 200 --lr_anneal_steps 100000"

# Guards run BEFORE any directory is created: parsed values, then the model's parameter count.
python - $ARGS <<'PY'
import sys
sys.argv = ['train_squidiff.py'] + sys.argv[1:]
import train_squidiff as t
from Squidiff.script_util import create_model_and_diffusion, model_and_diffusion_defaults, args_to_dict
from Squidiff.scrna_datasets import DRUG_ENCODER_DIMS
a = t.parse_args()
print('[guard1] drug_encoding=%r (%s)  fm_source=%r  use_adaln=%r  cond_drop=%r (%s)'
      % (a['drug_encoding'], type(a['drug_encoding']).__name__, a['fm_source'], a['use_adaln'],
         a['cond_drop'], type(a['cond_drop']).__name__))
if a['use_adaln'] is not True: sys.exit('FATAL: use_adaln did not resolve to True')
if a['fm_source'] != 'gaussian': sys.exit('FATAL: fm_source must be gaussian')
if a['drug_encoding'] != 'fingerprint_s10': sys.exit('FATAL: drug_encoding must be fingerprint_s10')
if not isinstance(a['cond_drop'], float) or abs(a['cond_drop'] - 0.1) > 1e-12:
    sys.exit('FATAL: cond_drop did not parse to the float 0.1: %r' % (a['cond_drop'],))
a['drug_dimension'] = DRUG_ENCODER_DIMS[a['drug_encoding']]
m, _ = create_model_and_diffusion(**args_to_dict(a, model_and_diffusion_defaults().keys()))
n = sum(p.numel() for p in m.parameters())
print('[guard2] drug_dimension=%d  parameters=%d' % (a['drug_dimension'], n))
if n != 58178180: sys.exit('FATAL: parameter count %d != 58178180' % n)
PY
if [ "${DRYRUN:-0}" = "1" ]; then echo "[dryrun] all guards passed; nothing created"; exit 0; fi
mkdir -p "$OUTDIR" logger_files provenance

echo "[$(date)] training $TAG"
python train_squidiff.py $ARGS

# Post: checkpoint present, dropout demonstrably ran at the intended rate, sidecar and provenance written.
python - <<PY
import json, os, hashlib, datetime, subprocess, sys, re, torch
ck = os.path.join('$OUTDIR', 'model_0.9999.pt')
if not os.path.isfile(ck) or os.path.getsize(ck) == 0:
    sys.exit('FATAL: checkpoint missing or empty')
sd = torch.load(ck, map_location='cpu', weights_only=True)
n = int(sum(v.numel() for v in sd.values() if hasattr(v, 'numel')))
log = 'logger_files/s12_${SLURM_ARRAY_JOB_ID}_${T}.out'
rates = re.findall(r'\[cond_drop\] call (\d+)\s+empirical rate ([0-9.]+)', open(log).read())
if not rates: sys.exit('FATAL: no [cond_drop] lines in %s - dropout never ran' % log)
calls, rate = int(rates[-1][0]), float(rates[-1][1])
if abs(rate - float('$CDROP')) > 0.005:
    sys.exit('FATAL: empirical dropout rate %.4f after %d calls, target $CDROP' % (rate, calls))
print('[post] empirical dropout rate %.4f after %d calls' % (rate, calls))
side = ck + '.encoding.json'
json.dump({'arm': '$ARM', 'drug_encoding': '$ENC', 'cond_drop': float('$CDROP'), 'seed': int('$SEED'),
           'split': '$SPLIT', 'commit': '$COMMIT', 'n_state_dict': n, 'prereg': '$PREREG'},
          open(side, 'w'), indent=2)
r = lambda c: subprocess.run(c, shell=True, capture_output=True, text=True).stdout.strip()
h = hashlib.sha256()
with open(ck, 'rb') as f:
    for c in iter(lambda: f.read(1<<20), b''): h.update(c)
json.dump(dict(status='completed', arm='$ARM', drug_encoding='$ENC', cond_drop=float('$CDROP'),
               empirical_cond_drop=rate, cond_drop_calls=calls, seed='$SEED',
               split='$SPLIT', prereg='$PREREG', commit='$COMMIT', fm_source='gaussian',
               completed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
               checkpoint=ck, checkpoint_sha256=h.hexdigest(), n_state_dict=n,
               encoding_sidecar=side, node=r('hostname'),
               gpu=torch.cuda.get_device_name(0)),
          open('$PROV', 'w'), indent=2)
print('[post] %s seed %s: %d state_dict entries' % ('$ARM', '$SEED', n))
PY
echo "[$(date)] $TAG DONE"
