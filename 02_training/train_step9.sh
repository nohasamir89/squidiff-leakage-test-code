#!/bin/bash
#SBATCH --job-name=sciplex_s9
#SBATCH --partition=nvidia
#SBATCH --gres=gpu:h100:1
#SBATCH -q a2s2
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=24:00:00
#SBATCH --array=0-14%2
#SBATCH --output=/scratch/ni2071/Diffusion/Squidiff_git/Squidiff/logger_files/s9_%A_%a.out
#SBATCH --error=/scratch/ni2071/Diffusion/Squidiff_git/Squidiff/logger_files/s9_%A_%a.err
# Step 9 under PREREG_step9_pathway_conditioning.md.
#  0-4  fingerprint (arm A) | 5-9 pathway (arm B) | 10-14 both (arm C)
# All arms: Gaussian-source flow matching at 20 NFE, the Step 6b configuration.
# Only the drug encoding differs. Parameter counts differ by design and are
# recorded in the prereg: 58,178,180 / 56,193,668 / 58,290,820.
set -eo pipefail
module load cuda/12.2.0
source /share/apps/NYUAD5/miniconda/3-4.11.0/etc/profile.d/conda.sh
conda activate /scratch/ni2071/conda_envs/squidiff
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
W=/scratch/ni2071/Diffusion/Squidiff_git/Squidiff
D=/scratch/ni2071/Diffusion/sciplex/datasets_corrected
SPLIT=drug_split_1
cd "$W"

T=$SLURM_ARRAY_TASK_ID
case "$T" in *[!0-9]*|"") echo "FATAL: non-numeric task id: $T"; exit 1 ;; esac
if [ "$T" -lt 0 ] || [ "$T" -gt 14 ]; then echo "FATAL: task out of range: $T"; exit 1; fi
N=$(grep -c '^## ' PREREG_step9_pathway_conditioning.md)
if [ "$N" -lt 8 ]; then echo "FATAL: prereg has $N sections, expected at least 8"; exit 1; fi
git diff --quiet -- PREREG_step9_pathway_conditioning.md || { echo "FATAL: prereg uncommitted"; exit 1; }
if [ -n "$(git status --porcelain -- train_squidiff.py Squidiff)" ]; then
  echo "FATAL: training sources dirty"; git status --porcelain -- train_squidiff.py Squidiff; exit 1
fi
COMMIT=$(git rev-parse HEAD)
echo "[prereg] $N sections committed | [commit] $COMMIT | [node] $(hostname)"

SEED=$(( T % 5 ))
case $(( T / 5 )) in
  0) ARM=fingerprint; EXPECT=58178180 ;;
  1) ARM=pathway;     EXPECT=56193668 ;;
  2) ARM=both;        EXPECT=58290820 ;;
esac
TAG=${SPLIT}_${ARM}_seed_${SEED}
OUTDIR=$W/step9/${ARM}/seed_${SEED}
LOGDIR=logger_files/logger_s9_${TAG}
PROV=$W/provenance/s9_${TAG}.json
if [ -e "$OUTDIR" ] && [ -n "$(find "$OUTDIR" -mindepth 1 -print -quit 2>/dev/null)" ]; then
  echo "FATAL: output dir not empty: $OUTDIR"; exit 1; fi
if [ -e "$W/$LOGDIR" ]; then echo "FATAL: logger path exists"; exit 1; fi
if [ -e "$PROV" ]; then echo "FATAL: provenance exists"; exit 1; fi
mkdir -p "$OUTDIR" logger_files provenance
export MASTER_PORT=$(( 18000 + T ))
export PYTHONHASHSEED=$SEED
echo "[task] $T  arm=$ARM  seed=$SEED  expected params=$EXPECT"

ARGS="--logger_path $LOGDIR --data_path $D/sci_plex_train_${SPLIT}.h5ad"
ARGS="$ARGS --control_data_path $D/sci_plex_train_${SPLIT}_control.h5ad"
ARGS="$ARGS --resume_checkpoint $OUTDIR --seed $SEED --use_adaln True"
ARGS="$ARGS --fm_source gaussian --drug_encoding $ARM"
ARGS="$ARGS --use_drug_structure True --gene_size 200 --output_dim 200 --lr_anneal_steps 100000"

python - $ARGS <<'PY'
import sys
sys.argv = ['train_squidiff.py'] + sys.argv[1:]
import train_squidiff as t
from Squidiff.script_util import create_model_and_diffusion, model_and_diffusion_defaults, args_to_dict
from Squidiff.scrna_datasets import DRUG_ENCODER_DIMS
a = t.parse_args()
print('[guard1] drug_encoding=%r (%s)  fm_source=%r  use_adaln=%r'
      % (a['drug_encoding'], type(a['drug_encoding']).__name__, a['fm_source'], a['use_adaln']))
if a['use_adaln'] is not True: sys.exit('FATAL: use_adaln did not resolve to True')
if a['fm_source'] != 'gaussian': sys.exit('FATAL: fm_source must be gaussian')
a['drug_dimension'] = DRUG_ENCODER_DIMS[a['drug_encoding']]
m, _ = create_model_and_diffusion(**args_to_dict(a, model_and_diffusion_defaults().keys()))
n = sum(p.numel() for p in m.parameters())
print('[guard2] drug_dimension=%d  parameters=%d' % (a['drug_dimension'], n))
PY

echo "[$(date)] training $TAG"
python train_squidiff.py $ARGS

python - <<PY
import json, os, hashlib, datetime, subprocess, sys, torch
ck = os.path.join('$OUTDIR', 'model_0.9999.pt')
if not os.path.isfile(ck) or os.path.getsize(ck) == 0:
    sys.exit('FATAL: checkpoint missing or empty')
sd = torch.load(ck, map_location='cpu', weights_only=True)
n = int(sum(v.numel() for v in sd.values() if hasattr(v, 'numel')))
# the encoder's first layer width is the only structural trace of the encoding
w = [tuple(v.shape) for k, v in sd.items() if 'encoder' in k and v.dim() == 2]
side = ck + '.encoding.json'
json.dump({'arm': '$ARM', 'drug_encoding': '$ARM', 'seed': int('$SEED'),
           'split': '$SPLIT', 'commit': '$COMMIT', 'n_state_dict': n,
           'prereg': 'PREREG_step9_pathway_conditioning.md'},
          open(side, 'w'), indent=2)
r = lambda c: subprocess.run(c, shell=True, capture_output=True, text=True).stdout.strip()
h = hashlib.sha256()
with open(ck, 'rb') as f:
    for c in iter(lambda: f.read(1<<20), b''): h.update(c)
json.dump(dict(status='completed', arm='$ARM', drug_encoding='$ARM', seed='$SEED',
               split='$SPLIT', prereg='PREREG_step9_pathway_conditioning.md',
               commit='$COMMIT', fm_source='gaussian',
               completed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
               checkpoint=ck, checkpoint_sha256=h.hexdigest(), n_state_dict=n,
               encoding_sidecar=side, node=r('hostname'),
               gpu=torch.cuda.get_device_name(0)),
          open('$PROV', 'w'), indent=2)
print('[post] %s seed %s: %d state_dict entries' % ('$ARM', '$SEED', n))
PY
echo "[$(date)] $TAG DONE"
