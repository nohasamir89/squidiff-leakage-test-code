#!/bin/bash
#SBATCH --job-name=sciplex_s13b
#SBATCH --partition=condo
#SBATCH -q a2s2
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=24:00:00
#SBATCH --array=0-19
#SBATCH --output=/scratch/ni2071/Diffusion/Squidiff_git/Squidiff/logger_files/s13b_%A_%a.out
#SBATCH --error=/scratch/ni2071/Diffusion/Squidiff_git/Squidiff/logger_files/s13b_%A_%a.err
# Step 13b under PREREG_step13b_confirmation.md: TWO arms x five seeds on each of
# drug_split_0 (tasks 0-9) and drug_split_2 (tasks 10-19).
#   within a split: tasks 0-4 attn_cross (drug x cell-state cross-attention),
#                   tasks 5-9 mlp_ctrl   (parameter-matched non-attention control)
# Same recipe as Step 13: Gaussian-source flow matching, semantic adaLN,
# --drug_encoding fingerprint (full construction), 100,000 steps, no condition dropout.
# The reference is NOT retrained: the Step 11 vectors (Step 6b checkpoints at 50 NFE).
# DRYRUN=1 SLURM_ARRAY_TASK_ID=0 bash train_step13b.sh  runs every guard and stops.
set -eo pipefail
module load cuda/12.2.0
source /share/apps/NYUAD5/miniconda/3-4.11.0/etc/profile.d/conda.sh
conda activate /scratch/ni2071/conda_envs/squidiff
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
W=/scratch/ni2071/Diffusion/Squidiff_git/Squidiff
D=/scratch/ni2071/Diffusion/sciplex/datasets_corrected
PREREG=PREREG_step13b_confirmation.md
cd "$W"

T=$SLURM_ARRAY_TASK_ID
case "$T" in *[!0-9]*|"") echo "FATAL: non-numeric task id: $T"; exit 1 ;; esac
if [ "$T" -gt 19 ]; then echo "FATAL: task out of range: $T"; exit 1; fi
N=$(grep -c '^## ' $PREREG)
if [ "$N" -lt 9 ]; then echo "FATAL: prereg has $N sections, expected at least 9"; exit 1; fi
git diff --quiet -- $PREREG || { echo "FATAL: prereg uncommitted"; exit 1; }
if [ -n "$(git status --porcelain -- train_squidiff.py Squidiff)" ]; then
  echo "FATAL: training sources dirty"; git status --porcelain -- train_squidiff.py Squidiff; exit 1
fi
COMMIT=$(git rev-parse HEAD)
echo "[prereg] $N sections committed | [commit] $COMMIT | [node] $(hostname)"

case $(( T / 10 )) in 0) SPLIT=drug_split_0 ;; 1) SPLIT=drug_split_2 ;; esac
R=$(( T % 10 )); SEED=$(( R % 5 ))
ENC=fingerprint
case $(( R / 5 )) in
  0) ARM=attn_cross; EXPECT=58532804 ;;
  1) ARM=mlp_ctrl;   EXPECT=58532300 ;;
esac
MEMSZ=0                 # no memory arm in Step 13b
export ARM EXPECT MEMSZ  # read by the guard and by nothing else
TAG=${SPLIT}_${ARM}_seed_${SEED}
OUTDIR=$W/step13b/${SPLIT}_${ARM}/seed_${SEED}
LOGDIR=logger_files/logger_s13b_${TAG}
PROV=$W/provenance/s13b_${TAG}.json
if [ -e "$OUTDIR" ] && [ -n "$(find "$OUTDIR" -mindepth 1 -print -quit 2>/dev/null)" ]; then
  echo "FATAL: output dir not empty: $OUTDIR"; exit 1; fi
if [ -e "$W/$LOGDIR" ]; then echo "FATAL: logger path exists"; exit 1; fi
if [ -e "$PROV" ]; then echo "FATAL: provenance exists"; exit 1; fi
export MASTER_PORT=$(( 18600 + T ))
export PYTHONHASHSEED=$SEED
echo "[task] $T  split=$SPLIT  arm=$ARM  drug_encoding=$ENC  cond_drop=0  seed=$SEED  expected params=$EXPECT"

ARGS="--logger_path $LOGDIR --data_path $D/sci_plex_train_${SPLIT}.h5ad"
ARGS="$ARGS --control_data_path $D/sci_plex_train_${SPLIT}_control.h5ad"
ARGS="$ARGS --resume_checkpoint $OUTDIR --seed $SEED --use_adaln True"
ARGS="$ARGS --fm_source gaussian --drug_encoding $ENC --encoder_arm $ARM"
ARGS="$ARGS --use_drug_structure True --gene_size 200 --output_dim 200 --lr_anneal_steps 100000"

# Guards run BEFORE any directory is created: parsed values, the memory size from the
# data, then the model's parameter count for THIS arm.
python - $ARGS <<'PY'
import sys, os
sys.argv = ['train_squidiff.py'] + sys.argv[1:]
import train_squidiff as t
from Squidiff.script_util import create_model_and_diffusion, model_and_diffusion_defaults, args_to_dict
from Squidiff.scrna_datasets import DRUG_ENCODER_DIMS
a = t.parse_args()
arm, expect, memsz = os.environ['ARM'], int(os.environ['EXPECT']), int(os.environ['MEMSZ'])
print('[guard1] drug_encoding=%r  fm_source=%r  use_adaln=%r  cond_drop=%r (%s)  encoder_arm=%r (%s)'
      % (a['drug_encoding'], a['fm_source'], a['use_adaln'], a['cond_drop'], type(a['cond_drop']).__name__,
         a['encoder_arm'], type(a['encoder_arm']).__name__))
if a['use_adaln'] is not True: sys.exit('FATAL: use_adaln did not resolve to True')
if a['fm_source'] != 'gaussian': sys.exit('FATAL: fm_source must be gaussian')
if a['drug_encoding'] != 'fingerprint': sys.exit('FATAL: drug_encoding must be fingerprint')
if not isinstance(a['cond_drop'], float) or a['cond_drop'] != 0.0: sys.exit('FATAL: cond_drop must be the float 0.0: %r' % (a['cond_drop'],))
if a['encoder_arm'] != arm: sys.exit('FATAL: encoder_arm parsed as %r, task expects %r' % (a['encoder_arm'], arm))
a['drug_dimension'] = DRUG_ENCODER_DIMS[a['drug_encoding']]
a['memory_size'] = 0
if arm == 'attn_mem':
    import anndata as ad
    tr = set(str(s) for s in ad.read_h5ad(a['data_path'], backed='r').obs['SMILES'])
    te = set(str(s) for s in ad.read_h5ad(a['data_path'].replace('_train_', '_test_'), backed='r').obs['SMILES'])
    print('[guard1b] training compounds %d, test compounds %d, overlap %d' % (len(tr), len(te), len(tr & te)))
    if len(tr) != memsz: sys.exit('FATAL: %d training compounds, PREREG says %d' % (len(tr), memsz))
    if tr & te: sys.exit('FATAL: training and test compounds overlap')
    a['memory_size'] = len(tr)
m, _ = create_model_and_diffusion(**args_to_dict(a, model_and_diffusion_defaults().keys()))
n = sum(p.numel() for p in m.parameters())
print('[guard2] arm=%s drug_dimension=%d memory_size=%d parameters=%d' % (arm, a['drug_dimension'], a['memory_size'], n))
if n != expect: sys.exit('FATAL: parameter count %d != %d for arm %s' % (n, expect, arm))
PY
if [ "${DRYRUN:-0}" = "1" ]; then echo "[dryrun] all guards passed; nothing created"; exit 0; fi
mkdir -p "$OUTDIR" logger_files provenance

echo "[$(date)] training $TAG"
python train_squidiff.py $ARGS

# Post: checkpoint present; the training-time assertions were printed; for attn_mem the
# memory table is inside the checkpoint and marked filled; sidecar and provenance written.
python - <<PY
import json, os, hashlib, datetime, subprocess, sys, re, torch
ck = os.path.join('$OUTDIR', 'model_0.9999.pt')
if not os.path.isfile(ck) or os.path.getsize(ck) == 0: sys.exit('FATAL: checkpoint missing or empty')
sd = torch.load(ck, map_location='cpu', weights_only=True)
n_sd = int(sum(v.numel() for v in sd.values() if hasattr(v, 'numel')))
log = open('logger_files/s13b_${SLURM_ARRAY_JOB_ID}_${T}.out').read()
m = re.search(r'\[step13\] encoder_arm=(\S+) params=(\d+) memory_size=(\d+) identity_at_init=pass', log)
if not m: sys.exit('FATAL: the [step13] assertion line is missing from the log')
if m.group(1) != '$ARM' or int(m.group(2)) != int('$EXPECT'): sys.exit('FATAL: log says arm %s params %s' % (m.group(1), m.group(2)))
mem = dict(memory_size=0, smiles_sha256=None, n_test_compounds=None)
if '$ARM' == 'attn_mem':
    mm = re.search(r'\[step13\] memory table: (\d+) training compounds, (\d+) test compounds, overlap 0, smiles_sha256=([0-9a-f]+)', log)
    if not mm: sys.exit('FATAL: memory-table line missing from the log')
    if int(mm.group(1)) != int('$MEMSZ'): sys.exit('FATAL: memory table has %s rows, expected $MEMSZ' % mm.group(1))
    kf, filled = sd.get('encoder.arm.keys_fp'), sd.get('encoder.arm.memory_filled')
    if kf is None or tuple(kf.shape) != (int('$MEMSZ'), 1024): sys.exit('FATAL: keys_fp buffer missing or wrong shape in the checkpoint')
    if filled is None or float(filled) != 1.0: sys.exit('FATAL: memory_filled flag not saved as 1 in the checkpoint')
    if float(kf.max()) != 1.0 or not bool(((kf == 0) | (kf == 1)).all()): sys.exit('FATAL: keys_fp is not an unscaled bit table')
    mem = dict(memory_size=int(mm.group(1)), smiles_sha256=mm.group(3), n_test_compounds=int(mm.group(2)))
    print('[post] memory table %d x 1024 saved in the checkpoint, filled flag 1, unscaled bits' % kf.shape[0])
arm_keys = sorted(k for k in sd if k.startswith('encoder.arm.'))
if not arm_keys: sys.exit('FATAL: no encoder.arm.* keys in the checkpoint')
side = ck + '.encoding.json'
json.dump({'arm': '$ARM', 'encoder_arm': '$ARM', 'drug_encoding': '$ENC', 'cond_drop': 0.0, 'seed': int('$SEED'),
           'split': '$SPLIT', 'commit': '$COMMIT', 'n_state_dict': n_sd, 'n_params': int('$EXPECT'), 'prereg': '$PREREG', **mem},
          open(side, 'w'), indent=2)
r = lambda c: subprocess.run(c, shell=True, capture_output=True, text=True).stdout.strip()
h = hashlib.sha256()
with open(ck, 'rb') as f:
    for c in iter(lambda: f.read(1<<20), b''): h.update(c)
json.dump(dict(status='completed', arm='$ARM', encoder_arm='$ARM', drug_encoding='$ENC', cond_drop=0.0, seed='$SEED',
               split='$SPLIT', prereg='$PREREG', commit='$COMMIT', fm_source='gaussian', n_params=int('$EXPECT'),
               arm_state_keys=arm_keys, **mem,
               completed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
               checkpoint=ck, checkpoint_sha256=h.hexdigest(), n_state_dict=n_sd,
               encoding_sidecar=side, node=r('hostname'), gpu=torch.cuda.get_device_name(0)),
          open('$PROV', 'w'), indent=2)
print('[post] %s seed %s: %d state_dict entries, %d arm keys' % ('$ARM', '$SEED', n_sd, len(arm_keys)))
PY
echo "[$(date)] $TAG DONE"
