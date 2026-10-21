#!/bin/bash
#SBATCH --job-name=s6b_eval
#SBATCH --partition=nvidia
#SBATCH --nodelist=cn274
#SBATCH --gres=gpu:h100:1
#SBATCH -q a2s2
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=12:00:00
#SBATCH --array=0-19%2
#SBATCH --output=/scratch/ni2071/Diffusion/Squidiff_git/Squidiff/logger_files/s6bev_%A_%a.out
#SBATCH --error=/scratch/ni2071/Diffusion/Squidiff_git/Squidiff/logger_files/s6bev_%A_%a.err
# Step 6b evaluation. Arm A: frozen harness, 1000 DDIM steps. Arm B: ODE
# wrapper at 20 NFE. Per-cell storage on from the start, so P2 needs no
# recovery run. Pinned to cn274, matching every Step 6 evaluation.
set -eo pipefail
module load cuda/12.2.0
source /share/apps/NYUAD5/miniconda/3-4.11.0/etc/profile.d/conda.sh
conda activate /scratch/ni2071/conda_envs/squidiff
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export SQUIDIFF_SAVE_PER_CELL=1
W=/scratch/ni2071/Diffusion/Squidiff_git/Squidiff
NFE=20
cd "$W"

T=$SLURM_ARRAY_TASK_ID
case "$T" in *[!0-9]*|"") echo "FATAL: non-numeric task id: $T"; exit 1 ;; esac
if [ "$T" -lt 0 ] || [ "$T" -gt 19 ]; then echo "FATAL: task out of range: $T"; exit 1; fi
N=$(grep -c '^## ' PREREG_step6b_confirmation.md)
if [ "$N" -lt 8 ]; then echo "FATAL: prereg has $N sections"; exit 1; fi
if [ -n "$(git status --porcelain -- leakage_test_base_sciplex_array.py sample_squidiff.py Squidiff)" ]; then
  echo "FATAL: evaluation sources dirty"; exit 1
fi
echo "[freeze] prereg $N sections, sources clean, commit $(git rev-parse --short HEAD)"
echo "[node] $(hostname)"

SEED=$(( T % 5 ))
case $(( T / 5 )) in
  0) SPLIT=drug_split_0; ARM=ddim ;;
  1) SPLIT=drug_split_0; ARM=fm_gaussian ;;
  2) SPLIT=drug_split_2; ARM=ddim ;;
  3) SPLIT=drug_split_2; ARM=fm_gaussian ;;
esac
CKPT=$W/step6b/${SPLIT}_${ARM}/seed_${SEED}/model_0.9999.pt
TAG=s6b_${SPLIT}_${ARM}_seed_${SEED}
SIDE=${CKPT}.objective.json
if [ ! -s "$CKPT" ]; then echo "FATAL: checkpoint missing: $CKPT"; exit 1; fi
if [ ! -s "$SIDE" ]; then echo "FATAL: objective sidecar missing"; exit 1; fi
if [ -e "$W/leakage_results/vectors_${TAG}.npz" ]; then echo "FATAL: output exists"; exit 1; fi
export MASTER_PORT=$(( 17400 + T ))

python - <<PY
import json, sys
s = json.load(open('$SIDE'))
if s.get('arm') != '$ARM' or s.get('split') != '$SPLIT':
    sys.exit('FATAL: sidecar says %r/%r, task expects $SPLIT/$ARM'
             % (s.get('split'), s.get('arm')))
print('[sidecar] %s %s seed %s fm_source=%s' % (s['split'], s['arm'], s['seed'], s['fm_source']))
PY

echo "[$(date)] task $T  $TAG"
if [ "$ARM" = "ddim" ]; then
  python leakage_test_base_sciplex_array.py "$SPLIT" "$CKPT" "$TAG"
else
  python leakage_test_fm_sciplex.py "$SPLIT" "$CKPT" "$TAG" fm_gaussian "$NFE" euler
fi

python - <<PY
import json, os, sys, hashlib, subprocess, datetime, torch
import numpy as np
v  = 'leakage_results/vectors_$TAG.npz'
pc = 'leakage_results/vectors_${TAG}_pc.npz'
for f in (v, pc):
    if not os.path.isfile(f) or os.path.getsize(f) == 0:
        sys.exit('FATAL: missing or empty ' + f)
d = np.load(v, allow_pickle=True)
n = len(d['cond'])
# drug_split_0 has 444 conditions, drug_split_2 has 455
expect = {'drug_split_0': 444, 'drug_split_2': 455}['$SPLIT']
if n != expect: sys.exit('FATAL: %d conditions, expected %d for $SPLIT' % (n, expect))
for k in ('true','A','B','C'):
    if not np.all(np.isfinite(d[k])): sys.exit('FATAL: non-finite %s' % k)
if float(np.asarray(d['B']).min()) < 0: sys.exit('FATAL: negative predicted expression')
p = np.load(pc, allow_pickle=True)
if len(p['pred']) != expect: sys.exit('FATAL: per-cell has %d conditions' % len(p['pred']))
nc = [np.asarray(a).shape[0] for a in p['pred']]
sha = lambda q: hashlib.sha256(open(q,'rb').read()).hexdigest()
r = lambda c: subprocess.run(c, shell=True, capture_output=True, text=True).stdout.strip()
json.dump(dict(status='completed', split='$SPLIT', arm='$ARM', seed='$SEED', tag='$TAG',
               nfe=(1000 if '$ARM' == 'ddim' else int('$NFE')),
               sampler=('ddim' if '$ARM' == 'ddim' else 'euler_ode'),
               prereg='PREREG_step6b_confirmation.md', commit=r('git rev-parse HEAD'),
               completed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
               checkpoint='$CKPT', checkpoint_sha256=sha('$CKPT'),
               vectors_npz=v, vectors_sha256=sha(v),
               per_cell_npz=pc, per_cell_sha256=sha(pc),
               n_conditions=int(n), cells_min=int(min(nc)), cells_max=int(max(nc)),
               node=r('hostname'), gpu=torch.cuda.get_device_name(0),
               harness_sha256=sha('leakage_test_base_sciplex_array.py'),
               sampler_sha256=sha('sample_squidiff.py')),
          open('provenance/s6bev_$TAG.json','w'), indent=2)
print('[post] %d conditions, cells %d-%d, finite, non-negative' % (n, min(nc), max(nc)))
PY
echo "[$(date)] task $T  $TAG  DONE"
