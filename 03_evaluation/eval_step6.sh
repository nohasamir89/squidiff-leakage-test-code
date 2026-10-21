#!/bin/bash
#SBATCH --job-name=sciplex_s6eval
#SBATCH --partition=nvidia
#SBATCH --gres=gpu:h100:1
#SBATCH -q a2s2
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=12:00:00
#SBATCH --array=0-14%2
#SBATCH --output=/scratch/ni2071/Diffusion/Squidiff_git/Squidiff/logger_files/s6eval_%A_%a.out
#SBATCH --error=/scratch/ni2071/Diffusion/Squidiff_git/Squidiff/logger_files/s6eval_%A_%a.err
# 0-4 ddim | 5-9 fm_gaussian | 10-14 fm_control_ot
# Arm A uses the frozen harness; arms B and C use leakage_test_fm_sciplex.py,
# which wraps it and substitutes ODE integration for the DDIM loop.
# The NODE is recorded per task: cn270 differs from cn274 by ~6e-06 on
# sampled outputs, demonstrated in CORRECTION_cn270.md.
set -eo pipefail
module load cuda/12.2.0
source /share/apps/NYUAD5/miniconda/3-4.11.0/etc/profile.d/conda.sh
conda activate /scratch/ni2071/conda_envs/squidiff
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
W=/scratch/ni2071/Diffusion/Squidiff_git/Squidiff
SPLIT=drug_split_1
NFE=20
SOLVER=euler
cd "$W"

[ -n "${SLURM_ARRAY_TASK_ID:-}" ] || { echo "FATAL: needs a SLURM array"; exit 1; }
T=$SLURM_ARRAY_TASK_ID
case "$T" in *[!0-9]*|"") echo "FATAL: non-numeric task id: $T"; exit 1 ;; esac
if [ "$T" -lt 0 ] || [ "$T" -gt 14 ]; then echo "FATAL: task out of range: $T"; exit 1; fi

N=$(grep -c '^## ' PREREG_step6_flow_matching.md)
# a minimum, not an equality: amendments add sections, and an equality
# guard silently invalidates the array every time one is written
if [ "$N" -lt 10 ]; then echo "FATAL: prereg has $N sections, expected at least 10"; exit 1; fi
if [ -n "$(git status --porcelain -- leakage_test_base_sciplex_array.py sample_squidiff.py Squidiff)" ]; then
  echo "FATAL: evaluation sources dirty"; exit 1
fi
echo "[freeze] prereg $N sections, sources clean, commit $(git rev-parse --short HEAD)"
echo "[node] $(hostname)"

SEED=$(( T % 5 ))
case $(( T / 5 )) in
  0) ARM=ddim ;;
  1) ARM=fm_gaussian ;;
  2) ARM=fm_control_ot ;;
esac
CKPT=$W/step6/${ARM}/seed_${SEED}/model_0.9999.pt
TAG=step6_${ARM}_seed_${SEED}
if [ ! -s "$CKPT" ]; then echo "FATAL: checkpoint missing: $CKPT"; exit 1; fi
for f in leakage_results/leakage_${TAG}.json leakage_results/vectors_${TAG}.npz \
         provenance/s6eval_${TAG}.json; do
  if [ -e "$W/$f" ]; then echo "FATAL: output exists: $f"; exit 1; fi
done
export MASTER_PORT=$(( 15600 + T ))

# the objective is NOT in the weights; the sidecar is the record
SIDE=${CKPT}.objective.json
if [ ! -s "$SIDE" ]; then echo "FATAL: objective sidecar missing: $SIDE"; exit 1; fi
python - <<PY
import json, sys
s = json.load(open('$SIDE'))
if s.get('arm') != '$ARM':
    sys.exit('FATAL: sidecar says %r, task expects $ARM' % s.get('arm'))
print('[sidecar] arm=%s fm_source=%s seed=%s' % (s['arm'], s['fm_source'], s['seed']))
PY

echo "[$(date)] task $T  $TAG"
if [ "$ARM" = "ddim" ]; then
  python leakage_test_base_sciplex_array.py "$SPLIT" "$CKPT" "$TAG"
else
  python leakage_test_fm_sciplex.py "$SPLIT" "$CKPT" "$TAG" "$ARM" "$NFE" "$SOLVER"
fi

python - <<PY
import json, os, hashlib, datetime, subprocess, sys, torch
import numpy as np
j = 'leakage_results/leakage_$TAG.json'; v = 'leakage_results/vectors_$TAG.npz'
for f in (j, v):
    if not os.path.isfile(f) or os.path.getsize(f) == 0:
        sys.exit('FATAL: missing or empty ' + f)
d = np.load(v, allow_pickle=True)
for k in ('true','A','B','C'):
    if k not in d.files: sys.exit('FATAL: %s absent' % k)
    if not np.all(np.isfinite(d[k])): sys.exit('FATAL: non-finite %s' % k)
n = len(d['cond'])
if n != 443: sys.exit('FATAL: %d conditions, expected 443' % n)
if float(np.asarray(d['B']).min()) < 0: sys.exit('FATAL: negative predicted expression')
r = lambda c: subprocess.run(c, shell=True, capture_output=True, text=True).stdout.strip()
sha = lambda p: hashlib.sha256(open(p,'rb').read()).hexdigest()
json.dump(dict(status='completed', arm='$ARM', seed='$SEED', split='$SPLIT', tag='$TAG',
               nfe=int('$NFE'), solver='$SOLVER',
               prereg='PREREG_step6_flow_matching.md', commit=r('git rev-parse HEAD'),
               completed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
               checkpoint='$CKPT', checkpoint_sha256=sha('$CKPT'),
               vectors_npz=v, vectors_sha256=sha(v), n_conditions=int(n),
               node=r('hostname'), gpu=torch.cuda.get_device_name(0),
               harness_sha256=sha('leakage_test_base_sciplex_array.py'),
               sampler_sha256=sha('sample_squidiff.py')),
          open('provenance/s6eval_$TAG.json','w'), indent=2)
print('[post] %d conditions, finite, non-negative, node %s' % (n, r('hostname')))
PY
echo "[$(date)] task $T  $TAG  DONE"
