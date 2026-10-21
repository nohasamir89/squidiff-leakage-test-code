#!/bin/bash
#SBATCH --job-name=s10_eval
#SBATCH --partition=nvidia
#SBATCH --gres=gpu:h100:1
#SBATCH -q nvidia
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=10:00:00
#SBATCH --array=0-14%2
#SBATCH --output=/scratch/ni2071/Diffusion/Squidiff_git/Squidiff/logger_files/s10ev_%A_%a.out
#SBATCH --error=/scratch/ni2071/Diffusion/Squidiff_git/Squidiff/logger_files/s10ev_%A_%a.err
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
N=$(grep -c '^## ' PREREG_step10_chemvae_stochastic.md)
if [ "$N" -lt 8 ]; then echo "FATAL: prereg has $N sections"; exit 1; fi
if [ -n "$(git status --porcelain -- leakage_test_base_sciplex_array.py sample_squidiff.py Squidiff)" ]; then
  echo "FATAL: evaluation sources dirty"; exit 1
fi
echo "[freeze] prereg $N sections, sources clean, commit $(git rev-parse --short HEAD)"
echo "[node] $(hostname)"

SEED=$(( T % 5 ))
SPLIT=drug_split_1
case $(( T / 5 )) in
  0) ARM=fingerprint_s10 ;;
  1) ARM=fp_chemvae ;;
  2) ARM=chemvae ;;
esac
CKPT=$W/step10/${ARM}/seed_${SEED}/model_0.9999.pt
TAG=s10_${ARM}_seed_${SEED}
SIDE=${CKPT}.encoding.json
if [ ! -s "$CKPT" ]; then echo "FATAL: checkpoint missing: $CKPT"; exit 1; fi
if [ ! -s "$SIDE" ]; then echo "FATAL: objective sidecar missing"; exit 1; fi
if [ -e "$W/leakage_results/vectors_${TAG}.npz" ]; then echo "FATAL: output exists"; exit 1; fi
export MASTER_PORT=$(( 17400 + T ))

python - <<PY
import json, sys
s = json.load(open('$SIDE'))
if s.get('arm') != '$ARM':
    sys.exit('FATAL: sidecar says arm=%r, task expects $ARM' % s.get('arm'))
if s.get('drug_encoding') != '$ARM':
    sys.exit('FATAL: sidecar says drug_encoding=%r, task expects $ARM' % s.get('drug_encoding'))
print('[sidecar] arm=%s drug_encoding=%s seed=%s' % (s['arm'], s['drug_encoding'], s['seed']))
PY

echo "[$(date)] task $T  $TAG"
# All three Step 9 arms are Gaussian-source flow matching at 20 NFE; only the
# drug encoding differs, and the sampler infers that from encoder.fc1.weight.
# Amendment 9: stochastic encoders are evaluated over 16 independent draws and
# the PREDICTIONS averaged. Each draw is a SEPARATE process, so nothing is shared
# between them - running the harness in a loop inside one process failed five
# times on namespace collisions, the harness reassigning TAG and other names into
# the caller's scope.
case "$ARM" in
  chemvae|fp_chemvae) N_DRAWS=16 ;;
  *)                  N_DRAWS=1  ;;
esac
echo "[eval] $ARM : $N_DRAWS draw(s)"
if [ "$N_DRAWS" -eq 1 ]; then
  python leakage_test_fm_sciplex.py "$SPLIT" "$CKPT" "$TAG" fm_gaussian "$NFE" euler
else
  for k in $(seq 0 $(( N_DRAWS - 1 ))); do
    echo "[eval] draw $k of $N_DRAWS"
    SQUIDIFF_DRAW_SEED=$(( 1000 + k )) SQUIDIFF_SAVE_PER_CELL=1 \
      python leakage_test_fm_sciplex.py "$SPLIT" "$CKPT" "${TAG}_draw${k}" \
             fm_gaussian "$NFE" euler
  done
  python average_draws.py "$TAG"
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
expect = 443
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
               prereg='PREREG_step10_chemvae_stochastic.md', commit=r('git rev-parse HEAD'),
               completed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
               checkpoint='$CKPT', checkpoint_sha256=sha('$CKPT'),
               vectors_npz=v, vectors_sha256=sha(v),
               per_cell_npz=pc, per_cell_sha256=sha(pc),
               n_conditions=int(n), cells_min=int(min(nc)), cells_max=int(max(nc)),
               node=r('hostname'), gpu=torch.cuda.get_device_name(0),
               harness_sha256=sha('leakage_test_base_sciplex_array.py'),
               sampler_sha256=sha('sample_squidiff.py')),
          open('provenance/s10ev_$TAG.json','w'), indent=2)
print('[post] %d conditions, cells %d-%d, finite, non-negative' % (n, min(nc), max(nc)))
PY
echo "[$(date)] task $T  $TAG  DONE"
