#!/bin/bash
#SBATCH --job-name=s12b_cfg
#SBATCH --partition=condo
#SBATCH --nodelist=cn274
#SBATCH -q a2s2
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=06:00:00
#SBATCH --array=0-49
#SBATCH --output=/scratch/ni2071/Diffusion/Squidiff_git/Squidiff/logger_files/s12bev_%A_%a.out
#SBATCH --error=/scratch/ni2071/Diffusion/Squidiff_git/Squidiff/logger_files/s12bev_%A_%a.err
# Step 12b evaluation under PREREG_step12b_confirmation.md. Every task: Gaussian-
# source flow matching, 50 NFE Euler, pinned to cn274, per-cell storage on.
# Task T: split = T / 25 (0 -> drug_split_0, 1 -> drug_split_2); within a split,
# block = (T % 25) / 5 and seed = T % 5:
#   block 0  unguided code path (S12_GUIDANCE_W unset)   tag wnone
#   block 1  guided path at w=1, identity check          tag w1
#   block 2  w=2                                  GATED  tag w2
#   block 3  w=3                            EXPLORATORY  tag w3
#   block 4  w=2, guidance only for t>=0.5  EXPLORATORY  tag w2h
# Blocks 2-4 refuse to start unless provenance/s12b_identity_<split>_seed_<k>.json
# says "pass": true. Submit in two batches:
#   sbatch --array=0-9,25-34 eval_step12b.sh      (blocks 0-1, both splits)
#   python check_s12b_identity.py
#   sbatch --array=10-24,35-49 eval_step12b.sh    (blocks 2-4, both splits)
set -eo pipefail
module load cuda/12.2.0
source /share/apps/NYUAD5/miniconda/3-4.11.0/etc/profile.d/conda.sh
conda activate /scratch/ni2071/conda_envs/squidiff
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export SQUIDIFF_SAVE_PER_CELL=1
W=/scratch/ni2071/Diffusion/Squidiff_git/Squidiff
cd "$W"
NFE=50; PREREG=PREREG_step12b_confirmation.md

T=$SLURM_ARRAY_TASK_ID
case "$T" in *[!0-9]*|"") echo "FATAL: non-numeric task id: $T"; exit 1 ;; esac
if [ "$T" -gt 49 ]; then echo "FATAL: task out of range: $T"; exit 1; fi
N=$(grep -c '^## ' $PREREG)
if [ "$N" -lt 9 ]; then echo "FATAL: prereg has $N sections, expected at least 9"; exit 1; fi
git ls-files --error-unmatch analyse_step12b.py check_s12b_identity.py >/dev/null 2>&1 || { echo "FATAL: analysis scripts not committed (PREREG section 8)"; exit 1; }
git diff --quiet -- $PREREG analyse_step12b.py check_s12b_identity.py leakage_test_fm_sciplex.py Squidiff/flow_matching.py \
  || { echo "FATAL: prereg, analysis or sampler sources uncommitted"; exit 1; }
COMMIT=$(git rev-parse HEAD)

case $(( T / 25 )) in 0) SPLIT=drug_split_0; EXPECT_N=444 ;; 1) SPLIT=drug_split_2; EXPECT_N=455 ;; esac
R=$(( T % 25 )); SEED=$(( R % 5 )); BLK=$(( R / 5 ))
unset S12_GUIDANCE_W
export S12_GUIDE_TMIN=0.0
case $BLK in
  0) WTAG=wnone ;;
  1) WTAG=w1;   export S12_GUIDANCE_W=1 ;;
  2) WTAG=w2;   export S12_GUIDANCE_W=2 ;;
  3) WTAG=w3;   export S12_GUIDANCE_W=3 ;;
  4) WTAG=w2h;  export S12_GUIDANCE_W=2; export S12_GUIDE_TMIN=0.5 ;;
esac
if [ "$BLK" -ge 2 ]; then
  ID=provenance/s12b_identity_${SPLIT}_seed_${SEED}.json
  if [ ! -s "$ID" ] || ! grep -q '"pass": true' "$ID"; then
    echo "FATAL: identity check not passed for $SPLIT seed $SEED ($ID); run check_s12b_identity.py first"; exit 1; fi
fi
CKPT=$W/step12b/${SPLIT}_cfg_fingerprint/seed_${SEED}/model_0.9999.pt
TAG=s12b_${SPLIT}_${WTAG}_seed_${SEED}
SIDE=${CKPT}.encoding.json
if [ ! -s "$CKPT" ]; then echo "FATAL: checkpoint missing: $CKPT"; exit 1; fi
if [ ! -s "$SIDE" ]; then echo "FATAL: encoding sidecar missing: $SIDE"; exit 1; fi
if [ -e "$W/leakage_results/vectors_${TAG}.npz" ]; then echo "FATAL: output exists: $TAG"; exit 1; fi
export MASTER_PORT=$(( 17700 + T ))

python - <<PY
import json, sys
s = json.load(open('$SIDE'))
if s.get('arm') != 'cfg_fingerprint' or int(s.get('seed', -1)) != int('$SEED') or s.get('split') != '$SPLIT':
    sys.exit('FATAL: sidecar says arm=%r split=%r seed=%r, task expects cfg_fingerprint $SPLIT seed $SEED'
             % (s.get('arm'), s.get('split'), s.get('seed')))
if abs(float(s.get('cond_drop', 0.0)) - 0.1) > 1e-9: sys.exit('FATAL: sidecar cond_drop=%r, expected 0.1' % s.get('cond_drop'))
if s.get('drug_encoding') != 'fingerprint': sys.exit('FATAL: sidecar drug_encoding=%r, expected fingerprint' % s.get('drug_encoding'))
print('[sidecar] arm=%s split=%s seed=%s cond_drop=%s trained_at=%s' % (s['arm'], s['split'], s['seed'], s['cond_drop'], str(s.get('commit', '?'))[:7]))
PY

echo "[prereg] $N sections | [commit] $COMMIT | [node] $(hostname)"
echo "[$(date)] task $T  $TAG  guidance_w=${S12_GUIDANCE_W:-none}  guide_t_min=$S12_GUIDE_TMIN"
python leakage_test_fm_sciplex.py "$SPLIT" "$CKPT" "$TAG" fm_gaussian "$NFE" euler

python - <<PY
import json, os, sys, re, hashlib, subprocess, datetime, torch
import numpy as np
v  = 'leakage_results/vectors_$TAG.npz'
pc = 'leakage_results/vectors_${TAG}_pc.npz'
for f in (v, pc):
    if not os.path.isfile(f) or os.path.getsize(f) == 0: sys.exit('FATAL: missing or empty ' + f)
d = np.load(v, allow_pickle=True)
n = len(d['cond'])
if n != int('$EXPECT_N'): sys.exit('FATAL: %d conditions, expected $EXPECT_N for $SPLIT' % n)
for k in ('true', 'A', 'B', 'C'):
    if not np.all(np.isfinite(d[k])): sys.exit('FATAL: non-finite %s' % k)
if float(np.asarray(d['B']).min()) < 0: sys.exit('FATAL: negative predicted expression')
p = np.load(pc, allow_pickle=True)
if len(p['pred']) != int('$EXPECT_N'): sys.exit('FATAL: per-cell has %d conditions' % len(p['pred']))
nc = [np.asarray(a).shape[0] for a in p['pred']]
log = open('logger_files/s12bev_${SLURM_ARRAY_JOB_ID}_${T}.out').read()
m = re.search(r'\[step12\] guidance_w=(\S+)\s+guide_t_min=([0-9.]+)', log)
if not m: sys.exit('FATAL: sampler did not print its guidance setting')
gw_env = os.environ.get('S12_GUIDANCE_W')
gw = None if gw_env in (None, '', 'none') else float(gw_env)
printed_w = None if m.group(1) == 'None' else float(m.group(1))
if (gw is None) != (printed_w is None) or (gw is not None and abs(gw - printed_w) > 1e-9):
    sys.exit('FATAL: sampler used guidance_w=%r, task set %r' % (printed_w, gw))
if abs(float(m.group(2)) - float('$S12_GUIDE_TMIN')) > 1e-9:
    sys.exit('FATAL: sampler used guide_t_min=%s, task set $S12_GUIDE_TMIN' % m.group(2))
sha = lambda q: hashlib.sha256(open(q, 'rb').read()).hexdigest()
r = lambda c: subprocess.run(c, shell=True, capture_output=True, text=True).stdout.strip()
json.dump(dict(status='completed', split='$SPLIT', arm='cfg_fingerprint', arm_key='cfg', wtag='$WTAG',
               cond_drop=0.1, drug_encoding='fingerprint', guidance_w=gw, guide_t_min=float('$S12_GUIDE_TMIN'),
               null_definition='all-zero fingerprint, same control cells', seed='$SEED', tag='$TAG',
               nfe=int('$NFE'), sampler='euler_ode', prereg='$PREREG', commit='$COMMIT',
               completed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
               checkpoint='$CKPT', checkpoint_sha256=sha('$CKPT'),
               vectors_npz=v, vectors_sha256=sha(v), per_cell_npz=pc, per_cell_sha256=sha(pc),
               n_conditions=int(n), cells_min=int(min(nc)), cells_max=int(max(nc)),
               node=r('hostname'), gpu=torch.cuda.get_device_name(0),
               harness_sha256=sha('leakage_test_base_sciplex_array.py'),
               fm_eval_sha256=sha('leakage_test_fm_sciplex.py'),
               flow_matching_sha256=sha('Squidiff/flow_matching.py'),
               sampler_sha256=sha('sample_squidiff.py')),
          open('provenance/s12bev_$TAG.json', 'w'), indent=2)
print('[post] %d conditions, cells %d-%d, finite, non-negative, guidance_w=%r bound' % (n, min(nc), max(nc), gw))
PY
echo "[$(date)] task $T  $TAG  DONE"
