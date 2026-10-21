#!/bin/bash
#SBATCH --job-name=s13b_eval
#SBATCH --partition=condo
#SBATCH --nodelist=cn274
#SBATCH -q a2s2
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=06:00:00
#SBATCH --array=0-19
#SBATCH --output=/scratch/ni2071/Diffusion/Squidiff_git/Squidiff/logger_files/s13bev_%A_%a.out
#SBATCH --error=/scratch/ni2071/Diffusion/Squidiff_git/Squidiff/logger_files/s13bev_%A_%a.err
# Step 13b evaluation under PREREG_step13b_confirmation.md: every task is Gaussian-source
# flow matching, 50 NFE Euler, NO guidance, pinned to cn274, per-cell storage on.
# Task T: split = T / 10 (drug_split_0, drug_split_2); within a split arm = (T % 10) / 5
# (attn_cross, mlp_ctrl), seed = T % 5. The reference is not re-evaluated: Step 11 vectors.
set -eo pipefail
module load cuda/12.2.0
source /share/apps/NYUAD5/miniconda/3-4.11.0/etc/profile.d/conda.sh
conda activate /scratch/ni2071/conda_envs/squidiff
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export SQUIDIFF_SAVE_PER_CELL=1
unset S12_GUIDANCE_W          # unguided code path; the sampler prints guidance_w=None and the post-check requires it
export S12_GUIDE_TMIN=0.0
W=/scratch/ni2071/Diffusion/Squidiff_git/Squidiff
cd "$W"
NFE=50; PREREG=PREREG_step13b_confirmation.md

T=$SLURM_ARRAY_TASK_ID
case "$T" in *[!0-9]*|"") echo "FATAL: non-numeric task id: $T"; exit 1 ;; esac
if [ "$T" -gt 19 ]; then echo "FATAL: task out of range: $T"; exit 1; fi
N=$(grep -c '^## ' $PREREG)
if [ "$N" -lt 9 ]; then echo "FATAL: prereg has $N sections, expected at least 9"; exit 1; fi
git ls-files --error-unmatch analyse_step13b.py >/dev/null 2>&1 || { echo "FATAL: analysis script not committed (PREREG section 8)"; exit 1; }
git diff --quiet -- $PREREG analyse_step13b.py leakage_test_fm_sciplex.py sample_squidiff.py Squidiff/flow_matching.py Squidiff/MLPModel.py Squidiff/drug_attention.py \
  || { echo "FATAL: prereg, analysis or sampler sources uncommitted"; exit 1; }
COMMIT=$(git rev-parse HEAD)

case $(( T / 10 )) in 0) SPLIT=drug_split_0; EXPECT_N=444 ;; 1) SPLIT=drug_split_2; EXPECT_N=455 ;; esac
R=$(( T % 10 )); SEED=$(( R % 5 )); MEMSZ=0
case $(( R / 5 )) in 0) ARM=attn_cross ;; 1) ARM=mlp_ctrl ;; esac
CKPT=$W/step13b/${SPLIT}_${ARM}/seed_${SEED}/model_0.9999.pt
TAG=s13b_${SPLIT}_${ARM}_seed_${SEED}
SIDE=${CKPT}.encoding.json
if [ ! -s "$CKPT" ]; then echo "FATAL: checkpoint missing: $CKPT"; exit 1; fi
if [ ! -s "$SIDE" ]; then echo "FATAL: encoding sidecar missing: $SIDE"; exit 1; fi
if [ -e "$W/leakage_results/vectors_${TAG}.npz" ]; then echo "FATAL: output exists: $TAG"; exit 1; fi
export MASTER_PORT=$(( 18800 + T ))

python - <<PY
import json, sys
s = json.load(open('$SIDE'))
if s.get('encoder_arm') != '$ARM' or s.get('arm') != '$ARM' or int(s.get('seed', -1)) != int('$SEED') or s.get('split') != '$SPLIT':
    sys.exit('FATAL: sidecar says arm=%r/%r split=%r seed=%r, task expects $ARM $SPLIT seed $SEED'
             % (s.get('arm'), s.get('encoder_arm'), s.get('split'), s.get('seed')))
if float(s.get('cond_drop', 1.0)) != 0.0: sys.exit('FATAL: sidecar cond_drop=%r, expected 0.0' % s.get('cond_drop'))
if s.get('drug_encoding') != 'fingerprint': sys.exit('FATAL: sidecar drug_encoding=%r' % s.get('drug_encoding'))
if int(s.get('memory_size', 0)) != int('$MEMSZ'): sys.exit('FATAL: sidecar memory_size=%r, expected $MEMSZ' % s.get('memory_size'))
print('[sidecar] arm=%s split=%s seed=%s n_params=%s memory_size=%s trained_at=%s' % (s['arm'], s['split'], s['seed'], s.get('n_params'), s.get('memory_size'), str(s.get('commit', '?'))[:7]))
PY

echo "[prereg] $N sections | [commit] $COMMIT | [node] $(hostname)"
echo "[$(date)] task $T  $TAG  arm=$ARM  guidance=none"
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
if n != int('$EXPECT_N'): sys.exit('FATAL: %d conditions, expected $EXPECT_N' % n)
for k in ('true', 'A', 'B', 'C'):
    if not np.all(np.isfinite(d[k])): sys.exit('FATAL: non-finite %s' % k)
if float(np.asarray(d['B']).min()) < 0: sys.exit('FATAL: negative predicted expression')
p = np.load(pc, allow_pickle=True)
if len(p['pred']) != int('$EXPECT_N'): sys.exit('FATAL: per-cell has %d conditions' % len(p['pred']))
nc = [np.asarray(a).shape[0] for a in p['pred']]
log = open('logger_files/s13bev_${SLURM_ARRAY_JOB_ID}_${T}.out').read()
m = re.search(r'encoder arm detected from checkpoint: (\S+) \(memory_size=(\d+)\)', log)
if not m: sys.exit('FATAL: the sampler did not print the detected encoder arm')
if m.group(1) != '$ARM' or int(m.group(2)) != int('$MEMSZ'):
    sys.exit('FATAL: sampler detected arm %s memory_size %s, task expects $ARM $MEMSZ' % (m.group(1), m.group(2)))
g = re.search(r'\[step12\] guidance_w=(\S+)\s+guide_t_min=([0-9.]+)', log)
if not g or g.group(1) != 'None': sys.exit('FATAL: sampler guidance setting is not None: %r' % (g.group(1) if g else None))
side = json.load(open('$SIDE'))
sha = lambda q: hashlib.sha256(open(q, 'rb').read()).hexdigest()
r = lambda c: subprocess.run(c, shell=True, capture_output=True, text=True).stdout.strip()
json.dump(dict(status='completed', split='$SPLIT', arm='$ARM', encoder_arm='$ARM', memory_size=int('$MEMSZ'),
               n_params=side.get('n_params'), smiles_sha256=side.get('smiles_sha256'), cond_drop=0.0, drug_encoding='fingerprint',
               guidance_w=None, guide_t_min=0.0, seed='$SEED', tag='$TAG', nfe=int('$NFE'), sampler='euler_ode',
               prereg='$PREREG', commit='$COMMIT', training_commit=side.get('commit'),
               completed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
               checkpoint='$CKPT', checkpoint_sha256=sha('$CKPT'),
               vectors_npz=v, vectors_sha256=sha(v), per_cell_npz=pc, per_cell_sha256=sha(pc),
               n_conditions=int(n), cells_min=int(min(nc)), cells_max=int(max(nc)),
               node=r('hostname'), gpu=torch.cuda.get_device_name(0),
               harness_sha256=sha('leakage_test_base_sciplex_array.py'), fm_eval_sha256=sha('leakage_test_fm_sciplex.py'),
               flow_matching_sha256=sha('Squidiff/flow_matching.py'), sampler_sha256=sha('sample_squidiff.py'),
               drug_attention_sha256=sha('Squidiff/drug_attention.py'), mlpmodel_sha256=sha('Squidiff/MLPModel.py')),
          open('provenance/s13bev_$TAG.json', 'w'), indent=2)
print('[post] %d conditions, cells %d-%d, finite, non-negative, arm %s bound, guidance None' % (n, min(nc), max(nc), '$ARM'))
PY
echo "[$(date)] task $T  $TAG  DONE"
