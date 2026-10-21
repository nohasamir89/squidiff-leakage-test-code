#!/bin/bash
#SBATCH --job-name=sciplex_repl_eval
#SBATCH --partition=nvidia
#SBATCH --gres=gpu:h100:1
##SBATCH -q a2s2
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=04:00:00
#SBATCH --array=0-19
#SBATCH --output=/scratch/ni2071/Diffusion/Squidiff_git/Squidiff/logger_files/repleval_%A_%a.out
#SBATCH --error=/scratch/ni2071/Diffusion/Squidiff_git/Squidiff/logger_files/repleval_%A_%a.err

# Evaluate all 20 replication checkpoints with the SAME harness used for Step 4.
# 0-4  drug_split_0 base | 5-9  drug_split_0 adaln
# 10-14 drug_split_2 base | 15-19 drug_split_2 adaln
# Vector generation only. No replication metric is computed here.

set -eo pipefail
module load cuda/12.2.0
source /share/apps/NYUAD5/miniconda/3-4.11.0/etc/profile.d/conda.sh
conda activate /scratch/ni2071/conda_envs/squidiff

WORKDIR=/scratch/ni2071/Diffusion/Squidiff_git/Squidiff
FREEZE_TAG=step4-replication-freeze
cd "$WORKDIR"

if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then echo "FATAL: needs a SLURM array context"; exit 1; fi
T=$SLURM_ARRAY_TASK_ID
if [ "$T" -lt 0 ] || [ "$T" -gt 19 ]; then echo "FATAL: task out of range: $T"; exit 1; fi

FROZEN=$(git rev-parse "${FREEZE_TAG}^{commit}")
for f in leakage_test_base_sciplex_array.py sample_squidiff.py; do
  if ! git diff --quiet "$FROZEN" -- "$f"; then
    echo "FATAL: $f differs from $FREEZE_TAG - all 20 must share one harness"; exit 1
  fi
done
if [ -n "$(git status --porcelain -- leakage_test_base_sciplex_array.py sample_squidiff.py Squidiff)" ]; then
  echo "FATAL: evaluation sources modified or untracked"
  git status --porcelain -- leakage_test_base_sciplex_array.py sample_squidiff.py Squidiff
  exit 1
fi
AUDIT=$WORKDIR/provenance/replication_pair_audit.json
[ -s "$AUDIT" ] || { echo "FATAL: pair audit missing: $AUDIT"; exit 1; }
python -c "
import json,sys
d=json.load(open('$AUDIT'))
if d.get('verdict')!='PASS': sys.exit('FATAL: pair audit verdict %r' % d.get('verdict'))
print('[audit] cross-arm protocol audit PASS, %d pairs' % d['n_pairs'])"
echo "[freeze] evaluation harness identical to $FREEZE_TAG ($FROZEN)"

SEED=$(( T % 5 ))
ARMS=(base adaln); ARM=${ARMS[$(( (T / 5) % 2 ))]}
SPLITS=(drug_split_0 drug_split_2); SPLIT=${SPLITS[$(( T / 10 ))]}
if [ "$ARM" = "base" ]; then
  CKPT=$WORKDIR/replication_base/${SPLIT}_base/seed_${SEED}/model_0.9999.pt
  EXPECT=54229254
else
  CKPT=$WORKDIR/replication/${SPLIT}_adaln/seed_${SEED}/model_0.9999.pt
  EXPECT=58186374
fi
TAG=repl_${SPLIT}_${ARM}_seed_${SEED}
[ -s "$CKPT" ] || { echo "FATAL: checkpoint missing or empty: $CKPT"; exit 1; }
for f in leakage_results/leakage_${TAG}.json leakage_results/vectors_${TAG}.npz \
         provenance/repleval_${TAG}.json; do
  [ -e "$WORKDIR/$f" ] && { echo "FATAL: output exists: $f"; exit 1; }
done
export MASTER_PORT=$(( 12700 + T ))

python - <<PY
import torch, sys
sd = torch.load('$CKPT', map_location='cpu', weights_only=True)
mod = sum(1 for k in sd if 'adaln_mod' in k)
n = int(sum(v.numel() for v in sd.values() if hasattr(v,'numel')))
arch = 'adaln' if mod else 'base'
print('[precheck] $TAG: %d params, %d adaln_mod -> %s' % (n, mod, arch))
if arch != '$ARM': sys.exit('FATAL: checkpoint is %s, task expects $ARM' % arch)
if n != $EXPECT: sys.exit('FATAL: expected $EXPECT params, got %d' % n)
print('[precheck] PASS')
PY

python -c "from Squidiff import MLPModel; print('[import] ' + MLPModel.__file__)"

python - <<PY > "$WORKDIR/provenance/repleval_${TAG}.json"
import json, subprocess, os, hashlib, datetime, torch, sys
r = lambda c: subprocess.run(c, shell=True, capture_output=True, text=True).stdout.strip()
def h(p):
    m = hashlib.sha256()
    with open(p,'rb') as f:
        for c in iter(lambda: f.read(1<<20), b''): m.update(c)
    return m.hexdigest()
gpu = torch.cuda.get_device_name(0)
if 'H100' not in gpu.upper(): sys.exit('FATAL: expected H100, got %r' % gpu)
W='$WORKDIR'
print(json.dumps({
    'status':'started', 'kind':'replication_evaluation',
    'started_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'prereg':'PREREG_step4_replication.md', 'freeze_tag':'$FREEZE_TAG',
    'commit': r('git rev-parse HEAD'),
    'split':'$SPLIT', 'arm':'$ARM', 'seed':'$SEED', 'tag':'$TAG',
    'checkpoint':'$CKPT', 'checkpoint_sha256': h('$CKPT'),
    'slurm_array_job': os.environ.get('SLURM_ARRAY_JOB_ID'),
    'slurm_task': os.environ.get('SLURM_ARRAY_TASK_ID'),
    'node': r('hostname'), 'gpu': gpu,
    'torch': torch.__version__, 'cuda': torch.version.cuda,
    'cudnn': torch.backends.cudnn.version(),
    'harness_sha256': h(W+'/leakage_test_base_sciplex_array.py'),
    'sampler_sha256': h(W+'/sample_squidiff.py'),
}, indent=2))
PY

echo "[$(date)] task $T  $TAG"
python leakage_test_base_sciplex_array.py "$SPLIT" "$CKPT" "$TAG"

python - <<PY
import json, os, hashlib, datetime, numpy as np, sys
p='$WORKDIR/provenance/repleval_${TAG}.json'
j='$WORKDIR/leakage_results/leakage_${TAG}.json'
v='$WORKDIR/leakage_results/vectors_${TAG}.npz'
for f in (j, v):
    if not os.path.isfile(f) or os.path.getsize(f)==0:
        sys.exit('FATAL: expected output missing or empty: '+f)
d=np.load(v, allow_pickle=True); n=len(d['cond'])
for k in ('true','B','C'):
    if k not in d.files: sys.exit('FATAL: %s absent from vectors' % k)
    if not np.all(np.isfinite(d[k])): sys.exit('FATAL: non-finite values in %s' % k)
rec=json.load(open(p))
rec.update({'status':'completed',
            'completed_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'n_conditions': int(n), 'leakage_json': j, 'vectors_npz': v,
            'vectors_sha256': hashlib.sha256(open(v,'rb').read()).hexdigest()})
tmp=p+'.tmp'; json.dump(rec, open(tmp,'w'), indent=2); os.replace(tmp, p)
print('[eval] %d conditions, all finite' % n)
PY

echo "[$(date)] task $T  $TAG  DONE"
