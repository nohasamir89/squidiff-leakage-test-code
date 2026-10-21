#!/bin/bash
#SBATCH --job-name=sciplex_s5eval
#SBATCH --partition=nvidia
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=12:00:00
#SBATCH --array=0-19
#SBATCH --output=/scratch/ni2071/Diffusion/Squidiff_git/Squidiff/logger_files/s5eval_%A_%a.out
#SBATCH --error=/scratch/ni2071/Diffusion/Squidiff_git/Squidiff/logger_files/s5eval_%A_%a.err
# Evaluate the 20 Step 5 checkpoints. 0-4 base | 5-9 adaln | 10-14 attention | 15-19 mlp
set -eo pipefail
module load cuda/12.2.0
source /share/apps/NYUAD5/miniconda/3-4.11.0/etc/profile.d/conda.sh
conda activate /scratch/ni2071/conda_envs/squidiff
W=/scratch/ni2071/Diffusion/Squidiff_git/Squidiff
SPLIT=drug_split_1
cd "$W"

[ -n "${SLURM_ARRAY_TASK_ID:-}" ] || { echo "FATAL: needs a SLURM array"; exit 1; }
T=$SLURM_ARRAY_TASK_ID
[ "$T" -ge 0 ] && [ "$T" -le 19 ] || { echo "FATAL: task out of range"; exit 1; }

N=$(grep -c '^## ' PREREG_step5_gene_module.md)
[ "$N" -eq 9 ] || { echo "FATAL: prereg has $N sections"; exit 1; }
[ -n "$(git status --porcelain -- sample_squidiff.py leakage_test_base_sciplex_array.py Squidiff)" ] && \
  { echo "FATAL: evaluation sources dirty"; exit 1; }
echo "[freeze] prereg complete, evaluation sources clean, commit $(git rev-parse --short HEAD)"

SEED=$(( T % 5 ))
case $(( T / 5 )) in
  0) ARM=base;      EXPECT=54229254; EGM=0 ;;
  1) ARM=adaln;     EXPECT=58186374; EGM=0 ;;
  2) ARM=attention; EXPECT=61442695; EGM=1 ;;
  3) ARM=mlp;       EXPECT=61442694; EGM=1 ;;
esac
CKPT=$W/step5/${ARM}/seed_${SEED}/model_0.9999.pt
TAG=step5_${ARM}_seed_${SEED}
[ -s "$CKPT" ] || { echo "FATAL: checkpoint missing: $CKPT"; exit 1; }
for f in leakage_results/leakage_${TAG}.json leakage_results/vectors_${TAG}.npz \
         provenance/s5eval_${TAG}.json; do
  [ -e "$W/$f" ] && { echo "FATAL: output exists: $f"; exit 1; }
done
export MASTER_PORT=$(( 14200 + T ))

python - <<PY
import torch, sys
sd = torch.load('$CKPT', map_location='cpu', weights_only=True)
n = int(sum(v.numel() for v in sd.values() if hasattr(v,'numel')))
nad = sum(1 for k in sd if 'adaln_mod' in k)
ngm = sum(1 for k in sd if 'gene_module' in k)
kind = ('attention' if any('gene_module.attn' in k for k in sd) else
        'mlp' if ngm else 'adaln' if nad else 'base')
print('[precheck] $TAG: %d entries, %d adaln, %d gene_module -> %s' % (n, nad, ngm, kind))
if kind != '$ARM': sys.exit('FATAL: checkpoint is %s, expected $ARM' % kind)
if n != $EXPECT: sys.exit('FATAL: expected $EXPECT entries, got %d' % n)
if ($EGM == 0) != (ngm == 0): sys.exit('FATAL: gene_module presence disagrees with arm')
print('[precheck] PASS')
PY

python -c "from Squidiff import MLPModel; print('[import] ' + MLPModel.__file__)"
echo "[$(date)] task $T  $TAG"
python leakage_test_base_sciplex_array.py "$SPLIT" "$CKPT" "$TAG"

python - <<PY
import json, os, hashlib, datetime, subprocess, sys, torch
import numpy as np
j = 'leakage_results/leakage_$TAG.json'; v = 'leakage_results/vectors_$TAG.npz'
for f in (j, v):
    if not os.path.isfile(f) or os.path.getsize(f) == 0:
        sys.exit('FATAL: missing or empty ' + f)
d = np.load(v, allow_pickle=True)
for k in ('true','B','C'):
    if k not in d.files: sys.exit('FATAL: %s absent' % k)
    if not np.all(np.isfinite(d[k])): sys.exit('FATAL: non-finite %s' % k)
n = len(d['cond'])
if n != 443: sys.exit('FATAL: %d conditions, expected 443' % n)
r = lambda c: subprocess.run(c, shell=True, capture_output=True, text=True).stdout.strip()
sha = lambda p: hashlib.sha256(open(p,'rb').read()).hexdigest()
json.dump(dict(status='completed', arm='$ARM', seed='$SEED', split='$SPLIT', tag='$TAG',
               prereg='PREREG_step5_gene_module.md', commit=r('git rev-parse HEAD'),
               completed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
               checkpoint='$CKPT', checkpoint_sha256=sha('$CKPT'),
               vectors_npz=v, vectors_sha256=sha(v), n_conditions=int(n),
               node=r('hostname'), gpu=torch.cuda.get_device_name(0),
               harness_sha256=sha('leakage_test_base_sciplex_array.py'),
               sampler_sha256=sha('sample_squidiff.py')),
          open('provenance/s5eval_$TAG.json','w'), indent=2)
print('[post] %d conditions, all finite' % n)
PY
echo "[$(date)] task $T  $TAG  DONE"
