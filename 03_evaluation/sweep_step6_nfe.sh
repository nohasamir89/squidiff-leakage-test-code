#!/bin/bash
#SBATCH --job-name=s6_sweep
#SBATCH --partition=nvidia
#SBATCH --nodelist=cn274
#SBATCH --gres=gpu:h100:1
#SBATCH -q a2s2
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=24:00:00
#SBATCH --array=0-39%2
#SBATCH --output=/scratch/ni2071/Diffusion/Squidiff_git/Squidiff/logger_files/s6sw_%A_%a.out
#SBATCH --error=/scratch/ni2071/Diffusion/Squidiff_git/Squidiff/logger_files/s6sw_%A_%a.err
# SECONDARY, non-gated. Cannot alter the section 4 verdict.
#
# The FM step sweep was pre-registered in section 3. The DDIM sweep was NOT:
# it is added after seeing B-A and is recorded as post-hoc.
#
# Purpose: B-A compared FM at 20 NFE against DDIM at 1000, so the objective and
# the sampling budget varied together. This does not isolate the objective -
# the arms also differ in target parameterisation, path and source - but it
# determines whether B-A depends materially on sampling budget.
#
# 0-19  ddim  seeds 0-4 x {1000, 250, 50, 20}
# 20-39 fm_gaussian seeds 0-4 x {50, 20, 10, 5}
set -eo pipefail
module load cuda/12.2.0
source /share/apps/NYUAD5/miniconda/3-4.11.0/etc/profile.d/conda.sh
conda activate /scratch/ni2071/conda_envs/squidiff
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export SQUIDIFF_SAVE_PER_CELL=1
W=/scratch/ni2071/Diffusion/Squidiff_git/Squidiff
SPLIT=drug_split_1
cd "$W"

T=$SLURM_ARRAY_TASK_ID
case "$T" in *[!0-9]*|"") echo "FATAL: non-numeric task id: $T"; exit 1 ;; esac
if [ "$T" -lt 0 ] || [ "$T" -gt 39 ]; then echo "FATAL: task out of range: $T"; exit 1; fi
if [ -n "$(git status --porcelain -- leakage_test_base_sciplex_array.py sample_squidiff.py Squidiff)" ]; then
  echo "FATAL: evaluation sources dirty"; exit 1
fi
echo "[node] $(hostname)  [commit] $(git rev-parse --short HEAD)"

if [ "$T" -lt 20 ]; then
  ARM=ddim; SEED=$(( T % 5 ))
  case $(( T / 5 )) in
    0) STEPS=1000 ;; 1) STEPS=250 ;; 2) STEPS=50 ;; 3) STEPS=20 ;;
  esac
else
  U=$(( T - 20 )); ARM=fm_gaussian; SEED=$(( U % 5 ))
  case $(( U / 5 )) in
    0) STEPS=50 ;; 1) STEPS=20 ;; 2) STEPS=10 ;; 3) STEPS=5 ;;
  esac
fi
CKPT=$W/step6/${ARM}/seed_${SEED}/model_0.9999.pt
TAG=sw_${ARM}_seed_${SEED}_n${STEPS}
if [ ! -s "$CKPT" ]; then echo "FATAL: checkpoint missing: $CKPT"; exit 1; fi
if [ -e "$W/leakage_results/vectors_${TAG}.npz" ]; then echo "FATAL: output exists"; exit 1; fi
export MASTER_PORT=$(( 16800 + T ))
echo "[sweep] arm=$ARM seed=$SEED steps=$STEPS"

echo "[$(date)] task $T  $TAG"
if [ "$ARM" = "ddim" ]; then
  python leakage_test_ddim_nfe.py "$SPLIT" "$CKPT" "$TAG" "$STEPS"
else
  python leakage_test_fm_sciplex.py "$SPLIT" "$CKPT" "$TAG" "$ARM" "$STEPS" euler
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
if len(d['cond']) != 443: sys.exit('FATAL: %d conditions' % len(d['cond']))
for k in ('true','A','B','C'):
    if not np.all(np.isfinite(d[k])): sys.exit('FATAL: non-finite %s' % k)
sha = lambda q: hashlib.sha256(open(q,'rb').read()).hexdigest()
r = lambda c: subprocess.run(c, shell=True, capture_output=True, text=True).stdout.strip()
json.dump(dict(status='completed', analysis='SECONDARY non-gated step sweep',
               ddim_sweep_posthoc=('$ARM' == 'ddim'),
               arm='$ARM', seed='$SEED', steps=int('$STEPS'), tag='$TAG',
               completed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
               commit=r('git rev-parse HEAD'), node=r('hostname'),
               gpu=torch.cuda.get_device_name(0),
               vectors_npz=v, vectors_sha256=sha(v),
               per_cell_npz=pc, per_cell_sha256=sha(pc)),
          open('provenance/s6sw_$TAG.json','w'), indent=2)
print('[post] $ARM seed $SEED at $STEPS steps: 443 conditions, finite')
PY
echo "[$(date)] task $T  $TAG  DONE"
