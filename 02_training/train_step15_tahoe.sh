#!/bin/bash
#SBATCH --job-name=s15_train
#SBATCH --partition=condo
#SBATCH -q a2s2
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=150G
#SBATCH --time=24:00:00
#SBATCH --array=0-44
#SBATCH --output=/scratch/ni2071/Diffusion/Squidiff_git/Squidiff/logger_files/s15tr_%A_%a.out
#SBATCH --error=/scratch/ni2071/Diffusion/Squidiff_git/Squidiff/logger_files/s15tr_%A_%a.err
# Step 15b (Tahoe brain/gut) training under PREREG_step15b_tahoe_model.md: FIVE arms x THREE seeds x THREE splits = 45 tasks.
#   task T:  split = T / 15  (0 -> drug_split_0, 1 -> drug_split_1, 2 -> drug_split_2)
#            within = T % 15;  arm = within / 3  (0 base, 1 recipe, 2 guidance_w2, 3 attn_cross, 4 mlp_ctrl);  seed = within % 3
# All arms: 100,000 steps, fingerprint drug encoding, 200 genes, on the Step 15a files (tahoe_{train}_drug_split_<k>{,_control}.h5ad).
#   base        : parser defaults (DDPM, no adaLN, fm_source none, cond_drop 0)
#   recipe      : + adaLN + Gaussian flow matching (cond_drop 0)              [Step 6/11]
#   guidance_w2 : recipe + cond_drop 0.1 (guided only at sampling time)        [Step 12b]  -- checkpoint differs from recipe by dropout
#   attn_cross  : recipe + --encoder_arm attn_cross (cond_drop 0)             [Step 13]  -- EXPECT params 58,532,804 (identical arch to sci-Plex, 200 genes)
#   mlp_ctrl    : recipe + --encoder_arm mlp_ctrl  (cond_drop 0)             [Step 13]  -- EXPECT params 58,532,300
# The parameter count is independent of the number of cell lines: the model reads the control cell's expression, not a per-line
# embedding (findings: base encoder concatenates control_feature + drug_dose; group label discarded, num_classes=None).
# DRYRUN=1 SLURM_ARRAY_TASK_ID=<T> bash train_step15_tahoe.sh  runs every guard, prints the parameter count, and stops.
# 2026-09-28 revision (record): array 18226073 completed 33/45 tasks; tasks 3-14 (split 0, non-base arms) were killed by the
#   memory cgroup after a 48 GB reservation was applied at runtime (RSS 46-50 GB at 27-57 min; same arms finished at 19-45 GB
#   on splits 1/2). They are re-run from scratch with sbatch --array=3-14 on the same partition/QOS with the original
#   150 GB, RSS sampled every 30 s to <outdir>/rss_kb.log, and the post block replaced by finalize_step15.py (the original
#   post block compared the state_dict size, parameters + buffers, with the parameters-only EXPECT and failed on every task).
set -eo pipefail
module load cuda/12.2.0
source /share/apps/NYUAD5/miniconda/3-4.11.0/etc/profile.d/conda.sh
conda activate /scratch/ni2071/conda_envs/squidiff
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
W=/scratch/ni2071/Diffusion/Squidiff_git/Squidiff
D=/scratch/ni2071/Diffusion/tahoe/datasets_step15
PREREG=PREREG_step15b_tahoe_model.md
cd "$W"

T=$SLURM_ARRAY_TASK_ID
case "$T" in *[!0-9]*|"") echo "FATAL: non-numeric task id: $T"; exit 1 ;; esac
if [ "$T" -gt 44 ]; then echo "FATAL: task out of range (0-44): $T"; exit 1; fi

# --- pre-registration and sources committed (section-count guard uses >=, never ==, to survive amendments) ---
if [ ! -f "$PREREG" ]; then echo "FATAL: missing $PREREG"; exit 1; fi
N=$(grep -c '^## ' $PREREG)
if [ "$N" -lt 9 ]; then echo "FATAL: prereg has $N sections, expected at least 9"; exit 1; fi
for f in $PREREG finalize_step15.py train_step15_tahoe.sh; do git ls-files --error-unmatch $f >/dev/null 2>&1 || { echo "FATAL: $f not committed"; exit 1; }; done
git diff --quiet -- $PREREG finalize_step15.py train_step15_tahoe.sh || { echo "FATAL: prereg, finalize or launcher uncommitted"; exit 1; }
if [ -n "$(git status --porcelain -- train_squidiff.py Squidiff)" ]; then
  echo "FATAL: training sources dirty"; git status --porcelain -- train_squidiff.py Squidiff; exit 1
fi
COMMIT=$(git rev-parse HEAD)

# --- decode the task ---
SPLIT_IDX=$(( T / 15 )); WITHIN=$(( T % 15 )); ARM_IDX=$(( WITHIN / 3 )); SEED=$(( WITHIN % 3 ))
SPLIT=drug_split_${SPLIT_IDX}
ENC=fingerprint; CDROP=0.0; EARM=none; ADALN=True; FM=gaussian; EXPECT=0
case $ARM_IDX in
  0) ARM=base;        ADALN=False; FM=none; EXPECT=54221060 ;;        # parser defaults: DDPM, no adaLN
  1) ARM=recipe;      EXPECT=58178180 ;;                              # adaLN + gaussian FM
  2) ARM=guidance_w2; CDROP=0.1; EXPECT=58178180 ;;                   # recipe + condition dropout (same arch as recipe)
  3) ARM=attn_cross;  EARM=attn_cross; EXPECT=58532804 ;;            # recipe + cross-attention encoder
  4) ARM=mlp_ctrl;    EARM=mlp_ctrl;   EXPECT=58532300 ;;            # recipe + parameter-matched control
  *) echo "FATAL: bad arm index $ARM_IDX"; exit 1 ;;
esac
export ARM EXPECT EARM ADALN FM CDROP
TAG=${SPLIT}_${ARM}_seed_${SEED}
OUTDIR=$W/step15/${ARM}/${SPLIT}_seed_${SEED}
LOGDIR=logger_files/logger_s15_${TAG}
PROV=$W/provenance/s15tr_${TAG}.json
TRAIN=$D/tahoe_train_${SPLIT}.h5ad
CTRL=$D/tahoe_train_${SPLIT}_control.h5ad
for f in "$TRAIN" "$CTRL"; do [ -s "$f" ] || { echo "FATAL: missing input $f"; exit 1; }; done
if [ -e "$OUTDIR" ] && [ -n "$(find "$OUTDIR" -mindepth 1 -print -quit 2>/dev/null)" ]; then echo "FATAL: output dir not empty: $OUTDIR"; exit 1; fi
if [ -e "$W/$LOGDIR" ]; then echo "FATAL: logger path exists: $LOGDIR"; exit 1; fi
if [ -e "$PROV" ]; then echo "FATAL: provenance exists: $PROV"; exit 1; fi
export MASTER_PORT=$(( 19000 + T )); export PYTHONHASHSEED=$SEED
echo "[prereg] $N sections committed | [commit] $COMMIT | [node] $(hostname)"
echo "[task] T=$T  split=$SPLIT  arm=$ARM  seed=$SEED  adaln=$ADALN  fm_source=$FM  cond_drop=$CDROP  encoder_arm=$EARM  expect_params=$EXPECT"

# --- shared training arguments; base omits adaLN and flow matching ---
ARGS="--logger_path $LOGDIR --data_path $TRAIN --control_data_path $CTRL --resume_checkpoint $OUTDIR"
ARGS="$ARGS --seed $SEED --use_drug_structure True --drug_encoding $ENC --gene_size 200 --output_dim 200 --lr_anneal_steps 100000"
if [ "$ARM" != base ]; then ARGS="$ARGS --use_adaln True --fm_source gaussian"; fi
if [ "$ARM" = guidance_w2 ]; then ARGS="$ARGS --cond_drop 0.1"; fi
if [ "$EARM" != none ]; then ARGS="$ARGS --encoder_arm $EARM"; fi

# --- guard: parse the arguments, confirm each flag took effect for THIS arm, build the model, print the parameter count ---
python - $ARGS <<'PY'
import sys, os
sys.argv = ['train_squidiff.py'] + sys.argv[1:]
import train_squidiff as t
from Squidiff.script_util import create_model_and_diffusion, model_and_diffusion_defaults, args_to_dict
from Squidiff.scrna_datasets import DRUG_ENCODER_DIMS
a = t.parse_args()
arm, expect, earm = os.environ['ARM'], int(os.environ['EXPECT']), os.environ['EARM']
adaln = os.environ['ADALN'] == 'True'; cdrop = float(os.environ['CDROP']); fm = os.environ['FM']
print('[guard] arm=%s  use_adaln=%r  fm_source=%r  cond_drop=%r (%s)  encoder_arm=%r  drug_encoding=%r'
      % (arm, a['use_adaln'], a['fm_source'], a['cond_drop'], type(a['cond_drop']).__name__, a.get('encoder_arm'), a['drug_encoding']))
# the flags that must have taken effect (the real failure mode: a flag silently not applied)
if a['use_adaln'] is not adaln: sys.exit('FATAL: use_adaln resolved to %r, expected %r' % (a['use_adaln'], adaln))
if a['fm_source'] != fm: sys.exit('FATAL: fm_source %r, expected %r' % (a['fm_source'], fm))
if not isinstance(a['cond_drop'], float) or abs(a['cond_drop'] - cdrop) > 1e-12: sys.exit('FATAL: cond_drop %r, expected %.3f' % (a['cond_drop'], cdrop))
if a['drug_encoding'] != 'fingerprint': sys.exit('FATAL: drug_encoding %r' % a['drug_encoding'])
if earm != 'none' and a.get('encoder_arm') != earm: sys.exit('FATAL: encoder_arm %r, expected %r' % (a.get('encoder_arm'), earm))
a['drug_dimension'] = DRUG_ENCODER_DIMS[a['drug_encoding']]; a['memory_size'] = 0
m, _ = create_model_and_diffusion(**args_to_dict(a, model_and_diffusion_defaults().keys()))
n = sum(p.numel() for p in m.parameters())
print('[guard] arm=%s parameters=%d' % (arm, n))
if expect and n != expect: sys.exit('FATAL: parameter count %d != expected %d for arm %s' % (n, expect, arm))
if earm != 'none':                                                                  # attention arms must carry encoder.arm.* weights
    keys = [k for k in m.state_dict() if k.startswith('encoder.arm.')]
    if not keys: sys.exit('FATAL: no encoder.arm.* parameters for arm %s' % arm)
    print('[guard] arm=%s encoder.arm.* parameter tensors: %d' % (arm, len(keys)))
PY
if [ "${DRYRUN:-0}" = "1" ]; then echo "[dryrun] guards passed for $TAG; nothing created"; exit 0; fi
mkdir -p "$OUTDIR" logger_files provenance

echo "[$(date)] training $TAG"
CG=/sys/fs/cgroup/memory/slurm/uid_$(id -u)/job_${SLURM_JOB_ID}/memory.usage_in_bytes   # what the memory limit is enforced on (cgroup v1 path; NA if absent)
python train_squidiff.py $ARGS &
TRAIN_PID=$!
( while kill -0 "$TRAIN_PID" 2>/dev/null; do echo "$(date +%s) rss_kb=$(ps -o rss= -p "$TRAIN_PID" | tr -d ' ') cgroup_bytes=$( [ -r "$CG" ] && cat "$CG" || echo NA )" >> "$OUTDIR/rss_kb.log"; sleep 30; done ) &
MON_PID=$!
RC=0; wait "$TRAIN_PID" || RC=$?
kill "$MON_PID" 2>/dev/null || true; wait "$MON_PID" 2>/dev/null || true
if [ "$RC" -ne 0 ]; then echo "FATAL: training exited with code $RC (peak sampled rss_kb: $(awk -F'rss_kb=' '{split($2,a," "); if(a[1]+0>m) m=a[1]+0} END{print m}' "$OUTDIR/rss_kb.log"))"; exit "$RC"; fi

# --- post: certify the checkpoint and write sidecar + provenance through the single shared implementation ---
python -u finalize_step15.py --jobid "$SLURM_ARRAY_JOB_ID" --tasks "$T" --expect-arm "$ARM" --expect-split "$SPLIT" --expect-seed "$SEED"
echo "[$(date)] $TAG DONE"
