#!/bin/bash
#SBATCH --job-name=s16_train
#SBATCH --partition=condo
#SBATCH -q a2s2
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=150G
#SBATCH --time=24:00:00
#SBATCH --array=0-17
#SBATCH --output=/scratch/ni2071/Diffusion/Squidiff_git/Squidiff/logger_files/s16tr_%A_%a.out
#SBATCH --error=/scratch/ni2071/Diffusion/Squidiff_git/Squidiff/logger_files/s16tr_%A_%a.err
# Step 16 (Tahoe brain/gut, target-aware compound representation) training under PREREG_step16_tahoe_targets.md:
# TWO arms x THREE seeds x THREE splits = 18 tasks.
#   task T:  split = T / 6  (0 -> drug_split_0, 1 -> drug_split_1, 2 -> drug_split_2)
#            within = T % 6;  arm = within / 3  (0 targets, 1 targets_shuf);  seed = within % 3
#   development first:  sbatch --array=6-11 train_step16_tahoe.sh      (drug_split_1, both arms, three seeds)
#   confirmation:       sbatch --array=0-5,12-17 train_step16_tahoe.sh
# Both arms are the Step 15b recipe (adaLN + Gaussian flow matching, cond_drop 0, encoder_arm mlp, 100,000 steps, 200 genes,
# the Step 15a files) with one change, the drug encoding:
#   targets      : --drug_encoding fp_targets       fingerprint (1,024) + bag of curated targets (544) + has-targets flag (1)
#   targets_shuf : --drug_encoding fp_targets_shuf  same width; the bag and flag of another drug under a fixed cyclic order
# EXPECT params = 58,178,180 (recipe) + (1,569 - 1,024) x width(encoder.fc1); the guard computes it from the built model and
# checks the recipe model (fingerprint) gives 58,178,180 by the same code path. The encoder self-check (prereg section 9)
# runs in the guard: every training and test SMILES is in the Step 15a record, the fingerprint block equals the published
# encoder, the bag of every drug equals its curated set, the flag equals has-targets, the shuffled bag equals the partner's.
# DRYRUN=1 SLURM_ARRAY_TASK_ID=<T> bash train_step16_tahoe.sh  runs every guard and self-check, prints the counts, and stops.
set -eo pipefail
module load cuda/12.2.0
source /share/apps/NYUAD5/miniconda/3-4.11.0/etc/profile.d/conda.sh
conda activate /scratch/ni2071/conda_envs/squidiff
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
W=/scratch/ni2071/Diffusion/Squidiff_git/Squidiff
D=/scratch/ni2071/Diffusion/tahoe/datasets_step15
PREREG=PREREG_step16_tahoe_targets.md
cd "$W"

T=$SLURM_ARRAY_TASK_ID
case "$T" in *[!0-9]*|"") echo "FATAL: non-numeric task id: $T"; exit 1 ;; esac
if [ "$T" -gt 17 ]; then echo "FATAL: task out of range (0-17): $T"; exit 1; fi

# --- pre-registration, launcher, finaliser, patch and the encoder's two inputs committed; training sources clean ---
if [ ! -f "$PREREG" ]; then echo "FATAL: missing $PREREG"; exit 1; fi
N=$(grep -c '^## ' $PREREG)
if [ "$N" -lt 9 ]; then echo "FATAL: prereg has $N sections, expected at least 9"; exit 1; fi
TRACKED="$PREREG finalize_step16.py train_step16_tahoe.sh patch_step16.py tahoe_results/step15_data_record.json tahoe_results/step15_targets_curated_hgnc.csv"
for f in $TRACKED; do git ls-files --error-unmatch $f >/dev/null 2>&1 || { echo "FATAL: $f not committed"; exit 1; }; done
git diff --quiet -- $TRACKED || { echo "FATAL: prereg, finaliser, launcher, patch or encoder inputs uncommitted"; exit 1; }
if [ -n "$(git status --porcelain -- train_squidiff.py Squidiff)" ]; then
  echo "FATAL: training sources dirty"; git status --porcelain -- train_squidiff.py Squidiff; exit 1
fi
grep -q "'fp_targets_shuf'" Squidiff/scrna_datasets.py || { echo "FATAL: Squidiff/scrna_datasets.py is not patched (run patch_step16.py and commit)"; exit 1; }
COMMIT=$(git rev-parse HEAD)

# --- decode the task ---
SPLIT_IDX=$(( T / 6 )); WITHIN=$(( T % 6 )); ARM_IDX=$(( WITHIN / 3 )); SEED=$(( WITHIN % 3 ))
SPLIT=drug_split_${SPLIT_IDX}
case $ARM_IDX in
  0) ARM=targets;      ENC=fp_targets ;;
  1) ARM=targets_shuf; ENC=fp_targets_shuf ;;
  *) echo "FATAL: bad arm index $ARM_IDX"; exit 1 ;;
esac
export ARM ENC SPLIT
TAG=${SPLIT}_${ARM}_seed_${SEED}
OUTDIR=$W/step16/${ARM}/${SPLIT}_seed_${SEED}
LOGDIR=logger_files/logger_s16_${TAG}
PROV=$W/provenance/s16tr_${TAG}.json
TRAIN=$D/tahoe_train_${SPLIT}.h5ad
CTRL=$D/tahoe_train_${SPLIT}_control.h5ad
TEST=$D/tahoe_test_${SPLIT}.h5ad
export TRAIN TEST
for f in "$TRAIN" "$CTRL" "$TEST"; do [ -s "$f" ] || { echo "FATAL: missing input $f"; exit 1; }; done
if [ -e "$OUTDIR" ] && [ -n "$(find "$OUTDIR" -mindepth 1 -print -quit 2>/dev/null)" ]; then echo "FATAL: output dir not empty: $OUTDIR"; exit 1; fi
if [ -e "$W/$LOGDIR" ]; then echo "FATAL: logger path exists: $LOGDIR"; exit 1; fi
if [ -e "$PROV" ]; then echo "FATAL: provenance exists: $PROV"; exit 1; fi
export MASTER_PORT=$(( 21000 + T )); export PYTHONHASHSEED=$SEED
echo "[prereg] $N sections committed | [commit] $COMMIT | [node] $(hostname)"
echo "[task] T=$T  split=$SPLIT  arm=$ARM  seed=$SEED  adaln=True  fm_source=gaussian  cond_drop=0.0  encoder_arm=none  drug_encoding=$ENC"

# --- the recipe's arguments with the arm's drug encoding ---
ARGS="--logger_path $LOGDIR --data_path $TRAIN --control_data_path $CTRL --resume_checkpoint $OUTDIR"
ARGS="$ARGS --seed $SEED --use_drug_structure True --drug_encoding $ENC --gene_size 200 --output_dim 200 --lr_anneal_steps 100000"
ARGS="$ARGS --use_adaln True --fm_source gaussian"

# --- guard: flags took effect; parameter rule; encoder self-check ---
python - $ARGS <<'PY'
import sys, os
import numpy as np
sys.argv = ['train_squidiff.py'] + sys.argv[1:]
import train_squidiff as t
from Squidiff.script_util import create_model_and_diffusion, model_and_diffusion_defaults, args_to_dict
from Squidiff.scrna_datasets import DRUG_ENCODER_DIMS, DRUG_ENCODERS, Drug_dose_encoder, _load_target_map, TARGET_VOCAB_SIZE
arm, enc = os.environ['ARM'], os.environ['ENC']
a = t.parse_args()
print('[guard] arm=%s  use_adaln=%r  fm_source=%r  cond_drop=%r (%s)  encoder_arm=%r  drug_encoding=%r'
      % (arm, a['use_adaln'], a['fm_source'], a['cond_drop'], type(a['cond_drop']).__name__, a.get('encoder_arm'), a['drug_encoding']))
if a['use_adaln'] is not True: sys.exit('FATAL: use_adaln resolved to %r, expected True' % a['use_adaln'])
if a['fm_source'] != 'gaussian': sys.exit('FATAL: fm_source %r, expected gaussian' % a['fm_source'])
if not isinstance(a['cond_drop'], float) or abs(a['cond_drop']) > 1e-12: sys.exit('FATAL: cond_drop %r, expected 0.0' % a['cond_drop'])
if a['drug_encoding'] != enc: sys.exit('FATAL: drug_encoding %r, expected %r' % (a['drug_encoding'], enc))
if a.get('encoder_arm', 'mlp') != 'mlp': sys.exit('FATAL: encoder_arm %r, expected mlp' % a.get('encoder_arm'))
if int(a['gene_size']) != 200: sys.exit('FATAL: gene_size %r' % a['gene_size'])
dd = DRUG_ENCODER_DIMS[enc]
if dd != 1024 + TARGET_VOCAB_SIZE + 1: sys.exit('FATAL: DRUG_ENCODER_DIMS[%s] = %d, expected %d' % (enc, dd, 1024 + TARGET_VOCAB_SIZE + 1))
def build(encoding):
    b = dict(a); b['drug_encoding'] = encoding; b['drug_dimension'] = DRUG_ENCODER_DIMS[encoding]; b['memory_size'] = 0
    m, _ = create_model_and_diffusion(**args_to_dict(b, model_and_diffusion_defaults().keys()))
    return m, sum(p.numel() for p in m.parameters())
m0, n0 = build('fingerprint')                                                       # the recipe, by the same code path
if n0 != 58178180: sys.exit('FATAL: the recipe model has %d parameters, expected 58178180' % n0)
m, n = build(enc)
fc1_w, fc1_in = int(m.encoder.fc1.weight.shape[0]), int(m.encoder.fc1.weight.shape[1])
if fc1_in != 200 + dd: sys.exit('FATAL: encoder.fc1 reads %d inputs, expected 200 + %d' % (fc1_in, dd))
expect = 58178180 + (dd - 1024) * fc1_w
print('[guard] arm=%s drug_dimension=%d fc1 width=%d parameters=%d expected=%d (= 58178180 + %d x %d); recipe by the same path %d'
      % (arm, dd, fc1_w, n, expect, dd - 1024, fc1_w, n0))
if n != expect: sys.exit('FATAL: parameter count %d != expected %d for arm %s' % (n, expect, arm))
if n - n0 != (dd - 1024) * fc1_w: sys.exit('FATAL: the arm differs from the recipe by %d parameters, expected %d' % (n - n0, (dd - 1024) * fc1_w))
if [k for k in m.state_dict() if k.startswith('encoder.arm.')]: sys.exit('FATAL: unexpected encoder.arm.* parameters')
# --- encoder self-check on every SMILES of the split's training and test files (dose 9 -> log10(10) = 1: unscaled blocks)
import anndata as ad
T = _load_target_map()
smi = {part: sorted(set(str(s) for s in ad.read_h5ad(os.environ[part], backed='r').obs['SMILES'])) for part in ('TRAIN', 'TEST')}
if set(smi['TRAIN']) & set(smi['TEST']): sys.exit('FATAL: a SMILES is in both the training and the test file')
for part, S in smi.items():
    missing = [s for s in S if s not in T['bag']]
    if missing: sys.exit('FATAL: %d %s SMILES absent from the Step 15a record, e.g. %r' % (len(missing), part, missing[0]))
    E = DRUG_ENCODERS[enc](S, [9.0] * len(S), 1)
    if E.shape != (len(S), dd): sys.exit('FATAL: encoder output %s, expected (%d, %d)' % (E.shape, len(S), dd))
    fp = Drug_dose_encoder(S, [9.0] * len(S), comb_num=1)
    if np.abs(E[:, :1024] - fp).max() > 0: sys.exit('FATAL: the fingerprint block differs from Drug_dose_encoder')
    bags = T['bag_shuf'] if enc == 'fp_targets_shuf' else T['bag']
    n_with, n_diff, sizes = 0, 0, []
    for i, s in enumerate(S):
        want = set(T['index'][x] for x in bags[s])
        got = set(np.where(E[i, 1024:1024 + TARGET_VOCAB_SIZE] == 1.0)[0])
        if got != want or set(np.unique(E[i, 1024:1024 + TARGET_VOCAB_SIZE])) - {0.0, 1.0}: sys.exit('FATAL: bag of %r differs from its curated set' % s)
        if float(E[i, -1]) != (1.0 if want else 0.0): sys.exit('FATAL: has-targets flag of %r is wrong' % s)
        n_with += bool(want); sizes.append(len(want)); n_diff += (bags[s] != T['bag'][s])
    Z = DRUG_ENCODERS[enc](S[:3], [0.0, 0.0, 0.0], 1)
    if np.abs(Z).max() != 0: sys.exit('FATAL: a zero-dose control is not all zeros')
    print('[self-check] %s: %d SMILES | %d with a non-empty bag | bag size mean %.2f max %d | %s | zero dose -> zeros'
          % (part, len(S), n_with, float(np.mean(sizes)), max(sizes), ('shuffled bag differs from the own bag for %d' % n_diff) if enc == 'fp_targets_shuf' else 'own bags'))
PY
if [ "${DRYRUN:-0}" = "1" ]; then echo "[dryrun] guards and self-checks passed for $TAG; nothing created"; exit 0; fi
mkdir -p "$OUTDIR" logger_files provenance

echo "[$(date)] training $TAG"
CG=/sys/fs/cgroup/memory/slurm/uid_$(id -u)/job_${SLURM_JOB_ID}/memory.usage_in_bytes   # cgroup v1 path; NA if absent
python train_squidiff.py $ARGS &
TRAIN_PID=$!
( while kill -0 "$TRAIN_PID" 2>/dev/null; do echo "$(date +%s) rss_kb=$(ps -o rss= -p "$TRAIN_PID" | tr -d ' ') cgroup_bytes=$( [ -r "$CG" ] && cat "$CG" || echo NA )" >> "$OUTDIR/rss_kb.log"; sleep 30; done ) &
MON_PID=$!
RC=0; wait "$TRAIN_PID" || RC=$?
kill "$MON_PID" 2>/dev/null || true; wait "$MON_PID" 2>/dev/null || true
if [ "$RC" -ne 0 ]; then echo "FATAL: training exited with code $RC (peak sampled rss_kb: $(awk -F'rss_kb=' '{split($2,a," "); if(a[1]+0>m) m=a[1]+0} END{print m}' "$OUTDIR/rss_kb.log"))"; exit "$RC"; fi

# --- post: certify the checkpoint and write sidecar + provenance ---
python -u finalize_step16.py --jobid "$SLURM_ARRAY_JOB_ID" --tasks "$T" --expect-arm "$ARM" --expect-split "$SPLIT" --expect-seed "$SEED"
echo "[$(date)] $TAG DONE"
