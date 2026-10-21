#!/bin/bash
#SBATCH --job-name=s21_train
#SBATCH --partition=condo
#SBATCH -q a2s2
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=150G
#SBATCH --time=24:00:00
#SBATCH --array=0-5%2
#SBATCH --output=/scratch/ni2071/Diffusion/Squidiff_git/Squidiff/logger_files/s21tr_%A_%a.out
#SBATCH --error=/scratch/ni2071/Diffusion/Squidiff_git/Squidiff/logger_files/s21tr_%A_%a.err
# Step 21 (m = 32 against full OT on the confirmation splits) training under PREREG_step21_mb32_confirmation.md.
# The arm is the Step 20 arm mb32 exactly - the Step 15b recipe (adaLN + Gaussian flow matching, cond_drop 0, encoder mlp,
# fingerprint encoding, 100,000 steps, 200 genes, the Step 15a training file) - with --control_data_path pointing at the
# minibatch-OT control file tahoe_train_<split>_control_mb32.h5ad (the same control cells, re-ordered within each
# condition; written and sha256-pinned by check_minibatch_pairing_tahoe.py in
# tahoe_results/minibatch_pairing_written_<split>_m32.json, committed with the pre-registration).
#
# One stage, 6 tasks (prereg sections 2 and 9).  task T:  split = (drug_split_0, drug_split_2)[T / 3];  seed = T % 3;  m = 32
#     sbatch train_step21_tahoe.sh
# The guard asserts the pin, the original control file and the training file against the data record, the row
# alignment of the minibatch file with the training file (backed read of obs only), and the parameter count 58,178,180
# built by the recipe's code path.
# DRYRUN=1 SLURM_ARRAY_TASK_ID=<T> bash train_step21_tahoe.sh  runs every guard and stops.
set -eo pipefail
module load cuda/12.2.0
source /share/apps/NYUAD5/miniconda/3-4.11.0/etc/profile.d/conda.sh
conda activate /scratch/ni2071/conda_envs/squidiff
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
W=${S21_W:-/scratch/ni2071/Diffusion/Squidiff_git/Squidiff}      # S21_W / S21_D override for synthetic tests only
D=${S21_D:-/scratch/ni2071/Diffusion/tahoe/datasets_step15}
PREREG=PREREG_step21_mb32_confirmation.md
RECORD=tahoe_results/step15_data_record.json
MBREC=tahoe_results/minibatch_pairing_check.json
cd "$W"

T=$SLURM_ARRAY_TASK_ID
case "$T" in *[!0-9]*|"") echo "FATAL: non-numeric task id: $T"; exit 1 ;; esac
if [ -n "${STAGE:-}" ]; then echo "FATAL: STAGE is a Step 20 setting; Step 21 has one stage"; exit 1; fi
if [ -n "${M:-}" ] && [ "${M}" != "32" ]; then echo "FATAL: Step 21 trains m = 32 only, got M=$M"; exit 1; fi
M=32

# --- decode the task (the finaliser decodes it the same way) ---
if [ "$T" -gt 5 ]; then echo "FATAL: task out of range (0-5): $T"; exit 1; fi
SI=(0 2); SPLIT_IDX=${SI[$(( T / 3 ))]}; SEED=$(( T % 3 ))
SPLIT=drug_split_${SPLIT_IDX}
ARM=mb${M}; ENC=fingerprint
PIN=tahoe_results/minibatch_pairing_written_${SPLIT}_m${M}.json

# --- pre-registration, launcher, finaliser, the pairing script, its records and this task's pin committed; sources clean ---
if [ ! -f "$PREREG" ]; then echo "FATAL: missing $PREREG"; exit 1; fi
N=$(grep -c '^## ' $PREREG)
if [ "$N" -lt 9 ]; then echo "FATAL: prereg has $N sections, expected at least 9"; exit 1; fi
TRACKED="$PREREG finalize_step21.py train_step21_tahoe.sh check_minibatch_pairing_tahoe.py $MBREC $PIN $RECORD"
for f in $TRACKED; do git ls-files --error-unmatch $f >/dev/null 2>&1 || { echo "FATAL: $f not committed"; exit 1; }; done
git diff --quiet -- $TRACKED || { echo "FATAL: prereg, finaliser, launcher, pairing script, records or pin uncommitted"; exit 1; }
if [ -n "$(git status --porcelain -- train_squidiff.py Squidiff)" ]; then
  echo "FATAL: training sources dirty"; git status --porcelain -- train_squidiff.py Squidiff; exit 1
fi
COMMIT=$(git rev-parse HEAD)

export ARM ENC SPLIT SPLIT_IDX RECORD MBREC PIN M
TAG=${SPLIT}_${ARM}_seed_${SEED}
OUTDIR=$W/step21/${ARM}/${SPLIT}_seed_${SEED}
LOGDIR=logger_files/logger_s21_${TAG}
PROV=$W/provenance/s21tr_${TAG}.json
TRAIN=$D/tahoe_train_${SPLIT}.h5ad
CTRL0=$D/tahoe_train_${SPLIT}_control.h5ad
CTRL=$D/tahoe_train_${SPLIT}_control_mb${M}.h5ad
TEST=$D/tahoe_test_${SPLIT}.h5ad
export TRAIN CTRL0 CTRL TEST
for f in "$TRAIN" "$CTRL0" "$CTRL" "$TEST"; do [ -s "$f" ] || { echo "FATAL: missing input $f"; exit 1; }; done
if [ -e "$OUTDIR" ] && [ -n "$(find "$OUTDIR" -mindepth 1 -print -quit 2>/dev/null)" ]; then echo "FATAL: output dir not empty: $OUTDIR"; exit 1; fi
if [ -e "$W/$LOGDIR" ]; then echo "FATAL: logger path exists: $LOGDIR"; exit 1; fi
if [ -e "$PROV" ]; then echo "FATAL: provenance exists: $PROV"; exit 1; fi
export MASTER_PORT=$(( 24200 + T )); export PYTHONHASHSEED=$SEED              # distinct from Step 20 (24000-24008, 24100-24105)
echo "[prereg] $N sections committed | [commit] $COMMIT | [node] $(hostname)"
echo "[task] T=$T  split=$SPLIT  arm=$ARM  m=$M  seed=$SEED  adaln=True  fm_source=gaussian  cond_drop=0.0  encoder_arm=none  drug_encoding=$ENC  control=$CTRL"

# --- the recipe's arguments with the minibatch-OT control file (identical to Step 18 apart from the control file) ---
ARGS="--logger_path $LOGDIR --data_path $TRAIN --control_data_path $CTRL --resume_checkpoint $OUTDIR"
ARGS="$ARGS --seed $SEED --use_drug_structure True --drug_encoding $ENC --gene_size 200 --output_dim 200 --lr_anneal_steps 100000"
ARGS="$ARGS --use_adaln True --fm_source gaussian"

# --- guard: flags took effect; the recipe's parameter count; the minibatch control file pinned and aligned ---
python - $ARGS <<'PY'
import hashlib, json, os, sys
import numpy as np
import anndata as ad
e = os.environ
# ---- the model: the recipe's flags and parameter count (verbatim from the Step 18 launcher)
sys.argv = ['train_squidiff.py'] + sys.argv[1:]
import train_squidiff as t
from Squidiff.script_util import create_model_and_diffusion, model_and_diffusion_defaults, args_to_dict
from Squidiff.scrna_datasets import DRUG_ENCODER_DIMS
a = t.parse_args()
print('[guard] arm=%s  use_adaln=%r  fm_source=%r  cond_drop=%r (%s)  encoder_arm=%r  drug_encoding=%r  control_data_path=%s'
      % (e['ARM'], a['use_adaln'], a['fm_source'], a['cond_drop'], type(a['cond_drop']).__name__, a.get('encoder_arm'), a['drug_encoding'], a['control_data_path']))
if a['use_adaln'] is not True: sys.exit('FATAL: use_adaln resolved to %r, expected True' % a['use_adaln'])
if a['fm_source'] != 'gaussian': sys.exit('FATAL: fm_source %r, expected gaussian' % a['fm_source'])
if not isinstance(a['cond_drop'], float) or abs(a['cond_drop']) > 1e-12: sys.exit('FATAL: cond_drop %r, expected 0.0' % a['cond_drop'])
if a['drug_encoding'] != 'fingerprint': sys.exit('FATAL: drug_encoding %r, expected fingerprint' % a['drug_encoding'])
if a.get('encoder_arm', 'mlp') != 'mlp': sys.exit('FATAL: encoder_arm %r, expected mlp' % a.get('encoder_arm'))
if int(a['gene_size']) != 200: sys.exit('FATAL: gene_size %r' % a['gene_size'])
if a['control_data_path'] != e['CTRL'] or not a['control_data_path'].endswith('_control_mb%s.h5ad' % e['M']): sys.exit('FATAL: control_data_path %r is not the minibatch file for m = %s' % (a['control_data_path'], e['M']))
if DRUG_ENCODER_DIMS['fingerprint'] != 1024: sys.exit('FATAL: fingerprint width %d' % DRUG_ENCODER_DIMS['fingerprint'])
b = dict(a); b['drug_dimension'] = 1024; b['memory_size'] = 0
m, _ = create_model_and_diffusion(**args_to_dict(b, model_and_diffusion_defaults().keys()))
n = sum(p.numel() for p in m.parameters())
if n != 58178180: sys.exit('FATAL: the model has %d parameters, expected the recipe\'s 58178180' % n)
if [k for k in m.state_dict() if k.startswith('encoder.arm.')]: sys.exit('FATAL: unexpected encoder.arm.* parameters')
if int(m.encoder.fc1.weight.shape[1]) != 200 + 1024: sys.exit('FATAL: encoder.fc1 reads %d inputs, expected 1224' % int(m.encoder.fc1.weight.shape[1]))
# ---- the minibatch control file: pinned by the pairing script, derived from the certified control file, aligned
def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for c in iter(lambda: f.read(1 << 24), b''): h.update(c)
    return h.hexdigest()
rec = json.load(open(e['RECORD'])); mb = json.load(open(e['MBREC'])); pin = json.load(open(e['PIN']))
files = rec['splits'][e['SPLIT_IDX']]['files']
if int(pin['m']) != int(e['M']) or pin['split'] != e['SPLIT']: sys.exit('FATAL: the pin is for %s m = %s' % (pin['split'], pin['m']))
if int(e['M']) not in [int(x) for x in mb['selection']['chosen']]: sys.exit('FATAL: m = %s was not chosen by the pairing rule' % e['M'])
if os.path.abspath(pin['path']) != os.path.abspath(e['CTRL']): sys.exit('FATAL: the pin names %s, the task uses %s' % (pin['path'], e['CTRL']))
h_mb, h_c0, h_tr = sha(e['CTRL']), sha(e['CTRL0']), sha(e['TRAIN'])
if h_mb != pin['sha256']: sys.exit('FATAL: minibatch control file sha256 %s differs from the pin %s' % (h_mb[:16], pin['sha256'][:16]))
if h_c0 != files['train_control']['sha256'] or h_c0 != pin['source_control_sha256']: sys.exit('FATAL: the original control file differs from the data record or the pin')
if h_tr != files['train']['sha256']: sys.exit('FATAL: the training file differs from the data record')
if mb['per_split'][e['SPLIT']]['train_control']['sha256'] != h_c0: sys.exit('FATAL: the pairing was measured on a different control file')
T_ = ad.read_h5ad(e['TRAIN'], backed='r'); C0 = ad.read_h5ad(e['CTRL0'], backed='r'); C1 = ad.read_h5ad(e['CTRL'], backed='r')
if not (T_.n_obs == C0.n_obs == C1.n_obs) or T_.n_vars != C1.n_vars: sys.exit('FATAL: training / control / minibatch-control files differ in shape')
for col in ('Group', 'cell_name', 'dose', 'SMILES'):
    if not np.array_equal(T_.obs[col].astype(str).values, C1.obs[col].astype(str).values): sys.exit('FATAL: minibatch control rows are not aligned with the training rows on %s' % col)
if not np.array_equal(np.sort(C1.obs_names.values), np.sort(C0.obs_names.values)): sys.exit('FATAL: the minibatch control file does not hold the rows of the original control file')
ms = mb['per_split'][e['SPLIT']]
print('[step21] arm=%s split=%s parameters=%d (= the recipe) | minibatch control %s pinned (%s), derived from the certified control (%s), %d rows aligned with the training file | pairing: cosine %.3f, f_cos %.3f'
      % (e['ARM'], e['SPLIT'], n, os.path.basename(e['CTRL']), h_mb[:12], h_c0[:12], C1.n_obs, ms['mean']['cos_m%s' % e['M']], ms['f_cos'][str(e['M'])]))
PY
if [ "${DRYRUN:-0}" = "1" ]; then echo "[dryrun] guards passed for $TAG; nothing created"; exit 0; fi
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
python -u finalize_step21.py --jobid "$SLURM_ARRAY_JOB_ID" --tasks "$T" --expect-split "$SPLIT" --expect-seed "$SEED"
echo "[$(date)] $TAG DONE"
