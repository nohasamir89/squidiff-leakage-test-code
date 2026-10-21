#!/bin/bash
#SBATCH -p compute
#SBATCH -c 32
#SBATCH --mem=120G
#SBATCH -t 24:00:00
#SBATCH -o logger_files/s22_%j.out
#SBATCH -e logger_files/s22_%j.err
# Step 22 launcher (PREREG_step22_fingerprint_identifiability.md, section 9): one split per job, stages 1 and 2.
#   sbatch run_step22_tahoe.sh drug_split_1                         development split (first)
#   sbatch run_step22_tahoe.sh drug_split_0 ; ... drug_split_2      confirmation splits, scripts frozen
#   sbatch -c 8 --mem=32G -t 4:00:00 run_step22_tahoe.sh mock       the end-to-end test on the synthetic split
# Stage 1 (prep_step22_fingerprint.py, squidiff env) writes tahoe_results/step22/<split>/ (or step22_mock/mock_split/);
# it is skipped when that folder's manifest already exists, so a stopped job can be resubmitted. Stage 2 runs the
# authors' package through run_step22_fingerprint.R in R_env, called by its full path (the R/4.5.3 module is the shared
# r-tess3 env and must not shadow R_env), with one BLAS thread per process so the package's forked loops use the cores.
# Guards: the pre-registration committed with at least 10 sections; the four Step 22 scripts committed and clean.
set -eo pipefail
SPLIT=${1:?usage: sbatch run_step22_tahoe.sh <drug_split_1|drug_split_0|drug_split_2|mock>}
cd /scratch/ni2071/Diffusion/Squidiff_git/Squidiff
PRE=PREREG_step22_fingerprint_identifiability.md
RSCRIPT=/scratch/ni2071/conda_envs/R_env/bin/Rscript
TRACKED="$PRE prep_step22_fingerprint.py run_step22_fingerprint.R run_step22_tahoe.sh"
case "$SPLIT" in drug_split_1|drug_split_0|drug_split_2|mock) ;; *) echo "FATAL: unknown split $SPLIT"; exit 1;; esac
[ "$(grep -c '^## ' "$PRE")" -ge 10 ] || { echo "FATAL: prereg incomplete"; exit 1; }
git ls-files --error-unmatch $TRACKED > /dev/null || { echo "FATAL: a Step 22 script is not committed"; exit 1; }
git diff --quiet -- $TRACKED || { echo "FATAL: a Step 22 script has uncommitted changes"; exit 1; }
[ -x "$RSCRIPT" ] || { echo "FATAL: no Rscript at $RSCRIPT"; exit 1; }
module unload R 2> /dev/null || true
module unload R/4.5.3 2> /dev/null || true
source /share/apps/NYUAD5/miniconda/3-4.11.0/etc/profile.d/conda.sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
CORES=${SLURM_CPUS_PER_TASK:-1}
echo "[step22] job ${SLURM_JOB_ID:-none} split $SPLIT cores $CORES commit $(git rev-parse --short HEAD) start $(date '+%F %T')"
if [ "$SPLIT" = mock ]; then DIR=tahoe_results/step22_mock/mock_split; ARGS="--mock"; else DIR=tahoe_results/step22/$SPLIT; ARGS="--split $SPLIT"; fi
if [ ! -f "$DIR/manifest.json" ]; then
  conda activate /scratch/ni2071/conda_envs/squidiff
  python -u prep_step22_fingerprint.py $ARGS
  conda deactivate
else
  echo "[step22] $DIR/manifest.json exists; stage 1 skipped (resume)"
fi
"$RSCRIPT" --version
"$RSCRIPT" run_step22_fingerprint.R --dir="$DIR" --cores="$CORES"
# cell-level answer files are kept gzipped (about 5 MB each uncompressed, twelve per split); the scorer reads either form
ls "$DIR"/map_*_cell.csv > /dev/null 2>&1 && gzip -9 -f "$DIR"/map_*_cell.csv
echo "[step22] done $(date '+%F %T')"
