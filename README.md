# Squidiff manuscript code package

Built 2026-10-22 from repository `/scratch/ni2071/Diffusion/Squidiff_git/Squidiff-code-tests` at commit `be6999b` (tag `manuscript-v3`), by `build_manuscript_package.py`
from the verified file list `manuscript_code_list.txt`. The repository was not changed; this folder is a copy
of exactly the files the manuscript's figures, tables and reported numbers depend on.

## How to read it

Folders are numbered in the order the work ran:

| folder | contents |
|---|---|
| `01_data_preparation/` | dataset preparation and split construction (Tahoe-100M Step 15a, sci-Plex, glioblastoma) |
| `02_training/` | the `Squidiff` package (model code), the training entry point, every training launcher and the checkpoint finalisers; the pairing scripts for optimal-transport and minibatch-OT training |
| `03_evaluation/` | the evaluation harness (`leakage_test_*`), its port scripts, the sampler, every evaluation launcher, the R toolchain record for RNA fingerprinting |
| `04_scoring/` | the scorers that produce the records (`score_*`, `analyse_*`), the post-hoc and diagnostic scripts the paper cites, the test-loss, bottleneck and ablation analyses |
| `05_figures/` | the nine figure scripts (`plot_fig1b.R` … `plot_fig5.R`; `plot_fig1b.py` is the value extractor for Fig. 1b) |
| `06_supplementary/` | the value tables the figure scripts write (`fig*_values.csv`), which feed the supplementary tables |
| `07_records/` | every JSON/CSV record a figure, a scorer or a reported number reads; `provenance/` holds the per-run evaluation provenance records (sha256 of checkpoint, arrays and harness) the scorers verify their inputs against; `step22/<split>/` the fingerprinting stage outputs the Step 22 scorer reads |
| `08_external_pipelines/` | pipelines that live outside the repository: `sciplex/` (preprocessing), `gbm_squidiff_robustness_test/` (gene-panel sweep), `gbm_run/` and `gbm_run_200/` (reproduction of the published analysis), `datasets_gbm_paper/` (its dataset preparation) |
| `prereg/` | the pre-registration document of every step |
| `tests/` | the automated test suite (see below), with `pytest.ini`, `ruff.toml` and `.github/workflows/tests.yml` at the root |

`FIGURE_MAP.md` maps every panel and table to the script that draws it, the record it reads, the script that
wrote the record and the pipeline behind it. `MANIFEST.csv` lists every file with its original path, size,
sha256 and the last commit that touched it. `PACKAGE_FIXES.md` lists the changes made on 21 Oct 2026 (test
suite, continuous integration, packaging fixes) and the files that were added to the package then.

## Tests

`python -m pytest` from this folder (about 4 s, CPU only, no data needed) runs `tests/`: every statistical
routine against SciPy/scikit-learn or an analytical value, every `fig*_values.csv` re-derived from `07_records/`,
and the package's integrity (sha256 vs `MANIFEST.csv`, compile, import graph). `python -m pytest -m slow` adds
the bootstrap-coverage simulation (about 30-90 s). `.github/workflows/tests.yml` runs the same on every push.
`tests/README.md` says what is and is not covered: training and evaluation are not re-run.

## Notes for anyone running the code

- The training entry point is `02_training/train_squidiff.py` (the file every `train_*.sh` runs); the copy inside
  `Squidiff_package/` is the upstream authors' original and is not used.
- `--resume_checkpoint` is the directory checkpoints are *saved* to; resuming is disabled in `train_util.py`
  (every recorded run trained its 100,000 steps in one job).
- Bootstrap intervals are deterministic in two conventions: Steps 15/15c draw all resamples from one sequential
  `RandomState(0)` (reproducible as a whole script); Steps 18-22 reset `RandomState(0)` before every contrast
  (each contrast reproducible on its own). Both give the committed records exactly.

## How to run things

The scripts expect the repository's own layout (for example the figure scripts look for `tahoe_results/` and
`leakage_results/` relative to the repository root), so to execute anything, clone the repository and check
out the tag:

    git clone <repository url> && cd Squidiff && git checkout manuscript-v3

Figures: `Rscript figures/squidiff_publication_figures/plot_fig2.R` (and 3, 4, 5) from any directory;
`Rscript plot_fig1c.R` etc. from the repository root; Fig. 1b needs `python plot_fig1b.py` first. Each script
prints the plotted values and writes `fig*_values.csv` next to the figure.

Not included (too large): checkpoints, per-cell `.npz` arrays and `.h5ad` data; `MANIFEST.csv` lists them with
their original location.

## Revising

Edit in place here and send the folder (or a diff) back; `MANIFEST.csv` gives the original path of every file,
so each change can be applied to the repository at the tag and committed with its provenance intact.
