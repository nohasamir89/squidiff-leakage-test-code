# Tests for the manuscript code

Run from the package root (or the repository root):

    python -m pytest            # about 4 seconds, CPU only, no data needed
    python -m pytest -m slow    # adds the bootstrap-coverage simulation (about 30 s)

Requirements: numpy, scipy, pandas, scikit-learn, anndata, pytest (the versions in
`00_environment/squidiff_env.yml`, or the pins in `.github/workflows/tests.yml`).
PyTorch is **not** needed: `conftest.py` imports the scorers through a small
torch-free shim.

## What is covered

| File | Question it answers | How |
| --- | --- | --- |
| `test_metrics.py` | Do the hand-written statistics compute the right thing? | 21 checks of the repo's own functions against scipy/scikit-learn or analytical values: per-gene Wilcoxon rank-sum (with ties), Benjamini–Hochberg, energy distance (independent implementation and a closed-form normal case), compound discrimination (independent implementation; perfect match = 1; chance = 0.5), the hypergeometric chance Jaccard 0.144, `fast_rank`, per-context Spearman (MRA), partial Spearman (spread decomposition), eta-squared, the harness r², interval levels and verdict rule; optional cluster-bootstrap coverage simulation (96.5 % observed for the nominal 97.5 % on 21 Oct 2026). |
| `test_figure_values.py` | Does every figure value equal what the committed result record says? | Reimplements each figure script's extraction in Python, recomputes every `fig*_values.csv` from `07_records/` (or `tahoe_results/` + `leakage_results/`), and compares row by row at 1e-12 relative tolerance; parses the GBM SLURM logs for Fig 1e; golden checks for the Fig 1c annotations and the Fig 1d reconstruction gate. 297 rows matched on 21 Oct 2026. |
| `test_package.py` | Is the released package complete and unaltered? | sha256 of every packaged file vs `MANIFEST.csv`; `py_compile` of every script; an import-graph walk that fails on any local module the package forgot (pending modules are declared in `known_gaps.txt` and reported as expected failures); refuses undocumented stubs. |
| `test_data_inputs.py` | Is every record the packaged scripts read (and every record a scorer writes) in the package? | Scans the string literals of the data-preparation, scoring, figure and external-pipeline scripts for .json/.csv/.tsv names, separates outputs (written on that line or assigned to an OUT* variable) from inputs, and fails on any input absent from the package unless `data_files_not_packaged.txt` lists it with a reason; result records written by `04_scoring/` scripts must be packaged as well. Added 22 Oct 2026 after the first package was found to lack fifteen such records. |

## What is not covered

Training and evaluation are not re-run: that needs the data, the checkpoints
and a GPU, and takes days on the cluster. The committed records in `07_records/`
are taken as the outputs of those runs. The repo's own guards are the evidence
for them: every scorer verifies the sha256 of its inputs against provenance
records, refuses to run unless the pre-registration and its own code are
committed and clean, and re-derives the previous step's values before adding
its own.

## When a test fails

* `test_manifest_sha256` after you edited a released file: expected. Rebuild
  the package (`build_manuscript_package.py` regenerates `MANIFEST.csv`).
* `test_local_imports_resolve_inside_package` naming a module: add that file
  to `manuscript_code_list.txt` and rebuild; if it is genuinely pending, list
  it in `known_gaps.txt` with a comment.
* Anything in `test_figure_values.py`: a record or a figure script changed.
  Check which; the message prints the row key, the committed value and the
  recomputed value.
* Anything in `test_metrics.py`: a statistical function changed behaviour.
  Do not loosen the tolerance; find out why the numbers differ.

## A sentence for the Code availability statement

"All statistical routines are covered by an automated test suite that checks
them against SciPy/scikit-learn reference implementations and analytical
values, re-derives every figure value from the committed result records, and
verifies the integrity and completeness of the released package; the suite
runs on every change via continuous integration (GitHub Actions)."

## The advisory lint

`ruff check --select F,B,NPY .` reports about 270 items. They were read one by
one on 21 Oct 2026: 130 loop-variable-capture warnings (B023) are closures that
are either consumed within their own loop iteration or bind their variables
through default arguments, so none can change a result; the rest are unused
imports, `zip` without `strict=`, and legacy `RandomState` calls that are
intentional (the pre-registrations name their seeds). CI therefore enforces
only the crash-class rules and prints the rest.
