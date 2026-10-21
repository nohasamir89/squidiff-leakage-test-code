"""pytest configuration for the Squidiff manuscript code tests.

What this file does
-------------------
1. Finds the code and the records whether the tests run inside the *release
   package* (folders 01_data_preparation/ ... 07_records/) or inside the
   *git repository* (scripts at the root, records in tahoe_results/ and
   leakage_results/). Every test asks the `layout` fixture for paths instead
   of hard-coding them.

2. Makes the scoring modules importable WITHOUT a GPU, PyTorch or the data:
   * The scorers import the model package as ``Squidiff``; the release ships it
     as ``Squidiff_package``. We alias one to the other when it imports
     (it needs torch), and otherwise register a minimal stub that provides
     only the names the scorers bind at import time. None of the functions
     under test ever call into it.
   * ``score_step16_tahoe`` imports ``score_step15d_utility`` and
     ``score_step15e_pooled``. If those files are present they are used; if
     not (they were missing from the first release package) a stub is
     registered so the rest of the suite still runs, and ``test_package.py``
     reports the gap. The ``spearman_fast`` stub is routed through scipy, so
     ``score_step16_tahoe.mra_self_check()`` becomes an independent check.

Why stubs instead of installing torch: the quantities tested here are pure
numpy/scipy functions. Keeping the suite CPU-only means it runs on any laptop
and in GitHub Actions in about a minute, so it actually gets run.
"""
import importlib
import pathlib
import sys
import types

import numpy as np
import pytest

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent

# --------------------------------------------------------------------------- layout detection
class Layout:
    """Paths for the two layouts the code lives in (release package vs git repo)."""

    def __init__(self, root: pathlib.Path):
        self.root = root
        self.is_package = (root / "04_scoring").is_dir()
        if self.is_package:
            self.scoring = root / "04_scoring"
            self.training = root / "02_training"          # holds Squidiff_package/
            self.evaluation = root / "03_evaluation"
            self.records = [root / "07_records"]
            self.figvalues = [root / "06_supplementary" / "figure_values",
                              root / "05_figures" / "fig1b_intermediate"]
            ext = root / "08_external_pipelines"
            self.gbm_sens_csv = ext / "gbm_squidiff_robustness_test" / "gene_sensitivity_SEEDED.csv"
            self.gbm_log_200 = ext / "gbm_run_200" / "logs" / "eval_200_16378137.out"
            self.gbm_log_2000 = ext / "gbm_run" / "logs" / "eval_16377417.out"
            self.manifest = root / "MANIFEST.csv"
        else:                                           # git repository layout
            self.scoring = root
            self.training = root
            self.evaluation = root
            self.records = [root / "tahoe_results", root / "leakage_results"]
            self.figvalues = [root / "figures" / "squidiff_publication_figures", root / "figures"]
            self.gbm_sens_csv = pathlib.Path("/scratch/ni2071/Diffusion/gbm_squidiff_robustness_test/gene_sensitivity_SEEDED.csv")
            self.gbm_log_200 = pathlib.Path("/scratch/ni2071/Diffusion/gbm_run_200/logs/eval_200_16378137.out")
            self.gbm_log_2000 = pathlib.Path("/scratch/ni2071/Diffusion/gbm_run/logs/eval_16377417.out")
            self.manifest = None

    def record(self, name: str) -> pathlib.Path:
        """Path of a committed result record (JSON/CSV) by file name; skips the test if absent."""
        for d in self.records:
            p = d / name
            if p.exists():
                return p
        pytest.skip("record %s not found under %s" % (name, [str(d) for d in self.records]))

    def figvalue(self, name: str) -> pathlib.Path:
        for d in self.figvalues:
            p = d / name
            if p.exists():
                return p
        pytest.skip("figure value table %s not found" % name)


LAYOUT = Layout(ROOT)

# --------------------------------------------------------------------------- import paths
for p in (LAYOUT.scoring, LAYOUT.training, LAYOUT.evaluation):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

STUBS_ACTIVE = {}   # name -> reason; reported by test_package.py so stubbing is never silent


def _try_real_squidiff() -> bool:
    """Import the real model package (needs torch). Alias Squidiff_package -> Squidiff for the release."""
    try:
        importlib.import_module("Squidiff.scrna_datasets")
        return True
    except Exception:
        pass
    try:
        pkg = importlib.import_module("Squidiff_package")
        sys.modules["Squidiff"] = pkg                      # same package object under the name the code uses
        importlib.import_module("Squidiff.scrna_datasets")
        return True
    except Exception:
        for k in [k for k in sys.modules if k == "Squidiff" or k.startswith("Squidiff.")]:
            del sys.modules[k]
        return False


if not _try_real_squidiff():
    squidiff = types.ModuleType("Squidiff")
    squidiff.__path__ = []                                 # mark as a package so submodule lookup works
    ds = types.ModuleType("Squidiff.scrna_datasets")

    def _not_in_tests(*a, **k):
        raise RuntimeError("stub Squidiff.scrna_datasets called: the unit tests never need the encoder")

    ds.Drug_dose_encoder = _not_in_tests
    ds._load_target_map = _not_in_tests
    ds.DRUG_ENCODERS = {}
    ds.DRUG_ENCODER_DIMS = {}
    ds.TARGET_VOCAB_SIZE = 544                           # the manuscript's curated-target vocabulary size
    sys.modules["Squidiff"] = squidiff
    sys.modules["Squidiff.scrna_datasets"] = ds
    STUBS_ACTIVE["Squidiff.scrna_datasets"] = "torch/rdkit not importable here; stub provides import-time names only"

# the two helper modules score_step16_tahoe.py imports; real files win when present
if not (LAYOUT.scoring / "score_step15d_utility.py").exists():
    s15d = types.ModuleType("score_step15d_utility")
    s15d.__getattr__ = lambda name: ""                   # PEP 562: any import-time attribute reads as ''
    sys.modules["score_step15d_utility"] = s15d
    STUBS_ACTIVE["score_step15d_utility"] = "file absent from this tree (add it to the package list)"

if not (LAYOUT.scoring / "score_step15e_pooled.py").exists():
    from scipy.stats import spearmanr

    s15e = types.ModuleType("score_step15e_pooled")

    def spearman_fast(x, y):
        """scipy's Spearman; makes score_step16_tahoe.mra_self_check an independent comparison."""
        r = spearmanr(x, y)[0]
        return 0.0 if np.isnan(r) else float(r)

    s15e.spearman_fast = spearman_fast
    sys.modules["score_step15e_pooled"] = s15e
    STUBS_ACTIVE["score_step15e_pooled"] = "file absent from this tree (add it to the package list)"


# --------------------------------------------------------------------------- fixtures
@pytest.fixture(scope="session")
def layout() -> Layout:
    return LAYOUT


@pytest.fixture(scope="session")
def stubs_active() -> dict:
    return dict(STUBS_ACTIVE)


@pytest.fixture
def rng() -> np.random.Generator:
    """One fixed seed per test: failures reproduce exactly."""
    return np.random.default_rng(12345)
