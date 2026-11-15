"""Integrity and completeness of the released code package.

These tests answer three questions a reviewer or a future reader asks first:
1. Is every file the one that was released? (sha256 of each packaged file vs MANIFEST.csv)
2. Does every Python file at least parse? (py_compile)
3. Does every local import resolve inside the package? (an import-graph walk;
   this is the test that would have caught the two scorer helper modules
   missing from the first release package)

A fourth test makes the conftest stubs visible: a stub is only acceptable for
a module that is documented as pending in tests/known_gaps.txt (or for the
torch-dependent model package, which the unit tests never call).

After you edit any released file, the sha256 test fails on purpose until the
package is rebuilt (build_manuscript_package.py regenerates MANIFEST.csv).
"""
import ast
import csv
import hashlib
import py_compile
import sys

import pytest

# Third-party distributions the code imports. Anything else that is not in the
# standard library and not a file in the package is reported as a missing local module.
THIRD_PARTY = {
    "numpy", "scipy", "pandas", "sklearn", "anndata", "scanpy", "torch", "th", "h5py", "matplotlib",
    "seaborn", "rdkit", "skmisc", "blobfile", "mpi4py", "PIL", "tqdm", "yaml", "requests", "numba",
    "pytest", "scvi", "umap", "leidenalg", "igraph", "louvain", "networkx", "statsmodels",
    "threadpoolctl",      # installed with scikit-learn; score_step29_sciplex.py uses it to set the number of threads
}
KNOWN_GAPS_FILE = "known_gaps.txt"


def _known_gaps(tests_dir):
    p = tests_dir / KNOWN_GAPS_FILE
    if not p.exists():
        return set()
    return {ln.split("#", 1)[0].strip() for ln in p.read_text().splitlines() if ln.split("#", 1)[0].strip()}


def test_manifest_sha256(layout):
    if layout.manifest is None or not layout.manifest.exists():
        pytest.skip("no MANIFEST.csv in this layout (repository checkout, not the release package)")
    bad, n_ok = [], 0
    with open(layout.manifest, newline="") as fh:
        for row in csv.DictReader(fh):
            rel = row["package_path"].strip()
            if not rel:                                             # listed but not packaged (checkpoints, data)
                continue
            p = layout.root / rel
            if not p.exists():
                bad.append("MISSING  " + rel)
                continue
            h = hashlib.sha256(p.read_bytes()).hexdigest()
            if h == row["sha256"].strip():
                n_ok += 1
            else:
                bad.append("CHANGED  " + rel)
    assert not bad, ("%d files differ from MANIFEST.csv (edited since the build? rebuild the package to "
                     "regenerate the manifest):\n  " % len(bad)) + "\n  ".join(bad)
    assert n_ok > 0


def test_every_python_file_compiles(layout):
    fails = []
    for p in sorted(layout.root.rglob("*.py")):
        try:
            py_compile.compile(str(p), doraise=True)
        except py_compile.PyCompileError as e:
            fails.append("%s: %s" % (p.relative_to(layout.root), str(e).splitlines()[0]))
    assert not fails, "\n".join(fails)


def test_local_imports_resolve_inside_package(layout):
    """Walk every import statement; a name that is neither standard library, nor a listed
    third-party distribution, nor a file/package in this tree is a module the release forgot."""
    pys = sorted(layout.root.rglob("*.py"))
    stems = {p.stem for p in pys}
    pkg_dirs = {p.parent.name for p in pys if p.name == "__init__.py"}
    stdlib = set(getattr(sys, "stdlib_module_names", ()))
    aliases = {"Squidiff": ("Squidiff" in pkg_dirs) or ("Squidiff_package" in pkg_dirs)}   # release ships it renamed
    missing = {}
    for p in pys:
        if "tests" in p.relative_to(layout.root).parts:
            continue
        try:
            tree = ast.parse(p.read_text(), filename=str(p))
        except SyntaxError:
            continue                                                # reported by the compile test
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                names = [node.module.split(".")[0]]
            for nm in names:
                if nm in stdlib or nm in THIRD_PARTY:
                    continue
                if nm in stems or nm in pkg_dirs or aliases.get(nm, False):
                    continue
                missing.setdefault(nm, set()).add(str(p.relative_to(layout.root)))
    known = _known_gaps(layout.root / "tests")
    unexpected = {k: v for k, v in missing.items() if k not in known}
    assert not unexpected, "local modules imported but absent from the package:\n" + "\n".join(
        "  %s  <- %s" % (k, ", ".join(sorted(v))) for k, v in sorted(unexpected.items()))
    pending = sorted(set(missing) & known)
    if pending:
        pytest.xfail("documented pending modules still missing (tests/known_gaps.txt): %s" % ", ".join(pending))


def test_stubs_are_documented(layout, stubs_active):
    """conftest may stub a module so the suite runs without torch or without the two helper
    scorers. Every such stub must be either the torch-dependent model package or a module listed
    in tests/known_gaps.txt; anything else is a silent gap and fails here."""
    known = _known_gaps(layout.root / "tests")
    allowed = {"Squidiff.scrna_datasets"} | known
    undocumented = sorted(set(stubs_active) - allowed)
    assert not undocumented, "undocumented stubs active: %s" % undocumented
