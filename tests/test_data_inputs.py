"""Every data record a packaged script reads must be in the package.

Why this test exists
--------------------
`test_package.py::test_local_imports_resolve_inside_package` catches a *module* the
release forgot. This test catches a *record* the release forgot. Scorers, extractors and
figure scripts read the records of earlier steps (`step7_interaction_baseline.json`,
`pathway_vs_fingerprint.json`, `step8_convergence.json`, ...). A number the manuscript
quotes from such a record has no source in the package unless the record is here. The
first release package was missing about fifteen of them; the figure-value tests did not
notice because the figures read only the final records.

How it works
------------
1. Every string literal ending in .json / .csv / .tsv in the packaged Python and R
   scripts (01_data_preparation, 04_scoring, 05_figures, 06_supplementary,
   08_external_pipelines) is a candidate file name. 06_supplementary holds the scripts
   that assemble the supplementary tables; it was added on 15 Nov 2026, when those
   scripts were complete, so that a record a table is built from cannot be left out.
   Templates (containing % or {) are ignored; per-cell arrays (.npz) are out of scope
   by design (too large to package).
2. A literal is an OUTPUT of the script when the line that names it writes it
   (to_csv, json.dump, open(..., 'w'), DictWriter, write.csv, ggsave, ...) or assigns it
   to a variable whose name starts with OUT / out. Every other literal is an INPUT.
3. Every INPUT name must exist somewhere in the package (file names are unique across
   the package folders), unless it is listed in tests/data_files_not_packaged.txt with
   a reason (raw external data, large intermediates). Being written by one packaged
   script does not excuse a file that another packaged script reads: the reader's
   number has to be checkable without re-running the writer.
4. Every .json OUTPUT of a script in 04_scoring/ is a result record (`step6_analysis.json`,
   `step8_convergence.json`, ...). Those must be packaged too: they are what the
   manuscript's numbers come from, whether or not another script reads them.

The failure message names each missing file and the scripts that read or write it, so
the fix is either a line in manuscript_code_list.txt (and a rebuild) or a line in the
allow-list.
"""
import pathlib
import re

import pytest

LITERAL = re.compile(r"""['"]([A-Za-z0-9][A-Za-z0-9_./\-]*\.(?:json|csv|tsv))['"]""")
WRITES = re.compile(r"""to_csv|json\.dump\(|DictWriter|open\([^)]*,\s*['"][wa]|['"]w['"]|\bwrite|fwrite|write\.csv|write_csv|saveRDS|ggsave""")
OUT_VAR = re.compile(r"""^\s*(?:[A-Za-z_][A-Za-z0-9_]*\s*,\s*)*(?:OUT|out)[A-Za-z0-9_]*\s*(?:,\s*[A-Za-z_][A-Za-z0-9_]*\s*)*=[^=]""")
FRAGMENT_LINE = re.compile(r"""['"][^'"]*%s[^'"]*['"]\s*%""")       # '%s_%s' % (stem, x): names on this line are suffixes
SCRIPT_DIRS = ("01_data_preparation", "04_scoring", "05_figures", "06_supplementary", "08_external_pipelines")
ALLOW_FILE = "data_files_not_packaged.txt"


def _allowed(tests_dir):
    p = tests_dir / ALLOW_FILE
    if not p.exists():
        return {}
    out = {}
    for ln in p.read_text().splitlines():
        name, _, reason = ln.partition("#")
        if name.strip():
            out[name.strip()] = reason.strip()
    return out


def scan(root: pathlib.Path):
    """Return {file name: {'reads': set(scripts), 'writes': set(scripts)}} over the packaged scripts."""
    uses = {}
    for d in SCRIPT_DIRS:
        for p in sorted((root / d).rglob("*")):
            if not p.is_file() or p.suffix not in (".py", ".R", ".r"):
                continue
            for ln in p.read_text(errors="ignore").splitlines():
                if FRAGMENT_LINE.search(ln):
                    continue
                for m in LITERAL.finditer(ln):
                    name = pathlib.Path(m.group(1)).name
                    if "%" in name or "{" in name:
                        continue
                    kind = "writes" if (WRITES.search(ln) or OUT_VAR.match(ln)) else "reads"
                    uses.setdefault(name, {"reads": set(), "writes": set()})[kind].add(str(p.relative_to(root)))
    return uses


def test_data_files_read_by_packaged_scripts_are_packaged(layout):
    if not layout.is_package:
        pytest.skip("package-only check (the repository holds every file by construction)")
    present = {p.name for p in layout.root.rglob("*") if p.is_file()}
    allowed = _allowed(layout.root / "tests")
    uses = scan(layout.root)
    missing = {n: u["reads"] for n, u in uses.items() if u["reads"] and n not in present and n not in allowed}
    records = {n: u["writes"] for n, u in uses.items()
               if n.endswith(".json") and any(s.startswith("04_scoring/") for s in u["writes"])
               and n not in present and n not in allowed and n not in missing}
    stale_allow = sorted(n for n in allowed if n in present)
    msg = []
    if missing:
        msg.append("records read by packaged scripts but absent from the package "
                   "(add to manuscript_code_list.txt and rebuild, or list in tests/%s with a reason):" % ALLOW_FILE)
        msg += ["  %-45s <- %s" % (n, ", ".join(sorted(r))) for n, r in sorted(missing.items())]
    if records:
        msg.append("result records written by packaged scorers but absent from the package:")
        msg += ["  %-45s <- written by %s" % (n, ", ".join(sorted(w))) for n, w in sorted(records.items())]
    if stale_allow:
        msg.append("listed in tests/%s but now packaged (delete the line): %s" % (ALLOW_FILE, ", ".join(stale_allow)))
    assert not msg, "\n".join(msg)
    assert uses, "no data-file literals found - scanner broken?"
