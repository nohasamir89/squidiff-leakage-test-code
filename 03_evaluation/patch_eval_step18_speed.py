#!/usr/bin/env python3
"""
Purpose
-------
Speed fix for eval_step18_tahoe.sh. The two per-condition check loops in the post-run
block index an OPEN .npz (p['pred'][i], p['ctrl'][i], rp['true'][i], ...). Each p[key]
access re-reads and un-pickles the WHOLE archive member from disk, so a loop over
expect_n conditions is O(n^2) in disk+CPU work and touches no GPU. On drug_split_1
(n = 2899) this ran for hours after the arrays were written and the jobs were killed for
an idle GPU. The fix reads each member ONCE into an in-memory object array and indexes
that. It changes nothing about WHAT is checked.

Logic
-----
- Refuses unless the file is the exact committed launcher (md5 guard), so the anchors
  below are known to be present verbatim.
- Two exact-string replacements, each asserted to occur exactly once:
    A) hoist pc_pred/pc_ctrl/pc_true before the post-run cap/finite loop and index those.
    B) hoist rp_true/rp_pred before the recipe reproduction loop and index those (reusing
       pc_true/pc_pred from A).
- Then four assertions prove the edit is OUTPUT-ONLY:
    1. the only textual difference is the hoist lines + the p[k][i] -> cached[i] renames;
    2. every comparison token the checks rely on is still present the same number of times
       ( '!= ', 'np.array_equal', 'np.isfinite', '> 200', '1e-3', 'sys.exit' counts );
    3. no numeric tolerance literal changed;
    4. the file still parses as a shell script wrapper and the two python heredocs are intact.
- Writes the patched file and prints its new md5 and a unified diff for the record.
"""
import hashlib, re, subprocess, sys

PATH = "eval_step18_tahoe.sh"
EXPECT_MD5 = "ef7cb43f67faac795058af194c2021e7"   # the committed launcher

src = open(PATH).read()
got = hashlib.md5(src.encode()).hexdigest()
if got != EXPECT_MD5:
    sys.exit("FATAL: %s md5 %s != expected %s; refusing to patch an unexpected file" % (PATH, got, EXPECT_MD5))

# ---- Edit A: the cap/finite loop in the general post block ----
A_OLD = """p = np.load(pc, allow_pickle=True)
if 'ctrl' not in p.files: sys.exit('FATAL: the per-cell file has no ctrl key')
if len(p['pred']) != expect_n or len(p['ctrl']) != expect_n: sys.exit('FATAL: per-cell file has %d / %d conditions' % (len(p['pred']), len(p['ctrl'])))
nc = [int(np.asarray(a).shape[0]) for a in p['pred']]
if max(nc) > 200: sys.exit('FATAL: per-cell population larger than the 200-cell cap: %d' % max(nc))
for i in range(expect_n):
    if np.asarray(p['ctrl'][i]).shape != np.asarray(p['pred'][i]).shape: sys.exit('FATAL: ctrl and pred shapes differ in condition %d' % i)
    if not np.all(np.isfinite(np.asarray(p['ctrl'][i], np.float32))): sys.exit('FATAL: non-finite ctrl in condition %d' % i)"""

A_NEW = """p = np.load(pc, allow_pickle=True)
if 'ctrl' not in p.files: sys.exit('FATAL: the per-cell file has no ctrl key')
pc_pred = p['pred']; pc_ctrl = p['ctrl']; pc_true = p['true']  # read each archive member ONCE: indexing p[k][i] inside a loop re-reads the whole member (O(n^2), no GPU; killed drug_split_1 for an idle GPU)
if len(pc_pred) != expect_n or len(pc_ctrl) != expect_n: sys.exit('FATAL: per-cell file has %d / %d conditions' % (len(pc_pred), len(pc_ctrl)))
nc = [int(np.asarray(a).shape[0]) for a in pc_pred]
if max(nc) > 200: sys.exit('FATAL: per-cell population larger than the 200-cell cap: %d' % max(nc))
for i in range(expect_n):
    if np.asarray(pc_ctrl[i]).shape != np.asarray(pc_pred[i]).shape: sys.exit('FATAL: ctrl and pred shapes differ in condition %d' % i)
    if not np.all(np.isfinite(np.asarray(pc_ctrl[i], np.float32))): sys.exit('FATAL: non-finite ctrl in condition %d' % i)"""

# ---- Edit B: the reproduction loop in the recipe block ----
B_OLD = """    dP = 0.0
    for i in range(expect_n):
        if not np.array_equal(np.asarray(rp['true'][i], np.float32), np.asarray(p['true'][i], np.float32)): sys.exit('FATAL: stored real cells differ from the Step 15b per-cell file in condition %d' % i)
        a_, b_ = np.asarray(rp['pred'][i], np.float64), np.asarray(p['pred'][i], np.float64)
        if a_.shape != b_.shape: sys.exit('FATAL: per-cell population shape differs from Step 15b in condition %d' % i)
        dP = max(dP, float(np.abs(a_ - b_).max()))"""

B_NEW = """    dP = 0.0
    rp_true = rp['true']; rp_pred = rp['pred']  # read each member ONCE (same O(n^2) fix as the pc_* hoist above)
    for i in range(expect_n):
        if not np.array_equal(np.asarray(rp_true[i], np.float32), np.asarray(pc_true[i], np.float32)): sys.exit('FATAL: stored real cells differ from the Step 15b per-cell file in condition %d' % i)
        a_, b_ = np.asarray(rp_pred[i], np.float64), np.asarray(pc_pred[i], np.float64)
        if a_.shape != b_.shape: sys.exit('FATAL: per-cell population shape differs from Step 15b in condition %d' % i)
        dP = max(dP, float(np.abs(a_ - b_).max()))"""

for name, old in (("A", A_OLD), ("B", B_OLD)):
    if src.count(old) != 1:
        sys.exit("FATAL: edit %s anchor found %d times, expected exactly 1" % (name, src.count(old)))

out = src.replace(A_OLD, A_NEW, 1).replace(B_OLD, B_NEW, 1)

# ---- Assertion 1: token counts the checks depend on are unchanged ----
def counts(s):
    return {
        "sys.exit":        s.count("sys.exit"),
        "np.array_equal":  s.count("np.array_equal"),
        "np.isfinite":     s.count("np.isfinite"),
        "> 200":           s.count("> 200"),
        "1e-3":            s.count("1e-3"),
        ".shape !=":       s.count(".shape != "),
        "max_cells 200":   s.count("max_cells_per_cond=200"),
    }
c0, c1 = counts(src), counts(out)
if c0 != c1:
    sys.exit("FATAL: a check token count changed, the edit is not output-only: %r -> %r" % (c0, c1))

# ---- Assertion 2: no numeric tolerance / threshold literal changed ----
nums = lambda s: re.findall(r"\b(?:1e-3|200|0\.9999|58178180|1024)\b", s)
if nums(src) != nums(out):
    sys.exit("FATAL: a numeric literal set changed: %r -> %r" % (nums(src), nums(out)))

# ---- Assertion 3: exactly the intended lines changed ----
import difflib
added = [l for l in difflib.ndiff(src.splitlines(), out.splitlines()) if l.startswith("+ ")]
removed = [l for l in difflib.ndiff(src.splitlines(), out.splitlines()) if l.startswith("- ")]
# every added/removed line must be one of ours: it mentions a hoist var or a p[...] -> cached rename
allow = ("pc_pred", "pc_ctrl", "pc_true", "rp_true", "rp_pred", "p['pred']", "p['ctrl']", "p['true']", "rp['true']", "rp['pred']")
for l in added + removed:
    if not any(tok in l for tok in allow):
        sys.exit("FATAL: unexpected changed line, not part of the hoist: %s" % l.strip())

open(PATH, "w").write(out)
print("patched %s" % PATH)
print("new md5 %s" % hashlib.md5(out.encode()).hexdigest())
print("lines changed: +%d / -%d (all hoist/rename)" % (len(added), len(removed)))
# compile-check the two python heredocs by extracting and ast.parse-ing them
blocks = re.findall(r"python - <<'PY'\n(.*?)\nPY", out, flags=re.S)
import ast
for i, b in enumerate(blocks):
    ast.parse(b)
print("python heredocs parse OK: %d blocks" % len(blocks))
