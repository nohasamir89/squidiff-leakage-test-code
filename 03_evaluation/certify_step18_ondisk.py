#!/usr/bin/env python3
"""
Purpose
-------
Certify Step 18 evaluation arrays that are ALREADY on disk, without re-running the
sampler. Needed because tasks 3 and 4 of array 18735564 finished sampling and wrote all
three output files, then stalled in the (now fixed) O(n^2) post-run check and were killed
for an idle GPU before their provenance record was written. The arrays are complete and
valid; only the record is missing. Re-running them would spend GPU hours on work already
done.

This runs the SAME checks as the post-run block of eval_step18_tahoe.sh (with the npz
hoist applied) and writes the same provenance record, plus explicit markers saying the
record was written by this tool rather than by the launcher. CPU only; reads nothing but
files already on disk.

Logic
-----
For each --tag:
  1. Decode the tag into kind / split / seed / arm and derive the SLURM array task id the
     launcher would have used (ot: T = split_idx*3 + seed; recipe: T = 9 + split_idx*3 + seed).
     The derived id must match the log file that exists, which links record to run.
  2. Verify the checkpoint named in the training provenance still hashes to the certified
     value - the launcher's pre-run guard.
  3. Verify the log of that task: it must contain the write line for THIS tag (so the log
     and the arrays belong together), the sampler's detected arm / architecture / drug
     encoding, the harness drug encoder, the Step 18 ctrl line, and guidance_w=None.
  4. Run the launcher's output checks: files present and non-empty, condition count equals
     the data record, labels unique, true/A/B/C finite, B non-negative, per-cell file has
     ctrl, pred and ctrl aligned per condition and finite, population <= the 200-cell cap.
  5. For kind=recipe only, run the reproduction check against the Step 15b arrays
     (true, C, cond bitwise; real cells bitwise; max |dB| and |dpred| <= 1e-3).
  6. Write provenance/s18ev_<tag>.json (ot) or provenance/s18rv_<tag>.json (recipe).
Refuses to overwrite an existing record. --dry-run runs every check and writes nothing.
"""
import argparse, ast, datetime, hashlib, json, os, re, subprocess, sys
import numpy as np

W = os.environ.get("S18_ROOT", "/scratch/ni2071/Diffusion/Squidiff_git/Squidiff")
PREREG = "PREREG_step18_tahoe_ot_pairing.md"
RECORD = "tahoe_results/step15_data_record.json"
OTREC = "tahoe_results/ot_pairing_check.json"
NFE = 50
TOL = 1e-3          # the launcher's reproduction tolerance, unchanged


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 24), b""):
            h.update(c)
    return h.hexdigest()


def decode(tag):
    m = re.fullmatch(r"s18_(drug_split_[012])_ot_pair_seed_([012])", tag)
    if m:
        si, seed = int(m.group(1)[-1]), int(m.group(2))
        return dict(kind="ot", arm="ot_pair", split=m.group(1), split_idx=si, seed=seed,
                    task=si * 3 + seed, prov="provenance/s18ev_%s.json" % tag,
                    prov_tr="provenance/s18tr_%s_ot_pair_seed_%d.json" % (m.group(1), seed))
    m = re.fullmatch(r"s18r_(drug_split_[012])_recipe_seed_([012])", tag)
    if m:
        si, seed = int(m.group(1)[-1]), int(m.group(2))
        return dict(kind="recipe", arm="recipe", split=m.group(1), split_idx=si, seed=seed,
                    task=9 + si * 3 + seed, prov="provenance/s18rv_%s.json" % tag,
                    prov_tr="provenance/s15tr_%s_recipe_seed_%d.json" % (m.group(1), seed))
    sys.exit("FATAL: tag %r is neither s18_<split>_ot_pair_seed_<k> nor s18r_<split>_recipe_seed_<k>" % tag)


def certify(tag, array_job, commit, dry):
    e = decode(tag)
    v = "tahoe_results/vectors_%s.npz" % tag
    pc = "tahoe_results/vectors_%s_pc.npz" % tag
    js = "tahoe_results/leakage_%s.json" % tag
    logp = "logger_files/s18ev_%s_%d.out" % (array_job, e["task"])

    if os.path.exists(e["prov"]):
        sys.exit("FATAL: %s already exists; refusing to overwrite a certification record" % e["prov"])
    for f in (v, pc, js, logp, e["prov_tr"], RECORD, OTREC):
        if not os.path.isfile(f) or os.path.getsize(f) == 0:
            sys.exit("FATAL: missing or empty %s" % f)

    # the data record keys `splits` by STRING index, exactly as the launcher reads it
    expect_n = int(json.load(open(RECORD))["splits"][str(e["split_idx"])]["n_test_conditions"])

    # --- 2. the checkpoint is still the certified one (launcher pre-run guard) ---
    tr = json.load(open(e["prov_tr"]))
    if tr.get("status") != "completed" or tr.get("split") != e["split"] or int(tr.get("seed", -1)) != e["seed"] or tr.get("arm") != e["arm"]:
        sys.exit("FATAL: %s is not a completed record for arm %s split %s seed %d" % (e["prov_tr"], e["arm"], e["split"], e["seed"]))
    ckpt = tr["checkpoint"]
    if sha(ckpt) != tr.get("checkpoint_sha256"):
        sys.exit("FATAL: checkpoint %s no longer hashes to the certified value in %s" % (ckpt, e["prov_tr"]))
    if e["kind"] == "ot":
        ot = json.load(open(OTREC))
        if tr.get("control_ot_sha256") != ot["written"][e["split"]]["sha256"]:
            sys.exit("FATAL: the run's OT control pin differs from the pairing record")

    # --- 3. the log belongs to these arrays and binds the architecture ---
    log = open(logp).read()
    if ("vectors_%s.npz" % tag) not in log:
        sys.exit("FATAL: %s does not contain the write line for %s; log and arrays may not belong together" % (logp, tag))
    m = re.search(r"encoder arm detected from checkpoint: (\S+) \(memory_size=(\d+)\)", log)
    if not m or m.group(1) != "mlp" or int(m.group(2)) != 0:
        sys.exit("FATAL: sampler detected arm %r, expected mlp 0" % (m.group(0) if m else None))
    a = re.search(r"checkpoint architecture: (\S+)", log)
    if not a or a.group(1) != "adaLN-Zero":
        sys.exit("FATAL: sampler architecture %r, expected adaLN-Zero" % (a.group(1) if a else None))
    de = re.search(r"drug encoding detected from checkpoint: (\S+) \(drug_dimension=(\d+)\)", log)
    if not de or de.group(1) != "fingerprint" or int(de.group(2)) != 1024:
        sys.exit("FATAL: sampler drug encoding %r, expected fingerprint (1024)" % (de.group(0) if de else None))
    he = re.search(r"\[harness\] drug encoder = (\S+) \(drug_dimension=(\d+)\)", log)
    if not he or he.group(1) != "fingerprint" or int(he.group(2)) != 1024:
        sys.exit("FATAL: harness drug encoder %r, expected fingerprint (1024)" % (he.group(0) if he else None))
    if "[step18] harness stores the conditioning control cells" not in log:
        sys.exit("FATAL: %s has no Step 18 ctrl line" % logp)
    g = re.search(r"\[step12\] guidance_w=(\S+)", log)
    if not g or g.group(1) != "None":
        sys.exit("FATAL: guidance_w=%s for an unguided arm" % (g.group(1) if g else None))

    # --- 4. the launcher's output checks (npz members read ONCE) ---
    d = np.load(v, allow_pickle=True)
    n = len(d["cond"])
    if n != expect_n:
        sys.exit("FATAL: %d conditions, data record expects %d for %s" % (n, expect_n, e["split"]))
    if len(set(map(str, d["cond"]))) != n:
        sys.exit("FATAL: duplicate condition labels")
    for k in ("true", "A", "B", "C"):
        if not np.all(np.isfinite(d[k])):
            sys.exit("FATAL: non-finite %s" % k)
    if float(np.asarray(d["B"]).min()) < 0:
        sys.exit("FATAL: negative predicted expression")
    p = np.load(pc, allow_pickle=True)
    if "ctrl" not in p.files:
        sys.exit("FATAL: the per-cell file has no ctrl key")
    pc_pred, pc_ctrl, pc_true = p["pred"], p["ctrl"], p["true"]
    if len(pc_pred) != expect_n or len(pc_ctrl) != expect_n:
        sys.exit("FATAL: per-cell file has %d / %d conditions" % (len(pc_pred), len(pc_ctrl)))
    nc = [int(np.asarray(x).shape[0]) for x in pc_pred]
    if max(nc) > 200:
        sys.exit("FATAL: per-cell population larger than the 200-cell cap: %d" % max(nc))
    for i in range(expect_n):
        if np.asarray(pc_ctrl[i]).shape != np.asarray(pc_pred[i]).shape:
            sys.exit("FATAL: ctrl and pred shapes differ in condition %d" % i)
        if not np.all(np.isfinite(np.asarray(pc_ctrl[i], np.float32))):
            sys.exit("FATAL: non-finite ctrl in condition %d" % i)

    # --- 5. recipe re-evaluations must reproduce the Step 15b arrays ---
    repro = None
    if e["kind"] == "recipe":
        ref = "tahoe_results/vectors_s15_%s_recipe_seed_%d.npz" % (e["split"], e["seed"])
        refpc = ref.replace(".npz", "_pc.npz")
        pj = json.load(open("provenance/s15ev_s15_%s_recipe_seed_%d.json" % (e["split"], e["seed"])))
        if pj.get("vectors_sha256") != sha(ref) or pj.get("per_cell_sha256") != sha(refpc):
            sys.exit("FATAL: the Step 15b recipe arrays differ from their provenance")
        r = np.load(ref, allow_pickle=True)
        rp = np.load(refpc, allow_pickle=True)
        if not np.array_equal(np.array([str(x) for x in r["cond"]]), np.array([str(x) for x in d["cond"]])):
            sys.exit("FATAL: condition order differs from the Step 15b arrays")
        for k in ("true", "C"):
            if not np.array_equal(np.asarray(r[k]), np.asarray(d[k])):
                sys.exit("FATAL: %s differs from the Step 15b arrays (data, must be bitwise equal)" % k)
        dB = float(np.abs(np.asarray(r["B"], np.float64) - np.asarray(d["B"], np.float64)).max())
        dA = float(np.abs(np.asarray(r["A"], np.float64) - np.asarray(d["A"], np.float64)).max())
        dP = 0.0
        rp_true, rp_pred = rp["true"], rp["pred"]
        for i in range(expect_n):
            if not np.array_equal(np.asarray(rp_true[i], np.float32), np.asarray(pc_true[i], np.float32)):
                sys.exit("FATAL: stored real cells differ from the Step 15b per-cell file in condition %d" % i)
            a_, b_ = np.asarray(rp_pred[i], np.float64), np.asarray(pc_pred[i], np.float64)
            if a_.shape != b_.shape:
                sys.exit("FATAL: per-cell population shape differs from Step 15b in condition %d" % i)
            dP = max(dP, float(np.abs(a_ - b_).max()))
        if dB > TOL or dP > TOL:
            sys.exit("FATAL: the re-evaluation differs from the Step 15b arrays (max |dB| %.2e, max |dpred| %.2e > %g)" % (dB, dP, TOL))
        repro = dict(reference_vectors=ref, reference_per_cell=refpc, max_abs_diff_B=dB,
                     max_abs_diff_A=dA, max_abs_diff_pred=dP,
                     true_C_cond_bitwise_equal=True, real_cells_bitwise_equal=True)
        print("[repro] %s reproduces the Step 15b arrays: max |dB| %.2e, max |dA| %.2e, max |dpred| %.2e" % (tag, dB, dA, dP))

    rec = dict(status="completed", tag=tag, kind=e["kind"], task=e["task"], slurm_array_job=str(array_job),
               split=e["split"], arm=e["arm"], seed=e["seed"], encoder_arm="mlp", use_adaln=True,
               fm_source="gaussian", cond_drop=0.0, drug_encoding="fingerprint", drug_dimension=1024,
               sampler="euler_ode", nfe=NFE, guidance_w=None, guide_t_min=0.0, max_cells_per_cond=200,
               n_conditions=n, cells_min=min(nc), cells_max=max(nc), expect_conditions=expect_n, ctrl_stored=True,
               prereg=PREREG, commit=commit, training_commit=tr.get("commit"), training_provenance=e["prov_tr"],
               checkpoint=ckpt, checkpoint_sha256=tr["checkpoint_sha256"], control_ot_sha256=tr.get("control_ot_sha256"),
               vectors_npz=v, vectors_sha256=sha(v), per_cell_npz=pc, per_cell_sha256=sha(pc),
               leakage_json=js, leakage_sha256=sha(js),
               data_record=RECORD, data_record_sha256=sha(RECORD), pairing_record=OTREC, pairing_record_sha256=sha(OTREC),
               harness_sha256=sha("leakage_test_fm_step18.py"), harness_src_sha256=sha("leakage_test_fm_tahoe.py"),
               harness_base_sha256=sha("leakage_test_base_tahoe.py"), sampler_sha256=sha("sample_squidiff.py"),
               datasets_sha256=sha("Squidiff/scrna_datasets.py"), flow_matching_sha256=sha("Squidiff/flow_matching.py"),
               reproduction=repro,
               node=subprocess.run("hostname", shell=True, capture_output=True, text=True).stdout.strip(),
               completed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
               # --- honesty markers: this record was NOT written by the launcher ---
               certified_from_disk=True,
               certified_by="certify_step18_ondisk.py",
               certified_reason="the launcher wrote these arrays then stalled in the pre-fix O(n^2) post-run check and was killed for an idle GPU; the arrays are the launcher's, the record is this tool's",
               eval_log=logp)
    if dry:
        print("[dryrun] %s: all checks passed, nothing written (%d conditions, cells %d-%d)" % (tag, n, min(nc), max(nc)))
        return
    json.dump(rec, open(e["prov"], "w"), indent=2)
    print("[certified] %s -> %s: %d conditions (record %d), cells %d-%d, ctrl stored, fingerprint bound by sampler and harness, unguided"
          % (tag, e["prov"], n, expect_n, min(nc), max(nc)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--array-job", required=True, help="the SLURM array job id whose logs hold these runs, e.g. 18735564")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    os.chdir(W)
    commit = subprocess.run("git rev-parse HEAD", shell=True, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run("git diff --quiet -- %s leakage_test_fm_step18.py leakage_test_fm_tahoe.py leakage_test_base_tahoe.py sample_squidiff.py Squidiff" % PREREG,
                           shell=True).returncode
    if dirty:
        sys.exit("FATAL: prereg, harness, sampler or encoder sources are uncommitted; commit before certifying")
    n = len(re.findall(r"^## ", open(PREREG).read(), flags=re.M))
    if n < 9:
        sys.exit("FATAL: prereg has %d sections, expected at least 9" % n)
    print("[guards] prereg %d sections | sources clean | commit %s" % (n, commit[:7]))
    for t in args.tags:
        certify(t, args.array_job, commit, args.dry_run)


if __name__ == "__main__":
    main()
