#!/usr/bin/env python
# v2 (patched by patch_gbm_eval_scale.py): the decoded predictions are NOT logged a
# second time; they are clipped at zero and compared with the log1p truth directly.
# The original script is kept unchanged next to this file for the record.
"""
GBM gene-count sensitivity — A/B/C three-way eval for ONE gene count N.

Built directly on your eval_gbm_deltaz_200.py: SAME encode_batch (MAX_CELLS=300,
RNG), SAME decode_mean (batch-decode then average; log1p-clip applied by caller),
SAME Delta-z construction (keepdim means), SAME reconstruction sanity check.
Extended to the full leakage three-way and to r2 + Pearson on all-genes and DEGs.

Protocol meaning on GBM (state in methods; differs from the Tahoe leakage test):
  A  RECONSTRUCTION      encode the held-out TREATED cells, decode. Sanity ceiling.
  B  GENUINE PREDICTION  encode the held-out type's CONTROL cells, ADD Delta-z
     (Delta-z transfer)  learned from MYELOID, decode. == your existing eval; the
                         published ~0.91 is a B-type number, not an A-type one.
  C  CONTROL MEAN        held-out type's vehicle mean, no model.

Mandatory guard (your Correction 3): vehicle-only reconstruction R is computed
FIRST. If < RECON_FLOOR at this N, prediction numbers are flagged unreliable
(broken decode, not biology).

Appends one row per (N, cell_type) to OUT_ROOT/gene_sensitivity_results.csv.

    python eval_gbm_sensitivity.py --n_hvg 300 --data_dir .../n300 --model .../model_0.9999.pt
"""
import argparse, csv, os, sys
import numpy as np, torch, scanpy as sc
from scipy.stats import pearsonr
from datetime import datetime

sys.path.insert(0, '/scratch/ni2071/Diffusion/Squidiff_gbm')
import sample_squidiff
from Squidiff.scrna_datasets import CELL_LINE_TO_ID, CELL_LINE_ORDER

OUT_ROOT = '/scratch/ni2071/Diffusion/gbm_squidiff_robustness_test'
RESULTS_CSV = os.path.join(OUT_ROOT, 'gene_sensitivity_SEEDED_v2.csv')
HELD_OUT = ['panobinostat', 'RO4929097', 'tazemetostat', 'ispinesib', 'ANA-12']
TYPES = ['tumor', 'oligodendrocyte']
MAX_CELLS = 300                 # identical to your eval
RECON_FLOOR = 0.70              # vehicle recon below this => decode suspect, flag row
RNG = np.random.RandomState(0)


print("[scale] v2: predictions compared on the training (log1p) scale; the second log1p of the original evaluation is removed")
def dense(x): return x.toarray() if hasattr(x, 'toarray') else np.asarray(x)
def r2(a, b):
    ss_res = np.sum((a - b) ** 2); ss_tot = np.sum((a - a.mean()) ** 2)
    return 0.0 if ss_tot == 0 else 1 - ss_res / ss_tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_hvg', type=int, required=True)
    ap.add_argument('--data_dir', required=True)
    ap.add_argument('--model', required=True)
    args = ap.parse_args()
    N = args.n_hvg
    print("=" * 70)
    print(f"GBM gene-sensitivity A/B/C eval — N={N}  ({datetime.now()})")
    print("=" * 70)

    # ---- fail-fast: confirm GBM mapping, not Tahoe (before any GPU work) ----
    NEEDED = {'myeloid', 'tumor', 'oligodendrocyte'}
    missing = NEEDED - set(CELL_LINE_TO_ID.keys())
    if missing:
        raise KeyError(
            f"CELL_LINE_TO_ID is missing GBM cell types: {missing}\n"
            f"Current dict: {CELL_LINE_TO_ID}\n"
            f"CELL_LINE_ORDER: {CELL_LINE_ORDER}\n"
            "This means Squidiff_gbm/scrna_datasets.py still has the Tahoe mapping. "
            "Fix it before training.")
    print(f"[guard] CELL_LINE_TO_ID = {CELL_LINE_TO_ID}  (confirmed GBM mapping)")

    # no-fingerprint path: 4th positional arg = use_drug_structure = False
    sampler = sample_squidiff.sampler(args.model, N, N, False, n_cell_lines=3, cell_emb_dim=32)

    tr   = sc.read_h5ad(f'{args.data_dir}/gbm_train.h5ad')
    test = sc.read_h5ad(f'{args.data_dir}/gbm_test.h5ad')
    ctrl = sc.read_h5ad(f'{args.data_dir}/gbm_control_to.h5ad')
    for a in (tr, test, ctrl): a.X = dense(a.X)
    fz_path = f'{args.data_dir}/frac_zero.txt'
    frac_zero = float(open(fz_path).read().split('\t')[1]) if os.path.exists(fz_path) else float('nan')

    def encode_batch(adata, mask, cell_type):
        X = adata.X[mask]
        if X.shape[0] == 0: return None
        if X.shape[0] > MAX_CELLS:
            X = X[RNG.choice(X.shape[0], MAX_CELLS, replace=False)]
        xb = torch.tensor(X, dtype=torch.float32).to('cuda')
        cid = torch.full((X.shape[0],), CELL_LINE_TO_ID[cell_type], dtype=torch.long).to('cuda')
        with torch.no_grad():
            z = sampler.model.encoder(xb, None, None, cid)     # drug_dose=None
        return z

    def encode_X(X, cell_type):
        """encode an explicit array (for Protocol A on treated cells)."""
        if X.shape[0] == 0: return None
        if X.shape[0] > MAX_CELLS:
            X = X[RNG.choice(X.shape[0], MAX_CELLS, replace=False)]
        xb = torch.tensor(X, dtype=torch.float32).to('cuda')
        cid = torch.full((X.shape[0],), CELL_LINE_TO_ID[cell_type], dtype=torch.long).to('cuda')
        with torch.no_grad():
            z = sampler.model.encoder(xb, None, None, cid)
        return z

    def decode_mean(z, seed=42):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        out = []
        for s in range(0, z.shape[0], 256):
            pb = sampler.pred(z[s:s + 256], gene_size=N)
            out.append(pb.detach().cpu().numpy())
        return np.concatenate(out, 0).mean(0)                  # batch-decode THEN average

    m_drug = tr.obs['drug'].astype(str).values
    m_ct   = tr.obs['cell_type'].astype(str).values

    # ---- Delta-z directions from MYELOID (identical to your eval) ----
    z_veh_m = encode_batch(tr, (m_ct == 'myeloid') & (m_drug == 'vehicle'), 'myeloid')
    assert z_veh_m is not None, "no myeloid vehicle cells"
    z_veh_m_mean = z_veh_m.mean(0, keepdim=True)
    Dz = {}
    for d in HELD_OUT:
        zd = encode_batch(tr, (m_ct == 'myeloid') & (m_drug == d), 'myeloid')
        if zd is None: continue
        Dz[d] = zd.mean(0, keepdim=True) - z_veh_m_mean
    dz_norm = float(np.mean([float(torch.linalg.norm(v.cpu())) for v in Dz.values()])) if Dz else float('nan')

    # ---- MANDATORY: vehicle-only reconstruction sanity, per type (your Correction 3) ----
    recon_R = {}
    z_start = {}
    for ct in TYPES:
        c_mask = ctrl.obs['cell_type'].astype(str).values == ct
        zb = encode_batch(ctrl, c_mask, ct)
        z_start[ct] = zb
        if zb is None: recon_R[ct] = float('nan'); continue
        pred_v = np.clip(decode_mean(zb), 0, None)
        true_v = ctrl.X[c_mask].mean(0)
        recon_R[ct], _ = pearsonr(true_v, pred_v)
    recon_ok = all((not np.isnan(v)) and v >= RECON_FLOOR for v in recon_R.values())

    # ---- SMOKE-TEST ASSERTION (self-reports pass/fail at N=200) ----
    if N == 200:
        recon_tumor = recon_R.get('tumor', float('nan'))
        recon_oligo = recon_R.get('oligodendrocyte', float('nan'))
        smoke_recon = (not np.isnan(recon_tumor)) and recon_tumor >= 0.95 \
                   and (not np.isnan(recon_oligo)) and recon_oligo >= 0.95
        smoke_dz    = (not np.isnan(dz_norm)) and 0.3 <= dz_norm <= 1.5
        smoke_pass  = smoke_recon and smoke_dz
        print("\n" + ("=" * 50))
        print(f"SMOKE TEST N=200")
        print(f"  recon_R tumor={recon_tumor:.3f} oligo={recon_oligo:.3f}  "
              f"(need >= 0.95)  {'PASS' if smoke_recon else 'FAIL'}")
        print(f"  dz_norm={dz_norm:.3f}  (need 0.3-1.5)  {'PASS' if smoke_dz else 'FAIL'}")
        print(f"  OVERALL: {'PASS — pipeline faithful, fire the full array' if smoke_pass else 'FAIL — do NOT launch remaining tasks; debug first'}")
        print("=" * 50 + "\n")
        if not smoke_pass:
            print("[v2] smoke-test threshold (0.95) was set for the old scale; not aborting, row is written")

    # ---- A / B / C per (held-out type, held-out drug) ----
    acc = {ct: {k: [] for k in
                ['A_r2_all','A_r2_deg','A_R_all','A_R_deg',
                 'B_r2_all','B_r2_deg','B_R_all','B_R_deg',
                 'C_r2_all','C_r2_deg','C_R_all','C_R_deg']} for ct in TYPES}

    for ct in TYPES:
        c_mask = ctrl.obs['cell_type'].astype(str).values == ct
        if c_mask.sum() == 0: continue
        ctrl_mean = ctrl.X[c_mask].mean(0)                     # Protocol C (all vehicle, uncapped)
        zb = z_start[ct]                                       # encoded vehicle latents (reused)
        for d in HELD_OUT:
            if d not in Dz: continue
            t_mask = ((test.obs['cell_type'].astype(str).values == ct) &
                      (test.obs['drug'].astype(str).values == d))
            if t_mask.sum() == 0: continue
            true_mean = test.X[t_mask].mean(0)
            degs = np.argsort(np.abs(true_mean - ctrl_mean))[-50:]

            # A: reconstruction of TREATED
            zA = encode_X(test.X[t_mask], ct)
            predA = np.clip(decode_mean(zA), 0, None)
            # B: genuine prediction (your existing protocol) — vehicle latents + Dz
            predB = np.clip(decode_mean(zb + Dz[d]), 0, None)
            # C: control mean
            predC = ctrl_mean

            for tag, pred in [('A', predA), ('B', predB), ('C', predC)]:
                acc[ct][f'{tag}_r2_all'].append(r2(true_mean, pred))
                acc[ct][f'{tag}_r2_deg'].append(r2(true_mean[degs], pred[degs]))
                ra, _ = pearsonr(true_mean, pred)
                rd, _ = pearsonr(true_mean[degs], pred[degs])
                acc[ct][f'{tag}_R_all'].append(float(ra))
                acc[ct][f'{tag}_R_deg'].append(float(rd))

    def agg(ct, k):
        v = acc[ct][k]; return float(np.mean(v)) if v else float('nan')

    header = ['n_hvg','cell_type','frac_zero','dz_norm','recon_R_vehicle','recon_ok',
              'A_r2_all','A_r2_deg','A_R_all','A_R_deg',
              'B_r2_all','B_r2_deg','B_R_all','B_R_deg',
              'C_r2_all','C_r2_deg','C_R_all','C_R_deg',
              'BminusC_r2_deg','BminusC_R_deg','leakage_gap_r2_all']
    os.makedirs(OUT_ROOT, exist_ok=True)
    write_header = not os.path.exists(RESULTS_CSV)
    # Release fix (21 Oct 2026): the CSV is appended to by every array task, so a re-run of one
    # panel size would add a second row per cell type and the figure script would read both.
    # Refuse if this N is already recorded (the shipped record holds exactly one row per
    # (n_hvg, cell_type)).
    if not write_header:
        with open(RESULTS_CSV, newline='') as f:
            seen = [row for row in csv.DictReader(f) if int(row['n_hvg']) == N]
        if seen:
            raise SystemExit(f'FATAL: {RESULTS_CSV} already holds {len(seen)} row(s) for n_hvg={N}; '
                             f'remove them before re-evaluating this size')
    with open(RESULTS_CSV, 'a', newline='') as f:
        w = csv.writer(f)
        if write_header: w.writerow(header)
        for ct in TYPES:
            bmc_r2  = agg(ct,'B_r2_deg') - agg(ct,'C_r2_deg')
            bmc_R   = agg(ct,'B_R_deg')  - agg(ct,'C_R_deg')
            gap_all = agg(ct,'A_r2_all') - agg(ct,'B_r2_all')
            w.writerow([N, ct, f'{frac_zero:.4f}', f'{dz_norm:.4f}',
                        f'{recon_R.get(ct, float("nan")):.4f}', int(recon_ok)]
                       + [f'{agg(ct,k):.4f}' for k in header[6:18]]
                       + [f'{bmc_r2:.4f}', f'{bmc_R:.4f}', f'{gap_all:.4f}'])

    print(f"\n[N={N}] frac_zero={frac_zero:.3f} dz_norm={dz_norm:.3f} recon_ok={recon_ok}")
    for ct in TYPES:
        print(f"  {ct:<16} recon_R={recon_R.get(ct, float('nan')):.3f} | "
              f"A_R_all={agg(ct,'A_R_all'):.3f} B_R_all={agg(ct,'B_R_all'):.3f} C_R_all={agg(ct,'C_R_all'):.3f} | "
              f"B-C(DEG r2)={agg(ct,'B_r2_deg')-agg(ct,'C_r2_deg'):+.3f} "
              f"B-C(DEG R)={agg(ct,'B_R_deg')-agg(ct,'C_R_deg'):+.3f}")
    if not recon_ok:
        print(f"  !!! recon below {RECON_FLOOR}: B/C numbers at N={N} are UNRELIABLE "
              f"(broken decode, not biology). See §9.5 Correction 3.")
    print(f"Finished: {datetime.now()}")


if __name__ == '__main__':
    main()