#!/usr/bin/env python
"""
GBM drug-response eval (Fig 3f) — CORRECTED decode.

KEY FIX (matches training_2 leakage-test proven pattern):
  Diffusion decode is nonlinear/stochastic, so decode(mean(z)) != mean(decode(z)).
  The earlier version decoded ONE mean latent -> reconstruction R ~0.13 (broken).
  Correct: encode the FULL batch of cells -> per-cell latents -> decode ALL ->
  average the decoded EXPRESSION. That is "mean predicted expression".

Delta-z method (paper task 3):
  direction Dz_drug = mean(z|myeloid,drug) - mean(z|myeloid,vehicle)   (latent arithmetic)
  predict (cell_type C, drug d):
     z_batch = per-cell vehicle latents of C  [N,D]
     z_pred  = z_batch + Dz_drug              [N,D]   (add direction to each cell)
     pred_mean = decode(z_pred).mean(0)       -> mean expression over genes
  score vs true held-out mean for (C,d).

Reconstruction sanity (vehicle, no Dz): decode(vehicle batch).mean(0) vs true vehicle mean.
GPU required.
"""
import sys
sys.path.insert(0, '/scratch/ni2071/Diffusion/Squidiff_gbm')
import numpy as np, torch, scanpy as sc
from scipy.stats import pearsonr
import sample_squidiff
from Squidiff.scrna_datasets import CELL_LINE_TO_ID
from datetime import datetime

MODEL_PATH = '/scratch/ni2071/Diffusion/gbm_run/results/model_0.9999.pt'
DATA_DIR   = '/scratch/ni2071/Diffusion/datasets_gbm_paper'
GENE_SIZE, N_CL, EMB = 2000, 3, 32
HELD_OUT   = ['panobinostat', 'RO4929097', 'tazemetostat', 'ispinesib', 'ANA-12']
MAX_CELLS  = 300          # cap per group for speed (stated approximation)
RNG = np.random.RandomState(0)

def dense(x): return x.toarray() if hasattr(x, 'toarray') else np.asarray(x)
def r2(a, b):
    ss_res = np.sum((a-b)**2); ss_tot = np.sum((a-a.mean())**2)
    return 0.0 if ss_tot == 0 else 1 - ss_res/ss_tot

def main():
    print("="*70)
    print("GBM Delta-z eval (Fig 3f) — CORRECTED decode (batch decode then average)")
    print(f"Started: {datetime.now()}"); print("="*70)

    sampler = sample_squidiff.sampler(MODEL_PATH, GENE_SIZE, GENE_SIZE, False,
                                      n_cell_lines=N_CL, cell_emb_dim=EMB)

    tr   = sc.read_h5ad(f'{DATA_DIR}/gbm_train.h5ad')
    test = sc.read_h5ad(f'{DATA_DIR}/gbm_test.h5ad')
    ctrl = sc.read_h5ad(f'{DATA_DIR}/gbm_control_to.h5ad')
    for a in (tr, test, ctrl): a.X = dense(a.X)

    def encode_batch(adata, mask, cell_type):
        """Encode a subset of cells -> per-cell latents [n, D]."""
        X = adata.X[mask]
        if X.shape[0] == 0: return None
        if X.shape[0] > MAX_CELLS:
            X = X[RNG.choice(X.shape[0], MAX_CELLS, replace=False)]
        xb = torch.tensor(X, dtype=torch.float32).to('cuda')
        cid = torch.full((X.shape[0],), CELL_LINE_TO_ID[cell_type], dtype=torch.long).to('cuda')
        with torch.no_grad():
            z = sampler.model.encoder(xb, None, None, cid)   # drug_dose=None
        return z

    def decode_mean(z):
        """Decode FULL per-cell batch, return mean expression over cells. (proven pattern)"""
        out = []
        for s in range(0, z.shape[0], 256):
            pb = sampler.pred(z[s:s+256], gene_size=GENE_SIZE)
            out.append(pb.detach().cpu().numpy())
        arr = np.concatenate(out, 0)              # [n, genes]
        if not hasattr(decode_mean, "_printed"):
            print(f"\n[decoder debug] decoded batch shape={arr.shape}, "
                  f"per-gene-mean min={arr.mean(0).min():.3f} max={arr.mean(0).max():.3f}")
            decode_mean._printed = True
        return arr.mean(0)                         # mean over cells -> [genes]

    m_drug = tr.obs['drug'].astype(str).values
    m_ct   = tr.obs['cell_type'].astype(str).values

    # ---- drug directions from myeloid (latent means) ----
    z_veh_m = encode_batch(tr, (m_ct=='myeloid') & (m_drug=='vehicle'), 'myeloid')
    assert z_veh_m is not None, "no myeloid vehicle cells"
    z_veh_m_mean = z_veh_m.mean(0, keepdim=True)
    Dz = {}
    print("\nDz norms (myeloid):")
    for d in HELD_OUT:
        zd = encode_batch(tr, (m_ct=='myeloid') & (m_drug==d), 'myeloid')
        if zd is None: print(f"  {d}: no cells; skip"); continue
        Dz[d] = zd.mean(0, keepdim=True) - z_veh_m_mean
        print(f"  {d:<14}: ||Dz|| = {float(torch.linalg.norm(Dz[d].cpu())):.3f}")

    # ---- reconstruction sanity (vehicle batch, no Dz) ----
    print("\nReconstruction sanity (vehicle batch, no Dz):")
    z_start = {}
    for ct in ['tumor', 'oligodendrocyte']:
        c_mask = ctrl.obs['cell_type'].astype(str).values == ct
        zb = encode_batch(ctrl, c_mask, ct)
        z_start[ct] = zb
        if zb is None: print(f"  {ct}: none"); continue
        pred_v = np.log1p(np.clip(decode_mean(zb), 0, None))   # match log1p truth scale
        true_v = ctrl.X[c_mask].mean(0)
        r,_ = pearsonr(pred_v, true_v)
        print(f"  {ct:<16}: recon R = {r:.3f}  r2 = {r2(true_v, pred_v):.3f}  (n_enc={zb.shape[0]})")

    # ---- predictions: add Dz to each vehicle cell latent, decode batch, average ----
    results = {'tumor': [], 'oligodendrocyte': []}
    print("\nPer-(cell_type, drug) R:")
    print(f"  {'cell_type':<16}{'drug':<14}{'R':>8}{'r2':>8}{'n_test':>8}")
    for ct in ['tumor', 'oligodendrocyte']:
        zb = z_start[ct]
        if zb is None: continue
        for d in HELD_OUT:
            if d not in Dz: continue
            t_mask = ((test.obs['cell_type'].astype(str).values == ct) &
                      (test.obs['drug'].astype(str).values == d))
            if t_mask.sum() == 0: continue
            true_mean = test.X[t_mask].mean(0)
            z_pred = zb + Dz[d]                    # add direction to EACH cell [N,D]
            pred_mean = np.log1p(np.clip(decode_mean(z_pred), 0, None))   # match log1p truth
            r,_ = pearsonr(pred_mean, true_mean)
            results[ct].append(r)
            print(f"  {ct:<16}{d:<14}{r:>8.3f}{r2(true_mean,pred_mean):>8.3f}{int(t_mask.sum()):>8}")

    print("\n" + "="*70)
    print("Fig 3f reproduction — mean Pearson R per cell type (over held-out drugs):")
    for ct in ['tumor', 'oligodendrocyte']:
        if results[ct]:
            print(f"  {ct:<16}: R = {np.mean(results[ct]):.3f}  (paper ~0.91; n_drugs={len(results[ct])})")
    print("="*70); print(f"Finished: {datetime.now()}")

if __name__ == '__main__':
    main()