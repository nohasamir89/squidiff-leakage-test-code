"""
GBM gene-count sensitivity — per-N prep (fast; runs inside each array task).

Loads the FROZEN labelled object + the FROZEN HVG ranking from prep_gbm_labels.py,
takes the first N genes (=> nested panel, top-N subset of top-M for N<M), and writes
the train/test/control split using the FROZEN labels and the paper's split logic.

Because labels and the ranking are frozen, the train/test/control CELL membership is
IDENTICAL across all N — only the gene set changes. That is the whole point of the
freeze: the curve reflects the model's gene-count sensitivity, nothing else.

    python prep_gbm_sensitivity.py --n_hvg 300
"""
import argparse, os
import numpy as np
import scanpy as sc

OUT_ROOT = '/scratch/ni2071/Diffusion/gbm_squidiff_robustness_test'
HELD_OUT = ['panobinostat', 'RO4929097', 'tazemetostat', 'ispinesib', 'ANA-12']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_hvg', type=int, required=True)
    args = ap.parse_args()
    N = args.n_hvg

    labelled = os.path.join(OUT_ROOT, 'gbm_labelled_full.h5ad')
    ranking  = os.path.join(OUT_ROOT, 'hvg_ranking.txt')
    if not (os.path.exists(labelled) and os.path.exists(ranking)):
        raise FileNotFoundError(
            "Frozen labels/ranking not found. Run prep_gbm_labels.py ONCE first:\n"
            f"  expected {labelled}\n  expected {ranking}")

    adata = sc.read_h5ad(labelled)
    ranked = [g.strip() for g in open(ranking) if g.strip()]
    # first N ranked genes that are actually present (defensive); preserves nesting
    panel = [g for g in ranked if g in set(adata.var_names)][:N]
    adata = adata[:, panel].copy()

    out_dir = os.path.join(OUT_ROOT, f'n{N}')
    os.makedirs(out_dir, exist_ok=True)

    # sparsity diagnostic for this panel
    X = adata.X.toarray() if hasattr(adata.X, 'toarray') else np.asarray(adata.X)
    frac_zero = float((X == 0).mean())
    with open(os.path.join(out_dir, 'frac_zero.txt'), 'w') as f:
        f.write(f"{N}\t{frac_zero:.4f}\n")

    # ---- the paper's split, on FROZEN labels (identical membership across N) ----
    is_myeloid = adata.obs['cell_type'] == 'myeloid'
    is_to      = adata.obs['cell_type'].isin(['tumor', 'oligodendrocyte'])
    in_train_to = is_to & adata.obs['drug'].isin(['vehicle', 'etoposide'])
    train_mask  = is_myeloid | in_train_to
    test_mask   = is_to & adata.obs['drug'].isin(HELD_OUT)
    ctrl_mask   = is_to & (adata.obs['drug'] == 'vehicle')

    adata[train_mask].copy().write(os.path.join(out_dir, 'gbm_train.h5ad'))
    adata[test_mask].copy().write(os.path.join(out_dir, 'gbm_test.h5ad'))
    adata[ctrl_mask].copy().write(os.path.join(out_dir, 'gbm_control_to.h5ad'))

    print(f"[N={N}] genes={adata.n_vars} frac_zero={frac_zero:.3f}  "
          f"train={int(train_mask.sum())} test={int(test_mask.sum())} "
          f"control={int(ctrl_mask.sum())}  -> {out_dir}")


if __name__ == '__main__':
    main()