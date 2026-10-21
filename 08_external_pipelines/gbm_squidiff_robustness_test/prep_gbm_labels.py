"""
GBM gene-count sensitivity — STEP 0 (run ONCE, before the array).

Freezes the cell-type labels and the gene ranking so the sweep changes ONLY the
number of genes the model sees, never the data partition. This isolates "model
robustness to gene count" from "clustering/labelling sensitivity to gene count".

Reproduces prep_gbm_paper.py EXACTLY up to cell-type assignment (same load_slice,
same QC, same normalize_total(1e4)+log1p, same markers, same Leiden), with ONE
deliberate change: clustering uses a FIXED gene count (CLUSTER_HVG), independent
of the swept N, so labels are constant across all N.

Writes:
  OUT_ROOT/gbm_labelled_full.h5ad   full post-QC log1p matrix (ALL genes) with
                                    obs['cell_type','drug','slice'] + counts layer
  OUT_ROOT/hvg_ranking.txt          genes ranked by dispersion on the FULL pool;
                                    per-N prep takes the first N (=> nested panels)

Run once:
    python prep_gbm_labels.py
"""
import os
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from scipy import sparse

# --- paths: RAW_DIR is where the 8 .cts.txt.gz live (your prep reads them here) ---
RAW_DIR  = '/scratch/ni2071/Diffusion/datasets_gbm_paper_200'   # EDIT if your .gz are elsewhere
OUT_ROOT = '/scratch/ni2071/Diffusion/gbm_squidiff_robustness_test'
CLUSTER_HVG = 2000          # FIXED clustering basis for labels (independent of the sweep)
RANDOM = 0

# verified slice -> condition map (identical to prep_gbm_paper.py)
SLICES = {
    'GSM4483743_PW030-701.cts.txt.gz': 'vehicle',
    'GSM4483744_PW030-702.cts.txt.gz': 'vehicle',
    'GSM4483745_PW030-703.cts.txt.gz': 'etoposide',
    'GSM4483746_PW030-704.cts.txt.gz': 'panobinostat',
    'GSM4483747_PW030-705.cts.txt.gz': 'RO4929097',
    'GSM4483748_PW030-706.cts.txt.gz': 'tazemetostat',
    'GSM4483749_PW030-707.cts.txt.gz': 'ispinesib',
    'GSM4483750_PW030-708.cts.txt.gz': 'ANA-12',
}
MARKERS = {
    'myeloid':         ['CD14', 'AIF1', 'TYROBP', 'CD163', 'PTPRC', 'C1QB', 'C1QA'],
    'oligodendrocyte': ['PLP1', 'MBP', 'MAG', 'SOX10', 'MOG', 'CLDN11'],
    'tumor':           ['GFAP', 'SOX2', 'EGFR', 'OLIG1', 'OLIG2', 'VIM', 'CHI3L1'],
}


def load_slice(fname, condition):
    """Identical to prep_gbm_paper.py: genes-as-rows count txt.gz -> cells x genes."""
    path = os.path.join(RAW_DIR, fname)
    df = pd.read_csv(path, sep='\t', compression='gzip')
    df = df.set_index('gene')
    df = df.drop(columns=['gid'])
    genes = df.index.astype(str).values
    barcodes = df.columns.values
    X = df.values.T.astype(np.float32)
    a = ad.AnnData(X=sparse.csr_matrix(X))
    a.var_names = genes
    a.var_names_make_unique()
    a.obs_names = [f"{condition}_{fname.split('_')[0]}_{b}" for b in barcodes]
    a.obs['drug'] = condition
    a.obs['slice'] = fname.split('_')[1].split('.')[0]
    return a


def main():
    os.makedirs(OUT_ROOT, exist_ok=True)
    print("Loading 8 PW030 slices ...")
    adatas = [load_slice(f, c) for f, c in SLICES.items()]
    adata = ad.concat(adatas, join='outer', index_unique=None)
    adata.X = sparse.csr_matrix(np.nan_to_num(adata.X.toarray()))
    print(f"  combined: {adata.n_obs} cells x {adata.n_vars} genes")

    # ---- QC, identical to prep_gbm_paper.py ----
    adata.var['mt']   = adata.var_names.str.upper().str.startswith('MT-')
    adata.var['ribo'] = adata.var_names.str.upper().str.match(r'^RP[SL]')
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], inplace=True, percent_top=None)
    n0 = adata.n_obs
    adata = adata[adata.obs['total_counts'] >= 1000]
    adata = adata[adata.obs['pct_counts_mt'] < 20]
    sc.pp.filter_cells(adata, max_genes=10000)
    sc.pp.filter_genes(adata, min_cells=3)
    print(f"  QC: {n0} -> {adata.n_obs} cells")
    adata = adata[:, ~(adata.var['mt'] | adata.var['ribo'])].copy()

    adata.layers['counts'] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # ---- cell-type assignment on a FIXED clustering basis (the freeze) ----
    print(f"Assigning cell types by Leiden + marker scoring (CLUSTER_HVG={CLUSTER_HVG}) ...")
    adata.raw = adata
    sc.pp.highly_variable_genes(adata, n_top_genes=CLUSTER_HVG, subset=False)
    ad_clust = adata[:, adata.var['highly_variable']].copy()
    sc.pp.scale(ad_clust, max_value=10)
    sc.tl.pca(ad_clust, n_comps=30, random_state=RANDOM)
    sc.pp.neighbors(ad_clust, n_neighbors=15, random_state=RANDOM)
    sc.tl.leiden(ad_clust, resolution=1.0, random_state=RANDOM)
    adata.obs['leiden'] = ad_clust.obs['leiden'].values

    for ct, genes in MARKERS.items():
        present = [g for g in genes if g in adata.raw.var_names]
        sc.tl.score_genes(adata, present, score_name=f'score_{ct}', use_raw=True)
    cluster_label = {}
    for cl in adata.obs['leiden'].cat.categories:
        m = adata.obs['leiden'] == cl
        means = {ct: adata.obs.loc[m, f'score_{ct}'].mean() for ct in MARKERS}
        cluster_label[cl] = max(means, key=means.get)
    adata.obs['cell_type'] = adata.obs['leiden'].map(cluster_label).astype(str)
    print("  cell-type counts:\n", adata.obs['cell_type'].value_counts())
    print("  cluster -> type:", cluster_label)

    # ---- deterministic HVG ranking on the FULL pool (=> nested panels per N) ----
    # Recompute dispersions on the full gene set; the ranking is N-independent, so
    # taking the first N gives nested top-N panels across the sweep.
    full = adata.copy()
    sc.pp.highly_variable_genes(full, n_top_genes=full.n_vars, subset=False)
    disp = full.var['dispersions_norm'].fillna(-np.inf)
    ranked = disp.sort_values(ascending=False).index.astype(str).tolist()
    with open(os.path.join(OUT_ROOT, 'hvg_ranking.txt'), 'w') as f:
        f.write('\n'.join(ranked) + '\n')
    print(f"  wrote hvg_ranking.txt ({len(ranked)} genes ranked)")

    # ---- freeze the labelled FULL-gene object (cell_name = cell_type for the loader) ----
    adata.obs['cell_name'] = adata.obs['cell_type'].astype(str)
    adata.obs['Group'] = adata.obs['drug'].astype(str)
    out = os.path.join(OUT_ROOT, 'gbm_labelled_full.h5ad')
    adata.write(out)
    print(f"\nFroze labels -> {out}")
    print("Labels and split membership are now CONSTANT across all N; only genes change.")


if __name__ == '__main__':
    np.random.seed(RANDOM)
    main()