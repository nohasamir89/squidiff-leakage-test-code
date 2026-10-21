"""
Preprocess GSE148842 (Zhao et al. GBM drug screen, patient PW030) to reproduce
the Squidiff drug-perturbation task (Fig. 3d-g) with YOUR pipeline.

Slice -> treatment mapping is VERIFIED from GEO SOFT metadata (not guessed):
  701 GSM4483743  vehicle (DMSO)
  702 GSM4483744  vehicle (DMSO)
  703 GSM4483745  2.5 uM etoposide
  704 GSM4483746  0.2 uM panobinostat
  705 GSM4483747  50 nM RO4929097
  706 GSM4483748  50 uM tazemetostat
  707 GSM4483749  1.8 nM ispinesib
  708 GSM4483750  40 nM ANA-12

Paper split (Methods task 3 / Fig 3e):
  TRAIN: myeloid cells (all 7 conditions) + tumor & oligo (vehicle + etoposide only)
  TEST : tumor & oligo on the 5 held-out drugs (panobinostat, RO4929097,
         tazemetostat, ispinesib, ANA-12)

Cell-type labels were NOT released by the Squidiff authors. We assign them by
Leiden clustering + canonical marker scoring (documented limitation: if our R
differs from the paper's 0.91, cell-type assignment is the most likely cause,
not the model).

File format: tab-delimited, genes as ROWS, first two cols = gid (Ensembl.version)
and gene (symbol); remaining cols = cell barcodes. ~60726 genes x ~14k cells/slice.

Output: /scratch/ni2071/Diffusion/datasets_gbm_paper/{gbm_train.h5ad, gbm_test.h5ad}
"""
import os, gzip
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from scipy import sparse

DATA_DIR = '/scratch/ni2071/Diffusion/datasets_gbm_paper_200'
N_HVG    = 200          # ablation: narrow head, matches paper iPSC task + Tahoe runs
RANDOM   = 0

# ---- verified slice -> condition map ----
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
    """Read one genes-as-rows count txt.gz into a cells x genes AnnData."""
    path = os.path.join(DATA_DIR, fname)
    # Two non-cell columns: 'gid' (Ensembl.version) and 'gene' (symbol), then barcodes.
    # Index by gene symbol and DROP gid explicitly (robust to column order).
    df = pd.read_csv(path, sep='\t', compression='gzip')
    df = df.set_index('gene')
    df = df.drop(columns=['gid'])            # df is now genes x cells
    genes = df.index.astype(str).values
    barcodes = df.columns.values
    X = df.values.T.astype(np.float32)        # cells x genes
    a = ad.AnnData(X=sparse.csr_matrix(X))
    a.var_names = genes
    a.var_names_make_unique()
    a.obs_names = [f"{condition}_{fname.split('_')[0]}_{b}" for b in barcodes]
    a.obs['drug'] = condition
    a.obs['slice'] = fname.split('_')[1].split('.')[0]
    return a

def main():
    print("Loading 8 PW030 slices ...")
    adatas = [load_slice(f, c) for f, c in SLICES.items()]
    adata = ad.concat(adatas, join='outer', index_unique=None)
    adata.X = sparse.csr_matrix(np.nan_to_num(adata.X.toarray()))
    print(f"  combined: {adata.n_obs} cells x {adata.n_vars} genes")

    # ---- QC per paper ----
    adata.var['mt']   = adata.var_names.str.upper().str.startswith('MT-')
    adata.var['ribo'] = adata.var_names.str.upper().str.match(r'^RP[SL]')
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], inplace=True, percent_top=None)
    n0 = adata.n_obs
    adata = adata[adata.obs['total_counts'] >= 1000]
    adata = adata[adata.obs['pct_counts_mt'] < 20]
    sc.pp.filter_cells(adata, max_genes=10000)        # drop likely multiplets
    sc.pp.filter_genes(adata, min_cells=3)
    print(f"  QC: {n0} -> {adata.n_obs} cells")
    # drop mito + ribo genes (paper excludes stress-associated)
    adata = adata[:, ~(adata.var['mt'] | adata.var['ribo'])].copy()

    adata.layers['counts'] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # ---- cell-type assignment by markers (documented limitation) ----
    print("Assigning cell types by Leiden + marker scoring ...")
    adata.raw = adata
    sc.pp.highly_variable_genes(adata, n_top_genes=N_HVG, subset=False)
    ad_clust = adata[:, adata.var['highly_variable']].copy()
    sc.pp.scale(ad_clust, max_value=10)
    sc.tl.pca(ad_clust, n_comps=30, random_state=RANDOM)
    sc.pp.neighbors(ad_clust, n_neighbors=15, random_state=RANDOM)
    sc.tl.leiden(ad_clust, resolution=1.0, random_state=RANDOM)
    adata.obs['leiden'] = ad_clust.obs['leiden'].values

    for ct, genes in MARKERS.items():
        present = [g for g in genes if g in adata.raw.var_names]
        sc.tl.score_genes(adata, present, score_name=f'score_{ct}', use_raw=True)
    score_cols = [f'score_{ct}' for ct in MARKERS]
    # label each leiden cluster by its highest mean marker score
    cluster_label = {}
    for cl in adata.obs['leiden'].cat.categories:
        m = adata.obs['leiden'] == cl
        means = {ct: adata.obs.loc[m, f'score_{ct}'].mean() for ct in MARKERS}
        cluster_label[cl] = max(means, key=means.get)
    adata.obs['cell_type'] = adata.obs['leiden'].map(cluster_label).astype(str)
    print("  cell-type counts:\n", adata.obs['cell_type'].value_counts())
    print("  cluster -> type:", cluster_label)

    # ---- subset to HVGs BEFORE the split (so train/test/control share genes) ----
    adata = adata[:, adata.var['highly_variable']].copy()
    print(f"  After HVG subset: {adata.n_vars} genes")

    # ---- build the paper's split ----
    held_out = ['panobinostat', 'RO4929097', 'tazemetostat', 'ispinesib', 'ANA-12']
    is_myeloid = adata.obs['cell_type'] == 'myeloid'
    is_to      = adata.obs['cell_type'].isin(['tumor', 'oligodendrocyte'])
    in_train_to = is_to & adata.obs['drug'].isin(['vehicle', 'etoposide'])
    train_mask  = is_myeloid | in_train_to
    test_mask   = is_to & adata.obs['drug'].isin(held_out)

    train = adata[train_mask].copy()
    test  = adata[test_mask].copy()
    print(f"\nTRAIN: {train.n_obs} cells")
    print(train.obs.groupby(['cell_type','drug']).size())
    print(f"\nTEST: {test.n_obs} cells")
    print(test.obs.groupby(['cell_type','drug']).size())

    os.makedirs(DATA_DIR, exist_ok=True)
    train.write(os.path.join(DATA_DIR, 'gbm_train.h5ad'))
    test.write(os.path.join(DATA_DIR, 'gbm_test.h5ad'))
    # also a control file (vehicle cells of tumor+oligo) for the Δz / encode-control eval
    ctrl = adata[is_to & (adata.obs['drug'] == 'vehicle')].copy()
    ctrl.write(os.path.join(DATA_DIR, 'gbm_control_to.h5ad'))
    print("\nWrote gbm_train.h5ad, gbm_test.h5ad, gbm_control_to.h5ad")

if __name__ == '__main__':
    main()