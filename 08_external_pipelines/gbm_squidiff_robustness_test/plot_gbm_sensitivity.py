"""
Plot the GBM gene-count sensitivity from gene_sensitivity_results.csv.

Run AFTER all array tasks have finished:
    python plot_gbm_sensitivity.py

Produces, under OUT_ROOT/figures/ :
  fig_sens_ABC_allgenes.png   A vs B vs C across N (all genes), per cell type
  fig_sens_ABC_degs.png       A vs B vs C across N (DEGs), per cell type
  fig_sens_BminusC.png        genuine prediction over baseline (B-C) vs N, DEGs
  fig_sens_diagnostics.png    recon R, Delta-z norm, sparsity vs N (why the curve bends)

These are diagnostic curves for choosing/justifying the gene regime; the figure a
reviewer cares about is fig_sens_BminusC (does genuine prediction beat baseline,
and over what range of N) together with fig_sens_diagnostics (the mechanism).
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUT_ROOT = '/scratch/ni2071/Diffusion/gbm_squidiff_robustness_test'
CSV = os.path.join(OUT_ROOT, 'gene_sensitivity_results.csv')
FIG_DIR = os.path.join(OUT_ROOT, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

TYPE_COLORS = {'tumor': '#d1495b', 'oligodendrocyte': '#30638e'}


def load():
    df = pd.read_csv(CSV).sort_values(['cell_type', 'n_hvg'])
    # flag any N where reconstruction was below floor (prediction unreliable there)
    if 'recon_ok' in df.columns:
        bad = df[df['recon_ok'] == 0]
        if len(bad):
            print("WARNING — reconstruction below floor at these (N, cell_type); "
                  "prediction there is a broken decode, not biology:")
            print(bad[['n_hvg', 'cell_type', 'recon_R_vehicle']].to_string(index=False))
    return df


def plot_ABC(df, gene_set):
    """gene_set in {'all','deg'} -> A/B/C Pearson R across N, one panel per cell type."""
    fig, axes = plt.subplots(1, len(df['cell_type'].unique()), figsize=(12, 5), sharey=True)
    if len(df['cell_type'].unique()) == 1:
        axes = [axes]
    for ax, ct in zip(axes, sorted(df['cell_type'].unique())):
        d = df[df['cell_type'] == ct]
        ax.plot(d['n_hvg'], d[f'A_R_{gene_set}'], 'o-', label='A: reconstruction', color='#9ecae1')
        ax.plot(d['n_hvg'], d[f'B_R_{gene_set}'], 's-', label='B: genuine prediction (Δz)', color='#2171b5')
        ax.plot(d['n_hvg'], d[f'C_R_{gene_set}'], '^--', label='C: control mean (no model)', color='gray')
        ax.set_title(f'{ct}'); ax.set_xlabel('number of HVGs (N)')
        ax.grid(alpha=0.3)
    axes[0].set_ylabel(f'Pearson R ({"all genes" if gene_set=="all" else "DEGs"})')
    axes[-1].legend(loc='lower right', fontsize=9)
    fig.suptitle(f'GBM gene-count sensitivity — A/B/C ({"all genes" if gene_set=="all" else "DEGs"})')
    fig.tight_layout()
    out = os.path.join(FIG_DIR, f'fig_sens_ABC_{"allgenes" if gene_set=="all" else "degs"}.png')
    fig.savefig(out, dpi=150, bbox_inches='tight'); plt.close(fig)
    print('wrote', out)


def plot_BminusC(df):
    """Genuine prediction over baseline (B - C) on DEGs, vs N — the headline curve."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for ct in sorted(df['cell_type'].unique()):
        d = df[df['cell_type'] == ct]
        ax.plot(d['n_hvg'], d['BminusC_R_deg'], 'o-', color=TYPE_COLORS.get(ct, None),
                label=f'{ct} (Pearson, DEGs)')
        ax.plot(d['n_hvg'], d['BminusC_r2_deg'], 's--', color=TYPE_COLORS.get(ct, None),
                alpha=0.6, label=f'{ct} (r², DEGs)')
    ax.axhline(0, color='k', lw=1)
    ax.set_xlabel('number of HVGs (N)'); ax.set_ylabel('B − C on DEGs  (model over baseline)')
    ax.set_title('Genuine prediction over baseline vs gene count\n(above 0 = the model beats "predict the control mean")')
    ax.grid(alpha=0.3); ax.legend(fontsize=9)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, 'fig_sens_BminusC.png')
    fig.savefig(out, dpi=150, bbox_inches='tight'); plt.close(fig)
    print('wrote', out)


def plot_diagnostics(df):
    """Recon R, Delta-z norm, sparsity vs N — the mechanism behind the curve."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    # recon R per type
    for ct in sorted(df['cell_type'].unique()):
        d = df[df['cell_type'] == ct]
        axes[0].plot(d['n_hvg'], d['recon_R_vehicle'], 'o-', color=TYPE_COLORS.get(ct, None), label=ct)
    axes[0].axhline(0.70, color='r', ls=':', label='recon floor')
    axes[0].set_title('Reconstruction R (vehicle sanity ceiling)'); axes[0].set_xlabel('N'); axes[0].legend(fontsize=8)
    # delta-z norm (one value per N; same across types) — take per-N first row
    dzn = df.groupby('n_hvg')['dz_norm'].first()
    axes[1].plot(dzn.index, dzn.values, 'o-', color='#6a51a3')
    axes[1].set_title('Δz norm (drug-signal strength)'); axes[1].set_xlabel('N')
    # sparsity
    fz = df.groupby('n_hvg')['frac_zero'].first()
    axes[2].plot(fz.index, fz.values, 'o-', color='#238b45')
    axes[2].set_title('Panel sparsity (fraction zeros)'); axes[2].set_xlabel('N')
    for a in axes: a.grid(alpha=0.3)
    fig.suptitle('GBM gene-count sensitivity — diagnostics (why the prediction curve bends)')
    fig.tight_layout()
    out = os.path.join(FIG_DIR, 'fig_sens_diagnostics.png')
    fig.savefig(out, dpi=150, bbox_inches='tight'); plt.close(fig)
    print('wrote', out)


if __name__ == '__main__':
    df = load()
    plot_ABC(df, 'all')
    plot_ABC(df, 'deg')
    plot_BminusC(df)
    plot_diagnostics(df)
    print('\nAll figures in', FIG_DIR)