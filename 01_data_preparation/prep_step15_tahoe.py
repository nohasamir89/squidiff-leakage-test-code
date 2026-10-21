#!/usr/bin/env python3
"""Step 15a (PREREG_step15a_tahoe_data.md, commit fe08221): prepare the Tahoe-100M brain / gut plates for Squidiff in the
sci-Plex layout. Every rule below cites its pre-registration section in [brackets]; every derived number is printed and
written to tahoe_results/step15_data_record.json.

Stages (each writes nothing until its assertions pass):
  0  guards: pre-registration committed, inputs present, libraries importable, output directory empty
  1  inventory: per-plate obs, gene lists; gene universe = genes present in all plates                       [4.3]
  2  per-plate 300,000-cell samples: detection, dominant-transcript test, seurat_v3 normalised variance      [4.5]
  3  panel: eligible genes ranked per plate, pooled by median rank, top 200                                  [4.5]
  4  drug table, structures (RDKit canonical SMILES without stereochemistry), excluded drug                  [4.2]
  5  three disjoint compound-held-out splits of 75 structures each                                           [4.6]
  6  cell draws (150 train / 500 test per condition), vehicle pairing within line and plate, normalisation,
     twelve h5ad files in the sci-Plex layout                                                                 [4.4, 4.7, 4.8, 5]
  7  record with SHA-256 of every output                                                                     [6]
The full-gene files are read with h5py in contiguous row blocks of the counts layer only (their stored X is never used;
opening them with anndata in backed mode would load the whole counts layer into memory).
`--dry` runs stages 0-1 and the schema read, then stops without writing.
"""
import os, sys, re, json, time, hashlib, subprocess
import numpy as np, pandas as pd, h5py
import anndata as ad, scipy.sparse as sp
from scipy.stats import rankdata

# ------------------------------------------------------------------ constants; [n] = section of PREREG_step15a_tahoe_data.md
DIR = '/scratch/ni2071/Tahoa_2/preprocessing_4/cpa_hvg_plates'
PLATES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14]                            # [2] plate 12 excluded before this step
FULL = DIR + '/cpa_ready_plate%d_full_genes.h5ad'
DRUG_TABLE = '/scratch/ni2071/Tahoa_2/preprocessing/tahoe_drug_metadata.csv'
VERTEPORFIN = dict(drug='Verteporfin', pubchem_cid=5362420, formula='C41H42N4O8',  # [2] PubChem ConnectivitySMILES, 2026-09-25
                   canonical_smiles='CC1=C(C2=CC3=NC(=CC4=C(C(=C(N4)C=C5C6(C(C(=CC=C6C(=N5)C=C1N2)C(=O)OC)C(=O)OC)C)C)CCC(=O)OC)C(=C3C)CCC(=O)O)C=C')
EXCLUDED_DRUGS = {'Sacubitril/Valsartan'}                                         # [4.2] two-molecule product, no single structure
SCIPLEX = '/scratch/ni2071/Diffusion/sciplex/datasets_corrected'                  # [3] schema only
OUT = '/scratch/ni2071/Diffusion/tahoe/datasets_step15'                           # [5]
REC, PANEL_TXT = 'tahoe_results/step15_data_record.json', 'tahoe_results/step15_gene_panel.txt'
PREREG, PREREG_SECTIONS = 'PREREG_step15a_tahoe_data.md', 8
CONTROL = 'DMSO_TF'                                                               # [4.1]
SAMPLE_CELLS, CHUNK = 300_000, 100_000                                            # [4.5] sample per plate; rows per block read
N_PANEL, DETECT_MIN, DOMINANT_SHARE, DOMINANT_FRAC = 200, 0.01, 0.05, 0.01        # [4.5]
EXCLUDED_PREFIXES = ('MT-', 'RPS', 'RPL')                                         # [4.5]
N_HELD, SPLIT_SEED, N_SPLITS = 75, 15, 3                                          # [4.6]
CAP_TRAIN, CAP_TEST, DRAW_SEED, MIN_TEST_CELLS = 150, 500, 0, 20                  # [4.7], [7]
TARGET, UM_TO_NM = 1e4, 1000.0                                                    # [4.4], [4.8]
DRY = '--dry' in sys.argv
T0 = time.time()
def log(m): print('[%5d s] %s' % (time.time() - T0, m)); sys.stdout.flush()
def die(m): sys.exit('FATAL: ' + m)
def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for c in iter(lambda: f.read(1 << 22), b''): h.update(c)
    return h.hexdigest()
try:
    from anndata.io import read_elem                                              # anndata >= 0.11
except Exception:
    try:
        from anndata.experimental import read_elem                                # anndata 0.8 - 0.10
    except Exception: die('anndata %s has no read_elem; 0.8 or newer is needed' % ad.__version__)

def parse_conc(s):
    """drugname_drugconc looks like "[('Gemcitabine', 0.05, 'uM')]" -> (name, conc, unit)."""
    m = re.search(r"\(\s*['\"](.+?)['\"]\s*,\s*([0-9.eE+-]+)\s*,\s*['\"](.*?)['\"]\s*\)", str(s))
    return (m.group(1), float(m.group(2)), m.group(3)) if m else (str(s), float('nan'), '')

class Counts:
    """The counts layer of one full-gene file, read block-wise with h5py. Supports the anndata CSR encoding and dense."""
    def __init__(self, path):
        self.f = h5py.File(path, 'r'); g = self.f['layers/counts']; self.g = g
        if isinstance(g, h5py.Group):
            enc = g.attrs.get('encoding-type', g.attrs.get('h5sparse_format', ''))
            enc = enc.decode() if isinstance(enc, bytes) else str(enc)
            if 'csr' not in enc: die('counts layer of %s is stored as %r; only CSR (or dense) can be read by row blocks' % (path, enc))
            self.shape = tuple(int(x) for x in g.attrs.get('shape', g.attrs.get('h5sparse_shape'))); self.sparse = True
        else:
            self.shape = tuple(int(x) for x in g.shape); self.sparse = False
    def rows(self, s0, s1):
        """Rows s0..s1-1 as a CSR matrix: one contiguous read of indptr, data and indices."""
        if not self.sparse: return sp.csr_matrix(np.asarray(self.g[s0:s1]))
        ip = self.g['indptr'][s0:s1 + 1].astype(np.int64); a0, a1 = int(ip[0]), int(ip[-1])
        return sp.csr_matrix((self.g['data'][a0:a1], self.g['indices'][a0:a1], ip - a0), shape=(s1 - s0, self.shape[1]))
    def select(self, idx, cols):
        """Rows idx (sorted, unique) restricted to columns cols, collected in one sequential pass over the file."""
        parts = []
        for s0 in range(0, self.shape[0], CHUNK):
            s1 = min(s0 + CHUNK, self.shape[0]); sel = idx[(idx >= s0) & (idx < s1)]
            if len(sel): parts.append(self.rows(s0, s1)[sel - s0][:, cols])
        M = sp.vstack(parts).tocsr() if parts else sp.csr_matrix((0, len(cols)), dtype=np.float32)
        if M.shape[0] != len(idx): die('selected %d rows, expected %d' % (M.shape[0], len(idx)))
        return M
    def close(self): self.f.close()

# ================================================================== stage 0: guards
if subprocess.run(['git', 'diff', '--quiet', '--', PREREG]).returncode != 0: die('%s has uncommitted changes' % PREREG)
if not os.path.isfile(PREREG): die('missing %s (run from the repository root)' % PREREG)
n_sec = sum(1 for l in open(PREREG) if l.startswith('## '))
if n_sec != PREREG_SECTIONS: die('%s has %d sections, expected %d' % (PREREG, n_sec, PREREG_SECTIONS))
COMMIT = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
for k in PLATES:
    if not os.path.isfile(FULL % k): die('missing plate file %s' % (FULL % k))
SCI = {p: '%s/sci_plex_%s_drug_split_1%s.h5ad' % (SCIPLEX, part, suf) for p, (part, suf) in
       {'train': ('train', ''), 'train_control': ('train', '_control'), 'test': ('test', ''), 'test_control': ('test', '_control')}.items()}
for p in [DRUG_TABLE] + list(SCI.values()):
    if not os.path.isfile(p): die('missing input %s' % p)
try:
    import skmisc, scanpy as sc                                                     # seurat_v3 needs scikit-misc (loess)
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors
except ImportError as e: die('missing library: %s (scanpy, scikit-misc and rdkit are required)' % e)
if os.path.isdir(OUT) and os.listdir(OUT): die('output directory %s is not empty' % OUT)
if os.path.exists(REC): die('record %s exists' % REC)
os.makedirs('tahoe_results', exist_ok=True)
log('guards passed | prereg %d sections committed | commit %s | dry=%s' % (n_sec, COMMIT[:7], DRY))

# ------------------------------------------------------------------ the sci-Plex schema the outputs must reproduce   [4.8]
schema = {}
for p, path in SCI.items():
    with h5py.File(path, 'r') as f:
        obs = read_elem(f['obs']); X_dense = isinstance(f['X'], h5py.Dataset)
        schema[p] = dict(columns=list(obs.columns), dtypes={c: str(obs[c].dtype) for c in obs.columns}, n=len(obs), X_dense=X_dense,
                         n_genes=int(f['X'].shape[1] if X_dense else f['X'].attrs['shape'][1]), X_dtype=str(f['X'].dtype if X_dense else f['X/data'].dtype))
    log('sci-Plex %-14s rows %7d genes %d X %s %s | %s' % (p, schema[p]['n'], schema[p]['n_genes'], 'dense' if X_dense else 'sparse', schema[p]['X_dtype'], schema[p]['dtypes']))
if schema['train']['n'] != schema['train_control']['n'] or schema['test']['n'] != schema['test_control']['n']: die('sci-Plex control files are not row-aligned')
SCHEMA_COLS = schema['train']['columns']
for p in SCI:
    if schema[p]['columns'] != SCHEMA_COLS and sorted(schema[p]['columns']) != sorted(SCHEMA_COLS): die('sci-Plex files disagree on obs columns')
for c in ('Group', 'dose', 'SMILES', 'cell_name'):
    if c not in SCHEMA_COLS: die('sci-Plex schema lacks column %s' % c)

# ================================================================== stage 1: inventory and gene universe   [4.1, 4.3]
plate = {}; universe = None
for k in PLATES:
    with h5py.File(FULL % k, 'r') as f:
        obs = read_elem(f['obs']); genes = np.array(read_elem(f['var']).index.astype(str))
    if len(set(genes)) != len(genes): die('plate %d: duplicate gene names' % k)
    for c in ('cell_name', 'drug', 'drugname_drugconc', 'sample'):
        if c not in obs.columns: die('plate %d: obs lacks %s' % (k, c))
    o = pd.DataFrame({c: pd.Categorical(obs[c].astype(str).values) for c in ('cell_name', 'drug', 'drugname_drugconc', 'sample')}, index=obs.index.astype(str))
    parsed = {v: parse_conc(v) for v in o['drugname_drugconc'].cat.categories}
    o['drug'] = pd.Categorical(o['drug'].astype(str).str.strip())                 # [4.2] whitespace-stripped names
    o['conc_um'] = o['drugname_drugconc'].map(lambda v: parsed[v][1]).astype(float); o['unit'] = o['drugname_drugconc'].map(lambda v: parsed[v][2]).astype(str)
    treated = o['drug'] != CONTROL
    if not (o.loc[treated, 'unit'] == 'uM').all(): die('plate %d: treated concentrations not all in uM: %s' % (k, sorted(set(o.loc[treated, 'unit']))))
    if o.loc[treated, 'conc_um'].isna().any(): die('plate %d: unparsed concentration' % k)
    o['dose'] = np.where(treated, np.round(o['conc_um'].fillna(0).values * UM_TO_NM, 6), 0.0)       # [4.8] nM; vehicle 0
    per_drug = o.loc[treated].groupby('drug', observed=True)['dose'].nunique()          # a drug may sit at more than one dose on a plate
    multi = sorted(per_drug[per_drug > 1].index)                                           # (plate 8: Adagrasib); each dose is its own condition [4.1]
    cnt = Counts(FULL % k); shape = cnt.shape; cnt.close()
    if shape != (len(o), len(genes)): die('plate %d: counts layer shape %s differs from obs x var %s' % (k, shape, (len(o), len(genes))))
    plate[k] = dict(obs=o, genes=genes, n=len(o), multi_dose_drugs=multi)
    universe = genes if universe is None else universe[np.isin(universe, genes)]
    log('plate %2d: %8d cells, %d genes, %d treated drugs (%d at more than one dose), %d vehicle cells, lines %d' % (k, len(o), len(genes), per_drug.shape[0], len(multi), int((~treated).sum()), o['cell_name'].nunique()))
G = len(universe); upos = {g: i for i, g in enumerate(universe)}
for k in PLATES:                                                                   # universe -> column positions in each plate's file
    gpos = {g: i for i, g in enumerate(plate[k]['genes'])}
    plate[k]['cols'] = np.array([gpos[g] for g in universe])
log('gene universe: %d genes present in all %d plates (plate 1 order)' % (G, len(PLATES)))
if DRY: log('dry run: stopping before any sample is read'); sys.exit(0)

# ================================================================== stage 2: per-plate samples and gene statistics   [4.5]
stats = {}
for k in PLATES:
    n = plate[k]['n']; idx = np.sort(np.random.RandomState(k).choice(n, min(SAMPLE_CELLS, n), replace=False))
    cnt = Counts(FULL % k); C = cnt.select(idx, plate[k]['cols']); cnt.close()
    C.data = C.data.astype(np.float32); ns = C.shape[0]
    if not np.allclose(C.data, np.round(C.data)): die('plate %d: counts are not integral' % k)
    tot = np.asarray(C.sum(1)).ravel()
    if (tot <= 0).any(): die('plate %d: a sampled cell has zero counts over the universe' % k)
    det = np.asarray((C > 0).sum(0)).ravel() / ns                                                     # detection rate per gene
    share = sp.diags(1.0 / tot) @ C                                                                   # each cell's counts as shares
    dom = np.asarray((share > DOMINANT_SHARE).sum(0)).ravel() / ns                                    # fraction of cells where the gene exceeds 5%
    s = ad.AnnData(X=C.copy(), var=pd.DataFrame(index=universe)); s.layers['counts'] = C
    sc.pp.highly_variable_genes(s, flavor='seurat_v3', layer='counts', n_top_genes=G)                # normalised variance for every gene
    vn = s.var['variances_norm'].values.astype(float); del s, share
    stats[k] = dict(n_sample=int(ns), det=det, dom=dom, vn=vn, total_q=[float(x) for x in np.percentile(tot, [5, 50, 95])])
    log('plate %2d sample %d cells | total counts q05/50/95 %s | genes detected >= %g%%: %d | dominant (> %g%% in >= %g%% of cells): %s'
        % (k, ns, stats[k]['total_q'], DETECT_MIN * 100, int((det >= DETECT_MIN).sum()), DOMINANT_SHARE * 100, DOMINANT_FRAC * 100, list(universe[dom >= DOMINANT_FRAC])))

# ================================================================== stage 3: panel   [4.5]
prefix_excl = np.array([any(g.startswith(p) for p in EXCLUDED_PREFIXES) for g in universe])
det_all = np.all([stats[k]['det'] >= DETECT_MIN for k in PLATES], axis=0)
dom_any = np.any([stats[k]['dom'] >= DOMINANT_FRAC for k in PLATES], axis=0)
elig = det_all & ~prefix_excl & ~dom_any
if elig.sum() < N_PANEL: die('only %d eligible genes, fewer than the panel size %d' % (elig.sum(), N_PANEL))
ranks = np.vstack([rankdata(-stats[k]['vn'][elig], method='min') for k in PLATES])                # rank 1 = most variable, per plate
pooled = np.median(ranks, axis=0); mean_rank = ranks.mean(axis=0); egenes = universe[elig]
order = sorted(range(int(elig.sum())), key=lambda i: (pooled[i], mean_rank[i], egenes[i]))          # ties: mean rank, then name
panel = [egenes[i] for i in order[:N_PANEL]]; panel_pos = np.array([upos[g] for g in panel])
panel_rows = [dict(gene=egenes[i], pooled_rank=float(pooled[i]), mean_rank=float(mean_rank[i]),
                   detection_min_over_plates=float(min(stats[k]['det'][upos[egenes[i]]] for k in PLATES))) for i in order[:N_PANEL]]
with open(PANEL_TXT, 'w') as fh:
    fh.write('# Step 15a gene panel: %d genes, pooled median seurat_v3 rank over %d plates (PREREG 4.5); prep commit %s\n# gene\tpooled_rank\tmean_rank\tmin_detection\n' % (N_PANEL, len(PLATES), COMMIT[:7]))
    for r in panel_rows: fh.write('%s\t%g\t%.2f\t%.4f\n' % (r['gene'], r['pooled_rank'], r['mean_rank'], r['detection_min_over_plates']))
log('eligible genes %d of %d (detected on every plate %d; prefix-excluded %d; dominant on some plate %d: %s)' % (elig.sum(), G, det_all.sum(), prefix_excl.sum(), dom_any.sum(), list(universe[dom_any])))
log('panel: %d genes | pooled rank of the last gene %g | detection (min over plates) q50 %.3f min %.3f | first 10: %s'
    % (len(panel), panel_rows[-1]['pooled_rank'], float(np.median([r['detection_min_over_plates'] for r in panel_rows])), min(r['detection_min_over_plates'] for r in panel_rows), panel[:10]))

# ================================================================== stage 4: drug table and structures   [4.2]
tab = pd.read_csv(DRUG_TABLE); tab['key'] = tab['drug'].astype(str).str.strip()
smiles = {r.key: r.canonical_smiles.strip() for r in tab.itertuples() if isinstance(r.canonical_smiles, str) and r.canonical_smiles.strip()}
cid = {r.key: (int(r.pubchem_cid) if pd.notna(r.pubchem_cid) else None) for r in tab.itertuples()}
mv = Chem.MolFromSmiles(VERTEPORFIN['canonical_smiles'])
if mv is None or rdMolDescriptors.CalcMolFormula(mv) != VERTEPORFIN['formula']: die('the recorded Verteporfin SMILES does not give %s' % VERTEPORFIN['formula'])
if 'Verteporfin' in smiles: die('the drug table now carries a Verteporfin SMILES; the pre-registration assumed none (section 2)')
smiles['Verteporfin'] = VERTEPORFIN['canonical_smiles']; cid['Verteporfin'] = VERTEPORFIN['pubchem_cid']
drugs_all = sorted(set().union(*[set(plate[k]['obs'].loc[plate[k]['obs']['drug'] != CONTROL, 'drug']) for k in PLATES]))
excluded_present = sorted(set(drugs_all) & EXCLUDED_DRUGS); usable = [d for d in drugs_all if d not in EXCLUDED_DRUGS]
missing = [d for d in usable if d not in smiles]
if missing: die('drugs without a structure: %s' % missing)
canon = {}
for d in usable:
    m = Chem.MolFromSmiles(smiles[d])
    if m is None: die('RDKit cannot parse the SMILES of %s: %s' % (d, smiles[d]))
    Chem.RemoveStereochemistry(m); canon[d] = Chem.MolToSmiles(m, isomericSmiles=False)
structs = sorted(set(canon.values())); sid = {s: 'S%04d' % i for i, s in enumerate(structs)}
groups = {}
for d in usable: groups.setdefault(sid[canon[d]], []).append(d)
shared = {s: v for s, v in groups.items() if len(v) > 1}
log('drugs on the plates %d | excluded %s | usable %d | structures %d | names sharing a structure: %s' % (len(drugs_all), excluded_present, len(usable), len(structs), shared))

# ================================================================== stage 5: splits   [4.6]
if N_SPLITS * N_HELD > len(structs): die('%d structures cannot give %d disjoint held-out sets of %d' % (len(structs), N_SPLITS, N_HELD))
perm = np.random.RandomState(SPLIT_SEED).permutation(len(structs))
held = {s: set(sid[structs[i]] for i in perm[s * N_HELD:(s + 1) * N_HELD]) for s in range(N_SPLITS)}
if any(held[a] & held[b] for a in held for b in held if a < b): die('held-out sets overlap')
held_drugs = {s: sorted(d for d in usable if sid[canon[d]] in held[s]) for s in range(N_SPLITS)}
for s in range(N_SPLITS): log('drug_split_%d: %d held-out structures = %d drugs, e.g. %s' % (s, len(held[s]), len(held_drugs[s]), held_drugs[s][:5]))

# ================================================================== stage 6: cell draws, pairing, normalisation, files   [4.4, 4.7, 4.8, 5]
rng = np.random.RandomState(DRAW_SEED)
name_map = {}
def safe(name):                                                                   # [4.8] '_' would break the Group convention
    if '_' in name: name_map[name] = name.replace('_', '-')
    return name_map.get(name, name)
# Group = <cell_name>_<drug>_<dose/10000>_p<plate>: 130 (drug, dose) pairs recur on several plates (plate 14 replicates plate 6;
# plate 13 re-runs drugs of plates 1, 5, 7, 8, 9; Adagrasib 50 nM on 12 plates), so the sci-Plex label alone would merge conditions.
PART_FILE = {'train': 'tahoe_train_drug_split_%d.h5ad', 'train_control': 'tahoe_train_drug_split_%d_control.h5ad',
             'test': 'tahoe_test_drug_split_%d.h5ad', 'test_control': 'tahoe_test_drug_split_%d_control.h5ad'}
buckets = {s: {p: [] for p in PART_FILE} for s in range(N_SPLITS)}
cond_records, skipped = [], []                                                    # skipped: conditions whose line has no vehicle cells on the plate
for k in PLATES:
    o = plate[k]['obs']; o['pos'] = np.arange(len(o)); treated = (o['drug'] != CONTROL).values
    line_of = o['cell_name'].astype(str).values
    vpos = {line: o['pos'].values[(~treated) & (line_of == line)] for line in np.unique(line_of)}
    t = o[treated & o['drug'].isin(usable).values]
    draws = []
    for (line, drug, dose), g in t.groupby(['cell_name', 'drug', 'dose'], sort=True, observed=True):
        rows = g['pos'].values
        if len(vpos.get(line, [])) == 0:                                              # [4.1] no same-plate vehicle: cannot be paired; excluded and recorded
            skipped.append(dict(plate=k, line=line, drug=drug, dose=float(dose), n_cells=int(len(rows)), reason='no vehicle cells of this line on this plate')); continue
        take = rng.choice(rows, min(CAP_TEST, len(rows)), replace=False)          # the first CAP_TRAIN of the draw are the training cells
        pairs = rng.choice(vpos[line], len(take), replace=True)                   # row i of the control file pairs with treated row i
        held_in = [s for s in range(N_SPLITS) if sid[canon[drug]] in held[s]]
        if held_in and len(take) < MIN_TEST_CELLS: die('plate %d: held-out condition %s|%s|%g has %d cells' % (k, line, drug, dose, len(take)))
        draws.append(dict(line=line, drug=drug, dose=float(dose), take=take, pairs=pairs, held=held_in[0] if held_in else None))
        cond_records.append(dict(plate=k, line=line, drug=drug, dose=float(dose), n_cells=int(len(rows)), n_test=int(len(take)),
                                 n_train=int(min(CAP_TRAIN, len(take))), vehicle_pool=int(len(vpos[line])), held_split=held_in[0] if held_in else None))
    need = np.unique(np.concatenate([d['take'] for d in draws] + [d['pairs'] for d in draws]))
    cnt = Counts(FULL % k); C = cnt.select(need, plate[k]['cols']); cnt.close()
    tot = np.asarray(C.sum(1)).ravel().astype(np.float64)
    if (tot <= 0).any(): die('plate %d: a selected cell has zero counts over the universe' % k)
    Xp = np.log1p(C[:, panel_pos].toarray().astype(np.float64) / tot[:, None] * TARGET).astype(np.float32)   # [4.4]
    del C
    names, wells = o.index.values, o['sample'].astype(str).values
    for d in draws:
        grp = '%s_%s_%s_p%d' % (safe(d['line']), safe(d['drug']), str(d['dose'] / 10000.0), k); smi = smiles[d['drug']]   # [4.8] sci-Plex form + plate: a condition sits on one plate
        for part in ('test', 'train'):
            if part == 'test' and d['held'] is None: continue
            m = len(d['take']) if part == 'test' else min(CAP_TRAIN, len(d['take']))
            take, pairs = d['take'][:m], d['pairs'][:m]
            mk = lambda rows: pd.DataFrame(dict(Group=grp, dose=d['dose'], SMILES=smi, cell_name=d['line'], plate=k, sample=wells[rows]), index=['p%d_%s' % (k, b) for b in names[rows]])
            xt, xc = Xp[np.searchsorted(need, take)], Xp[np.searchsorted(need, pairs)]; ot, oc = mk(take), mk(pairs)
            for s in ([d['held']] if part == 'test' else [s for s in range(N_SPLITS) if s != d['held']]):
                buckets[s][part].append((xt, ot)); buckets[s][part + '_control'].append((xc, oc))
    log('plate %2d: %d conditions, %d distinct cells read (%d treated draws + vehicle pairs) | skipped so far %d' % (k, len(draws), len(need), sum(len(d['take']) for d in draws), len(skipped)))
    del Xp, o['pos']

os.makedirs(OUT, exist_ok=True)
files = {}
for s in range(N_SPLITS):
    for part, fname in PART_FILE.items():
        items = buckets[s][part]
        if not items: die('drug_split_%d: no rows for %s' % (s, part))
        X = np.concatenate([x for x, _ in items]); obs = pd.concat([ob for _, ob in items])
        obs = obs[[c for c in SCHEMA_COLS] + ['plate', 'sample']]
        for c in ('Group', 'SMILES', 'cell_name', 'sample'): obs[c] = obs[c].astype('category')
        obs['dose'] = obs['dose'].astype('float64'); obs['plate'] = obs['plate'].astype('int64')
        a = ad.AnnData(X=X if schema['train']['X_dense'] else sp.csr_matrix(X), obs=obs, var=pd.DataFrame(index=pd.Index(panel, name=None)))
        a.obs_names_make_unique()
        for c in SCHEMA_COLS:
            if str(a.obs[c].dtype) != schema['train']['dtypes'][c]: die('%s: column %s dtype %s, sci-Plex has %s' % (fname % s, c, a.obs[c].dtype, schema['train']['dtypes'][c]))
        if a.X.dtype != np.float32 or a.shape[1] != N_PANEL: die('%s: X %s %s' % (fname % s, a.X.dtype, a.shape))
        path = os.path.join(OUT, fname % s); a.write_h5ad(path)
        files[(s, part)] = dict(path=path, rows=int(a.shape[0]), n_conditions=int(a.obs['Group'].nunique()), sha256=sha256(path))
        del a, X, obs
    for part in ('train', 'test'):
        t, c = files[(s, part)], files[(s, part + '_control')]
        if t['rows'] != c['rows'] or t['n_conditions'] != c['n_conditions']: die('drug_split_%d: %s and its control file are not aligned' % (s, part))
    buckets[s] = None
    log('drug_split_%d: train %d rows / %d conditions | test %d rows / %d conditions | control files aligned' % (s, files[(s, 'train')]['rows'], files[(s, 'train')]['n_conditions'], files[(s, 'test')]['rows'], files[(s, 'test')]['n_conditions']))

# ================================================================== stage 7: record   [6]
cr = pd.DataFrame(cond_records)
record = dict(prereg=PREREG, prereg_sections=n_sec, commit=COMMIT, date=time.strftime('%Y-%m-%d'),
              constants=dict(plates=PLATES, sample_cells=SAMPLE_CELLS, n_panel=N_PANEL, detect_min=DETECT_MIN, dominant_share=DOMINANT_SHARE, dominant_frac=DOMINANT_FRAC,
                             excluded_prefixes=EXCLUDED_PREFIXES, n_held=N_HELD, split_seed=SPLIT_SEED, cap_train=CAP_TRAIN, cap_test=CAP_TEST, draw_seed=DRAW_SEED, target=TARGET, control=CONTROL),
              inputs=dict(plates={k: dict(path=FULL % k, n_cells=plate[k]['n'], n_genes=int(len(plate[k]['genes'])), drugs_at_more_than_one_dose=plate[k]['multi_dose_drugs']) for k in PLATES},
                          drug_table=dict(path=DRUG_TABLE, sha256=sha256(DRUG_TABLE)), sciplex_schema=schema),
              universe=dict(n=int(G), first=list(universe[:5])),
              samples={k: dict(n_sample=stats[k]['n_sample'], total_counts_q05_50_95=stats[k]['total_q'], n_detected_ge_min=int((stats[k]['det'] >= DETECT_MIN).sum()),
                               dominant=list(universe[stats[k]['dom'] >= DOMINANT_FRAC])) for k in PLATES},
              eligibility=dict(n_eligible=int(elig.sum()), n_detected_on_every_plate=int(det_all.sum()), n_prefix_excluded=int(prefix_excl.sum()),
                               prefix_excluded=list(universe[prefix_excl]), dominant_on_some_plate=list(universe[dom_any])),
              panel=panel_rows,
              drugs=dict(n_names_on_plates=len(drugs_all), excluded=excluded_present, n_usable=len(usable), n_structures=len(structs), shared_structures=shared,
                         verteporfin=VERTEPORFIN, structure={d: sid[canon[d]] for d in usable}, canonical_smiles={d: canon[d] for d in usable}, table_smiles={d: smiles[d] for d in usable}, pubchem_cid={d: cid.get(d) for d in usable}),
              splits={s: dict(held_structures=sorted(held[s]), held_drugs=held_drugs[s],
                              n_test_conditions=int((cr['held_split'] == s).sum()), n_train_conditions=int((cr['held_split'] != s).sum()),
                              test_cells_per_condition_q05_50_95=[float(x) for x in np.percentile(cr.loc[cr['held_split'] == s, 'n_test'], [5, 50, 95])],
                              files={p: files[(s, p)] for p in PART_FILE}) for s in range(N_SPLITS)},
              name_map=name_map, conditions=cond_records,
              skipped_conditions=dict(n=len(skipped), n_cells=int(sum(x['n_cells'] for x in skipped)), by_plate_line={'%d|%s' % (x['plate'], x['line']): 0 for x in skipped}, items=skipped),
              seconds=int(time.time() - T0))
for x in skipped: record['skipped_conditions']['by_plate_line']['%d|%s' % (x['plate'], x['line'])] += 1
json.dump(record, open(REC, 'w'), indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))
log('conditions %d | skipped (no same-plate vehicle) %d: %s | cells per condition q05/50/95 %s | wrote %s and %s (%d s)' % (len(cr), len(skipped), record['skipped_conditions']['by_plate_line'], [float(x) for x in np.percentile(cr['n_cells'], [5, 50, 95])], REC, PANEL_TXT, int(time.time() - T0)))
