import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import scanpy as sc
import os
import pickle
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

def Drug_dose_encoder(drug_SMILES_list: list, dose_list: list, num_Bits=1024, comb_num=1, drug_encoding='fingerprint'):
    """
    adopted from PRnet @Author: Xiaoning Qi.
    Encode SMILES of drug to rFCFP fingerprint
    """
    drug_len = len(drug_SMILES_list)
    fcfp4_array = np.zeros((drug_len, num_Bits))

    if comb_num==1:
        for i, smiles in enumerate(drug_SMILES_list):
            smi = smiles
            mol = Chem.MolFromSmiles(smi)
            fcfp4 = AllChem.GetMorganFingerprintAsBitVect(mol, 2, useFeatures=True, nBits=num_Bits).ToBitString()
            fcfp4_list = np.array(list(fcfp4), dtype=np.float32)
            fcfp4_list = fcfp4_list*np.log10(dose_list[i]+1)
            fcfp4_array[i] = fcfp4_list
    else:
        for i, smiles in enumerate(drug_SMILES_list):
            smiles_list = smiles.split('+')
            for smi in smiles_list:
                mol = Chem.MolFromSmiles(smi)
                fcfp4 = AllChem.GetMorganFingerprintAsBitVect(mol, 2, useFeatures=True, nBits=num_Bits).ToBitString()
                fcfp4_list = np.array(list(fcfp4), dtype=np.float32)
                fcfp4_list = fcfp4_list*np.log10(float(dose_list[i])+1)
                fcfp4_array[i] += fcfp4_list
    return fcfp4_array 


# --- Step 9: mechanism-of-action conditioning. PREREG_step9_pathway_conditioning.md
_PATHWAY_MAP = None
_PATHWAY_VOCAB = None

def _load_pathway_map():
    """SMILES -> pathway_level_2, built once from the source object.

    The split files carry only Group, dose, SMILES and cell_name, so the
    annotation is looked up rather than stored. Verified: 188 SMILES, each
    mapping to exactly one pathway, zero unresolved across all splits.
    The vocabulary is all 55 pathways in the source, NOT the subset present in
    any one split, so the input width is fixed across splits and partitions.
    """
    global _PATHWAY_MAP, _PATHWAY_VOCAB
    if _PATHWAY_MAP is not None: return
    import scanpy as _sc
    _fo = _sc.read_h5ad('/scratch/ni2071/Diffusion/sciplex/Sci_Plex.h5ad').obs
    _sm = _fo['SMILES'].astype(str).values
    _p2 = _fo['pathway_level_2'].astype(str).values
    m = {}
    for a, b in zip(_sm, _p2):
        if a in m and m[a] != b:
            raise ValueError('SMILES %s maps to both %r and %r' % (a, m[a], b))
        m[a] = b
    _PATHWAY_MAP = m
    _PATHWAY_VOCAB = sorted(set(_p2))
    print('[pathway] %d SMILES mapped, %d pathway categories'
          % (len(m), len(_PATHWAY_VOCAB)))


def Pathway_dose_encoder(drug_SMILES_list, dose_list, comb_num=1):
    """One-hot pathway_level_2, scaled by log10(dose+1).

    Mirrors Drug_dose_encoder exactly, including the multiplicative dose
    scaling, so a zero-dose control yields an all-zero vector in both arms.
    Only the block being scaled differs: 55 mechanism categories rather than
    1024 Morgan bits.
    """
    _load_pathway_map()
    n = len(drug_SMILES_list)
    out = np.zeros((n, len(_PATHWAY_VOCAB)), dtype=np.float32)
    idx = {p: i for i, p in enumerate(_PATHWAY_VOCAB)}
    for i, smi in enumerate(drug_SMILES_list):
        parts = [smi] if comb_num == 1 else smi.split('+')
        for p in parts:
            key = _PATHWAY_MAP.get(str(p))
            if key is None:
                raise KeyError('no pathway annotation for SMILES %r' % p)
            out[i, idx[key]] += 1.0
        out[i] *= np.log10(float(dose_list[i]) + 1)
    return out


def Combined_dose_encoder(drug_SMILES_list, dose_list, comb_num=1, num_Bits=1024):
    """Step 9 arm C: fingerprint concatenated with pathway one-hot.

    Keeps the published 1024-bit input intact and ADDS the mechanistic signal,
    so a null result cannot be attributed to having removed information. Both
    blocks carry the same multiplicative dose scaling as the originals, so a
    zero-dose control still yields an all-zero vector.
    """
    fp = Drug_dose_encoder(drug_SMILES_list, dose_list, comb_num=comb_num, num_Bits=num_Bits)
    pw = Pathway_dose_encoder(drug_SMILES_list, dose_list, comb_num=comb_num)
    return np.concatenate([fp, pw], axis=1).astype(np.float32)


# --- Step 10: stochastic ChemicalVAE conditioning ---
# PREREG_step10_chemvae_stochastic.md. Unlike the other encoders, this one is
# STATEFUL at construction: Chemvae_dose_encoder is called once in AnnDataDataset.__init__,
# so each CELL receives one random draw from the pool that stays fixed for the whole
# training run (cells of the same compound differ; nothing is re-drawn per step or epoch).
# Fresh draws across evaluation processes come from SQUIDIFF_DRAW_SEED. The pool holds 64
# draws per compound, precomputed with their pretrained weights. (Comment corrected for the
# release, 21 Oct 2026; the behaviour is unchanged.)
_CHEMVAE_POOL = None

def _load_chemvae_pool():
    global _CHEMVAE_POOL
    if _CHEMVAE_POOL is not None: return _CHEMVAE_POOL
    d = np.load('/scratch/ni2071/PerturbNet/chemvae_pool.npz', allow_pickle=True)
    _CHEMVAE_POOL = ({str(x): i for i, x in enumerate(d['smiles'])},
                     d['pool'], int(d['n_pool']))
    print('[chemvae] pool loaded: %d compounds, %d draws, dim %d'
          % (_CHEMVAE_POOL[1].shape[0], _CHEMVAE_POOL[2], _CHEMVAE_POOL[1].shape[2]))
    return _CHEMVAE_POOL

def Chemvae_dose_encoder(drug_SMILES_list, dose_list, comb_num=1, rng=None):
    """One RANDOM draw per call, dose-scaled exactly as every other encoder is."""
    idx, pool, n_pool = _load_chemvae_pool()
    if rng is None:
        # Each evaluation draw is a separate process, so the pool
        # selection must be seeded per process or every draw would be
        # identical and the averaging would be over nothing.
        _s = os.environ.get('SQUIDIFF_DRAW_SEED')
        rng = np.random.RandomState(int(_s)) if _s is not None else np.random
    out = np.zeros((len(drug_SMILES_list), pool.shape[2]), dtype=np.float32)
    for i, smi in enumerate(drug_SMILES_list):
        j = idx.get(str(smi))
        if j is None:
            raise KeyError('no ChemicalVAE embedding for SMILES %r; it must be '
                           'excluded from every arm, not silently zeroed' % smi)
        out[i] = pool[j, rng.randint(n_pool)]
    return out * np.log10(np.asarray(dose_list, dtype=float) + 1)[:, None]

def Fp_chemvae_dose_encoder(drug_SMILES_list, dose_list, comb_num=1, rng=None):
    """GATED arm: the published fingerprint with ChemicalVAE appended, so a null
    cannot be attributed to having removed information."""
    fp = Drug_dose_encoder(drug_SMILES_list, dose_list, comb_num=comb_num)
    cv = Chemvae_dose_encoder(drug_SMILES_list, dose_list, comb_num=comb_num, rng=rng)
    return np.concatenate([fp, cv], axis=1).astype(np.float32)

# --- Step 16: target-aware compound representation. PREREG_step16_tahoe_targets.md
# The compound enters as the 1,024-bit fingerprint, then a BAG of its curated targets (one input per symbol of a fixed
# vocabulary V), then a has-targets flag; every block is scaled by log10(dose + 1) exactly as Drug_dose_encoder, so a
# zero-dose control is all zeros. Lookup: SMILES (as written in the h5ad files) -> usable drug name(s) through the Step
# 15a record (drugs.table_smiles) -> curated_targets_hgnc (tahoe_results/step15_targets_curated_hgnc.csv, Repurposing
# Hub targets in current HGNC symbols). Names sharing a SMILES contribute the union of their sets (counted and printed).
# V = the sorted union of the curated symbols of all usable drugs; its size is pinned (TARGET_VOCAB_SIZE) because the
# encoder's width is part of the model architecture.
# The shuffled control gives every drug the bag and flag of ANOTHER drug: one fixed cyclic order of the usable drugs
# (RandomState(0) permutation of the sorted names; a drug receives the bag of the next drug in the cycle, so no drug
# keeps its own); the same order at training and at sampling, printed with its hash. Width, parameter count and input
# statistics equal the targets arm; the correspondence between a structure and its targets is destroyed.
# Both inputs are read from the repository root (the parent of this package): nothing is typed in.
TARGET_VOCAB_SIZE = 544
_TARGETS = None


def _load_target_map():
    global _TARGETS
    if _TARGETS is not None:
        return _TARGETS
    import csv as _csv, hashlib as _hl, json as _json
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    rec_path = os.path.join(root, 'tahoe_results', 'step15_data_record.json')
    tab_path = os.path.join(root, 'tahoe_results', 'step15_targets_curated_hgnc.csv')
    for q in (rec_path, tab_path):
        if not os.path.isfile(q):
            raise FileNotFoundError('[targets] missing %s' % q)
    table_smiles = _json.load(open(rec_path))['drugs']['table_smiles']        # usable drug name -> SMILES in the files
    bag_of = {}
    with open(tab_path) as f:
        for row in _csv.DictReader(f):
            bag_of[row['tahoe_drug']] = frozenset(x for x in (row.get('curated_targets_hgnc') or '').split('|') if x)
    names = sorted(table_smiles)
    if set(names) != set(bag_of):
        raise ValueError('[targets] the target table lists %d drugs, the data record %d usable drugs; they must be the '
                         'same set' % (len(bag_of), len(names)))
    vocab = sorted(set().union(*bag_of.values()))
    if len(vocab) != TARGET_VOCAB_SIZE:
        raise ValueError('[targets] %d curated target symbols, the pre-registered vocabulary has %d'
                         % (len(vocab), TARGET_VOCAB_SIZE))
    by_smiles = {}
    for d in names:
        by_smiles.setdefault(str(table_smiles[d]), []).append(d)
    shared = {s: ds for s, ds in by_smiles.items() if len(ds) > 1}
    perm = np.random.RandomState(0).permutation(len(names))
    partner = {names[perm[i]]: names[perm[(i + 1) % len(names)]] for i in range(len(names))}
    if len(names) < 2 or any(partner[d] == d for d in names):
        raise ValueError('[targets] the shuffled order has a fixed point')
    bag = {s: frozenset().union(*[bag_of[d] for d in ds]) for s, ds in by_smiles.items()}
    bag_shuf = {s: frozenset().union(*[bag_of[partner[d]] for d in ds]) for s, ds in by_smiles.items()}
    perm_sha = _hl.sha256('\n'.join('%s>%s' % (d, partner[d]) for d in names).encode()).hexdigest()
    n_with = sum(1 for s in by_smiles if bag[s])
    n_diff = sum(1 for s in by_smiles if bag_shuf[s] != bag[s])
    _TARGETS = dict(vocab=vocab, index={t: i for i, t in enumerate(vocab)}, bag=bag, bag_shuf=bag_shuf, names=names,
                    bag_of_name=bag_of, partner=partner, perm_sha256=perm_sha, shared=shared, record=rec_path, table=tab_path)
    print('[targets] %d usable drugs, %d SMILES (%d shared by two names: %s), %d SMILES with curated targets, vocabulary '
          '%d symbols | shuffled order sha256 %s, no fixed point, shuffled bag differs from the own bag for %d of %d SMILES'
          % (len(names), len(by_smiles), len(shared), shared, n_with, len(vocab), perm_sha[:12], n_diff, len(by_smiles)))
    return _TARGETS


def Target_dose_encoder(drug_SMILES_list, dose_list, comb_num=1, shuffled=False):
    """Bag of curated targets (|V| inputs) followed by a has-targets flag, scaled by log10(dose + 1) exactly as the
    fingerprint is; shuffled=True uses the bag and flag of the drug's partner under the fixed cyclic order."""
    T = _load_target_map()
    bags = T['bag_shuf'] if shuffled else T['bag']
    n = len(drug_SMILES_list)
    out = np.zeros((n, len(T['vocab']) + 1), dtype=np.float32)
    for i, smi in enumerate(drug_SMILES_list):
        parts = [smi] if comb_num == 1 else str(smi).split('+')
        for p in parts:
            key = str(p)
            if key not in bags:
                raise KeyError('[targets] no usable drug with SMILES %r in the Step 15a record' % key)
            for t in bags[key]:
                out[i, T['index'][t]] = 1.0
            if bags[key]:
                out[i, -1] = 1.0
        out[i] *= np.log10(float(dose_list[i]) + 1)
    return out


def Fp_targets_dose_encoder(drug_SMILES_list, dose_list, comb_num=1, num_Bits=1024, shuffled=False):
    """Step 16 arms: the published fingerprint with the target bag and flag appended (shuffled=True: the control)."""
    fp = Drug_dose_encoder(drug_SMILES_list, dose_list, comb_num=comb_num, num_Bits=num_Bits)
    tg = Target_dose_encoder(drug_SMILES_list, dose_list, comb_num=comb_num, shuffled=shuffled)
    return np.concatenate([fp, tg], axis=1).astype(np.float32)


DRUG_ENCODERS = {
    'fingerprint': lambda s, d, c: Drug_dose_encoder(s, d, comb_num=c),
    'pathway':     lambda s, d, c: Pathway_dose_encoder(s, d, comb_num=c),
    'both':        lambda s, d, c: Combined_dose_encoder(s, d, comb_num=c),
    'chemvae':     lambda s, d, c: Chemvae_dose_encoder(s, d, comb_num=c),
    'fp_chemvae':  lambda s, d, c: Fp_chemvae_dose_encoder(s, d, comb_num=c),
    'fingerprint_s10': lambda s, d, c: Drug_dose_encoder(s, d, comb_num=c),
    'fp_targets':      lambda s, d, c: Fp_targets_dose_encoder(s, d, comb_num=c),                   # Step 16
    'fp_targets_shuf': lambda s, d, c: Fp_targets_dose_encoder(s, d, comb_num=c, shuffled=True),    # Step 16 control
}
DRUG_ENCODER_DIMS = {'fingerprint': 1024, 'pathway': 55, 'both': 1079,
                     'chemvae': 196, 'fp_chemvae': 1220,
                     'fingerprint_s10': 1024,
                     'fp_targets': 1024 + TARGET_VOCAB_SIZE + 1,            # Step 16: 1,569
                     'fp_targets_shuf': 1024 + TARGET_VOCAB_SIZE + 1}       # same width, selected by name

class AnnDataDataset(Dataset):
    def __init__(self, adata, control_adata=None,use_drug_structure=False,comb_num=1,
                 drug_encoding='fingerprint'):
        if drug_encoding not in DRUG_ENCODERS:
            raise ValueError('drug_encoding must be one of %s, got %r'
                             % (sorted(DRUG_ENCODERS), drug_encoding))
        self.drug_encoding = drug_encoding
        self.use_drug_structure = use_drug_structure
        if type(adata.X)==np.ndarray:
            self.features = torch.tensor(adata.X, dtype=torch.float32)
        else:
            self.features = torch.tensor(adata.X.toarray(), dtype=torch.float32)
        
        if self.use_drug_structure:
            if type(control_adata.X)==np.ndarray:
                self.control_features = torch.tensor(control_adata.X, dtype=torch.float32)
            else:
                self.control_features = torch.tensor(control_adata.X.toarray(), dtype=torch.float32)
                
            self.drug_type_list = adata.obs['SMILES'].to_list()
            self.dose_list = adata.obs['dose'].to_list()
            #self.encoded_obs_tensor = torch.tensor(adata.obs['Group'].copy().values, dtype=torch.float32)
            self.encoded_obs_tensor = adata.obs['Group'].copy().values
            _enc = DRUG_ENCODERS[getattr(self, 'drug_encoding', 'fingerprint')]
            self.encode_drug_doses = _enc(self.drug_type_list, self.dose_list, comb_num)
            self.encode_drug_doses = torch.tensor(self.encode_drug_doses, dtype=torch.float32)
        else:
            self.encoded_obs_tensor = adata.obs['Group'].copy().values
        
    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
       
        if self.use_drug_structure:
            return {'feature':self.features[idx], 'drug_dose':self.encode_drug_doses[idx], 'group': self.encoded_obs_tensor[idx],'control_feature':self.control_features[idx]}
        else:
            return {'feature':self.features[idx], 'group': self.encoded_obs_tensor[idx]}
            
    

def prepared_data(data_dir=None,control_data_dir=None, batch_size=64,use_drug_structure=False,comb_num=1, drug_encoding='fingerprint'):
     
    
    train_adata = sc.read_h5ad(data_dir)
    # Step 10: three training compounds have no ChemicalVAE embedding. They are
    # dropped from EVERY arm, the fingerprint control included, so all arms see
    # identical data. Filtering here rather than writing new h5ad files keeps the
    # frozen datasets untouched. PREREG_step10_chemvae_stochastic.md section 4.
    if drug_encoding in ('chemvae', 'fp_chemvae', 'fingerprint_s10'):
        _p = np.load('/scratch/ni2071/PerturbNet/chemvae_pool.npz', allow_pickle=True)
        _ok = set(str(x) for x in _p['smiles'])
        _m = np.array([str(x) in _ok for x in train_adata.obs['SMILES'].astype(str)])
        print('[chemvae_filter] %d of %d cells retained, %d compounds dropped'
              % (_m.sum(), len(_m),
                 len(set(train_adata.obs['SMILES'].astype(str)[~_m]))))
        train_adata = train_adata[_m].copy()
        _S10_MASK = _m        # the control file is paired 1:1 BY ROW and must be
                              # filtered identically, or every pair after the first
                              # exclusion is misaligned
        if drug_encoding == 'fingerprint_s10':
            drug_encoding = 'fingerprint'   # control arm: same data, published encoder
    if use_drug_structure:
        control_adata = sc.read_h5ad(control_data_dir)
        if '_S10_MASK' in dir():
            if control_adata.n_obs != len(_S10_MASK):
                raise ValueError('control has %d rows, treated mask has %d - the 1:1 '
                                 'row pairing does not hold and filtering is unsafe'
                                 % (control_adata.n_obs, len(_S10_MASK)))
            control_adata = control_adata[_S10_MASK].copy()
            print('[chemvae_filter] control filtered identically: %d rows'
                  % control_adata.n_obs)
        # Release fix (21 Oct 2026): the pairing is positional (row i of the control file is the
        # vehicle partner of treated row i, written that way by prep_step15_tahoe.py). The prep
        # asserts this when it writes the files; assert it again here so a mismatched pair of
        # files can never be trained on silently.
        if control_adata.n_obs != train_adata.n_obs:
            raise ValueError('control file has %d rows, treated file has %d; the files must be '
                             'row-aligned 1:1 (same prep run)' % (control_adata.n_obs, train_adata.n_obs))
    else:
        control_adata = None
    
    _data_dataset = AnnDataDataset(train_adata,control_adata,use_drug_structure,comb_num,
                                   drug_encoding=drug_encoding)


    dataloader = DataLoader(
                _data_dataset, 
                batch_size=batch_size,
                shuffle=True, 
                )
        
    return dataloader