#!/usr/bin/env python3
"""Step 16 patch (PREREG_step16_tahoe_targets.md section 9): the target-aware compound representation.

Exact-string edits; every anchor must occur exactly the expected number of times, every patched file must still parse,
and nothing is written unless every edit succeeds. Run from the repository root:  python patch_step16.py

  Squidiff/scrna_datasets.py   _load_target_map(), Target_dose_encoder, Fp_targets_dose_encoder (plain and shuffled),
                               entries 'fp_targets' and 'fp_targets_shuf' in DRUG_ENCODERS / DRUG_ENCODER_DIMS (both
                               1,024 + |V| + 1 = 1,569 wide; the shuffled variant is selected by NAME, never by width)
  train_squidiff.py            the parameter-count expectation becomes base + (drug_dimension - 1024) x width(fc1), for the
                               mlp encoder arm only
  sample_squidiff.py           the "known encoding" table is replaced by a DRUG_ENCODER_DIMS lookup; when several encodings
                               share the checkpoint's width, SQUIDIFF_DRUG_ENCODING names the arm's encoding and is
                               asserted against that width

Backward compatibility: with drug_encoding 'fingerprint' nothing changes - the added term is (1024 - 1024) x width = 0,
the width 1,024 maps to exactly one encoding, and no new module is imported unless a Step 16 encoding is requested.
"""
import ast
import hashlib
import os
import sys

EDITS = []       # (path, old, new, expected_count)
SPANS = []       # (path, first_line, last_line, new): replace from the start of first_line to the end of last_line


def edit(path, old, new, n=1):
    EDITS.append((path, old, new, n))


def span(path, first, last, new):
    SPANS.append((path, first, last, new))


# ---------------------------------------------------------------- Squidiff/scrna_datasets.py
P = 'Squidiff/scrna_datasets.py'
TARGET_BLOCK = '''
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
    perm_sha = _hl.sha256('\\n'.join('%s>%s' % (d, partner[d]) for d in names).encode()).hexdigest()
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

'''
edit(P, "\nDRUG_ENCODERS = {\n", TARGET_BLOCK + "\nDRUG_ENCODERS = {\n")
edit(P, "    'fingerprint_s10': lambda s, d, c: Drug_dose_encoder(s, d, comb_num=c),\n}\n"
        "DRUG_ENCODER_DIMS = {'fingerprint': 1024, 'pathway': 55, 'both': 1079,\n"
        "                     'chemvae': 196, 'fp_chemvae': 1220,\n"
        "                     'fingerprint_s10': 1024}\n",
        "    'fingerprint_s10': lambda s, d, c: Drug_dose_encoder(s, d, comb_num=c),\n"
        "    'fp_targets':      lambda s, d, c: Fp_targets_dose_encoder(s, d, comb_num=c),                   # Step 16\n"
        "    'fp_targets_shuf': lambda s, d, c: Fp_targets_dose_encoder(s, d, comb_num=c, shuffled=True),    # Step 16 control\n"
        "}\n"
        "DRUG_ENCODER_DIMS = {'fingerprint': 1024, 'pathway': 55, 'both': 1079,\n"
        "                     'chemvae': 196, 'fp_chemvae': 1220,\n"
        "                     'fingerprint_s10': 1024,\n"
        "                     'fp_targets': 1024 + TARGET_VOCAB_SIZE + 1,            # Step 16: 1,569\n"
        "                     'fp_targets_shuf': 1024 + TARGET_VOCAB_SIZE + 1}       # same width, selected by name\n")

# ---------------------------------------------------------------- train_squidiff.py
P = 'train_squidiff.py'
edit(P, "                 'attn_cross': 58178180 + 354624, 'mlp_ctrl': 58178180 + 354120}[_arm]\n"
        "    if _n_params != _expected:\n",
        "                 'attn_cross': 58178180 + 354624, 'mlp_ctrl': 58178180 + 354120}[_arm]\n"
        "    # Step 16 (PREREG_step16_tahoe_targets.md section 3): a drug encoding wider than the 1,024-bit fingerprint changes\n"
        "    # the encoder's first layer only, by (drug_dimension - 1024) x width(fc1) parameters. The attention arms' own\n"
        "    # modules also read the drug width, so a width other than 1,024 is accepted for the mlp arm alone.\n"
        "    _extra_in = int(args['drug_dimension']) - 1024\n"
        "    if _extra_in != 0:\n"
        "        if _arm != 'mlp':\n"
        "            raise RuntimeError('[step16] drug_dimension %d is supported for encoder_arm mlp only, got %s' % (args['drug_dimension'], _arm))\n"
        "        _fc1_w = int(model.encoder.fc1.weight.shape[0])\n"
        "        print('[step16] drug_dimension=%d: expected parameters %d = %d + %d x %d (fc1 width)'\n"
        "              % (args['drug_dimension'], _expected + _extra_in * _fc1_w, _expected, _extra_in, _fc1_w))\n"
        "        _expected += _extra_in * _fc1_w\n"
        "    if _n_params != _expected:\n")

# ---------------------------------------------------------------- sample_squidiff.py
P = 'sample_squidiff.py'
span(P, "        if _dd not in (1024, 55, 1079, 196, 1220):\n",
        "                  196: 'chemvae', 1220: 'fp_chemvae'}[_dd], _dd))\n",
        "        # Step 16 (PREREG_step16_tahoe_targets.md section 9): the width is looked up in DRUG_ENCODER_DIMS. Two Step 16\n"
        "        # encodings share the width 1,569 (fp_targets and its shuffled control), so the harness names the arm's encoding\n"
        "        # in SQUIDIFF_DRUG_ENCODING, which must agree with the checkpoint's width; a width with a single encoding needs\n"
        "        # no variable (fingerprint_s10 is an alias of fingerprint and is not listed).\n"
        "        import os as _os16\n"
        "        from Squidiff.scrna_datasets import DRUG_ENCODER_DIMS as _DIMS16\n"
        "        _cands = sorted(k for k, v in _DIMS16.items() if v == _dd and k != 'fingerprint_s10')\n"
        "        _env = _os16.environ.get('SQUIDIFF_DRUG_ENCODING')\n"
        "        if _env is not None:\n"
        "            if _env not in _DIMS16:\n"
        "                raise SystemExit('FATAL: SQUIDIFF_DRUG_ENCODING=%r is not a known encoding %s' % (_env, sorted(_DIMS16)))\n"
        "            if _DIMS16[_env] != _dd:\n"
        "                raise SystemExit('FATAL: SQUIDIFF_DRUG_ENCODING=%r has width %d, the checkpoint has %d' % (_env, _DIMS16[_env], _dd))\n"
        "            _enc_name = _env\n"
        "        elif len(_cands) == 1:\n"
        "            _enc_name = _cands[0]\n"
        "        elif not _cands:\n"
        "            raise SystemExit('FATAL: inferred drug_dimension %d is not a known encoding %s' % (_dd, _DIMS16))\n"
        "        else:\n"
        "            raise SystemExit('FATAL: drug_dimension %d is shared by %s; set SQUIDIFF_DRUG_ENCODING' % (_dd, _cands))\n"
        "        args['drug_dimension'] = _dd\n"
        "        print('drug encoding detected from checkpoint: %s (drug_dimension=%d)' % (_enc_name, _dd))\n")

# ---------------------------------------------------------------- apply, all-or-nothing
texts = {}


def load(path):
    if path not in texts:
        if not os.path.isfile(path):
            sys.exit('FATAL: %s not found (run from the repository root)' % path)
        texts[path] = open(path).read()
    return texts[path]


for path, old, new, n in EDITS:
    t = load(path)
    c = t.count(old)
    if c != n:
        sys.exit('FATAL: anchor found %d times in %s, expected %d:\n%s' % (c, path, n, old))
    texts[path] = t.replace(old, new)
for path, first, last, new in SPANS:
    t = load(path)
    if t.count(first) != 1 or t.count(last) != 1:
        sys.exit('FATAL: span anchors found %d / %d times in %s (expected 1 / 1):\n%s\n%s' % (t.count(first), t.count(last), path, first, last))
    a, b = t.index(first), t.index(last) + len(last)
    if b <= a:
        sys.exit('FATAL: span anchors out of order in %s' % path)
    print('--- %s: replacing the block\n%s--- with the Step 16 lookup' % (path, t[a:b]))
    texts[path] = t[:a] + new + t[b:]
for path, txt in texts.items():
    ast.parse(txt)                       # every patched file must still parse before anything is written
for path, txt in texts.items():
    open(path, 'w').write(txt)
    print('patched %s  lines %d  md5 %s' % (path, txt.count('\n'), hashlib.md5(txt.encode()).hexdigest()))
print('OK: %d edits and %d span replacements applied' % (len(EDITS), len(SPANS)))
