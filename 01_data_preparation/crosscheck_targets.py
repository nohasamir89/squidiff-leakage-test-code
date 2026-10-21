#!/usr/bin/env python
"""Roadmap step 2 (2026-10-04): cross-check the Tahoe drug-target annotation (GPT-4o-proposed, MedChemExpress-validated)
against the Broad Drug Repurposing Hub (Corsello et al. 2017; release 2020-03-24; CC-BY 4.0), and produce the curated
target set the thesis will use.

Matching each of the 375 usable Tahoe drugs to a Hub compound, in this priority:
  1. PubChem CID (Tahoe table vs Hub samples file);
  2. structure: InChIKey connectivity block (first 14 characters, so salts/stereo variants match) of the Tahoe
     canonical SMILES vs the Hub samples' InChIKey; fallback RDKit canonical SMILES of the largest fragment;
  3. name (case-insensitive, trailing parenthesised qualifier dropped) vs the Hub's pert_iname.
Agreement per matched drug between the Tahoe targets and the Hub targets: identical / Tahoe subset of Hub / Hub subset
of Tahoe / partial overlap / disjoint / one side empty.
Curated targets = the Hub's targets where a Hub match exists and is non-empty; otherwise none (the Tahoe GPT targets are
kept in a separate column and flagged, never silently used).

Writes tahoe_results/step15_targets_crosscheck.json and tahoe_results/step15_targets_curated.csv.
"""
import json
import os
import re
import sys

import pandas as pd
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')
HUB_DIR = '/scratch/ni2071/Diffusion/tahoe/external/repurposing_hub'
HUB_DRUGS = os.path.join(HUB_DIR, 'repurposing_drugs_20200324.txt')
HUB_SAMPLES = os.path.join(HUB_DIR, 'repurposing_samples_20200324.txt')
DRUG_TABLE = '/scratch/ni2071/Tahoa_2/preprocessing/tahoe_drug_metadata.csv'
RECORD = 'tahoe_results/step15_data_record.json'
INV_AB = 'tahoe_results/step15_coverage_inventory.json'
OUT = 'tahoe_results/step15_targets_crosscheck.json'
OUT_CSV = 'tahoe_results/step15_targets_curated.csv'


def norm_name(s):
    s = re.sub(r'\s*\(.*\)\s*$', '', str(s)).strip().lower()
    return re.sub(r'\s+', ' ', s)


def largest(smiles):
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    m = Chem.MolFromSmiles(smiles.strip())
    if m is None:
        return None
    frags = Chem.GetMolFrags(m, asMols=True)
    return max(frags, key=lambda x: x.GetNumHeavyAtoms())


def ikey_block(mol):
    try:
        k = Chem.MolToInchiKey(mol)
        return k[:14] if k else None
    except Exception:
        return None


def split_targets(s):
    if not isinstance(s, str) or not s.strip():
        return set()
    return set(t.strip() for t in re.split(r'[|,;]', s) if t.strip())


def main():
    for q in (HUB_DRUGS, HUB_SAMPLES, DRUG_TABLE, RECORD, INV_AB):
        if not os.path.isfile(q):
            sys.exit('FATAL: missing %s' % q)
    for q in (OUT, OUT_CSV):
        if os.path.exists(q):
            sys.exit('FATAL: %s exists' % q)
    rec = json.load(open(RECORD))
    inv = json.load(open(INV_AB))
    name_map = rec.get('name_map', {})
    usable = sorted(rec['drugs']['structure'])
    tab = pd.read_csv(DRUG_TABLE)
    tab['key'] = tab['drug'].astype(str).str.strip()
    tab = tab.drop_duplicates('key').set_index('key')

    hd = pd.read_csv(HUB_DRUGS, sep='\t', comment='!', dtype=str)
    hs = pd.read_csv(HUB_SAMPLES, sep='\t', comment='!', dtype=str)
    print('[hub] drugs %d rows | samples %d rows' % (len(hd), len(hs)))
    hub_targets = {str(r['pert_iname']): split_targets(r['target']) for _, r in hd.iterrows()}
    hub_moa = {str(r['pert_iname']): str(r['moa']) if isinstance(r['moa'], str) else '' for _, r in hd.iterrows()}
    by_cid, by_ikey, by_smiles, by_name = {}, {}, {}, {}
    for _, r in hs.iterrows():
        nm = str(r['pert_iname'])
        by_name.setdefault(norm_name(nm), set()).add(nm)
        cid = r.get('pubchem_cid')
        if isinstance(cid, str) and cid.strip() and cid.strip().lower() != 'nan':
            try:
                by_cid.setdefault(int(float(cid)), set()).add(nm)
            except ValueError:
                pass
        ik = r.get('InChIKey')
        if isinstance(ik, str) and len(ik) >= 14:
            by_ikey.setdefault(ik[:14], set()).add(nm)
        mol = largest(r.get('smiles'))
        if mol is not None:
            by_smiles.setdefault(Chem.MolToSmiles(mol), set()).add(nm)
    print('[hub] index sizes: cid %d | inchikey blocks %d | canonical smiles %d | names %d' % (len(by_cid), len(by_ikey), len(by_smiles), len(by_name)))

    rows, summary = [], dict(n_usable=len(usable), matched=0, by=dict(cid=0, inchikey=0, smiles=0, name=0), agreement={}, hub_match_no_targets=0)
    for d in usable:
        key = d if d in tab.index else name_map.get(d, d)
        r = tab.loc[key] if key in tab.index else None
        tahoe_t = set(inv['per_drug'].get(d, {}).get('targets', []))
        hits, how = set(), ''
        if r is not None:
            cid = r.get('pubchem_cid')
            try:
                cid = int(float(cid)) if pd.notna(cid) else None
            except (TypeError, ValueError):
                cid = None
            if cid is not None and cid in by_cid:
                hits, how = by_cid[cid], 'cid'
            if not hits:
                mol = largest(r.get('canonical_smiles'))
                if mol is not None:
                    blk = ikey_block(mol)
                    if blk and blk in by_ikey:
                        hits, how = by_ikey[blk], 'inchikey'
                    elif Chem.MolToSmiles(mol) in by_smiles:
                        hits, how = by_smiles[Chem.MolToSmiles(mol)], 'smiles'
        if not hits:
            for cand in (norm_name(d), norm_name(name_map.get(d, d))):
                if cand in by_name:
                    hits, how = by_name[cand], 'name'
                    break
        hub_names = sorted(hits)
        hub_t = set().union(*[hub_targets.get(n, set()) for n in hub_names]) if hub_names else set()
        moa = '; '.join(sorted(set(hub_moa.get(n, '') for n in hub_names) - {''}))
        if hub_names:
            summary['matched'] += 1
            summary['by'][how] += 1
            if not hub_t:
                summary['hub_match_no_targets'] += 1
                agree = 'hub match, no hub targets'
            elif not tahoe_t:
                agree = 'hub targets, no tahoe targets'
            elif tahoe_t == hub_t:
                agree = 'identical'
            elif tahoe_t < hub_t:
                agree = 'tahoe subset of hub'
            elif hub_t < tahoe_t:
                agree = 'hub subset of tahoe'
            elif tahoe_t & hub_t:
                agree = 'partial overlap'
            else:
                agree = 'disjoint'
        else:
            agree = 'no hub match'
        summary['agreement'][agree] = summary['agreement'].get(agree, 0) + 1
        rows.append(dict(tahoe_drug=d, hub_names='|'.join(hub_names), matched_by=how, agreement=agree,
                         tahoe_targets_gpt='|'.join(sorted(tahoe_t)), hub_targets='|'.join(sorted(hub_t)), hub_moa=moa,
                         curated_targets='|'.join(sorted(hub_t)), curated_source='repurposing_hub_20200324' if hub_t else ''))
    df = pd.DataFrame(rows)
    per_split = {}
    for k, s in rec['splits'].items():
        held = [name_map.get(x, x) for x in s['held_drugs']]
        held = [h if h in usable else next((d for d in usable if name_map.get(d, d) == h), h) for h in held]
        sub = df[df['tahoe_drug'].isin(held)]
        per_split['drug_split_%s' % k] = dict(n_held=len(held), hub_matched=int((sub['hub_names'] != '').sum()),
                                              with_curated_targets=int((sub['curated_targets'] != '').sum()),
                                              with_gpt_targets=int((sub['tahoe_targets_gpt'] != '').sum()))
    summary['per_split'] = per_split
    summary['with_curated_targets'] = int((df['curated_targets'] != '').sum())
    summary['with_gpt_targets'] = int((df['tahoe_targets_gpt'] != '').sum())
    print('[match] %d / %d usable drugs matched to the Hub (by cid %d, inchikey %d, smiles %d, name %d)'
          % (summary['matched'], len(usable), summary['by']['cid'], summary['by']['inchikey'], summary['by']['smiles'], summary['by']['name']))
    print('[targets] curated (Hub) targets for %d drugs | GPT targets for %d | agreement: %s'
          % (summary['with_curated_targets'], summary['with_gpt_targets'], summary['agreement']))
    for k, v in per_split.items():
        print('[%s] held-out %d | hub matched %d | curated targets %d | gpt targets %d' % (k, v['n_held'], v['hub_matched'], v['with_curated_targets'], v['with_gpt_targets']))
    df.to_csv(OUT_CSV, index=False)
    json.dump(dict(hub_release='2020-03-24', hub_drugs_rows=len(hd), hub_samples_rows=len(hs), summary=summary), open(OUT, 'w'), indent=1)
    print('wrote %s and %s' % (OUT, OUT_CSV))


if __name__ == '__main__':
    main()
