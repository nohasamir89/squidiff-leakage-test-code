#!/usr/bin/env python
"""Roadmap step 3 (2026-10-04): gene-symbol harmonisation. The 200-gene panel (Tahoe names), the curated drug targets
(Repurposing Hub 2020 names) and the CPTAC GBM/COAD tables (names of their own vintage) are each mapped to the CURRENT
HGNC symbol with the HGNC complete set (quarterly archive 2026-01-06, kept permanently by HGNC), and the coverage counts
of inventory (c) are recomputed on the harmonised names.

Mapping of one name, in this priority (same precedence as HGNChelper: approved, then previous, then alias):
  1. a bare Ensembl gene id (ENSG..., version suffix dropped)   -> HGNC entry with that ensembl_gene_id       [ensembl]
  2. the name IS a current approved symbol                      -> itself                                     [current]
  3. the name is the previous symbol of exactly one entry       -> that entry's current symbol                [previous]
  4. the name is an alias of exactly one entry                  -> that entry's current symbol                [alias]
  5. the name equals a current symbol ignoring case (unique)    -> that symbol                                [case]
  otherwise unmapped (an ambiguous previous symbol or alias, hitting several entries, is also left unmapped).
Each entry also carries HGNC's locus_group (protein-coding gene / non-coding RNA / pseudogene / other), which replaces
the hand classification of the panel's non-coding entries in the coverage note.

Reproduction guard: the raw-name counts recomputed here must equal inventory (c) exactly (panel in proteome /
transcriptome / both; GPT target symbols in proteome; drugs with a quantified GPT target), otherwise the script stops.

Writes tahoe_results/step15_symbol_harmonisation.json, tahoe_results/step15_panel_harmonised.csv and
tahoe_results/step15_targets_curated_hgnc.csv (per drug: curated targets as current HGNC symbols).

v2 (2026-10-05, before anything was committed): a panel gene also counts as present in a CPTAC table when its Ensembl
id - the bare ENSG name itself, or HGNC's ensembl_gene_id of the entry it mapped to - is among the table's Database_ID
ids (the WashU transcriptome carries an Ensembl id for every column). v1 matched on symbols only, which left the 15
Ensembl-only panel genes (absent from HGNC) unmatched in the transcriptome.
Usage (repo root, conda env active, login node not needed): python -u harmonise_symbols.py
"""
import csv
import hashlib
import json
import os
import sys

import pandas as pd

HGNC = '/scratch/ni2071/Diffusion/tahoe/external/hgnc/hgnc_complete_set_2026-01-06.tsv'
HGNC_MD5 = 'cf5ae3bd5e2432e9def5e5a1c2be7a86'                 # the file as downloaded on 2026-10-04 (44,810 lines)
CPTAC_DIR = '/scratch/ni2071/Diffusion/tahoe/external/cptac'
CPTAC_REC = os.path.join(CPTAC_DIR, 'cptac_fetch_record.json')
PANEL = 'tahoe_results/step15_gene_panel.txt'
DATA_REC = 'tahoe_results/step15_data_record.json'
INV_C = 'tahoe_results/step15_coverage_inventory_c.json'
CURATED = 'tahoe_results/step15_targets_curated.csv'
OUT = 'tahoe_results/step15_symbol_harmonisation.json'
OUT_PANEL = 'tahoe_results/step15_panel_harmonised.csv'
OUT_TARGETS = 'tahoe_results/step15_targets_curated_hgnc.csv'
COHORTS = ('gbm', 'coad')


def digest(p, algo):
    m = hashlib.new(algo)
    with open(p, 'rb') as f:
        for c in iter(lambda: f.read(1 << 20), b''):
            m.update(c)
    return m.hexdigest()


class Hgnc:
    """Lookup tables built once from the HGNC complete set."""

    def __init__(self, path):
        t = pd.read_csv(path, sep='\t', dtype=str, quoting=csv.QUOTE_NONE, keep_default_na=False)
        for col in ('hgnc_id', 'symbol', 'status', 'locus_group', 'locus_type', 'prev_symbol', 'alias_symbol', 'ensembl_gene_id'):
            t[col] = t[col].str.strip().str.strip('"')          # multi-valued fields may be wrapped in double quotes
        self.n_rows = len(t)
        t = t[t['status'] == 'Approved']
        self.n_approved = len(t)
        self.locus_counts = t['locus_group'].value_counts().to_dict()
        self.sym = dict(zip(t['hgnc_id'], t['symbol']))
        self.group = dict(zip(t['hgnc_id'], t['locus_group']))
        self.ltype = dict(zip(t['hgnc_id'], t['locus_type']))
        self.current = dict(zip(t['symbol'], t['hgnc_id']))
        self.ensg_of = dict(zip(t['hgnc_id'], t['ensembl_gene_id']))
        self.prev, self.alias, self.ensg, lower = {}, {}, {}, {}
        for hid, s, p, a, e in zip(t['hgnc_id'], t['symbol'], t['prev_symbol'], t['alias_symbol'], t['ensembl_gene_id']):
            for x in p.split('|'):
                if x:
                    self.prev.setdefault(x, set()).add(hid)
            for x in a.split('|'):
                if x:
                    self.alias.setdefault(x, set()).add(hid)
            if e:
                self.ensg[e] = hid
            lower.setdefault(s.lower(), set()).add(hid)
        self.lower = {k: next(iter(v)) for k, v in lower.items() if len(v) == 1}
        self.cache = {}

    def resolve(self, name):
        """-> (route, hgnc_id or None). Cached: the CPTAC tables have tens of thousands of names."""
        if name in self.cache:
            return self.cache[name]
        n = str(name).strip()
        if n.startswith('ENSG'):
            hid = self.ensg.get(n.split('.')[0])
            r = ('ensembl', hid) if hid else ('unmapped_ensembl', None)
        elif n in self.current:
            r = ('current', self.current[n])
        elif n in self.prev:
            r = ('previous', next(iter(self.prev[n]))) if len(self.prev[n]) == 1 else ('ambiguous_previous', None)
        elif n in self.alias:
            r = ('alias', next(iter(self.alias[n]))) if len(self.alias[n]) == 1 else ('ambiguous_alias', None)
        elif n.lower() in self.lower:
            r = ('case', self.lower[n.lower()])
        else:
            r = ('unmapped', None)
        self.cache[name] = r
        return r

    def symbol(self, name):
        """Current symbol if mapped, else the raw name (so unmapped names still match themselves)."""
        route, hid = self.resolve(name)
        return self.sym[hid] if hid else str(name).strip()


def routes(h, names):
    c = {}
    for n in names:
        r = h.resolve(n)[0]
        c[r] = c.get(r, 0) + 1
    return dict(sorted(c.items()))


def table_symbols(h, path, expect_sha, levels):
    """Raw and harmonised symbol sets of one CPTAC table from its header only. Columns were flattened by
    fetch_cptac.py as 'level1|level2|...'; the gene symbol is the 'Name' level, and when a 'Database_ID' level holds
    an Ensembl id it is used for names that fail to map by symbol (transcriptome tables)."""
    if not os.path.isfile(path):
        sys.exit('FATAL: missing %s' % path)
    if digest(path, 'sha256') != expect_sha:
        sys.exit('FATAL: %s differs from the fetch record' % path)
    cols = [str(c) for c in pd.read_csv(path, nrows=0).columns][1:]           # first column = sample index
    multi = len(levels) > 1
    name_i = levels.index('Name') if 'Name' in levels else 0
    db_i = levels.index('Database_ID') if 'Database_ID' in levels else None
    raw, harm, ensg_ids, changed, via_ensg, unmapped = set(), set(), set(), 0, 0, 0
    for c in cols:
        parts = c.split('|') if multi else [c]
        name = parts[name_i] if name_i < len(parts) else parts[0]
        raw.add(name)
        if db_i is not None and db_i < len(parts) and parts[db_i].startswith('ENSG'):
            ensg_ids.add(parts[db_i].split('.')[0])
        route, hid = h.resolve(name)
        if hid is None and db_i is not None and db_i < len(parts) and parts[db_i].startswith('ENSG'):
            route, hid = h.resolve(parts[db_i])
            if hid:
                via_ensg += 1
        if hid:
            s = h.sym[hid]
            if s != name:
                changed += 1
        else:
            s = name
            unmapped += 1
        harm.add(s)
    return dict(n_columns=len(cols), raw=raw, harmonised=harm, ensg_ids=ensg_ids, n_raw_symbols=len(raw), n_harmonised_symbols=len(harm),
                n_ensembl_ids=len(ensg_ids), columns_renamed=changed, columns_mapped_via_ensembl=via_ensg, columns_unmapped=unmapped)


def main():
    for q in (HGNC, CPTAC_REC, PANEL, DATA_REC, INV_C, CURATED):
        if not os.path.isfile(q):
            sys.exit('FATAL: missing %s' % q)
    for q in (OUT, OUT_PANEL, OUT_TARGETS):
        if os.path.exists(q):
            sys.exit('FATAL: %s exists' % q)
    md5 = digest(HGNC, 'md5')
    if md5 != HGNC_MD5:
        sys.exit('FATAL: %s md5 %s differs from the pinned %s' % (HGNC, md5, HGNC_MD5))
    h = Hgnc(HGNC)
    print('[hgnc] %s | md5 %s | %d entries, %d approved | locus groups %s' % (os.path.basename(HGNC), md5, h.n_rows, h.n_approved, h.locus_counts))

    drec = json.load(open(DATA_REC))
    inv_c = json.load(open(INV_C))
    cptac = json.load(open(CPTAC_REC))
    name_map = drec.get('name_map', {})

    # ------------------------------------------------------------------ panel
    panel = [l.split('\t')[0].strip() for l in open(PANEL) if l.strip() and not l.startswith('#')]
    rec_panel = [r['gene'] for r in drec['panel']]
    if len(panel) != 200 or len(set(panel)) != 200 or panel != rec_panel:
        sys.exit('FATAL: panel file (%d genes) differs from the data record panel (%d)' % (len(panel), len(rec_panel)))
    prow = []
    for g in panel:
        route, hid = h.resolve(g)
        prow.append(dict(gene=g, route=route, hgnc_id=hid or '', symbol_current=h.sym[hid] if hid else '',
                         locus_group=h.group[hid] if hid else 'unmapped', locus_type=h.ltype[hid] if hid else ''))
    pdf = pd.DataFrame(prow)
    pdf['ensembl_id'] = [g.split('.')[0] if g.startswith('ENSG') else (h.ensg_of.get(h.resolve(g)[1]) or '') for g in panel]
    panel_routes = pdf['route'].value_counts().sort_index().to_dict()
    panel_groups = pdf['locus_group'].value_counts().sort_index().to_dict()
    renamed = [(r['gene'], r['symbol_current']) for r in prow if r['symbol_current'] and r['symbol_current'] != r['gene']]
    print('[panel] 200 genes | routes %s | locus groups %s' % (panel_routes, panel_groups))
    print('[panel] renamed to the current symbol (%d): %s' % (len(renamed), ', '.join('%s->%s' % x for x in renamed)))
    print('[panel] unmapped: %s' % sorted(r['gene'] for r in prow if not r['hgnc_id']))
    panel_harm = {g: h.symbol(g) for g in panel}
    panel_ensg = {g: (g.split('.')[0] if g.startswith('ENSG') else (h.ensg_of.get(h.resolve(g)[1]) or None)) for g in panel}

    def present(g, tab):
        """Panel gene g is in table tab if its harmonised symbol is among the table's harmonised symbols, or its Ensembl
        id is among the table's Database_ID ids (v2)."""
        return panel_harm[g] in tab['harmonised'] or (panel_ensg[g] is not None and panel_ensg[g] in tab['ensg_ids'])
    dup = sorted(set(s for s in panel_harm.values() if list(panel_harm.values()).count(s) > 1))
    print('[panel] distinct symbols after harmonisation: %d of 200%s' % (len(set(panel_harm.values())), (' | collisions: %s' % dup) if dup else ''))

    # ------------------------------------------------------------------ targets (curated Hub set; GPT set only as a guard)
    cur = pd.read_csv(CURATED, dtype=str, keep_default_na=False)
    usable = list(cur['tahoe_drug'])
    split_t = lambda s: [x for x in str(s).split('|') if x]
    curated_of = {d: split_t(t) for d, t in zip(cur['tahoe_drug'], cur['curated_targets'])}
    gpt_of = {d: split_t(t) for d, t in zip(cur['tahoe_drug'], cur['tahoe_targets_gpt'])}
    cur_syms = sorted(set(x for v in curated_of.values() for x in v))
    gpt_syms = sorted(set(x for v in gpt_of.values() for x in v))
    cur_routes, gpt_routes = routes(h, cur_syms), routes(h, gpt_syms)
    cur_unmapped = sorted(s for s in cur_syms if h.resolve(s)[1] is None)
    print('[targets] curated symbols %d | routes %s | unmapped (%d): %s' % (len(cur_syms), cur_routes, len(cur_unmapped), cur_unmapped))
    print('[targets] GPT symbols %d | routes %s' % (len(gpt_syms), gpt_routes))
    curated_h = {d: sorted(set(h.symbol(x) for x in v)) for d, v in curated_of.items()}
    gpt_h = {d: sorted(set(h.symbol(x) for x in v)) for d, v in gpt_of.items()}
    tdf = pd.DataFrame(dict(tahoe_drug=usable, curated_targets=[ '|'.join(curated_of[d]) for d in usable],
                            curated_targets_hgnc=['|'.join(curated_h[d]) for d in usable],
                            n_curated_unmapped=[sum(1 for x in curated_of[d] if h.resolve(x)[1] is None) for d in usable]))

    held = {}
    for k, s in drec['splits'].items():
        hh = [name_map.get(x, x) for x in s['held_drugs']]
        held['drug_split_%s' % k] = [x if x in curated_of else next((d for d in usable if name_map.get(d, d) == x), x) for x in hh]

    # ------------------------------------------------------------------ CPTAC tables, raw vs harmonised
    out_c = {}
    P_h_all = {}
    for co in COHORTS:
        tabs = cptac['cohorts'][co]['tables']
        t = {}
        for dtype in ('proteomics', 'transcriptomics'):
            info = tabs[dtype]
            t[dtype] = table_symbols(h, os.path.join(CPTAC_DIR, info['file']), info['sha256'], info['column_levels'])
            print('[%s] %s: %d columns | raw symbols %d -> harmonised %d | columns renamed %d, mapped via Ensembl id %d, unmapped %d | Ensembl ids %d'
                  % (co, dtype, t[dtype]['n_columns'], t[dtype]['n_raw_symbols'], t[dtype]['n_harmonised_symbols'],
                     t[dtype]['columns_renamed'], t[dtype]['columns_mapped_via_ensembl'], t[dtype]['columns_unmapped'], t[dtype]['n_ensembl_ids']))
        P_raw, T_raw, P_h, T_h = t['proteomics']['raw'], t['transcriptomics']['raw'], t['proteomics']['harmonised'], t['transcriptomics']['harmonised']
        pset_raw = set(panel)
        raw = dict(panel_in_proteome=len(pset_raw & P_raw), panel_in_transcriptome=len(pset_raw & T_raw), panel_with_both=len(pset_raw & P_raw & T_raw),
                   gpt_symbols_in_proteome=len(set(gpt_syms) & P_raw), drugs_with_gpt_target_in_proteome=sum(1 for v in gpt_of.values() if set(v) & P_raw))
        c = inv_c['cohorts'][co]
        expect = dict(panel_in_proteome=c['panel_in_proteome'], panel_in_transcriptome=c['panel_in_transcriptome'], panel_with_both=c['panel_with_both'],
                      gpt_symbols_in_proteome=c['target_symbols_in_proteome'], drugs_with_gpt_target_in_proteome=c['drugs_with_target_in_proteome'])
        if raw != expect:
            sys.exit('FATAL: raw-name recount %s does not reproduce inventory (c) %s' % (raw, expect))
        tP, tT = t['proteomics'], t['transcriptomics']
        harm = dict(panel_in_proteome=sum(1 for g in panel if present(g, tP)), panel_in_transcriptome=sum(1 for g in panel if present(g, tT)),
                    panel_with_both=sum(1 for g in panel if present(g, tP) and present(g, tT)),
                    curated_symbols_in_proteome=len(set(x for v in curated_h.values() for x in v) & P_h),
                    drugs_with_curated_target_in_proteome=sum(1 for v in curated_h.values() if set(v) & P_h),
                    curated_symbols_in_proteome_raw=len(set(cur_syms) & P_raw),
                    drugs_with_curated_target_in_proteome_raw=sum(1 for v in curated_of.values() if set(v) & P_raw))
        gained_p = sorted(g for g in panel if present(g, tP) and g not in P_raw)
        gained_t = sorted(g for g in panel if present(g, tT) and g not in T_raw)
        by_id_only_t = sorted(g for g in panel if present(g, tT) and panel_harm[g] not in T_h)
        per_split = {k: dict(n_held=len(v), with_curated_targets=sum(1 for d in v if curated_h.get(d)),
                             with_curated_target_in_proteome=sum(1 for d in v if set(curated_h.get(d, [])) & P_h)) for k, v in held.items()}
        out_c[co] = dict(tables={k: {kk: vv for kk, vv in v.items() if kk not in ('raw', 'harmonised', 'ensg_ids')} for k, v in t.items()},
                         raw_names=raw, harmonised=harm, panel_gained_in_proteome=gained_p, panel_gained_in_transcriptome=gained_t,
                         panel_in_transcriptome_by_ensembl_id_only=by_id_only_t,
                         panel_not_in_proteome_after=sorted(g for g in panel if not present(g, tP)),
                         panel_not_in_transcriptome_after=sorted(g for g in panel if not present(g, tT)), per_split=per_split)
        P_h_all[co] = (P_raw, tP)
        print('[%s] panel: proteome %d -> %d, transcriptome %d -> %d, both %d -> %d (raw -> harmonised) | gained in proteome: %s | gained in transcriptome: %s | of which by Ensembl id only: %s'
              % (co, raw['panel_in_proteome'], harm['panel_in_proteome'], raw['panel_in_transcriptome'], harm['panel_in_transcriptome'],
                 raw['panel_with_both'], harm['panel_with_both'], gained_p, gained_t, by_id_only_t))
        print('[%s] curated targets: symbols in proteome %d -> %d | drugs with a quantified curated target %d -> %d of %d | held-out per split %s'
              % (co, harm['curated_symbols_in_proteome_raw'], harm['curated_symbols_in_proteome'], harm['drugs_with_curated_target_in_proteome_raw'],
                 harm['drugs_with_curated_target_in_proteome'], sum(1 for v in curated_h.values() if v),
                 {k: '%d/%d' % (v['with_curated_target_in_proteome'], v['n_held']) for k, v in per_split.items()}))
    both_raw = sum(1 for g in panel if all(g in P_h_all[co][0] for co in COHORTS))
    both_h = sum(1 for g in panel if all(present(g, P_h_all[co][1]) for co in COHORTS))
    print('[both] panel genes quantified in both proteomes: %d -> %d' % (both_raw, both_h))

    summary = dict(hgnc=dict(file=os.path.basename(HGNC), md5=md5, sha256=digest(HGNC, 'sha256'), bytes=os.path.getsize(HGNC), n_entries=h.n_rows,
                             n_approved=h.n_approved, locus_groups=h.locus_counts,
                             url='https://storage.googleapis.com/public-download-files/hgnc/archive/archive/quarterly/tsv/hgnc_complete_set_2026-01-06.tsv'),
                   panel=dict(n=len(panel), routes=panel_routes, locus_groups=panel_groups, renamed=renamed, unmapped=sorted(r['gene'] for r in prow if not r['hgnc_id']),
                              distinct_after=len(set(panel_harm.values())), collisions=dup),
                   targets=dict(curated_symbols=len(cur_syms), curated_routes=cur_routes, curated_unmapped=cur_unmapped,
                                curated_symbols_harmonised=len(set(x for v in curated_h.values() for x in v)),
                                gpt_symbols=len(gpt_syms), gpt_routes=gpt_routes, drugs_with_curated_targets=sum(1 for v in curated_h.values() if v)),
                   cohorts=out_c, panel_in_both_proteomes=dict(raw=both_raw, harmonised=both_h))
    json.dump(summary, open(OUT, 'w'), indent=1)
    pdf.to_csv(OUT_PANEL, index=False)
    tdf.to_csv(OUT_TARGETS, index=False)
    print('wrote %s, %s and %s' % (OUT, OUT_PANEL, OUT_TARGETS))


if __name__ == '__main__':
    main()
