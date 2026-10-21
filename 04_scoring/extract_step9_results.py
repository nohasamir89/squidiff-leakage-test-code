"""Extract Step 7/9 values to long-format CSV, one row per plotted point.

Nothing typed by hand; each row carries its source file and SHA-256.
"""
import json, os, sys, csv, hashlib
def sha(p):
    m=hashlib.sha256()
    with open(p,'rb') as f:
        for c in iter(lambda: f.read(1<<20), b''): m.update(c)
    return m.hexdigest()
RES='leakage_results'; rows=[]
def add(exp, arm, seed, metric, value, src, split='drug_split_1'):
    rows.append(dict(experiment=exp, split=split, arm=arm, seed=seed, metric=metric,
                     value=float(value), source_file=os.path.basename(src),
                     source_sha256=sha(src)[:16]))

# ridge feature-set comparison, three splits
p=os.path.join(RES,'pathway_vs_fingerprint.json')
if not os.path.isfile(p): sys.exit('FATAL: missing '+p)
d=json.load(open(p))
for sp,r in d['per_split'].items():
    for k,lab in (('pathway_x_line','ridge pathway x line'),
                  ('fp_x_line','ridge FP x line'),
                  ('target_x_line','ridge target x line')):
        add('ridge', lab, -1, 'dpc50', r[k], p, sp)

# Step 9 model arms
p9=os.path.join(RES,'step9_analysis.json')
if not os.path.isfile(p9): sys.exit('FATAL: missing '+p9)
d9=json.load(open(p9))
for arm,v in d9['dpc_marginal'].items():
    add('step9', 'model '+arm, -1, 'dpc50_marginal', v, p9)
for arm,vals in d9['de_overlap'].items():
    for s,x in enumerate(vals): add('step9','model '+arm, s, 'de_overlap', x, p9)
for arm,vals in d9['energy'].items():
    for s,x in enumerate(vals): add('step9','model '+arm, s, 'energy_distance', x, p9)
for lab,c in d9['contrasts'].items():
    tag='C_minus_A' if lab.startswith('C') else 'B_minus_A'
    add('step9', tag, -1, 'paired_median_dpc50', c['paired_median'], p9)
    add('step9', tag+'_lo', -1, 'paired_ci', c['ci'][0], p9)
    add('step9', tag+'_hi', -1, 'paired_ci', c['ci'][1], p9)

# Step 6c: flow matching (fingerprint conditioning) already has DPC-50 on all
# three splits. Step 9's conditioning arms ran on drug_split_1 only, by design -
# splits 0 and 2 had been used twice already and section 6 recorded UNRESOLVED
# as the expected outcome, so spending them a third time was not justified.
p6=os.path.join(RES,'ridge_vs_fm.json')
if not os.path.isfile(p6): sys.exit('FATAL: missing '+p6)
d6=json.load(open(p6))
for sp,r in d6['per_split'].items():
    add('step6c','flow matching (fingerprint)',-1,'dpc50_marginal',r['marginal_median_fm'],p6,sp)
    add('step6c','ridge fingerprint only',-1,'dpc50_marginal',r['marginal_median_ridge'],p6,sp)

out=os.path.join(RES,'step9_results_long.csv')
with open(out,'w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print('wrote %s  (%d rows)' % (out,len(rows)))
from collections import Counter
for k,v in sorted(Counter((r['experiment'],r['metric']) for r in rows).items()):
    print('  %-8s %-22s %d' % (k[0],k[1],v))
