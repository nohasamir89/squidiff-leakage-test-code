"""Extract Step 10 values to long-format CSV, one row per plotted point.

Nothing typed by hand; each row carries its source file and SHA-256.
"""
import json, os, sys, csv, hashlib
def sha(p):
    m=hashlib.sha256()
    with open(p,'rb') as f:
        for c in iter(lambda: f.read(1<<20), b''): m.update(c)
    return m.hexdigest()
RES='leakage_results'; rows=[]
def add(exp, arm, metric, value, src, split='drug_split_1'):
    rows.append(dict(experiment=exp, split=split, arm=arm, metric=metric,
                     value=float(value), source_file=os.path.basename(src),
                     source_sha256=sha(src)[:16]))

p=os.path.join(RES,'step10_analysis.json')
if not os.path.isfile(p): sys.exit('FATAL: missing '+p)
d=json.load(open(p))
for arm,v in d['dpc_marginal'].items(): add('step10', arm, 'dpc50_marginal', v, p)
for arm,vals in d['de_overlap'].items():
    add('step10', arm, 'de_overlap', sum(vals)/len(vals), p)
for arm,vals in d['energy'].items():
    add('step10', arm, 'energy_distance', sum(vals)/len(vals), p)
for lab,c in d['contrasts'].items():
    tag='B_minus_A' if lab.startswith('B') else 'C_minus_A'
    add('step10', tag, 'paired_median', c['paired_median'], p)
    add('step10', tag+'_lo', 'paired_ci', c['ci'][0], p)
    add('step10', tag+'_hi', 'paired_ci', c['ci'][1], p)

# the ridge diagnostic that motivated the step
q=os.path.join(RES,'chemicalvae_comparison.json')
if os.path.isfile(q):
    e=json.load(open(q))
    for sp,r in e['dpc50_by_split'].items():
        for k,lab in (('chemicalvae','ridge ChemicalVAE'),('fingerprint','ridge fingerprint'),
                      ('pathway','ridge pathway')):
            add('ridge', lab, 'dpc50', r[k], q, sp)
out=os.path.join(RES,'step10_results_long.csv')
with open(out,'w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print('wrote %s  (%d rows)' % (out,len(rows)))
from collections import Counter
for k,v in sorted(Counter((r['experiment'],r['metric']) for r in rows).items()):
    print('  %-8s %-22s %d' % (k[0],k[1],v))
