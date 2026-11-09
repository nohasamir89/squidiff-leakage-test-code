#!/usr/bin/env python3
"""
make_supp_table_s15.py - Supplementary Table S15 (revised): the discrimination score on Tahoe-100M (Results 5, Fig. 6).

Purpose
-------
The Results (Section 5) and the Fig. 6 caption cite Supplementary Table S15 for every discrimination-score value:
the published cross-plate scores of the models and of the linear references, the same arms within a plate, the
alignment with the plate's vehicle offset, the pre-registered contrasts, and the shares quoted in the text ("about
one eighth" left within a plate; the plate accounting for about 88 % of the rewired model's advantage). This script
assembles that table from the committed records, so that every number in it traces to a record and is rounded once,
from full precision. It replaces the Step 15c version of S15 (published construction only).

Three parts:
  a  score per arm, mean over conditions, under four readings: the published cross-plate construction (S-orig), the
     same candidates with predicted and measured changes taken against disjoint vehicle halves (S-split), the
     within-plate construction (S-split), and the cosine of the arm's change with the plate's vehicle offset (S-split);
  b  the pre-registered contrasts with their 97.5 % structure-clustered bootstrap intervals (* = excludes zero);
  c  layout facts and the shares quoted in the text (point estimates).

Logic
-----
- Inputs (committed records only; the script stops unless each is tracked by git and identical to HEAD):
    tahoe_results/step15c_pds.json               Step 15c contrasts (per_split/<split>/contrasts/<name>: diff, ci)
    tahoe_results/step15c_pds_per_condition.csv  Step 15c per-condition PDS (<arm>_pds) -> arm means
    tahoe_results/step25_swap_summary.csv        Step 25 fingerprint-swap contrasts
    tahoe_results/step26_source_summary.csv      Step 26 values (<arm>__<scheme>_pds) and contrasts
    tahoe_results/step26_source.json             Step 26 plate structure per split
    tahoe_results/step27_within_plate_summary.csv Step 27 within-plate values, plate-offset cosines and contrasts
- Reproduction checks, all before anything is written (a failure stops the script):
    1. Step 15c internal: for every Step 15c arm with a "<arm> - 0.5" contrast, the per-condition mean minus 0.5
       equals the contrast's point estimate in step15c_pds.json (1e-9).
    2. Across records: the Step 15c means of the rewired and original models equal Step 26's published-construction
       values (1e-9; Step 26 reproduced the Step 15c per-condition scores before scoring anything new).
    3. Against the compiled results table of 5 Nov 2026 (results_table_steps24_27.md, itself built from these
       records): every part-a value of the Step 26/27 arms to its 4 printed decimals (5e-5), the 19 contrasts that
       table lists to their 3 printed decimals (5e-4), and the part-c shares, layout facts and shared-estimate effect
       to their printed precision. The EXPECTED blocks below are those printed values; they are checks only, never
       inputs to a computation.
- Rounding: every value is printed with 3 decimals (shares as % with 1 decimal; the shared-estimate effect with 4),
  rounded once from the full-precision record value. Negative numbers use the typographic minus.
- Outputs (never overwritten): supplementary_tables/supp_table_S15.csv (long format: part, row, split, value, lo,
  hi, excludes_zero, source record, source key) and supplementary_tables/supp_table_S15.md (the three tables in
  Markdown); the Markdown is also printed. Reads JSON/CSV only (no numpy/scipy/anndata), so it runs on the login node.

Run from the research worktree root:
    python make_supp_table_s15.py
"""
import csv
import json
import math
import os
import subprocess
import sys

# ---------------------------------------------------------------------------------------------------------------
# Inputs, outputs, fixed settings
# ---------------------------------------------------------------------------------------------------------------
REC = 'tahoe_results'
IN = dict(
    s15c_json=os.path.join(REC, 'step15c_pds.json'),
    s15c_pc=os.path.join(REC, 'step15c_pds_per_condition.csv'),
    s25=os.path.join(REC, 'step25_swap_summary.csv'),
    s26=os.path.join(REC, 'step26_source_summary.csv'),
    s26_json=os.path.join(REC, 'step26_source.json'),
    s27=os.path.join(REC, 'step27_within_plate_summary.csv'),
)
OUT_DIR = 'supplementary_tables'
OUT_CSV = os.path.join(OUT_DIR, 'supp_table_S15.csv')
OUT_MD = os.path.join(OUT_DIR, 'supp_table_S15.md')

SPLITS = ['drug_split_1', 'drug_split_0', 'drug_split_2']   # the paper's order: development first, then 0 and 2
SPLIT_HEAD = ['Split 1 (development)', 'Split 0', 'Split 2']
TOL_EXACT = 1e-9          # checks 1-2: same quantity read from two records
TOL_4DP = 5e-5 + 1e-12    # check 3: half a unit in the 4th printed decimal
TOL_3DP = 5e-4 + 1e-12    # check 3: half a unit in the 3rd printed decimal
TOL_PCT = 0.05 + 1e-9     # check 3: shares printed as % with one decimal

# ---------------------------------------------------------------------------------------------------------------
# Check values: printed in results_table_steps24_27.md (5 Nov 2026), parsed from that file, not typed.
# Part a: arm -> split -> (S-orig cross-plate, S-split cross-plate, S-split within plate, plate-offset cosine S-split)
# ---------------------------------------------------------------------------------------------------------------
EXPECTED_T1 = {
    'recipe': {'drug_split_1': (0.7223, 0.7181, 0.5030, 0.6277), 'drug_split_0': (0.7181, 0.7132, 0.5172, 0.6352), 'drug_split_2': (0.7142, 0.7106, 0.5264, 0.6235)},
    'recipe_fpshuf': {'drug_split_1': (0.7160, 0.7123, 0.4886, 0.6141), 'drug_split_0': (0.7012, 0.6968, 0.4906, 0.6364), 'drug_split_2': (0.6907, 0.6868, 0.4859, 0.6074)},
    'recipe_fpzero': {'drug_split_1': (0.7360, 0.7316, 0.4969, 0.7224), 'drug_split_0': (0.7180, 0.7132, 0.4983, 0.7122), 'drug_split_2': (0.7144, 0.7108, 0.4972, 0.7006)},
    'base': {'drug_split_1': (0.6415, 0.6392, 0.4934, 0.2508), 'drug_split_0': (0.6444, 0.6419, 0.5078, 0.2589), 'drug_split_2': (0.6549, 0.6528, 0.5200, 0.2636)},
    'base_fpshuf': {'drug_split_1': (0.6426, 0.6402, 0.4961, 0.2488), 'drug_split_0': (0.6360, 0.6338, 0.5007, 0.2588), 'drug_split_2': (0.6441, 0.6421, 0.4994, 0.2617)},
    'base_fpzero': {'drug_split_1': (0.6431, 0.6408, 0.4970, 0.2458), 'drug_split_0': (0.6366, 0.6342, 0.4986, 0.2610), 'drug_split_2': (0.6460, 0.6437, 0.5022, 0.2613)},
    'recon_recipe': {'drug_split_1': (0.7368, 0.7329, 0.5245, 0.6261), 'drug_split_0': (0.7304, 0.7255, 0.5314, 0.6336), 'drug_split_2': (0.7262, 0.7225, 0.5384, 0.6198)},
    'recon_base': {'drug_split_1': (0.6564, 0.6540, 0.5049, 0.2597), 'drug_split_0': (0.6591, 0.6567, 0.5210, 0.2674), 'drug_split_2': (0.6688, 0.6664, 0.5295, 0.2706)},
    'ctrl_line': {'drug_split_1': (0.7513, 0.7475, 0.5000, 0.9817), 'drug_split_0': (0.7335, 0.7284, 0.5000, 0.9830), 'drug_split_2': (0.7321, 0.7292, 0.4999, 0.9807)},
    'ctrl_sample': {'drug_split_1': (0.4992, 0.4847, 0.5024, -0.0209), 'drug_split_0': (0.5023, 0.4829, 0.5049, -0.0067), 'drug_split_2': (0.4947, 0.4785, 0.4938, -0.0166)},
    'duplicate': {'drug_split_1': (0.9531, 0.9512, 0.8836, 0.3134), 'drug_split_0': (0.9622, 0.9600, 0.9095, 0.2771), 'drug_split_2': (0.9530, 0.9500, 0.8911, 0.2944)},
    'true': {'drug_split_1': (None, None, None, 0.3642), 'drug_split_0': (None, None, None, 0.3195), 'drug_split_2': (None, None, None, 0.3381)},
    'ctrl_sample_h2': {'drug_split_1': (None, None, None, -0.0029), 'drug_split_0': (None, None, None, 0.0005), 'drug_split_2': (None, None, None, 0.0019)},
}
# Part b: (record, contrast name) -> split -> (point, lo, hi), as printed with 3 decimals in that table's Table 2.
EXPECTED_T2 = {
    ('s25', 'recipe_Q1_own_minus_shuf'): {'drug_split_1': (+0.006, -0.015, +0.029), 'drug_split_0': (+0.017, -0.004, +0.041), 'drug_split_2': (+0.024, -0.003, +0.053)},
    ('s25', 'recipe_Q2_shuf_minus_0.5'): {'drug_split_1': (+0.216, +0.186, +0.246), 'drug_split_0': (+0.201, +0.157, +0.239), 'drug_split_2': (+0.191, +0.156, +0.225)},
    ('s25', 'base_Q1_own_minus_shuf'): {'drug_split_1': (-0.001, -0.009, +0.007), 'drug_split_0': (+0.008, -0.004, +0.023), 'drug_split_2': (+0.011, -0.001, +0.025)},
    ('s25', 'Q3_gap_own_minus_shuf'): {'drug_split_1': (+0.007, -0.012, +0.027), 'drug_split_0': (+0.009, -0.008, +0.026), 'drug_split_2': (+0.013, -0.009, +0.035)},
    ('s25', 'ref_gap_own'): {'drug_split_1': (+0.081, +0.036, +0.124), 'drug_split_0': (+0.074, +0.018, +0.124), 'drug_split_2': (+0.059, +0.017, +0.103)},
    ('s26', 'G1_recipe_split_minus_0.5'): {'drug_split_1': (+0.218, +0.184, +0.250), 'drug_split_0': (+0.213, +0.172, +0.253), 'drug_split_2': (+0.211, +0.180, +0.240)},
    ('s26', 'G3_split_recipe_minus_fpshuf'): {'drug_split_1': (+0.006, -0.016, +0.028), 'drug_split_0': (+0.016, -0.005, +0.040), 'drug_split_2': (+0.024, -0.003, +0.053)},
    ('s26', 'G4_ctrl_line_orig_minus_0.5'): {'drug_split_1': (+0.251, +0.224, +0.279), 'drug_split_0': (+0.233, +0.202, +0.262), 'drug_split_2': (+0.232, +0.201, +0.262)},
    ('s26', 'G4b_ctrl_sample_orig_minus_0.5'): {'drug_split_1': (-0.001, -0.011, +0.009), 'drug_split_0': (+0.002, -0.010, +0.015), 'drug_split_2': (-0.005, -0.020, +0.009)},
    ('s26', 'E_orig_recipe_minus_ctrl_line'): {'drug_split_1': (-0.029, -0.054, -0.005), 'drug_split_0': (-0.015, -0.040, +0.009), 'drug_split_2': (-0.018, -0.044, +0.007)},
    ('s27', 'H1_recipe_split_minus_0.5'): {'drug_split_1': (+0.003, -0.041, +0.048), 'drug_split_0': (+0.017, -0.039, +0.071), 'drug_split_2': (+0.026, -0.021, +0.076)},
    ('s27', 'H2_split_recipe_minus_base'): {'drug_split_1': (+0.010, -0.053, +0.069), 'drug_split_0': (+0.009, -0.072, +0.082), 'drug_split_2': (+0.006, -0.054, +0.065)},
    ('s27', 'H3_split_recipe_minus_fpshuf'): {'drug_split_1': (+0.014, -0.016, +0.042), 'drug_split_0': (+0.027, -0.001, +0.058), 'drug_split_2': (+0.041, +0.005, +0.082)},
    ('s27', 'H3f_flip_recipe_minus_fpshuf'): {'drug_split_1': (+0.015, -0.015, +0.043), 'drug_split_0': (+0.029, +0.001, +0.062), 'drug_split_2': (+0.041, +0.007, +0.082)},
    ('s27', 'base_H3_split_own_minus_fpshuf'): {'drug_split_1': (-0.003, -0.012, +0.005), 'drug_split_0': (+0.007, -0.010, +0.028), 'drug_split_2': (+0.021, +0.005, +0.038)},
    ('s27', 'H4_ctrl_line_orig_minus_0.5'): {'drug_split_1': (+0.000, -0.039, +0.041), 'drug_split_0': (+0.000, -0.040, +0.041), 'drug_split_2': (-0.000, -0.044, +0.044)},
    ('s27', 'duplicate_split_minus_0.5'): {'drug_split_1': (+0.384, +0.367, +0.400), 'drug_split_0': (+0.410, +0.391, +0.426), 'drug_split_2': (+0.391, +0.372, +0.409)},
    ('s27', 'H5_residual_true_split'): {'drug_split_1': (+0.364, +0.318, +0.410), 'drug_split_0': (+0.320, +0.279, +0.357), 'drug_split_2': (+0.338, +0.291, +0.381)},
    ('s27', 'H5null_residual_ctrl_h2_split'): {'drug_split_1': (-0.003, -0.011, +0.006), 'drug_split_0': (+0.000, -0.007, +0.008), 'drug_split_2': (+0.002, -0.007, +0.011)},
}
# Part c, as printed in that table's Table 3 (and its header for the within-plate coverage).
EXPECTED_T3 = {
    'share_E_pct': {'drug_split_1': 113.0, 'drug_split_0': 107.0, 'drug_split_2': 108.3},
    'share_I_pct': {'drug_split_1': 11.8, 'drug_split_0': 12.7, 'drug_split_2': 10.9},
    'shared_estimate': {'drug_split_1': 0.0042, 'drug_split_0': 0.0049, 'drug_split_2': 0.0036},
    'plates_min_median_max': {'drug_split_1': (3, 5, 5), 'drug_split_0': (3, 5, 12), 'drug_split_2': (3, 5, 5)},
    'same_plate_fraction': {'drug_split_1': 0.2177, 'drug_split_0': 0.2137, 'drug_split_2': 0.2234},
    'covered_of_total': {'drug_split_1': (2645, 2899), 'drug_split_0': (2678, 3115), 'drug_split_2': (2532, 2853)},
}

# ---------------------------------------------------------------------------------------------------------------
# Table layout. Part a rows: (label, Step 26/27 arm key or None, Step 15c arm key or None)
# ---------------------------------------------------------------------------------------------------------------
ROWS_A = [
    ('Rewired model', 'recipe', None),
    ('Rewired model, fingerprint of another held-out compound', 'recipe_fpshuf', None),
    ('Rewired model, no fingerprint', 'recipe_fpzero', None),
    ('Rewired model, encoder given the measured treated cells', 'recon_recipe', None),
    ('Original model', 'base', None),
    ('Original model, fingerprint of another held-out compound', 'base_fpshuf', None),
    ('Original model, no fingerprint', 'base_fpzero', None),
    ('Original model, encoder given the measured treated cells', 'recon_base', None),
    ('Rewired model with classifier-free guidance (weight 2)', None, 'guidance_w2'),
    ('Rewired model with cross-attention', None, 'attn_cross'),
    ('Rewired model, parameter-matched control without attention', None, 'mlp_ctrl'),
    ('Training-set ridge', None, 'ridge_train'),
    ('Cross-validated ridge', None, 'ridge_S7'),
    ('Compound-blind average', None, 'mean_response'),
    ('Cell-line vehicle average (no model)', 'ctrl_line', None),
    ("Resample of the condition's vehicle pool (no model)", 'ctrl_sample', None),
    ('Second draw of the measured cells (ceiling)', 'duplicate', None),
    ('Measured change of the real treated cells', 'true', None),            # plate-offset cosine only
    ('Vehicle resample from the measured-change half (null)', 'ctrl_sample_h2', None),  # plate-offset cosine only
]
COS_ONLY = ('true', 'ctrl_sample_h2')
HEAD_A = ['Arm', 'Cross-plate, published (S-orig)', 'Cross-plate, vehicle halves (S-split)',
          'Within plate (S-split)', 'Cosine with the plate offset (S-split)']

# Part b rows: a group header is (label, None, None); a contrast is (label, record, name).
ROWS_B = [
    ('Published construction (cross-plate, S-orig)', None, None),
    ('Rewired model − 0.5', 's15c', 'Q1_primary_pds_recipe_minus_0.5'),
    ('Original model − 0.5', 's15c', 'secondary_pds_base_minus_0.5'),
    ('Training-set ridge − 0.5', 's15c', 'secondary_pds_ridge_train_minus_0.5'),
    ('Cross-validated ridge − 0.5', 's15c', 'secondary_pds_ridge_S7_minus_0.5'),
    ('Compound-blind average − 0.5', 's15c', 'secondary_pds_mean_response_minus_0.5'),
    ('Rewired model − training-set ridge', 's15c', 'Q2_coprimary_pds_recipe_minus_ridge_train'),
    ('Rewired model − cross-validated ridge', 's15c', 'secondary_pds_recipe_minus_ridge_S7'),
    ('Rewired model with guidance − 0.5', 's15c', 'secondary_pds_guidance_w2_minus_0.5'),
    ('Rewired model with guidance − rewired model', 's15c', 'secondary_pds_guidance_w2_minus_recipe'),
    ('Rewired model with cross-attention − 0.5', 's15c', 'secondary_pds_attn_cross_minus_0.5'),
    ('Parameter-matched control − 0.5', 's15c', 'secondary_pds_mlp_ctrl_minus_0.5'),
    ('Cross-attention − parameter-matched control (exploratory)', 's15c', 'exploratory_pds_attn_cross_minus_mlp_ctrl'),
    ('Rewired model, another compound’s fingerprint − 0.5', 's25', 'recipe_Q2_shuf_minus_0.5'),
    ('Rewired model, no fingerprint − 0.5', 's25', 'recipe_ref_zero_minus_0.5'),
    ('Rewired model: own − another compound’s fingerprint', 's25', 'recipe_Q1_own_minus_shuf'),
    ('Rewired model: own − no fingerprint', 's25', 'recipe_ref_own_minus_zero'),
    ('Original model, another compound’s fingerprint − 0.5', 's25', 'base_Q2_shuf_minus_0.5'),
    ('Original model: own − another compound’s fingerprint', 's25', 'base_Q1_own_minus_shuf'),
    ('Rewired model − original model', 's25', 'ref_gap_own'),
    ('(Rewired − original): own − another compound’s fingerprint', 's25', 'Q3_gap_own_minus_shuf'),
    ('Cell-line vehicle average − 0.5', 's26', 'G4_ctrl_line_orig_minus_0.5'),
    ('Vehicle resample − 0.5', 's26', 'G4b_ctrl_sample_orig_minus_0.5'),
    ('Rewired model − cell-line vehicle average', 's26', 'E_orig_recipe_minus_ctrl_line'),
    ('Cross-plate, disjoint vehicle halves (S-split)', None, None),
    ('Rewired model − 0.5', 's26', 'G1_recipe_split_minus_0.5'),
    ('Rewired model, another compound’s fingerprint − 0.5', 's26', 'G2_fpshuf_split_minus_0.5'),
    ('Rewired model: own − another compound’s fingerprint', 's26', 'G3_split_recipe_minus_fpshuf'),
    ('Original model − 0.5', 's26', 'base_G1_split_minus_0.5'),
    ('Within plate (S-split unless stated)', None, None),
    ('Rewired model − 0.5', 's27', 'H1_recipe_split_minus_0.5'),
    ('Original model − 0.5', 's27', 'base_H1_split_minus_0.5'),
    ('Rewired model, no fingerprint − 0.5', 's27', 'fpzero_split_minus_0.5'),
    ('Rewired model − original model', 's27', 'H2_split_recipe_minus_base'),
    ('Rewired model: own − another compound’s fingerprint', 's27', 'H3_split_recipe_minus_fpshuf'),
    ('Rewired model: own − another compound’s fingerprint (S-flip)', 's27', 'H3f_flip_recipe_minus_fpshuf'),
    ('Original model: own − another compound’s fingerprint', 's27', 'base_H3_split_own_minus_fpshuf'),
    ('Cell-line vehicle average − 0.5 (S-orig)', 's27', 'H4_ctrl_line_orig_minus_0.5'),
    ('Vehicle resample − 0.5', 's27', 'ctrl_sample_split_minus_0.5'),
    ('Second draw of the measured cells − 0.5 (ceiling)', 's27', 'duplicate_split_minus_0.5'),
    ('Alignment with the plate offset (cosine, S-split)', None, None),
    ('Measured change of the real treated cells', 's27', 'H5_residual_true_split'),
    ('Vehicle resample from the measured-change half (null)', 's27', 'H5null_residual_ctrl_h2_split'),
]


# ---------------------------------------------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------------------------------------------
def die(msg):
    sys.exit('FATAL: ' + msg)


def git(*args):
    """Run one git command in the worktree; return (exit code, stripped stdout)."""
    p = subprocess.run(['git'] + list(args), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                       universal_newlines=True)
    return p.returncode, p.stdout.strip()


def guard():
    """Every input tracked and identical to HEAD; no output present (outputs are never overwritten)."""
    for path in IN.values():
        if git('ls-files', '--error-unmatch', path)[0] != 0:
            die('%s is not tracked by git' % path)
        if git('diff', '--quiet', 'HEAD', '--', path)[0] != 0:
            die('%s differs from HEAD; this script reads committed records only' % path)
    for path in (OUT_CSV, OUT_MD):
        if os.path.exists(path):
            die('%s exists; outputs are never overwritten' % path)


def load_summary(path):
    """A Step 25/26/27 summary CSV -> values, contrasts and condition counts, keyed by (split, name)."""
    vals, cons, ncond = {}, {}, {}
    for r in csv.DictReader(open(path)):
        key = (r['split'], r['name'])
        if r['kind'] == 'value':
            vals[key] = float(r['value'])
            ncond[key] = int(float(r['n_conditions']))
        elif r['kind'] == 'contrast':
            cons[key] = (float(r['value']), float(r['lo']), float(r['hi']))
    return vals, cons, ncond


def find_split_block(obj):
    """Locate, anywhere in the Step 26 JSON, the mapping split -> results that carries 'plate_structure'."""
    if isinstance(obj, dict):
        if all(isinstance(obj.get(s), dict) and 'plate_structure' in obj[s] for s in SPLITS):
            return obj
        for v in obj.values():
            found = find_split_block(v)
            if found is not None:
                return found
    return None


def num(x, dp=3, sign=False):
    """Format once from full precision, with the typographic minus. With sign=True a positive value carries '+'.
    A value that rounds to zero at this precision is printed without a sign (e.g. -0.00013 -> '0.000')."""
    s = '%.*f' % (dp, x)
    if float(s) == 0.0:
        return '%.*f' % (dp, 0.0)
    if sign and x > 0:
        s = '+' + s
    return s.replace('-', '−')


def interval(c):
    """'+0.222 [+0.188, +0.254]*' with * when the interval excludes zero."""
    v, lo, hi = c
    star = '*' if (lo > 0 or hi < 0) else ''
    return '%s [%s, %s]%s' % (num(v, sign=True), num(lo, sign=True), num(hi, sign=True), star)


def close(a, b, tol, what):
    if a is None or b is None or not (abs(a - b) <= tol):
        die('%s: %r does not reproduce %r (tolerance %g)' % (what, a, b, tol))


# ---------------------------------------------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------------------------------------------
def main():
    guard()
    _, commit = git('rev-parse', '--short', 'HEAD')

    # ---- read every record
    j15 = json.load(open(IN['s15c_json']))
    pc15 = list(csv.DictReader(open(IN['s15c_pc'])))
    v26, c26, n26 = load_summary(IN['s26'])
    v27, c27, n27 = load_summary(IN['s27'])
    _, c25, _ = load_summary(IN['s25'])
    blk26 = find_split_block(json.load(open(IN['s26_json'])))
    if blk26 is None:
        die('no per-split plate_structure found in %s' % IN['s26_json'])
    cons = {'s15c': {}, 's25': c25, 's26': c26, 's27': c27}
    for s in SPLITS:
        for name, c in j15['per_split'][s]['contrasts'].items():
            cons['s15c'][(s, name)] = (float(c['diff']), float(c['ci'][0]), float(c['ci'][1]))

    def mean15(split, arm):
        """Step 15c arm mean over the split's conditions (finite per-condition PDS only)."""
        col = '%s_pds' % arm
        if pc15 and col not in pc15[0]:
            die('column %s missing from %s' % (col, IN['s15c_pc']))
        xs = [float(r[col]) for r in pc15 if r['split'] == split and r[col] not in ('', 'nan', 'NaN')]
        xs = [x for x in xs if math.isfinite(x)]
        if not xs:
            die('no finite %s values on %s' % (col, split))
        return sum(xs) / len(xs)

    def con(src, name, split):
        c = cons[src].get((split, name))
        if c is None:
            die('contrast %s missing from the %s record on %s' % (name, src, split))
        return c

    # ---- check 1: Step 15c means minus 0.5 equal the Step 15c contrast points
    zero_name = {'recipe': 'Q1_primary_pds_recipe_minus_0.5'}
    for arm in ('recipe', 'base', 'ridge_train', 'ridge_S7', 'mean_response', 'guidance_w2', 'attn_cross', 'mlp_ctrl'):
        name = zero_name.get(arm, 'secondary_pds_%s_minus_0.5' % arm)
        for s in SPLITS:
            close(mean15(s, arm) - 0.5, con('s15c', name, s)[0], TOL_EXACT, 'check 1, %s %s' % (s, name))
    # ---- check 2: Step 15c and Step 26 agree on the published-construction scores of both models
    for arm in ('recipe', 'base'):
        for s in SPLITS:
            close(mean15(s, arm), v26.get((s, '%s__orig_pds' % arm)), TOL_EXACT, 'check 2, %s %s' % (s, arm))

    # ---- part a values
    def a_values(key15, key, s):
        """(S-orig, S-split cross-plate, within plate S-split, plate-offset cosine S-split) for one arm and split."""
        if key15 is not None:
            return (mean15(s, key15), None, None, None)
        if key in COS_ONLY:
            return (None, None, None, v27.get((s, 'residual__%s__split' % key)))
        return (v26.get((s, '%s__orig_pds' % key)), v26.get((s, '%s__split_pds' % key)),
                v27.get((s, '%s__split_wp_pds' % key)), v27.get((s, 'residual__%s__split' % key)))

    # ---- check 3a: part a against the printed table
    for _, key, key15 in ROWS_A:
        if key is None:
            continue
        for s in SPLITS:
            got = a_values(key15, key, s)
            for i, exp in enumerate(EXPECTED_T1[key][s]):
                if exp is not None:
                    close(got[i], exp, TOL_4DP, 'check 3a, %s %s column %d' % (s, key, i + 1))
    # ---- check 3b: the 19 printed contrasts
    for (src, name), per in EXPECTED_T2.items():
        for s in SPLITS:
            got = con(src, name, s)
            for g, e in zip(got, per[s]):
                close(g, e, TOL_3DP, 'check 3b, %s %s' % (s, name))

    # ---- part c quantities (+ check 3c)
    part_c = {}
    for s in SPLITS:
        ps = blk26[s]['plate_structure']
        total = n26.get((s, 'recipe__orig_pds'))
        covered = n27.get((s, 'recipe__split_wp_pds'))
        share_e = 100.0 * (v26[(s, 'ctrl_line__orig_pds')] - 0.5) / (v26[(s, 'recipe__orig_pds')] - 0.5)
        share_i = 100.0 * con('s27', 'H2_split_recipe_minus_base', s)[0] / con('s25', 'ref_gap_own', s)[0]
        shared = v26[(s, 'recipe__orig_pds')] - v26[(s, 'recipe__split_pds')]
        mmm = (ps['plates_per_context_min'], ps['plates_per_context_median'], ps['plates_per_context_max'])
        frac = ps['candidate_pair_same_plate_fraction']
        e = EXPECTED_T3
        close(share_e, e['share_E_pct'][s], TOL_PCT, 'check 3c, %s share E' % s)
        close(share_i, e['share_I_pct'][s], TOL_PCT, 'check 3c, %s share I' % s)
        close(shared, e['shared_estimate'][s], TOL_4DP, 'check 3c, %s shared estimate' % s)
        close(frac, e['same_plate_fraction'][s], TOL_4DP, 'check 3c, %s same-plate fraction' % s)
        if tuple(mmm) != e['plates_min_median_max'][s] or (covered, total) != e['covered_of_total'][s]:
            die('check 3c, %s plates %s or coverage %s/%s differ from the printed table' % (s, mmm, covered, total))
        part_c[s] = dict(covered=covered, total=total, plates=mmm, frac=frac, share_e=share_e, share_i=share_i,
                         shared=shared)

    # ---- build the long CSV and the Markdown
    long_rows = []
    md = ['**Supplementary Table S15a** | Score per arm, mean over conditions (splits 1 / 0 / 2; 0.5 = chance).', '',
          '| ' + ' | '.join(HEAD_A) + ' |', '|' + '---|' * len(HEAD_A)]
    cols = ('S-orig cross-plate', 'S-split cross-plate', 'S-split within plate', 'plate-offset cosine S-split')
    for label, key, key15 in ROWS_A:
        cells = []
        per = {s: a_values(key15, key, s) for s in SPLITS}
        for i in range(4):
            vs = [per[s][i] for s in SPLITS]
            cells.append('—' if any(v is None for v in vs) else ' / '.join(num(v) for v in vs))
            for s in SPLITS:
                if per[s][i] is not None:
                    src = 'step15c_pds_per_condition.csv' if key15 else (
                        'step26_source_summary.csv' if i < 2 else 'step27_within_plate_summary.csv')
                    long_rows.append(dict(part='a', row=label, column=cols[i], split=s, value=per[s][i], lo='',
                                          hi='', excludes_zero='', source_record=src,
                                          source_key=key15 or key))
        md.append('| %s | %s |' % (label, ' | '.join(cells)))

    md += ['', '**Supplementary Table S15b** | Pre-registered contrasts, 97.5 % structure-clustered bootstrap '
               'intervals (* excludes zero).', '',
           '| Contrast | ' + ' | '.join(SPLIT_HEAD) + ' |', '|---|---|---|---|']
    srcname = {'s15c': 'step15c_pds.json', 's25': 'step25_swap_summary.csv', 's26': 'step26_source_summary.csv',
               's27': 'step27_within_plate_summary.csv'}
    for label, src, name in ROWS_B:
        if src is None:
            md.append('| **%s** | | | |' % label)
            continue
        cs = [con(src, name, s) for s in SPLITS]
        md.append('| %s | %s |' % (label, ' | '.join(interval(c) for c in cs)))
        for s, c in zip(SPLITS, cs):
            long_rows.append(dict(part='b', row=label, column='contrast', split=s, value=c[0], lo=c[1], hi=c[2],
                                  excludes_zero=int(c[1] > 0 or c[2] < 0), source_record=srcname[src],
                                  source_key=name))

    md += ['', '**Supplementary Table S15c** | Layout of the candidate sets and the shares quoted in the text '
               '(point estimates).', '',
           '| Quantity | ' + ' | '.join(SPLIT_HEAD) + ' |', '|---|---|---|---|']
    c_rows = [
        ('Conditions scored within a plate, of all held-out conditions',
         lambda d: '{:,} of {:,}'.format(d['covered'], d['total']), lambda d: d['covered']),
        ('Plates per cell line and dose: median (range)',
         lambda d: '%d (%d–%d)' % (d['plates'][1], d['plates'][0], d['plates'][2]), lambda d: d['plates'][1]),
        ("Share of candidate pairs measured on the condition's own plate",
         lambda d: num(d['frac']), lambda d: d['frac']),
        ("Cell-line vehicle average: distance from chance as a share of the rewired model's (S-orig)",
         lambda d: ('%.1f %%' % d['share_e']).replace('-', '−'), lambda d: d['share_e']),
        ('Rewired − original difference remaining within a plate, as a share of the cross-plate difference',
         lambda d: ('%.1f %%' % d['share_i']).replace('-', '−'), lambda d: d['share_i']),
        ('Shared vehicle estimate: rewired model, S-orig minus S-split (cross-plate)',
         lambda d: num(d['shared'], dp=4), lambda d: d['shared']),
    ]
    for label, fmt, raw in c_rows:
        md.append('| %s | %s |' % (label, ' | '.join(fmt(part_c[s]) for s in SPLITS)))
        for s in SPLITS:
            long_rows.append(dict(part='c', row=label, column='point', split=s, value=raw(part_c[s]), lo='', hi='',
                                  excludes_zero='', source_record='step26_source.json / summaries',
                                  source_key=label))
    md += ['', 'Sources: step15c_pds.json and step15c_pds_per_condition.csv (Step 15c), step25_swap_summary.csv '
               '(Step 25), step26_source_summary.csv and step26_source.json (Step 26), step27_within_plate_summary.csv '
               '(Step 27); research commit %s; assembled by make_supp_table_s15.py.' % commit]

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(OUT_CSV, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(long_rows[0]), lineterminator='\n')
        w.writeheader()
        w.writerows(long_rows)
    with open(OUT_MD, 'w') as f:
        f.write('\n'.join(md) + '\n')
    print('\n'.join(md))
    print('\n[checks] 1 (Step 15c means vs contrasts), 2 (Step 15c vs Step 26), 3 (results table: part a %d values, '
          'part b %d contrasts, part c) all passed' % (sum(v is not None for d in EXPECTED_T1.values() for t in d.values()
                                                          for v in t), 3 * len(EXPECTED_T2)))
    print('[written] %s, %s  (commit %s)' % (OUT_CSV, OUT_MD, commit))


if __name__ == '__main__':
    main()
