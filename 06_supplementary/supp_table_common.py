#!/usr/bin/env python3
"""
supp_table_common.py - helpers shared by the scripts that build the supplementary tables (make_supp_table_sN.py).

Purpose
-------
Every supplementary table is assembled from committed records by its own script. All of those scripts must behave
the same way on four points, so the code for them lives here once:
  1. read committed records only (tracked by git, identical to HEAD, and with the sha256 the table was designed on);
  2. never overwrite a table that exists;
  3. format every number once, from full precision, in one style;
  4. write the two output files together or not at all, and print their sha256.
This file builds nothing by itself; it is imported by the table scripts and must sit in the same folder.

Logic
-----
- guard(inputs, table_id): stops unless each input is tracked, identical to HEAD and has the recorded sha256, and
  unless neither output file of the table exists.
- num(), interval(), triple(): number formatting. Negative numbers carry the typographic minus; a value that
  rounds to zero is printed without a sign; an interval is marked with * when it excludes zero.
- close(), same(): checks. A failed check stops the script with a message that names the check.
- table(): one Markdown table from a header and rows.
- write_outputs(table_id, long_rows, md_lines): writes supplementary_tables/supp_table_<id>.csv (long format, full
  precision) and .md (caption, tables, sources). Both are first written under temporary names and renamed only
  when both are complete; if anything fails (for example a full disk quota) the temporary files are removed, so a
  failed run leaves nothing behind. The Markdown is printed, then the sha256 of both files. Nothing in the outputs
  depends on the date, the machine or the commit, so a run elsewhere on the same records gives the same sha256.

Standard library only (no numpy, scipy or anndata), so every table script runs on the login node.
"""
import csv
import hashlib
import json
import os
import subprocess
import sys

OUT_DIR = 'supplementary_tables'
SPLITS = ['drug_split_1', 'drug_split_0', 'drug_split_2']          # the paper's order: development first
SPLIT_HEAD = ['Split 1 (development)', 'Split 0', 'Split 2']
DASH = '—'                                                          # a cell with no value


def die(msg):
    sys.exit('FATAL: ' + msg)


def git(*args):
    """Run one git command in the worktree; return (exit code, stripped stdout)."""
    p = subprocess.run(['git'] + list(args), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                       universal_newlines=True)
    return p.returncode, p.stdout.strip()


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def out_paths(table_id):
    """('supplementary_tables/supp_table_S5.csv', '.../supp_table_S5.md') for table_id 'S5'."""
    base = os.path.join(OUT_DIR, 'supp_table_%s' % table_id)
    return base + '.csv', base + '.md'


def guard(inputs, table_id):
    """inputs: {name: (path, sha256)}. Every input tracked, identical to HEAD and with its recorded sha256;
    neither output of the table present (outputs are never overwritten)."""
    for name, (path, digest) in inputs.items():
        if not os.path.isfile(path):
            die('%s not found; run this script from the research worktree root' % path)
        if git('ls-files', '--error-unmatch', path)[0] != 0:
            die('%s is not tracked by git' % path)
        if git('diff', '--quiet', 'HEAD', '--', path)[0] != 0:
            die('%s differs from HEAD; this script reads committed records only' % path)
        found = sha256_of(path)
        if found != digest:
            die('%s is not the record this table was designed on (sha256 %s, expected %s)' % (path, found, digest))
    for path in out_paths(table_id):
        if os.path.exists(path):
            die('%s exists; outputs are never overwritten' % path)


def load_json(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def load_csv(path):
    with open(path, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def is_number(x):
    """True for a real, finite-or-not float/int that is not NaN and not None."""
    return isinstance(x, (int, float)) and not isinstance(x, bool) and x == x


def num(x, dp=3, sign=False):
    """Format once from full precision, with the typographic minus. With sign=True a positive value carries '+'.
    A value that rounds to zero at this precision is printed without a sign (e.g. -0.00013 -> '0.000').
    None or NaN gives the dash."""
    if not is_number(x):
        return DASH
    s = '%.*f' % (dp, x)
    if float(s) == 0.0:
        return '%.*f' % (dp, 0.0)
    if sign and x > 0:
        s = '+' + s
    return s.replace('-', '−')


def excludes_zero(lo, hi):
    return lo > 0 or hi < 0


def interval(diff, lo, hi, dp=3):
    """'+0.196 [+0.184, +0.207]*' with * when the interval excludes zero; the dash when there is no value."""
    if not (is_number(diff) and is_number(lo) and is_number(hi)):
        return DASH
    return '%s [%s, %s]%s' % (num(diff, dp, True), num(lo, dp, True), num(hi, dp, True),
                              '*' if excludes_zero(lo, hi) else '')


def triple(values, dp=3):
    """'0.268 / 0.285 / 0.291' for the three splits; a single dash when no split has a value."""
    if not any(is_number(v) for v in values):
        return DASH
    return ' / '.join(num(v, dp) for v in values)


def close(a, b, tol, what):
    if not (is_number(a) and is_number(b)) or not (abs(a - b) <= tol):
        die('%s: %r does not reproduce %r (tolerance %g)' % (what, a, b, tol))


def same(a, b, what):
    if a != b:
        die('%s: %r is not %r' % (what, a, b))


def table(header, rows):
    """Markdown lines of one table."""
    lines = ['| ' + ' | '.join(header) + ' |', '|' + '---|' * len(header)]
    for r in rows:
        if len(r) != len(header):
            die('table row has %d cells, header has %d: %r' % (len(r), len(header), r))
        lines.append('| ' + ' | '.join(r) + ' |')
    return lines


def csv_value(v):
    """Full precision for numbers, empty for a missing value, text as it is."""
    if v is None or (isinstance(v, float) and v != v):
        return ''
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, float):
        return repr(v)
    return v


def write_outputs(table_id, long_rows, md_lines):
    """Write the .csv (long format) and the .md of one table, both or neither; print the Markdown and the sha256."""
    out_csv, out_md = out_paths(table_id)
    fields = list(long_rows[0])
    for r in long_rows:
        if list(r) != fields:
            die('long-format rows of table %s do not all have the same columns' % table_id)
    tmp_csv, tmp_md = out_csv + '.tmp', out_md + '.tmp'
    try:
        os.makedirs(OUT_DIR, exist_ok=True)
        with open(tmp_csv, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=fields, lineterminator='\n')
            w.writeheader()
            for r in long_rows:
                w.writerow(dict((k, csv_value(v)) for k, v in r.items()))
        with open(tmp_md, 'w', encoding='utf-8') as f:
            f.write('\n'.join(md_lines) + '\n')
        os.replace(tmp_csv, out_csv)
        os.replace(tmp_md, out_md)
    except OSError as e:
        for p in (tmp_csv, tmp_md, out_csv, out_md):          # leave nothing half-written behind
            try:
                os.remove(p)
            except OSError:
                pass
        die('could not write table %s (%s). If this says "Disk quota exceeded", free files on /scratch '
            '(myquota) and run again; nothing was left behind.' % (table_id, e))
    print('\n'.join(md_lines))
    print('')
    print('[written] %s  sha256 %s' % (out_csv, sha256_of(out_csv)))
    print('[written] %s  sha256 %s' % (out_md, sha256_of(out_md)))
