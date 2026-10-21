#!/usr/bin/env python
"""finalize_step16.py -- certify a finished Step 16 (Tahoe, target-aware encoding) training run; write its sidecar and
provenance. The Step 15 finaliser with the drug encoding passed through (PREREG_step16_tahoe_targets.md section 9).

What it checks for task T (0-17); every check must pass before anything is written
  1. decode T -> split, arm, seed exactly as train_step16_tahoe.sh does (split = T // 6, arm = (T % 6) // 3 with
     0 targets / 1 targets_shuf, seed = T % 3), cross-checked against the '[task] T=..' line the launcher printed in
     the run's .out log (and against --expect-* if given)
  2. the run's .out log contains the training 'end_time' line, the '[targets]' line of the encoder (its shuffled-order
     hash is recorded) and the '[step16]' parameter line; the .err log records no oom-kill event
  3. the checkpoint model_0.9999.pt loads STRICTLY into a model rebuilt from the launcher's exact flags (every key and
     shape must match); the parameters-only count equals the pre-registered rule
         58,178,180 (the recipe) + (drug_dimension - 1,024) x width(encoder.fc1)
     with drug_dimension = DRUG_ENCODER_DIMS[encoding] = 1,569; the first layer's input width equals
     gene_size + drug_dimension
  4. the sidecar (model_0.9999.pt.encoding.json) and provenance (provenance/s16tr_<TAG>.json) do not exist yet
Written on success: the sidecar and the provenance record, with the checkpoint sha256, the sha256 of the two encoder
inputs (the Step 15a data record and the curated-target table) and the training commit parsed from the run's own log.

Usage
  python -u finalize_step16.py --jobid $SLURM_ARRAY_JOB_ID --tasks $T \
         --expect-arm $ARM --expect-split $SPLIT --expect-seed $SEED              # from the launcher
  python -u finalize_step16.py --jobid <id> --tasks 6-11                           # back-fill completed runs
  --dry-run validates and prints, writes nothing.  Exit code 1 if any task fails.
"""
import argparse
import datetime
import hashlib
import json
import os
import re
import sys

W = '/scratch/ni2071/Diffusion/Squidiff_git/Squidiff'
D = '/scratch/ni2071/Diffusion/tahoe/datasets_step15'
PREREG = 'PREREG_step16_tahoe_targets.md'
CKPT_NAME = 'model_0.9999.pt'
BASE_PARAMS = 58178180                      # the recipe (Step 15b): adaLN + Gaussian flow matching, fingerprint, 200 genes
FP_WIDTH = 1024
RECORD = 'tahoe_results/step15_data_record.json'
TARGETS = 'tahoe_results/step15_targets_curated_hgnc.csv'

# Arm table: name, drug encoding. Must stay identical to the case-table in train_step16_tahoe.sh.
ARMS = [('targets', 'fp_targets'), ('targets_shuf', 'fp_targets_shuf')]
N_TASKS = 18


class Fail(Exception):
    """A validation failure for one task; the message says which check failed."""


def decode(T):
    """T -> (split index, arm index, seed): split = T // 6, arm = (T % 6) // 3, seed = T % 3 (as the launcher)."""
    if not 0 <= T < N_TASKS:
        raise Fail('task out of range 0-%d: %d' % (N_TASKS - 1, T))
    split_idx, within = divmod(T, 6)
    arm_idx, seed = divmod(within, 3)
    return split_idx, arm_idx, seed


def parse_tasks(spec):
    """'6-11,0' -> [6, 7, 8, 9, 10, 11, 0]."""
    out = []
    for part in spec.split(','):
        a, _, b = part.strip().partition('-')
        out.extend(range(int(a), int(b or a) + 1))
    return out


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 24), b''):
            h.update(chunk)
    return h.hexdigest()


def launcher_argv(split, seed, enc, outdir, logdir):
    """The exact argument list train_step16_tahoe.sh passes to train_squidiff.py (the recipe's flags + the encoding)."""
    train = '%s/tahoe_train_%s.h5ad' % (D, split)
    ctrl = '%s/tahoe_train_%s_control.h5ad' % (D, split)
    argv = ['--logger_path', logdir, '--data_path', train, '--control_data_path', ctrl,
            '--resume_checkpoint', outdir, '--seed', str(seed), '--use_drug_structure', 'True',
            '--drug_encoding', enc, '--gene_size', '200', '--output_dim', '200',
            '--lr_anneal_steps', '100000', '--use_adaln', 'True', '--fm_source', 'gaussian']
    return argv, train, ctrl


def build_model(argv, enc):
    """Parse the launcher's flags with train_squidiff's own parser, confirm each flag took effect, build the model on CPU
    and return it with the pre-registered parameter expectation."""
    import train_squidiff as t
    from Squidiff.script_util import create_model_and_diffusion, model_and_diffusion_defaults, args_to_dict
    from Squidiff.scrna_datasets import DRUG_ENCODER_DIMS
    saved = sys.argv
    sys.argv = ['train_squidiff.py'] + argv
    try:
        a = t.parse_args()
    finally:
        sys.argv = saved
    if a['use_adaln'] is not True:
        raise Fail('use_adaln resolved to %r, expected True' % a['use_adaln'])
    if a['fm_source'] != 'gaussian':
        raise Fail('fm_source %r, expected gaussian' % a['fm_source'])
    if not isinstance(a['cond_drop'], float) or abs(a['cond_drop']) > 1e-12:
        raise Fail('cond_drop %r, expected 0.0' % a['cond_drop'])
    if a['drug_encoding'] != enc:
        raise Fail('drug_encoding %r, expected %r' % (a['drug_encoding'], enc))
    if a.get('encoder_arm', 'mlp') != 'mlp':
        raise Fail('encoder_arm %r, expected mlp' % a.get('encoder_arm'))
    if int(a['gene_size']) != 200:
        raise Fail('gene_size %r, expected 200' % a['gene_size'])
    a['drug_dimension'] = DRUG_ENCODER_DIMS[enc]
    a['memory_size'] = 0
    model, _ = create_model_and_diffusion(**args_to_dict(a, model_and_diffusion_defaults().keys()))
    fc1 = model.encoder.fc1.weight
    if int(fc1.shape[1]) != 200 + a['drug_dimension']:
        raise Fail('encoder.fc1 reads %d inputs, expected gene_size 200 + drug_dimension %d' % (int(fc1.shape[1]), a['drug_dimension']))
    expected = BASE_PARAMS + (a['drug_dimension'] - FP_WIDTH) * int(fc1.shape[0])
    return model, a['drug_dimension'], int(fc1.shape[0]), expected


def load_checkpoint(path):
    import torch
    return torch.load(path, map_location='cpu', weights_only=True)


def read_log(out_log, err_log, T, split, name, seed):
    """Completion evidence from the run's own logs: the [task] line matches, 'end_time' was reached, the encoder and the
    parameter rule printed, no oom-kill."""
    if not os.path.isfile(out_log):
        raise Fail('missing log %s' % out_log)
    text = open(out_log).read()
    m = re.search(r'\[task\] T=(\d+)\s+split=(\S+)\s+arm=(\S+)\s+seed=(\d+)', text)
    if not m:
        raise Fail('no [task] line in %s' % out_log)
    if (int(m.group(1)), m.group(2), m.group(3), int(m.group(4))) != (T, split, name, seed):
        raise Fail('log says T=%s split=%s arm=%s seed=%s, expected T=%d %s %s %d'
                   % (m.group(1), m.group(2), m.group(3), m.group(4), T, split, name, seed))
    end_lines = [l for l in text.splitlines() if 'end_time' in l]
    if not end_lines:
        raise Fail('training did not reach end_time in %s' % out_log)
    if os.path.isfile(err_log) and 'oom-kill' in open(err_log).read():
        raise Fail('oom-kill recorded in %s' % err_log)
    tg = re.search(r'\[targets\] .*shuffled order sha256 ([0-9a-f]{12})', text)
    if not tg:
        raise Fail('no [targets] encoder line in %s' % out_log)
    s16 = re.search(r'\[step16\] drug_dimension=(\d+): expected parameters (\d+)', text)
    if not s16:
        raise Fail('no [step16] parameter line in %s' % out_log)
    c = re.search(r'\[commit\] ([0-9a-f]{7,40})', text)
    n = re.search(r'\[node\] (\S+)', text)
    return dict(commit=c.group(1) if c else None, node=n.group(1) if n else None, end_time_line=end_lines[-1].strip(),
                targets_line=tg.group(0).strip(), perm_sha256_prefix=tg.group(1),
                logged_drug_dimension=int(s16.group(1)), logged_expected=int(s16.group(2)))


def finalize(T, jobid, force, dry, expect):
    split_idx, arm_idx, seed = decode(T)
    split = 'drug_split_%d' % split_idx
    name, enc = ARMS[arm_idx]
    if expect is not None and expect != (name, split, seed):
        raise Fail('launcher expects arm=%s split=%s seed=%d, decode gives %s %s %d' % (expect + (name, split, seed)))
    tag = '%s_%s_seed_%d' % (split, name, seed)
    outdir = '%s/step16/%s/%s_seed_%d' % (W, name, split, seed)
    logdir = 'logger_files/logger_s16_%s' % tag
    out_log = '%s/logger_files/s16tr_%s_%d.out' % (W, jobid, T)
    err_log = '%s/logger_files/s16tr_%s_%d.err' % (W, jobid, T)
    ck = os.path.join(outdir, CKPT_NAME)
    side = ck + '.encoding.json'
    prov = '%s/provenance/s16tr_%s.json' % (W, tag)

    info = read_log(out_log, err_log, T, split, name, seed)                       # check 1 + 2
    if not os.path.isfile(ck) or os.path.getsize(ck) == 0:                        # check 3
        raise Fail('checkpoint missing or empty: %s' % ck)
    for q in (RECORD, TARGETS):
        if not os.path.isfile(q):
            raise Fail('missing encoder input %s' % q)
    argv, train, ctrl = launcher_argv(split, seed, enc, outdir, logdir)
    model, dd, fc1_w, exp_n = build_model(argv, enc)
    if info['logged_drug_dimension'] != dd or info['logged_expected'] != exp_n:
        raise Fail('the run logged drug_dimension %d / expected %d, the rule gives %d / %d'
                   % (info['logged_drug_dimension'], info['logged_expected'], dd, exp_n))
    sd = load_checkpoint(ck)
    try:
        model.load_state_dict(sd, strict=True)                                    # every key and shape must match
    except Exception as e:
        raise Fail('checkpoint does not load into the rebuilt %s model: %s' % (name, str(e)[:300]))
    n_params = int(sum(p.numel() for p in model.parameters()))
    n_sd = int(sum(int(v.numel()) for v in sd.values() if hasattr(v, 'numel')))
    if n_params != exp_n:
        raise Fail('parameter count %d != expected %d (= %d + (%d - %d) x %d) for arm %s' % (n_params, exp_n, BASE_PARAMS, dd, FP_WIDTH, fc1_w, name))
    if any(k.startswith('encoder.arm.') for k in sd):
        raise Fail('unexpected encoder.arm.* keys in the checkpoint (the Step 16 arms use the mlp encoder)')
    if not force and (os.path.exists(side) or os.path.exists(prov)):              # check 4
        raise Fail('sidecar or provenance already exists for %s (use --force to overwrite)' % tag)

    meta = dict(arm=name, split=split, seed=seed, encoder_arm='mlp', memory_size=0, use_adaln=True, fm_source='gaussian',
                cond_drop=0.0, drug_encoding=enc, drug_dimension=dd, fc1_width=fc1_w, gene_size=200,
                n_params=n_params, n_state_dict=n_sd, expected_rule='%d + (%d - %d) x %d' % (BASE_PARAMS, dd, FP_WIDTH, fc1_w),
                prereg=PREREG, commit=info['commit'], shuffled_order_sha256_prefix=info['perm_sha256_prefix'])
    record = dict(status='completed', tag=tag, task=T, slurm_array_job=str(jobid), **meta,
                  checkpoint=ck, checkpoint_sha256=sha256(ck), encoding_sidecar=side,
                  data_path=train, control_data_path=ctrl, data_record=RECORD, data_record_sha256=sha256(RECORD),
                  targets_table=TARGETS, targets_table_sha256=sha256(TARGETS), targets_line=info['targets_line'],
                  node=info['node'], log_out=out_log, log_end_time_line=info['end_time_line'],
                  finalized_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  finalize_script_sha256=sha256(os.path.abspath(__file__)))
    if not dry:
        os.makedirs(os.path.dirname(prov), exist_ok=True)
        json.dump(meta, open(side, 'w'), indent=2)
        json.dump(record, open(prov, 'w'), indent=2)
    return tag, n_params, n_sd, record['checkpoint_sha256'][:12], info['commit']


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--jobid', required=True, help='SLURM array job id whose logs certify the runs')
    p.add_argument('--tasks', required=True, help='e.g. 6-11')
    p.add_argument('--expect-arm')
    p.add_argument('--expect-split')
    p.add_argument('--expect-seed', type=int)
    p.add_argument('--force', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    expect = None
    if a.expect_arm or a.expect_split or a.expect_seed is not None:
        if not (a.expect_arm and a.expect_split and a.expect_seed is not None):
            sys.exit('FATAL: give all three of --expect-arm/--expect-split/--expect-seed or none')
        expect = (a.expect_arm, a.expect_split, a.expect_seed)
    os.chdir(W)
    if W not in sys.path:
        sys.path.insert(0, W)
    tasks = parse_tasks(a.tasks)
    failures = 0
    for T in tasks:
        try:
            tag, n_params, n_sd, sha, commit = finalize(T, a.jobid, a.force, a.dry_run, expect)
            print('[ok%s] T=%2d %-34s params=%d state_dict=%d sha256=%s commit=%s'
                  % (' dry' if a.dry_run else '', T, tag, n_params, n_sd, sha, (commit or '?')[:7]), flush=True)
        except Fail as e:
            failures += 1
            print('[FAIL] T=%2d %s' % (T, e), flush=True)
    print('[done] %d tasks, %d failures%s' % (len(tasks), failures, ', nothing written (dry run)' if a.dry_run else ''))
    sys.exit(1 if failures else 0)


if __name__ == '__main__':
    main()
