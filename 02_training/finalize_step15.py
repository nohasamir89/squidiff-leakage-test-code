#!/usr/bin/env python
"""finalize_step15.py -- certify a finished Step 15b (Tahoe) training run; write its sidecar and provenance.

Why this script exists
  The launcher's original post block compared the checkpoint's state_dict size (parameters + buffers, e.g.
  54,229,254 for base) with the parameters-only EXPECT count (54,221,060) and therefore failed on every task.
  Training itself was complete for the 33 runs that reached 'end_time'; the other 12 (tasks 3-14 of array
  18226073) were killed by the memory cgroup and are re-run. This script is the single implementation of the
  post-training check: it back-fills the 33 completed runs and is called by the launcher for every later run.

What it checks for task T (0-44); every check must pass before anything is written
  1. decode T -> split, arm, seed exactly as train_step15_tahoe.sh does, and cross-check against the
     '[task] T=..' line the launcher printed in the run's .out log (and against --expect-* if given)
  2. the run's .out log contains the training 'end_time' line (the loop reached its last step) and the .err
     log records no oom-kill event
  3. the checkpoint model_0.9999.pt loads STRICTLY into a model rebuilt from the launcher's exact flags
     (every key and shape must match the architecture), the parameters-only count equals the pre-registered
     EXPECT count for the arm, and attention arms carry encoder.arm.* weights
  4. the sidecar (model_0.9999.pt.encoding.json) and provenance (provenance/s15tr_<TAG>.json) do not exist yet
     (refuse to overwrite unless --force)
Written on success: the sidecar and the provenance record, with the checkpoint sha256 and the training commit
parsed from the run's own log (not the current HEAD, which may be later than the training).

Usage
  python -u finalize_step15.py --jobid 18226073 --tasks 0-2,15-44                 # back-fill completed runs
  python -u finalize_step15.py --jobid $SLURM_ARRAY_JOB_ID --tasks $T \
         --expect-arm $ARM --expect-split $SPLIT --expect-seed $SEED              # from the launcher
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
PREREG = 'PREREG_step15b_tahoe_model.md'
CKPT_NAME = 'model_0.9999.pt'

# Arm table: name, use_adaln, fm_source, cond_drop, encoder_arm flag ('none' = parser default 'mlp'), EXPECT params.
# Must stay identical to the case-table in train_step15_tahoe.sh; the launcher passes --expect-* so a divergence fails loudly.
ARMS = [
    ('base',        False, 'none',     0.0, 'none',       54221060),
    ('recipe',      True,  'gaussian', 0.0, 'none',       58178180),
    ('guidance_w2', True,  'gaussian', 0.1, 'none',       58178180),
    ('attn_cross',  True,  'gaussian', 0.0, 'attn_cross', 58532804),
    ('mlp_ctrl',    True,  'gaussian', 0.0, 'mlp_ctrl',   58532300),
]


class Fail(Exception):
    """A validation failure for one task; the message says which check failed."""


def decode(T):
    """T -> (split index, arm index, seed): split = T // 15, arm = (T % 15) // 3, seed = T % 3 (as the launcher)."""
    if not 0 <= T <= 44:
        raise Fail('task out of range 0-44: %d' % T)
    split_idx, within = divmod(T, 15)
    arm_idx, seed = divmod(within, 3)
    return split_idx, arm_idx, seed


def parse_tasks(spec):
    """'0-2,15-44' -> [0, 1, 2, 15, ..., 44]."""
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


def launcher_argv(split, seed, row, outdir, logdir):
    """The exact argument list train_step15_tahoe.sh passes to train_squidiff.py for this arm."""
    name, adaln, fm, cdrop, earm, _ = row
    train = '%s/tahoe_train_%s.h5ad' % (D, split)
    ctrl = '%s/tahoe_train_%s_control.h5ad' % (D, split)
    argv = ['--logger_path', logdir, '--data_path', train, '--control_data_path', ctrl,
            '--resume_checkpoint', outdir, '--seed', str(seed), '--use_drug_structure', 'True',
            '--drug_encoding', 'fingerprint', '--gene_size', '200', '--output_dim', '200',
            '--lr_anneal_steps', '100000']
    if name != 'base':
        argv += ['--use_adaln', 'True', '--fm_source', 'gaussian']
    if name == 'guidance_w2':
        argv += ['--cond_drop', '0.1']
    if earm != 'none':
        argv += ['--encoder_arm', earm]
    return argv, train, ctrl


def build_model(argv, row):
    """Parse the launcher's flags with train_squidiff's own parser, confirm each flag took effect for this arm
    (the guard the launcher runs before training), and build the model on CPU."""
    import train_squidiff as t
    from Squidiff.script_util import create_model_and_diffusion, model_and_diffusion_defaults, args_to_dict
    from Squidiff.scrna_datasets import DRUG_ENCODER_DIMS
    name, adaln, fm, cdrop, earm, _ = row
    saved = sys.argv
    sys.argv = ['train_squidiff.py'] + argv
    try:
        a = t.parse_args()
    finally:
        sys.argv = saved
    if a['use_adaln'] is not adaln:
        raise Fail('use_adaln resolved to %r, expected %r' % (a['use_adaln'], adaln))
    if a['fm_source'] != fm:
        raise Fail('fm_source %r, expected %r' % (a['fm_source'], fm))
    if not isinstance(a['cond_drop'], float) or abs(a['cond_drop'] - cdrop) > 1e-12:
        raise Fail('cond_drop %r, expected %.3f' % (a['cond_drop'], cdrop))
    if a['drug_encoding'] != 'fingerprint':
        raise Fail('drug_encoding %r' % a['drug_encoding'])
    if earm != 'none' and a.get('encoder_arm') != earm:
        raise Fail('encoder_arm %r, expected %r' % (a.get('encoder_arm'), earm))
    a['drug_dimension'] = DRUG_ENCODER_DIMS[a['drug_encoding']]
    a['memory_size'] = 0
    model, _ = create_model_and_diffusion(**args_to_dict(a, model_and_diffusion_defaults().keys()))
    return model


def load_checkpoint(path):
    import torch
    return torch.load(path, map_location='cpu', weights_only=True)


def read_log(out_log, err_log, T, split, name, seed):
    """Completion evidence from the run's own logs: the [task] line matches, 'end_time' was reached, no oom-kill."""
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
    c = re.search(r'\[commit\] ([0-9a-f]{7,40})', text)
    n = re.search(r'\[node\] (\S+)', text)
    return dict(commit=c.group(1) if c else None, node=n.group(1) if n else None, end_time_line=end_lines[-1].strip())


def finalize(T, jobid, force, dry, expect):
    split_idx, arm_idx, seed = decode(T)
    split = 'drug_split_%d' % split_idx
    row = ARMS[arm_idx]
    name, adaln, fm, cdrop, earm, exp_n = row
    if expect is not None and expect != (name, split, seed):
        raise Fail('launcher expects arm=%s split=%s seed=%d, decode gives %s %s %d' % (expect + (name, split, seed)))
    tag = '%s_%s_seed_%d' % (split, name, seed)
    outdir = '%s/step15/%s/%s_seed_%d' % (W, name, split, seed)
    logdir = 'logger_files/logger_s15_%s' % tag
    out_log = '%s/logger_files/s15tr_%s_%d.out' % (W, jobid, T)
    err_log = '%s/logger_files/s15tr_%s_%d.err' % (W, jobid, T)
    ck = os.path.join(outdir, CKPT_NAME)
    side = ck + '.encoding.json'
    prov = '%s/provenance/s15tr_%s.json' % (W, tag)

    info = read_log(out_log, err_log, T, split, name, seed)                       # check 1 + 2
    if not os.path.isfile(ck) or os.path.getsize(ck) == 0:                        # check 3
        raise Fail('checkpoint missing or empty: %s' % ck)
    argv, train, ctrl = launcher_argv(split, seed, row, outdir, logdir)
    model = build_model(argv, row)
    sd = load_checkpoint(ck)
    try:
        model.load_state_dict(sd, strict=True)                                    # every key and shape must match
    except Exception as e:
        raise Fail('checkpoint does not load into the rebuilt %s model: %s' % (name, str(e)[:300]))
    n_params = int(sum(p.numel() for p in model.parameters()))
    n_sd = int(sum(int(v.numel()) for v in sd.values() if hasattr(v, 'numel')))
    if n_params != exp_n:
        raise Fail('parameter count %d != expected %d for arm %s' % (n_params, exp_n, name))
    if earm != 'none' and not any(k.startswith('encoder.arm.') for k in sd):
        raise Fail('no encoder.arm.* keys in the checkpoint for arm %s' % name)
    if not force and (os.path.exists(side) or os.path.exists(prov)):              # check 4
        raise Fail('sidecar or provenance already exists for %s (use --force to overwrite)' % tag)

    meta = dict(arm=name, split=split, seed=seed, encoder_arm=(earm if earm != 'none' else 'mlp'), memory_size=0,
                use_adaln=adaln, fm_source=fm, cond_drop=cdrop, drug_encoding='fingerprint', gene_size=200,
                n_params=n_params, n_state_dict=n_sd, prereg=PREREG, commit=info['commit'])
    record = dict(status='completed', tag=tag, task=T, slurm_array_job=str(jobid), **meta,
                  checkpoint=ck, checkpoint_sha256=sha256(ck), encoding_sidecar=side,
                  data_path=train, control_data_path=ctrl, node=info['node'],
                  log_out=out_log, log_end_time_line=info['end_time_line'],
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
    p.add_argument('--tasks', required=True, help='e.g. 0-2,15-44')
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
            print('[ok%s] T=%2d %-32s params=%d state_dict=%d sha256=%s commit=%s'
                  % (' dry' if a.dry_run else '', T, tag, n_params, n_sd, sha, (commit or '?')[:7]), flush=True)
        except Fail as e:
            failures += 1
            print('[FAIL] T=%2d %s' % (T, e), flush=True)
    print('[done] %d tasks, %d failures%s' % (len(tasks), failures, ', nothing written (dry run)' if a.dry_run else ''))
    sys.exit(1 if failures else 0)


if __name__ == '__main__':
    main()
