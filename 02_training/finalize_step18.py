#!/usr/bin/env python
"""finalize_step18.py -- certify a finished Step 18 (Tahoe, OT-paired training controls) training run; write its sidecar
and provenance. The Step 16 finaliser for one arm at the recipe's width (PREREG_step18_tahoe_ot_pairing.md section 9).

What it checks for task T (0-8); every check must pass before anything is written
  1. decode T -> split, seed exactly as train_step18_tahoe.sh does (split = T // 3, seed = T % 3; the arm is ot_pair),
     cross-checked against the '[task] T=..' line the launcher printed in the run's .out log (and --expect-* if given);
     that line must name the OT-paired control file
  2. the run's .out log contains the training 'end_time' line and the launcher's '[step18]' guard line; the .err log
     records no oom-kill event
  3. the checkpoint model_0.9999.pt loads STRICTLY into a model rebuilt from the launcher's exact flags; the parameter
     count equals the recipe's 58,178,180; no encoder.arm.* keys; the first layer reads 200 + 1,024 inputs
  4. the OT control file still hashes to the pin in tahoe_results/ot_pairing_check.json, and the original control file
     to the data record
  5. the sidecar (model_0.9999.pt.encoding.json) and provenance (provenance/s18tr_<TAG>.json) do not exist yet
Written on success: the sidecar and the provenance record, with the checkpoint sha256, the OT control file sha256,
the data record and pairing record sha256 and the training commit parsed from the run's own log.

Usage
  python -u finalize_step18.py --jobid $SLURM_ARRAY_JOB_ID --tasks $T --expect-split $SPLIT --expect-seed $SEED
  python -u finalize_step18.py --jobid <id> --tasks 3-5                            # back-fill completed runs
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
PREREG = 'PREREG_step18_tahoe_ot_pairing.md'
CKPT_NAME = 'model_0.9999.pt'
BASE_PARAMS = 58178180                      # the recipe (Step 15b): adaLN + Gaussian flow matching, fingerprint, 200 genes
RECORD = 'tahoe_results/step15_data_record.json'
OTREC = 'tahoe_results/ot_pairing_check.json'
ARM, ENC = 'ot_pair', 'fingerprint'
N_TASKS = 9


class Fail(Exception):
    """A validation failure for one task; the message says which check failed."""


def decode(T):
    """T -> (split index, seed): split = T // 3, seed = T % 3 (as the launcher)."""
    if not 0 <= T < N_TASKS:
        raise Fail('task out of range 0-%d: %d' % (N_TASKS - 1, T))
    return divmod(T, 3)


def parse_tasks(spec):
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


def launcher_argv(split, seed, outdir, logdir):
    """The exact argument list train_step18_tahoe.sh passes to train_squidiff.py (the recipe's flags + the OT control)."""
    train = '%s/tahoe_train_%s.h5ad' % (D, split)
    ctrl0 = '%s/tahoe_train_%s_control.h5ad' % (D, split)
    ctrl = '%s/tahoe_train_%s_control_ot.h5ad' % (D, split)
    argv = ['--logger_path', logdir, '--data_path', train, '--control_data_path', ctrl,
            '--resume_checkpoint', outdir, '--seed', str(seed), '--use_drug_structure', 'True',
            '--drug_encoding', ENC, '--gene_size', '200', '--output_dim', '200',
            '--lr_anneal_steps', '100000', '--use_adaln', 'True', '--fm_source', 'gaussian']
    return argv, train, ctrl0, ctrl


def build_model(argv):
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
    if a['drug_encoding'] != ENC:
        raise Fail('drug_encoding %r, expected %r' % (a['drug_encoding'], ENC))
    if a.get('encoder_arm', 'mlp') != 'mlp':
        raise Fail('encoder_arm %r, expected mlp' % a.get('encoder_arm'))
    if int(a['gene_size']) != 200:
        raise Fail('gene_size %r, expected 200' % a['gene_size'])
    if not a['control_data_path'].endswith('_control_ot.h5ad'):
        raise Fail('control_data_path %r is not the OT-paired file' % a['control_data_path'])
    a['drug_dimension'] = DRUG_ENCODER_DIMS[ENC]
    a['memory_size'] = 0
    model, _ = create_model_and_diffusion(**args_to_dict(a, model_and_diffusion_defaults().keys()))
    fc1 = model.encoder.fc1.weight
    if int(fc1.shape[1]) != 200 + a['drug_dimension']:
        raise Fail('encoder.fc1 reads %d inputs, expected 200 + %d' % (int(fc1.shape[1]), a['drug_dimension']))
    return model, a['drug_dimension'], int(fc1.shape[0])


def load_checkpoint(path):
    import torch
    return torch.load(path, map_location='cpu', weights_only=True)


def read_log(out_log, err_log, T, split, seed, ctrl):
    if not os.path.isfile(out_log):
        raise Fail('missing log %s' % out_log)
    text = open(out_log).read()
    m = re.search(r'\[task\] T=(\d+)\s+split=(\S+)\s+arm=(\S+)\s+seed=(\d+).*?control=(\S+)', text)
    if not m:
        raise Fail('no [task] line with a control= field in %s' % out_log)
    if (int(m.group(1)), m.group(2), m.group(3), int(m.group(4))) != (T, split, ARM, seed):
        raise Fail('log says T=%s split=%s arm=%s seed=%s, expected T=%d %s %s %d'
                   % (m.group(1), m.group(2), m.group(3), m.group(4), T, split, ARM, seed))
    if os.path.abspath(m.group(5)) != os.path.abspath(ctrl):
        raise Fail('the run was launched with control=%s, expected %s' % (m.group(5), ctrl))
    end_lines = [l for l in text.splitlines() if 'end_time' in l]
    if not end_lines:
        raise Fail('training did not reach end_time in %s' % out_log)
    if os.path.isfile(err_log) and 'oom-kill' in open(err_log).read():
        raise Fail('oom-kill recorded in %s' % err_log)
    g = re.search(r'\[step18\] arm=ot_pair split=(\S+) parameters=(\d+) .*OT control (\S+) pinned \(([0-9a-f]{12})\)', text)
    if not g:
        raise Fail('no [step18] guard line in %s' % out_log)
    if g.group(1) != split or int(g.group(2)) != BASE_PARAMS:
        raise Fail('the [step18] line says split %s, parameters %s' % (g.group(1), g.group(2)))
    c = re.search(r'\[commit\] ([0-9a-f]{7,40})', text)
    n = re.search(r'\[node\] (\S+)', text)
    return dict(commit=c.group(1) if c else None, node=n.group(1) if n else None, end_time_line=end_lines[-1].strip(),
                guard_line=g.group(0).strip(), logged_ot_sha_prefix=g.group(4))


def finalize(T, jobid, force, dry, expect):
    split_idx, seed = decode(T)
    split = 'drug_split_%d' % split_idx
    if expect is not None and expect != (split, seed):
        raise Fail('launcher expects split=%s seed=%d, decode gives %s %d' % (expect + (split, seed)))
    tag = '%s_%s_seed_%d' % (split, ARM, seed)
    outdir = '%s/step18/%s/%s_seed_%d' % (W, ARM, split, seed)
    logdir = 'logger_files/logger_s18_%s' % tag
    out_log = '%s/logger_files/s18tr_%s_%d.out' % (W, jobid, T)
    err_log = '%s/logger_files/s18tr_%s_%d.err' % (W, jobid, T)
    ck = os.path.join(outdir, CKPT_NAME)
    side = ck + '.encoding.json'
    prov = '%s/provenance/s18tr_%s.json' % (W, tag)

    argv, train, ctrl0, ctrl = launcher_argv(split, seed, outdir, logdir)
    info = read_log(out_log, err_log, T, split, seed, ctrl)                        # checks 1 + 2
    if not os.path.isfile(ck) or os.path.getsize(ck) == 0:                        # check 3
        raise Fail('checkpoint missing or empty: %s' % ck)
    for q in (RECORD, OTREC, train, ctrl0, ctrl):
        if not os.path.isfile(q):
            raise Fail('missing input %s' % q)
    model, dd, fc1_w = build_model(argv)
    sd = load_checkpoint(ck)
    try:
        model.load_state_dict(sd, strict=True)
    except Exception as e:
        raise Fail('checkpoint does not load into the rebuilt model: %s' % str(e)[:300])
    n_params = int(sum(p.numel() for p in model.parameters()))
    n_sd = int(sum(int(v.numel()) for v in sd.values() if hasattr(v, 'numel')))
    if n_params != BASE_PARAMS:
        raise Fail('parameter count %d != the recipe\'s %d' % (n_params, BASE_PARAMS))
    if any(k.startswith('encoder.arm.') for k in sd):
        raise Fail('unexpected encoder.arm.* keys in the checkpoint')
    rec, ot = json.load(open(RECORD)), json.load(open(OTREC))                    # check 4
    w = ot['written'][split]
    h_ot, h_c0 = sha256(ctrl), sha256(ctrl0)
    if h_ot != w['sha256'] or not h_ot.startswith(info['logged_ot_sha_prefix']):
        raise Fail('OT control file sha256 %s differs from the pairing record %s or from the run log' % (h_ot[:16], w['sha256'][:16]))
    if h_c0 != rec['splits'][str(split_idx)]['files']['train_control']['sha256'] or h_c0 != w['source_control_sha256']:
        raise Fail('the original control file differs from the data record or the pairing record')
    if not force and (os.path.exists(side) or os.path.exists(prov)):              # check 5
        raise Fail('sidecar or provenance already exists for %s (use --force to overwrite)' % tag)

    meta = dict(arm=ARM, split=split, seed=seed, encoder_arm='mlp', memory_size=0, use_adaln=True, fm_source='gaussian',
                cond_drop=0.0, drug_encoding=ENC, drug_dimension=dd, fc1_width=fc1_w, gene_size=200,
                n_params=n_params, n_state_dict=n_sd, prereg=PREREG, commit=info['commit'],
                control_data_path=ctrl, control_ot_sha256=h_ot)
    record = dict(status='completed', tag=tag, task=T, slurm_array_job=str(jobid), **meta,
                  checkpoint=ck, checkpoint_sha256=sha256(ck), encoding_sidecar=side,
                  data_path=train, original_control_path=ctrl0, original_control_sha256=h_c0,
                  data_record=RECORD, data_record_sha256=sha256(RECORD), pairing_record=OTREC, pairing_record_sha256=sha256(OTREC),
                  guard_line=info['guard_line'], node=info['node'], log_out=out_log, log_end_time_line=info['end_time_line'],
                  finalized_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  finalize_script_sha256=sha256(os.path.abspath(__file__)))
    if not dry:
        os.makedirs(os.path.dirname(prov), exist_ok=True)
        json.dump(meta, open(side, 'w'), indent=2)
        json.dump(record, open(prov, 'w'), indent=2)
    return tag, n_params, n_sd, record['checkpoint_sha256'][:12], info['commit']


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--jobid', required=True)
    p.add_argument('--tasks', required=True, help='e.g. 3-5')
    p.add_argument('--expect-split')
    p.add_argument('--expect-seed', type=int)
    p.add_argument('--force', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    expect = None
    if a.expect_split or a.expect_seed is not None:
        if not (a.expect_split and a.expect_seed is not None):
            sys.exit('FATAL: give both --expect-split and --expect-seed or neither')
        expect = (a.expect_split, a.expect_seed)
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
