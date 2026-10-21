"""Evaluate a Step 6 flow-matching checkpoint. PREREG_step6_flow_matching.md.

Wraps the frozen harness rather than editing it: leakage_test_base_sciplex_array.py
and sample_squidiff.py stay byte-identical, so arm A remains comparable to
Steps 4, 4b and 5 and the arm-A reproduction check in section 7 is meaningful.

The substitution is one function. The harness calls decode(z) -> (n, 200) via
sampler.pred, which passes model_kwargs={'z_mod': z} and runs the DDIM loop.
Flow matching passes the same single key and integrates the ODE instead.

  arm B: x0 = Gaussian noise, seeded per condition as the harness does
  arm C: x0 = the SAME control cells the encoder was conditioned on

Arm C's start point is deterministic, so its per-condition seed has no effect -
recorded rather than silently differing from arm B.

NFE is fixed at 20 with Euler for the gated comparison, per section 3.
"""
import sys, os, types
import numpy as np
import torch
from Squidiff.flow_matching import ode_sample

if len(sys.argv) < 5:
    sys.exit('usage: leakage_test_fm_sciplex.py SPLIT CKPT TAG ARM [NFE] [SOLVER]')
SPLIT, CKPT, TAG, ARM = sys.argv[1:5]
NFE = int(sys.argv[5]) if len(sys.argv) > 5 else 20
SOLVER = sys.argv[6] if len(sys.argv) > 6 else 'euler'
if ARM not in ('fm_gaussian', 'fm_control_ot'):
    sys.exit('FATAL: ARM must be fm_gaussian or fm_control_ot, got %r' % ARM)
print('[fm-eval] arm=%s  nfe=%d  solver=%s' % (ARM, NFE, SOLVER))
# Step 12: guidance is switched on by environment, so the Step 11 call signature
# and behaviour are unchanged when the variables are absent.
import os as _os
GUIDANCE_W = _os.environ.get('S12_GUIDANCE_W')
GUIDANCE_W = None if GUIDANCE_W in (None, '', 'none') else float(GUIDANCE_W)
GUIDE_TMIN = float(_os.environ.get('S12_GUIDE_TMIN', '0.0'))
_Z_DIM = [None]
print('[step12] guidance_w=%s  guide_t_min=%.2f' % (GUIDANCE_W, GUIDE_TMIN))

import sample_squidiff

_orig_sampler = sample_squidiff.sampler

class FMSampler(_orig_sampler):
    """Same construction and checkpoint routing; pred integrates an ODE."""
    _fm_source = None      # set per instance below
    _fm_x0 = None          # control cells for arm C, set by the patched harness

    def pred(self, z_sem, gene_size):
        z_null = None
        if GUIDANCE_W is not None:
            # _guided_encode returns [z | z_null]; the harness only slices rows.
            if _Z_DIM[0] is None or z_sem.shape[1] != 2 * _Z_DIM[0]:
                raise RuntimeError('guided pred expects [z | z_null] of width %s, got %d'
                                   % (None if _Z_DIM[0] is None else 2 * _Z_DIM[0], z_sem.shape[1]))
            z_sem, z_null = z_sem[:, :_Z_DIM[0]], z_sem[:, _Z_DIM[0]:]
        if self._fm_source == 'control':
            if self._fm_x0 is None:
                raise RuntimeError('arm C requires control cells; _fm_x0 is unset')
            x0 = self._fm_x0
            if x0.shape[0] != z_sem.shape[0]:
                raise RuntimeError('control rows %d != latent rows %d'
                                   % (x0.shape[0], z_sem.shape[0]))
        else:
            x0 = torch.randn(z_sem.shape[0], gene_size, device=z_sem.device)
        return ode_sample(self.model, x0, n_steps=NFE, solver=SOLVER,
                          model_kwargs={'z_mod': z_sem}, clamp_min=0.0,
                          null_kwargs=None if z_null is None else {'z_mod': z_null},
                          guidance_w=GUIDANCE_W, guide_t_min=GUIDE_TMIN)

def _make(*a, **k):
    s = FMSampler(*a, **k)
    s._fm_source = 'control' if ARM == 'fm_control_ot' else 'gaussian'
    return s

sample_squidiff.sampler = _make
# Step 9: the harness imports Drug_dose_encoder at its line 29 and calls it at
# 109, so it always builds a 1024-bit fingerprint regardless of what the
# checkpoint expects. The model is constructed correctly - the sampler infers
# drug_dimension from encoder.fc1.weight - so a pathway or combined checkpoint
# fails with (n,1224) x (255,2048). Rather than edit the FROZEN harness, the
# encoder is selected here from the checkpoint's own width and injected into
# the namespace the harness executes in.
import torch as _th
from Squidiff.scrna_datasets import DRUG_ENCODERS
_sd = _th.load(CKPT, map_location='cpu', weights_only=True)
_fc1 = _sd.get('encoder.fc1.weight')
if _fc1 is None:
    sys.exit('FATAL: encoder.fc1.weight absent; cannot infer the drug encoding')
_dd = int(_fc1.shape[1]) - 200
_name = {1024: 'fingerprint', 55: 'pathway', 1079: 'both',
         196: 'chemvae', 1220: 'fp_chemvae'}.get(_dd)
if _name is None:
    sys.exit('FATAL: inferred drug_dimension %d is not a known encoding' % _dd)
print('[harness] drug encoder = %s (drug_dimension=%d)' % (_name, _dd))
del _sd, _fc1

def Drug_dose_encoder(drug_SMILES_list, dose_list, comb_num=1, num_Bits=1024):
    """Shadows the harness's import so the encoding matches the checkpoint."""
    return DRUG_ENCODERS[_name](drug_SMILES_list, dose_list, comb_num)

def _guided_encode(sampler, dummy, fp, cf):
    """Real embedding, and under guidance the null embedding (zero fingerprint,
    same control cells - the training-time null), concatenated along the width."""
    z = sampler.model.encoder(dummy, None, fp, cf)
    if GUIDANCE_W is None:
        return z
    z_null = sampler.model.encoder(dummy, None, torch.zeros_like(fp), cf)
    _Z_DIM[0] = z.shape[1]
    return torch.cat([z, z_null], dim=1)

sys.argv = ['leakage_test_base_sciplex_array.py', SPLIT, CKPT, TAG]
_src = open('leakage_test_base_sciplex_array.py').read()
# Step 12: route both encoder calls through _guided_encode (in memory only; the
# harness file on disk is frozen). Each anchor must occur exactly once.
for _a, _b in (('zA = sampler.model.encoder(dummy, None, fp, cfA)', 'zA = _guided_encode(sampler, dummy, fp, cfA)'),
               ('zB = sampler.model.encoder(dummy, None, fp, cfB)', 'zB = _guided_encode(sampler, dummy, fp, cfB)')):
    if _src.count(_a) != 1:
        sys.exit('FATAL: harness anchor %r found %d times' % (_a, _src.count(_a)))
    _src = _src.replace(_a, _b)
# drop the harness's own import so it does not overwrite the shadow above
_src = _src.replace('from Squidiff.scrna_datasets import Drug_dose_encoder',
                    '# import shadowed by leakage_test_fm_sciplex.py (Step 9)')
exec(_src)
