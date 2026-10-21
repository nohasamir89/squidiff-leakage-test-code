# -*- coding: utf-8 -*-
# @Author: SiyuHe
# @Last Modified by:   SiyuHe
# @Last Modified time: 2025-02-19


import argparse
import os
import numpy as np
import torch.distributed as dist
import torch
from Squidiff import dist_util, logger
from Squidiff.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from Squidiff.resample import create_named_schedule_sampler
from sklearn.metrics import r2_score
import scipy
class sampler:
    def __init__(self,model_path,gene_size,output_dim,use_drug_structure):
        args = self.parse_args(model_path,gene_size,output_dim,use_drug_structure)
        print("load model and diffusion...")
        # Detect the architecture FROM THE CHECKPOINT rather than taking a flag.
        # A caller that forgot to pass use_adaln would build a base model and
        # then raise on the 12 unexpected adaln_mod keys, or with strict=False
        # silently DROP them - producing plausible numbers from the wrong
        # architecture. Detection cannot be forgotten by a caller.
        _sd = dist_util.load_state_dict(args['model_path'])
        args['use_adaln'] = any('adaln_mod' in k for k in _sd.keys())
        # Step 9: the encoder's input width encodes the drug representation.
        # encoder.fc1.weight is (hidden, gene_size + drug_dimension): 1224, 255
        # or 1279 for a 200-gene panel. Read by key name rather than by shape,
        # so a future checkpoint cannot match a heuristic on the wrong tensor.
        _fc1 = _sd.get('encoder.fc1.weight')
        if _fc1 is None:
            raise SystemExit('FATAL: encoder.fc1.weight absent; cannot infer drug_dimension')
        _dd = int(_fc1.shape[1]) - int(args['gene_size'])
        # Step 16 (PREREG_step16_tahoe_targets.md section 9): the width is looked up in DRUG_ENCODER_DIMS. Two Step 16
        # encodings share the width 1,569 (fp_targets and its shuffled control), so the harness names the arm's encoding
        # in SQUIDIFF_DRUG_ENCODING, which must agree with the checkpoint's width; a width with a single encoding needs
        # no variable (fingerprint_s10 is an alias of fingerprint and is not listed).
        import os as _os16
        from Squidiff.scrna_datasets import DRUG_ENCODER_DIMS as _DIMS16
        _cands = sorted(k for k, v in _DIMS16.items() if v == _dd and k != 'fingerprint_s10')
        _env = _os16.environ.get('SQUIDIFF_DRUG_ENCODING')
        if _env is not None:
            if _env not in _DIMS16:
                raise SystemExit('FATAL: SQUIDIFF_DRUG_ENCODING=%r is not a known encoding %s' % (_env, sorted(_DIMS16)))
            if _DIMS16[_env] != _dd:
                raise SystemExit('FATAL: SQUIDIFF_DRUG_ENCODING=%r has width %d, the checkpoint has %d' % (_env, _DIMS16[_env], _dd))
            _enc_name = _env
        elif len(_cands) == 1:
            _enc_name = _cands[0]
        elif not _cands:
            raise SystemExit('FATAL: inferred drug_dimension %d is not a known encoding %s' % (_dd, _DIMS16))
        else:
            raise SystemExit('FATAL: drug_dimension %d is shared by %s; set SQUIDIFF_DRUG_ENCODING' % (_dd, _cands))
        args['drug_dimension'] = _dd
        print('drug encoding detected from checkpoint: %s (drug_dimension=%d)' % (_enc_name, _dd))
        # gene_module is a constructor argument, but unlike the time-only mode it
        # IS recoverable from the weights: the two variants have different keys.
        _gm_keys = [k for k in _sd.keys() if 'gene_module' in k]
        if not _gm_keys:
            args['gene_module'] = 'none'
        elif any('gene_module.attn' in k for k in _gm_keys):
            args['gene_module'] = 'attention'
        elif any('gene_module.fc1' in k for k in _gm_keys):
            args['gene_module'] = 'mlp'
        else:
            raise SystemExit('FATAL: unrecognised gene_module keys: %s' % _gm_keys[:5])
        # Step 13: the encoder arm (and the memory table's size) are read from the
        # checkpoint keys, the same way use_adaln and gene_module are.
        from Squidiff.drug_attention import infer_arm_from_state_dict
        args['encoder_arm'], args['memory_size'] = infer_arm_from_state_dict(_sd)
        print('encoder arm detected from checkpoint: %s (memory_size=%d)' % (args['encoder_arm'], args['memory_size']))
        if args['gene_module'] != 'none':
            print('gene module detected from checkpoint: %s' % args['gene_module'])
        # The modulation MODE is not recoverable from the weights: the semantic and
        # time-only arms are capacity-matched with identical parameter names and
        # shapes. A sidecar beside the checkpoint carries it. No sidecar means a
        # pre-control checkpoint, all of which are semantic, so behaviour for every
        # existing checkpoint is unchanged.
        import json as _json, os as _os
        _side = args['model_path'] + '.mode.json'
        if _os.path.exists(_side):
            _m = _json.load(open(_side))
            # bool("False") is True. That is the fault that invalidated the first
            # replication base arm; require genuine JSON Booleans rather than
            # coercing whatever the file holds.
            for _k in ('use_adaln', 'use_adaln_timeonly'):
                if type(_m.get(_k)) is not bool:
                    raise SystemExit('FATAL: sidecar %s must be a JSON Boolean, got %r (%s)'
                                     % (_k, _m.get(_k), type(_m.get(_k)).__name__))
            # The sidecar and checkpoint jointly define the model, so bind them:
            # a stale sidecar beside a different checkpoint would otherwise pass.
            import hashlib as _hl
            _want = _m.get('checkpoint_sha256')
            if _want:
                _h = _hl.sha256()
                with open(args['model_path'], 'rb') as _f:
                    for _c in iter(lambda: _f.read(1 << 20), b''): _h.update(_c)
                if _h.hexdigest() != _want:
                    raise SystemExit('FATAL: sidecar is bound to a different checkpoint '
                                     '(expects %s)' % _want[:12])
            if _m['use_adaln'] != bool(args['use_adaln']):
                raise SystemExit('FATAL: sidecar use_adaln=%r contradicts the state dict (%r)'
                                 % (_m['use_adaln'], args['use_adaln']))
            args['use_adaln_timeonly'] = _m['use_adaln_timeonly']
            print('mode sidecar: use_adaln=%s use_adaln_timeonly=%s'
                  % (args['use_adaln'], args['use_adaln_timeonly']))
        else:
            args['use_adaln_timeonly'] = False
        print("checkpoint architecture: %s" % ('adaLN-Zero' if args['use_adaln'] else 'base'))

        model, diffusion = create_model_and_diffusion(
                **args_to_dict(args, model_and_diffusion_defaults().keys())
            )

        model.load_state_dict(
            _sd
        )
        # EFFECTIVE mode on the constructed object. Requested values were not
        # sufficient last time: the bool('False') fault was a requested/effective
        # mismatch that no artifact revealed.
        _eff = (bool(getattr(model, 'use_adaln', False)),
                bool(getattr(model, 'use_adaln_timeonly', False)))
        if _eff != (bool(args['use_adaln']), bool(args['use_adaln_timeonly'])):
            raise SystemExit('FATAL: effective mode %r != requested %r'
                             % (_eff, (args['use_adaln'], args['use_adaln_timeonly'])))
        for _i, _b in enumerate(model.mlp_blocks):
            if bool(getattr(_b, 'use_adaln_timeonly', False)) != _eff[1]:
                raise SystemExit('FATAL: block %d mode disagrees with the model' % _i)
        print('effective use_adaln=%s use_adaln_timeonly=%s (%d blocks)'
              % (_eff[0], _eff[1], len(model.mlp_blocks)))
        model.to(dist_util.dev())
        if args['use_fp16']:
            model.convert_to_fp16()
        model.eval()
        self.model = model
        self.arg = args
        self.diffusion = diffusion
        self.sample_fn = (diffusion.p_sample_loop if not args['use_ddim'] else diffusion.ddim_sample_loop)
    
    def stochastic_encode(
        self, model, x, t, model_kwargs):
        """
        ddim reverse sample
        """
        sample = x
        sample_t = []
        xstart_t = []
        T = []
        indices = list(range(t))

        for i in indices:
            timestep = torch.full((x.shape[0],), i, device='cuda').long()
            with torch.no_grad():
                out = self.diffusion.ddim_reverse_sample(model, 
                                                    sample, 
                                                    timestep, 
                                                    model_kwargs=model_kwargs)
                sample = out['sample']
                sample_t.append(sample)
                xstart_t.append(out['pred_xstart'])
                T.append(timestep)

        return {
        'sample': sample,
        'sample_t': sample_t,
        'xstart_t': xstart_t,
        'T': T,
    }

    def parse_args(self,model_path,gene_size,output_dim,use_drug_structure):
        """Parse command-line arguments and update with default values."""
        # Define default arguments
        default_args = {}
        default_args.update(model_and_diffusion_defaults())
        default_args['use_adaln'] = False   # overridden by checkpoint detection
        updated_args = {
            'data_path': '',
            'schedule_sampler': 'uniform',
            'lr': 1e-4,
            'weight_decay': 0.0,
            'lr_anneal_steps': 1e5,
            'batch_size': 16,
            'microbatch': -1,
            'ema_rate': '0.9999',
            'log_interval': 1e4,
            'save_interval': 1e4,
            'resume_checkpoint': '',
            'use_fp16': False,
            'fp16_scale_growth': 1e-3,
            'gene_size': gene_size,
            'output_dim': output_dim,
            'num_layers': 3,
            'class_cond': False,
            'use_encoder': True,
            'use_ddim':True,
            'diffusion_steps': 1000,
            'logger_path': '',
            'model_path': model_path,
            'use_drug_structure':use_drug_structure,
            'comb_num':1,
            'drug_dimension':1024
        }
        default_args.update(updated_args)

        # Return the updated arguments as a dictionary
        return default_args

    def load_squidiff_model(self):
        print("load model and diffusion...")
        return self.model

    def load_sample_fn(self):
        
        return self.sample_fn

    def get_diffused_data(self,model, x, t, model_kwargs):
        sample = x
        sample_t = [x]  # Store initial data for plotting
        xstart_t = []
        T = []

        indices = list(range(t))

        for i in indices:
            timestep = torch.full((x.shape[0],), i, device='cuda').long()
            with torch.no_grad():
                # Replacing ddim_reverse_sample with a simpler forward diffusion process
                noise = torch.randn_like(sample)  # Add noise at each step
                out = sample + noise * (i / t)    # Simulating diffusion based on time step
                sample = out
                sample_t.append(sample.cpu())  # Store the samples for visualization
                xstart_t.append(sample.cpu())
                T.append(timestep)

        return {
            'sample': sample,
            'sample_t': sample_t,
            'xstart_t': xstart_t,
            'T': T
        }

    def sample_around_point(self, point, num_samples=None, scale=0.7):
        return point + scale * np.random.randn(num_samples, point.shape[0])

    def pred(self,z_sem, gene_size):
        
        pred_result = self.sample_fn(
                        self.model,
                        shape = (z_sem.shape[0], gene_size),
                        model_kwargs={
                            'z_mod': z_sem
                        },
                        noise =  None
                )
        return pred_result
    
    def interp_with_direction(self, z_sem_origin = None, gene_size = None, direction = None, scale = 1, add_noise_term = True):

        z_sem_origin = z_sem_origin.detach().cpu().numpy()
        z_sem_interp_ = z_sem_origin.mean(axis=0) + direction.detach().cpu().numpy() * scale
        if add_noise_term:
            z_sem_interp_ = self.sample_around_point(z_sem_interp_, num_samples=z_sem_origin.shape[0])

        z_sem_interp_ = torch.tensor(z_sem_interp_,dtype=torch.float32).to('cuda')
        sample_interp = self.sample_fn(
                            self.model,
                            shape = (z_sem_origin.shape[0],gene_size),
                            model_kwargs={
                                'z_mod': z_sem_interp_
                            },
                            noise =  None
        )
        return sample_interp
        
    def cal_metric(self,x1,x2):
        r2 = r2_score(x1.detach().cpu().numpy().mean(axis=0),
                      x2.X.mean(axis=0))
        pearsonr,_ = scipy.stats.pearsonr(x1.detach().cpu().numpy().mean(axis=0),
                      x2.X.mean(axis=0))
        return r2, pearsonr

        

