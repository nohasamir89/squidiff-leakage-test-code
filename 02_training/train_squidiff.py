# -*- coding: utf-8 -*-
# @Author: SiyuHe
# @Last Modified by:   SiyuHe
# @Last Modified time: 2025-02-19

import io
import os
import socket

import torch as th
import torch.distributed as dist
import argparse
import random
import numpy as np
from datetime import datetime
from Squidiff import dist_util,logger

from Squidiff.scrna_datasets import prepared_data
from Squidiff.resample import create_named_schedule_sampler
from Squidiff.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
    str2bool,
)
from Squidiff.train_util import TrainLoop,plot_loss

GPUS_PER_NODE = 1  # Set this to the actual number of GPUs per node

def load_state_dict(path, **kwargs):
    """
    Load a PyTorch file without redundant fetches across ranks.
    """
    chunk_size = 2 ** 30  # Size limit for data chunks
    if dist.get_rank() == 0:
        with open(path, "rb") as f:
            data = f.read()
        num_chunks = len(data) // chunk_size
        if len(data) % chunk_size:
            num_chunks += 1
        dist.broadcast(th.tensor(num_chunks), 0)
        for i in range(0, len(data), chunk_size):
            dist.broadcast(th.tensor(data[i: i + chunk_size]), 0)
    else:
        num_chunks = dist.broadcast(th.tensor(0), 0).item()
        data = bytes()
        for _ in range(num_chunks):
            chunk = th.zeros(chunk_size, dtype=th.uint8)
            dist.broadcast(chunk, 0)
            data += bytes(chunk.numpy())

    return th.load(io.BytesIO(data), **kwargs)

def set_seed(seed):
    """Seed every RNG that affects training. Called before model creation
    and dataloader construction so weight init and batch order are both fixed."""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed_all(seed)
    th.backends.cudnn.deterministic = True
    th.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    return seed


def run_training(args):
        
    logger.configure(dir=args['logger_path'])
    _seed = set_seed(args.get('seed', 0))
    logger.log(f'[seed] set_seed({_seed}) applied before model creation')
    logger.log("*********creating model and diffusion**********")
    # Step 9: the encoder's input width MUST match the chosen drug encoding.
    # A mismatch would construct a model whose first layer disagrees with the
    # data, so this asserts rather than defaulting to 1024.
    from Squidiff.scrna_datasets import DRUG_ENCODER_DIMS
    _enc = args.get('drug_encoding', 'fingerprint')
    if _enc not in DRUG_ENCODER_DIMS:
        raise ValueError('drug_encoding must be one of %s, got %r'
                         % (sorted(DRUG_ENCODER_DIMS), _enc))
    args['drug_dimension'] = DRUG_ENCODER_DIMS[_enc]
    print('[step9] drug_encoding=%r -> drug_dimension=%d' % (_enc, args['drug_dimension']))

    # Step 13: the memory arm needs one slot per TRAINING compound. The table is
    # built from the training file only; the split's test compounds must be
    # disjoint from it (leakage guard), and both facts are printed for provenance.
    _arm = args.get('encoder_arm', 'mlp')
    from Squidiff.drug_attention import ARMS
    if _arm not in ARMS:
        raise ValueError('encoder_arm must be one of %s, got %r' % (ARMS, _arm))
    args['memory_size'] = 0
    _mem_smiles = None
    if _arm == 'attn_mem':
        import anndata as _ad, hashlib as _hl
        _tr = _ad.read_h5ad(args['data_path'], backed='r')
        _mem_smiles = sorted(set(str(s) for s in _tr.obs['SMILES']))
        _te_path = args['data_path'].replace('_train_', '_test_')
        if _te_path == args['data_path'] or not os.path.isfile(_te_path):
            raise RuntimeError('attn_mem: cannot locate the test file for the leakage guard: %r' % _te_path)
        _te_smiles = set(str(s) for s in _ad.read_h5ad(_te_path, backed='r').obs['SMILES'])
        _overlap = sorted(set(_mem_smiles) & _te_smiles)
        if _overlap:
            raise RuntimeError('attn_mem: %d test compounds would enter the memory table' % len(_overlap))
        args['memory_size'] = len(_mem_smiles)
        print('[step13] memory table: %d training compounds, %d test compounds, overlap 0, smiles_sha256=%s'
              % (len(_mem_smiles), len(_te_smiles), _hl.sha256('\n'.join(_mem_smiles).encode()).hexdigest()))

    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )

    # Step 13 assertions, part of the run (PREREG section 3).
    if _arm == 'attn_mem':
        from Squidiff.scrna_datasets import Drug_dose_encoder as _fpenc
        # dose 9 -> log10(10) = 1 exactly: the unscaled FCFP4 bits, by the same function
        _fps = _fpenc(_mem_smiles, [9.0] * len(_mem_smiles), comb_num=1)
        model.encoder.arm.fill_memory(_fps)
    _n_params = sum(p.numel() for p in model.parameters())
    _expected = {'mlp': (58178180 if args.get('use_adaln') is True else 54221060), 'attn_fp': 58178180 + 182080, 'attn_mem': 58178180 + 197760 + 64 * int(args['memory_size']),
                 'attn_cross': 58178180 + 354624, 'mlp_ctrl': 58178180 + 354120}[_arm]
    # Step 16 (PREREG_step16_tahoe_targets.md section 3): a drug encoding wider than the 1,024-bit fingerprint changes
    # the encoder's first layer only, by (drug_dimension - 1024) x width(fc1) parameters. The attention arms' own
    # modules also read the drug width, so a width other than 1,024 is accepted for the mlp arm alone.
    _extra_in = int(args['drug_dimension']) - 1024
    if _extra_in != 0:
        if _arm != 'mlp':
            raise RuntimeError('[step16] drug_dimension %d is supported for encoder_arm mlp only, got %s' % (args['drug_dimension'], _arm))
        _fc1_w = int(model.encoder.fc1.weight.shape[0])
        print('[step16] drug_dimension=%d: expected parameters %d = %d + %d x %d (fc1 width)'
              % (args['drug_dimension'], _expected + _extra_in * _fc1_w, _expected, _extra_in, _fc1_w))
        _expected += _extra_in * _fc1_w
    if _n_params != _expected:
        raise RuntimeError('[step13] parameter count %d, expected %d for arm %s' % (_n_params, _expected, _arm))
    if model.encoder.arm is not None:
        with th.no_grad():
            _g = th.Generator().manual_seed(0)
            _fp = (th.rand(8, args['drug_dimension'], generator=_g) < 0.06).float() * 3.0
            _cf = th.randn(8, args['gene_size'], generator=_g)
            _r = model.encoder.arm(_fp, _cf)
        if float(_r.abs().max()) != 0.0:
            raise RuntimeError('[step13] arm residual is not zero at initialisation')
    print('[step13] encoder_arm=%s params=%d memory_size=%d identity_at_init=pass' % (_arm, _n_params, int(args['memory_size'])))
    _p0 = next(model.parameters())
    logger.log('[seed] seed=%s first-param sum=%.8f shape=%s' % (_seed, float(_p0.detach().float().sum()), tuple(_p0.shape)))
    
    model.to(dist_util.dev())
    # Step 6 arms. ONE string flag with three values, never a bool: the parser
    # uses type=type(value), so a Boolean flag would hit the bool('False') trap.
    #   none        -> arm A, the existing DDIM diffusion object, unchanged
    #   gaussian    -> arm B, flow matching from Gaussian noise
    #   control_ot  -> arm C, flow matching from control cells with OT coupling
    _fm = args.get('fm_source', 'none')
    _cd = float(args.get('cond_drop', 0.0))
    if _cd > 0 and _fm == 'none':
        raise ValueError('cond_drop is implemented only in FlowMatchingWrapper; fm_source is none')
    if _fm != 'none':
        from Squidiff.flow_matching import FlowMatchingWrapper
        if _fm == 'gaussian':
            diffusion = FlowMatchingWrapper(source='gaussian', ot=False, cond_drop=_cd)
        elif _fm == 'control_ot':
            diffusion = FlowMatchingWrapper(source='control', ot=True, cond_drop=_cd)
        else:
            raise ValueError("fm_source must be none, gaussian or control_ot, got %r" % _fm)
        print('[step6] objective = flow matching, source=%s' % _fm)
        print('[step12] cond_drop=%.3f (%s)' % (_cd, type(args.get('cond_drop')).__name__))
    else:
        print('[step6] objective = DDIM denoising (arm A)')


    schedule_sampler = create_named_schedule_sampler(args['schedule_sampler'], diffusion)

    logger.log("creating data loader...")
    data = prepared_data(
        data_dir = args['data_path'],
        control_data_dir = args['control_data_path'],
        batch_size = args['batch_size'],
        use_drug_structure= args['use_drug_structure'],
        comb_num = args['comb_num'],
        drug_encoding = args['drug_encoding']
    )
    #logger.log(f'with gpu {dist_util.dev()}')
    start_time = datetime.now()
    logger.log(f'**********training started at {start_time} **********')
    train_ = TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args['batch_size'],
        microbatch=args['microbatch'],
        lr=args['lr'],
        ema_rate=args['ema_rate'],
        log_interval=args['log_interval'],
        save_interval=args['save_interval'],
        resume_checkpoint=args['resume_checkpoint'],
        use_fp16=args['use_fp16'],
        fp16_scale_growth=args['fp16_scale_growth'],
        schedule_sampler=schedule_sampler,
        weight_decay=args['weight_decay'],
        lr_anneal_steps=args['lr_anneal_steps'],
        use_drug_structure= args['use_drug_structure'],
        comb_num=args['comb_num'],
    )
    train_.run_loop()
    
    end_time = datetime.now()

    during_time = (end_time-start_time).seconds/60

    logger.log(f'start time: {start_time} end_time: {end_time} time:{during_time} min')
    
    return train_.loss_list


def parse_args():
    """Parse command-line arguments and update with default values."""
    # Define default arguments
    default_args = {}
    default_args.update(model_and_diffusion_defaults())
    updated_args = {
        'data_path': '',
        'control_data_path': '',
        'schedule_sampler': 'uniform',
        'fm_source': 'none',
        'cond_drop': 0.0,
        'seed': 0,
        'lr': 1e-4,
        'weight_decay': 0.0,
        'lr_anneal_steps': 1e5,
        'batch_size': 64,
        'microbatch': -1,
        'ema_rate': '0.9999',
        'log_interval': 1e4,
        'save_interval': 1e4,
        'resume_checkpoint': '',
        'use_fp16': False,
        'fp16_scale_growth': 1e-3,
        'gene_size': 100,
        'output_dim': 100,
        'num_layers': 3,
        'class_cond': False,
        'use_encoder': True,
        'diffusion_steps': 1000,
        'logger_path': '',
        'use_drug_structure':False,
        'comb_num':1,
        'drug_encoding':'fingerprint',
        'encoder_arm':'mlp',
        'use_ddim':True,
    }
    default_args.update(updated_args)
    # Initialize argument parser
    parser = argparse.ArgumentParser(description='Perturbation-conditioned generative diffusion model')
    
    # Add arguments to the parser (these should correspond to the keys in default_args)
    for key, value in default_args.items():
        # Booleans parse through str2bool: with type=bool, argparse would call bool('False'),
        # which is True, so `--use_adaln False` would silently select the adaLN arm. Every
        # recorded run passed only `True` (verified 21 Oct 2026); this guards third-party reruns.
        kind = str2bool if isinstance(value, bool) else type(value)
        parser.add_argument(f'--{key}', default=value, type=kind, help=f'{key} (default: {value})')

    # Parse command-line arguments
    args = parser.parse_args()

    # Convert the parsed arguments to a dictionary and update the defaults
    updated_args = vars(args)
    
    # Check if 'logger_path' is None and raise an error if so
    if updated_args['logger_path']=='':
        logger.log('ERROR:Please specify the logger path --logger_path.')
        raise ValueError("Logger path is required. Please specify the logger path.")

            # Check if 'logger_path' is None and raise an error if so
    if updated_args['data_path']=='':
        logger.log("ERROR:Please specify the data path --data_path.")
        raise ValueError("Dataset path is required. Please specify the path where the training adata is.")


    # Return the updated arguments as a dictionary
    return updated_args



if __name__ == "__main__":
    args_train = parse_args()
    print('**************training args*************')
    print(args_train)
    losses = run_training(args_train)
    
    plot_loss(losses,args_train)
    
    

