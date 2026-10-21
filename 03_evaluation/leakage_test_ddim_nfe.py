"""Evaluate a DDIM checkpoint at reduced sampling steps. SECONDARY, post-hoc.

Wraps the frozen harness like leakage_test_fm_sciplex.py, overriding only
timestep_respacing so the DDIM loop takes N steps rather than 1000. Everything
else - conditions, cell selection, seeds, encoder path, metrics - is unchanged.
"""
import sys, os
if len(sys.argv) < 5:
    sys.exit('usage: leakage_test_ddim_nfe.py SPLIT CKPT TAG STEPS')
SPLIT, CKPT, TAG, STEPS = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
import sample_squidiff
_orig = sample_squidiff.sampler

class RespacedSampler(_orig):
    def parse_args(self, model_path, gene_size, output_dim, use_drug_structure):
        a = _orig.parse_args(self, model_path, gene_size, output_dim, use_drug_structure)
        a['timestep_respacing'] = '' if STEPS >= 1000 else str(STEPS)
        print('[respacing] %r -> DDIM will take %d steps'
              % (a['timestep_respacing'], STEPS))
        return a

sample_squidiff.sampler = RespacedSampler
sys.argv = ['leakage_test_base_sciplex_array.py', SPLIT, CKPT, TAG]
exec(open('leakage_test_base_sciplex_array.py').read())
