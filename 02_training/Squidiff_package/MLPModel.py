from abc import abstractmethod

import math

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from .fp16_util import convert_module_to_f16, convert_module_to_f32
from .nn import (
    checkpoint,
    linear,
    zero_module,
    normalization,
    timestep_embedding,
)

class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb, z_sem):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """
        
        

class MLPBlock(TimestepBlock):
    """
    Basic MLP block with an optional timestep embedding.
    """

    def __init__(self, input_dim, output_dim, time_embed_dim=None, latent_dim = None, use_adaln=False, use_adaln_timeonly=False):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)
        self.fc2 = nn.Linear(output_dim, output_dim)
        self.layer_norm1 = nn.LayerNorm(output_dim)
        self.layer_norm2 = nn.LayerNorm(output_dim)
        self.time_embed_dim = time_embed_dim
        if time_embed_dim is not None:
            self.time_dense = nn.Linear(time_embed_dim, output_dim)
        if latent_dim is not None:
            self.zsem_dense = nn.Linear(latent_dim, output_dim)
        self.use_adaln = use_adaln
        # Capacity control (PREREG amendment 13). The modulation MLP is identical
        # in parameter count, topology, rank and initialisation; only its INPUT
        # changes, from time_dense(emb) + zsem_dense(z_sem) to time_dense(emb)
        # alone. The base ADDITIVE path is unaffected in both cases.
        # NOTE the saved state dict cannot distinguish the two arms - that is what
        # capacity matching means - so verification must be behavioural.
        self.use_adaln_timeonly = use_adaln_timeonly
        if use_adaln_timeonly and not use_adaln:
            raise ValueError('use_adaln_timeonly requires use_adaln')
        if use_adaln:
            # adaLN-Zero adapted for a stack with NO residual connections.
            # The conditioning vector c that the base ADDS is also mapped to a
            # per-feature shift and scale applied after each LayerNorm. Both are
            # zero-initialised, so at step 0 h*(1+0)+0 == h and the block computes
            # EXACTLY the base forward. The usual DiT construction (zero-gated
            # residual) is unavailable here: with no skip path, a zero gate would
            # output zero rather than the identity and the network would be dead.
            # Bottlenecked to keep the parameter cost small. A full-rank
            # Linear(2048, 4*2048) adds 16.8M params per block, +93% on a 54M
            # model, which would make any Step 4 improvement unattributable
            # (capacity vs conditioning). c carries at most z_sem (60 dims) plus
            # the timestep embedding, so a rank-128 factorisation loses nothing
            # the conditioning did not already have. ~1.3M per block, +7% total.
            # Only the SECOND layer is zero-initialised, which is sufficient for
            # the whole modulation to vanish at init.
            adaln_bottleneck = 128
            self.adaln_mod = nn.Sequential(
                nn.Linear(output_dim, adaln_bottleneck),
                nn.SiLU(),
                nn.Linear(adaln_bottleneck, 4 * output_dim),
            )
            nn.init.zeros_(self.adaln_mod[2].weight)
            nn.init.zeros_(self.adaln_mod[2].bias)
    
    def forward(self, x, emb, z_sem):
        c = None
        c_time = None
        if emb is not None:
            c_time = self.time_dense(emb)
            c = c_time
            if z_sem is not None:
                c = c + self.zsem_dense(z_sem)
        use_mod = self.use_adaln and (c is not None)
        if use_mod:
            # semantic arm uses c; time-only arm uses c_time. When
            # use_adaln_timeonly is False this is bitwise identical to before.
            c_mod = c_time if self.use_adaln_timeonly else c
            s1, b1, s2, b2 = self.adaln_mod(F.silu(c_mod)).chunk(4, dim=-1)
        h = self.layer_norm1(self.fc1(x))
        if use_mod:
            h = h * (1 + s1) + b1
        h = F.silu(h)
        if c is not None:
            h = h + c
        h = self.layer_norm2(self.fc2(h))
        if use_mod:
            h = h * (1 + s2) + b2
        h = F.silu(h)
        return h
    
    


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb, z_sem):
        
        for layer in self:
            
            
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb, z_sem)
            else:
                x = layer(x)
        return x

class MLPModel(nn.Module):
    """
    MLP model for single-cell RNA-seq data with timestep embedding.
    """

    def __init__(self, 
                 gene_size, 
                 output_dim, 
                 num_layers, 
                 hidden_sizes=2048,
                 time_pos_dim=2048,
                 num_classes = None,
                 latent_dim=60,
                 use_checkpoint=False,
                 use_fp16 = False,
                 use_scale_shift_norm =False,
                 dropout=0,
                 time_embed_dim=2048,
                 use_encoder=False,
                 use_drug_structure = False,
                 drug_dimension = 1024,
                 comb_num=1,
                 use_adaln=False,
                 use_adaln_timeonly=False,
                 encoder_arm='mlp',
                 memory_size=0,
                 gene_module='none',
                 gene_embed_dim=512,
                 gene_heads=16,
                 gene_bottleneck=8120,
                 
                ):
        super().__init__()
        
        self.use_encoder = use_encoder
        self.time_embed_dim = time_embed_dim
        self.latent_dim = latent_dim
        self.time_embed = None
        self.hidden_sizes = hidden_sizes
        self.num_classes = num_classes
        self.drug_dimension = drug_dimension
        self.use_adaln = use_adaln
        self.use_adaln_timeonly = use_adaln_timeonly
        if use_encoder: 
            self.encoder = EncoderMLPModel(gene_size,self.hidden_sizes, self.num_classes, use_drug_structure, self.drug_dimension, comb_num,
                                           encoder_arm=encoder_arm, memory_size=memory_size)
        
        
        if time_embed_dim is not None:
            self.time_embed = nn.Sequential(
                nn.Linear(time_pos_dim, time_embed_dim),
                nn.SiLU(),
                nn.Linear(time_embed_dim, time_embed_dim),
            )

        if self.use_encoder: 
            layers = []
            for _ in range(num_layers):
                layers.append(MLPBlock(hidden_sizes, hidden_sizes, time_embed_dim, latent_dim, use_adaln=use_adaln, use_adaln_timeonly=use_adaln_timeonly))
            self.mlp_blocks = TimestepEmbedSequential(*layers)
        else:
            layers = []
            for _ in range(num_layers):
                layers.append(MLPBlock(hidden_sizes, hidden_sizes, time_embed_dim, use_adaln=use_adaln, use_adaln_timeonly=use_adaln_timeonly))
            self.mlp_blocks = TimestepEmbedSequential(*layers)
        
        self.input_layer = nn.Linear(gene_size, hidden_sizes)
        self.output_layer = nn.Linear(hidden_sizes, output_dim)
        # Gene-level mixing (Step 5). input_layer is a DENSE 200->2048 map, so no
        # hidden unit corresponds to a gene and 2048 does not factor into 200
        # tokens: there is no point inside the block stack where gene tokens
        # exist. The module attaches AFTER output_layer as a residual refinement.
        # The flag is a STRING, not None: train_squidiff.py builds its parser with
        # type=type(value), so a None default would yield type=NoneType and reject
        # any value. Same divergence that produced the bool('False') fault.
        self.gene_module_kind = gene_module
        if gene_module in (None, 'none'):
            self.gene_module = None
        elif gene_module == 'attention':
            from Squidiff.gene_modules import GeneAttention
            self.gene_module = GeneAttention(n_genes=output_dim,
                                             embed_dim=gene_embed_dim,
                                             n_heads=gene_heads)
        elif gene_module == 'mlp':
            from Squidiff.gene_modules import GeneMLP
            self.gene_module = GeneMLP(n_genes=output_dim,
                                       bottleneck=gene_bottleneck)
        else:
            raise ValueError("gene_module must be 'none', 'attention' or 'mlp', got %r"
                             % gene_module)
        
    def forward(self, x, timesteps=None, **model_kwargs):
        
        
        if self.time_embed is not None and timesteps is not None:
            
            emb = self.time_embed(timestep_embedding(timesteps, self.hidden_sizes))
            
        else:
            emb = None
            
        if self.use_encoder: 
            if 'z_mod' in model_kwargs.keys():
                z_sem = model_kwargs['z_mod']
            elif self.num_classes is None:
                z_sem = self.encoder(model_kwargs['x_start'],label = None,drug_dose = model_kwargs['drug_dose'],control_feature = model_kwargs['control_feature'])
            else: 
                z_sem = self.encoder(model_kwargs['x_start'],label = model_kwargs['group'],drug_dose = model_kwargs['drug_dose'],control_feature = model_kwargs['control_feature'])

            h = self.input_layer(x)
            
            h = self.mlp_blocks(x=h, emb=emb, z_sem=z_sem)
            h = self.output_layer(h)
        else:
            z_sem = None
            h = self.input_layer(x)
            h = self.mlp_blocks(x=h, emb=emb, z_sem=z_sem)
            h = self.output_layer(h)
        if self.gene_module is not None:
            h = self.gene_module(h)
        return h

    
class EncoderMLPModel(nn.Module):

    def __init__(self, input_size, hidden_sizes, num_classes=None, use_drug_structure=False, drug_dimension=1024,comb_num=1,output_size=60, dropout=0.1, use_fp16=False,
                 encoder_arm='mlp', memory_size=0):
        super(EncoderMLPModel, self).__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.dtype = th.float16 if use_fp16 else th.float32
        self.drug_dimension = drug_dimension
    
        if num_classes is None:
            l1 = 0
        else: 
            l1 = hidden_sizes
        if use_drug_structure:
            l2 = drug_dimension
        else:
            l2 = 0
        
        self.fc1 = nn.Linear(input_size+l1+l2, hidden_sizes)
        self.bn1 = nn.BatchNorm1d(hidden_sizes)
        self.bn2 = nn.BatchNorm1d(hidden_sizes)
        self.fc2 = nn.Linear(hidden_sizes, hidden_sizes)
        self.fc3 = nn.Linear(hidden_sizes, output_size)
       
        self.label_embed = nn.Linear(1, hidden_sizes)
        # Step 13 (PREREG_step13_attention.md): an optional module that maps the drug
        # input (and the control feature) to a residual of the drug input's width.
        # 'mlp' builds nothing, so existing checkpoints load strictly as before.
        from Squidiff.drug_attention import build_arm
        self.encoder_arm = encoder_arm
        self.arm = build_arm(encoder_arm, drug_dimension, input_size, memory_size)
    
    def forward(self, x_start, label=None, drug_dose=None, control_feature = None):
        
        if label is not None:
            label_emb = self.label_embed(label)
            x_start = th.concat([x_start,label_emb],axis=1)
        
        if drug_dose is not None:
            if self.arm is not None:
                # Step 13: the arm's residual enters the unchanged MLP through the same
                # fc1 column block the fingerprint uses (zero at initialisation).
                drug_dose = drug_dose + self.arm(drug_dose, control_feature)
            x_start = th.concat([control_feature,drug_dose],axis=1)
            
        h = x_start.type(self.dtype)
        h = F.relu(self.bn1(self.fc1(h)))
        h = F.relu(self.bn2(self.fc2(h)))
        h = self.fc3(h)
        return h
    

    
class EncoderMLPModel2(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes=None, output_size=60, dropout=0.1, use_fp16=False):
        super(EncoderMLPModel2, self).__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.dropout = dropout
        self.dtype = th.float16 if use_fp16 else th.float32
        
        self.fc1 = nn.Linear(input_size, hidden_sizes)
        self.bn1 = nn.BatchNorm1d(hidden_sizes)
        self.fc2 = nn.Linear(hidden_sizes, hidden_sizes)
        self.bn2 = nn.BatchNorm1d(hidden_sizes)
        self.fc3 = nn.Linear(hidden_sizes, output_size)
        self.dropout_layer = nn.Dropout(dropout)
        
        self.label_embed = nn.Linear(1, hidden_sizes)  

    def forward(self, x_start, label=None):
        h = x_start.type(self.dtype)
        h = F.relu(self.bn1(self.fc1(h)))
    

        if label is not None:
            label = label.type(self.dtype) 
            label_emb = self.label_embed(label)
            
            h = h + label_emb  # Add label embedding as a residual connection


        h = self.dropout_layer(h)
        h = self.fc3(h)
       
        return h