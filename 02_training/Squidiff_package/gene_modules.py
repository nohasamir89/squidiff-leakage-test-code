"""Gene-level mixing modules for Step 5.

WHERE THESE ATTACH, AND WHY
MLPModel maps 200 -> 2048 (input_layer), runs three MLPBlocks entirely in
2048-d, then maps 2048 -> 200 (output_layer). input_layer is DENSE, so hidden
units carry no gene correspondence and 2048 does not factor into 200 tokens:
there is no point inside the block stack where gene tokens exist. These modules
therefore attach AFTER output_layer, as a residual refinement of the predicted
200-gene profile.

PERMUTATION EQUIVARIANCE
Attention over 200 tokens whose only content is a scalar cannot distinguish
gene i from gene j. A learned per-gene embedding is required, analogous to
positional encoding. Without it the module cannot represent gene identity and
the hypothesis is not testable.

IDENTITY AT INITIALISATION
Both modules end in a zero-initialised projection and are applied residually,
so a model built with either is numerically identical to the un-augmented model
at step 0, mirroring the adaLN-Zero construction from Step 4.

WHAT THE CONTROL CONTROLS FOR
Both allow all-to-all gene interaction and are parameter matched. The attention
module's mixing weights are INPUT DEPENDENT; the MLP's are fixed once trained.
The contrast isolates dynamic content-dependent mixing from static learned
mixing, NOT interaction versus no interaction.

PARAMETER COUNTS (n_genes=200, ffn_mult=4)
  GeneAttention:  200E + 12E^2 + 16E + 1
  GeneMLP:        401b + 200
  At E=512: 3,256,321.  Matched bottleneck b=8120 gives 3,256,320.
  That is 5.60% of the 58,178,180-parameter adaLN model, comparable to the
  7.3% that adaLN itself added.
"""
import torch as th
import torch.nn as nn
import torch.nn.functional as F


class GeneAttention(nn.Module):
    """Multi-head self-attention over n_genes tokens."""

    def __init__(self, n_genes=200, embed_dim=512, n_heads=16, ffn_mult=4):
        super().__init__()
        if embed_dim % n_heads:
            raise ValueError('embed_dim %d not divisible by n_heads %d' % (embed_dim, n_heads))
        self.n_genes, self.embed_dim = n_genes, embed_dim
        self.gene_embed = nn.Parameter(th.randn(n_genes, embed_dim) * 0.02)
        self.value_proj = nn.Linear(1, embed_dim)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_mult * embed_dim), nn.GELU(),
            nn.Linear(ffn_mult * embed_dim, embed_dim))
        self.out_proj = nn.Linear(embed_dim, 1)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x):
        if x.shape[-1] != self.n_genes:
            raise ValueError('expected %d genes, got %d' % (self.n_genes, x.shape[-1]))
        z = self.value_proj(x.unsqueeze(-1)) + self.gene_embed
        h = self.ln1(z)
        a, _ = self.attn(h, h, h, need_weights=False)
        z = z + a
        z = z + self.ffn(self.ln2(z))
        return x + self.out_proj(z).squeeze(-1)


class GeneMLP(nn.Module):
    """Parameter-matched control: fixed all-to-all gene mixing, no attention."""

    def __init__(self, n_genes=200, bottleneck=8120):
        super().__init__()
        self.n_genes = n_genes
        self.fc1 = nn.Linear(n_genes, bottleneck)
        self.fc2 = nn.Linear(bottleneck, n_genes)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        if x.shape[-1] != self.n_genes:
            raise ValueError('expected %d genes, got %d' % (self.n_genes, x.shape[-1]))
        return x + self.fc2(F.gelu(self.fc1(x)))
