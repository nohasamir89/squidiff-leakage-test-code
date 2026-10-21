"""Step 13: attention on the drug condition - four encoder residual modules.

Each module maps the drug input (and, for the cross-attention arm, the control
feature) to a residual r of the SAME width as the drug input (1024 for the Morgan
fingerprint). EncoderMLPModel then feeds  drug_dose + r  into its unchanged MLP.

Design rules shared by all four, and why they are there:
  * The LAST projection of every module is zero-initialised (as adaLN-Zero was in
    Step 4). At initialisation every arm is bitwise the reference encoder; whatever
    differs at the end of training was learned, not injected.
  * The residual lives in the fingerprint space, so encoder.fc1.weight keeps its
    (hidden, gene_size + drug_dimension) shape. Both samplers infer the drug encoding
    from that shape, and arm identity is inferred from the extra state-dict keys.
  * A row whose drug input is all zeros (a vehicle cell, or a hidden condition under
    Step 12's cond_drop) yields r = 0 in the fingerprint self-attention arm by
    construction (no tokens); the other arms are not zero there, which is why this
    step trains WITHOUT cond_drop and guidance is out of scope (see PREREG_step13).

Arms
  attn_fp     FingerprintSelfAttention  - tokens are the active fingerprint bits
  attn_mem    CompoundMemoryAttention   - attention over the split's training compounds
  attn_cross  DrugCellCrossAttention    - the drug queries tokens made from the control cell
  mlp_ctrl    MatchedMLPResidual        - non-attention residual matched to attn_cross's size
  mlp         (no module)               - the reference encoder, r = 0
"""
import math
import torch as th
import torch.nn as nn
import torch.nn.functional as F

ARMS = ('mlp', 'attn_fp', 'attn_mem', 'attn_cross', 'mlp_ctrl')
D_MODEL = 64          # width of every attention module
N_HEADS = 4
MAX_ACTIVE_BITS = 256 # token budget of the fingerprint arm; asserted, never silently truncated
N_CELL_TOKENS = 16    # cross-attention arm: tokens made from the control feature
CTRL_HIDDEN = 264     # control arm width; matches attn_cross's parameter count to within 0.2%


def _zero_linear(d_in, d_out):
    """A linear layer whose weight AND bias start at zero (adaLN-Zero convention)."""
    lin = nn.Linear(d_in, d_out)
    nn.init.zeros_(lin.weight)
    nn.init.zeros_(lin.bias)
    return lin


class FingerprintSelfAttention(nn.Module):
    """Arm attn_fp. Each active bit of the fingerprint is a token (a learned 64-dim
    embedding of that bit's identity, multiplied by the dose scale the fingerprint
    already carries). One transformer layer lets substructures attend to each other;
    a masked mean over the active tokens is projected back to fingerprint width."""

    def __init__(self, drug_dim=1024, d=D_MODEL, heads=N_HEADS, max_active=MAX_ACTIVE_BITS):
        super().__init__()
        self.drug_dim, self.max_active = drug_dim, max_active
        self.bit_embed = nn.Embedding(drug_dim, d)
        self.layer = nn.TransformerEncoderLayer(d_model=d, nhead=heads, dim_feedforward=4 * d,
                                                dropout=0.0, batch_first=True)
        self.out = _zero_linear(d, drug_dim)

    def forward(self, drug_dose, control_feature=None):
        mask = drug_dose != 0                                       # (B, 1024) active bits
        n_active = mask.sum(1)
        if int(n_active.max()) > self.max_active:
            raise RuntimeError('FingerprintSelfAttention: %d active bits exceed the token budget %d'
                               % (int(n_active.max()), self.max_active))
        # positions of the active bits, padded with inactive positions (topk on the mask
        # returns the ones first; order among them is irrelevant to attention)
        idx = th.topk(mask.float(), self.max_active, dim=1).indices  # (B, L)
        valid = mask.gather(1, idx)                                  # (B, L) True where a real token
        # the dose scale: every active bit carries the same value log10(dose + 1)
        scale = drug_dose.gather(1, idx)                             # (B, L), 0 on padding
        tok = self.bit_embed(idx) * scale.unsqueeze(-1)              # (B, L, d)
        # rows with no active bits would give an all-True padding mask (NaN in softmax);
        # give them one dummy valid slot and zero the result afterwards
        empty = n_active == 0
        pad = ~valid
        pad[empty, 0] = False
        h = self.layer(tok, src_key_padding_mask=pad)               # (B, L, d)
        w = valid.float().unsqueeze(-1)
        pooled = (h * w).sum(1) / w.sum(1).clamp(min=1.0)           # masked mean over real tokens
        r = self.out(pooled)
        r = r.masked_fill(empty.unsqueeze(-1), 0.0)                  # no tokens -> no residual
        return r


class CompoundMemoryAttention(nn.Module):
    """Arm attn_mem. A memory with one slot per TRAINING compound of the split: the key
    is a learned projection of that compound's (unscaled) fingerprint, the value a free
    learned vector. The input drug's fingerprint is projected to a query; the softmax
    over keys blends the values, so an unseen compound is represented through the seen
    compounds it resembles. keys_fp is a buffer filled by the training script from the
    training compounds only, and saved inside the checkpoint."""

    def __init__(self, memory_size, drug_dim=1024, d=D_MODEL):
        super().__init__()
        self.d = d
        self.register_buffer('keys_fp', th.zeros(int(memory_size), drug_dim))
        self.register_buffer('memory_filled', th.zeros(1))          # 1 once the training script filled keys_fp
        self.q = nn.Linear(drug_dim, d)
        self.k = nn.Linear(drug_dim, d)
        self.values = nn.Parameter(th.randn(int(memory_size), d) * 0.02)
        self.out = _zero_linear(d, drug_dim)

    def forward(self, drug_dose, control_feature=None):
        if float(self.memory_filled) != 1.0:
            raise RuntimeError('CompoundMemoryAttention: keys_fp was never filled (training script must call fill_memory)')
        q = self.q(drug_dose)                                        # (B, d)
        k = self.k(self.keys_fp)                                     # (M, d)
        att = th.softmax(q @ k.t() / math.sqrt(self.d), dim=-1)      # (B, M)
        return self.out(att @ self.values)                           # (B, 1024)

    @th.no_grad()
    def fill_memory(self, fps):
        fps = th.as_tensor(fps, dtype=self.keys_fp.dtype)
        if tuple(fps.shape) != tuple(self.keys_fp.shape):
            raise ValueError('fill_memory: got %s, buffer is %s' % (tuple(fps.shape), tuple(self.keys_fp.shape)))
        self.keys_fp.copy_(fps)
        self.memory_filled.fill_(1.0)


class DrugCellCrossAttention(nn.Module):
    """Arm attn_cross. The control feature (the cell's baseline expression) is projected
    into N_CELL_TOKENS tokens; the drug fingerprint becomes a single query that attends
    over them. The first multiplicative drug x cell-state interaction in the encoder."""

    def __init__(self, drug_dim=1024, gene_size=200, d=D_MODEL, heads=N_HEADS, n_tokens=N_CELL_TOKENS):
        super().__init__()
        self.n_tokens, self.d = n_tokens, d
        self.q = nn.Linear(drug_dim, d)
        self.cell_tokens = nn.Linear(gene_size, n_tokens * d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.out = _zero_linear(d, drug_dim)

    def forward(self, drug_dose, control_feature):
        B = drug_dose.shape[0]
        q = self.q(drug_dose).unsqueeze(1)                           # (B, 1, d)
        kv = self.cell_tokens(control_feature).view(B, self.n_tokens, self.d)
        h, _ = self.attn(q, kv, kv, need_weights=False)              # (B, 1, d)
        return self.out(h.squeeze(1))


class MatchedMLPResidual(nn.Module):
    """Control arm mlp_ctrl. A plain residual MLP on the fingerprint with no attention,
    sized to DrugCellCrossAttention's parameter count, so that an attention arm's gain
    can be told apart from the gain of extra parameters plus a new pathway (Step 4b)."""

    def __init__(self, drug_dim=1024, hidden=CTRL_HIDDEN, d=D_MODEL):
        super().__init__()
        self.fc1 = nn.Linear(drug_dim, hidden)
        self.fc2 = nn.Linear(hidden, d)
        self.out = _zero_linear(d, drug_dim)

    def forward(self, drug_dose, control_feature=None):
        return self.out(F.relu(self.fc2(F.relu(self.fc1(drug_dose)))))


def build_arm(arm, drug_dim, gene_size, memory_size=0):
    """Construct the module for an arm name; None for the reference arm."""
    if arm not in ARMS:
        raise ValueError('encoder_arm must be one of %s, got %r' % (ARMS, arm))
    if arm == 'mlp':
        return None
    if arm == 'attn_fp':
        return FingerprintSelfAttention(drug_dim)
    if arm == 'attn_mem':
        if int(memory_size) <= 0:
            raise ValueError('attn_mem needs memory_size = number of training compounds')
        return CompoundMemoryAttention(int(memory_size), drug_dim)
    if arm == 'attn_cross':
        return DrugCellCrossAttention(drug_dim, gene_size)
    return MatchedMLPResidual(drug_dim)


def infer_arm_from_state_dict(sd, prefix='encoder.arm.'):
    """Read the arm (and the memory size) from a checkpoint's keys, the way the samplers
    read use_adaln and the drug encoding. Returns (arm, memory_size)."""
    keys = [k[len(prefix):] for k in sd.keys() if k.startswith(prefix)]
    if not keys:
        return 'mlp', 0
    if any(k.startswith('bit_embed.') for k in keys):
        return 'attn_fp', 0
    if any(k == 'keys_fp' for k in keys):
        return 'attn_mem', int(sd[prefix + 'keys_fp'].shape[0])
    if any(k.startswith('cell_tokens.') for k in keys):
        return 'attn_cross', 0
    if any(k.startswith('fc1.') for k in keys):
        return 'mlp_ctrl', 0
    raise ValueError('unrecognised encoder.arm keys: %s' % keys[:5])
