"""Conditional flow matching for Squidiff. PREREG_step6_flow_matching.md.

Written standalone rather than importing torchcfm: the loss is short, the conda
environment is frozen, and nothing here touches Squidiff/diffusion.py, so arm A
(DDIM) remains byte-identical to Steps 4, 4b and 5.

PATH. Straight-line interpolation x_t = (1-t)*x_0 + t*x_1 with constant
conditional velocity target v = x_1 - x_0. Chosen over a Gaussian diffusion
path because the target is constant along the path and straight paths are the
basis of the low-NFE claim.

TIME. Sampled uniformly on [0, 1] and rescaled to [0, 1000) before the existing
sinusoidal timestep embedding, so the network's conditioning path is unchanged.

WHAT THE NETWORK PREDICTS. Velocity, not epsilon. The architecture is
identical; only the regression target and the sampler differ.
"""
import numpy as np
import torch as th


def _t_to_embedding_scale(t, n_steps=1000):
    """Map continuous t in [0,1] onto the discrete embedding range.

    The existing timestep_embedding is sinusoidal over integer-scaled inputs;
    passing a float tensor is well defined and keeps the conditioning path
    identical between arms.
    """
    return t.float() * float(n_steps)


def ot_couple(x0, x1):
    """Exact minibatch optimal-transport coupling by squared Euclidean cost.

    For equal batch sizes the linear assignment problem gives the exact OT map
    on the empirical measures, which is what OT-CFM approximates. Returns x0
    permuted to align with x1.
    """
    from scipy.optimize import linear_sum_assignment
    if x0.shape[0] != x1.shape[0]:
        raise ValueError('OT coupling needs equal batch sizes, got %d and %d'
                         % (x0.shape[0], x1.shape[0]))
    with th.no_grad():
        cost = th.cdist(x0.float(), x1.float()) ** 2
        r, c = linear_sum_assignment(cost.detach().cpu().numpy())
        perm = th.as_tensor(r[np.argsort(c)], device=x0.device, dtype=th.long)
    return x0[perm]


class ConditionalFlowMatcher:
    """Straight-path CFM. sigma > 0 adds Gaussian smoothing around the path."""

    def __init__(self, sigma=0.0, ot=False):
        self.sigma = float(sigma)
        self.ot = bool(ot)

    def sample(self, x0, x1, t=None):
        """Return (t, x_t, target velocity).

        x0: source sample. Gaussian noise for arm B, control cells for arm C.
        x1: target sample, the observed treated expression profile.
        """
        if x0.shape != x1.shape:
            raise ValueError('shape mismatch: %s vs %s' % (x0.shape, x1.shape))
        if self.ot:
            x0 = ot_couple(x0, x1)
        if t is None:
            t = th.rand(x0.shape[0], device=x0.device)
        tb = t.view(-1, *([1] * (x0.dim() - 1)))
        xt = (1.0 - tb) * x0 + tb * x1
        if self.sigma > 0:
            xt = xt + self.sigma * th.randn_like(xt)
        return t, xt, x1 - x0


@th.no_grad()
def ode_sample(model, x0, n_steps=20, solver='euler', model_kwargs=None,
               clamp_min=0.0, n_embed_steps=1000,
               null_kwargs=None, guidance_w=None, guide_t_min=0.0):
    """Integrate dx/dt = v(x,t) from t=0 to t=1.

    n_steps is the number of function evaluations for Euler; Heun uses two per
    step and is reported separately. The non-negativity clamp is applied ONCE at
    t=1, not per step: log1p expression is non-negative, but clamping inside the
    integration would alter the learned flow. This differs from arm A, which
    clamps at every reverse step - an asymmetry intrinsic to the two samplers
    and recorded as a limitation in the pre-registration, not controlled.
    """
    if model_kwargs is None: model_kwargs = {}
    if solver not in ('euler', 'heun'):
        raise ValueError("solver must be 'euler' or 'heun', got %r" % solver)
    # Step 12 (PREREG_step12_guidance.md s.3): classifier-free guidance. With
    # guidance_w set, each step evaluates the model twice, with the real
    # condition (model_kwargs) and the null condition (null_kwargs), and steps
    # along v_null + w * (v_cond - v_null). w = 1 reproduces the unguided path
    # up to rounding; guide_t_min > 0 restricts guidance to t >= guide_t_min
    # (the exploratory half-trajectory variant). Euler only.
    if guidance_w is not None:
        if null_kwargs is None:
            raise ValueError('guidance_w requires null_kwargs')
        if solver != 'euler':
            raise ValueError('guidance is implemented for the euler solver only')
        guidance_w = float(guidance_w)
    x = x0.clone()
    dt = 1.0 / float(n_steps)
    for i in range(n_steps):
        t = th.full((x.shape[0],), i * dt, device=x.device)
        v = model(x, _t_to_embedding_scale(t, n_embed_steps), **model_kwargs)
        if guidance_w is not None and i * dt >= guide_t_min:
            v_null = model(x, _t_to_embedding_scale(t, n_embed_steps), **null_kwargs)
            v = v_null + guidance_w * (v - v_null)
        if solver == 'euler':
            x = x + v * dt
        else:
            t2 = th.full((x.shape[0],), min((i + 1) * dt, 1.0), device=x.device)
            v2 = model(x + v * dt, _t_to_embedding_scale(t2, n_embed_steps), **model_kwargs)
            x = x + 0.5 * (v + v2) * dt
    if clamp_min is not None:
        x = x.clamp(min=clamp_min)
    return x


def fm_loss(model, x0, x1, matcher, model_kwargs=None, n_embed_steps=1000):
    """Conditional flow matching loss: MSE between predicted and target velocity."""
    if model_kwargs is None: model_kwargs = {}
    t, xt, target = matcher.sample(x0, x1)
    pred = model(xt, _t_to_embedding_scale(t, n_embed_steps), **model_kwargs)
    if pred.shape != target.shape:
        raise ValueError('model output %s does not match target %s'
                         % (tuple(pred.shape), tuple(target.shape)))
    return ((pred - target) ** 2).mean()


class FlowMatchingWrapper:
    """Exposes `training_losses` with the diffusion signature, so TrainLoop is
    used unchanged and arms A, B and C share one training path byte-for-byte.

    TrainLoop calls  self.diffusion.training_losses(model, micro, t, model_kwargs)
    and expects {'loss': tensor}. The `t` it passes comes from ScheduleSampler
    over discrete diffusion steps; flow matching needs t ~ U[0,1], so that
    argument is IGNORED and a continuous t is drawn here. The importance weights
    TrainLoop applies are uniform under the default sampler, so the product is
    unaffected -- verify this if a non-uniform sampler is ever configured.

    ARM B: source='gaussian'.  ARM C: source='control', ot=True.
    """

    num_timesteps = 1000  # only for TrainLoop's logging helpers

    def __init__(self, source='gaussian', ot=False, sigma=0.0, n_embed_steps=1000,
                 cond_drop=0.0):
        if source not in ('gaussian', 'control'):
            raise ValueError("source must be 'gaussian' or 'control', got %r" % source)
        if not (0.0 <= float(cond_drop) < 1.0):
            raise ValueError('cond_drop must be in [0, 1), got %r' % cond_drop)
        self.cond_drop = float(cond_drop)       # Step 12: P(fingerprint hidden) per row
        self._calls = 0; self._rows = 0; self._dropped = 0
        self.source = source
        self.matcher = ConditionalFlowMatcher(sigma=sigma, ot=ot)
        self.n_embed_steps = n_embed_steps

    def training_losses(self, model, x_start, t, model_kwargs=None):
        # MLPModel.forward reads model_kwargs['x_start'] to build z_sem. Arm A's
        # GaussianDiffusion.training_losses inserts it at diffusion.py:756, and it
        # is the CLEAN treated profile, not x_t. Flow matching must pass the same
        # thing or the arms differ in conditioning as well as objective.
        # Copy rather than mutate: arm C reads control_feature from this dict.
        model_kwargs = dict(model_kwargs or {})
        model_kwargs['x_start'] = x_start
        # Step 12 (PREREG_step12_guidance.md s.3): with probability cond_drop a
        # row's fingerprint is replaced by the all-zero vector, never a real
        # condition here (every training target has dose >= 10 and the encoder
        # scales by log10(dose+1)). MLPModel.forward hands model_kwargs['drug_dose']
        # to the semantic encoder, whose input is concat([control_feature,
        # drug_dose]) (MLPModel.py:283), so zeroing here zeros the fingerprint half
        # of the encoder input. Cloned: TrainLoop's micro-batch is a view.
        if self.cond_drop > 0:
            dd0 = model_kwargs['drug_dose']
            mask = th.rand(dd0.shape[0], device=dd0.device) < self.cond_drop
            dd = dd0.clone(); dd[mask] = 0
            if dd[mask].abs().sum().item() != 0: raise RuntimeError('cond_drop: masked rows not zero')
            if not th.equal(dd[~mask], dd0[~mask]): raise RuntimeError('cond_drop: unmasked rows changed')
            model_kwargs['drug_dose'] = dd
            self._calls += 1; self._rows += int(dd.shape[0]); self._dropped += int(mask.sum())
            if self._calls % 10000 == 0:
                print('[cond_drop] call %d  empirical rate %.4f (target %.2f)'
                      % (self._calls, self._dropped / self._rows, self.cond_drop), flush=True)
        if self.source == 'gaussian':
            x0 = th.randn_like(x_start)
        else:
            cf = model_kwargs.get('control_feature')
            if cf is None:
                raise ValueError("source='control' requires control_feature in "
                                 "model_kwargs; check that TrainLoop forwards it")
            if cf.shape != x_start.shape:
                raise ValueError('control_feature %s does not match x_start %s'
                                 % (tuple(cf.shape), tuple(x_start.shape)))
            x0 = cf.to(x_start.device).clone()
        tc, xt, target = self.matcher.sample(x0, x_start)
        pred = model(xt, _t_to_embedding_scale(tc, self.n_embed_steps), **model_kwargs)
        if pred.shape != target.shape:
            raise ValueError('model output %s != target %s'
                             % (tuple(pred.shape), tuple(target.shape)))
        return {'loss': ((pred - target) ** 2).mean(dim=list(range(1, target.dim())))}
