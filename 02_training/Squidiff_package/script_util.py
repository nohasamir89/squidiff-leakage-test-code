import argparse

from . import diffusion
from .respace import SpacedDiffusion, space_timesteps
from .MLPModel import MLPModel

NUM_CLASSES = 4


def diffusion_defaults():
    """
    Defaults for image and classifier training.
    """
    return dict(
        learn_sigma=False,
        diffusion_steps=1000,
        noise_schedule="linear",
        timestep_respacing="",
        use_kl=False,
        predict_xstart=False,
        rescale_timesteps=False,
        rescale_learned_sigmas=False,
    )


def model_and_diffusion_defaults():
    """
    Defaults for training.
    """
    res = dict(
        output_dim=1000,
        num_layers=3,
        gene_size=1000,
        num_channels=128,
        dropout=0.0,
        class_cond=False,
        use_checkpoint=False,
        use_scale_shift_norm=True,
        use_fp16=False,
        use_encoder=False,
        use_drug_structure=False,
        drug_dimension=1024,
        comb_num=1,
        use_adaln=False,
        use_adaln_timeonly=False,
        encoder_arm='mlp',
        memory_size=0,
        gene_module='none',
        gene_embed_dim=512,
        gene_heads=16,
        gene_bottleneck=8120,
    )
    res.update(diffusion_defaults())
    return res


def create_model_and_diffusion(
    gene_size,
    num_layers,
    output_dim,
    class_cond,
    learn_sigma,
    num_channels,
    dropout,
    diffusion_steps,
    noise_schedule,
    timestep_respacing,
    use_kl,
    predict_xstart,
    rescale_timesteps,
    rescale_learned_sigmas,
    use_checkpoint,
    use_scale_shift_norm,
    use_fp16,
    use_encoder,
    use_drug_structure,
    drug_dimension,
    comb_num,
    use_adaln,
    use_adaln_timeonly,
    encoder_arm,
    memory_size,
    gene_module,
    gene_embed_dim,
    gene_heads,
    gene_bottleneck,
):
    model = create_model(
        gene_size,
        num_layers,
        output_dim,
        learn_sigma=learn_sigma,
        class_cond=class_cond,
        use_checkpoint=use_checkpoint,
        use_scale_shift_norm=use_scale_shift_norm,
        dropout=dropout,
        use_fp16=use_fp16,
        use_encoder=use_encoder,
        use_drug_structure = use_drug_structure,
        drug_dimension = drug_dimension,
        comb_num=comb_num,
        use_adaln=use_adaln,
        use_adaln_timeonly=use_adaln_timeonly,
        encoder_arm=encoder_arm,
        memory_size=memory_size,
        gene_module=gene_module,
        gene_embed_dim=gene_embed_dim,
        gene_heads=gene_heads,
        gene_bottleneck=gene_bottleneck,
    )
    diffusion = create_gaussian_diffusion(
        steps=diffusion_steps,
        learn_sigma=learn_sigma,
        noise_schedule=noise_schedule,
        use_kl=use_kl,
        predict_xstart=predict_xstart,
        rescale_timesteps=rescale_timesteps,
        rescale_learned_sigmas=rescale_learned_sigmas,
        timestep_respacing=timestep_respacing,
        use_encoder=use_encoder
    )
    return model, diffusion


def create_model(
    gene_size,
    num_layers,
    output_dim,
    learn_sigma=False,
    class_cond=False,
    use_checkpoint=False,
    use_scale_shift_norm=False,
    dropout=0,
    use_fp16=False,
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

    return MLPModel(
        gene_size  = gene_size,
        output_dim = output_dim,
        num_layers = num_layers,
        dropout=dropout,
        num_classes=(NUM_CLASSES if class_cond else None),
        use_checkpoint=use_checkpoint,
        use_fp16=use_fp16,
        use_scale_shift_norm=use_scale_shift_norm,
        use_encoder = use_encoder,
        use_drug_structure = use_drug_structure,
        drug_dimension = drug_dimension,
        comb_num=comb_num,
        use_adaln=use_adaln,
        use_adaln_timeonly=use_adaln_timeonly,
        encoder_arm=encoder_arm,
        memory_size=memory_size,
        gene_module=gene_module,
        gene_embed_dim=gene_embed_dim,
        gene_heads=gene_heads,
        gene_bottleneck=gene_bottleneck,
    )


# Removed for the manuscript release (21 Oct 2026): the upstream guided-diffusion
# classifier and super-resolution builders (classifier_defaults,
# classifier_and_diffusion_defaults, create_classifier_and_diffusion, create_classifier,
# sr_model_and_diffusion_defaults, sr_create_model_and_diffusion, sr_create_model) were
# never called by any script in this repository and referenced names that do not exist
# here (EncoderUNetModel, SuperResModel); they would have raised NameError if used.


def create_gaussian_diffusion(
    *,
    steps=1000,
    learn_sigma=False,
    sigma_small=False,
    noise_schedule="linear",
    use_kl=False,
    predict_xstart=False,
    rescale_timesteps=False,
    rescale_learned_sigmas=False,
    timestep_respacing="",
    use_encoder = False
):
    print('diffusion num of steps = ',steps)
    betas = diffusion.get_named_beta_schedule(noise_schedule, steps)
    if use_kl:
        loss_type = diffusion.LossType.RESCALED_KL
    elif rescale_learned_sigmas:
        loss_type = diffusion.LossType.RESCALED_MSE
    else:
        loss_type = diffusion.LossType.MSE
    if not timestep_respacing:
        timestep_respacing = [steps]
        
    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            diffusion.ModelMeanType.EPSILON if not predict_xstart else diffusion.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                diffusion.ModelVarType.FIXED_LARGE
                if not sigma_small
                else diffusion.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else diffusion.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
        use_encoder=use_encoder
    )


def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)


def args_to_dict(args, keys):
    return {k: args[k] for k in keys if k in args}


def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")
