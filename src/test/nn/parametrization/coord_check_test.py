import logging
from contextlib import contextmanager
from functools import partial
from typing import Callable, Dict, List, Optional, Tuple, Type

import numpy as np
import pytest
import torch

from olmo_core.nn.attention import AttentionConfig, AttentionType
from olmo_core.nn.cross_entropy_loss import CrossEntropyLoss
from olmo_core.nn.feed_forward import FeedForwardConfig
from olmo_core.nn.moe.moe import MoEConfig
from olmo_core.nn.moe.router import MoERouterConfig
from olmo_core.nn.parametrization import MupScalingStrategy, ParametrizationConfig
from olmo_core.nn.parametrization.config import ParametrizationType
from olmo_core.nn.parametrization.parametrization import ParametrizationBase
from olmo_core.nn.transformer.config import TransformerBlockType, TransformerConfig
from olmo_core.nn.transformer.init import InitMethod
from olmo_core.optim.adam import AdamConfig
from olmo_core.optim.adamw import AdamWConfig, SkipStepAdamWConfig
from olmo_core.optim.config import OptimConfig, OptimGroupOverride
from olmo_core.optim.scheduler import LinearWithWarmup
from olmo_core.testing.utils import requires_gpu

log = logging.getLogger(__name__)


def get_transformer_config(
    parametrization_config: Optional[ParametrizationConfig] = None,
    vocab_size: int = 100,
    init_seed: int = 0,
    d_model: int = 8,
    n_heads: int = 2,
    **kwargs,
) -> TransformerConfig:
    return TransformerConfig.llama_like(
        d_model=d_model,
        vocab_size=vocab_size,
        n_layers=2,
        n_heads=n_heads,
        hidden_size_multiplier=1,
        hidden_size_multiple_of=16,
        block_name=TransformerBlockType.reordered_norm,
        qk_norm=True,
        rope_theta=500_000,
        layer_norm_eps=1e-6,
        fused_ops=False,
        use_flash=False,
        parametrization=parametrization_config,
        init_seed=init_seed,
        **kwargs,
    )


def get_transformer_inputs(vocab_size: int = 100) -> torch.Tensor:
    return torch.arange(0, vocab_size).unsqueeze(0)


@contextmanager
def _submodule_output_collection(model: torch.nn.Module):
    submodule_outputs: Dict[str, torch.Tensor] = {}

    def collect_submodule_outputs(
        debug_state: Dict[str, torch.Tensor],
        name: str,
        module: torch.nn.Module,
        args,
        output,
    ):
        del module
        if isinstance(output, torch.Tensor):
            state_name = f"{name}|{len(debug_state)}|output"
            debug_state[state_name] = output.detach()

    handles = []
    try:
        for name, module in model.named_modules():
            handles.append(
                module.register_forward_hook(
                    partial(
                        collect_submodule_outputs,
                        submodule_outputs,
                        name,
                    )
                )
            )
        yield submodule_outputs
    finally:
        for handle in handles:
            handle.remove()


def train_and_collect_coordinate_data(
    model_generator: Callable[[int, int], Tuple[int, torch.nn.Module]],
    optim_config: OptimConfig,
    num_widths: int,
    train_batch_size: int = 1,
    eval_batch_size: int = 8,
    sequence_length: int = 16,
    vocab_size: int = 100,
    steps: int = 50,
    seeds: int = 10,
    has_embedding_dim: bool = False,
    device: Optional[torch.device] = None,
) -> Tuple[Dict[str, List[List[float]]], Dict[str, List[List[float]]]]:
    """
    Trains a series of models and collect coordinate data for them.

    First piece of parametrization data is average abs value of elements at initialization
    (a.k.a. "coordinate magnitudes" at init).

    Second piece of parametrization data is std deviation of the change in "coordinates"
    from initialization to end of training.

    Outputs are returns as a Dict[str, List[List]], where the dictionary key is the activation name,
    the outer list is for each width, and the inner list is for each seed.
    """
    device = device or torch.device("cpu")

    def _get_input(*, is_training: bool = True, input_dim: Optional[int] = None) -> torch.Tensor:
        batch_size = train_batch_size if is_training else eval_batch_size
        if has_embedding_dim:
            assert input_dim is not None
            return torch.randn(batch_size, sequence_length, input_dim)

        return torch.randint(0, vocab_size, (batch_size, sequence_length))

    seeded_initial_submodule_outputs: List[List[Dict[str, torch.Tensor]]] = []
    seeded_final_submodule_outputs: List[List[Dict[str, torch.Tensor]]] = []

    # Dict[activation_name, [<field>_for_different_multipliers_and_seeds]]

    for width_idx in range(num_widths):
        width_initial_submodule_outputs = []
        width_final_submodule_outputs = []
        seeded_initial_submodule_outputs.append(width_initial_submodule_outputs)
        seeded_final_submodule_outputs.append(width_final_submodule_outputs)

        for seed_idx in range(seeds):
            input_dim, model = model_generator(width_idx, seed_idx)

            optim = optim_config.build(model)
            scheduler = LinearWithWarmup(warmup=0)

            with _submodule_output_collection(model) as submodule_outputs:
                _ = model(_get_input(is_training=False, input_dim=input_dim).to(device=device))
                width_initial_submodule_outputs.append(submodule_outputs)

            # Train for a bit
            for i_step in range(steps):
                inp = _get_input(is_training=True, input_dim=input_dim).to(device=device)
                labels = (inp.sum(dim=-1) > 0).long() if has_embedding_dim else inp
                # labels = inp
                optim.zero_grad(set_to_none=True)
                logits = model(inp)

                ce_loss, z_loss = CrossEntropyLoss()(
                    logits,
                    labels,
                )

                # Get loss to optimize for.
                loss = ce_loss
                if z_loss is not None:
                    loss += z_loss

                loss.backward()
                optim.step()

                for param_group in optim.param_groups:
                    new_lr = scheduler.get_lr(
                        param_group[scheduler.initial_lr_field],
                        i_step,
                        steps,
                    )
                    param_group[scheduler.lr_field] = new_lr

            with _submodule_output_collection(model) as submodule_outputs:
                _ = model(_get_input(is_training=False, input_dim=input_dim).to(device=device))
                width_final_submodule_outputs.append(submodule_outputs)

    state_keys = list(seeded_initial_submodule_outputs[0][0].keys())

    # First piece of parametrization data is average abs value of elements at initialization
    # (a.k.a. "coordinate magnitudes" at init).
    seeded_init_coord_magnitudes: Dict[str, List[List[float]]] = {}
    for state_key in state_keys:
        seeded_init_coord_magnitudes[state_key] = []
        for width_initial_submodule_outputs in seeded_initial_submodule_outputs:
            width_init_coord_magnitudes = []
            seeded_init_coord_magnitudes[state_key].append(width_init_coord_magnitudes)
            for initial_submodule_outputs in width_initial_submodule_outputs:
                width_init_coord_magnitudes.append(
                    initial_submodule_outputs[state_key].float().norm(p=1.0).item()
                    / initial_submodule_outputs[state_key].numel()
                )

    # Second piece of parametrization data is std deviation of the change in "coordinates"
    # from initialization to end of training.
    seeded_coord_change_stds: Dict[str, List[List[float]]] = {}
    for state_key in state_keys:
        seeded_coord_change_stds[state_key] = []
        for width_initial_submodule_outputs, width_final_submodule_outputs in zip(
            seeded_initial_submodule_outputs, seeded_final_submodule_outputs
        ):
            width_coord_change_stds = []
            seeded_coord_change_stds[state_key].append(width_coord_change_stds)
            for initial_submodule_outputs, final_submodule_outputs in zip(
                width_initial_submodule_outputs, width_final_submodule_outputs
            ):
                initial_output = initial_submodule_outputs[state_key]
                final_output = final_submodule_outputs[state_key]
                coord_change_std = (final_output - initial_output).float().std()
                width_coord_change_stds.append(coord_change_std.item())

    return seeded_init_coord_magnitudes, seeded_coord_change_stds


def validate_parametrization_data(
    widths: List[int],
    seeded_init_coord_magnitudes: Dict[str, List[List[float]]],
    seeded_coord_change_stds: Optional[Dict[str, List[List[float]]]] = None,
    max_normalized_slope: float = 0.05,
    min_normalized_slope: float = -0.05,
    init_decay_allowlist: Optional[List[str]] = None,
):
    """
    Validate that the parametrization data is good. At this point, this means passing
    the coordinate check. The coordinate check requires that:
    1. Coordinate magnitudes at initialization don't grow/decay too much with width. The exception is
    that output layers are allowed to decay.
    2. Change in coordinates from initialization to end of training don't grow/decay too much with width.
    """

    assert len(seeded_init_coord_magnitudes) > 0

    growing_param_names = []
    decaying_param_names = []
    for name, width_magnitudes_by_seed in seeded_init_coord_magnitudes.items():
        mean_width_magnitudes = np.array(
            [np.mean(width_seed_magnitudes) for width_seed_magnitudes in width_magnitudes_by_seed]
        )

        log_widths = np.log2(np.array(widths))
        slope = np.polynomial.polynomial.Polynomial.fit(
            log_widths, mean_width_magnitudes / mean_width_magnitudes[0], 1
        ).coef[1]
        if slope > max_normalized_slope:
            log.error(
                f"Coordinate magnitudes for {name} grow too much with width. "
                f"slope {slope}, magnitudes {mean_width_magnitudes}, normalized magnitudes {mean_width_magnitudes / mean_width_magnitudes[0]}"
            )
            growing_param_names.append(name)
        elif slope < min_normalized_slope:
            param_name = name.split("|")[0] if "|" in name else None
            if (
                init_decay_allowlist is not None
                and param_name is not None
                and param_name in init_decay_allowlist
            ):
                log.info(
                    f"Coordinate magnitudes for {name} decay too much with width, but '{param_name}' is in decay allowlist. "
                    f"slope {slope}, magnitudes {mean_width_magnitudes}, normalized magnitudes {mean_width_magnitudes / mean_width_magnitudes[0]}"
                )
                continue

            log.error(
                f"Coordinate magnitudes for {name} decay too much with width. "
                f"slope {slope}, magnitudes {mean_width_magnitudes}, normalized magnitudes {mean_width_magnitudes / mean_width_magnitudes[0]}"
            )
            decaying_param_names.append(name)

    assert (
        not growing_param_names and not decaying_param_names
    ), f"Initial coords failed mup check. Growing params: {growing_param_names}, decaying params: {decaying_param_names}"

    if seeded_coord_change_stds is None:
        return

    growing_param_names = []
    decaying_param_names = []
    for name, width_magnitudes_by_seed in seeded_coord_change_stds.items():
        mean_width_stds = np.array(
            [np.mean(width_seed_stds) for width_seed_stds in width_magnitudes_by_seed]
        )

        log_d_models = np.log2(np.array(widths))
        slope = np.polynomial.polynomial.Polynomial.fit(
            log_d_models, mean_width_stds / mean_width_stds[0], 1
        ).coef[1]
        if slope > max_normalized_slope:
            log.error(
                f"Coordinate change stds for {name} grow too much with width. "
                f"slope {slope}, stds {mean_width_stds}, normalized stds {mean_width_stds / mean_width_stds[0]}"
            )
            growing_param_names.append(name)
        elif slope < min_normalized_slope:
            log.error(
                f"Coordinate change stds for {name} decay too much with width. "
                f"slope {slope}, stds {mean_width_stds}, normalized stds {mean_width_stds / mean_width_stds[0]}"
            )
            decaying_param_names.append(name)

    assert (
        not growing_param_names and not decaying_param_names
    ), f"Coordinate change stds failed mup check. Growing params: {growing_param_names}, decaying params: {decaying_param_names}"


@pytest.mark.parametrize(
    "optim_config_cls, parametrization_scaling_strategy",
    [
        (optim_config_cls, scaling_strategy)
        for optim_config_cls in [AdamConfig, AdamWConfig, SkipStepAdamWConfig]
        for scaling_strategy in MupScalingStrategy
    ],
)
@pytest.mark.parametrize("base_d_model, base_hidden_size", [(64, 32), (None, None)])
def test_ffn_coord_check_at_init_with_scaling_input_output(
    optim_config_cls, parametrization_scaling_strategy, base_d_model, base_hidden_size
):
    MIN_NORMALIZED_SLOPE = -0.05
    MAX_NORMALIZED_SLOPE = 0.05
    D_MODELS = [16, 32, 64, 128, 256]
    HIDDEN_SIZES = [10, 20, 30, 40, 50]
    SEEDS = 10
    STEPS = 1

    optim_config: OptimConfig = optim_config_cls(
        lr=1e-3,
        betas=(0.9, 0.95),
    )
    parametrization_optimizer_type = optim_config.parametrization_optimizer_type()
    assert parametrization_optimizer_type is not None, "Optimizer does not support parametrization"

    def generate_model(width_idx: int, seed_idx: int) -> Tuple[int, torch.nn.Module]:
        d_model = D_MODELS[width_idx]
        hidden_size = HIDDEN_SIZES[width_idx]

        parametrization = ParametrizationConfig(
            name=ParametrizationType.mup,
            optimizer=parametrization_optimizer_type,
            scaling_strategy=parametrization_scaling_strategy,
        )

        ffn = FeedForwardConfig(
            hidden_size,
            parametrization=parametrization,
            bias=False,
        ).build(d_model)

        if base_d_model is not None and base_hidden_size is not None:
            base_ffn = FeedForwardConfig(
                base_hidden_size,
                parametrization=parametrization,
                bias=False,
            ).build(base_d_model)
            ParametrizationBase.set_base_model_dims(ffn, base_ffn)

        InitMethod.normal.init_feed_forward(
            ffn,
            d_model=d_model,
            block_idx=0,
            num_blocks=1,
            generator=torch.Generator().manual_seed(seed_idx),
        )

        return d_model, ffn

    seeded_init_coord_magnitudes, _ = train_and_collect_coordinate_data(
        generate_model,
        optim_config,
        len(D_MODELS),
        steps=STEPS,
        seeds=SEEDS,
        has_embedding_dim=True,
    )

    validate_parametrization_data(
        D_MODELS,
        seeded_init_coord_magnitudes,
        seeded_coord_change_stds=None,
        min_normalized_slope=MIN_NORMALIZED_SLOPE,
        max_normalized_slope=MAX_NORMALIZED_SLOPE,
    )


@pytest.mark.parametrize(
    "optim_config_cls, parametrization_scaling_strategy",
    [
        (optim_config_cls, scaling_strategy)
        for optim_config_cls in [AdamConfig, AdamWConfig, SkipStepAdamWConfig]
        for scaling_strategy in MupScalingStrategy
    ],
)
@pytest.mark.parametrize("base_d_model, base_hidden_size", [(64, 32), (None, None)])
def test_ffn_coord_check_after_training_with_scaling_input_output(
    optim_config_cls, parametrization_scaling_strategy, base_d_model, base_hidden_size
):
    MIN_NORMALIZED_SLOPE = -0.05
    MAX_NORMALIZED_SLOPE = 0.05
    D_MODELS = [16, 32, 64, 128, 256]
    HIDDEN_SIZES = [10, 20, 30, 40, 50]
    SEEDS = 10
    STEPS = 10

    optim_config: OptimConfig = optim_config_cls(
        lr=1e-3,
        betas=(0.9, 0.95),
    )
    parametrization_optimizer_type = optim_config.parametrization_optimizer_type()
    assert parametrization_optimizer_type is not None, "Optimizer does not support parametrization"

    def generate_model(width_idx: int, seed_idx: int) -> Tuple[int, torch.nn.Module]:
        d_model = D_MODELS[width_idx]
        hidden_size = HIDDEN_SIZES[width_idx]

        parametrization = ParametrizationConfig(
            name=ParametrizationType.mup,
            optimizer=parametrization_optimizer_type,
            scaling_strategy=parametrization_scaling_strategy,
        )

        ffn = FeedForwardConfig(
            hidden_size,
            parametrization=parametrization,
            bias=False,
        ).build(d_model)

        if base_d_model is not None and base_hidden_size is not None:
            base_ffn = FeedForwardConfig(
                base_hidden_size,
                parametrization=parametrization,
                bias=False,
            ).build(base_d_model)
            ParametrizationBase.set_base_model_dims(ffn, base_ffn)

        InitMethod.normal.init_feed_forward(
            ffn,
            d_model=d_model,
            block_idx=0,
            num_blocks=1,
            generator=torch.Generator().manual_seed(seed_idx),
        )

        return d_model, ffn

    seeded_init_coord_magnitudes, seeded_coord_change_stds = train_and_collect_coordinate_data(
        generate_model,
        optim_config,
        len(D_MODELS),
        steps=STEPS,
        seeds=SEEDS,
        has_embedding_dim=True,
    )

    validate_parametrization_data(
        D_MODELS,
        seeded_init_coord_magnitudes,
        seeded_coord_change_stds,
        min_normalized_slope=MIN_NORMALIZED_SLOPE,
        max_normalized_slope=MAX_NORMALIZED_SLOPE,
    )


@pytest.mark.parametrize(
    "optim_config_cls, parametrization_scaling_strategy",
    [
        (optim_config_cls, scaling_strategy)
        for optim_config_cls in [AdamConfig, AdamWConfig, SkipStepAdamWConfig]
        for scaling_strategy in MupScalingStrategy
    ],
)
@pytest.mark.parametrize("base_d_model, base_hidden_size", [(32, 64), (None, None)])
def test_ffn_coord_check_at_init_with_scaling_hidden_layers(
    optim_config_cls, parametrization_scaling_strategy, base_d_model, base_hidden_size
):
    MIN_NORMALIZED_SLOPE = -0.05
    MAX_NORMALIZED_SLOPE = 0.05
    D_MODELS = [32, 32, 32, 32, 32]
    HIDDEN_SIZES = [16, 32, 64, 128, 256]
    SEEDS = 10
    STEPS = 1

    optim_config: OptimConfig = optim_config_cls(
        lr=1e-3,
        betas=(0.9, 0.95),
    )
    parametrization_optimizer_type = optim_config.parametrization_optimizer_type()
    assert parametrization_optimizer_type is not None, "Optimizer does not support parametrization"

    def generate_model(width_idx: int, seed_idx: int) -> Tuple[int, torch.nn.Module]:
        d_model = D_MODELS[width_idx]
        hidden_size = HIDDEN_SIZES[width_idx]

        parametrization = ParametrizationConfig(
            name=ParametrizationType.mup,
            optimizer=parametrization_optimizer_type,
            scaling_strategy=parametrization_scaling_strategy,
        )

        ffn = FeedForwardConfig(
            hidden_size,
            parametrization=parametrization,
            bias=False,
        ).build(d_model)

        if base_d_model is not None and base_hidden_size is not None:
            base_ffn = FeedForwardConfig(
                base_hidden_size,
                parametrization=parametrization,
                bias=False,
            ).build(base_d_model)
            ParametrizationBase.set_base_model_dims(ffn, base_ffn)

        InitMethod.normal.init_feed_forward(
            ffn,
            d_model=d_model,
            block_idx=0,
            num_blocks=1,
            generator=torch.Generator().manual_seed(seed_idx),
        )

        return d_model, ffn

    seeded_init_coord_magnitudes, _ = train_and_collect_coordinate_data(
        generate_model,
        optim_config,
        len(HIDDEN_SIZES),
        steps=STEPS,
        seeds=SEEDS,
        has_embedding_dim=True,
    )

    validate_parametrization_data(
        HIDDEN_SIZES,
        seeded_init_coord_magnitudes,
        seeded_coord_change_stds=None,
        min_normalized_slope=MIN_NORMALIZED_SLOPE,
        max_normalized_slope=MAX_NORMALIZED_SLOPE,
    )


@pytest.mark.parametrize(
    "optim_config_cls, parametrization_scaling_strategy",
    [
        (optim_config_cls, scaling_strategy)
        for optim_config_cls in [AdamConfig, AdamWConfig, SkipStepAdamWConfig]
        for scaling_strategy in MupScalingStrategy
    ],
)
@pytest.mark.parametrize("base_d_model, base_hidden_size", [(32, 64), (None, None)])
def test_ffn_coord_check_after_training_with_scaling_hidden_layers(
    optim_config_cls, parametrization_scaling_strategy, base_d_model, base_hidden_size
):
    MIN_NORMALIZED_SLOPE = -0.05
    MAX_NORMALIZED_SLOPE = 0.05
    D_MODELS = [32, 32, 32, 32, 32]
    HIDDEN_SIZES = [16, 32, 64, 128, 256]
    SEEDS = 10
    STEPS = 10

    optim_config: OptimConfig = optim_config_cls(
        lr=1e-3,
        betas=(0.9, 0.95),
    )
    parametrization_optimizer_type = optim_config.parametrization_optimizer_type()
    assert parametrization_optimizer_type is not None, "Optimizer does not support parametrization"

    def generate_model(width_idx: int, seed_idx: int) -> Tuple[int, torch.nn.Module]:
        d_model = D_MODELS[width_idx]
        hidden_size = HIDDEN_SIZES[width_idx]

        parametrization = ParametrizationConfig(
            name=ParametrizationType.mup,
            optimizer=parametrization_optimizer_type,
            scaling_strategy=parametrization_scaling_strategy,
        )

        ffn = FeedForwardConfig(
            hidden_size,
            parametrization=parametrization,
            bias=False,
        ).build(d_model)

        if base_d_model is not None and base_hidden_size is not None:
            base_ffn = FeedForwardConfig(
                base_hidden_size,
                parametrization=parametrization,
                bias=False,
            ).build(base_d_model)
            ParametrizationBase.set_base_model_dims(ffn, base_ffn)

        InitMethod.normal.init_feed_forward(
            ffn,
            d_model=d_model,
            block_idx=0,
            num_blocks=1,
            generator=torch.Generator().manual_seed(seed_idx),
        )

        return d_model, ffn

    seeded_init_coord_magnitudes, seeded_coord_change_stds = train_and_collect_coordinate_data(
        generate_model,
        optim_config,
        len(HIDDEN_SIZES),
        steps=STEPS,
        seeds=SEEDS,
        has_embedding_dim=True,
    )

    validate_parametrization_data(
        HIDDEN_SIZES,
        seeded_init_coord_magnitudes,
        seeded_coord_change_stds,
        min_normalized_slope=MIN_NORMALIZED_SLOPE,
        max_normalized_slope=MAX_NORMALIZED_SLOPE,
    )


@pytest.mark.parametrize(
    "optim_config_cls, parametrization_scaling_strategy",
    [
        (optim_config_cls, scaling_strategy)
        for optim_config_cls in [AdamConfig, AdamWConfig, SkipStepAdamWConfig]
        for scaling_strategy in MupScalingStrategy
    ],
)
@pytest.mark.parametrize("base_d_model, base_n_heads", [(64, 8), (None, None)])
def test_attention_coord_check_at_init_with_scaling_input_output(
    optim_config_cls, parametrization_scaling_strategy, base_d_model, base_n_heads
):
    MIN_NORMALIZED_SLOPE = -0.05
    MAX_NORMALIZED_SLOPE = 0.05
    D_MODELS = [16, 32, 64, 128, 256]
    N_HEADS = [2, 4, 8, 16, 32]
    SEEDS = 10
    STEPS = 1

    optim_config: OptimConfig = optim_config_cls(
        lr=1e-3,
        betas=(0.9, 0.95),
    )
    parametrization_optimizer_type = optim_config.parametrization_optimizer_type()
    assert parametrization_optimizer_type is not None, "Optimizer does not support parametrization"

    def generate_model(width_idx: int, seed_idx: int) -> Tuple[int, torch.nn.Module]:
        d_model = D_MODELS[width_idx]
        n_heads = N_HEADS[width_idx]

        parametrization = ParametrizationConfig(
            name=ParametrizationType.mup,
            optimizer=parametrization_optimizer_type,
            scaling_strategy=parametrization_scaling_strategy,
        )

        attention = AttentionConfig(
            AttentionType.default,
            n_heads,
            bias=False,
            parametrization=parametrization,
        ).build(d_model, layer_idx=0, n_layers=1)

        if base_d_model is not None and base_n_heads is not None:
            base_attention = AttentionConfig(
                AttentionType.default,
                base_n_heads,
                bias=False,
                parametrization=parametrization,
            ).build(base_d_model, layer_idx=0, n_layers=1)
            ParametrizationBase.set_base_model_dims(attention, base_attention)

        InitMethod.normal.init_attention(
            attention,
            d_model=d_model,
            block_idx=0,
            num_blocks=1,
            generator=torch.Generator().manual_seed(seed_idx),
        )

        return d_model, attention

    seeded_init_coord_magnitudes, _ = train_and_collect_coordinate_data(
        generate_model,
        optim_config,
        len(N_HEADS),
        steps=STEPS,
        seeds=SEEDS,
        has_embedding_dim=True,
    )

    validate_parametrization_data(
        N_HEADS,
        seeded_init_coord_magnitudes,
        seeded_coord_change_stds=None,
        min_normalized_slope=MIN_NORMALIZED_SLOPE,
        max_normalized_slope=MAX_NORMALIZED_SLOPE,
    )


@pytest.mark.parametrize(
    "optim_config_cls, parametrization_scaling_strategy",
    [
        (optim_config_cls, scaling_strategy)
        for optim_config_cls in [AdamConfig, AdamWConfig, SkipStepAdamWConfig]
        for scaling_strategy in MupScalingStrategy
    ],
)
@pytest.mark.parametrize("base_d_model, base_n_heads", [(64, 8), (None, None)])
def test_attention_coord_check_after_training_with_scaling_hidden_layers(
    optim_config_cls, parametrization_scaling_strategy, base_d_model, base_n_heads
):
    MIN_NORMALIZED_SLOPE = -0.05
    MAX_NORMALIZED_SLOPE = 0.05
    D_MODELS = [32, 64, 128, 256, 512]
    N_HEADS = [4, 8, 16, 32, 64]
    SEEDS = 10
    STEPS = 10

    optim_config: OptimConfig = optim_config_cls(
        lr=1e-3,
        betas=(0.9, 0.95),
    )
    parametrization_optimizer_type = optim_config.parametrization_optimizer_type()
    assert parametrization_optimizer_type is not None, "Optimizer does not support parametrization"

    def generate_model(width_idx: int, seed_idx: int) -> Tuple[int, torch.nn.Module]:
        d_model = D_MODELS[width_idx]
        n_heads = N_HEADS[width_idx]

        parametrization = ParametrizationConfig(
            name=ParametrizationType.mup,
            optimizer=parametrization_optimizer_type,
            scaling_strategy=parametrization_scaling_strategy,
        )

        attention = AttentionConfig(
            AttentionType.default,
            n_heads,
            bias=False,
            parametrization=parametrization,
        ).build(d_model, layer_idx=0, n_layers=1)

        if base_d_model is not None and base_n_heads is not None:
            base_attention = AttentionConfig(
                AttentionType.default,
                base_n_heads,
                bias=False,
                parametrization=parametrization,
            ).build(base_d_model, layer_idx=0, n_layers=1)
            ParametrizationBase.set_base_model_dims(attention, base_attention)

        InitMethod.normal.init_attention(
            attention,
            d_model=d_model,
            block_idx=0,
            num_blocks=1,
            generator=torch.Generator().manual_seed(seed_idx),
        )

        return d_model, attention

    seeded_init_coord_magnitudes, seeded_coord_change_stds = train_and_collect_coordinate_data(
        generate_model,
        optim_config,
        len(N_HEADS),
        steps=STEPS,
        seeds=SEEDS,
        has_embedding_dim=True,
    )

    validate_parametrization_data(
        N_HEADS,
        seeded_init_coord_magnitudes,
        seeded_coord_change_stds,
        min_normalized_slope=MIN_NORMALIZED_SLOPE,
        max_normalized_slope=MAX_NORMALIZED_SLOPE,
    )


@requires_gpu
@pytest.mark.parametrize(
    "optim_config_cls, parametrization_scaling_strategy",
    [
        (optim_config_cls, scaling_strategy)
        for optim_config_cls in [AdamConfig, AdamWConfig, SkipStepAdamWConfig]
        for scaling_strategy in MupScalingStrategy
    ],
)
def test_moe_mup_const_coord_norm_at_init_scaling_input_output(
    optim_config_cls, parametrization_scaling_strategy
):
    MIN_NORMALIZED_SLOPE = -0.05
    MAX_NORMALIZED_SLOPE = 0.05
    D_MODELS = [128, 256, 512, 1024, 2048]
    HIDDEN_SIZES = [100, 200, 300, 400, 500]
    SHARED_EXPERT_HIDDEN_SIZES = [16, 32, 48, 64, 72]
    TOP_K = [2, 2, 2, 2, 2]
    NUM_EXPERTS = [2, 4, 8, 16, 32]
    SEEDS = 5
    STEPS = 3

    torch.cuda.init()

    optim_config: OptimConfig = optim_config_cls(
        lr=1e-3,
        betas=(0.9, 0.95),
    )
    parametrization_optimizer_type = optim_config.parametrization_optimizer_type()
    assert parametrization_optimizer_type is not None, "Optimizer does not support parametrization"

    def generate_model(width_idx: int, seed_idx: int) -> Tuple[int, torch.nn.Module]:
        d_model = D_MODELS[width_idx]
        hidden_size = HIDDEN_SIZES[width_idx]
        num_experts = NUM_EXPERTS[width_idx]
        top_k = TOP_K[width_idx]
        shared_expert_hidden_size = SHARED_EXPERT_HIDDEN_SIZES[width_idx]

        parametrization = ParametrizationConfig(
            name=ParametrizationType.mup,
            optimizer=parametrization_optimizer_type,
            scaling_strategy=parametrization_scaling_strategy,
        )

        router = MoERouterConfig(
            top_k=top_k,
            normalize_expert_weights=2.0,
            parametrization=parametrization,
        )

        moe = MoEConfig(
            num_experts=num_experts,
            hidden_size=hidden_size,
            router=router,
            parametrization=parametrization,
            shared_mlp=FeedForwardConfig(
                shared_expert_hidden_size,
                parametrization=parametrization,
                bias=False,
            ),
        ).build(d_model, init_device="cuda")

        InitMethod.normal.init_feed_forward_moe(
            moe,
            d_model=d_model,
            block_idx=0,
            num_blocks=1,
            generator=torch.Generator(device="cuda").manual_seed(seed_idx),
        )

        return d_model, moe

    seeded_init_coord_magnitudes, seeded_coord_change_stds = train_and_collect_coordinate_data(
        generate_model,
        optim_config,
        len(D_MODELS),
        steps=STEPS,
        seeds=SEEDS,
        has_embedding_dim=True,
        device=torch.device("cuda"),
    )

    validate_parametrization_data(
        D_MODELS,
        seeded_init_coord_magnitudes,
        seeded_coord_change_stds,
        min_normalized_slope=MIN_NORMALIZED_SLOPE,
        max_normalized_slope=MAX_NORMALIZED_SLOPE,
    )


@pytest.mark.parametrize(
    "optim_config_cls, parametrization_scaling_strategy",
    [
        (optim_config_cls, scaling_strategy)
        for optim_config_cls in [AdamConfig, AdamWConfig, SkipStepAdamWConfig]
        for scaling_strategy in MupScalingStrategy
    ],
)
@pytest.mark.parametrize("base_d_model", [128, None])
def test_mup_non_growing_coordinates(
    optim_config_cls, parametrization_scaling_strategy, base_d_model
):
    MIN_NORMALIZED_SLOPE = -0.2
    MAX_NORMALIZED_SLOPE = 0.1
    SEQ_LEN = 32
    VOCAB_SIZE = 100
    SEEDS = 10
    STEPS = 10
    D_MODELS = [32, 64, 128, 256, 512]

    optim_config: OptimConfig = optim_config_cls(
        lr=5e-3,
        betas=(0.9, 0.95),
        group_overrides=[
            OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0))
        ],
    )
    parametrization_optimizer_type = optim_config.parametrization_optimizer_type()
    assert parametrization_optimizer_type is not None, "Optimizer does not support parametrization"

    def generate_model(width_idx: int, seed_idx: int) -> Tuple[int, torch.nn.Module]:
        d_model = D_MODELS[width_idx]

        parametrization_config = ParametrizationConfig(
            name=ParametrizationType.mup,
            optimizer=parametrization_optimizer_type,
            scaling_strategy=parametrization_scaling_strategy,
        )

        model_config = get_transformer_config(
            parametrization_config,
            vocab_size=VOCAB_SIZE,
            d_model=d_model,
            init_seed=seed_idx,
        )

        model = model_config.build(init_device="meta")

        if base_d_model is not None:
            base_model_config = get_transformer_config(
                parametrization_config,
                vocab_size=VOCAB_SIZE,
                d_model=base_d_model,
                init_seed=seed_idx,
            )
            base_model = base_model_config.build(init_device="meta")

            ParametrizationBase.set_base_model_dims(model, base_model)

        model.init_weights(max_seq_len=SEQ_LEN, device=torch.device("cpu"))

        return SEQ_LEN, model

    seeded_init_coord_magnitudes, seeded_coord_change_stds = train_and_collect_coordinate_data(
        generate_model,
        optim_config,
        len(D_MODELS),
        vocab_size=VOCAB_SIZE,
        steps=STEPS,
        seeds=SEEDS,
    )

    # MuP allows for the final layer to decay at init
    init_decay_allowlist = ["lm_head.w_out", "lm_head", ""]
    validate_parametrization_data(
        D_MODELS,
        seeded_init_coord_magnitudes,
        seeded_coord_change_stds,
        min_normalized_slope=MIN_NORMALIZED_SLOPE,
        max_normalized_slope=MAX_NORMALIZED_SLOPE,
        init_decay_allowlist=init_decay_allowlist,
    )
