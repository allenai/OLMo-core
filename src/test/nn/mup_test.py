import logging
from collections import defaultdict
from contextlib import contextmanager
from functools import partial
from typing import Callable, Dict, List, Optional, Tuple, Type

import numpy as np
import pytest
import torch

from olmo_core.nn.cross_entropy_loss import CrossEntropyLoss
from olmo_core.nn.feed_forward import FeedForwardConfig
from olmo_core.nn.moe.moe import MoEConfig
from olmo_core.nn.moe.router import MoERouterConfig
from olmo_core.nn.mup import (
    MuPConfig,
    MuPHyperParam,
    MuPOptimizerType,
    MuPScalingStrategy,
)
from olmo_core.nn.transformer.config import TransformerBlockType, TransformerConfig
from olmo_core.nn.transformer.init import InitMethod
from olmo_core.optim.adam import AdamConfig
from olmo_core.optim.adamw import AdamWConfig, SkipStepAdamWConfig
from olmo_core.optim.config import OptimConfig, OptimGroupOverride
from olmo_core.optim.scheduler import LinearWithWarmup
from olmo_core.testing.utils import requires_gpu

log = logging.getLogger(__name__)


def get_transformer_config(
    mup_config: Optional[MuPConfig] = None,
    vocab_size: int = 100,
    init_seed: int = 0,
    d_model_multiplier: int = 1,
    n_heads_multiplier: int = 1,
) -> TransformerConfig:
    return TransformerConfig.llama_like(
        d_model=8 * d_model_multiplier,
        vocab_size=vocab_size,
        n_layers=2,
        n_heads=2 * n_heads_multiplier,
        hidden_size_multiplier=1,
        hidden_size_multiple_of=16,
        block_name=TransformerBlockType.reordered_norm,
        qk_norm=True,
        rope_theta=500_000,
        layer_norm_eps=1e-6,
        fused_ops=False,
        use_flash=False,
        mup=mup_config,
        init_seed=init_seed,
    )


def get_transformer_inputs(vocab_size: int = 100) -> torch.Tensor:
    return torch.arange(0, vocab_size).unsqueeze(0)


@pytest.mark.parametrize(
    "mup_optimizer_type, mup_scaling_strategy, expected_multiplier",
    [
        pytest.param(MuPOptimizerType.adam, MuPScalingStrategy.constant_inputs, None),
        pytest.param(MuPOptimizerType.adam, MuPScalingStrategy.constant_lr, 1 / (2**3 * 0.7)),
        pytest.param(
            MuPOptimizerType.adam, MuPScalingStrategy.constant_init_std, 1 / (2**3 * 0.7)
        ),
    ],
)
def test_mup_input_multiplier_scaling_input(
    mup_optimizer_type, mup_scaling_strategy, expected_multiplier
):
    mup_config = MuPConfig(
        mup_optimizer_type,
        scaling_strategy=mup_scaling_strategy,
        width_scalings={
            MuPHyperParam.d_model: 2,
            MuPHyperParam.hidden_size: 0.7,
            MuPHyperParam.head_dim: 2,
        },
    )
    mup = mup_config.build({MuPHyperParam.d_model: 3, MuPHyperParam.hidden_size: 1}, {})

    torch.testing.assert_close(mup.input_multiplier, expected_multiplier)


@pytest.mark.parametrize(
    "mup_optimizer_type, mup_scaling_strategy, expected_multiplier",
    [
        pytest.param(MuPOptimizerType.adam, MuPScalingStrategy.constant_inputs, None),
        pytest.param(MuPOptimizerType.adam, MuPScalingStrategy.constant_lr, 1 / (2**3)),
        pytest.param(
            MuPOptimizerType.adam, MuPScalingStrategy.constant_init_std, 1 / (2 ** (3 / 2))
        ),
    ],
)
def test_mup_input_multiplier_scaling_input_and_output(
    mup_optimizer_type, mup_scaling_strategy, expected_multiplier
):
    mup_config = MuPConfig(
        mup_optimizer_type,
        scaling_strategy=mup_scaling_strategy,
        width_scalings={
            MuPHyperParam.d_model: 2,
            MuPHyperParam.hidden_size: 0.7,
            MuPHyperParam.head_dim: 2,
        },
    )
    mup = mup_config.build({MuPHyperParam.d_model: 3}, {MuPHyperParam.hidden_size: 1})

    torch.testing.assert_close(mup.input_multiplier, expected_multiplier)


@pytest.mark.parametrize(
    "mup_optimizer_type, mup_scaling_strategy, expected_multiplier",
    [
        pytest.param(MuPOptimizerType.adam, MuPScalingStrategy.constant_inputs, 1 / (2**3 * 0.7)),
        pytest.param(MuPOptimizerType.adam, MuPScalingStrategy.constant_lr, None),
        pytest.param(MuPOptimizerType.adam, MuPScalingStrategy.constant_init_std, None),
    ],
)
def test_mup_init_std_multiplier_scaling_input(
    mup_optimizer_type, mup_scaling_strategy, expected_multiplier
):
    mup_config = MuPConfig(
        mup_optimizer_type,
        scaling_strategy=mup_scaling_strategy,
        width_scalings={
            MuPHyperParam.d_model: 2,
            MuPHyperParam.hidden_size: 0.7,
            MuPHyperParam.head_dim: 2,
        },
    )
    mup = mup_config.build({MuPHyperParam.d_model: 3, MuPHyperParam.hidden_size: 1}, {})

    torch.testing.assert_close(mup.init_std_multiplier, expected_multiplier)


@pytest.mark.parametrize(
    "mup_optimizer_type, mup_scaling_strategy, expected_multiplier",
    [
        pytest.param(MuPOptimizerType.adam, MuPScalingStrategy.constant_inputs, 1 / (2 ** (3 / 2))),
        pytest.param(MuPOptimizerType.adam, MuPScalingStrategy.constant_lr, 2 ** (3 / 2)),
        pytest.param(MuPOptimizerType.adam, MuPScalingStrategy.constant_init_std, None),
    ],
)
def test_mup_init_std_multiplier_scaling_input_and_output(
    mup_optimizer_type, mup_scaling_strategy, expected_multiplier
):
    mup_config = MuPConfig(
        mup_optimizer_type,
        scaling_strategy=mup_scaling_strategy,
        width_scalings={
            MuPHyperParam.d_model: 2,
            MuPHyperParam.hidden_size: 0.7,
            MuPHyperParam.head_dim: 2,
        },
    )
    mup = mup_config.build({MuPHyperParam.d_model: 3}, {MuPHyperParam.hidden_size: 1})

    torch.testing.assert_close(mup.init_std_multiplier, expected_multiplier)


@pytest.mark.parametrize(
    "mup_optimizer_type, mup_scaling_strategy",
    [
        pytest.param(mup_optimizer_type, scaling_strategy)
        for mup_optimizer_type in MuPOptimizerType
        for scaling_strategy in MuPScalingStrategy
    ],
)
def test_mup_no_width_scaling_same_output(mup_optimizer_type, mup_scaling_strategy):
    model_config = get_transformer_config()

    mup_config = MuPConfig(mup_optimizer_type, scaling_strategy=mup_scaling_strategy)
    mup_model_config = get_transformer_config(mup_config)

    model = model_config.build()
    mup_model = mup_model_config.build()

    model.init_weights()
    mup_model.to_empty(device=model.device)
    mup_model.load_state_dict(model.state_dict())

    input_ids = get_transformer_inputs()
    logits = model(input_ids=input_ids)
    mup_logits = mup_model(input_ids=input_ids)

    torch.testing.assert_close(logits, mup_logits)


@pytest.mark.parametrize(
    "optim_config_cls, mup_scaling_strategy",
    [
        pytest.param(optim_config_cls, scaling_strategy)
        for optim_config_cls in [AdamConfig, AdamWConfig, SkipStepAdamWConfig]
        for scaling_strategy in MuPScalingStrategy
    ],
)
def test_mup_no_width_scaling_same_optim_groups(
    optim_config_cls: Type[OptimConfig], mup_scaling_strategy
):
    model_config = get_transformer_config()

    mup_optimizer_type = optim_config_cls.mup_optimizer_type()
    assert mup_optimizer_type is not None, "Optimizer does not support muP"

    mup_config = MuPConfig(mup_optimizer_type, scaling_strategy=mup_scaling_strategy)
    mup_model_config = get_transformer_config(mup_config)

    model = model_config.build()
    mup_model = mup_model_config.build()

    optim_config = optim_config_cls(
        lr=1e-2,
        betas=(0.9, 0.95),
        group_overrides=[
            OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0))
        ],
    )

    model.init_weights()
    mup_model.to_empty(device=model.device)
    mup_model.load_state_dict(model.state_dict())

    optim: torch.optim.Optimizer = optim_config.build(model)
    mup_optim: torch.optim.Optimizer = optim_config.build(mup_model)

    torch.testing.assert_close(optim.param_groups, mup_optim.param_groups)


@pytest.mark.parametrize(
    "mup_optimizer_type",
    [pytest.param(mup_optimizer_type) for mup_optimizer_type in MuPOptimizerType],
)
def test_mup_no_input_scaling_same_output(mup_optimizer_type):
    model_config = get_transformer_config()

    mup_config = MuPConfig(
        mup_optimizer_type,
        scaling_strategy=MuPScalingStrategy.constant_inputs,
        width_scalings={MuPHyperParam.hidden_size: 2.0},
    )
    mup_model_config = get_transformer_config(mup_config)

    model = model_config.build()
    mup_model = mup_model_config.build()

    model.init_weights()
    mup_model.to_empty(device=model.device)
    mup_model.load_state_dict(model.state_dict())

    input_ids = get_transformer_inputs()
    logits = model(input_ids=input_ids)
    mup_logits = mup_model(input_ids=input_ids)

    torch.testing.assert_close(logits, mup_logits)


@pytest.mark.parametrize(
    "optim_config_cls",
    [
        pytest.param(AdamConfig),
        pytest.param(AdamWConfig),
        pytest.param(SkipStepAdamWConfig),
    ],
)
def test_mup_no_lr_scaling_same_optim_groups(optim_config_cls):
    model_config = get_transformer_config()

    mup_optimizer_type = optim_config_cls.mup_optimizer_type()
    assert mup_optimizer_type is not None, "Optimizer does not support muP"

    mup_config = MuPConfig(
        mup_optimizer_type,
        scaling_strategy=MuPScalingStrategy.constant_lr,
        width_scalings={
            MuPHyperParam.d_model: 2.0,
            MuPHyperParam.head_dim: 2.0,
            MuPHyperParam.hidden_size: 2.0,
        },
    )
    mup_model_config = get_transformer_config(mup_config)

    model = model_config.build()
    mup_model = mup_model_config.build()

    optim_config = optim_config_cls(
        lr=1e-2,
        betas=(0.9, 0.95),
        group_overrides=[
            OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0))
        ],
    )

    model.init_weights()
    mup_model.to_empty(device=model.device)
    mup_model.load_state_dict(model.state_dict())

    optim: torch.optim.Optimizer = optim_config.build(model)
    mup_optim: torch.optim.Optimizer = optim_config.build(mup_model)

    torch.testing.assert_close(optim.param_groups, mup_optim.param_groups)


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


def train_and_collect_mup_data(
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
):
    device = device or torch.device("cpu")

    def _get_input(*, is_training: bool = True, input_dim: Optional[int] = None) -> torch.Tensor:
        batch_size = train_batch_size if is_training else eval_batch_size
        if has_embedding_dim:
            assert input_dim is not None
            return torch.randn(batch_size, sequence_length, input_dim)

        return torch.randint(0, vocab_size, (batch_size, sequence_length))

    seeded_initial_submodule_outputs: List[Dict[str, torch.Tensor]] = []
    seeded_final_submodule_outputs: List[Dict[str, torch.Tensor]] = []

    # Dict[activation_name, [<field>_for_different_multipliers_and_seeds]]

    for width_idx in range(num_widths):
        for seed_idx in range(seeds):
            input_dim, model = model_generator(width_idx, seed_idx)

            optim = optim_config.build(model)
            scheduler = LinearWithWarmup(warmup=0)

            with _submodule_output_collection(model) as submodule_outputs:
                _ = model(_get_input(is_training=False, input_dim=input_dim).to(device=device))
                seeded_initial_submodule_outputs.append(submodule_outputs)

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
                seeded_final_submodule_outputs.append(submodule_outputs)

    # First piece of muP data is average abs value of elements at initialization
    # (a.k.a. "coordinate magnitudes" at init).
    seeded_init_coord_magnitudes: defaultdict[str, List[float]] = defaultdict(list)
    for initial_submodule_outputs in seeded_initial_submodule_outputs:
        for state_key, output in initial_submodule_outputs.items():
            seeded_init_coord_magnitudes[state_key].append(
                output.float().norm(p=1.0).item() / output.numel()
            )

    # Second piece of muP data is std deviation of the change in "coordinates"
    # from initialization to end of training.
    seeded_coord_change_stds: defaultdict[str, List[float]] = defaultdict(list)
    for initial_submodule_outputs, final_submodule_outputs in zip(
        seeded_initial_submodule_outputs, seeded_final_submodule_outputs
    ):
        for state_key in initial_submodule_outputs.keys():
            initial_output = initial_submodule_outputs[state_key]
            final_output = final_submodule_outputs[state_key]
            coord_change_std = (
                final_output - initial_output
            ).float().std() / initial_output.numel()
            seeded_coord_change_stds[state_key].append(coord_change_std.item())

    return seeded_init_coord_magnitudes, seeded_coord_change_stds


@pytest.mark.parametrize(
    "optim_config_cls, mup_scaling_strategy",
    [
        pytest.param(optim_config_cls, scaling_strategy)
        for optim_config_cls in [AdamConfig, AdamWConfig, SkipStepAdamWConfig]
        for scaling_strategy in MuPScalingStrategy
    ],
)
def test_ffn_mup_const_coord_norm_at_init_scaling_input_output(
    optim_config_cls, mup_scaling_strategy
):
    """
    This is the most important (and slowest) test of muP validity. It runs a base model and its muP
    up-/down-scalings, and checks that the magnitude of the coordinates (i.e. entries) of the
    intermediate activations are not growing/decaying.

    Growing coordinates are never expected in muP beyond noise, so the acceptable threshold of growth
    is set closer to 1. Decaying is expected in attention logits and layers with scaling inputs at initialization,
    and in general it takes a while for muP to get rid of decaying. Thus decay threshold is set further from 1.
    These thresholds are broken very easily by the standard parametrization but we try to keep them tight
    because muP bugs might not break them as easily.
    """
    MIN_NORMALIZED_SLOPE = -0.05
    MAX_NORMALIZED_SLOPE = 0.05
    D_MODELS = [128, 256, 512, 1024, 2048]
    HIDDEN_SIZES = [100, 200, 300, 400, 500]
    BASE_MULTIPLIER_IDX = 2
    BASE_D_MODEL = D_MODELS[BASE_MULTIPLIER_IDX]
    BASE_HIDDEN_SIZE = HIDDEN_SIZES[BASE_MULTIPLIER_IDX]
    SEEDS = 10
    STEPS = 1

    optim_config: OptimConfig = optim_config_cls(
        lr=1e-3,
        betas=(0.9, 0.95),
    )
    mup_optimizer_type = optim_config.mup_optimizer_type()
    assert mup_optimizer_type is not None, "Optimizer does not support muP"

    def generate_model(width_idx: int, seed_idx: int) -> Tuple[int, torch.nn.Module]:
        d_model = D_MODELS[width_idx]
        hidden_size = HIDDEN_SIZES[width_idx]

        mup = MuPConfig(
            mup_optimizer_type,
            scaling_strategy=mup_scaling_strategy,
            width_scalings={
                MuPHyperParam.d_model: d_model / BASE_D_MODEL,
                MuPHyperParam.hidden_size: hidden_size / BASE_HIDDEN_SIZE,
                MuPHyperParam.head_dim: d_model / BASE_D_MODEL,
            },
        )

        ffn = FeedForwardConfig(
            hidden_size,
            mup=mup,
            bias=False,
        ).build(d_model)

        InitMethod.normal.init_feed_forward(
            ffn,
            d_model=d_model,
            block_idx=0,
            num_blocks=1,
            generator=torch.Generator().manual_seed(seed_idx),
        )

        return d_model, ffn

    coord_magnitudes, activation_shapes = train_and_collect_mup_data(
        generate_model,
        optim_config,
        len(D_MODELS),
        steps=STEPS,
        seeds=SEEDS,
        has_embedding_dim=True,
    )

    for name, magnitudes in coord_magnitudes.items():
        magnitudes = np.array(
            [np.mean(magnitudes[i : i + SEEDS]) for i in range(0, len(magnitudes), SEEDS)]
        )
        shapes = [activation_shapes[name][SEEDS * i] for i in range(len(magnitudes))]

        log_d_models = np.log2(np.array(D_MODELS))
        slope = np.polynomial.polynomial.Polynomial.fit(
            log_d_models, magnitudes / magnitudes[0], 1
        ).coef[1]
        assert slope <= MAX_NORMALIZED_SLOPE, (
            f"Coordinate magnitudes for {name} grow too much with width. "
            f"slope {slope}, magnitudes {magnitudes}, normalized magnitudes {magnitudes / magnitudes[0]} shapes {shapes}"
        )
        assert slope >= MIN_NORMALIZED_SLOPE, (
            f"Coordinate magnitudes for {name} decay too much with width. "
            f"slope {slope}, magnitudes {magnitudes}, normalized magnitudes {magnitudes / magnitudes[0]} shapes {shapes}"
        )


@requires_gpu
@pytest.mark.parametrize(
    "optim_config_cls, mup_scaling_strategy",
    [
        pytest.param(optim_config_cls, scaling_strategy)
        for optim_config_cls in [AdamConfig, AdamWConfig, SkipStepAdamWConfig]
        for scaling_strategy in MuPScalingStrategy
    ],
)
def test_moe_mup_const_coord_norm_at_init_scaling_input_output(
    optim_config_cls, mup_scaling_strategy
):
    """
    This is the most important (and slowest) test of muP validity. It runs a base model and its muP
    up-/down-scalings, and checks that the magnitude of the coordinates (i.e. entries) of the
    intermediate activations are not growing/decaying.

    Growing coordinates are never expected in muP beyond noise, so the acceptable threshold of growth
    is set closer to 1. Decaying is expected in attention logits and layers with scaling inputs at initialization,
    and in general it takes a while for muP to get rid of decaying. Thus decay threshold is set further from 1.
    These thresholds are broken very easily by the standard parametrization but we try to keep them tight
    because muP bugs might not break them as easily.
    """
    MIN_NORMALIZED_SLOPE = -0.05
    MAX_NORMALIZED_SLOPE = 0.05
    D_MODELS = [128, 256, 512, 1024, 2048]
    HIDDEN_SIZES = [100, 200, 300, 400, 500]
    SHARED_EXPERT_HIDDEN_SIZES = [16, 32, 48, 64, 72]
    TOP_K = [2, 2, 2, 2, 2]
    NUM_EXPERTS = [2, 4, 8, 16, 32]
    BASE_MULTIPLIER_IDX = 2
    BASE_NUM_EXPERTS = NUM_EXPERTS[BASE_MULTIPLIER_IDX]
    BASE_D_MODEL = D_MODELS[BASE_MULTIPLIER_IDX]
    BASE_HIDDEN_SIZE = HIDDEN_SIZES[BASE_MULTIPLIER_IDX]
    BASE_SHARED_EXPERT_HIDDEN_SIZE = SHARED_EXPERT_HIDDEN_SIZES[BASE_MULTIPLIER_IDX]
    SEEDS = 5
    STEPS = 3

    torch.cuda.init()

    optim_config: OptimConfig = optim_config_cls(
        lr=1e-3,
        betas=(0.9, 0.95),
    )
    mup_optimizer_type = optim_config.mup_optimizer_type()
    assert mup_optimizer_type is not None, "Optimizer does not support muP"

    def generate_model(width_idx: int, seed_idx: int) -> Tuple[int, torch.nn.Module]:
        d_model = D_MODELS[width_idx]
        hidden_size = HIDDEN_SIZES[width_idx]
        num_experts = NUM_EXPERTS[width_idx]
        top_k = TOP_K[width_idx]
        shared_expert_hidden_size = SHARED_EXPERT_HIDDEN_SIZES[width_idx]

        mup = MuPConfig(
            mup_optimizer_type,
            scaling_strategy=mup_scaling_strategy,
            width_scalings={
                MuPHyperParam.d_model: d_model / BASE_D_MODEL,
                MuPHyperParam.hidden_size: hidden_size / BASE_HIDDEN_SIZE,
                MuPHyperParam.head_dim: d_model / BASE_D_MODEL,
                MuPHyperParam.num_experts: num_experts / BASE_NUM_EXPERTS,
                MuPHyperParam.shared_expert_hidden_size: shared_expert_hidden_size
                / BASE_SHARED_EXPERT_HIDDEN_SIZE,
            },
        )

        router = MoERouterConfig(
            top_k=top_k,
            normalize_expert_weights=2.0,
        )

        moe = MoEConfig(
            num_experts=num_experts,
            hidden_size=hidden_size,
            router=router,
            mup=mup,
            shared_mlp=FeedForwardConfig(
                shared_expert_hidden_size,
                mup=mup,
                bias=False,
                hidden_size_mup_hyper_param=MuPHyperParam.shared_expert_hidden_size,
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

    coord_magnitudes, activation_shapes = train_and_collect_mup_data(
        generate_model,
        optim_config,
        len(D_MODELS),
        steps=STEPS,
        seeds=SEEDS,
        has_embedding_dim=True,
        device=torch.device("cuda"),
    )

    growing_param_names = []
    decaying_param_names = []
    for name, magnitudes in coord_magnitudes.items():
        magnitudes = np.array(
            [np.mean(magnitudes[i : i + SEEDS]) for i in range(0, len(magnitudes), SEEDS)]
        )
        shapes = [activation_shapes[name][SEEDS * i] for i in range(len(magnitudes))]

        log_d_models = np.log2(np.array(D_MODELS))
        slope = np.polynomial.polynomial.Polynomial.fit(
            log_d_models, magnitudes / magnitudes[0], 1
        ).coef[1]
        if slope >= MAX_NORMALIZED_SLOPE:
            log.error(
                f"Coordinate magnitudes for {name} grow too much with width. "
                f"slope {slope}, magnitudes {magnitudes}, normalized magnitudes {magnitudes / magnitudes[0]} shapes {shapes}"
            )
            growing_param_names.append(name)
        if slope < MIN_NORMALIZED_SLOPE:
            log.error(
                f"Coordinate magnitudes for {name} decay too much with width. "
                f"slope {slope}, magnitudes {magnitudes}, normalized magnitudes {magnitudes / magnitudes[0]} shapes {shapes}"
            )
            decaying_param_names.append(name)

    assert (
        len(growing_param_names) == 0 and len(decaying_param_names) == 0
    ), f"Growing parameters: {growing_param_names}. Decaying parameters: {decaying_param_names}."


@pytest.mark.parametrize(
    "optim_config_cls, mup_scaling_strategy",
    [
        pytest.param(optim_config_cls, scaling_strategy)
        for optim_config_cls in [AdamConfig, AdamWConfig, SkipStepAdamWConfig]
        for scaling_strategy in MuPScalingStrategy
    ],
)
def test_ffn_mup_non_growing_coord_norm_scaling_input_output(
    optim_config_cls, mup_scaling_strategy
):
    """
    This is the most important (and slowest) test of muP validity. It runs a base model and its muP
    up-/down-scalings, and checks that the magnitude of the coordinates (i.e. entries) of the
    intermediate activations are not growing/decaying.

    Growing coordinates are never expected in muP beyond noise, so the acceptable threshold of growth
    is set closer to 1. Decaying is expected in attention logits and layers with scaling inputs at initialization,
    and in general it takes a while for muP to get rid of decaying. Thus decay threshold is set further from 1.
    These thresholds are broken very easily by the standard parametrization but we try to keep them tight
    because muP bugs might not break them as easily.
    """
    MIN_NORMALIZED_SLOPE = -0.5
    MAX_NORMALIZED_SLOPE = 0.05
    D_MODELS = [32, 64, 128, 256, 512]
    HIDDEN_SIZES = [24, 48, 96, 192, 384]
    BASE_MULTIPLIER_IDX = 2
    BASE_D_MODEL = D_MODELS[BASE_MULTIPLIER_IDX]
    BASE_HIDDEN_SIZE = HIDDEN_SIZES[BASE_MULTIPLIER_IDX]
    SEEDS = 10
    STEPS = 50

    optim_config: OptimConfig = optim_config_cls(
        lr=1e-3,
        betas=(0.9, 0.95),
    )
    mup_optimizer_type = optim_config.mup_optimizer_type()
    assert mup_optimizer_type is not None, "Optimizer does not support muP"

    def generate_model(width_idx: int, seed_idx: int) -> Tuple[int, torch.nn.Module]:
        d_model = D_MODELS[width_idx]
        hidden_size = HIDDEN_SIZES[width_idx]

        mup = MuPConfig(
            mup_optimizer_type,
            scaling_strategy=mup_scaling_strategy,
            width_scalings={
                MuPHyperParam.d_model: d_model / BASE_D_MODEL,
                MuPHyperParam.hidden_size: hidden_size / BASE_HIDDEN_SIZE,
                MuPHyperParam.head_dim: d_model / BASE_D_MODEL,
            },
        )

        ffn = FeedForwardConfig(
            hidden_size,
            mup=mup,
            bias=False,
        ).build(d_model)

        InitMethod.normal.init_feed_forward(
            ffn,
            d_model=d_model,
            block_idx=0,
            num_blocks=1,
            generator=torch.Generator().manual_seed(seed_idx),
        )

        return d_model, ffn

    coord_magnitudes, activation_shapes = train_and_collect_mup_data(
        generate_model,
        optim_config,
        len(D_MODELS),
        steps=STEPS,
        seeds=SEEDS,
        has_embedding_dim=True,
    )

    for name, magnitudes in coord_magnitudes.items():
        magnitudes = np.array(
            [np.mean(magnitudes[i : i + SEEDS]) for i in range(0, len(magnitudes), SEEDS)]
        )
        shapes = [activation_shapes[name][SEEDS * i] for i in range(len(magnitudes))]

        log_d_models = np.log2(np.array(D_MODELS))
        slope = np.polynomial.polynomial.Polynomial.fit(
            log_d_models, magnitudes / magnitudes[0], 1
        ).coef[1]
        assert slope <= MAX_NORMALIZED_SLOPE, (
            f"Coordinate magnitudes for {name} grow too much with width. "
            f"slope {slope}, magnitudes {magnitudes}, normalized magnitudes {magnitudes / magnitudes[0]} shapes {shapes}"
        )
        assert slope >= MIN_NORMALIZED_SLOPE, (
            f"Coordinate magnitudes for {name} decay too much with width. "
            f"slope {slope}, magnitudes {magnitudes}, normalized magnitudes {magnitudes / magnitudes[0]} shapes {shapes}"
        )


@pytest.mark.parametrize(
    "optim_config_cls, mup_scaling_strategy",
    [
        pytest.param(optim_config_cls, scaling_strategy)
        for optim_config_cls in [AdamConfig, AdamWConfig, SkipStepAdamWConfig]
        for scaling_strategy in MuPScalingStrategy
    ],
)
def test_ffn_mup_non_growing_coord_norm_scaling_d_model(optim_config_cls, mup_scaling_strategy):
    """
    This is the most important (and slowest) test of muP validity. It runs a base model and its muP
    up-/down-scalings, and checks that the magnitude of the coordinates (i.e. entries) of the
    intermediate activations are not growing/decaying.

    Growing coordinates are never expected in muP beyond noise, so the acceptable threshold of growth
    is set closer to 1. Decaying is expected in attention logits and layers with scaling inputs at initialization,
    and in general it takes a while for muP to get rid of decaying. Thus decay threshold is set further from 1.
    These thresholds are broken very easily by the standard parametrization but we try to keep them tight
    because muP bugs might not break them as easily.
    """
    MIN_NORMALIZED_SLOPE = -0.3
    MAX_NORMALIZED_SLOPE = 0.05
    D_MODELS = [32, 64, 128, 256, 512]
    HIDDEN_SIZES = [32] * 5
    BASE_MULTIPLIER_IDX = 2
    BASE_D_MODEL = D_MODELS[BASE_MULTIPLIER_IDX]
    BASE_HIDDEN_SIZE = HIDDEN_SIZES[BASE_MULTIPLIER_IDX]
    SEEDS = 10
    STEPS = 10

    optim_config: OptimConfig = optim_config_cls(
        lr=1e-3,
        betas=(0.9, 0.95),
    )
    mup_optimizer_type = optim_config.mup_optimizer_type()
    assert mup_optimizer_type is not None, "Optimizer does not support muP"

    def generate_model(width_idx: int, seed_idx: int) -> Tuple[int, torch.nn.Module]:
        d_model = D_MODELS[width_idx]
        hidden_size = HIDDEN_SIZES[width_idx]

        mup = MuPConfig(
            mup_optimizer_type,
            scaling_strategy=mup_scaling_strategy,
            width_scalings={
                MuPHyperParam.d_model: d_model / BASE_D_MODEL,
                MuPHyperParam.hidden_size: hidden_size / BASE_HIDDEN_SIZE,
                MuPHyperParam.head_dim: d_model / BASE_D_MODEL,
            },
        )

        ffn = FeedForwardConfig(
            hidden_size,
            mup=mup,
            bias=False,
        ).build(d_model)

        InitMethod.normal.init_feed_forward(
            ffn,
            d_model=d_model,
            block_idx=0,
            num_blocks=1,
            generator=torch.Generator().manual_seed(seed_idx),
        )

        return d_model, ffn

    coord_magnitudes, activation_shapes = train_and_collect_mup_data(
        generate_model,
        optim_config,
        len(D_MODELS),
        steps=STEPS,
        seeds=SEEDS,
        has_embedding_dim=True,
    )

    for name, magnitudes in coord_magnitudes.items():
        magnitudes = np.array(
            [np.mean(magnitudes[i : i + SEEDS]) for i in range(0, len(magnitudes), SEEDS)]
        )
        shapes = [activation_shapes[name][SEEDS * i] for i in range(len(magnitudes))]

        log_d_models = np.log2(np.array(D_MODELS))
        slope = np.polynomial.polynomial.Polynomial.fit(
            log_d_models, magnitudes / magnitudes[0], 1
        ).coef[1]
        assert slope <= MAX_NORMALIZED_SLOPE, (
            f"Coordinate magnitudes for {name} grow too much with width. "
            f"slope {slope}, magnitudes {magnitudes}, normalized magnitudes {magnitudes / magnitudes[0]} shapes {shapes}"
        )
        assert slope >= MIN_NORMALIZED_SLOPE, (
            f"Coordinate magnitudes for {name} decay too much with width. "
            f"slope {slope}, magnitudes {magnitudes}, normalized magnitudes {magnitudes / magnitudes[0]} shapes {shapes}"
        )


@pytest.mark.parametrize(
    "optim_config_cls, mup_scaling_strategy",
    [
        pytest.param(optim_config_cls, scaling_strategy)
        for optim_config_cls in [AdamConfig, AdamWConfig, SkipStepAdamWConfig]
        for scaling_strategy in MuPScalingStrategy
    ],
)
def test_mup_non_growing_coordinates(optim_config_cls: Type[OptimConfig], mup_scaling_strategy):
    """
    This is the most important (and slowest) test of muP validity. It runs a base model and its muP
    up-/down-scalings, and checks that the magnitude of the coordinates (i.e. entries) of the
    intermediate activations are not growing/decaying.

    Growing coordinates are never expected in muP beyond noise, so the acceptable threshold of growth
    is set closer to 1. Decaying is expected in attention logits and layers with scaling inputs at initialization,
    and in general it takes a while for muP to get rid of decaying. Thus decay threshold is set further from 1.
    These thresholds are broken very easily by the standard parametrization but we try to keep them tight
    because muP bugs might not break them as easily.
    """
    MIN_NORMALIZED_SLOPE = -0.6
    MAX_NORMALIZED_SLOPE = 0.05
    SEQ_LEN = 32
    VOCAB_SIZE = 1024
    SEEDS = 10
    STEPS = 100
    D_MODEL_MULTIPLIERS = [4, 8, 16]
    BASE_MULTIPLIER_IDX = 1
    base_d_model_multiplier = D_MODEL_MULTIPLIERS[BASE_MULTIPLIER_IDX]

    optim_config: OptimConfig = optim_config_cls(
        lr=5e-3,
        betas=(0.9, 0.95),
        group_overrides=[
            OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0))
        ],
    )
    mup_optimizer_type = optim_config.mup_optimizer_type()
    assert mup_optimizer_type is not None, "Optimizer does not support muP"

    def generate_model(width_idx: int, seed_idx: int) -> Tuple[int, torch.nn.Module]:
        d_model_multiplier = D_MODEL_MULTIPLIERS[width_idx]

        non_mup_model_config = get_transformer_config(
            None,
            vocab_size=VOCAB_SIZE,
            d_model_multiplier=d_model_multiplier,
            init_seed=seed_idx,
        )
        base_model_config = get_transformer_config(
            None,
            vocab_size=VOCAB_SIZE,
            d_model_multiplier=base_d_model_multiplier,
            init_seed=seed_idx,
        )

        mup_config = MuPConfig(
            mup_optimizer_type,
            scaling_strategy=mup_scaling_strategy,
            width_scalings=non_mup_model_config.get_mup_width_scalings(base_model_config),
        )

        model_config = get_transformer_config(
            mup_config,
            vocab_size=VOCAB_SIZE,
            d_model_multiplier=d_model_multiplier,
            init_seed=seed_idx,
        )

        model = model_config.build()

        model.init_weights(max_seq_len=SEQ_LEN)

        return SEQ_LEN, model

    seeded_init_coord_magnitudes, seeded_coord_change_stds = train_and_collect_mup_data(
        generate_model,
        optim_config,
        len(D_MODEL_MULTIPLIERS),
        vocab_size=VOCAB_SIZE,
        steps=STEPS,
        seeds=SEEDS,
    )

    for name, magnitudes in seeded_init_coord_magnitudes.items():
        magnitudes = np.array(
            [np.mean(magnitudes[i : i + SEEDS]) for i in range(0, len(magnitudes), SEEDS)]
        )

        log_d_model_multipliers = np.log2(np.array(D_MODEL_MULTIPLIERS))
        slope = np.polynomial.polynomial.Polynomial.fit(
            log_d_model_multipliers, magnitudes / magnitudes[0], 1
        ).coef[1]
        assert slope <= MAX_NORMALIZED_SLOPE, (
            f"Coordinate magnitudes for {name} grow too much with width. "
            f"slope {slope}, magnitudes {magnitudes}, normalized magnitudes {magnitudes / magnitudes[0]}"
        )
        assert slope >= MIN_NORMALIZED_SLOPE, (
            f"Coordinate magnitudes for {name} decay too much with width. "
            f"slope {slope}, magnitudes {magnitudes}, normalized magnitudes {magnitudes / magnitudes[0]}"
        )

    for name, stds in seeded_coord_change_stds.items():
        stds = np.array([np.mean(stds[i : i + SEEDS]) for i in range(0, len(stds), SEEDS)])

        log_d_model_multipliers = np.log2(np.array(D_MODEL_MULTIPLIERS))
        slope = np.polynomial.polynomial.Polynomial.fit(
            log_d_model_multipliers, stds / stds[0], 1
        ).coef[1]
        assert slope <= MAX_NORMALIZED_SLOPE, (
            f"Coordinate change stds for {name} grow too much with width. "
            f"slope {slope}, stds {stds}, normalized stds {stds / stds[0]}"
        )
        assert slope >= MIN_NORMALIZED_SLOPE, (
            f"Coordinate change stds for {name} decay too much with width. "
            f"slope {slope}, stds {stds}, normalized stds {stds / stds[0]}"
        )
