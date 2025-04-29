from collections import defaultdict
from functools import partial
from typing import List, Optional, Tuple, Type

import numpy as np
import pytest
import torch

from olmo_core.nn.cross_entropy_loss import CrossEntropyLoss
from olmo_core.nn.feed_forward import FeedForwardConfig
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
        pytest.param(MuPOptimizerType.adam, MuPScalingStrategy.constant_init_std, 1 / (2**3 * 0.7)),
        pytest.param(MuPOptimizerType.adam, MuPScalingStrategy.table_8, 1 / (2**3 * 0.7)),
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
        pytest.param(MuPOptimizerType.adam, MuPScalingStrategy.table_8, None),
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
        pytest.param(MuPOptimizerType.adam, MuPScalingStrategy.table_8, None),
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
        pytest.param(MuPOptimizerType.adam, MuPScalingStrategy.table_8, 1 / (2 ** (3 / 2))),
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
    ]
)
def test_mup_no_width_scaling_same_optim_groups(optim_config_cls: Type[OptimConfig], mup_scaling_strategy):
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


@pytest.mark.parametrize(
    "optim_config_cls, mup_scaling_strategy",
    [
        pytest.param(optim_config_cls, scaling_strategy)
        for optim_config_cls in [AdamConfig, AdamWConfig, SkipStepAdamWConfig]
        for scaling_strategy in MuPScalingStrategy
    ]
)
def test_ffn_mup_non_growing_coordinates(optim_config_cls, mup_scaling_strategy):
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
    SEEDS = 10
    STEPS = 50
    BATCH_SIZE = 8
    DECAY_THRESHOLD = 0.7
    GROWTH_THRESHOLD = 1.2
    D_MODELS = [2, 4, 8, 16, 32]
    # HIDDEN_SIZES = [3, 6, 9, 12, 15]
    MULTIPLIER = 16
    HIDDEN_SIZES = [8, 8, 8, 8, 8]
    BASE_MULTIPLIER_IDX = 2
    BASE_D_MODEL = D_MODELS[BASE_MULTIPLIER_IDX] * MULTIPLIER
    BASE_HIDDEN_SIZE = HIDDEN_SIZES[BASE_MULTIPLIER_IDX] * MULTIPLIER

    def collect_mean_coord_magnitude(
        debug_state: List[Tuple[str, tuple, float]],
        name: str,
        module: torch.nn.Module,
        args,
        output,
    ):
        del module
        if isinstance(args[0], torch.Tensor):
            inp_vector = args[0].detach().float()
            state_name = f"{name}|input"
            debug_state.append(
                (state_name, inp_vector.shape, inp_vector.norm(p=1.0).item() / inp_vector.numel())
            )
        if isinstance(output, torch.Tensor):
            state_name = f"{name}|{len(debug_state)}|output"
            debug_state.append(
                (
                    state_name,
                    output.shape,
                    output.detach().float().norm(p=1.0).item() / output.numel(),
                )
            )

    optim_config: OptimConfig = optim_config_cls(
        lr=5e-3,
        betas=(0.9, 0.95),
    )
    mup_optimizer_type = optim_config.mup_optimizer_type()
    assert mup_optimizer_type is not None, "Optimizer does not support muP"

    # Dict[activation_name, [field_for_different_multiplers_and_seeds]]
    coord_magnitudes: defaultdict[str, List[float]] = defaultdict(list)
    activation_shapes: defaultdict[str, List[tuple]] = defaultdict(list)
    for d_model, hidden_size in zip(D_MODELS, HIDDEN_SIZES):
        d_model *= MULTIPLIER
        hidden_size *= MULTIPLIER
        for i_seed in range(SEEDS):
            mup = MuPConfig(
                mup_optimizer_type,
                scaling_strategy=mup_scaling_strategy,
                width_scalings={
                    MuPHyperParam.d_model: d_model / BASE_D_MODEL,
                    MuPHyperParam.hidden_size: hidden_size / BASE_HIDDEN_SIZE,
                    MuPHyperParam.head_dim: d_model / BASE_D_MODEL,
                }
            )

            ffn = FeedForwardConfig(
                hidden_size,
                mup=mup,
            ).build(d_model)

            InitMethod.normal.init_feed_forward(ffn, d_model=d_model, block_idx=0, num_blocks=1)
            optim = optim_config.build(ffn)


            # Train for a bit
            for i in range(STEPS):
                inp = torch.randn(BATCH_SIZE, d_model)
                labels = torch.randint(0, d_model, (BATCH_SIZE,))
                optim.zero_grad(set_to_none=True)
                logits = ffn(inp)

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

            activation_shapes_and_coord_magnitudes: List[Tuple[str, tuple, float]] = []
            for name, module in ffn.named_modules():
                module.register_forward_hook(
                    partial(
                        collect_mean_coord_magnitude,
                        activation_shapes_and_coord_magnitudes,
                        name,
                    )
                )

            ffn(torch.randn(BATCH_SIZE, d_model))

            for name, shape, magnitude in activation_shapes_and_coord_magnitudes:
                coord_magnitudes[name].append(magnitude)
                activation_shapes[name].append(shape)

    for name, magnitudes in coord_magnitudes.items():
        magnitudes = [np.mean(magnitudes[i : i + SEEDS]) for i in range(0, len(magnitudes), SEEDS)]
        shapes = [activation_shapes[name][SEEDS * i] for i in range(len(magnitudes))]

        for i, magnitude in enumerate(magnitudes):
            if i < BASE_MULTIPLIER_IDX:
                ratio = magnitudes[BASE_MULTIPLIER_IDX] / magnitude
            else:
                ratio = magnitude / magnitudes[BASE_MULTIPLIER_IDX]

            assert ratio <= GROWTH_THRESHOLD, (
                f"Coordinate magnitudes for {name} grow too much with width. "
                f"d_model {D_MODELS[i]}, "
                f"base d_model {D_MODELS[BASE_MULTIPLIER_IDX]}, "
                f"ratio {ratio}, magnitudes {magnitudes}, shapes {shapes}"
            )
            assert ratio >= DECAY_THRESHOLD, (
                f"Coordinate magnitudes for {name} decay too much with width. "
                f"d_model {D_MODELS[i]}, "
                f"base d_model {D_MODELS[BASE_MULTIPLIER_IDX]}, "
                f"ratio {ratio}, magnitudes {magnitudes}, shapes {shapes}"
            )



@pytest.mark.parametrize(
    "optim_config_cls, mup_scaling_strategy",
    [
        pytest.param(optim_config_cls, scaling_strategy)
        for optim_config_cls in [AdamConfig, AdamWConfig, SkipStepAdamWConfig]
        for scaling_strategy in MuPScalingStrategy
    ]
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
    SEEDS = 10
    STEPS = 50
    BATCH_SIZE = 2
    SEQ_LEN = 32
    DECAY_THRESHOLD = 0.3
    GROWTH_THRESHOLD = 2
    D_MODEL_MULTIPLIERS = [1, 2, 4, 8, 16]
    BASE_MULTIPLIER_IDX = 1
    base_d_model_multiplier = D_MODEL_MULTIPLIERS[BASE_MULTIPLIER_IDX]

    def collect_mean_coord_magnitude(
        debug_state: List[Tuple[str, tuple, float]],
        name: str,
        module: torch.nn.Module,
        args,
        output,
    ):
        del module
        # if isinstance(args[0], torch.Tensor):
        #     inp_vector = args[0].detach().float()
        #     state_name = f"{name}|input"
        #     debug_state.append(
        #         (state_name, inp_vector.shape, inp_vector.norm(p=1.0).item() / inp_vector.numel())
        #     )
        if isinstance(output, torch.Tensor):
            state_name = f"{name}|{len(debug_state)}|output"
            debug_state.append(
                (
                    state_name,
                    output.shape,
                    output.detach().float().norm(p=1.0).item() / output.numel(),
                )
            )

    mup_optimizer_type = optim_config_cls.mup_optimizer_type()
    assert mup_optimizer_type is not None, "Optimizer does not support muP"

    optim_config: OptimConfig = optim_config_cls(
        lr=5e-3,
        weight_decay=0.1,
        betas=(0.9, 0.95),
        group_overrides=[
            OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0))
        ],
    )

    # Dict[activation_name, [field_for_different_multiplers_and_seeds]]
    coord_magnitudes: defaultdict[str, List[float]] = defaultdict(list)
    activation_shapes: defaultdict[str, List[tuple]] = defaultdict(list)
    for d_model_multiplier in D_MODEL_MULTIPLIERS:
        for i_seed in range(SEEDS):
            vocab_size = BATCH_SIZE * SEQ_LEN + STEPS

            non_mup_model_config = get_transformer_config(
                None,
                vocab_size=vocab_size,
                d_model_multiplier=d_model_multiplier,
                init_seed=i_seed,
            )
            base_model_config = get_transformer_config(
                None,
                vocab_size=vocab_size,
                d_model_multiplier=base_d_model_multiplier,
                init_seed=i_seed,
            )

            mup_config = MuPConfig(
                mup_optimizer_type,
                scaling_strategy=mup_scaling_strategy,
                width_scalings=non_mup_model_config.get_mup_width_scalings(base_model_config),
            )
            model_config = get_transformer_config(
                mup_config,
                vocab_size=vocab_size,
                d_model_multiplier=d_model_multiplier,
                init_seed=i_seed,
            )

            model = model_config.build()
            optim = optim_config.build(model)

            model.init_weights(max_seq_len=SEQ_LEN)

            # Train for a bit
            for i in range(STEPS):
                inp = torch.arange(0, BATCH_SIZE * SEQ_LEN).reshape(BATCH_SIZE, SEQ_LEN) + i
                labels = inp.detach().clone().contiguous()
                optim.zero_grad(set_to_none=True)
                _, loss, _, _ = model(inp, labels=labels, return_logits=False)

                # # Get loss to optimize for.
                # loss = ce_loss
                # if z_loss is not None:
                #     loss += z_loss

                loss.backward()
                optim.step()

            activation_shapes_and_coord_magnitudes: List[Tuple[str, tuple, float]] = []
            for name, module in model.named_modules():
                module.register_forward_hook(
                    partial(
                        collect_mean_coord_magnitude,
                        activation_shapes_and_coord_magnitudes,
                        name,
                    )
                )

            model(torch.arange(0, BATCH_SIZE * SEQ_LEN).reshape(BATCH_SIZE, SEQ_LEN))

            for name, shape, magnitude in activation_shapes_and_coord_magnitudes:
                coord_magnitudes[name].append(magnitude)
                activation_shapes[name].append(shape)

    for name, magnitudes in coord_magnitudes.items():
        magnitudes = [np.mean(magnitudes[i : i + SEEDS]) for i in range(0, len(magnitudes), SEEDS)]
        shapes = [activation_shapes[name][SEEDS * i] for i in range(len(magnitudes))]

        for i, magnitude in enumerate(magnitudes):
            if i < BASE_MULTIPLIER_IDX:
                ratio = magnitudes[BASE_MULTIPLIER_IDX] / magnitude
            else:
                ratio = magnitude / magnitudes[BASE_MULTIPLIER_IDX]

            assert ratio <= GROWTH_THRESHOLD, (
                f"Coordinate magnitudes for {name} grow too much with width. "
                f"d_model multiplier {D_MODEL_MULTIPLIERS[i]}, "
                f"base d_model multiplier {D_MODEL_MULTIPLIERS[BASE_MULTIPLIER_IDX]}, "
                f"ratio {ratio}, magnitudes {magnitudes}, shapes {shapes}"
            )
            assert ratio >= DECAY_THRESHOLD, (
                f"Coordinate magnitudes for {name} decay too much with width. "
                f"d_model multiplier {D_MODEL_MULTIPLIERS[i]}, "
                f"base d_model multiplier {D_MODEL_MULTIPLIERS[BASE_MULTIPLIER_IDX]}, "
                f"ratio {ratio}, magnitudes {magnitudes}, shapes {shapes}"
            )
