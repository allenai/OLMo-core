from collections import defaultdict
from functools import partial
from typing import List, Optional, Tuple

import numpy as np
import pytest
import torch

from olmo_core.nn.mup import MuPConfig, MuPHyperParam, MuPScalingStrategy
from olmo_core.nn.transformer.config import TransformerBlockType, TransformerConfig
from olmo_core.optim.adamw import AdamWConfig
from olmo_core.optim.config import OptimGroupOverride


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
    "mup_scaling_strategy, expected_multiplier",
    [
        pytest.param(MuPScalingStrategy.constant_inputs, None),
        pytest.param(MuPScalingStrategy.constant_lr, 1 / (2**3 * 5)),
        pytest.param(MuPScalingStrategy.constant_init_std, 1 / (2**3 * 5)),
        pytest.param(MuPScalingStrategy.table_8, 1 / (2**3 * 5)),
    ],
)
def test_mup_input_multiplier_scaling_input(mup_scaling_strategy, expected_multiplier):
    mup_config = MuPConfig(
        scaling_strategy=mup_scaling_strategy,
        width_scalings={
            MuPHyperParam.d_model: 2,
            MuPHyperParam.hidden_size: 5,
            MuPHyperParam.head_dim: 2,
        },
    )
    mup = mup_config.build({MuPHyperParam.d_model: 3, MuPHyperParam.hidden_size: 1}, {})

    torch.testing.assert_close(mup.input_multiplier, expected_multiplier)


@pytest.mark.parametrize(
    "mup_scaling_strategy, expected_multiplier",
    [
        pytest.param(MuPScalingStrategy.constant_inputs, None),
        pytest.param(MuPScalingStrategy.constant_lr, 1 / (2**3)),
        pytest.param(MuPScalingStrategy.constant_init_std, 1 / (2 ** (3 / 2))),
        pytest.param(MuPScalingStrategy.table_8, None),
    ],
)
def test_mup_input_multiplier_scaling_input_and_output(mup_scaling_strategy, expected_multiplier):
    mup_config = MuPConfig(
        scaling_strategy=mup_scaling_strategy,
        width_scalings={
            MuPHyperParam.d_model: 2,
            MuPHyperParam.hidden_size: 5,
            MuPHyperParam.head_dim: 2,
        },
    )
    mup = mup_config.build({MuPHyperParam.d_model: 3}, {MuPHyperParam.hidden_size: 1})

    torch.testing.assert_close(mup.input_multiplier, expected_multiplier)


@pytest.mark.parametrize(
    "mup_scaling_strategy, expected_multiplier",
    [
        pytest.param(MuPScalingStrategy.constant_inputs, 1 / (2**3 * 5)),
        pytest.param(MuPScalingStrategy.constant_lr, None),
        pytest.param(MuPScalingStrategy.constant_init_std, None),
        pytest.param(MuPScalingStrategy.table_8, None),
    ],
)
def test_mup_init_std_multiplier_scaling_input(mup_scaling_strategy, expected_multiplier):
    mup_config = MuPConfig(
        scaling_strategy=mup_scaling_strategy,
        width_scalings={
            MuPHyperParam.d_model: 2,
            MuPHyperParam.hidden_size: 5,
            MuPHyperParam.head_dim: 2,
        },
    )
    mup = mup_config.build({MuPHyperParam.d_model: 3, MuPHyperParam.hidden_size: 1}, {})

    torch.testing.assert_close(mup.init_std_multiplier, expected_multiplier)


@pytest.mark.parametrize(
    "mup_scaling_strategy, expected_multiplier",
    [
        pytest.param(MuPScalingStrategy.constant_inputs, 1 / (2 ** (3 / 2))),
        pytest.param(MuPScalingStrategy.constant_lr, 2 ** (3 / 2)),
        pytest.param(MuPScalingStrategy.constant_init_std, None),
        pytest.param(MuPScalingStrategy.table_8, 1 / (2 ** (3 / 2))),
    ],
)
def test_mup_init_std_multiplier_scaling_input_and_output(
    mup_scaling_strategy, expected_multiplier
):
    mup_config = MuPConfig(
        scaling_strategy=mup_scaling_strategy,
        width_scalings={
            MuPHyperParam.d_model: 2,
            MuPHyperParam.hidden_size: 5,
            MuPHyperParam.head_dim: 2,
        },
    )
    mup = mup_config.build({MuPHyperParam.d_model: 3}, {MuPHyperParam.hidden_size: 1})

    torch.testing.assert_close(mup.init_std_multiplier, expected_multiplier)


@pytest.mark.parametrize(
    "mup_scaling_strategy",
    [pytest.param(scaling_strategy) for scaling_strategy in MuPScalingStrategy],
)
def test_mup_no_width_scaling_same_output(mup_scaling_strategy):
    model_config = get_transformer_config()

    mup_config = MuPConfig(scaling_strategy=mup_scaling_strategy)
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
    "mup_scaling_strategy",
    [pytest.param(scaling_strategy) for scaling_strategy in MuPScalingStrategy],
)
def test_mup_no_width_scaling_same_optim_groups(mup_scaling_strategy):
    model_config = get_transformer_config()

    mup_config = MuPConfig(scaling_strategy=mup_scaling_strategy)
    mup_model_config = get_transformer_config(mup_config)

    model = model_config.build()
    mup_model = mup_model_config.build()

    optim_config = AdamWConfig(
        lr=1e-2,
        weight_decay=0.1,
        betas=(0.9, 0.95),
        group_overrides=[
            OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0))
        ],
        fused=True,
    )

    model.init_weights()
    mup_model.to_empty(device=model.device)
    mup_model.load_state_dict(model.state_dict())

    optim: torch.optim.Optimizer = optim_config.build(model)
    mup_optim: torch.optim.Optimizer = optim_config.build(mup_model)

    torch.testing.assert_close(optim.param_groups, mup_optim.param_groups)


def test_mup_no_input_scaling_same_output():
    model_config = get_transformer_config()

    mup_config = MuPConfig(
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


def test_mup_no_lr_scaling_same_optim_groups():
    model_config = get_transformer_config()

    mup_config = MuPConfig(
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

    optim_config = AdamWConfig(
        lr=1e-2,
        weight_decay=0.1,
        betas=(0.9, 0.95),
        group_overrides=[
            OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0))
        ],
        fused=True,
    )

    model.init_weights()
    mup_model.to_empty(device=model.device)
    mup_model.load_state_dict(model.state_dict())

    optim: torch.optim.Optimizer = optim_config.build(model)
    mup_optim: torch.optim.Optimizer = optim_config.build(mup_model)

    torch.testing.assert_close(optim.param_groups, mup_optim.param_groups)


@pytest.mark.parametrize(
    "mup_scaling_strategy",
    [pytest.param(scaling_strategy) for scaling_strategy in MuPScalingStrategy],
)
def test_mup_non_growing_coordinates(mup_scaling_strategy):
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
    BASE_MULTIPLIER_IDX = 2
    base_d_model_multiplier = D_MODEL_MULTIPLIERS[BASE_MULTIPLIER_IDX]

    def collect_mean_coord_magnitude(
        debug_state: List[Tuple[str, tuple, float]],
        name: str,
        module: torch.nn.Module,
        args,
        output,
    ):
        del module, args
        if isinstance(output, torch.Tensor):
            state_name = f"{name}|{len(debug_state)}|output"
            debug_state.append(
                (
                    state_name,
                    output.shape,
                    output.detach().float().norm(p=1.0).item() / output.numel(),
                )
            )

    optim_config = AdamWConfig(
        lr=1e-2,
        weight_decay=0.1,
        betas=(0.9, 0.95),
        group_overrides=[
            OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0))
        ],
        fused=True,
    )

    #
    # Dict[activation_name, [field_for_different_multiplers_and_seeds]]
    coord_magnitudes: defaultdict[str, List[float]] = defaultdict(list)
    activation_shapes: defaultdict[str, List[tuple]] = defaultdict(list)
    for d_model_multiplier in D_MODEL_MULTIPLIERS:
        mup_config = MuPConfig(
            scaling_strategy=mup_scaling_strategy,
            width_scalings={
                MuPHyperParam.d_model: d_model_multiplier / base_d_model_multiplier,
                MuPHyperParam.hidden_size: d_model_multiplier / base_d_model_multiplier,
                MuPHyperParam.head_dim: d_model_multiplier / base_d_model_multiplier,
            },
        )
        for i_seed in range(SEEDS):
            vocab_size = BATCH_SIZE * SEQ_LEN + STEPS
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
                _, ce_loss, z_loss = model(inp, labels=labels, return_logits=False)

                # Get loss to optimize for.
                loss = ce_loss
                if z_loss is not None:
                    loss += z_loss

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
