from collections import defaultdict
from functools import partial
from typing import List, Optional, Tuple

import numpy as np
import pytest
import torch

from olmo_core.nn.attention import Attention, AttentionConfig
from olmo_core.nn.feed_forward import FeedForwardConfig
from olmo_core.nn.mup import MuPConfig, MuPHyperParam, MuPScalingStrategy
from olmo_core.nn.transformer.config import TransformerBlockType, TransformerConfig
from olmo_core.nn.transformer.init import InitMethod
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


def assert_distributions_close(
    base_tensor: torch.Tensor,
    actual_tensor: torch.Tensor,
    scale_factor: float = 1.0,
    name: Optional[str] = None,
    atol: float = 1e-3,
    rtol: float = 1e-3,
):
    expected_mean = base_tensor.mean()
    actual_mean = actual_tensor.mean() * scale_factor

    torch.testing.assert_close(
        expected_mean,
        actual_mean,
        atol=atol,
        rtol=rtol,
        msg=f"Expected mean value for {name} = {expected_mean}, actual = {actual_mean}",
    )

    expected_std = base_tensor.std()
    actual_std = actual_tensor.std() * scale_factor

    torch.testing.assert_close(
        expected_std,
        actual_std,
        atol=atol,
        rtol=rtol,
        msg=f"Expected std value for {name} = {expected_std}, actual = {actual_std}",
    )


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


@pytest.mark.parametrize(
    "mup_scaling_strategy",
    [pytest.param(scaling_strategy) for scaling_strategy in MuPScalingStrategy],
)
def test_mup_no_width_scaling_same_init(mup_scaling_strategy):
    d_model_multiplier = 16
    model_config = get_transformer_config(d_model_multiplier=d_model_multiplier)

    mup_config = MuPConfig(scaling_strategy=mup_scaling_strategy)
    mup_model_config = get_transformer_config(mup_config, d_model_multiplier=d_model_multiplier)

    model = model_config.build()
    mup_model = mup_model_config.build()

    model.init_weights()
    mup_model.init_weights()

    model_params = dict(model.named_parameters())
    mup_model_params = dict(mup_model.named_parameters())

    for name, param in model_params.items():
        assert_distributions_close(param, mup_model_params[name], name=name)


def test_mup_no_output_scaling_same_output():
    model_config = get_transformer_config()

    mup_config = MuPConfig(
        scaling_strategy=MuPScalingStrategy.constant_outputs,
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


def test_mup_no_init_scaling_same_init():
    d_model_multiplier = 16
    model_config = get_transformer_config(d_model_multiplier=d_model_multiplier)

    mup_config = MuPConfig(
        scaling_strategy=MuPScalingStrategy.constant_init_std,
        width_scalings={
            MuPHyperParam.d_model: d_model_multiplier,
            MuPHyperParam.head_dim: d_model_multiplier,
            MuPHyperParam.hidden_size: d_model_multiplier,
        },
    )
    mup_model_config = get_transformer_config(mup_config, d_model_multiplier=d_model_multiplier)

    model = model_config.build()
    mup_model = mup_model_config.build()

    model.init_weights()
    mup_model.init_weights()

    model_params = dict(model.named_parameters())
    mup_model_params = dict(mup_model.named_parameters())

    for name, param in model_params.items():
        assert_distributions_close(param, mup_model_params[name], name=name)


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
def test_feed_forward_mup_scaling_init_std(mup_scaling_strategy):
    d_model_multiplier = 16
    d_model = 128 * d_model_multiplier
    hidden_size = 64

    feed_forward = FeedForwardConfig(hidden_size, bias=False).build(d_model)

    mup_config = MuPConfig(
        scaling_strategy=mup_scaling_strategy,
        width_scalings={
            MuPHyperParam.d_model: d_model_multiplier,
            MuPHyperParam.hidden_size: 1.0,
            MuPHyperParam.head_dim: d_model_multiplier,
        },
    )
    mup_feed_forward = FeedForwardConfig(hidden_size, bias=False, mup=mup_config).build(d_model)

    init = InitMethod.normal
    init.init_feed_forward(feed_forward, d_model=d_model, block_idx=2, num_blocks=8)
    init.init_feed_forward(mup_feed_forward, d_model=d_model, block_idx=2, num_blocks=8)

    params = dict(feed_forward.named_parameters())
    mup_params = dict(mup_feed_forward.named_parameters())

    for name in ["w1.weight", "w2.weight", "w3.weight"]:
        mup_init_std_multiplier = mup_feed_forward.mups[name].init_std_multiplier or 1.0
        assert_distributions_close(
            params[name],
            mup_params[name],
            scale_factor=1 / mup_init_std_multiplier,
            name=name,
            atol=1e-3,
        )


@pytest.mark.parametrize(
    "mup_scaling_strategy",
    [pytest.param(scaling_strategy) for scaling_strategy in MuPScalingStrategy],
)
def test_attention_mup_scaling_init_std(mup_scaling_strategy):
    d_model_multiplier = 16
    d_model = 128 * d_model_multiplier
    n_heads = 32

    attention = AttentionConfig(n_heads=n_heads, bias=False).build(d_model)

    mup_config = MuPConfig(
        scaling_strategy=mup_scaling_strategy,
        width_scalings={
            MuPHyperParam.d_model: d_model_multiplier,
            MuPHyperParam.hidden_size: 1.0,
            MuPHyperParam.head_dim: d_model_multiplier,
        },
    )
    mup_attention = AttentionConfig(n_heads=n_heads, bias=False, mup=mup_config).build(d_model)
    assert isinstance(mup_attention, Attention)

    init = InitMethod.normal
    init.init_attention(attention, d_model=d_model, block_idx=2, num_blocks=8)
    init.init_attention(mup_attention, d_model=d_model, block_idx=2, num_blocks=8)

    params = dict(attention.named_parameters())
    mup_params = dict(mup_attention.named_parameters())

    for name in ["w_q.weight", "w_k.weight", "w_v.weight", "w_out.weight"]:
        mup_init_std_multiplier = mup_attention.mups[name].init_std_multiplier or 1.0
        assert_distributions_close(
            params[name],
            mup_params[name],
            scale_factor=1 / mup_init_std_multiplier,
            name=name,
            atol=1e-3,
        )


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

    # Dict[activation_name, [activation_norm_for_different_multiplers]]
    activation_norms: defaultdict[str, List[float]] = defaultdict(list)
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

            model_activation_shapes_and_norms: List[Tuple[str, tuple, float]] = []
            for name, module in model.named_modules():
                module.register_forward_hook(
                    partial(collect_mean_coord_magnitude, model_activation_shapes_and_norms, name)
                )

            model(torch.arange(0, BATCH_SIZE * SEQ_LEN).reshape(BATCH_SIZE, SEQ_LEN))

            for name, shape, norm in model_activation_shapes_and_norms:
                activation_norms[name].append(norm)
                activation_shapes[name].append(shape)

    for name, norms in activation_norms.items():
        norms = [np.mean(norms[i : i + SEEDS]) for i in range(0, len(norms), SEEDS)]
        shapes = [activation_shapes[name][SEEDS * i] for i in range(len(norms))]

        for i, norm in enumerate(norms):
            if i < BASE_MULTIPLIER_IDX:
                ratio = norms[BASE_MULTIPLIER_IDX] / norm
            else:
                ratio = norm / norms[BASE_MULTIPLIER_IDX]

            print(name, shapes, norms)
            assert ratio <= GROWTH_THRESHOLD, (
                f"Coordinate magnitudes for {name} grow too much with width. "
                f"d_model multiplier {D_MODEL_MULTIPLIERS[i]}, "
                f"base d_model multiplier {D_MODEL_MULTIPLIERS[BASE_MULTIPLIER_IDX]}, "
                f"ratio {ratio}, norms {norms}, shapes {shapes}"
            )
            assert ratio >= DECAY_THRESHOLD, (
                f"Coordinate magnitudes for {name} decay too much with width. "
                f"d_model multiplier {D_MODEL_MULTIPLIERS[i]}, "
                f"base d_model multiplier {D_MODEL_MULTIPLIERS[BASE_MULTIPLIER_IDX]}, "
                f"ratio {ratio}, norms {norms}, shapes {shapes}"
            )
