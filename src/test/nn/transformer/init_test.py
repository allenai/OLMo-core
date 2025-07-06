from typing import Optional

import pytest
import torch

from olmo_core.nn.attention import Attention, AttentionConfig
from olmo_core.nn.feed_forward import FeedForwardConfig
from olmo_core.nn.lm_head import LMHeadConfig
from olmo_core.nn.mup import MuPConfig, MuPHyperParam, MuPScalingStrategy
from olmo_core.nn.transformer.config import TransformerBlockType, TransformerConfig
from olmo_core.nn.transformer.init import InitMethod


def get_transformer_config(
    mup_config: Optional[MuPConfig] = None,
    vocab_size: int = 100,
) -> TransformerConfig:
    return TransformerConfig.llama_like(
        d_model=8,
        vocab_size=vocab_size,
        n_layers=2,
        n_heads=2,
        hidden_size_multiplier=1,
        hidden_size_multiple_of=16,
        block_name=TransformerBlockType.reordered_norm,
        qk_norm=True,
        rope_theta=500_000,
        layer_norm_eps=1e-6,
        fused_ops=False,
        use_flash=False,
        mup=mup_config,
    )


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
def test_mup_no_width_scaling_same_init(mup_scaling_strategy):
    model_config = get_transformer_config()

    mup_config = MuPConfig(scaling_strategy=mup_scaling_strategy)
    mup_model_config = get_transformer_config(mup_config)

    model = model_config.build()
    mup_model = mup_model_config.build()

    model.init_weights()
    mup_model.init_weights()

    model_params = dict(model.named_parameters())
    mup_model_params = dict(mup_model.named_parameters())

    for name, param in model_params.items():
        assert_distributions_close(param, mup_model_params[name], name=name)


def test_mup_no_init_scaling_same_init():
    d_model_multiplier = 16
    model_config = get_transformer_config()

    mup_config = MuPConfig(
        scaling_strategy=MuPScalingStrategy.constant_init_std,
        width_scalings={
            MuPHyperParam.d_model: d_model_multiplier,
            MuPHyperParam.head_dim: d_model_multiplier,
            MuPHyperParam.hidden_size: d_model_multiplier,
        },
    )
    mup_model_config = get_transformer_config(mup_config)

    model = model_config.build()
    mup_model = mup_model_config.build()

    model.init_weights()
    mup_model.init_weights()

    model_params = dict(model.named_parameters())
    mup_model_params = dict(mup_model.named_parameters())

    for name, param in model_params.items():
        assert_distributions_close(param, mup_model_params[name], name=name)


@pytest.mark.parametrize(
    "mup_scaling_strategy",
    [pytest.param(scaling_strategy) for scaling_strategy in MuPScalingStrategy],
)
def test_feed_forward_mup_scaling_init_std(mup_scaling_strategy):
    d_model_multiplier = 16
    d_model = 8 * d_model_multiplier
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
    d_model = 8 * d_model_multiplier
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
def test_lm_head_mup_scaling_init_std(mup_scaling_strategy):
    d_model_multiplier = 16
    d_model = 8 * d_model_multiplier
    vocab_size = 100

    lm_head = LMHeadConfig().build(d_model=d_model, vocab_size=vocab_size)

    mup_config = MuPConfig(
        scaling_strategy=mup_scaling_strategy,
        width_scalings={
            MuPHyperParam.d_model: d_model_multiplier,
            MuPHyperParam.hidden_size: 1.0,
            MuPHyperParam.head_dim: d_model_multiplier,
        },
    )
    mup_lm_head = LMHeadConfig(mup=mup_config).build(d_model=d_model, vocab_size=vocab_size)

    init = InitMethod.normal
    init.init_final_w_out(lm_head.w_out, d_model=d_model)
    init.init_final_w_out(mup_lm_head.w_out, d_model=d_model, mup=mup_lm_head.mups["w_out.weight"])

    params = dict(lm_head.named_parameters())
    mup_params = dict(mup_lm_head.named_parameters())

    mup_init_std_multiplier = mup_lm_head.mups["w_out.weight"].init_std_multiplier or 1.0
    assert_distributions_close(
        params["w_out.weight"],
        mup_params["w_out.weight"],
        scale_factor=1 / mup_init_std_multiplier,
        name="w_out.weight",
        atol=1e-3,
    )
