import math

import torch
import torch.nn as nn

from olmo_core.config import DType
from olmo_core.nn.attention import AttentionConfig
from olmo_core.nn.feed_forward import FeedForwardConfig
from olmo_core.nn.layer_norm import LayerNormConfig, LayerNormType
from olmo_core.nn.lm_head import LMHeadConfig
from olmo_core.nn.transformer import (
    TransformerBlockConfig,
    TransformerBlockType,
    TransformerConfig,
)
from olmo_core.optim import AdamWConfig
from olmo_core.train.train_module import TransformerTrainModuleConfig


def _tiny_transformer_config() -> TransformerConfig:
    layer_norm = LayerNormConfig(name=LayerNormType.rms, bias=False)
    return TransformerConfig(
        d_model=8,
        vocab_size=6,
        n_layers=1,
        block=TransformerBlockConfig(
            name=TransformerBlockType.default,
            attention=AttentionConfig(n_heads=2),
            layer_norm=layer_norm,
            feed_forward=FeedForwardConfig(hidden_size=16, bias=False),
        ),
        lm_head=LMHeadConfig(layer_norm=layer_norm, bias=False, dtype=DType.float32),
    )


def test_early_fusion_weighted_unembedding_sum_uses_raw_kn_mass():
    model = _tiny_transformer_config().build(init_device="cpu")
    assert model.lm_head is not None
    model.lm_head.register_parameter(
        "early_fusion_alpha_log",
        nn.Parameter(torch.tensor([math.log(0.5)], dtype=torch.float32)),
    )
    with torch.no_grad():
        model.lm_head.w_out.weight.copy_(
            torch.arange(6 * 8, dtype=torch.float32).reshape(6, 8)
        )

    ngram_token_ids = torch.tensor([[[1, 3, 0], [2, 5, 0]]], dtype=torch.long)
    ngram_log_probs = torch.log(
        torch.tensor([[[0.25, 0.50, 0.0], [0.10, 0.20, 0.0]]], dtype=torch.float32)
    )

    out = model._compute_early_fusion_ngram_embedding(
        ngram_token_ids,
        ngram_log_probs,
        dtype=torch.float32,
    )

    weights = model.lm_head.w_out.weight
    expected = torch.stack(
        [
            0.5 * (0.25 * weights[1] + 0.50 * weights[3]),
            0.5 * (0.10 * weights[2] + 0.20 * weights[5]),
        ],
        dim=0,
    ).unsqueeze(0)
    assert out.shape == (1, 2, 8)
    torch.testing.assert_close(out, expected)


def test_train_module_registers_early_fusion_alpha_with_optimizer_override():
    model = _tiny_transformer_config().build(init_device="cpu")
    train_module = TransformerTrainModuleConfig(
        rank_microbatch_size=4,
        max_sequence_length=4,
        optim=AdamWConfig(lr=1e-3),
        early_fusion_ngram=True,
        early_fusion_alpha_init=0.2,
        early_fusion_alpha_lr=1e-4,
        early_fusion_ngram_table_dir="/tmp/nonexistent-ngram-table",
    ).build(model, device=torch.device("cpu"))

    alpha_log = train_module._early_fusion_alpha_log_param()
    torch.testing.assert_close(
        torch.exp(alpha_log.detach()),
        torch.tensor([0.2], dtype=torch.float32),
    )

    alpha_group = next(
        group
        for group in train_module.optim.param_groups
        if any(param is alpha_log for param in group["params"])
    )
    assert alpha_group["lr"] == 1e-4
    assert alpha_group["weight_decay"] == 0.0
