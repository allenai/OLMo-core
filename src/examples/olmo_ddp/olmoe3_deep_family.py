"""Candidate deeper OLMoE3 production family.

This module is intentionally separate from ``olmoe3_final_family`` so the
qualified shallow production candidates remain reproducible while we test the
new 16/24/40-layer family.
"""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass

from olmoe3_final_family import (
    NUM_EXPERTS,
    TOP_K,
    VOCAB_SIZE,
    _attention,
    _dense_first,
    _kda,
    _moe_block,
    _norm,
)

from olmo_core.config import DType
from olmo_core.nn.lm_head import LMHeadConfig
from olmo_core.nn.transformer import OLMoDDPModelConfig, TransformerType

LATENT_COMPRESSION = 2
MODEL_SIZES = ("small", "medium", "large")


@dataclass(frozen=True)
class Geometry:
    """Geometry for one rung of the controlled deeper family."""

    d_model: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    expected_active_params: int
    expected_active_non_embedding_params: int
    expected_total_params: int

    @property
    def latent_dim(self) -> int:
        return self.d_model // LATENT_COMPRESSION

    @property
    def expert_hidden_size(self) -> int:
        return self.d_model

    @property
    def full_attention_layers(self) -> tuple[int, ...]:
        # Keep layer 0 dense+KDA and use a controlled 7:1 KDA:FA cadence.
        return tuple(range(7, self.n_layers, 8))


GEOMETRIES: dict[str, Geometry] = {
    "small": Geometry(
        d_model=1024,
        n_layers=16,
        n_heads=8,
        n_kv_heads=4,
        expected_active_params=794_230_912,
        expected_active_non_embedding_params=691_470_464,
        expected_total_params=12_496_339_072,
    ),
    "medium": Geometry(
        d_model=1536,
        n_layers=24,
        n_heads=16,
        n_kv_heads=8,
        expected_active_params=2_387_524_992,
        expected_active_non_embedding_params=2_233_384_320,
        expected_total_params=42_759_798_144,
    ),
    "large": Geometry(
        d_model=1536,
        n_layers=40,
        n_heads=16,
        n_kv_heads=8,
        expected_active_params=3_780_501_120,
        expected_active_non_embedding_params=3_626_360_448,
        expected_total_params=72_237_833_856,
    ),
}


def geometry(model_size: str) -> Geometry:
    """Return the geometry for ``model_size``."""

    try:
        return GEOMETRIES[model_size.lower()]
    except KeyError as exc:
        raise ValueError(f"Unknown model size {model_size!r}; choose from {MODEL_SIZES}") from exc


def build_model_config(
    model_size: str,
    *,
    vocab_size: int = VOCAB_SIZE,
    num_experts: int = NUM_EXPERTS,
    top_k: int = TOP_K,
) -> OLMoDDPModelConfig:
    """Build one rung with no activation or block recomputation enabled."""

    g = geometry(model_size)
    norm = _norm()
    default_block = _moe_block(
        g,
        norm,
        _kda(g),
        num_experts=num_experts,
        top_k=top_k,
    )
    attention_block = _moe_block(
        g,
        norm,
        _attention(g, norm),
        num_experts=num_experts,
        top_k=top_k,
    )
    model = OLMoDDPModelConfig(
        name=TransformerType.moe_fused_v2,
        d_model=g.d_model,
        vocab_size=vocab_size,
        n_layers=g.n_layers,
        block=default_block,
        block_overrides={
            0: _dense_first(g, norm),
            **{index: deepcopy(attention_block) for index in g.full_attention_layers},
        },
        lm_head=LMHeadConfig(
            layer_norm=deepcopy(norm),
            bias=False,
            dtype=DType.float32,
        ),
        embedding_norm=deepcopy(norm),
        dtype=DType.float32,
        init_method="normal",
        init_seed=0,
        init_std=0.02,
        embed_scale=math.sqrt(g.d_model),
        tie_word_embeddings=False,
        two_batch_overlap=False,
        recompute_all_blocks_by_chunk=False,
        recompute_each_block=False,
    )
    model.validate()
    if model.recompute_all_blocks_by_chunk or model.recompute_each_block:
        raise ValueError("The deep-family candidates must not use block recomputation")
    if vocab_size == VOCAB_SIZE and num_experts == NUM_EXPERTS and top_k == TOP_K:
        actual = (
            model.num_active_params,
            model.num_active_non_embedding_params,
            model.num_params,
        )
        expected = (
            g.expected_active_params,
            g.expected_active_non_embedding_params,
            g.expected_total_params,
        )
        if actual != expected:
            raise ValueError(f"{model_size} parameter-count drift: {actual=} != {expected=}")
    return model
