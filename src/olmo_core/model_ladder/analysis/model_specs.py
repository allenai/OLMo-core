"""
Compute parameter counts and FLOPs for OLMo3, hybrid GatedDeltaNet-Transformer,
and hybrid/pure Mamba2 models.

Provides :class:`ModelSpec` definitions for each model size and functions to compute
exact parameter counts and forward-pass FLOP estimates. Used by both the standalone
``compute_hybrid_model_specs.py`` CLI and the Chinchilla scaling-law fitting pipeline.
"""

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

__all__ = [
    "ModelSpec",
    "OLMO3_SPECS",
    "OLMO3_SPECS_BY_NAME",
    "ensure_multiple_of",
    "gdn_head_dim",
    "gdn_dims",
    "count_gdn_params",
    "mamba2_dims",
    "count_mamba2_params",
    "count_attention_params",
    "count_mlp_params",
    "count_layernorm_params",
    "gdn_macs_per_token",
    "gdn_parallel_macs_per_token",
    "mamba2_macs_per_token",
    "mamba2_parallel_macs_per_token",
    "attention_macs_per_token",
    "mlp_macs_per_token",
    "compute_hybrid_specs",
    "compute_hybrid_specs_parallel",
    "compute_hybrid_mamba2_specs",
    "compute_hybrid_mamba2_specs_parallel",
    "compute_olmo3_specs",
    "compute_specs_for_size",
    "compute_specs_for_size_parallel",
    "fmt",
    "LADDER_ARCH_CONFIGS",
    "DISPLAY_NAMES",
    "get_display_name",
    "get_param_count",
]


def ensure_multiple_of(x: int, of: int) -> int:
    return of * math.ceil(x / of)


# ---------------------------------------------------------------------------
# OLMo3 model size specs (from TransformerConfig.olmo2_* / llama_like)
# ---------------------------------------------------------------------------
@dataclass
class ModelSpec:
    name: str
    d_model: int
    n_heads: int
    n_layers: int
    hidden_size_multiplier: float = 1.5
    hidden_size_multiple_of: int = 256
    vocab_size: int = 100352  # dolma2 tokenizer padded vocab

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def mlp_hidden_size(self) -> int:
        h = int(8 * self.d_model / 3)
        h = int(self.hidden_size_multiplier * h)
        return ensure_multiple_of(h, self.hidden_size_multiple_of)


OLMO3_SPECS = [
    ModelSpec("60M", d_model=384, n_heads=8, n_layers=8),
    ModelSpec("100M", d_model=512, n_heads=8, n_layers=12),
    ModelSpec("190M", d_model=768, n_heads=12, n_layers=12),
    ModelSpec("370M", d_model=1024, n_heads=16, n_layers=16),
    ModelSpec("600M", d_model=1280, n_heads=16, n_layers=16),
    ModelSpec("760M", d_model=1536, n_heads=16, n_layers=16),
    ModelSpec("1B", d_model=2048, n_heads=16, n_layers=16),
    ModelSpec("3B", d_model=3328, n_heads=16, n_layers=16),
    ModelSpec("7B", d_model=4096, n_heads=32, n_layers=32),
    ModelSpec("13B", d_model=5120, n_heads=40, n_layers=40),
]

OLMO3_SPECS_BY_NAME: Dict[str, ModelSpec] = {s.name: s for s in OLMO3_SPECS}


# ---------------------------------------------------------------------------
# GatedDeltaNet dimension helpers
# ---------------------------------------------------------------------------
def gdn_head_dim(d_model: int, n_heads: int) -> int:
    return ensure_multiple_of(int(0.75 * d_model / n_heads), 128)


def gdn_dims(d_model: int, n_heads: int, expand_v: float = 2.0) -> Tuple[int, int, int, int]:
    hd = gdn_head_dim(d_model, n_heads)
    head_k_dim = hd
    head_v_dim = int(hd * expand_v)
    key_dim = n_heads * head_k_dim
    value_dim = n_heads * head_v_dim
    return key_dim, value_dim, head_k_dim, head_v_dim


# ---------------------------------------------------------------------------
# Mamba2 dimension helpers
# ---------------------------------------------------------------------------
def mamba2_dims(
    d_model: int, n_heads: int, expand: int = 2, state_size: int = 128, n_groups: int = 1
) -> dict:
    head_dim = (expand * d_model) // n_heads
    intermediate_size = expand * d_model
    conv_dim = intermediate_size + 2 * n_groups * state_size
    projection_size = intermediate_size + conv_dim + n_heads
    return {
        "head_dim": head_dim,
        "intermediate_size": intermediate_size,
        "conv_dim": conv_dim,
        "projection_size": projection_size,
        "state_size": state_size,
        "n_groups": n_groups,
    }


# ---------------------------------------------------------------------------
# Parameter counting
# ---------------------------------------------------------------------------
def count_mamba2_params(
    d_model: int,
    n_heads: int,
    expand: int = 2,
    state_size: int = 128,
    n_groups: int = 1,
    conv_kernel: int = 4,
    use_bias: bool = True,
) -> int:
    dims = mamba2_dims(d_model, n_heads, expand, state_size, n_groups)
    intermediate_size = dims["intermediate_size"]
    conv_dim = dims["conv_dim"]
    projection_size = dims["projection_size"]

    params = 0
    params += d_model * projection_size
    if use_bias:
        params += projection_size
    params += conv_dim * conv_kernel
    params += n_heads * 3  # dt_bias, A_log, D
    params += intermediate_size  # RMSNorm
    params += intermediate_size * d_model
    if use_bias:
        params += d_model
    return params


def count_gdn_params(d_model: int, n_heads: int, conv_size: int = 4, use_gate: bool = True) -> int:
    key_dim, value_dim, head_k_dim, head_v_dim = gdn_dims(d_model, n_heads)
    num_v_heads = n_heads

    params = 0
    # Projections (q, k, v, a, b)
    params += d_model * (2 * key_dim + value_dim + 2 * num_v_heads)
    # Per-head params (A_log, dt_bias)
    params += 2 * num_v_heads
    # Short convs (q, k, v)
    params += (2 * key_dim + value_dim) * conv_size

    if use_gate:
        params += d_model * value_dim  # g_proj
        params += head_v_dim  # FusedRMSNormGated
    else:
        params += head_v_dim

    params += value_dim * d_model  # o_proj
    return params


def count_attention_params(d_model: int, n_heads: int, qk_norm: bool = True) -> int:
    head_dim = d_model // n_heads
    params = 4 * d_model * d_model  # Q, K, V, O
    if qk_norm:
        params += 2 * d_model
    return params


def count_mlp_params(d_model: int, mlp_hidden: int) -> int:
    return 3 * d_model * mlp_hidden


def count_layernorm_params(d_model: int) -> int:
    return d_model


# ---------------------------------------------------------------------------
# FLOP counting
# ---------------------------------------------------------------------------
def gdn_macs_per_token(
    d_model: int, n_heads: int, conv_size: int = 4, use_gate: bool = True
) -> int:
    """MACs per token for GatedDeltaNet (Recurrent Inference Mode)."""
    key_dim, value_dim, head_k_dim, head_v_dim = gdn_dims(d_model, n_heads)
    num_v_heads = n_heads

    macs = 0
    # Linear projections
    macs += d_model * (2 * key_dim + value_dim + 2 * num_v_heads)
    # Depthwise convolutions
    macs += (2 * key_dim + value_dim) * conv_size
    # Gate & Output
    if use_gate:
        macs += d_model * value_dim
    macs += value_dim * d_model

    # Recurrence: 3 interactions per head (Retain, Update, Output)
    # 3 * [d_k * d_v] per head
    macs += 3 * num_v_heads * head_k_dim * head_v_dim

    return macs


def mamba2_macs_per_token(
    d_model: int,
    n_heads: int,
    expand: int = 2,
    state_size: int = 128,
    n_groups: int = 1,
    conv_kernel: int = 4,
) -> int:
    """MACs per token for Mamba2 (Recurrent Inference Mode)."""
    dims = mamba2_dims(d_model, n_heads, expand, state_size, n_groups)
    intermediate_size = dims["intermediate_size"]
    conv_dim = dims["conv_dim"]
    projection_size = dims["projection_size"]
    head_dim = dims["head_dim"]

    macs = 0
    macs += d_model * projection_size
    macs += conv_dim * conv_kernel
    macs += intermediate_size * d_model
    # Recurrence: d_head * d_state per head
    macs += n_heads * head_dim * state_size
    return macs


def attention_macs_per_token(
    d_model: int, n_heads: int, seq_len: int, chinchilla_flops: bool = True
) -> int:
    head_dim = d_model // n_heads
    macs = 4 * d_model * d_model  # Projections

    effective_seq_len = seq_len // 2
    # Attention Score (QK^T) + Output (AV)
    macs += 2 * n_heads * head_dim * effective_seq_len

    if chinchilla_flops:
        # Softmax overhead
        macs += (5 * n_heads * effective_seq_len + 1) // 2

    return macs


def gdn_parallel_macs_per_token(
    d_model: int, n_heads: int, chunk_size: int = 256, conv_size: int = 4, use_gate: bool = True
) -> int:
    """
    MACs per token for GatedDeltaNet using the Extended WY Parallel Algorithm.

    References: Gated DeltaNet (arXiv:2412.06464v3), Equations 8-9 & 6-7.
    This algorithm avoids log-space tiling by folding decay into the WY representation,
    allowing full use of Tensor Cores.

    Components:
    1. Projections & Convolutions (same as recurrent).
    2. Intra-chunk Overhead (Extended WY Construction + Local Attention):
       - Kernel Matrix (KK^T): C * d_k
       - W Matrix (TK): C * d_k
       - U Matrix (TV): C * d_v
       - Local Attention (QK^T): C * d_k
       - Local Output (Score * (U-WS)): C * d_v
       -> Total Intra-chunk coefficient per token: 3*d_k + 2*d_v
    3. Inter-chunk State Passing:
       - 3 matrix ops of size (d_k * d_v) per head.
    """
    key_dim, value_dim, head_k_dim, head_v_dim = gdn_dims(d_model, n_heads)
    num_v_heads = n_heads

    macs = 0

    # 1. Sequence-Independent Operations (Projections + Convs)
    macs += d_model * (2 * key_dim + value_dim + 2 * num_v_heads)  # q,k,v,a,b proj
    macs += (2 * key_dim + value_dim) * conv_size  # q,k,v convs
    if use_gate:
        macs += d_model * value_dim  # g proj
    macs += value_dim * d_model  # o proj

    # 2. Intra-chunk Operations (scaled by chunk_size)
    # Coeff: 3*key_dim + 2*value_dim
    # Reflects: KK^T + W + QK^T (3*K) and U + Output (2*V)
    macs += chunk_size * (3 * key_dim + 2 * value_dim)

    # 3. Inter-chunk State Propagation (WY State Update)
    # 3 Ops: Querying state (QS^T), Updating State (U'^T K), Correction (WS^T)
    # Cost is 3 * d_k * d_v per head (amortized per token)
    macs += 3 * num_v_heads * head_k_dim * head_v_dim

    return macs


def mamba2_parallel_macs_per_token(
    d_model: int,
    n_heads: int,
    chunk_size: int = 256,
    expand: int = 2,
    state_size: int = 128,
    n_groups: int = 1,
    conv_kernel: int = 4,
) -> int:
    """
    MACs per token for Mamba2 using the SSD Parallel Algorithm.

    Includes intra-chunk semi-separable attention and inter-chunk state passing.
    """
    dims = mamba2_dims(d_model, n_heads, expand, state_size, n_groups)
    intermediate_size = dims["intermediate_size"]
    conv_dim = dims["conv_dim"]
    projection_size = dims["projection_size"]
    head_dim = dims["head_dim"]

    macs = 0
    # Projections
    macs += d_model * projection_size
    macs += conv_dim * conv_kernel
    macs += intermediate_size * d_model

    # Intra-chunk SSD: QK^T + Score*V
    # 2 * C * d_model (per token)
    macs += 2 * chunk_size * n_heads * head_dim

    # Inter-chunk State Passing (Read + Write)
    macs += 4 * n_heads * head_dim * state_size

    return macs


def mlp_macs_per_token(d_model: int, mlp_hidden: int) -> int:
    return 3 * d_model * mlp_hidden


# ---------------------------------------------------------------------------
# Main computation functions (mostly unchanged structure)
# ---------------------------------------------------------------------------
def compute_hybrid_specs(
    spec: ModelSpec,
    transformer_ratio: int = 4,
    seq_len: int = 8192,
    force_final_attention: bool = True,
    placement: str = "every_nth",
    chinchilla_flops: bool = True,
) -> dict:
    d = spec.d_model
    nh = spec.n_heads
    nl = spec.n_layers
    mlp_h = spec.mlp_hidden_size
    V = spec.vocab_size

    if transformer_ratio == 0:
        attn_indices = []
    elif placement == "middle":
        n_transformer_layers = nl // transformer_ratio
        start_idx = (nl - n_transformer_layers) // 2
        attn_indices = list(range(start_idx, start_idx + n_transformer_layers))
    else:
        attn_indices = [i for i in range(nl) if i % transformer_ratio == transformer_ratio - 1]

    if force_final_attention and nl - 1 not in attn_indices:
        attn_indices.append(nl - 1)
    attn_indices = sorted(set(attn_indices))
    n_attn = len(attn_indices)
    n_gdn = nl - n_attn

    embed_params = V * d
    head_params = V * d
    gdn_params = count_gdn_params(d, nh) + count_mlp_params(d, mlp_h) + 2 * d
    attn_params = count_attention_params(d, nh) + count_mlp_params(d, mlp_h) + 2 * d
    final_norm = d

    total_params = (
        embed_params + head_params + n_gdn * gdn_params + n_attn * attn_params + final_norm
    )
    non_embed_params = total_params - embed_params

    embed_macs = d * V if chinchilla_flops else 0
    head_macs = d * V
    gdn_layer_macs = gdn_macs_per_token(d, nh) + mlp_macs_per_token(d, mlp_h)
    attn_layer_macs = attention_macs_per_token(
        d, nh, seq_len, chinchilla_flops
    ) + mlp_macs_per_token(d, mlp_h)

    total_macs = embed_macs + head_macs + n_gdn * gdn_layer_macs + n_attn * attn_layer_macs

    return {
        "spec": spec,
        "n_attn": n_attn,
        "n_gdn": n_gdn,
        "attn_indices": attn_indices,
        "total_params": total_params,
        "non_embed_params": non_embed_params,
        "total_flops_per_token": 2 * total_macs,
        "total_macs_per_token": total_macs,
        "gdn_layer_macs": gdn_layer_macs,
        "attn_layer_macs": attn_layer_macs,
    }


def compute_hybrid_mamba2_specs(
    spec: ModelSpec,
    transformer_ratio: int = 4,
    seq_len: int = 8192,
    force_final_attention: bool = True,
    strip_mamba2_mlp: bool = False,
    chinchilla_flops: bool = True,
) -> dict:
    d = spec.d_model
    nh = spec.n_heads
    nl = spec.n_layers
    mlp_h = spec.mlp_hidden_size
    V = spec.vocab_size

    if transformer_ratio == 0:
        attn_indices = []
    else:
        attn_indices = [i for i in range(nl) if i % transformer_ratio == transformer_ratio - 1]

    if force_final_attention and nl - 1 not in attn_indices:
        attn_indices.append(nl - 1)
    attn_indices = sorted(set(attn_indices))
    n_attn = len(attn_indices)
    n_mamba = nl - n_attn

    embed_params = V * d
    head_params = V * d
    mamba_mixer = count_mamba2_params(d, nh)
    mamba_params = (
        mamba_mixer + d if strip_mamba2_mlp else mamba_mixer + count_mlp_params(d, mlp_h) + 2 * d
    )
    attn_params = count_attention_params(d, nh) + count_mlp_params(d, mlp_h) + 2 * d
    final_norm = d

    total_params = (
        embed_params + head_params + n_mamba * mamba_params + n_attn * attn_params + final_norm
    )
    non_embed_params = total_params - embed_params

    embed_macs = d * V if chinchilla_flops else 0
    head_macs = d * V
    mamba_layer_macs = mamba2_macs_per_token(d, nh)
    if not strip_mamba2_mlp:
        mamba_layer_macs += mlp_macs_per_token(d, mlp_h)
    attn_layer_macs = attention_macs_per_token(
        d, nh, seq_len, chinchilla_flops
    ) + mlp_macs_per_token(d, mlp_h)

    total_macs = embed_macs + head_macs + n_mamba * mamba_layer_macs + n_attn * attn_layer_macs

    return {
        "spec": spec,
        "n_attn": n_attn,
        "n_mamba2": n_mamba,
        "total_params": total_params,
        "non_embed_params": non_embed_params,
        "total_flops_per_token": 2 * total_macs,
        "total_macs_per_token": total_macs,
        "mamba2_layer_macs": mamba_layer_macs,
        "attn_layer_macs": attn_layer_macs,
    }


def compute_olmo3_specs(
    spec: ModelSpec, seq_len: int = 8192, chinchilla_flops: bool = True
) -> dict:
    d = spec.d_model
    nh = spec.n_heads
    nl = spec.n_layers
    mlp_h = spec.mlp_hidden_size
    V = spec.vocab_size

    embed_params = V * d
    head_params = V * d
    attn_params = count_attention_params(d, nh) + count_mlp_params(d, mlp_h) + 2 * d
    final_norm = d
    total_params = embed_params + head_params + nl * attn_params + final_norm
    non_embed_params = total_params - embed_params

    embed_macs = d * V if chinchilla_flops else 0
    head_macs = d * V
    attn_layer_macs = attention_macs_per_token(
        d, nh, seq_len, chinchilla_flops
    ) + mlp_macs_per_token(d, mlp_h)
    total_macs = embed_macs + head_macs + nl * attn_layer_macs

    return {
        "spec": spec,
        "n_attn": nl,
        "total_params": total_params,
        "non_embed_params": non_embed_params,
        "total_flops_per_token": 2 * total_macs,
    }


def compute_hybrid_specs_parallel(
    spec: ModelSpec,
    transformer_ratio: int = 4,
    seq_len: int = 8192,
    chunk_size: int = 256,
    force_final_attention: bool = True,
    placement: str = "every_nth",
    chinchilla_flops: bool = True,
) -> dict:
    result = compute_hybrid_specs(
        spec, transformer_ratio, seq_len, force_final_attention, placement, chinchilla_flops
    )
    d = spec.d_model
    nh = spec.n_heads
    mlp_h = spec.mlp_hidden_size
    V = spec.vocab_size

    # Recompute GDN MACs using Extended WY parallel algorithm
    gdn_layer_macs = gdn_parallel_macs_per_token(d, nh, chunk_size=chunk_size) + mlp_macs_per_token(
        d, mlp_h
    )

    embed_macs = d * V if chinchilla_flops else 0
    head_macs = d * V
    n_gdn = result["n_gdn"]
    n_attn = result["n_attn"]
    attn_layer_macs = result["attn_layer_macs"]

    total_macs = embed_macs + head_macs + n_gdn * gdn_layer_macs + n_attn * attn_layer_macs
    result["gdn_layer_macs"] = gdn_layer_macs
    result["total_macs_per_token"] = total_macs
    result["total_flops_per_token"] = 2 * total_macs
    result["chunk_size"] = chunk_size
    return result


def compute_hybrid_mamba2_specs_parallel(
    spec: ModelSpec,
    transformer_ratio: int = 4,
    seq_len: int = 8192,
    chunk_size: int = 256,
    force_final_attention: bool = True,
    strip_mamba2_mlp: bool = False,
    chinchilla_flops: bool = True,
) -> dict:
    result = compute_hybrid_mamba2_specs(
        spec, transformer_ratio, seq_len, force_final_attention, strip_mamba2_mlp, chinchilla_flops
    )
    d = spec.d_model
    nh = spec.n_heads
    mlp_h = spec.mlp_hidden_size
    V = spec.vocab_size

    # Recompute Mamba2 MACs using SSD parallel algorithm
    mamba2_layer_macs = mamba2_parallel_macs_per_token(d, nh, chunk_size=chunk_size)
    if not strip_mamba2_mlp:
        mamba2_layer_macs += mlp_macs_per_token(d, mlp_h)

    embed_macs = d * V if chinchilla_flops else 0
    head_macs = d * V
    n_mamba = result["n_mamba2"]
    n_attn = result["n_attn"]
    attn_layer_macs = result["attn_layer_macs"]

    total_macs = embed_macs + head_macs + n_mamba * mamba2_layer_macs + n_attn * attn_layer_macs
    result["mamba2_layer_macs"] = mamba2_layer_macs
    result["total_macs_per_token"] = total_macs
    result["total_flops_per_token"] = 2 * total_macs
    result["chunk_size"] = chunk_size
    return result


def compute_specs_for_size(
    ladder_name: str,
    size: str,
    seq_len: Optional[int] = None,
    chinchilla_flops: bool = True,
) -> Optional[dict]:
    config = _get_arch_config(ladder_name)
    if config is None:
        return None
    spec = OLMO3_SPECS_BY_NAME.get(size)
    if spec is None:
        return None
    effective_seq_len = seq_len or config.seq_len

    if config.is_transformer:
        return compute_olmo3_specs(spec, effective_seq_len, chinchilla_flops)
    elif config.layer_type == "mamba2":
        return compute_hybrid_mamba2_specs(
            spec,
            config.transformer_ratio,
            effective_seq_len,
            config.force_final_attention,
            chinchilla_flops=chinchilla_flops,
        )
    else:
        return compute_hybrid_specs(
            spec,
            config.transformer_ratio,
            effective_seq_len,
            config.force_final_attention,
            config.placement,
            chinchilla_flops,
        )


def compute_specs_for_size_parallel(
    ladder_name: str,
    size: str,
    seq_len: Optional[int] = None,
    chunk_size: int = 256,
    chinchilla_flops: bool = True,
) -> Optional[dict]:
    config = _get_arch_config(ladder_name)
    if config is None:
        return None
    spec = OLMO3_SPECS_BY_NAME.get(size)
    if spec is None:
        return None
    effective_seq_len = seq_len or config.seq_len

    if config.is_transformer:
        return compute_olmo3_specs(spec, effective_seq_len, chinchilla_flops)
    elif config.layer_type == "mamba2":
        return compute_hybrid_mamba2_specs_parallel(
            spec,
            config.transformer_ratio,
            effective_seq_len,
            chunk_size,
            config.force_final_attention,
            chinchilla_flops=chinchilla_flops,
        )
    else:
        return compute_hybrid_specs_parallel(
            spec,
            config.transformer_ratio,
            effective_seq_len,
            chunk_size,
            config.force_final_attention,
            config.placement,
            chinchilla_flops,
        )


# Helper config classes (LADDER_ARCH_CONFIGS, etc) remain unchanged...
@dataclass
class LadderArchConfig:
    transformer_ratio: int
    is_transformer: bool = False
    force_final_attention: bool = True
    placement: str = "every_nth"
    seq_len: int = 8192
    layer_type: str = "gdn"


LADDER_ARCH_CONFIGS: Dict[str, LadderArchConfig] = {
    "olmo3": LadderArchConfig(0, True),
    "olmo3-1": LadderArchConfig(0, True),
    "olmo3-2": LadderArchConfig(0, True),
    "olmo3-3": LadderArchConfig(0, True),
    "hybrid-gdn": LadderArchConfig(4),
    "hybrid-gdn-half": LadderArchConfig(2),
    "hybrid-gdn-eight": LadderArchConfig(8),
    "hybrid-gdn-middle": LadderArchConfig(4, placement="middle"),
    "pure-gdn": LadderArchConfig(0, force_final_attention=False),
    "hybrid-mamba": LadderArchConfig(4, layer_type="mamba2"),
    "pure-mamba": LadderArchConfig(0, force_final_attention=False, layer_type="mamba2"),
}


def _get_arch_config(ladder_name: str) -> Optional[LadderArchConfig]:
    lower = ladder_name.lower()
    if lower in LADDER_ARCH_CONFIGS:
        return LADDER_ARCH_CONFIGS[lower]
    for p, c in LADDER_ARCH_CONFIGS.items():
        if p in lower:
            return c
    return None


DISPLAY_NAMES: Dict[str, str] = {
    "olmo3": "Olmo 3",
    "olmo3-1": "Olmo 3 v1",
    "olmo3-2": "Olmo 3 v2",
    "olmo3-3": "Olmo 3 v3",
    "pure-gdn": "Pure GDN",
    "hybrid-gdn": "Hybrid GDN (1:3)",
    "hybrid-gdn-half": "Hybrid GDN (1:1)",
    "hybrid-gdn-eight": "Hybrid GDN (1:7)",
    "hybrid-gdn-middle": "Hybrid GDN (Middle)",
    "pure-mamba": "Pure Mamba",
    "hybrid-mamba": "Hybrid Mamba",
}


def fmt(n: float) -> str:
    if abs(n) >= 1e9:
        return f"{n/1e9:.2f}B"
    elif abs(n) >= 1e6:
        return f"{n/1e6:.2f}M"
    elif abs(n) >= 1e3:
        return f"{n/1e3:.1f}K"
    else:
        return f"{n:.0f}"


def get_display_name(ladder_name: str) -> str:
    return DISPLAY_NAMES.get(ladder_name, ladder_name)


def get_param_count(ladder_name: str, size: str) -> Optional[int]:
    specs = compute_specs_for_size(ladder_name, size)
    return specs["non_embed_params"] if specs else None
