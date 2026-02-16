"""
Compute parameter counts and FLOPs for OLMo3, hybrid GatedDeltaNet-Transformer,
and hybrid/pure Mamba2 models.

Provides :class:`ModelSpec` definitions for each model size and functions to compute
exact parameter counts and forward-pass FLOP estimates.  Used by both the standalone
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
    ModelSpec("600M", d_model=1280, n_heads=16, n_layers=16),  # olmo3 overrides d_model to 1280
    ModelSpec("760M", d_model=1536, n_heads=16, n_layers=16),
    ModelSpec(
        "1B", d_model=2048, n_heads=16, n_layers=16
    ),  # olmo3_1B -> olmo2_1B_v2 -> llama2_1B with n_layers=16
    ModelSpec("3B", d_model=3328, n_heads=16, n_layers=16),
    ModelSpec("7B", d_model=4096, n_heads=32, n_layers=32),
    ModelSpec("13B", d_model=5120, n_heads=40, n_layers=40),
]

OLMO3_SPECS_BY_NAME: Dict[str, ModelSpec] = {s.name: s for s in OLMO3_SPECS}


# ---------------------------------------------------------------------------
# GatedDeltaNet dimension helpers
# ---------------------------------------------------------------------------
def gdn_head_dim(d_model: int, n_heads: int) -> int:
    """
    Matches: ensure_multiple_of(int(0.75 * d_model / n_heads), 128)
    from hybrid-gdn-ladder.py configure_model().
    """
    return ensure_multiple_of(int(0.75 * d_model / n_heads), 128)


def gdn_dims(d_model: int, n_heads: int, expand_v: float = 2.0) -> Tuple[int, int, int, int]:
    """Return GatedDeltaNet projection dimensions (with use_gate=True)."""
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
    """
    Compute Mamba2 intermediate dimensions.

    Matches FLAConfig.build() for name="Mamba2":
        head_dim = (expand * d_model) // n_heads
    And Mamba2.__init__ from fla/layers/mamba2.py.
    """
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
    """
    Count parameters for one Mamba2 layer.
    Mirrors Mamba2.__init__ in fla/layers/mamba2.py.
    """
    dims = mamba2_dims(d_model, n_heads, expand, state_size, n_groups)
    intermediate_size = dims["intermediate_size"]
    conv_dim = dims["conv_dim"]
    projection_size = dims["projection_size"]

    params = 0

    # in_proj: Linear(hidden_size -> projection_size, bias=use_bias)
    params += d_model * projection_size
    if use_bias:
        params += projection_size

    # conv1d: depthwise Conv1d(conv_dim, conv_dim, kernel, groups=conv_dim)
    # weight: (conv_dim, 1, kernel), no bias by default (use_conv_bias=False)
    params += conv_dim * conv_kernel

    # Per-head parameters
    params += n_heads  # dt_bias
    params += n_heads  # A_log
    params += n_heads  # D

    # RMSNormGated(intermediate_size) -> weight of size intermediate_size
    params += intermediate_size

    # out_proj: Linear(intermediate_size -> hidden_size, bias=use_bias)
    params += intermediate_size * d_model
    if use_bias:
        params += d_model

    return params


def count_gdn_params(d_model: int, n_heads: int, conv_size: int = 4, use_gate: bool = True) -> int:
    """
    Count parameters for one GatedDeltaNet layer (use_gate=True, use_short_conv=True).
    Mirrors GatedDeltaNet.__init__ in gated_deltanet.py.
    """
    key_dim, value_dim, head_k_dim, head_v_dim = gdn_dims(d_model, n_heads)
    num_v_heads = n_heads  # default

    params = 0

    # Projections (no bias)
    params += d_model * key_dim  # q_proj
    params += d_model * key_dim  # k_proj
    params += d_model * value_dim  # v_proj
    params += d_model * num_v_heads  # a_proj
    params += d_model * num_v_heads  # b_proj

    # Small per-head params
    params += num_v_heads  # A_log
    params += num_v_heads  # dt_bias

    # Short convolutions (depthwise, no bias by default)
    # ShortConvolution: groups=hidden_size, kernel_size=conv_size
    # weight shape: (hidden_size, 1, kernel_size) -> hidden_size * kernel_size params
    params += key_dim * conv_size  # q_conv1d
    params += key_dim * conv_size  # k_conv1d
    params += value_dim * conv_size  # v_conv1d

    if use_gate:
        params += d_model * value_dim  # g_proj
        # FusedRMSNormGated(head_v_dim) -> head_v_dim params
        params += head_v_dim
    else:
        # RMSNorm(head_v_dim) -> head_v_dim params
        params += head_v_dim

    params += value_dim * d_model  # o_proj

    return params


def count_attention_params(d_model: int, n_heads: int, qk_norm: bool = True) -> int:
    """
    Count parameters for one attention sub-layer (no bias, with optional qk_norm).
    n_kv_heads = n_heads (no GQA), head_dim = d_model // n_heads.
    """
    head_dim = d_model // n_heads
    params = 0

    # Q, K, V, O projections (no bias)
    params += d_model * n_heads * head_dim  # Q
    params += d_model * n_heads * head_dim  # K
    params += d_model * n_heads * head_dim  # V
    params += n_heads * head_dim * d_model  # O
    # Simplifies to 4 * d_model^2

    # QK norm (RMSNorm, no bias): q_norm + k_norm, each over (n_heads * head_dim) = d_model
    if qk_norm:
        params += d_model  # q_norm
        params += d_model  # k_norm

    return params


def count_mlp_params(d_model: int, mlp_hidden: int) -> int:
    """SwiGLU MLP: w1 (gate), w2 (up), w3 (down), no bias. 3 * d_model * hidden."""
    return 3 * d_model * mlp_hidden


def count_layernorm_params(d_model: int) -> int:
    """RMSNorm: just the weight vector (no bias)."""
    return d_model


# ---------------------------------------------------------------------------
# FLOP counting (forward pass multiply-accumulate operations)
# ---------------------------------------------------------------------------
def gdn_macs_per_token(
    d_model: int, n_heads: int, conv_size: int = 4, use_gate: bool = True
) -> int:
    """
    MACs per token for GatedDeltaNet projections + convolutions.
    The recurrence itself is O(1) per token per head: ~num_v_heads * head_k_dim * head_v_dim.
    """
    key_dim, value_dim, head_k_dim, head_v_dim = gdn_dims(d_model, n_heads)
    num_v_heads = n_heads

    macs = 0

    # Linear projections
    macs += d_model * key_dim  # q_proj
    macs += d_model * key_dim  # k_proj
    macs += d_model * value_dim  # v_proj
    macs += d_model * num_v_heads  # a_proj
    macs += d_model * num_v_heads  # b_proj

    # Depthwise convolutions: kernel_size MACs per channel per token
    macs += key_dim * conv_size  # q_conv1d
    macs += key_dim * conv_size  # k_conv1d
    macs += value_dim * conv_size  # v_conv1d

    if use_gate:
        macs += d_model * value_dim  # g_proj

    macs += value_dim * d_model  # o_proj

    # Recurrence: per-token state read (query) + write (update), each ~d_k * d_v per head
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
    """
    MACs per token for Mamba2 layer (forward pass).

    Includes:
    - in_proj and out_proj linear projections
    - Depthwise conv1d
    - SSM recurrence: per-token state update ~num_heads * head_dim * state_size
    """
    dims = mamba2_dims(d_model, n_heads, expand, state_size, n_groups)
    intermediate_size = dims["intermediate_size"]
    conv_dim = dims["conv_dim"]
    projection_size = dims["projection_size"]
    head_dim = dims["head_dim"]

    macs = 0

    # in_proj: d_model -> projection_size
    macs += d_model * projection_size

    # depthwise conv1d: conv_dim channels, kernel_size MACs per channel per token
    macs += conv_dim * conv_kernel

    # out_proj: intermediate_size -> d_model
    macs += intermediate_size * d_model

    # SSM recurrence: per-token state read (query) + write (update) per head
    # Each head: head_dim * state_size for each of read and write
    macs += 2 * n_heads * head_dim * state_size

    return macs


def attention_macs_per_token(
    d_model: int, n_heads: int, seq_len: int, chinchilla_flops: bool = False
) -> int:
    """
    MACs per token for attention (forward pass).
    Projections are O(1) per token; QK^T and attn*V are O(seq_len) per token.

    If *chinchilla_flops* is True, includes softmax ops (3 non-MAC ops per score).
    """
    head_dim = d_model // n_heads
    macs = 0

    # Q, K, V, O projections: 4 * d_model^2
    macs += 4 * d_model * d_model

    # Halved to account for causal attention
    effective_seq_len = seq_len // 2

    # QK^T: each token attends to seq_len keys -> n_heads * head_dim * seq_len
    macs += n_heads * head_dim * effective_seq_len

    if chinchilla_flops:
        # Softmax: 3 non-MAC ops per score (subtract max, exp, divide by sum).
        # Chinchilla counts 3 * num_heads * seq_len * seq_len FLOPs (no 2× factor).
        # Since we convert MACs -> FLOPs via 2× later, store as 1.5 to get 3 after doubling.
        macs += (3 * n_heads * effective_seq_len + 1) // 2  # ceil(3/2) to stay integer

    # attn * V: n_heads * seq_len * head_dim
    macs += n_heads * effective_seq_len * head_dim

    return macs


def gdn_parallel_macs_per_token(
    d_model: int, n_heads: int, chunk_size: int = 256, conv_size: int = 4, use_gate: bool = True
) -> int:
    """
    MACs per token for GatedDeltaNet using chunkwise parallel training algorithm.

    Compared to the recurrent formulation, training uses the WY representation
    (arXiv:2412.06464v3, Equations 8-10) which adds:
    - Intra-chunk local attention: O(chunk_size) per token
    - Inter-chunk state propagation via WY form: ~3x the per-head state cost
    """
    key_dim, value_dim, head_k_dim, head_v_dim = gdn_dims(d_model, n_heads)
    num_v_heads = n_heads

    macs = 0

    # Linear projections (same as recurrent)
    macs += d_model * key_dim  # q_proj
    macs += d_model * key_dim  # k_proj
    macs += d_model * value_dim  # v_proj
    macs += d_model * num_v_heads  # a_proj
    macs += d_model * num_v_heads  # b_proj

    # Depthwise convolutions (same as recurrent)
    macs += key_dim * conv_size  # q_conv1d
    macs += key_dim * conv_size  # k_conv1d
    macs += value_dim * conv_size  # v_conv1d

    if use_gate:
        macs += d_model * value_dim  # g_proj

    macs += value_dim * d_model  # o_proj

    # # Intra-chunk: local attention within each chunk of size C
    # # Q@K^T (C x d_k @ d_k x C -> C x C) and scores@V (C x C @ C x d_v -> C x d_v)
    # # Per token (amortized): C * d_k + C * d_v = C * (d_k + d_v)
    # macs += chunk_size * (key_dim + value_dim)

    # GDN Intra-chunk involves:
    # 1. Kernel Matrix K*K^T (C*d_k)
    # 2. W & U construction (roughly 2 * C * (d_k + d_v))
    # 3. Attention Q*K^T (C*d_k)
    # 4. Output Score*(U-WS) (C*d_v)
    # Conservative estimate based on paper Eq 8-9:
    macs += chunk_size * (2 * key_dim + 2 * value_dim + key_dim)

    # Inter-chunk: state propagation using WY representation
    # Three matrix multiplications involving the (d_k x d_v) state per head per chunk,
    # amortized to per-token cost
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
    MACs per token for Mamba2 using SSD parallel training algorithm.

    Compared to the recurrent formulation, the SSD algorithm adds intra-chunk
    semi-separable matmuls that scale with chunk_size.
    """
    dims = mamba2_dims(d_model, n_heads, expand, state_size, n_groups)
    intermediate_size = dims["intermediate_size"]
    conv_dim = dims["conv_dim"]
    projection_size = dims["projection_size"]
    head_dim = dims["head_dim"]

    macs = 0

    # Projections (same as recurrent)
    macs += d_model * projection_size  # in_proj
    macs += conv_dim * conv_kernel  # depthwise conv1d
    macs += intermediate_size * d_model  # out_proj

    # Intra-chunk: semi-separable attention within each chunk
    # Per token (amortized): C * n_heads * head_dim
    macs += 2 * chunk_size * n_heads * head_dim

    # Inter-chunk: blockwise state propagation (read + write)
    macs += 4 * n_heads * head_dim * state_size

    return macs


def mlp_macs_per_token(d_model: int, mlp_hidden: int) -> int:
    """MACs for SwiGLU MLP: 3 matmuls of d_model x hidden."""
    return 3 * d_model * mlp_hidden


# ---------------------------------------------------------------------------
# Main computation
# ---------------------------------------------------------------------------
def compute_hybrid_specs(
    spec: ModelSpec,
    transformer_ratio: int = 4,
    seq_len: int = 4096,
    force_final_attention: bool = True,
    placement: str = "every_nth",
    chinchilla_flops: bool = False,
) -> dict:
    """Compute params and FLOPs for a hybrid GDN model.

    Args:
        spec: Model architecture specification.
        transformer_ratio: Ratio of transformer layers (e.g. 4 = every 4th layer is attention).
            Set to 0 for pure-GDN.
        seq_len: Sequence length for FLOP calculation.
        force_final_attention: If True, the final layer is always attention.
            Set to False for pure-GDN models.
        placement: How to place attention layers.
            ``"every_nth"``: attention at every *transformer_ratio*-th layer (default).
            ``"middle"``: attention layers are placed in a centered block.
        chinchilla_flops: If True, include embedding and softmax FLOPs
            following the Chinchilla accounting convention.
    """
    d = spec.d_model
    nh = spec.n_heads
    nl = spec.n_layers
    mlp_h = spec.mlp_hidden_size
    V = spec.vocab_size

    # Determine which layers are attention
    if transformer_ratio == 0:
        attn_indices: list = []
    elif placement == "middle":
        # Centered block (same logic as hybrid-gdn-middle-ladder.py)
        n_transformer_layers = nl // transformer_ratio
        start_idx = (nl - n_transformer_layers) // 2
        end_idx = start_idx + n_transformer_layers
        attn_indices = list(range(start_idx, end_idx))
    else:
        # Every-Nth (same logic as hybrid-gdn-ladder.py)
        attn_indices = [i for i in range(nl) if i % transformer_ratio == transformer_ratio - 1]

    if force_final_attention and nl - 1 not in attn_indices:
        attn_indices.append(nl - 1)
    attn_indices = sorted(set(attn_indices))
    n_attn = len(attn_indices)
    n_gdn = nl - n_attn

    # --- Parameters ---
    # Embeddings
    embed_params = V * d
    head_params = V * d  # untied

    # Per-layer
    gdn_layer_params = (
        count_gdn_params(d, nh) + count_mlp_params(d, mlp_h) + 2 * count_layernorm_params(d)
    )
    attn_layer_params = (
        count_attention_params(d, nh) + count_mlp_params(d, mlp_h) + 2 * count_layernorm_params(d)
    )

    # Final layer norm (before LM head)
    final_norm_params = count_layernorm_params(d)

    total_params = (
        embed_params
        + head_params
        + n_gdn * gdn_layer_params
        + n_attn * attn_layer_params
        + final_norm_params
    )
    non_embed_params = total_params - embed_params

    # --- FLOPs (using 2 * MACs convention) ---
    embed_macs = d * V if chinchilla_flops else 0  # embedding lookup (counted by Chinchilla)
    head_macs = d * V
    gdn_layer_macs = gdn_macs_per_token(d, nh) + mlp_macs_per_token(d, mlp_h)
    attn_layer_macs = attention_macs_per_token(
        d, nh, seq_len, chinchilla_flops=chinchilla_flops
    ) + mlp_macs_per_token(d, mlp_h)

    total_macs = embed_macs + head_macs + n_gdn * gdn_layer_macs + n_attn * attn_layer_macs
    total_flops = 2 * total_macs

    return {
        "spec": spec,
        "n_attn": n_attn,
        "n_gdn": n_gdn,
        "attn_indices": attn_indices,
        "embed_params": embed_params,
        "head_params": head_params,
        "gdn_layer_params": gdn_layer_params,
        "attn_layer_params": attn_layer_params,
        "final_norm_params": final_norm_params,
        "total_params": total_params,
        "non_embed_params": non_embed_params,
        "total_flops_per_token": total_flops,
        "total_macs_per_token": total_macs,
        "gdn_layer_macs": gdn_layer_macs,
        "attn_layer_macs": attn_layer_macs,
        "head_macs": head_macs,
        "seq_len": seq_len,
    }


def compute_hybrid_mamba2_specs(
    spec: ModelSpec,
    transformer_ratio: int = 4,
    seq_len: int = 4096,
    force_final_attention: bool = True,
    strip_mamba2_mlp: bool = False,
    chinchilla_flops: bool = False,
) -> dict:
    """Compute params and FLOPs for a hybrid Mamba2-Transformer model.

    Args:
        spec: Model architecture specification.
        transformer_ratio: Ratio of transformer layers (e.g. 4 = every 4th layer is attention).
            Set to 0 for pure-Mamba2.
        seq_len: Sequence length for FLOP calculation.
        force_final_attention: If True, the final layer is always attention.
            Set to False for pure-Mamba2 models.
        strip_mamba2_mlp: If True, Mamba2 layers have no MLP.
            Default is False: Mamba2 layers include FFNs between layers,
            the same way as GDN and Transformer layers.
        chinchilla_flops: If True, include embedding and softmax FLOPs
            following the Chinchilla accounting convention.
    """
    d = spec.d_model
    nh = spec.n_heads
    nl = spec.n_layers
    mlp_h = spec.mlp_hidden_size
    V = spec.vocab_size

    # Determine which layers are attention
    if transformer_ratio == 0:
        attn_indices: list = []
    else:
        attn_indices = [i for i in range(nl) if i % transformer_ratio == transformer_ratio - 1]

    if force_final_attention and nl - 1 not in attn_indices:
        attn_indices.append(nl - 1)
    attn_indices = sorted(set(attn_indices))
    n_attn = len(attn_indices)
    n_mamba2 = nl - n_attn

    # --- Parameters ---
    embed_params = V * d
    head_params = V * d  # untied

    # Mamba2 layer: Mamba2 mixer + (optional MLP) + layer norms
    mamba2_mixer_params = count_mamba2_params(d, nh)
    if strip_mamba2_mlp:
        # Only 1 layer norm (for the Mamba2 mixer, no MLP norm)
        mamba2_layer_params = mamba2_mixer_params + count_layernorm_params(d)
    else:
        mamba2_layer_params = (
            mamba2_mixer_params + count_mlp_params(d, mlp_h) + 2 * count_layernorm_params(d)
        )

    # Attention layer: attention + MLP + 2 layer norms
    attn_layer_params = (
        count_attention_params(d, nh) + count_mlp_params(d, mlp_h) + 2 * count_layernorm_params(d)
    )

    final_norm_params = count_layernorm_params(d)

    total_params = (
        embed_params
        + head_params
        + n_mamba2 * mamba2_layer_params
        + n_attn * attn_layer_params
        + final_norm_params
    )
    non_embed_params = total_params - embed_params

    # --- FLOPs (using 2 * MACs convention) ---
    embed_macs = d * V if chinchilla_flops else 0  # embedding lookup (counted by Chinchilla)
    head_macs = d * V
    mamba2_layer_macs = mamba2_macs_per_token(d, nh)
    if not strip_mamba2_mlp:
        mamba2_layer_macs += mlp_macs_per_token(d, mlp_h)
    attn_layer_macs = attention_macs_per_token(
        d, nh, seq_len, chinchilla_flops=chinchilla_flops
    ) + mlp_macs_per_token(d, mlp_h)

    total_macs = embed_macs + head_macs + n_mamba2 * mamba2_layer_macs + n_attn * attn_layer_macs
    total_flops = 2 * total_macs

    return {
        "spec": spec,
        "n_attn": n_attn,
        "n_mamba2": n_mamba2,
        "n_gdn": 0,
        "attn_indices": attn_indices,
        "embed_params": embed_params,
        "head_params": head_params,
        "mamba2_layer_params": mamba2_layer_params,
        "attn_layer_params": attn_layer_params,
        "final_norm_params": final_norm_params,
        "total_params": total_params,
        "non_embed_params": non_embed_params,
        "total_flops_per_token": total_flops,
        "total_macs_per_token": total_macs,
        "mamba2_layer_macs": mamba2_layer_macs,
        "attn_layer_macs": attn_layer_macs,
        "head_macs": head_macs,
        "seq_len": seq_len,
    }


def compute_olmo3_specs(
    spec: ModelSpec, seq_len: int = 4096, chinchilla_flops: bool = False
) -> dict:
    """Compute params and FLOPs for a pure OLMo3 (all-attention) model."""
    d = spec.d_model
    nh = spec.n_heads
    nl = spec.n_layers
    mlp_h = spec.mlp_hidden_size
    V = spec.vocab_size

    embed_params = V * d
    head_params = V * d
    attn_layer_params = (
        count_attention_params(d, nh) + count_mlp_params(d, mlp_h) + 2 * count_layernorm_params(d)
    )
    final_norm_params = count_layernorm_params(d)
    total_params = embed_params + head_params + nl * attn_layer_params + final_norm_params
    non_embed_params = total_params - embed_params

    embed_macs = d * V if chinchilla_flops else 0  # embedding lookup (counted by Chinchilla)
    head_macs = d * V
    attn_layer_macs = attention_macs_per_token(
        d, nh, seq_len, chinchilla_flops=chinchilla_flops
    ) + mlp_macs_per_token(d, mlp_h)
    total_macs = embed_macs + head_macs + nl * attn_layer_macs
    total_flops = 2 * total_macs

    return {
        "spec": spec,
        "n_attn": nl,
        "n_gdn": 0,
        "total_params": total_params,
        "non_embed_params": non_embed_params,
        "total_flops_per_token": total_flops,
        "seq_len": seq_len,
    }


def compute_hybrid_specs_parallel(
    spec: ModelSpec,
    transformer_ratio: int = 4,
    seq_len: int = 4096,
    chunk_size: int = 256,
    force_final_attention: bool = True,
    placement: str = "every_nth",
    chinchilla_flops: bool = False,
) -> dict:
    """Compute params and parallel-training FLOPs for a hybrid GDN model.

    Same as :func:`compute_hybrid_specs` but uses chunkwise parallel training
    FLOP estimates (intra-chunk attention + WY inter-chunk overhead) instead of
    recurrent inference FLOPs for GDN layers.
    """
    result = compute_hybrid_specs(
        spec,
        transformer_ratio=transformer_ratio,
        seq_len=seq_len,
        force_final_attention=force_final_attention,
        placement=placement,
        chinchilla_flops=chinchilla_flops,
    )

    d = spec.d_model
    nh = spec.n_heads
    mlp_h = spec.mlp_hidden_size
    V = spec.vocab_size

    # Recompute GDN layer MACs using parallel algorithm
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
    seq_len: int = 4096,
    chunk_size: int = 256,
    force_final_attention: bool = True,
    strip_mamba2_mlp: bool = False,
    chinchilla_flops: bool = False,
) -> dict:
    """Compute params and parallel-training FLOPs for a hybrid Mamba2-Transformer model.

    Same as :func:`compute_hybrid_mamba2_specs` but uses SSD parallel training
    FLOP estimates instead of recurrent inference FLOPs for Mamba2 layers.
    """
    result = compute_hybrid_mamba2_specs(
        spec,
        transformer_ratio=transformer_ratio,
        seq_len=seq_len,
        force_final_attention=force_final_attention,
        strip_mamba2_mlp=strip_mamba2_mlp,
        chinchilla_flops=chinchilla_flops,
    )

    d = spec.d_model
    nh = spec.n_heads
    mlp_h = spec.mlp_hidden_size
    V = spec.vocab_size

    # Recompute Mamba2 layer MACs using parallel algorithm
    mamba2_layer_macs = mamba2_parallel_macs_per_token(d, nh, chunk_size=chunk_size)
    if not strip_mamba2_mlp:
        mamba2_layer_macs += mlp_macs_per_token(d, mlp_h)

    embed_macs = d * V if chinchilla_flops else 0
    head_macs = d * V
    n_mamba2 = result["n_mamba2"]
    n_attn = result["n_attn"]
    attn_layer_macs = result["attn_layer_macs"]

    total_macs = embed_macs + head_macs + n_mamba2 * mamba2_layer_macs + n_attn * attn_layer_macs

    result["mamba2_layer_macs"] = mamba2_layer_macs
    result["total_macs_per_token"] = total_macs
    result["total_flops_per_token"] = 2 * total_macs
    result["chunk_size"] = chunk_size

    return result


def compute_specs_for_size_parallel(
    ladder_name: str,
    size: str,
    seq_len: Optional[int] = None,
    chunk_size: int = 256,
    chinchilla_flops: bool = False,
) -> Optional[dict]:
    """
    Compute architecture specs with parallel-training FLOPs for a given ladder name and size.

    Like :func:`compute_specs_for_size` but uses chunkwise parallel FLOP estimates
    for GDN/Mamba2 layers. Pure transformer models are unaffected.
    """
    config = _get_arch_config(ladder_name)
    if config is None:
        return None

    spec = OLMO3_SPECS_BY_NAME.get(size)
    if spec is None:
        return None

    effective_seq_len = seq_len or config.seq_len

    if config.is_transformer:
        return compute_olmo3_specs(
            spec, seq_len=effective_seq_len, chinchilla_flops=chinchilla_flops
        )
    elif config.layer_type == "mamba2":
        return compute_hybrid_mamba2_specs_parallel(
            spec,
            transformer_ratio=config.transformer_ratio,
            seq_len=effective_seq_len,
            chunk_size=chunk_size,
            force_final_attention=config.force_final_attention,
            chinchilla_flops=chinchilla_flops,
        )
    else:
        return compute_hybrid_specs_parallel(
            spec,
            transformer_ratio=config.transformer_ratio,
            seq_len=effective_seq_len,
            chunk_size=chunk_size,
            force_final_attention=config.force_final_attention,
            placement=config.placement,
            chinchilla_flops=chinchilla_flops,
        )


def fmt(n: float) -> str:
    if abs(n) >= 1e9:
        return f"{n/1e9:.2f}B"
    elif abs(n) >= 1e6:
        return f"{n/1e6:.2f}M"
    elif abs(n) >= 1e3:
        return f"{n/1e3:.1f}K"
    else:
        return f"{n:.0f}"


# ---------------------------------------------------------------------------
# Ladder name → architecture config mapping
# ---------------------------------------------------------------------------
@dataclass
class LadderArchConfig:
    """Architecture configuration for a ladder type."""

    transformer_ratio: int  # 0 for pure-GDN/Mamba2, n for every-nth attention
    is_transformer: bool = False  # True for pure transformer (OLMo3)
    force_final_attention: bool = True  # Hybrids force final attn; pure models do not
    placement: str = "every_nth"  # "every_nth" or "middle"
    seq_len: int = 4096
    layer_type: str = "gdn"  # "gdn" or "mamba2"


LADDER_ARCH_CONFIGS: Dict[str, LadderArchConfig] = {
    # Transformer models
    "olmo3": LadderArchConfig(transformer_ratio=0, is_transformer=True),
    "olmo3-1": LadderArchConfig(transformer_ratio=0, is_transformer=True),
    "olmo3-2": LadderArchConfig(transformer_ratio=0, is_transformer=True),
    "olmo3-3": LadderArchConfig(transformer_ratio=0, is_transformer=True),
    # Hybrid GDN models
    "hybrid-gdn": LadderArchConfig(transformer_ratio=4),
    "hybrid-gdn-half": LadderArchConfig(transformer_ratio=2),
    "hybrid-gdn-eight": LadderArchConfig(transformer_ratio=8),
    "hybrid-gdn-middle": LadderArchConfig(transformer_ratio=4, placement="middle"),
    # Pure GDN
    "pure-gdn": LadderArchConfig(transformer_ratio=0, force_final_attention=False),
    # Hybrid Mamba2 models
    "hybrid-mamba": LadderArchConfig(transformer_ratio=4, layer_type="mamba2"),
    # Pure Mamba2
    "pure-mamba": LadderArchConfig(
        transformer_ratio=0, force_final_attention=False, layer_type="mamba2"
    ),
}


def _get_arch_config(ladder_name: str) -> Optional[LadderArchConfig]:
    """Look up architecture config by ladder name (exact then substring match)."""
    lower = ladder_name.lower()
    if lower in LADDER_ARCH_CONFIGS:
        return LADDER_ARCH_CONFIGS[lower]
    for pattern, cfg in LADDER_ARCH_CONFIGS.items():
        if pattern in lower:
            return cfg
    return None


def compute_specs_for_size(
    ladder_name: str,
    size: str,
    seq_len: Optional[int] = None,
    chinchilla_flops: bool = False,
) -> Optional[dict]:
    """
    Compute architecture specs for a given ladder name and model size.

    Returns a dict with ``non_embed_params``, ``total_params``,
    ``total_flops_per_token``, etc., or ``None`` if the architecture
    cannot be determined.
    """
    config = _get_arch_config(ladder_name)
    if config is None:
        return None

    spec = OLMO3_SPECS_BY_NAME.get(size)
    if spec is None:
        return None

    effective_seq_len = seq_len or config.seq_len

    if config.is_transformer:
        return compute_olmo3_specs(
            spec, seq_len=effective_seq_len, chinchilla_flops=chinchilla_flops
        )
    elif config.layer_type == "mamba2":
        return compute_hybrid_mamba2_specs(
            spec,
            transformer_ratio=config.transformer_ratio,
            seq_len=effective_seq_len,
            force_final_attention=config.force_final_attention,
            chinchilla_flops=chinchilla_flops,
        )
    else:
        return compute_hybrid_specs(
            spec,
            transformer_ratio=config.transformer_ratio,
            seq_len=effective_seq_len,
            force_final_attention=config.force_final_attention,
            placement=config.placement,
            chinchilla_flops=chinchilla_flops,
        )


# ---------------------------------------------------------------------------
# Display names and param-count helper
# ---------------------------------------------------------------------------

DISPLAY_NAMES: Dict[str, str] = {
    "olmo3": "Olmo 3",
    "olmo3-1": "Olmo 3 v1",
    "olmo3-2": "Olmo 3 v2",
    "olmo3-3": "Olmo 3 v3",
    "pure-gdn": "Pure GDN",
    "hybrid-gdn": "Hybrid GDN (1/4)",
    "hybrid-gdn-half": "Hybrid GDN (1/2)",
    "hybrid-gdn-eight": "Hybrid GDN (1/8)",
    "hybrid-gdn-middle": "Hybrid GDN (Middle)",
    "pure-mamba": "Pure Mamba",
    "hybrid-mamba": "Hybrid Mamba",
}


def get_display_name(ladder_name: str) -> str:
    """Return a human-readable display name for a ladder."""
    return DISPLAY_NAMES.get(ladder_name, ladder_name)


def get_param_count(ladder_name: str, size: str) -> Optional[int]:
    """Return the non-embedding parameter count for a ladder architecture and size.

    Uses :func:`compute_specs_for_size` to compute the count dynamically.
    Returns ``None`` if the architecture/size combination is not recognized.
    """
    specs = compute_specs_for_size(ladder_name, size)
    if specs is None:
        return None
    return specs["non_embed_params"]
