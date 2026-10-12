"""Resolve an HF Qwen3-dense base model to its olmo-core builder.

olmo-core only ships `qwen3` dense builders (0.6B–32B) and a `qwen3`-only HF weight
converter — no qwen2, no hybrid Qwen3.5, no MoE, no LoRA. This module is the single
place that maps an HF model id -> (builder name, vocab, dims) and rejects anything the
olmo-core path can't train, so the dispatcher (train.py), the trainer (olmo_train.py),
the converter and the exporter all stay in sync on what's supported.

Detection reads the HF `config.json` (`model_type`, `hidden_size`, `num_hidden_layers`)
rather than the model-id string: names like `Qwen3-4B-Base` vs `Qwen3-4B` vs community
remixes vary, and the `Qwen3.5`/hybrid models share the `Qwen3` prefix but a different
`model_type`. Size is keyed on (hidden_size, num_hidden_layers) — 14B and 32B share
hidden_size=5120 and are disambiguated by layer count.
"""

from dataclasses import dataclass

# (hidden_size, num_hidden_layers) -> olmo-core TransformerConfig builder method.
# These are the canonical Qwen3-dense shapes; an exact match guarantees the builder's
# own defaults (head_dim, n_kv_heads, ffn size, qk-norm, rope) are already correct, so
# we don't override them.
QWEN3_BUILDERS = {
    (1024, 28): "qwen3_0_6B",
    (2048, 28): "qwen3_1_7B",
    (2560, 36): "qwen3_4B",
    (4096, 36): "qwen3_8B",
    (5120, 48): "qwen3_14B",
    (5120, 64): "qwen3_32B",
}

# Hybrid Qwen3.5 (GDN linear + gated-softmax). model_type='qwen3_5' is a VL wrapper whose
# text decoder lives under config['text_config'] (model_type='qwen3_5_text'). The olmo-core
# builders + convert_qwen3_5_state_from_hf live on the qwen35/amandab/prasann branches.
# Keyed on (text hidden_size, text num_hidden_layers).
QWEN35_BUILDERS = {
    (1024, 24): "qwen3_5_0_8B",
    (2048, 24): "qwen3_5_2B",
    (2560, 32): "qwen3_5_4B",
    (4096, 32): "qwen3_5_9B",
    (5120, 64): "qwen3_5_27B",
}


def _raw_hf_config(base_model: str) -> dict:
    """Read a model's raw config.json as a dict, without instantiating a transformers
    config class. Needed for `qwen3_5`, which transformers <release-with-qwen3_5 cannot
    parse via AutoConfig (raises 'does not recognize this architecture')."""
    import json
    import os

    if os.path.isdir(base_model):
        path = os.path.join(base_model, "config.json")
    else:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(base_model, "config.json")
    with open(path) as f:
        return json.load(f)


@dataclass
class OlmoModelSpec:
    """Everything the olmo-core path needs to know about a base model."""
    base_model: str        # HF id, e.g. "Qwen/Qwen3-4B-Base"
    builder: str           # TransformerConfig builder name, e.g. "qwen3_4B"
    vocab_size: int        # from HF config (don't hardcode 151936 — respect resized vocabs)
    n_layers: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    hidden_size: int
    is_hybrid: bool = False   # True for Qwen3.5 (GDN linear + softmax); needs FLA + direct-convert


def resolve_olmo_model(base_model: str) -> OlmoModelSpec:
    """Map an HF model id to an OlmoModelSpec, or raise ValueError if unsupported.

    Handles Qwen3 *dense* (model_type='qwen3') and hybrid *Qwen3.5* (model_type='qwen3_5',
    text decoder under config['text_config']). Raises with an actionable message for qwen2,
    MoE variants, and shapes with no matching olmo-core builder.
    """
    raw = _raw_hf_config(base_model)
    model_type = raw.get("model_type")

    # --- hybrid Qwen3.5: read the nested text_config; needs the FLA + qwen3_5 olmo branch ---
    if model_type in ("qwen3_5", "qwen3_5_text"):
        tc = raw.get("text_config", raw)
        hidden = tc["hidden_size"]
        layers = tc["num_hidden_layers"]
        builder = QWEN35_BUILDERS.get((hidden, layers))
        if builder is None:
            raise ValueError(
                f"No olmo-core qwen3_5 builder for text hidden_size={hidden}, "
                f"num_hidden_layers={layers} ({base_model!r}). Known: {sorted(QWEN35_BUILDERS)}."
            )
        return OlmoModelSpec(
            base_model=base_model,
            builder=builder,
            vocab_size=tc["vocab_size"],
            n_layers=layers,
            n_heads=tc["num_attention_heads"],
            n_kv_heads=tc.get("num_key_value_heads", tc["num_attention_heads"]),
            head_dim=tc.get("head_dim", hidden // tc["num_attention_heads"]),
            hidden_size=hidden,
            is_hybrid=True,
        )

    if model_type != "qwen3":
        raise ValueError(
            f"olmo-core backend supports only Qwen3 *dense* models (model_type='qwen3'); "
            f"got model_type='{model_type}' for {base_model!r}. Qwen2.5 ('qwen2') and the "
            f"hybrid Qwen3.5 are not supported by olmo-core — drop `backend: olmo-core` and "
            f"use the axolotl/chunked path instead."
        )
    # Dense only: refuse anything that smells like a mixture-of-experts config.
    if raw.get("num_experts") or raw.get("num_local_experts"):
        raise ValueError(
            f"{base_model!r} looks like an MoE model; the olmo-core path is Qwen3 dense only."
        )

    hidden = raw["hidden_size"]
    layers = raw["num_hidden_layers"]
    builder = QWEN3_BUILDERS.get((hidden, layers))
    if builder is None:
        raise ValueError(
            f"No olmo-core qwen3 builder for hidden_size={hidden}, num_hidden_layers={layers} "
            f"({base_model!r}). Known shapes: {sorted(QWEN3_BUILDERS)}."
        )

    head_dim = raw.get("head_dim") or (hidden // raw["num_attention_heads"])
    return OlmoModelSpec(
        base_model=base_model,
        builder=builder,
        vocab_size=raw["vocab_size"],
        n_layers=layers,
        n_heads=raw["num_attention_heads"],
        n_kv_heads=raw.get("num_key_value_heads", raw["num_attention_heads"]),
        head_dim=head_dim,
        hidden_size=hidden,
    )


def build_transformer_config(spec: OlmoModelSpec, dtype=None):
    """Build the olmo-core TransformerConfig for a resolved spec.

    `dtype` (an olmo_core.config.DType) is forwarded to the builder for the "full bf16"
    arm; pass None for the bf16-mixed baseline (model built fp32).
    """
    from olmo_core.nn.transformer import TransformerConfig

    builder_fn = getattr(TransformerConfig, spec.builder)
    kwargs = {}
    if dtype is not None:
        kwargs["dtype"] = dtype
    # Optional attention-backend override (e.g. OLMO_ATTN_BACKEND=torch). The qwen3_5 builders
    # default to flash_2; the torch backend is the one validated to match HF forward+backward.
    import os
    _ab = os.environ.get("OLMO_ATTN_BACKEND")
    if _ab:
        from olmo_core.nn.attention import AttentionBackendName
        kwargs["attn_backend"] = AttentionBackendName(_ab)
    return builder_fn(vocab_size=spec.vocab_size, **kwargs)
