"""Inverse of olmo-core's convert_qwen3_5_state_from_hf: OLMo-core hybrid state -> HF.

The branch ships only the from_hf direction for qwen3_5 (HF fused projections -> OLMo
separate). To export a trained Qwen3.5 olmo checkpoint for HF/vLLM eval we need the inverse:
re-fuse in_proj_qkv = cat(w_q,w_k,w_v), conv1d = cat(q,k,v convs), interleave q_proj from
(w_q,w_g), and undo the norm +1 transform (HF zero-init `1+w` vs OLMo ones-init `w`).

Targets the TEXT-only decoder keys (model.layers.N.*, model.embed_tokens, model.norm,
lm_head) so the result loads as a text `Qwen3_5ForCausalLM` — no vision tower needed for eval.

Validated by round_trip_check(): from_hf(to_hf(state)) must equal the original HF state.
"""
from typing import Any, Dict, List

import torch

# norms that from_hf shifts by +1 (olmo = hf + 1); inverse subtracts 1. NOTE: GDN o_norm
# (linear_attn.norm) is NOT in this set.
_OLMO_PLUS1_SUFFIXES = (
    "attention_norm.weight",
    "feed_forward_norm.weight",
    "lm_head.norm.weight",
    "attention.q_norm.weight",
    "attention.k_norm.weight",
)


def _unshift(olmo_key: str, v: torch.Tensor) -> torch.Tensor:
    return v - 1.0 if any(olmo_key.endswith(s) for s in _OLMO_PLUS1_SUFFIXES) else v


def _refuse_q_proj(w_q: torch.Tensor, w_g: torch.Tensor, n_heads: int, head_dim: int) -> torch.Tensor:
    """Inverse of _split_qwen3_5_q_proj: interleave [q_0,g_0,q_1,g_1,...] per head."""
    hidden = w_q.shape[1]
    q = w_q.reshape(n_heads, head_dim, hidden)
    g = w_g.reshape(n_heads, head_dim, hidden)
    qg = torch.stack([q, g], dim=1)  # (n_heads, 2, head_dim, hidden)
    return qg.reshape(n_heads * 2 * head_dim, hidden)


def convert_qwen3_5_state_to_hf(olmo_state: Dict[str, Any], text_config) -> Dict[str, Any]:
    """OLMo-core hybrid Qwen3.5 state -> HF text-decoder state dict.

    text_config: object/namespace with layer_types, num_hidden_layers, num_attention_heads,
    head_dim, linear_num_key_heads, linear_key_head_dim, linear_num_value_heads,
    linear_value_head_dim, tie_word_embeddings.
    """
    g = olmo_state.get
    layer_types: List[str] = list(text_config.layer_types)
    n_layers = int(text_config.num_hidden_layers)
    n_heads = int(text_config.num_attention_heads)
    head_dim = int(text_config.head_dim)
    key_dim = int(text_config.linear_num_key_heads) * int(text_config.linear_key_head_dim)
    value_dim = int(text_config.linear_num_value_heads) * int(text_config.linear_value_head_dim)

    hf: Dict[str, Any] = {}
    # shared
    if "embeddings.weight" in olmo_state:
        hf["model.embed_tokens.weight"] = g("embeddings.weight")
    if "lm_head.norm.weight" in olmo_state:
        hf["model.norm.weight"] = _unshift("lm_head.norm.weight", g("lm_head.norm.weight"))
    # ALWAYS emit the trained output head from olmo lm_head.w_out. Even with
    # tie_word_embeddings, olmo-core stores (and may update) w_out separately, so relying on
    # HF tie_weights() (lm_head <- embed) would silently discard the trained head -> the model
    # then produces format-plausible but content-wrong output. Emit it explicitly.
    if "lm_head.w_out.weight" in olmo_state:
        hf["lm_head.weight"] = g("lm_head.w_out.weight")

    for i in range(n_layers):
        op = f"blocks.{i}."
        hp = f"model.layers.{i}."
        # shared per-layer norms + mlp
        hf[f"{hp}input_layernorm.weight"] = _unshift(f"{op}attention_norm.weight", g(f"{op}attention_norm.weight"))
        hf[f"{hp}post_attention_layernorm.weight"] = _unshift(f"{op}feed_forward_norm.weight", g(f"{op}feed_forward_norm.weight"))
        hf[f"{hp}mlp.gate_proj.weight"] = g(f"{op}feed_forward.w1.weight")
        hf[f"{hp}mlp.down_proj.weight"] = g(f"{op}feed_forward.w2.weight")
        hf[f"{hp}mlp.up_proj.weight"] = g(f"{op}feed_forward.w3.weight")

        if layer_types[i] == "linear_attention":
            lp = f"{hp}linear_attn."
            hf[f"{lp}in_proj_qkv.weight"] = torch.cat(
                [g(f"{op}attention.w_q.weight"), g(f"{op}attention.w_k.weight"), g(f"{op}attention.w_v.weight")], dim=0)
            hf[f"{lp}in_proj_z.weight"] = g(f"{op}attention.w_g.weight")
            hf[f"{lp}in_proj_a.weight"] = g(f"{op}attention.w_a.weight")
            hf[f"{lp}in_proj_b.weight"] = g(f"{op}attention.w_b.weight")
            hf[f"{lp}out_proj.weight"] = g(f"{op}attention.w_out.weight")
            hf[f"{lp}conv1d.weight"] = torch.cat(
                [g(f"{op}attention.q_conv1d.weight"), g(f"{op}attention.k_conv1d.weight"), g(f"{op}attention.v_conv1d.weight")], dim=0)
            hf[f"{lp}norm.weight"] = g(f"{op}attention.o_norm.weight")  # GDN o_norm: NOT shifted
            hf[f"{lp}A_log"] = g(f"{op}attention.A_log")
            hf[f"{lp}dt_bias"] = g(f"{op}attention.dt_bias")
        elif layer_types[i] == "full_attention":
            sp = f"{hp}self_attn."
            hf[f"{sp}q_proj.weight"] = _refuse_q_proj(
                g(f"{op}attention.w_q.weight"), g(f"{op}attention.w_g.weight"), n_heads, head_dim)
            hf[f"{sp}k_proj.weight"] = g(f"{op}attention.w_k.weight")
            hf[f"{sp}v_proj.weight"] = g(f"{op}attention.w_v.weight")
            hf[f"{sp}o_proj.weight"] = g(f"{op}attention.w_out.weight")
            hf[f"{sp}q_norm.weight"] = _unshift(f"{op}attention.q_norm.weight", g(f"{op}attention.q_norm.weight"))
            hf[f"{sp}k_norm.weight"] = _unshift(f"{op}attention.k_norm.weight", g(f"{op}attention.k_norm.weight"))
        else:
            raise ValueError(f"unknown layer type {layer_types[i]!r} at {i}")

    return hf


def round_trip_check(base_snapshot: str, atol: float = 0.02) -> None:
    """from_hf(to_hf(.)) == identity check, using the cached base model. Raises on mismatch."""
    import glob
    import json
    import os
    import types

    from safetensors.torch import load_file
    from olmo_core.nn.hf.convert import convert_qwen3_5_state_from_hf

    raw = json.load(open(os.path.join(base_snapshot, "config.json")))
    cfg = types.SimpleNamespace(**raw)
    cfg.text_config = types.SimpleNamespace(**raw["text_config"])

    hf0: Dict[str, Any] = {}
    for s in sorted(glob.glob(os.path.join(base_snapshot, "*.safetensors"))):
        hf0.update(load_file(s))

    olmo = convert_qwen3_5_state_from_hf(cfg, hf0)          # HF -> olmo (validated path)
    hf1 = convert_qwen3_5_state_to_hf(olmo, cfg.text_config)  # olmo -> HF (this module)

    # Build the text-decoder reference exactly as from_hf normalizes it (drop vision tower:
    # only the text decoder lives under language_model / the top-level text keys).
    ref: Dict[str, Any] = {}
    for k, v in hf0.items():
        if any(skip in k for skip in ("visual.", "vision_tower", "vision_config", "multi_modal")):
            continue
        if ".layers." in k:
            ref["model" + k[k.index(".layers."):]] = v
        elif k.endswith("embed_tokens.weight"):
            ref["model.embed_tokens.weight"] = v
        elif k.endswith("norm.weight") and ".layers." not in k:
            ref["model.norm.weight"] = v
        elif k.endswith("lm_head.weight"):
            ref["lm_head.weight"] = v
    # if base ties embeddings (no separate lm_head.weight), our converter copies w_out==embed;
    # don't fail the round-trip on a tie-only difference
    if "lm_head.weight" not in ref and "lm_head.weight" in hf1:
        hf1.pop("lm_head.weight")

    missing = sorted(set(ref) - set(hf1))
    extra = sorted(set(hf1) - set(ref))
    print(f"[round-trip] produced {len(hf1)} keys; ref {len(ref)}; missing={len(missing)} extra={len(extra)}", flush=True)
    if missing[:5]:
        print("  missing:", missing[:5], flush=True)
    if extra[:5]:
        print("  extra:", extra[:5], flush=True)

    bad = 0
    for k in sorted(set(ref) & set(hf1)):
        a, b = ref[k].float(), hf1[k].float()
        if a.shape != b.shape or not torch.allclose(a, b, atol=atol, rtol=0):
            bad += 1
            if bad <= 5:
                d = (a - b).abs().max().item() if a.shape == b.shape else float("nan")
                print(f"  MISMATCH {k}: shape {tuple(a.shape)} vs {tuple(b.shape)} maxdiff={d}", flush=True)
    ok = (bad == 0 and not missing and not extra)
    print(f"[round-trip] {'PASS' if ok else 'FAIL'} (value-mismatches={bad})", flush=True)
    if not ok:
        raise SystemExit("round-trip inverse-converter check FAILED")
