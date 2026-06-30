"""Olmo-3 mask patch: make sliding-window-attention layers behave as full
causal attention so the chunked attention mask passed via `attention_mask=`
is not intersected with a sliding window.

Olmo-3's `forward()` rebuilds per-layer masks via:

    causal_mask_mapping = {
        "full_attention":    create_causal_mask(**mask_kwargs),
        "sliding_attention": create_sliding_window_causal_mask(**mask_kwargs),
    }

The user-provided 4D `attention_mask` is folded into `mask_kwargs` and
intersected (logical AND) with each per-layer pattern. For chunked-mask
training/eval this means sliding layers see `chunked ∩ sliding(4096)`,
which silently truncates inter-chunk reach once seq_len > 4096.

We replace `create_sliding_window_causal_mask` with `create_causal_mask`
inside the olmo3 modeling module so all softmax layers receive the same
`chunked ∩ causal` (= chunked) mask. Idempotent; no-op for non-olmo3.
"""
from __future__ import annotations


def maybe_patch_olmo3_sliding_to_full(model) -> bool:
    cfg = getattr(model, "config", None)
    if getattr(cfg, "model_type", None) != "olmo3":
        return False
    from transformers.models.olmo3 import modeling_olmo3
    if getattr(modeling_olmo3, "_corpus_reasoning_sliding_patched", False):
        return True
    modeling_olmo3.create_sliding_window_causal_mask = modeling_olmo3.create_causal_mask
    modeling_olmo3._corpus_reasoning_sliding_patched = True
    print("  [olmo3 patch] sliding_attention layers now use create_causal_mask "
          "(no window restriction); chunked mask applies uniformly.")
    return True
