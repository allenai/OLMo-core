"""LongLoRA: shifted sparse attention (S^2-Attn) + trainable embed/norm.

Paper: "LongLoRA: Efficient Fine-tuning of Long-Context Large Language Models"
       (Chen et al., 2023, arxiv:2309.12307)
Reference impl: https://github.com/dvlab-research/LongLoRA

S^2-Attn (training only):
  - split the sequence into groups of size G (paper default G = seq_len // 4)
  - roll the *back half* of attention heads by -G/2 along the sequence dim
    (key/query/value alike)
  - compute causal attention independently within each group
  - un-roll the back half by +G/2 after attention
At inference, we fall back to plain SDPA (which is the LongLoRA-recommended
setup), so no attention swap is needed at eval time.

LoRA+ (LongLoRA's other half): on top of LoRA on q/k/v/o, embed_tokens and
all RMSNorms are made fully trainable. We rely on PEFT's `modules_to_save`
to (a) flip requires_grad and (b) save those module weights alongside the
LoRA adapter on disk.

Caveat for Qwen3.5: the family is hybrid (softmax + GDN-linear). HF's
ALL_ATTENTION_FUNCTIONS slot only routes the softmax layers, so S^2-Attn
applies to the 6/24 softmax layers and the 18/24 linear layers are
unaffected. This is the same caveat that applies to attn_smoothing on
this architecture.
"""

from __future__ import annotations

import torch


# Module-name suffixes that PEFT will treat as `modules_to_save`. PEFT matches
# by `name.endswith(target)`, so "norm" catches model.norm, input_layernorm,
# post_attention_layernorm, q_norm, k_norm — i.e. every RMSNorm in the model.
LONGLORA_MODULES_TO_SAVE = ["embed_tokens", "norm"]


def install_s2_attn(model, group_size: int, force_eval: bool = False) -> None:
    """Register an `s2_attn` attention function and switch the model to it.

    The implementation wraps HF's stock SDPA: at training time, when the
    sequence length is a multiple of `group_size` and at least 2*group_size,
    we apply the head-half roll + per-group causal SDPA recipe from the
    LongLoRA paper. Otherwise (eval, short seqs, partial groups) we fall
    through to plain SDPA so behavior degrades gracefully.

    `force_eval=True` removes the `module.training` gate, so S^2-Attn also
    fires at inference (used to study the eval-time effect of the swap; the
    LongLoRA paper recommends plain SDPA at eval).
    """
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    sdpa_fn = ALL_ATTENTION_FUNCTIONS["sdpa"]
    G = int(group_size)
    if G <= 0 or G % 2 != 0:
        raise ValueError(f"group_size must be a positive even int, got {G}")

    def s2_attn(module, query, key, value, attention_mask, **kwargs):
        # query: [B, Hq, T, D], key/value: [B, Hkv, T, D]
        B, Hq, T, D = query.shape
        Hkv = key.shape[1]

        # Skip S^2-Attn at eval time, when the seq is too short, or when
        # the seq doesn't tile evenly into groups. In all of these we want
        # plain causal SDPA — same as LongLoRA's "optional at inference".
        if (not (module.training or force_eval)) or T < 2 * G or (T % G) != 0:
            return sdpa_fn(module, query, key, value, attention_mask, **kwargs)

        # Both Hq and Hkv must be even for the half-head split. Qwen2/3.5
        # GQA always satisfies this in practice (even head counts).
        if Hq % 2 != 0 or Hkv % 2 != 0:
            return sdpa_fn(module, query, key, value, attention_mask, **kwargs)

        half_q = Hq // 2
        half_kv = Hkv // 2
        shift = G // 2
        n_g = T // G

        # 1) Roll the back half of heads by -G/2 along the seq dim.
        q = query.clone()
        q[:, half_q:] = torch.roll(q[:, half_q:], shifts=-shift, dims=2)
        k = key.clone()
        k[:, half_kv:] = torch.roll(k[:, half_kv:], shifts=-shift, dims=2)
        v = value.clone()
        v[:, half_kv:] = torch.roll(v[:, half_kv:], shifts=-shift, dims=2)

        # 2) Reshape into groups: [B, H, T, D] -> [B*n_g, H, G, D]
        q = q.view(B, Hq, n_g, G, D).transpose(1, 2).reshape(B * n_g, Hq, G, D)
        k = k.view(B, Hkv, n_g, G, D).transpose(1, 2).reshape(B * n_g, Hkv, G, D)
        v = v.view(B, Hkv, n_g, G, D).transpose(1, 2).reshape(B * n_g, Hkv, G, D)

        # 3) Causal SDPA inside each group. Drop the user-provided 4D mask:
        #    shifted heads see a different effective causal pattern, and the
        #    group reshape makes the original mask unusable. is_causal=True
        #    gives plain triangular within each (G, G) group.
        kw = dict(kwargs)
        kw["is_causal"] = True
        out, weights = sdpa_fn(module, q, k, v, attention_mask=None, **kw)

        # The HF SDPA wrapper returns [B*n_g, G, Hq, D] (transpose-applied),
        # though older versions returned [B*n_g, Hq, G, D]. Detect & normalize
        # to [B, Hq, T, D] for the un-roll step. Both reshapes need a FULL
        # permute (not a single transpose) to bring Hq to axis 1.
        if out.dim() == 4 and out.shape[0] == B * n_g and out.shape[1] == G:
            # [B*n_g, G, Hq, D] -> [B, n_g, G, Hq, D] -> [B, Hq, n_g, G, D] -> [B, Hq, T, D]
            out = (out.view(B, n_g, G, Hq, D)
                      .permute(0, 3, 1, 2, 4).contiguous()
                      .view(B, Hq, T, D))
        else:
            # [B*n_g, Hq, G, D] -> [B, n_g, Hq, G, D] -> [B, Hq, n_g, G, D] -> [B, Hq, T, D]
            out = (out.view(B, n_g, Hq, G, D)
                      .permute(0, 2, 1, 3, 4).contiguous()
                      .view(B, Hq, T, D))

        # 4) Un-roll the back half by +G/2.
        unrolled = out.clone()
        unrolled[:, half_q:] = torch.roll(out[:, half_q:], shifts=shift, dims=2)

        # Match the SDPA wrapper's [B, T, Hq, D] output convention.
        return unrolled.transpose(1, 2).contiguous(), weights

    ALL_ATTENTION_FUNCTIONS["s2_attn"] = s2_attn

    # Switch the live model. HF propagates `_attn_implementation` to every
    # attention module that holds the slot.
    model.config._attn_implementation = "s2_attn"
    for sub in model.modules():
        if hasattr(sub, "_attn_implementation"):
            sub._attn_implementation = "s2_attn"


def parse_longlora_config(cfg: dict) -> dict | None:
    """Return {'group_size': int} when `longlora.enabled` is set; else None.

    The block lives at top level so attention_pattern can stay independent
    (longlora is a *training-time attention swap*, not a chunked mask).
    """
    block = cfg.get("longlora")
    if not block:
        return None
    if not block.get("enabled", False):
        return None
    G = int(block.get("group_size", 0))
    if G <= 0:
        seq_len = int(cfg.get("sequence_len", 0))
        if seq_len <= 0:
            raise ValueError(
                "longlora.group_size unset and sequence_len missing — "
                "cannot derive group size."
            )
        # Paper default: G = seq_len // 4
        G = seq_len // 4
    return {"group_size": G}
