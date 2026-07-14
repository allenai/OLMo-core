"""Attention smoothing with curriculum decay.

Idea: for "selected" query rows (question and/or answer tokens), blend the
attention output with a uniform-over-visible-keys output:

    out_t = (1 - alpha) * attn_out_t + alpha * mean_V[visible to t]

This is equivalent to mixing the attention distribution with uniform, but
avoids rewriting the attention kernel — we just add a cheap running-mean
of V in parallel and blend per row.

Curriculum: alpha(step) = alpha_init * max(0, 1 - step / (decay_end_fraction * total_steps))
so alpha decays linearly to 0 at `decay_end_fraction` of training and stays 0.

Row selection:
  - Answer rows: tokens where `labels != -100`.
  - Question rows: tokens inside <|q_start|>...<|q_end|> marker pairs (exclusive
    of the markers themselves).
Union of the enabled selectors.

Installation:
  - `setup_smoothing_tokens(tokenizer)` registers the two markers.
  - `install_smoothed_attention(model, state)` registers a custom attention
    function `sdpa_smoothed` that reads `state["alpha"]`, `state["row_mask"]`,
    `state["pad_mask"]` each forward. The caller populates those fields per
    step (via a Trainer callback) and per batch (via a forward pre-hook).

Debug invariant:
  - With `alpha == 0.0` the custom path returns the unpatched SDPA output
    exactly, so a training run is bit-identical to an unsmoothed one.
"""

from __future__ import annotations

from typing import Any

import torch

Q_START = "<|q_start|>"
Q_END = "<|q_end|>"


# ---------------------------------------------------------------------------
# Tokenizer setup
# ---------------------------------------------------------------------------

def setup_smoothing_tokens(tokenizer):
    """Register Q_START/Q_END as additional special tokens. Returns their ids."""
    tokenizer.add_special_tokens(
        {"additional_special_tokens": [Q_START, Q_END]}
    )
    q_start_id = tokenizer.convert_tokens_to_ids(Q_START)
    q_end_id = tokenizer.convert_tokens_to_ids(Q_END)
    return q_start_id, q_end_id


# ---------------------------------------------------------------------------
# Row mask
# ---------------------------------------------------------------------------

def build_row_mask(
    input_ids: torch.Tensor,
    labels: torch.Tensor | None,
    q_start_id: int | None,
    q_end_id: int | None,
    smooth_question: bool,
    smooth_answer: bool,
) -> torch.Tensor:
    """Return a BoolTensor[B, T] marking rows to smooth.

    - Answer rows (when smooth_answer): labels != -100.
    - Question rows (when smooth_question): positions strictly between matched
      <|q_start|>...<|q_end|> pairs. Markers themselves are excluded.

    Unpaired markers (e.g. truncated at EOS) are ignored safely.
    """
    B, T = input_ids.shape
    device = input_ids.device
    mask = torch.zeros((B, T), dtype=torch.bool, device=device)

    if smooth_answer and labels is not None:
        mask |= (labels != -100)

    if smooth_question and q_start_id is not None and q_end_id is not None:
        # Build an "inside question" flag via cumulative diff:
        #   +1 at the token AFTER q_start
        #   -1 at the position OF q_end
        # cumsum then > 0 between markers (exclusive of markers).
        delta = torch.zeros((B, T), dtype=torch.int32, device=device)
        is_start = (input_ids == q_start_id)
        is_end = (input_ids == q_end_id)
        # Shift starts by +1 so the start marker itself isn't flagged.
        start_shifted = torch.zeros_like(is_start)
        start_shifted[:, 1:] = is_start[:, :-1]
        delta += start_shifted.to(torch.int32)
        delta -= is_end.to(torch.int32)
        inside = torch.cumsum(delta, dim=1) > 0
        mask |= inside

    return mask


# ---------------------------------------------------------------------------
# Curriculum
# ---------------------------------------------------------------------------

def alpha_from_step(
    step: int, total_steps: int, alpha_init: float, decay_end_fraction: float
) -> float:
    """Linear decay: alpha_init at step 0, 0 by step decay_end_fraction*total_steps."""
    if alpha_init <= 0.0 or decay_end_fraction <= 0.0 or total_steps <= 0:
        return 0.0
    end = max(1, int(decay_end_fraction * total_steps))
    if step >= end:
        return 0.0
    return float(alpha_init) * (1.0 - step / end)


# ---------------------------------------------------------------------------
# Mean-V blending core
# ---------------------------------------------------------------------------

def _mean_v_cumulative(
    value: torch.Tensor, pad_mask: torch.Tensor
) -> torch.Tensor:
    """Per-row causal mean of V over visible (non-pad) keys.

    value:    [B, H_kv, T, D]  (bf16/fp16/fp32)
    pad_mask: [B, T] bool — True for valid tokens, False for padding.

    Returns:  [B, H_kv, T, D] — mean_v_t = mean(value[b, h, 0:t+1] over k where pad_mask[b, k])
    """
    B, H, T, D = value.shape
    m = pad_mask.to(value.dtype).view(B, 1, T, 1)  # [B,1,T,1]
    v_masked = value * m
    cum_v = torch.cumsum(v_masked, dim=2)
    cum_n = torch.cumsum(m, dim=2).clamp(min=1.0)  # avoid div-by-zero
    return cum_v / cum_n


def _expand_kv_to_q_heads(mean_v: torch.Tensor, n_q_heads: int) -> torch.Tensor:
    """Repeat-interleave the KV heads to match the query head count (GQA)."""
    B, H_kv, T, D = mean_v.shape
    if H_kv == n_q_heads:
        return mean_v
    assert n_q_heads % H_kv == 0, (
        f"q heads ({n_q_heads}) must be a multiple of kv heads ({H_kv})"
    )
    rep = n_q_heads // H_kv
    return mean_v.repeat_interleave(rep, dim=1)


def blend_with_mean_v(
    attn_out: torch.Tensor,
    value: torch.Tensor,
    row_mask: torch.Tensor,
    pad_mask: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    """Blend attn_out with the causal mean of V on rows where row_mask=True.

    attn_out: [B, H_q, T, D]
    value:    [B, H_kv, T, D]
    row_mask: [B, T]  bool — which query rows to smooth
    pad_mask: [B, T]  bool — True for valid tokens
    alpha:    float in [0, 1]

    Returns blended tensor same shape/dtype as attn_out. alpha=0 → attn_out.
    """
    if alpha <= 0.0:
        return attn_out
    if not row_mask.any():
        return attn_out
    mean_v = _mean_v_cumulative(value, pad_mask)
    mean_v = _expand_kv_to_q_heads(mean_v, attn_out.shape[1])
    sel = row_mask.view(row_mask.shape[0], 1, row_mask.shape[1], 1)
    blended = (1.0 - alpha) * attn_out + alpha * mean_v.to(attn_out.dtype)
    return torch.where(sel, blended, attn_out)


# ---------------------------------------------------------------------------
# HF attention registration
# ---------------------------------------------------------------------------

def install_smoothed_attention(model, state: dict) -> None:
    """Register a `sdpa_smoothed` attention function that wraps HF's SDPA path.

    The wrapper reads `state["alpha"]`, `state["row_mask"]`, `state["pad_mask"]`
    each forward call and applies `blend_with_mean_v` to the attention output.
    If `alpha` is 0 (or missing), the result is identical to plain SDPA.

    The caller is responsible for populating `state` each step/batch:
      - `state["alpha"]`: float (updated by a Trainer callback each step)
      - `state["row_mask"]`: BoolTensor[B, T] (updated by a model pre-hook)
      - `state["pad_mask"]`: BoolTensor[B, T] (updated by a model pre-hook)

    Switches `model.config._attn_implementation` to "sdpa_smoothed".
    """
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    sdpa_fn = ALL_ATTENTION_FUNCTIONS["sdpa"]

    def sdpa_smoothed(
        module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask,
        **kwargs,
    ):
        # Call the stock SDPA path to compute attn_out (same numerical behavior
        # as the stock model when alpha=0).
        attn_out, attn_weights = sdpa_fn(
            module, query, key, value, attention_mask, **kwargs
        )

        alpha = float(state.get("alpha", 0.0))
        if alpha <= 0.0:
            return attn_out, attn_weights
        row_mask = state.get("row_mask", None)
        pad_mask = state.get("pad_mask", None)
        if row_mask is None or pad_mask is None:
            return attn_out, attn_weights

        # attn_out may be [B, T, H_q, D] or [B, H_q, T, D] depending on the HF
        # version. The SDPA wrapper returns [B, T, H_q, D] on current releases;
        # reshape to [B, H_q, T, D] for the blend then back.
        if attn_out.dim() == 4 and attn_out.shape[1] == query.shape[2]:
            # layout: [B, T, H_q, D]  (T==query seq len, H_q==query.shape[1] after perm)
            out_btdh = attn_out
            out_bhtd = out_btdh.transpose(1, 2).contiguous()
            value_bhtd = value
        else:
            out_bhtd = attn_out
            value_bhtd = value

        B, T = row_mask.shape
        # Align masks to current batch/seq (handle padding mismatch defensively).
        if out_bhtd.shape[0] != B or out_bhtd.shape[2] != T:
            return attn_out, attn_weights

        blended_bhtd = blend_with_mean_v(
            out_bhtd, value_bhtd, row_mask.to(out_bhtd.device),
            pad_mask.to(out_bhtd.device), alpha,
        )

        if attn_out is out_bhtd:
            return blended_bhtd, attn_weights
        return blended_bhtd.transpose(1, 2).contiguous(), attn_weights

    ALL_ATTENTION_FUNCTIONS["sdpa_smoothed"] = sdpa_smoothed
    # Switch the model over. HF propagates the change to every attention layer.
    model.config._attn_implementation = "sdpa_smoothed"
    for module in model.modules():
        if hasattr(module, "_attn_implementation"):
            module._attn_implementation = "sdpa_smoothed"


# ---------------------------------------------------------------------------
# Trainer glue
# ---------------------------------------------------------------------------

def attach_row_mask_pre_hook(
    model,
    state: dict,
    q_start_id: int | None,
    q_end_id: int | None,
    smooth_question: bool,
    smooth_answer: bool,
    pad_token_id: int | None,
):
    """Install a forward pre-hook that computes row_mask & pad_mask each batch.

    Reads input_ids / labels / attention_mask from the forward kwargs and stashes
    the computed masks on `state`. Safe when alpha=0 (masks are still computed,
    blend_with_mean_v short-circuits).
    """
    def pre_hook(module, args, kwargs):
        input_ids = kwargs.get("input_ids", None)
        labels = kwargs.get("labels", None)
        attn_mask = kwargs.get("attention_mask", None)
        if input_ids is None:
            return
        # Compute pad_mask: prefer attention_mask if 2D bool/int, else use pad_token.
        if attn_mask is not None and attn_mask.dim() == 2:
            pad_mask = attn_mask.bool()
        elif pad_token_id is not None:
            pad_mask = input_ids != pad_token_id
        else:
            pad_mask = torch.ones_like(input_ids, dtype=torch.bool)
        row_mask = build_row_mask(
            input_ids, labels, q_start_id, q_end_id,
            smooth_question=smooth_question, smooth_answer=smooth_answer,
        )
        state["row_mask"] = row_mask
        state["pad_mask"] = pad_mask

    # Register on the top-level model.
    return model.register_forward_pre_hook(pre_hook, with_kwargs=True)


# ---------------------------------------------------------------------------
# Config parsing
# ---------------------------------------------------------------------------

def parse_smoothing_config(cfg: dict) -> dict[str, Any] | None:
    """Return a normalized dict, or None if the `attn_smoothing` block is absent.

    A present block with `alpha_init: 0.0` still returns a config so the
    smoothing code path is exercised (for bit-identity debug runs) — the
    runtime alpha from `alpha_from_step` will still be 0 at every step.
    """
    block = cfg.get("attn_smoothing")
    if block is None:
        return None
    return {
        "alpha_init": float(block.get("alpha_init", 0.0)),
        "decay_end_fraction": float(block.get("decay_end_fraction", 0.5)),
        "smooth_question": bool(block.get("smooth_question", True)),
        "smooth_answer": bool(block.get("smooth_answer", True)),
    }
