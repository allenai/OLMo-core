"""Training-time helpers for adding the stupid-backoff PoE bias to LM
logits.

This module is the runtime-side parallel of the KN-smoothed-PoE code in
``train_module.py`` (``_apply_poe_eval_bias``/``_compute_poe_loss``),
factored out into a standalone helper so it can be unit-tested without
spinning up a TransformerTrainModule.

The bias math
=============

Stupid-backoff scores aren't normalized to a distribution, so the
"non-overridden" tokens at each position aren't a constant the way the
KN-smoothed log_residual is. We instead split the per-position bias
``b[w]`` into two pieces:

1. **A length-V unigram floor**, the same at every position, that
   would-be the per-token score *if* every token fell back to unigram.
   Concretely::

       unigram_floor[w] = (N_max - 1) · log α + log((C_1(w) + 1) / (T + V_dolma2))

   (Laplace +1 smoothing, with the SB discount folded in.) Loaded once
   at startup from ``counts_index/order1.*`` via
   :class:`olmo_core.data.stupid_backoff_ngram.StupidBackoffNgramLM`.

2. **Per-position sparse overrides** for tokens that were actually
   observed at some order > 1 for the position's history. These come
   in as four flat tensors (the collator's ``sb_override_*`` fields).

The math the train step needs:

  logits[b, t, w] += λ · unigram_floor[w]                              # broadcast V-add
  logits[b, t, w] += λ · (sb_score(w | h_t) − unigram_floor[w])         # at observed (b, t, w)

Net per-position-per-token::

  bias[b, t, w] = λ · sb_score(w | h_t)      if (b, t, w) ∈ overrides
                  λ · unigram_floor[w]        otherwise

so every token in the vocab gets a real SB bias, with the higher-order
override winning at the (b, t, w) triples that have one.

Memory: we never materialize a `(B, S, V)` bias tensor. The unigram
floor is V floats (≈400 KB). The override deltas are
`(total_overrides_in_batch,)` floats (~MB-range). PyTorch handles both
via broadcasting + scatter_add_ on the existing logits tensor in place.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import torch


def apply_sb_bias_inplace(
    logits: torch.Tensor,
    unigram_floor: torch.Tensor,
    sb_override_batch_idx: torch.Tensor,
    sb_override_position: torch.Tensor,
    sb_override_token_id: torch.Tensor,
    sb_override_log_score: torch.Tensor,
    poe_lambda: float,
) -> None:
    """Add the stupid-backoff PoE bias to LM logits *in place*.

    :param logits: shape ``(B, S, V)``, must be contiguous and float.
        Modified in place.
    :param unigram_floor: shape ``(V,)``, same device and dtype as
        ``logits``. The pre-computed log-SB unigram-floor vector.
    :param sb_override_batch_idx: shape ``(n_overrides,)`` long.
        ``b`` of each override.
    :param sb_override_position: shape ``(n_overrides,)`` long.
        ``t`` of each override.
    :param sb_override_token_id: shape ``(n_overrides,)`` long.
        ``w`` (dolma2 token id) of each override.
    :param sb_override_log_score: shape ``(n_overrides,)`` float.
        natural-log SB score for that (b, t, w).
    :param poe_lambda: scalar λ. If 0, returns immediately.
    """
    if poe_lambda == 0.0:
        return
    if not logits.is_contiguous():
        raise ValueError(
            f"apply_sb_bias_inplace requires contiguous logits; got "
            f"contiguous={logits.is_contiguous()}, shape={tuple(logits.shape)}"
        )
    if unigram_floor.shape != (logits.shape[-1],):
        raise ValueError(
            f"unigram_floor shape {tuple(unigram_floor.shape)} doesn't match "
            f"logits V={logits.shape[-1]}"
        )

    # 1. Broadcast the unigram floor onto every (b, t) position.
    #    `logits.add_(unigram_floor, alpha=λ)` does logits += λ · floor with V-broadcast.
    logits.add_(unigram_floor.to(dtype=logits.dtype), alpha=float(poe_lambda))

    if sb_override_batch_idx.numel() == 0:
        return

    # 2. Scatter the per-position override deltas.
    #    delta[i] = λ · (sb_score[i] − unigram_floor[w[i]])
    #    logits[batch_idx[i], position[i], token_id[i]] += delta[i]
    B, S, V = logits.shape
    bidx = sb_override_batch_idx.to(device=logits.device, dtype=torch.long)
    pos = sb_override_position.to(device=logits.device, dtype=torch.long)
    tok = sb_override_token_id.to(device=logits.device, dtype=torch.long)
    sc = sb_override_log_score.to(device=logits.device, dtype=torch.float32)
    floor_at_tok = unigram_floor.index_select(0, tok).to(dtype=torch.float32)
    delta = (float(poe_lambda) * (sc - floor_at_tok)).to(dtype=logits.dtype)
    flat_idx = bidx * (S * V) + pos * V + tok
    logits.view(-1).scatter_add_(0, flat_idx, delta)


def compute_sparse_sb_ce_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    unigram_floor: torch.Tensor,
    sb_override_batch_idx: torch.Tensor,
    sb_override_position: torch.Tensor,
    sb_override_token_id: torch.Tensor,
    sb_override_log_score: torch.Tensor,
    poe_lambda: float,
    label_ignore_index: int,
) -> torch.Tensor:
    """Compute exact per-token CE for SB PoE without materializing biased logits.

    This is equivalent to calling :func:`apply_sb_bias_inplace` on an fp32
    logits clone and then computing ``-(label_logit - logsumexp(logits))``.
    It keeps the dense unigram-floor normalization and applies the observed
    higher-order SB continuations as sparse partition-function corrections.
    """
    if logits.ndim != 3:
        raise ValueError(
            f"expected logits with shape (B, S, V), got {tuple(logits.shape)}"
        )
    B, S, V = logits.shape
    if unigram_floor.shape != (V,):
        raise ValueError(
            f"unigram_floor shape {tuple(unigram_floor.shape)} doesn't match logits V={V}"
        )

    logits_f32 = logits.float()
    unigram_floor = unigram_floor.to(device=logits_f32.device, dtype=logits_f32.dtype)
    floor_bias = float(poe_lambda) * unigram_floor
    base_logits = logits_f32 + floor_bias.view(1, 1, V)
    log_sum_exp_base = torch.logsumexp(base_logits, dim=-1)

    safe_labels = labels.to(logits_f32.device, non_blocking=True)
    safe_labels = safe_labels.masked_fill(safe_labels == label_ignore_index, 0)
    label_logits = base_logits.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)

    tok = sb_override_token_id.to(device=logits_f32.device, dtype=torch.long)
    if tok.numel() > 0:
        bidx = sb_override_batch_idx.to(device=logits_f32.device, dtype=torch.long)
        pos = sb_override_position.to(device=logits_f32.device, dtype=torch.long)
        sc = sb_override_log_score.to(device=logits_f32.device, dtype=logits_f32.dtype)

        flat_pos = bidx * S + pos
        override_delta = float(poe_lambda) * (sc - unigram_floor[tok])
        override_base_logits = base_logits[bidx, pos, tok]
        z_delta_flat = torch.zeros(B * S, device=logits_f32.device, dtype=logits_f32.dtype)
        z_delta_flat.index_add_(
            0,
            flat_pos,
            torch.exp(override_base_logits) * torch.expm1(override_delta),
        )
        log_sum_exp = log_sum_exp_base + torch.log1p(
            z_delta_flat.view(B, S) * torch.exp(-log_sum_exp_base)
        )

        flat_labels = safe_labels.reshape(-1)
        label_match = tok == flat_labels[flat_pos]
        if label_match.any():
            label_delta_flat = torch.zeros(
                B * S, device=logits_f32.device, dtype=logits_f32.dtype
            )
            label_delta_flat.index_add_(
                0, flat_pos[label_match], override_delta[label_match]
            )
            label_logits = label_logits + label_delta_flat.view(B, S)
    else:
        log_sum_exp = log_sum_exp_base

    ce_loss = -(label_logits - log_sum_exp)
    return torch.where(
        labels.to(ce_loss.device, non_blocking=True) == label_ignore_index,
        torch.zeros_like(ce_loss),
        ce_loss,
    )


def compute_sb_overrides_for_batch(
    input_ids: torch.Tensor,
    reader,  # StupidBackoffNgramLM
    *,
    batch_threads: int = 1,
) -> Dict[str, torch.Tensor]:
    """Compute the per-batch flat SB-override tensors synchronously.

    Mirrors what
    :class:`olmo_core.data.composable.NgramStupidBackoffInstanceSource`
    does in a dataloader worker, but batched and run inline — used by
    eval callbacks that don't go through the training-time dataloader.

    Returns a dict with the same four flat tensor keys the collator
    produces, on CPU (the caller is expected to move them to the LM's
    device before calling :func:`apply_sb_bias_inplace`).
    """
    B = int(input_ids.shape[0])
    cpu_input_ids = input_ids.detach().to("cpu").numpy()

    def compute_one(b: int) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
        if batch_threads > 1 and hasattr(reader, "_compute_overrides_for_sequence_serial"):
            pos_np, tok_np, sc_np = reader._compute_overrides_for_sequence_serial(  # noqa: SLF001
                cpu_input_ids[b],
                log_timing=False,
            )
        else:
            pos_np, tok_np, sc_np = reader.compute_overrides_for_sequence(cpu_input_ids[b])
        return b, pos_np, tok_np, sc_np

    if batch_threads > 1 and B > 1:
        from concurrent.futures import ThreadPoolExecutor

        max_workers = min(B, int(batch_threads))
        with ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="sb-eval-row",
        ) as pool:
            parts = list(pool.map(compute_one, range(B)))
    else:
        parts = [compute_one(b) for b in range(B)]

    all_bidx, all_pos, all_tok, all_score = [], [], [], []
    for b, pos_np, tok_np, sc_np in parts:
        if pos_np.size == 0:
            continue
        all_bidx.append(np.full(pos_np.size, b, dtype=np.int64))
        all_pos.append(pos_np.astype(np.int64, copy=False))
        all_tok.append(tok_np.astype(np.int64, copy=False))
        all_score.append(sc_np.astype(np.float32, copy=False))
    if not all_bidx:
        return {
            "sb_override_batch_idx": torch.zeros(0, dtype=torch.long),
            "sb_override_position": torch.zeros(0, dtype=torch.long),
            "sb_override_token_id": torch.zeros(0, dtype=torch.long),
            "sb_override_log_score": torch.zeros(0, dtype=torch.float32),
        }
    return {
        "sb_override_batch_idx": torch.as_tensor(np.concatenate(all_bidx)),
        "sb_override_position": torch.as_tensor(np.concatenate(all_pos)),
        "sb_override_token_id": torch.as_tensor(np.concatenate(all_tok)),
        "sb_override_log_score": torch.as_tensor(np.concatenate(all_score)),
    }


def sb_bias_from_batch(batch: Dict[str, Any]) -> Optional[Dict[str, torch.Tensor]]:
    """Extract the four sb_override_* tensors from a collated batch.

    Returns ``None`` if the batch has no SB overrides (i.e. the batch
    didn't come through ``NgramStupidBackoffInstanceSource``). Otherwise
    returns a dict with the four flat tensors.
    """
    if "sb_override_batch_idx" not in batch:
        return None
    return {
        "sb_override_batch_idx": batch["sb_override_batch_idx"],
        "sb_override_position": batch["sb_override_position"],
        "sb_override_token_id": batch["sb_override_token_id"],
        "sb_override_log_score": batch["sb_override_log_score"],
    }
