"""Gold-document gradient masking for olmo-core models (standard attention).

Goal
----
Keep every document in the forward pass (so the loss is computed over the full
context exactly as usual), but make sure **only the ground-truth documents** (plus
the instruction / query / answer tokens we actually supervise) drive the weight
update. Distractor-document tokens become *read-only context*: they influence the
forward output, but receive and contribute **zero** gradient.

Why detach K and V (and nothing else)
-------------------------------------
A context token at position ``t`` influences the loss **only by being attended to**
-- i.e. only through its key ``K_t`` and value ``V_t``. Concretely, the weight grad
of an attention projection is ``dL/dW = sum_t  g_t · a_t^T`` over all positions; to
zero a non-gold token's contribution we need its output-side gradient ``g_t`` to be
zero, *not* its activation (zeroing the activation would corrupt the forward pass).

The two "obvious" interventions both fail:

  * **Detaching the input embeddings** of non-gold tokens -- the hidden state is
    still a nonzero *value*, and ``W_K``/``W_V`` still receive
    ``(dL/dK_t)·h_t^T`` because gold/output queries attend to ``t`` (the *read*
    side-door delivers gradient regardless of the input being detached).
  * **Masking the residual-stream gradient** at non-gold positions -- that masks
    gradient flowing *further down* the stream, but ``dL/dK_t``/``dL/dV_t`` arrive
    from the querying positions, *upstream* of any residual-stream gate.

So the exact tensor to cut is ``K``/``V`` themselves. With ``K_t``/``V_t`` detached
at **every** layer, a non-gold token has no backward path to the loss at all (its
only downstream influence -- being attended to -- is severed), so it contributes
zero to *every* weight gradient (``W_Q``/``W_K``/``W_V``/``W_O``/MLP), while the
forward pass is bit-identical (``detach()`` preserves values).

Mechanism
---------
``torch.where(gold[:, :, None, None], k, k.detach())``: forward value is ``k``
everywhere (both branches are equal), but in backward, gradient reaches ``k`` only
at gold key positions. Applied to the ``k``/``v`` tensors right before they enter
the attention backend, so it is **backend-agnostic** (SDPA / flash / flex) and needs
no kernel changes. We apply it at the same seam the FlexAttention installer uses --
each ``olmo_core.nn.attention.Attention`` instance's bound ``sdpa`` method -- so
RoPE, QK-norm and gating are all upstream and untouched.

Important interaction with loss masking
---------------------------------------
The zero-gradient guarantee for a distractor token holds when the loss is **not**
placed on that token's own next-token prediction. In SFT with a ``label_mask`` over
only the completion/answer tokens (our setup), distractor positions never appear in
the loss, so their *query* path (``W_Q``, which we do **not** detach) also receives
no gradient and the severance is total. Under a full unmasked LM loss, a distractor
token's own prediction would still train ``W_Q`` via its query -- see the unit test
``test_gold_grad_mask.py`` which exercises both regimes.

Scope: single-rank, full-sequence (CP=1), same as the FlexAttention installer.
"""

from __future__ import annotations

import hashlib
import types
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Union

import torch

# Role conventions are shared with the chunked/flex path: doc idx >= 0, FREE = -1
# (instruction / query / answer), PAD = -2.
from corpus_reasoning.lib.chunked_attention import FREE_CHUNK_ID, PAD_CHUNK_ID
from corpus_reasoning.lib.olmo_flex_attention import build_roles


# ---------------------------------------------------------------------------
# Core op
# ---------------------------------------------------------------------------

def detach_kv(
    k: torch.Tensor, v: torch.Tensor, gold_mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Detach ``k``/``v`` at non-gold key positions; forward values unchanged.

    :param k, v: ``(B, S, n_kv_heads, head_dim)`` (the layout olmo-core's
        ``Attention.sdpa`` receives, post-RoPE / post-QK-norm).
    :param gold_mask: ``(B, S)`` bool -- ``True`` = keep gradient (gold doc / free),
        ``False`` = detach (distractor / pad).
    :returns: ``(k, v)`` with gradient severed at the ``False`` positions. The
        returned tensors equal the inputs elementwise, so the attention output (and
        hence the loss) is identical; only the backward graph differs.
    """
    # (B, S) -> (B, S, 1, 1) so it broadcasts over kv-heads and head_dim. Detaching
    # here (before GQA expansion in the backend) is correct: a key/value head shared
    # across query groups is severed for non-gold positions regardless of expansion.
    m = gold_mask.unsqueeze(-1).unsqueeze(-1)
    k = torch.where(m, k, k.detach())
    v = torch.where(m, v, v.detach())
    return k, v


# ---------------------------------------------------------------------------
# Gold-mask policy helper
# ---------------------------------------------------------------------------

def _normalize_gold_indices(
    gold_doc_indices: Union[Iterable[int], Sequence[Iterable[int]]], B: int
) -> List[Set[int]]:
    """Coerce the ``gold_doc_indices`` arg into a per-row list of int sets.

    Accepts either a single iterable of doc indices (applied to *every* row) or a
    per-row sequence of iterables (one per batch element).
    """
    items = list(gold_doc_indices)
    is_per_row = len(items) > 0 and all(
        isinstance(x, (list, tuple, set)) for x in items
    )
    if is_per_row:
        if len(items) != B:
            raise ValueError(
                f"per-row gold_doc_indices has {len(items)} rows, expected B={B}")
        return [set(int(i) for i in row) for row in items]
    shared = set(int(i) for i in items)
    return [set(shared) for _ in range(B)]


def build_gold_key_mask(
    input_ids: torch.Tensor,
    *,
    doc_start_id: int,
    doc_end_id: int,
    eos_id: int,
    gold_doc_indices: Union[Iterable[int], Sequence[Iterable[int]]],
) -> torch.Tensor:
    """Build the ``(B, S)`` bool gold mask from marker-wrapped ``input_ids``.

    ``True`` (keep gradient) for: tokens of any document whose index is in
    ``gold_doc_indices`` for that row, **plus** all FREE tokens (instruction prefix,
    query, answer -- these we always supervise/learn through). ``False`` (detach)
    for: distractor-document tokens and padding.

    Document indices follow :func:`scripts.lib.olmo_flex_attention.build_roles`
    (0-based, in stream order; the ``<doc_start>``/``<doc_end>`` markers belong to
    their document). ``gold_doc_indices`` may be a single set applied to all rows or
    a per-row sequence.
    """
    roles = build_roles(input_ids, doc_start_id, doc_end_id, eos_id, mode="chunked")
    B, S = roles.shape
    gold_sets = _normalize_gold_indices(gold_doc_indices, B)

    keep = roles == FREE_CHUNK_ID  # instruction / query / answer always learn
    for b in range(B):
        for gi in gold_sets[b]:
            keep[b] |= roles[b] == gi
    # PAD (-2) and distractor docs stay False.
    return keep.to(torch.bool)


# ---------------------------------------------------------------------------
# Per-example gold lookup by content fingerprint (training plumbing)
# ---------------------------------------------------------------------------
#
# The gold-document set is per-example and known only at tokenize time, but it must
# NOT enter the token stream (a distinct gold marker would let the model trivially
# identify the contradicting docs). The olmo-core data collator only stacks fixed
# keys (``input_ids`` / ``label_mask``), so a parallel array can't ride along without
# editing the data stack. Instead we key each example's gold set by a fingerprint of
# its (unpadded) token content, written to a sidecar at tokenize time and looked up in
# the forward pre-hook from ``input_ids`` -- which the hook already receives. Robust to
# the data loader's shuffling, and adds nothing to what the model sees.

def content_fingerprint(ids: Sequence[int]) -> str:
    """Stable fingerprint of an example's *content* token ids (everything up to and
    including the single real eos, i.e. excluding the right-pad region).

    Must be computed identically at tokenize time (over ``p_ids + o_ids``) and at
    train time (over ``input_ids[:first_eos+1]``). NumpyPaddedFSLDataset preserves the
    content verbatim and only right-pads, so the two slices are byte-identical.
    """
    return hashlib.sha1(",".join(map(str, ids)).encode()).hexdigest()


def content_fingerprint_from_row(ids: Sequence[int], eos_id: int) -> str:
    """Fingerprint a padded training row: slice to the first eos, then fingerprint."""
    ids = list(ids)
    try:
        end = ids.index(eos_id) + 1
    except ValueError:
        end = len(ids)
    return content_fingerprint(ids[:end])


def gold_chunks_from_gold_doc_indices(gold_doc_indices) -> Set[int]:
    """Map a row's ``gold_doc_indices`` to a set of 0-based document/chunk indices.

    Contradiction rows store pairs of **1-indexed** "Claim N" display ids (possibly
    nested, e.g. ``[[2, 8], [11, 17]]``); document order is preserved through rendering
    + wrapping, so wrapped-doc/chunk index == Claim id - 1 (verified end-to-end by
    smoke_olmo_gold_grad_contradiction.py). We flatten any nesting and subtract 1.
    """
    out: Set[int] = set()

    def _walk(x):
        if isinstance(x, (list, tuple)):
            for y in x:
                _walk(y)
        else:
            out.add(int(x) - 1)

    _walk(gold_doc_indices)
    return out


def make_fingerprint_gold_mask_fn(
    gold_table: Dict[str, Iterable[int]],
    *,
    doc_start_id: int,
    doc_end_id: int,
    eos_id: int,
    debug_once: bool = True,
):
    """Build a ``gold_mask_fn(input_ids)->(B,S) bool`` that looks up each row's gold
    doc set from ``gold_table`` (fingerprint -> chunk indices) and returns the keep
    mask. A row whose fingerprint is absent from the table is left **all-True** (no
    masking — trains normally), so a partial table degrades gracefully.

    :param gold_table: {content_fingerprint: [gold chunk indices]} from the sidecar.
    """
    table = {k: set(int(i) for i in v) for k, v in gold_table.items()}
    # The trainer's very first model() call is a synthetic warmup (data_loader.
    # get_mock_batch) whose fingerprint is intentionally absent -> it correctly misses
    # and trains unmasked. Real batches follow. We accumulate hits across calls and log
    # the first few so the steady-state hit-rate (should be ~1.0) is visible, not just
    # the mock warmup.
    state = {"calls": 0, "rows": 0, "hits": 0}

    def fn(input_ids: torch.Tensor) -> torch.Tensor:
        roles = build_roles(input_ids.cpu(), doc_start_id, doc_end_id, eos_id, mode="chunked")
        B, S = roles.shape
        keep = roles == FREE_CHUNK_ID  # instruction / query / answer always learn
        ids2d = input_ids.tolist()
        n_found = 0
        for b in range(B):
            fp = content_fingerprint_from_row(ids2d[b], eos_id)
            gold = table.get(fp)
            if gold is None:
                keep[b] = True  # unknown example (e.g. warmup mock) -> no masking
                continue
            n_found += 1
            for gi in gold:
                keep[b] |= roles[b] == gi
        state["calls"] += 1
        state["rows"] += B
        state["hits"] += n_found
        if debug_once and state["calls"] <= 12:
            kept = int(keep.sum().item())
            tag = " (warmup mock)" if state["calls"] == 1 else ""
            print(f"[gold-grad] call#{state['calls']}{tag}: B={B} S={S} "
                  f"fp_hits={n_found}/{B} cum_hits={state['hits']}/{state['rows']} "
                  f"detached={B * S - kept}/{B * S}", flush=True)
        return keep.to(torch.bool)

    return fn


# ---------------------------------------------------------------------------
# Runtime installer
# ---------------------------------------------------------------------------

@dataclass
class GoldGradMaskHolder:
    """Shared per-step holder: the pre-hook sets ``gold_mask`` each forward; every
    patched ``sdpa`` reads it."""
    gold_mask: Optional[torch.Tensor] = None
    n_patched: int = 0


# A function that maps a batch's ``input_ids`` (B, S) -> ``(B, S)`` bool gold mask.
GoldMaskFn = Callable[[torch.Tensor], torch.Tensor]


def install_gold_grad_mask(
    model: torch.nn.Module, gold_mask_fn: GoldMaskFn
) -> GoldGradMaskHolder:
    """Install gold-document gradient masking on ``model`` in place.

    Registers a ``forward_pre_hook`` on the top-level model that computes the gold
    mask from the batch's ``input_ids`` via ``gold_mask_fn`` and stashes it, and
    monkeypatches each :class:`olmo_core.nn.attention.Attention` instance's bound
    ``sdpa`` to :func:`detach_kv` non-gold ``k``/``v`` before the attention backend.

    ``gold_mask_fn`` keeps *policy* (which tokens are gold) out of the mechanism --
    e.g. ``lambda ids: build_gold_key_mask(ids, ..., gold_doc_indices=...)``. It is
    called every forward with the live ``input_ids`` and must return a ``(B, S)``
    bool tensor (placed on any device; it is moved to the ``input_ids`` device here).

    Composes with the FlexAttention installer: if ``sdpa`` was already patched
    (custom mask pattern), call this *after* and the detach wraps the flex ``sdpa``.

    :returns: the :class:`GoldGradMaskHolder` (records how many modules were patched).
    """
    from olmo_core.nn.attention import Attention

    holder = GoldGradMaskHolder()

    def pre_hook(module, args, kwargs):
        input_ids = kwargs.get("input_ids", args[0] if args else None)
        if input_ids is None:
            return
        gm = gold_mask_fn(input_ids)
        holder.gold_mask = gm.to(device=input_ids.device, dtype=torch.bool)

    model.register_forward_pre_hook(pre_hook, with_kwargs=True)

    def make_gated_sdpa(orig_sdpa):
        # orig_sdpa is the module's *bound* method (possibly already flex-patched);
        # we ignore the rebound `self` and call it directly.
        def gated_sdpa(self, q, k, v, **kwargs):
            gm = holder.gold_mask
            if gm is not None:
                k, v = detach_kv(k, v, gm)
            return orig_sdpa(q, k, v, **kwargs)

        return gated_sdpa

    n = 0
    for m in model.modules():
        if isinstance(m, Attention):
            m.sdpa = types.MethodType(make_gated_sdpa(m.sdpa), m)
            n += 1
    holder.n_patched = n
    return holder
