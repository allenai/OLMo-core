"""
Gold-document gradient masking -- make the SFT **backward pass O(1)** in the number of context
documents by detaching the K/V of every document *except* the ground-truth (gold) documents plus a
few random ones.

Goal
----
Keep every document in the **forward** pass (so the loss is computed over the full context exactly as
usual), but make sure **only a small selected set of documents** (the gold / ground-truth
contradiction docs, plus optionally a handful of random distractors) -- together with the free
instruction / query / answer tokens we actually supervise -- drive the weight update. The remaining
distractor-document tokens become *read-only context*: they influence the forward output, but receive
and contribute **zero** gradient, so autograd never traverses their per-layer sub-graphs. Because the
``"chunked"`` document mask makes each context document an **isolated tower** (it attends only within
itself; only FREE tokens bridge across documents), severing a document's K/V at every layer's FREE
read fully cuts its path to the loss -> its backward is skipped. Backward compute then scales with the
number of *selected* documents, not the number of documents in the context.

Why detach K and V (and nothing else)
-------------------------------------
A context token at position ``t`` influences the loss **only by being attended to** -- i.e. only
through its key ``K_t`` and value ``V_t``. To zero a non-selected token's contribution we need its
output-side gradient to be zero, *not* its activation (zeroing the activation would corrupt the
forward pass). Detaching the input embeddings fails (the *read* side-door still delivers gradient to
``W_K`` / ``W_V`` because selected/output queries attend to ``t``); masking the residual-stream
gradient fails (``dL/dK_t`` / ``dL/dV_t`` arrive from the querying positions, upstream of any
residual gate). The exact tensor to cut is ``K`` / ``V`` themselves, at **every** layer.

Mechanism
---------
``torch.where(keep[:, :, None, None], k, k.detach())``: the forward value is ``k`` everywhere (both
branches are equal), but in backward gradient reaches ``k`` only at kept key positions. Applied to the
``k`` / ``v`` tensors right before they enter the attention backend, so it is backend-agnostic
(SDPA / flash / flex) and needs no kernel changes -- installed at each
:class:`~olmo_core.nn.attention.Attention` instance's bound ``sdpa`` seam (RoPE, QK-norm and gating
are all upstream and untouched).

Non-leaky per-example gold lookup
---------------------------------
The gold-document set is per-example and known only at tokenize time, but it must **not** enter the
token stream (a distinct gold marker would let the model trivially identify the contradicting docs).
The data collator only stacks fixed keys (``input_ids`` / ``label_mask``), so a parallel array can't
ride along without editing the data stack. Instead each example's gold set is keyed by a fingerprint
of its (unpadded) token content, written to a sidecar at tokenize time and looked up in a forward
pre-hook from ``input_ids`` -- robust to shuffling, and adding nothing to what the model sees.

Scope: single-rank, full-sequence (CP=1). Run with ``compile_model=False`` (the per-forward Python
fingerprint / RNG is not ``torch.compile``-capturable, exactly like the mask-mix curriculum).
"""

from __future__ import annotations

import hashlib
import random
import types
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Union

import torch

from .chunked_mask import FREE_CHUNK_ID, build_chunk_ids_from_tokens

__all__ = [
    "detach_kv",
    "content_fingerprint",
    "content_fingerprint_from_row",
    "gold_chunks_from_gold_doc_indices",
    "select_keep_docs",
    "make_fingerprint_gold_mask_fn",
    "install_gold_grad_mask",
    "GoldGradMaskHolder",
]


# ---------------------------------------------------------------------------
# Core op
# ---------------------------------------------------------------------------


def detach_kv(
    k: torch.Tensor, v: torch.Tensor, keep_mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Detach ``k`` / ``v`` at non-kept key positions; forward values unchanged.

    :param k: Keys, shape ``(B, S, n_kv_heads, head_dim)`` (the layout an
        :class:`~olmo_core.nn.attention.Attention` ``sdpa`` receives, post-RoPE / post-QK-norm).
    :param v: Values, same shape as ``k``.
    :param keep_mask: ``(B, S)`` bool -- ``True`` = keep gradient (selected doc / free token),
        ``False`` = detach (non-selected distractor / pad).

    :returns: ``(k, v)`` with gradient severed at the ``False`` positions. The returned tensors equal
        the inputs elementwise, so the attention output (and hence the loss) is identical; only the
        backward graph differs.
    """
    # (B, S) -> (B, S, 1, 1) so it broadcasts over kv-heads and head_dim. Detaching here (before GQA
    # expansion in the backend) is correct: a kv-head shared across query groups is severed for
    # non-kept positions regardless of expansion.
    m = keep_mask.unsqueeze(-1).unsqueeze(-1)
    # ``.contiguous()``: flash-attn's backward reads k/v (and their grads) with hard assumptions about
    # memory layout. ``torch.where`` may hand back a tensor whose layout differs from the one the
    # backend was given, which the CUDA kernel does not validate -- it just reads out of bounds and
    # SIGSEGVs (observed deterministically at seq 6144; never at 2048). Forcing the standard contiguous
    # layout costs one copy of k/v per layer and changes no value.
    k = torch.where(m, k, k.detach()).contiguous()
    v = torch.where(m, v, v.detach()).contiguous()
    return k, v


# ---------------------------------------------------------------------------
# Per-example gold lookup by content fingerprint
# ---------------------------------------------------------------------------


def content_fingerprint(ids: Sequence[int]) -> str:
    """
    Stable fingerprint of an example's *content* token ids (everything up to and including the single
    real EOS, i.e. excluding the right-pad region).

    Must be computed identically at tokenize time (over the emitted ``token_ids`` incl. the trailing
    EOS) and at train time (over ``input_ids[:first_eos+1]``). The padded reader preserves the content
    verbatim and only right-pads, so the two slices are byte-identical.
    """
    return hashlib.sha1(",".join(map(str, (int(i) for i in ids))).encode()).hexdigest()


def content_fingerprint_from_row(ids: Sequence[int], eos_id: int) -> str:
    """Fingerprint a padded training row: slice to the first EOS (inclusive), then fingerprint."""
    ids = list(ids)
    try:
        end = ids.index(eos_id) + 1
    except ValueError:
        end = len(ids)
    return content_fingerprint(ids[:end])


def gold_chunks_from_gold_doc_indices(gold_doc_indices) -> Set[int]:
    """
    Map a row's ``gold_doc_indices`` to a set of 0-based document/chunk indices.

    Contradiction rows store pairs of **1-indexed** "Claim N" display ids (possibly nested, e.g.
    ``[[2, 8], [11, 17]]``); document order is preserved through rendering + wrapping, so
    wrapped-doc / chunk index == ``Claim id - 1``. We flatten any nesting and subtract 1.
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


# ---------------------------------------------------------------------------
# Keep-set policy
# ---------------------------------------------------------------------------


def select_keep_docs(
    present_docs: Sequence[int],
    gold_docs: Set[int],
    *,
    n_random: int,
    mode: str,
    rng: random.Random,
    n_gold: int = 0,
    gold_pairs: Optional[Sequence[Sequence[int]]] = None,
    n_pairs: int = 1,
) -> Set[int]:
    """
    Choose the set of context-document indices that keep gradient this example.

    :param present_docs: The document indices actually present in the row (unique ``chunk_ids >= 0``).
    :param gold_docs: The ground-truth (gold) document indices for the row.
    :param n_random: Number of *extra* random documents to keep beyond the forced set.
    :param mode: One of:

        * ``"gold_plus_random"`` -- keep **every** gold doc plus ``n_random`` random non-gold docs.
          O(1) in N, but over-represents positives vs the true base rate (see ``"gold_subsample"``).
        * ``"gold_subsample"`` -- keep ``n_gold`` *randomly chosen* gold docs plus ``n_random`` random
          non-gold docs. O(1) in N **and** base-rate preserving.
        * ``"random_only"`` -- keep ``len(gold_docs) + n_random`` docs drawn from **all** docs. NB this
          keeps gold BY CHANCE (~``|gold|*n_keep/|present|``), so it is a same-sparsity control, *not*
          a gold-free one.
        * ``"random_nongold"`` -- STRICT gold-free control: same sparsity, sampled only from non-gold.
        * ``"gold_pair"`` -- keep ``n_pairs`` COMPLETE gold pairs (both halves of each) plus
          ``n_random`` random non-gold docs. Requires ``gold_pairs``.
        * ``"gold_halves"`` -- the matched control for ``"gold_pair"``: keep ``n_gold`` gold docs drawn
          from ``n_gold`` DISTINCT pairs (i.e. orphaned halves, never a complete pair) plus
          ``n_random`` random non-gold. With ``n_gold=2`` vs ``gold_pair`` ``n_pairs=1`` the two arms
          hold the gold COUNT, the doc count and the gold fraction fixed and differ *only* in whether
          the two gold docs form a contradicting pair.

    :param rng: A seeded :class:`random.Random` (seed per-example so the choice is stable across
        epochs and reproducible).
    :param n_gold: ``"gold_subsample"`` / ``"gold_halves"`` only -- how many gold docs to keep.
    :param gold_pairs: The row's gold docs grouped into contradicting pairs, e.g.
        ``[[6, 19], [18, 48], [32, 72]]``. Required by the pair-aware modes.
    :param n_pairs: ``"gold_pair"`` only -- how many complete pairs to keep.

    :returns: The set of document indices to keep (a subset of ``present_docs``).

    .. note::
        The pair-aware modes exist because the label is a *relation* (the model must emit
        ``[[9, 28], ...]``), yet every doc-level mode treats gold as an unordered set -- so
        ``gold_subsample`` with ``n_gold=1`` keeps one half of a pair and detaches its partner, and
        gradient never flows through both ends of a contradiction at once.
    """
    present = [d for d in present_docs if d >= 0]
    present_set = set(present)
    if mode == "gold_plus_random":
        keep = {g for g in gold_docs if g in present_set}
        pool = sorted(present_set - keep)
        rng.shuffle(pool)
        keep.update(pool[: max(0, n_random)])
        return keep
    if mode == "random_only":
        n_keep = min(
            len(present), len([g for g in gold_docs if g in present_set]) + max(0, n_random)
        )
        pool = sorted(present_set)
        rng.shuffle(pool)
        return set(pool[:n_keep])
    if mode == "gold_subsample":
        # O(1) *and* base-rate preserving: keep a CONSTANT, SMALL number of gold docs (n_gold, drawn at
        # random from the gold set) plus n_random random NON-gold docs.
        #
        # Why not keep ALL gold (= "gold_plus_random"): the gold docs are a biased subpopulation (the
        # contradicting claims). Forcing every one of them into the backward pass over-represents
        # positives relative to the true base rate (6:14 at n20), and the model's retrieval comes out
        # miscalibrated -- empirically gold_plus_random lost to an *ignorant* uniform sample at the same
        # negative count (F1 0.833 @ 6 gold + 6 neg vs 0.950 @ ~2.3 gold + ~5.7 neg). Keeping a small
        # constant n_gold restores the natural positive:negative ratio while staying O(1) in N.
        gold_present = sorted(g for g in gold_docs if g in present_set)
        rng.shuffle(gold_present)
        keep = set(gold_present[: max(0, n_gold)])
        pool = sorted(present_set - set(gold_docs))
        rng.shuffle(pool)
        keep.update(pool[: max(0, n_random)])
        return keep
    if mode in ("gold_pair", "gold_halves"):
        if not gold_pairs:
            raise ValueError(f"mode {mode!r} needs gold_pairs (the pair-preserving sidecar)")
        # Only pairs whose BOTH halves are present can be kept intact.
        pairs = [[int(a) for a in p] for p in gold_pairs if all(int(a) in present_set for a in p)]
        rng.shuffle(pairs)
        keep: Set[int] = set()
        if mode == "gold_pair":
            for p in pairs[: max(0, n_pairs)]:
                keep.update(p)
        else:  # gold_halves: one orphan from each of n_gold DISTINCT pairs -> never a complete pair
            for p in pairs[: max(0, n_gold)]:
                keep.add(p[rng.randrange(len(p))])
        pool = sorted(present_set - set(gold_docs))
        rng.shuffle(pool)
        keep.update(pool[: max(0, n_random)])
        return keep
    if mode == "random_nongold":
        # STRICT control: sample the same NUMBER of docs but ONLY from non-gold ones, so the gold
        # gradient is exactly zero. Needed because "random_only" draws from ALL docs and therefore
        # keeps ~len(gold)*n_keep/len(present) gold docs BY CHANCE (with 6 gold of 20 and 8 kept that
        # is ~2.4 gold docs/example, i.e. it is NOT a gold-free arm) -- so "random_only" cannot, on its
        # own, tell us whether knowing the true contradiction is what matters.
        n_keep = min(
            len(present), len([g for g in gold_docs if g in present_set]) + max(0, n_random)
        )
        pool = sorted(present_set - set(gold_docs))
        rng.shuffle(pool)
        return set(pool[:n_keep])
    raise ValueError(
        f"Unknown gold-grad mode {mode!r}; expected 'gold_plus_random', 'gold_subsample', "
        "'random_only', 'random_nongold', 'gold_pair' or 'gold_halves'."
    )


# ---------------------------------------------------------------------------
# Fingerprint -> keep-mask function
# ---------------------------------------------------------------------------

# A function that maps a batch's ``input_ids`` (B, S) -> ``(B, S)`` bool keep mask.
GoldMaskFn = Callable[[torch.Tensor], torch.Tensor]


def make_fingerprint_gold_mask_fn(
    gold_table: Dict[str, Iterable[int]],
    *,
    doc_start_id: int,
    doc_end_id: int,
    eos_id: int,
    n_random: int = 0,
    mode: str = "gold_plus_random",
    seed: int = 0,
    debug_calls: int = 12,
    n_gold: int = 0,
    n_pairs: int = 1,
) -> GoldMaskFn:
    """
    Build a ``keep_mask_fn(input_ids) -> (B, S) bool`` that looks up each row's gold document set from
    ``gold_table`` (fingerprint -> chunk indices), applies the :func:`select_keep_docs` policy, and
    returns the per-token keep mask (``True`` = keep gradient).

    A row whose fingerprint is absent from the table is left **all-True** (trains normally / no
    masking), so a partial table -- and the trainer's synthetic warmup mock batch -- degrade
    gracefully.

    :param gold_table: ``{content_fingerprint: [gold chunk indices]}`` from the sidecar.
    :param doc_start_id: ``<|box_start|>`` id (for role reconstruction).
    :param doc_end_id: ``<|box_end|>`` id.
    :param eos_id: EOS id (row fingerprint slice + PAD boundary).
    :param n_random: Extra random docs to keep (see :func:`select_keep_docs`).
    :param mode: ``"gold_plus_random"`` or ``"random_only"``.
    :param seed: Base seed for the per-example random doc choice.
    :param debug_calls: Log a ``[gold-grad]`` line for the first this-many forwards.
    """
    # A sidecar value is either a FLAT list of gold doc ids ([6, 18, 19, ...]) or a list of
    # contradicting PAIRS ([[6, 19], [18, 48], ...]). Accept both: the doc-level modes only need the
    # flat set, the pair-aware modes need the grouping.
    table: Dict[str, Set[int]] = {}
    pair_table: Dict[str, List[List[int]]] = {}
    for k, v in gold_table.items():
        items = list(v)
        if items and isinstance(items[0], (list, tuple)):
            pair_table[k] = [[int(a) for a in p] for p in items]  # type: ignore[union-attr]
            table[k] = {int(a) for p in items for a in p}  # type: ignore[union-attr]
        else:
            table[k] = {int(i) for i in items}  # type: ignore[arg-type]
    if mode in ("gold_pair", "gold_halves") and not pair_table:
        raise ValueError(
            f"mode {mode!r} needs a PAIR-preserving gold sidecar (values like [[6, 19], [18, 48]]); "
            "the flat gold_fingerprints.json cannot express which docs contradict which."
        )
    state = {"calls": 0, "rows": 0, "hits": 0}

    def fn(input_ids: torch.Tensor) -> torch.Tensor:
        ids_cpu = input_ids.detach().to("cpu")
        roles = build_chunk_ids_from_tokens(
            ids_cpu, doc_start_id=doc_start_id, doc_end_id=doc_end_id, eos_id=eos_id, mode="chunked"
        )
        B, S = roles.shape
        keep = roles == FREE_CHUNK_ID  # instruction / query / answer always learn
        ids2d = ids_cpu.tolist()
        n_found = 0
        for b in range(B):
            fp = content_fingerprint_from_row(ids2d[b], eos_id)
            gold = table.get(fp)
            if gold is None:
                keep[b] = True  # unknown example (e.g. warmup mock) -> no masking
                continue
            n_found += 1
            row_roles = roles[b]
            present_docs = [int(d) for d in torch.unique(row_roles[row_roles >= 0]).tolist()]
            rng = random.Random(f"{seed}:{fp}")
            keep_docs = select_keep_docs(
                present_docs,
                gold,
                n_random=n_random,
                mode=mode,
                rng=rng,
                n_gold=n_gold,
                gold_pairs=pair_table.get(fp),
                n_pairs=n_pairs,
            )
            for gi in keep_docs:
                keep[b] |= row_roles == gi
        state["calls"] += 1
        state["rows"] += B
        state["hits"] += n_found
        if state["calls"] <= debug_calls:
            kept = int(keep.sum().item())
            tag = " (warmup mock)" if n_found == 0 and state["calls"] == 1 else ""
            print(
                f"[gold-grad] call#{state['calls']}{tag}: B={B} S={S} mode={mode} "
                f"n_random={n_random} fp_hits={n_found}/{B} cum_hits={state['hits']}/{state['rows']} "
                f"detached={B * S - kept}/{B * S}",
                flush=True,
            )
        return keep.to(torch.bool)

    return fn


# ---------------------------------------------------------------------------
# Runtime installer
# ---------------------------------------------------------------------------


@dataclass
class GoldGradMaskHolder:
    """
    Shared per-step holder: the forward pre-hook sets ``keep_mask`` each forward; every patched
    ``sdpa`` reads it. Also records how many attention modules were patched.
    """

    keep_mask: Optional[torch.Tensor] = None
    n_patched: int = 0
    _hook_handle: object = field(default=None, repr=False)


def install_gold_grad_mask(model: torch.nn.Module, keep_mask_fn: GoldMaskFn) -> GoldGradMaskHolder:
    """
    Install gold-document gradient masking on ``model`` in place.

    Registers a ``forward_pre_hook`` on the top-level model that computes the per-token keep mask from
    the batch's ``input_ids`` via ``keep_mask_fn`` and stashes it, and monkeypatches each
    :class:`~olmo_core.nn.attention.Attention` instance's bound ``sdpa`` to :func:`detach_kv` the
    non-kept ``k`` / ``v`` before the attention backend.

    ``keep_mask_fn`` keeps *policy* (which tokens keep gradient) out of the mechanism -- e.g.
    :func:`make_fingerprint_gold_mask_fn`. It is called every forward with the live ``input_ids`` and
    must return a ``(B, S)`` bool tensor (placed on any device; it is moved to the ``input_ids``
    device here).

    Composes with any pre-existing ``sdpa`` patch (custom mask pattern / document-chunked mask): the
    detach simply wraps whatever ``sdpa`` was bound.

    :returns: the :class:`GoldGradMaskHolder` (records how many modules were patched).
    """
    from . import Attention

    holder = GoldGradMaskHolder()

    def pre_hook(module, args, kwargs):
        input_ids = kwargs.get("input_ids", args[0] if args else None)
        if input_ids is None:
            return
        km = keep_mask_fn(input_ids)
        holder.keep_mask = km.to(device=input_ids.device, dtype=torch.bool)

    holder._hook_handle = model.register_forward_pre_hook(pre_hook, with_kwargs=True)

    def make_gated_sdpa(orig_sdpa):
        # orig_sdpa is the module's *bound* method (possibly already patched); ignore the rebound
        # ``self`` and call it directly.
        def gated_sdpa(self, q, k, v, **kwargs):
            km = holder.keep_mask
            if km is not None:
                k, v = detach_kv(k, v, km)
            return orig_sdpa(q, k, v, **kwargs)

        return gated_sdpa

    n = 0
    for m in model.modules():
        if isinstance(m, Attention):
            m.sdpa = types.MethodType(make_gated_sdpa(m.sdpa), m)
            n += 1
    holder.n_patched = n
    return holder
