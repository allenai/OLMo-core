"""
Prefill once, decode many: KV reuse across queries that share a token prefix.

A 32k rung costs 500 full 32k prefills (8,885 s measured on one H200), and if those 500 prompts
share a leading block of documents, that block can be prefilled once and its KV reused. This module
is that optimisation, and **only** that -- it changes where a prompt's KV comes from, never how
tokens are chosen. Stop rules, sampling and post-processing stay in the evaluator via ``decode_fn``.

.. important::
   **Two separate things get called "shared-corpus eval", and only one of them is safe.**

   *This module* -- reusing KV across prompts that already share a token prefix -- is score
   preserving by construction, and was verified to produce byte-identical generations
   (``records/shared-corpus-eval-results.md``, 2026-08-07).

   *Rebuilding the eval set* so that many queries share one corpus is a different thing, and it
   changes the task. The measured cost of the "prefix+tail" construction was **+0.215 / +0.261** on
   outlier and **−0.102 … −0.175** on contradiction. A scatter control isolated the cause to
   document *placement*, not to the sharing. Do not conflate the two; the second is not ported.

.. warning::
   **Shared documents are not shared tokens. Measure before quoting a speedup.** These checkpoints
   train with ``query_position="both"``, rendering ``{questions}\\n\\n{documents}\\n\\n{questions}``,
   so the per-query question precedes the corpus and a byte-identical document block is *not* a
   token prefix. Measured reusable prefix: nq 0.5%, rerank 0.9%, oolong 0.1% -- against
   contradiction 95.0% and outlier 94.7%, and those two only because their question block is empty
   or a fixed string. :func:`measure_reuse` is the check, it needs no GPU, and
   :func:`generate_group_with_shared_prefix` falls back to plain prefills when there is nothing to
   reuse.

   The obvious fix is not available: ``query_position="after"`` makes the corpus a true prefix and
   **collapses the model** (nq 0.860 -> 0.074). This is the same decision as the open
   ``query_position`` question for the suite -- prefill reuse is one of the things riding on it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Sequence, Tuple

__all__ = [
    "longest_common_token_prefix",
    "group_by_corpus",
    "ReuseEstimate",
    "measure_reuse",
    "generate_group_with_shared_prefix",
]

#: Below this many shared tokens the rewind bookkeeping costs more than the prefill it saves.
MIN_PREFIX_TOKENS = 64


def longest_common_token_prefix(sequences: Sequence[Sequence[int]]) -> int:
    """
    :param sequences: Tokenized prompts.

    :returns: Length of the longest token prefix every sequence shares.

    Measured at the *token* level rather than trusting the document boundary: tokenizer merges can
    span a boundary, so a prompt-level split can differ from the token-level one by a token or two.
    Measuring costs one pass and removes the whole class of bug.
    """
    if not sequences:
        return 0
    first = sequences[0]
    for i in range(min(len(s) for s in sequences)):
        if any(s[i] != first[i] for s in sequences):
            return i
    return min(len(s) for s in sequences)


def group_by_corpus(examples: Sequence[Mapping[str, Any]]) -> Dict[str, List[int]]:
    """
    Map ``corpus_id`` to row indices, preserving file order.

    :param examples: Eval examples.

    :returns: group id -> row indices. Rows with no ``corpus_id`` each become their own single-row
        group, so an ordinary independent file runs correctly through this path with nothing to
        share.
    """
    groups: Dict[str, List[int]] = {}
    for i, example in enumerate(examples):
        raw = example.get("ex", example) if isinstance(example, Mapping) else {}
        groups.setdefault(raw.get("corpus_id", f"__row{i}"), []).append(i)
    return groups


@dataclass(frozen=True)
class ReuseEstimate:
    """
    :param groups: Corpus groups examined.
    :param rows: Prompts examined.
    :param mean_prompt_tokens: Mean prompt length.
    :param mean_prefix_tokens: Mean shared prefix length within a group.
    :param fraction: Reusable share of the mean prompt.
    """

    groups: int
    rows: int
    mean_prompt_tokens: float
    mean_prefix_tokens: float
    fraction: float

    @property
    def speedup(self) -> float:
        """:returns: Best-case reduction in prefilled prompt tokens, as a multiple."""
        return 1.0 / max(1e-9, 1.0 - self.fraction)

    def __str__(self) -> str:
        verdict = "worth caching" if self.fraction >= 0.5 else "NOT worth caching"
        return (
            f"{self.groups} group(s), {self.rows} row(s): mean prompt "
            f"{self.mean_prompt_tokens:.0f} tok, shared prefix {self.mean_prefix_tokens:.0f} tok "
            f"({self.fraction:.1%} reusable, up to {self.speedup:.1f}x fewer prefill tokens) "
            f"-- {verdict}"
        )


def measure_reuse(groups: Sequence[Sequence[Sequence[int]]]) -> ReuseEstimate:
    """
    Report how much of a tokenized eval is actually a shared token prefix. CPU only.

    Run this before quoting any speedup. It is the difference between "these rows share a corpus"
    and "these rows share tokens", and the two came apart by two orders of magnitude across the
    tasks measured.

    :param groups: One list of tokenized prompts per corpus group.

    :returns: The estimate.
    """
    non_empty = [g for g in groups if g]
    if not non_empty:
        return ReuseEstimate(0, 0, 0.0, 0.0, 0.0)
    prefixes = [longest_common_token_prefix(g) for g in non_empty]
    lengths = [len(p) for g in non_empty for p in g]
    mean_prompt = sum(lengths) / len(lengths)
    mean_prefix = sum(prefixes) / len(prefixes)
    return ReuseEstimate(
        groups=len(non_empty),
        rows=len(lengths),
        mean_prompt_tokens=mean_prompt,
        mean_prefix_tokens=mean_prefix,
        fraction=mean_prefix / mean_prompt if mean_prompt else 0.0,
    )


# ── the GPU half ────────────────────────────────────────────────────────────────────────────────


def _mixers(model):
    for block in model.blocks.values():
        yield block.attention


def snapshot_recurrent_states(model) -> List[Tuple[Any, Dict[str, Any]]]:
    """
    Clone every recurrent mixer's state so it can be rewound.

    Qwen3.5 is a hybrid and the two block types need different treatment. Softmax-attention blocks
    write into a preallocated KV buffer at ``cache_seqlens``, so rewinding is just resetting that
    cursor -- positions ``[0, L)`` still hold the prefix's own K/V and no later write touches them.
    GatedDeltaNet blocks instead carry a constant-size recurrent and conv state that *every* token
    mutates, so after one query's suffix that state is polluted and must be copied back.

    Rewinding the KV cursor and forgetting the GDN state silently feeds query N+1 a state that has
    already consumed query N -- no error, just quietly wrong generations for every query after the
    first in each group.

    :param model: The model being evaluated.

    :returns: Opaque snapshots for :func:`restore_recurrent_states`.
    """
    from olmo_core.nn.attention import Attention

    snapshots = []
    for mixer in _mixers(model):
        cache = getattr(mixer, "state_cache", None)
        if cache is None or isinstance(mixer, Attention):
            continue
        recurrent = getattr(cache, "recurrent_state", None)
        snapshots.append(
            (
                cache,
                {
                    "conv_state_q": cache.conv_state_q.clone(),
                    "conv_state_k": cache.conv_state_k.clone(),
                    "conv_state_v": cache.conv_state_v.clone(),
                    "recurrent_state": None if recurrent is None else recurrent.clone(),
                    "has_state": bool(getattr(cache, "has_state", False)),
                },
            )
        )
    return snapshots


def restore_recurrent_states(snapshots: Sequence[Tuple[Any, Dict[str, Any]]]) -> None:
    """
    Rewind every recurrent mixer to its snapshot.

    Copies rather than aliases: the next query's suffix mutates these tensors in place, so handing
    back a reference would let query N corrupt the snapshot for query N+1.

    :param snapshots: From :func:`snapshot_recurrent_states`.
    """
    for cache, saved in snapshots:
        cache.conv_state_q.copy_(saved["conv_state_q"])
        cache.conv_state_k.copy_(saved["conv_state_k"])
        cache.conv_state_v.copy_(saved["conv_state_v"])
        recurrent = saved["recurrent_state"]
        cache.recurrent_state = None if recurrent is None else recurrent.clone()
        cache.has_state = saved["has_state"]


def rewind_kv_cursor(model, length: int) -> None:
    """
    Point every KV cache back at ``length``, keeping the prefix K/V already written there.

    :param model: The model being evaluated.
    :param length: The shared prefix length.
    """
    for mixer in _mixers(model):
        manager = getattr(mixer, "kv_cache_manager", None)
        if manager is not None:
            manager.cache_seqlens.fill_(length)


def generate_group_with_shared_prefix(
    gm,
    prefills: Sequence[Sequence[int]],
    device,
    max_length: int,
    decode_fn: Callable[[Any], str],
    min_prefix: int = MIN_PREFIX_TOKENS,
) -> Tuple[List[str], Dict[str, float]]:
    """
    Generate for one corpus group, prefilling the common token prefix exactly once.

    :param gm: The generation module (exposes ``model`` and ``prepare_inference_cache``).
    :param prefills: Tokenized prompts for the group.
    :param device: Torch device.
    :param max_length: Inference cache size.
    :param decode_fn: Given the logits of the last prompt token, run the task's own decode loop and
        return the text. Passing this in is what keeps every stop rule and post-processing detail
        in the evaluator, so this path cannot diverge from the plain one on anything but KV origin.
    :param min_prefix: Below this many shared tokens, fall back to plain per-example prefills.

    :returns: ``(texts in the order of prefills, stats)``.
    """
    import torch

    stats: Dict[str, float] = {"prefix_len": 0, "prefill_tokens": 0, "suffix_tokens": 0}
    shared = longest_common_token_prefix(prefills)

    if shared < min_prefix or len(prefills) < 2:
        texts = []
        with torch.no_grad():
            for ids in prefills:
                gm.prepare_inference_cache(1, max_length)
                leftpad = torch.zeros(1, dtype=torch.int32, device=device)
                logits = gm.model(
                    torch.tensor([ids], device=device), logits_to_keep=1, cache_leftpad=leftpad
                )
                texts.append(decode_fn(logits))
                stats["prefill_tokens"] += len(ids)
        stats["shared"] = 0.0
        return texts, stats

    with torch.no_grad():
        # Prefill the shared prefix once. The returned logits belong to prefix token L-1 and are
        # deliberately discarded: the first generated token must be predicted from the LAST token
        # of each query's own suffix, not from the end of the shared block.
        started = time.time()
        gm.prepare_inference_cache(1, max_length)
        leftpad = torch.zeros(1, dtype=torch.int32, device=device)
        gm.model(
            torch.tensor([prefills[0][:shared]], device=device),
            logits_to_keep=1,
            cache_leftpad=leftpad,
        )
        snapshots = snapshot_recurrent_states(gm.model)
        stats["prefix_len"] = shared
        stats["prefill_tokens"] = shared
        stats["prefill_seconds"] = time.time() - started

        texts = []
        for ids in prefills:
            rewind_kv_cursor(gm.model, shared)
            restore_recurrent_states(snapshots)
            suffix = list(ids[shared:])
            stats["suffix_tokens"] += len(suffix)
            logits = gm.model(torch.tensor([suffix], device=device), logits_to_keep=1)
            texts.append(decode_fn(logits))

    naive = sum(len(ids) for ids in prefills)
    total = stats["prefill_tokens"] + stats["suffix_tokens"]
    stats["shared"] = 1.0 - (total / naive) if naive else 0.0
    return texts, stats
