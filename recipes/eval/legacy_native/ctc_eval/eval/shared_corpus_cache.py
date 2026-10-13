"""
Prefix-KV reuse across queries that share a corpus. EVAL ONLY.

A shared-corpus eval file (``corpus_id`` / ``shared_prefix_len`` / ``shared_prefix_sha1``, built by
``corpus_reasoning.data.build_shared_corpus_evals``) guarantees that every row in a corpus group
begins with a byte-identical block of documents. Because the alpaca template puts the fixed
instruction and then the documents *before* the query
(``...### Instruction:\\n{instruction}\\n\\n### Input:\\n{docs}\\n\\n{query}\\n\\n### Response:\\n``),
that block is a true **token prefix** of every prompt in the group -- so it can be prefilled once
and its KV reused for all of them.

The saving is exactly the thing that makes these rungs expensive: a 32k rung costs 500 full 32k
prefills today (8,885 s measured on one H200), and the shared form collapses that to one prefill
per corpus plus a short per-query suffix.

Two details make this exact rather than approximate:

**The reuse point is measured, not assumed.** We take the longest common *token* prefix of the
group's tokenized prompts rather than trusting the document boundary. Tokenizer merges can span a
document boundary, so a prompt-level split could differ from the token-level one by a token or two;
measuring removes that whole class of bug and costs one pass over the ids.

**Qwen3.5 is a hybrid, so two kinds of state must be rewound.** Softmax-attention blocks write into
a preallocated KV buffer at ``cache_seqlens``, so rewinding is just resetting that cursor to L --
positions ``[0, L)`` are still the prefix's own K/V and no later write touches them. GatedDeltaNet
blocks instead carry a constant-size recurrent + conv state that *every* token mutates, so after
one query's suffix that state is polluted; those must be snapshotted after the prefix and copied
back before each query. Rewinding only the KV cursor and forgetting the GDN state would silently
feed query N+1 a state that had already consumed query N's suffix.
"""

from __future__ import annotations

import time
from typing import Callable, Dict, List, Sequence, Tuple

import torch

from olmo_core.nn.attention import Attention


def _mixers(model):
    for block in model.blocks.values():
        yield block.attention


def longest_common_token_prefix(seqs: Sequence[Sequence[int]]) -> int:
    """Length of the longest token prefix shared by every sequence."""
    if not seqs:
        return 0
    shortest = min(len(s) for s in seqs)
    first = seqs[0]
    for i in range(shortest):
        tok = first[i]
        if any(s[i] != tok for s in seqs):
            return i
    return shortest


def snapshot_recurrent_states(model) -> List[Tuple[object, Dict[str, object]]]:
    """Clone every recurrent mixer's state so it can be rewound to this point.

    Attention blocks need no snapshot -- their prefix K/V stays put in the preallocated buffer and
    is rewound by :func:`rewind_kv_cursor` alone.
    """
    snaps = []
    for mixer in _mixers(model):
        sc = getattr(mixer, "state_cache", None)
        if sc is None or isinstance(mixer, Attention):
            continue
        rs = getattr(sc, "recurrent_state", None)
        snaps.append((sc, {
            "conv_state_q": sc.conv_state_q.clone(),
            "conv_state_k": sc.conv_state_k.clone(),
            "conv_state_v": sc.conv_state_v.clone(),
            "recurrent_state": None if rs is None else rs.clone(),
            "has_state": bool(getattr(sc, "has_state", False)),
        }))
    return snaps


def restore_recurrent_states(snaps) -> None:
    """Rewind every recurrent mixer to its snapshotted state (copy, not alias -- the next query's
    suffix will mutate these tensors in place)."""
    for sc, s in snaps:
        sc.conv_state_q.copy_(s["conv_state_q"])
        sc.conv_state_k.copy_(s["conv_state_k"])
        sc.conv_state_v.copy_(s["conv_state_v"])
        rs = s["recurrent_state"]
        sc.recurrent_state = None if rs is None else rs.clone()
        sc.has_state = s["has_state"]


def rewind_kv_cursor(model, length: int) -> None:
    """Point every KV cache back at ``length``, keeping the prefix K/V already written there."""
    for mixer in _mixers(model):
        mgr = getattr(mixer, "kv_cache_manager", None)
        if mgr is not None:
            mgr.cache_seqlens.fill_(length)


def group_by_corpus(examples) -> "Dict[str, List[int]]":
    """Map ``corpus_id`` -> row indices, preserving file order.

    Rows with no ``corpus_id`` each become their own single-row group, so a plain independent file
    still runs correctly through this path (just with no sharing to exploit).
    """
    groups: Dict[str, List[int]] = {}
    for i, ex in enumerate(examples):
        raw = ex.get("ex", ex) if isinstance(ex, dict) else ex
        cid = raw.get("corpus_id", f"__row{i}") if isinstance(raw, dict) else f"__row{i}"
        groups.setdefault(cid, []).append(i)
    return groups


@torch.no_grad()
def generate_group_with_shared_prefix(
    gm,
    prefills: List[List[int]],
    device: torch.device,
    max_length: int,
    decode_fn: Callable[[torch.Tensor], str],
    min_prefix: int = 64,
) -> Tuple[List[str], Dict[str, float]]:
    """Generate for one corpus group, prefilling the common token prefix exactly once.

    :param decode_fn: given the logits of the last prompt token, run the task's own decode loop and
        return the generated text. Passing this in keeps every stop rule, top-k and post-processing
        detail in the evaluator where it already lives -- this module only changes *where the
        prompt's KV comes from*, never how tokens are chosen.
    :param min_prefix: below this many shared tokens the rewind bookkeeping costs more than it
        saves, so fall back to a plain per-example prefill.

    :returns: (texts in the order of ``prefills``, timing//coverage stats).
    """
    stats = {"prefix_len": 0, "prefill_tokens": 0, "suffix_tokens": 0, "shared": 0.0}
    L = longest_common_token_prefix(prefills)

    if L < min_prefix or len(prefills) < 2:
        texts = []
        for ids in prefills:
            gm.prepare_inference_cache(1, max_length)
            leftpad = torch.zeros(1, dtype=torch.int32, device=device)
            logits = gm.model(torch.tensor([ids], device=device), logits_to_keep=1,
                              cache_leftpad=leftpad)
            texts.append(decode_fn(logits))
            stats["prefill_tokens"] += len(ids)
        return texts, stats

    # Prefill the shared prefix once. logits_to_keep=1 keeps the call shape identical to the normal
    # path; the returned logits belong to prefix token L-1 and are deliberately discarded -- the
    # first generated token must be predicted from the LAST token of each query's own suffix.
    t0 = time.time()
    gm.prepare_inference_cache(1, max_length)
    leftpad = torch.zeros(1, dtype=torch.int32, device=device)
    gm.model(torch.tensor([prefills[0][:L]], device=device), logits_to_keep=1,
             cache_leftpad=leftpad)
    snaps = snapshot_recurrent_states(gm.model)
    stats["prefix_len"] = L
    stats["prefill_tokens"] = L
    stats["prefill_seconds"] = time.time() - t0

    texts = []
    for ids in prefills:
        rewind_kv_cursor(gm.model, L)
        restore_recurrent_states(snaps)
        suffix = ids[L:]
        stats["suffix_tokens"] += len(suffix)
        logits = gm.model(torch.tensor([suffix], device=device), logits_to_keep=1)
        texts.append(decode_fn(logits))

    total = stats["prefill_tokens"] + stats["suffix_tokens"]
    naive = sum(len(ids) for ids in prefills)
    stats["shared"] = 1.0 - (total / naive) if naive else 0.0
    return texts, stats
