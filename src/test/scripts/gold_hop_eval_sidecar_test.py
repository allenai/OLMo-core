"""
End-to-end CPU checks that the **eval-side** gold-hop mask is really gold-edited, on REAL eval
prefills, through the ACTUAL install path.

Why this file exists, separately from ``src/test/nn/attention/gold_hop_mask_test.py``: that suite
proves the graph edit and the mask are correct on synthetic roles. It cannot catch the eval-specific
failure, which is not in the mask at all -- it is that the **sidecar key space is different**. The eval
prefill is prompt-only (no answer, no EOS, no padding), so a training-shard ``gold_pairs.json``
fingerprints **0 of 488** eval rows (measured). Every row would then silently degrade to an all-True
graph = plain causal over the context, scoring every arm as unrestricted ``standard`` near the 0.943
ceiling -- a failure that reads as a triumphant result.

So the checks here are: the eval sidecar hits, the resulting mask really has the gold edit, and the
hard assert fires when handed the wrong sidecar.

⚠ Data-dependent (the eval JSONL + the Qwen3 tokenizer), so every test skips cleanly when they are
absent. That is deliberate: these assert a property of *our data + our eval path*, which a hermetic
fixture cannot speak to.
"""

import json
import os

import pytest
import torch

from olmo_core.nn.attention.chunked_mask import build_chunk_ids_from_tokens
from olmo_core.nn.attention.gold_grad_mask import content_fingerprint_from_row
from olmo_core.nn.attention.gold_hop_mask import (
    GOLD_HOP_VALUES,
    GOLD_HOPS_INF,
    make_fingerprint_gold_hop_fn,
    shortest_gold_hops,
)

EVAL_JSONL = (
    "/scratch/users/prasann/corpus-reasoning/data/contradiction_eval_pubmed_both_n50_k3.jsonl"
)
EVAL_SIDECAR = "/scratch/users/prasann/longctx_sft_qwen/contra_n50_v2_orig/gold_pairs_eval_n50.json"
TRAIN_SIDECAR = "/scratch/users/prasann/longctx_sft_qwen/contra_n50_v2_orig/gold_pairs.json"

DOC_START, DOC_END, EOS = 151648, 151649, 151643
KEEP_PROB, SEED = 0.25, 42
N_EX = 8  # a handful of real prefills is enough; the property is per example

pytestmark = pytest.mark.skipif(
    not (os.path.exists(EVAL_JSONL) and os.path.exists(EVAL_SIDECAR)),
    reason="needs the n50 contradiction eval JSONL + its gold_pairs sidecar (local /scratch data)",
)


def _prefills(n=N_EX):
    """Real eval prefills, rendered by the EVAL'S OWN function -- never a reimplementation, because a
    fingerprint is a hash and a one-token drift would silently zero the hit rate."""
    from transformers import AutoTokenizer

    from corpus_reasoning.eval.eval_lc_native_docchunk_contra import build_eval_prefill
    from corpus_reasoning.eval.evaluate import load_unified_examples

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    examples = load_unified_examples(
        EVAL_JSONL, 0, task="contradiction", query_position="both", use_alpaca=True
    )
    out = []
    for ex in examples[:n]:
        raw = ex.get("ex", ex)
        out.append(
            (
                build_eval_prefill(
                    tok,
                    raw,
                    variant="dense",
                    cot_mode="none",
                    doc_start_id=DOC_START,
                    doc_end_id=DOC_END,
                ),
                raw,
            )
        )
    return out


@pytest.fixture(scope="module")
def prefills():
    return _prefills()


def test_the_training_sidecar_does_not_hit_a_single_eval_prefill(prefills):
    """The premise of the whole eval-side builder, asserted rather than assumed. If this ever starts
    passing keys, the two key spaces have merged and the separate builder is redundant."""
    if not os.path.exists(TRAIN_SIDECAR):
        pytest.skip("training sidecar not present")
    train = json.load(open(TRAIN_SIDECAR))
    hits = sum(1 for ids, _ in prefills if content_fingerprint_from_row(ids, EOS) in train)
    assert hits == 0, (
        "a training-shard fingerprint matched an eval prefill -- unexpected, and it would mean the "
        "prompt-only prefill now carries the answer + EOS"
    )


def test_every_eval_prefill_hits_the_eval_sidecar(prefills):
    table = json.load(open(EVAL_SIDECAR))
    for ids, _ in prefills:
        assert content_fingerprint_from_row(ids, EOS) in table


@pytest.mark.parametrize("hops", GOLD_HOP_VALUES)
def test_eval_mask_on_a_real_prefill_is_really_gold_edited(prefills, hops):
    """PER EXAMPLE, on real eval prefills, through the real lookup: no gold->gold edge in the deleting
    arms, shortest gold path exactly ``h``, unreachable for hop_inf. If the sidecar silently missed,
    the graph would be all-True and every one of these assertions fails."""
    table = json.load(open(EVAL_SIDECAR))
    fn = make_fingerprint_gold_hop_fn(
        table,
        doc_start_id=DOC_START,
        doc_end_id=DOC_END,
        eos_id=EOS,
        hops=hops,
        doc_keep_prob=KEEP_PROB,
        seed=SEED,
    )
    for ids, raw in prefills:
        adj = fn(torch.tensor([ids]))[0].numpy()
        assert not adj.all(), "an all-True graph means the fingerprint MISSED and the arm is inert"
        for p in raw["gold_doc_indices"]:
            a, b = sorted(int(x) - 1 for x in p)
            realized = shortest_gold_hops(adj, a, b)
            if hops == 1:
                assert adj[b, a] and realized == 1
                continue
            assert not adj[b, a], f"gold edge {b}->{a} survived in hop{hops}"
            routable = (b - a) >= hops and hops != GOLD_HOPS_INF
            assert realized == (hops if routable else GOLD_HOPS_INF)

    assert fn.counters["hits"] == fn.counters["graph_rows"] == len(prefills)
    assert fn.misses == []


def test_the_hard_assert_fires_on_a_deliberately_wrong_sidecar(prefills):
    """⚠ The guard that matters most. Feed the TRAINING sidecar (the realistic mistake -- it exists, it
    is named gold_pairs.json, and it is for the same corpus) and require a loud failure. Without this,
    the run reports a near-ceiling f1 and looks like a win."""
    from olmo_core.nn.attention.gold_hop_mask import GoldHopMaskHolder

    if not os.path.exists(TRAIN_SIDECAR):
        pytest.skip("training sidecar not present")
    fn = make_fingerprint_gold_hop_fn(
        json.load(open(TRAIN_SIDECAR)),
        doc_start_id=DOC_START,
        doc_end_id=DOC_END,
        eos_id=EOS,
        hops=2,
        doc_keep_prob=KEEP_PROB,
        seed=SEED,
    )
    for ids, _ in prefills:
        adj = fn(torch.tensor([ids]))
        assert bool(adj.all()), "the wrong sidecar degrades to all-True -- this is the silent failure"

    holder = GoldHopMaskHolder(counters=fn.counters, misses=fn.misses)
    assert holder.hit_rate == 0.0
    with pytest.raises(SystemExit, match="MISSED the gold-pairs sidecar"):
        holder.require_full_hit_rate(context="wrong-sidecar test")


def test_gold_indices_name_real_chunks_in_the_prefill(prefills):
    """chunk index == 'Claim N' - 1 must hold in the PREFILL's own token structure, or the mask edits
    the wrong documents while looking perfectly healthy."""
    for ids, raw in prefills:
        roles = build_chunk_ids_from_tokens(
            torch.tensor([ids]), doc_start_id=DOC_START, doc_end_id=DOC_END, eos_id=EOS
        )[0]
        present = {int(d) for d in torch.unique(roles[roles >= 0]).tolist()}
        assert len(present) == 50
        flat = {int(x) - 1 for p in raw["gold_doc_indices"] for x in p}
        assert flat.issubset(present)
