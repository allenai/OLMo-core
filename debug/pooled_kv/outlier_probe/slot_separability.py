"""
Is the OUTLIER signal present in the slot vector at all?  (oracle readout, no model forward)

``outlier_richer_slot_probe.py`` asks whether a FROZEN DENSE model can read a pooled outlier out
of a slot.  That conflates two failures: the slot may not carry the signal, or it may carry it in
a form this model was never trained to read.  This script separates them.

Outlier = "which k of n documents are topically odd".  The natural oracle readout is therefore
**distance to the corpus centroid in slot space**: rank the row's documents by how far their slot
vector is from the mean slot vector, take the top k, and score that against the gold set.  If a
construction's oracle recall is at the ``k/n`` guess floor, that construction's vector does not
carry the signal at all and no reader -- trained or frozen -- can recover it.  If the oracle recall
is high while the probe's generation recall is at the floor, the signal IS there and the gap is a
READOUT problem, which is what a trained reader (or summarizer) would attack.

The same readout on a plain lexical TF-IDF vector per document is the reference: it says how much
of the task a purely topical, embedding-free summary solves on this corpus.

    python debug/pooled_kv/outlier_probe/slot_separability.py --shard <tokenized shard> --rows 200
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, ".."))

import eval_side_slot_probe as P  # noqa: E402
import outlier_richer_slot_probe as R  # noqa: E402
import outlier_slot_probe as O  # noqa: E402

from olmo_core.distributed.checkpoint import load_model_and_optim_state  # noqa: E402
from olmo_core.nn.attention.chunked_mask import (  # noqa: E402
    build_chunk_ids_from_tokens,
    mark_doc_headers_free,
)
from olmo_core.nn.attention.gold_grad_mask import content_fingerprint_from_row  # noqa: E402
from olmo_core.nn.lm_head import LMLossImplementation  # noqa: E402

IDS = O.IDS


def log(m):
    print(f"[separability] {m}", flush=True)


def topk_recall(dist, gold, k):
    """Recall of the gold set among the k FARTHEST documents from the centroid."""
    order = np.argsort(-dist)
    top = set(order[:k].tolist())
    return len(top & set(gold)) / max(1, len(gold))


def mean_rank_pct(dist, gold):
    """Mean percentile rank of the gold documents when sorted by distance (1.0 = farthest)."""
    order = np.argsort(dist)
    rank = np.empty_like(order)
    rank[order] = np.arange(len(dist))
    n = max(1, len(dist) - 1)
    return float(np.mean([rank[g] / n for g in gold]))


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rung", default="2k")
    ap.add_argument("--rows", type=int, default=200)
    ap.add_argument("--shard", default=None)
    ap.add_argument("--jsonl", default=None)
    ap.add_argument("--ckpt-name", default="ds64-outlier-dense-u64M")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--slots", default="mean,meanrn,cmean100,cmean500,centered,cent_cmean,idf,g2,g4,g2cent,g2cc,enc2,enc4")
    ap.add_argument("--work", default="/results/sep_work")
    ap.add_argument("--out", default="/results/slot_separability.json")
    a = ap.parse_args()

    if a.tokenizer:
        P.TOKENIZER = a.tokenizer
    shard = a.shard or f"{a.work}/outlier_{a.rung}"
    if a.shard is None:
        P.convert("outlier", a.jsonl or O.RUNGS[a.rung], a.rows, shard)
    rows, _ = P.load_rows(shard, a.rows)
    log(f"{len(rows)} rows")

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(P.TOKENIZER)
    cfg = P.build_cfg()
    cfg.lm_head.loss_implementation = LMLossImplementation.default
    model = cfg.build(init_device="cpu")
    ck = P.find_ckpt(a.ckpt or O.CKPTS[a.ckpt_name])
    load_model_and_optim_state(ck, model)
    model = model.cuda().to(torch.bfloat16)
    model.eval()
    R.ST["corpus"] = R.corpus_stats(rows, tok, model)
    idf_np = R.ST["corpus"]["idf"].cpu().numpy()

    slots = [s for s in a.slots.split(",") if s]
    mk = lambda: {"recall": [], "recall_nn": [], "rankpct": [], "cos_gold": [], "cos_other": []}
    res = {s: mk() for s in slots}
    res["lexidf"] = mk()
    floor = []

    gold_table = json.load(open(f"{shard}/gold_fingerprints.json"))
    gold_table = {
        fp: sorted({int(i) for v in val for i in (v if isinstance(v, (list, tuple)) else [v])})
        for fp, val in gold_table.items()
    }

    for ri, row in enumerate(rows):
        x = torch.tensor(row[None], device="cuda")
        gold = gold_table.get(content_fingerprint_from_row(row.tolist(), IDS.eos))
        if gold is None:
            continue
        cid = build_chunk_ids_from_tokens(
            x, doc_start_id=IDS.doc_start, doc_end_id=IDS.doc_end, eos_id=IDS.eos, mode="chunked"
        )
        n_docs = int(cid.max()) + 1
        cid = mark_doc_headers_free(
            cid, x, doc_start_id=IDS.doc_start, doc_end_id=IDS.doc_end,
            stop_id=R.HEADER_STOP_ID, stop_count=1, cap=32,
        )
        gold = [g for g in gold if 0 <= g < n_docs]
        k = len(gold)
        floor.append(k / max(1, n_docs))

        for s in slots:
            params = R.SLOTS[s]
            G = params.get("G", 1)
            cid_seg, n_seg = R._seg_split(cid, n_docs, G)
            want = torch.arange(n_seg, device="cuda")
            R.ST["enc_cache"] = {}
            feats, _ = R.build_slot_feats(model, x, cid_seg, n_seg, want, params, ri)
            v = feats.float().cpu().numpy()
            # G > 1: the document's oddness is its ODDEST segment (averaging the segment means back
            # to a document vector would just reconstruct the plain mean and measure nothing).
            _score(res[s], v, gold, k, G=G, n_docs=n_docs)

        # lexical TF-IDF reference (no embeddings at all)
        c = cid[0].cpu().numpy()
        ctx = c >= 0
        uniq, inv = np.unique(np.asarray(row)[ctx], return_inverse=True)
        M = np.zeros((n_docs, len(uniq)))
        np.add.at(M, (c[ctx], inv), 1.0)
        M *= idf_np[uniq][None, :]
        _score(res["lexidf"], M, gold, k)

        if ri + 1 in (1, 2, 5, 10) or (ri + 1) % 25 == 0:
            log(f"row {ri + 1}/{len(rows)}")
            table(res, floor)

    table(res, floor)
    out = {
        "task": "outlier", "rung": a.rung, "eval_size": len(floor), "ckpt": ck,
        "ckpt_name": a.ckpt_name, "floor_kn": float(np.mean(floor)),
        "readouts": {
            s: {
                "oracle_topk_recall": float(np.mean(r["recall"])),
                "oracle_topk_recall_nn": float(np.mean(r["recall_nn"])),
                "oracle_topk_recall_nn_se": float(np.std(r["recall_nn"]) / max(1, len(r["recall_nn"])) ** 0.5),
                "oracle_topk_recall_se": float(np.std(r["recall"]) / max(1, len(r["recall"])) ** 0.5),
                "gold_mean_rank_pct": float(np.mean(r["rankpct"])),
                "cos_gold_centroid": float(np.mean(r["cos_gold"])),
                "cos_nongold_centroid": float(np.mean(r["cos_other"])),
                "n": len(r["recall"]),
            }
            for s, r in res.items() if r["recall"]
        },
    }
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    log(f"wrote {a.out}")


def _score(acc, v, gold, k, G=1, n_docs=None):
    c = v.mean(0, keepdims=True)
    vn = v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9)
    cn = c / (np.linalg.norm(c, axis=-1, keepdims=True) + 1e-9)
    cos = (vn * cn).sum(-1)
    # A second, task-shaped readout: an outlier has no topical NEIGHBOUR, while every ordinary
    # document has at least one other document about the same thing. Rank by max similarity to any
    # other document (lowest = oddest). Strictly a better oracle than centroid distance whenever
    # the ordinary documents are a tight topical cluster.
    sim = vn @ vn.T
    np.fill_diagonal(sim, -2.0)
    nn = sim.max(1)
    if G > 1:
        cos = cos.reshape(n_docs, G).min(1)  # oddest segment wins
        nn = nn.reshape(n_docs, G).min(1)
    acc["recall_nn"].append(topk_recall(-nn, gold, k))
    acc["recall"].append(topk_recall(-cos, gold, k))  # farthest from centroid = most cosine-distant
    acc["rankpct"].append(mean_rank_pct(cos, gold))  # low cosine = high "oddness"; 0.0 = oddest
    acc["cos_gold"].append(float(np.mean([cos[g] for g in gold])))
    acc["cos_other"].append(float(np.mean([cos[i] for i in range(len(cos)) if i not in gold])))


def table(res, floor):
    print(f"{'readout':12} {'R@k centroid':>14} {'R@k nn':>14} {'gold rankpct':>13} "
          f"{'cos(gold,c)':>12} {'cos(other,c)':>13} {'rows':>5}   (k/n floor {np.mean(floor):.3f})",
          flush=True)
    for s, r in res.items():
        if not r["recall"]:
            continue
        n = len(r["recall"])
        se = float(np.std(r["recall"]) / max(1, n) ** 0.5)
        sn = float(np.std(r["recall_nn"]) / max(1, n) ** 0.5)
        print(f"{s:12} {np.mean(r['recall']):9.3f}±{se:.3f} {np.mean(r['recall_nn']):9.3f}±{sn:.3f} "
              f"{np.mean(r['rankpct']):13.3f} {np.mean(r['cos_gold']):12.4f} "
              f"{np.mean(r['cos_other']):13.4f} {n:5d}", flush=True)


if __name__ == "__main__":
    main()
