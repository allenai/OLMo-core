"""
GDN-intact pooling probe (Prasann, 2026-09-08): keep EVERY token in the sequence -- so the 24 GDN
layers (and the FFNs) see the real documents exactly as at full attention -- and pool only the
attention layers' keys/values: for queries outside a pooled document, its per-token K/V collapse to
a single mean slot (+log L logit bias), i.e. :class:`PooledDocKVAttention` on the 8 full-attention
blocks of the dense-trained Qwen3.5-4B checkpoints. Compared with the soft-token compaction (which
removes the tokens from the whole stack), the gap between the two isolates how much of the
compaction loss is the GDN/FFN path seeing one mean-embedding token instead of the document.

No train-time saving in this construction beyond attention (every token still runs the network);
it is a diagnostic, not a candidate.

    python debug/pooled_kv/pooledkv_eval_probe.py --task contradiction --rung 32k --rows 24
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_side_slot_probe as P  # noqa: E402

from olmo_core.nn.attention import AttentionType  # noqa: E402
from olmo_core.nn.attention.pooled_doc_kv import PooledDocKVAttention, install_pooled_doc_keep, make_fingerprint_keep_docs_fn  # noqa: E402

IDS = P.IDS


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="contradiction", choices=["contradiction", "oolong", "nq", "outlier"])
    ap.add_argument("--rung", default="32k")
    ap.add_argument("--rows", type=int, default=24)
    ap.add_argument("--keeps", default="0.3333,0.1667,0.0833,0")
    ap.add_argument("--work", default="/results/probe_work")
    ap.add_argument("--out", default="/results/pooledkv.json")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--jsonl", default=None)
    ap.add_argument("--shard", default=None)
    a = ap.parse_args()

    shard = a.shard or f"{a.work}/{P.FAMILY}_{a.task}_{a.rung}"
    if a.shard is None:
        P.convert(a.task, a.jsonl or P.EVAL_JSONL[a.task][a.rung], a.rows, shard)
    rows, masks = P.load_rows(shard, a.rows)
    P.log(f"{len(rows)} rows")

    from olmo_core.distributed.checkpoint import load_model_and_optim_state
    from olmo_core.nn.lm_head import LMLossImplementation

    cfg = P.build_cfg()
    cfg.lm_head.loss_implementation = LMLossImplementation.default
    blk = cfg.block
    mixer = blk["attn"].sequence_mixer if isinstance(blk, dict) else blk.sequence_mixer
    mixer.name = AttentionType.pooled_doc_kv
    mixer.pooled_keep_prob = 0.0
    mixer.pooled_keep_seed = a.seed
    mixer.pooled_len_bias = True
    cfg.document_chunk_attention = {"doc_start_id": IDS.doc_start, "doc_end_id": IDS.doc_end, "eos_id": IDS.eos, "mode": "chunked"}
    model = cfg.build(init_device="cpu")
    ck = P.find_ckpt(a.ckpt) if a.ckpt else P.find_ckpt(P.CKPT[a.task])
    t0 = time.time(); load_model_and_optim_state(ck, model); P.log(f"loaded {ck} in {time.time() - t0:.0f}s")
    model = model.cuda().to(torch.bfloat16)
    pooled_layers = [m for m in model.modules() if isinstance(m, PooledDocKVAttention)]
    P.log(f"{len(pooled_layers)} pooled-KV attention layers")
    # chunk ids are reconstructed (and pooling happens) whenever this config is set, train or eval;
    # the full-attention reference is the same model with reconstruction switched off.
    dc_cfg = model._document_chunk_attention

    gold_table = json.load(open(f"{shard}/gold_fingerprints.json")) if a.task in P.GOLD_TASKS else None
    keeps = [float(k) for k in a.keeps.split(",")]
    names = ["full"] + [f"pooledKV k={k:.3f} {b}" for k in keeps for b in ("+logL", "no-bias")]
    res = {n: {"ce": [], "top1": [], "kl": [], "correct": [], "compaction": [], "sec": []} for n in names}
    holder = None
    for ri, (row, rmask) in enumerate(zip(rows, masks)):
        x = torch.tensor(row[None], device="cuda")
        ans_pos = torch.tensor(np.nonzero(rmask)[0], device="cuda")
        pred_pos = ans_pos - 1
        targets = x[0, ans_pos]
        model.eval()
        t0 = time.time()
        model._document_chunk_attention = None
        lf = model(x, logits_to_keep=pred_pos[None])[0].float()
        model._document_chunk_attention = dc_cfg
        r = res["full"]; r["ce"].append(float(F.cross_entropy(lf, targets))); r["top1"].append(1.0); r["kl"].append(0.0)
        r["correct"].append(float((lf.argmax(-1) == targets).all())); r["compaction"].append(1.0); r["sec"].append(time.time() - t0)
        for keep in keeps:
            if a.task in P.GOLD_TASKS:
                keep_fn = make_fingerprint_keep_docs_fn(gold_table, doc_start_id=IDS.doc_start, doc_end_id=IDS.doc_end, eos_id=IDS.eos,
                                                        n_random_frac=keep, mode="gold_plus_random", seed=a.seed)
                if holder is not None:
                    holder._hook_handle.remove()
                holder = install_pooled_doc_keep(model, keep_fn)
            else:
                for m in pooled_layers:
                    m._pooled_keep_holder = None
                    m.keep_prob = keep
                    m.keep_seed = a.seed
            for bias in ("+logL", "no-bias"):
                for m in pooled_layers:
                    m.len_bias = bias == "+logL"
                name = f"pooledKV k={keep:.3f} {bias}"
                t0 = time.time()
                lg = model(x, logits_to_keep=pred_pos[None])[0].float()
                r = res[name]
                r["ce"].append(float(F.cross_entropy(lg, targets)))
                r["top1"].append(float((lg.argmax(-1) == lf.argmax(-1)).float().mean()))
                r["kl"].append(float(F.kl_div(F.log_softmax(lg, -1), F.log_softmax(lf, -1), log_target=True, reduction="batchmean")))
                r["correct"].append(float((lg.argmax(-1) == targets).all()))
                r["compaction"].append(1.0); r["sec"].append(time.time() - t0)
        if ri + 1 in (1, 2, 5) or (ri + 1) % 8 == 0:
            P.log(f"row {ri + 1}/{len(rows)}")
            P.summary(res, [(n, None, None) for n in names])
    P.summary(res, [(n, None, None) for n in names])
    out = {"task": a.task, "rung": a.rung, "rows": len(rows), "ckpt": ck, "configs": {n: {k: float(np.mean(v)) for k, v in res[n].items()} for n in names}}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    P.log(f"wrote {a.out}")


if __name__ == "__main__":
    main()
