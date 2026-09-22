"""
Dev loss for the soft-detach CPT comparison (records/softdetach-cpt-plan.md), one checkpoint,
one GPU. Rows come from the HELD-OUT marker-wrapped shard (``cpt_dev``, cut from a source part no
training arm read). Every checkpoint is scored the same way:

  full_ce        teacher-forced CE over every body token of the 64k row, FULL attention
  tail20_ce      the same, restricted to the last 20% of the row's pseudo-documents (the "cpt80"
                 metric: what the model predicts at the end of a long document)
  own_*          for a soft arm (--arm sd20|sfl20): the same two numbers under the arm's OWN
                 training construction (pooled slots, detached), i.e. is the compression usable
                 at inference too, not only cheaper to train with

    python eval_cpt_devloss.py --ckpt <run>/model_and_optim --dev <shards>/cpt_dev --arm sd20 --rows 32 --out x.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import time

import numpy as np
import torch

from olmo_core.data.document_chunk_landmark import RESERVED_IDS
from olmo_core.distributed.checkpoint import load_model_and_optim_state
from olmo_core.nn.attention import AttentionBackendName
from olmo_core.nn.attention.chunked_mask import build_chunk_ids_from_tokens
from olmo_core.nn.attention.pooled_doc_kv import PooledDocKeepHolder, resolve_keep_docs
from olmo_core.nn.lm_head import LMLossImplementation
from olmo_core.nn.transformer import TransformerConfig

VOCAB = 248320
ARMS = {  # mirrors launch_softdetach_cpt.py
    "dense": None,
    "sd20": dict(keep_prob=0.2, rule="none", k=0.0),
    "sfl20": dict(keep_prob=0.0, rule="first_last", k=0.2),
}


def log(m):
    print(f"[cpt-devloss] {m}", flush=True)


def load_rows(shard, n):
    meta = json.load(open(f"{shard}/metadata.json"))
    ids = np.memmap(sorted(glob.glob(f"{shard}/token_ids_part_*.npy"))[0], dtype=np.dtype(meta["dtype"]), mode="r")
    msk = np.memmap(sorted(glob.glob(f"{shard}/labels_mask_*.npy"))[0], dtype=np.dtype(meta.get("mask_dtype", "bool")), mode="r")
    L = int(meta["max_example_len"])
    assert meta["min_example_len"] == L, "cpt shards are fixed-length rows"
    n = min(n, int(meta["num_instances"]))
    rows = [np.asarray(ids[i * L : (i + 1) * L], dtype=np.int64) for i in range(n)]
    masks = [np.asarray(msk[i * L : (i + 1) * L], dtype=bool) for i in range(n)]
    return rows, masks, meta


def stop_ids_from_rows(rows, ids, topk=100):
    flat = np.concatenate(rows)
    vals, cnt = np.unique(flat, return_counts=True)
    top = vals[np.argsort(-cnt)[:topk]].tolist()
    return sorted(set(int(t) for t in top) | {ids.doc_start, ids.doc_end, ids.eos, ids.landmark, ids.pad})


@torch.no_grad()
def per_token_loss(model, x, labels):
    """(S,) per-token CE in the ORIGINAL row frame, NaN where no label survives."""
    out = model(x, labels=labels, loss_reduction="none")
    loss = out.loss if hasattr(out, "loss") else out
    loss = loss.float().reshape(-1)
    full = torch.full((x.shape[1],), float("nan"), device=x.device)
    sel = labels[0] != -100
    if loss.numel() == x.shape[1]:
        full[sel] = loss[sel]
        return full
    if loss.numel() == int(sel.sum()):  # per-label-token vector (fused head)
        full[sel] = loss
        return full
    # compacted frame (shorter row): only the count is trustworthy -> spread as a constant (mean preserved)
    nz = loss[loss != 0]
    full[sel] = nz.mean() if nz.numel() else float("nan")
    return full


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help=".../model_and_optim (distcp)")
    ap.add_argument("--dev", required=True, help="held-out marker-wrapped shard dir (cpt_dev)")
    ap.add_argument("--arm", default="dense", choices=sorted(ARMS))
    ap.add_argument("--rows", type=int, default=32)
    ap.add_argument("--tail-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--attn-backend", default="flash_2")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ids = RESERVED_IDS["qwen3_5"]
    rows, masks, meta = load_rows(a.dev, a.rows)
    log(f"{len(rows)} dev rows x {len(rows[0])} tokens from {a.dev} (source parts {meta.get('source_parts')})")
    cfg = TransformerConfig.qwen3_5_4B(vocab_size=VOCAB, attn_backend=AttentionBackendName(a.attn_backend))
    cfg.lm_head.loss_implementation = LMLossImplementation.fused_linear
    model = cfg.build(init_device="cpu")
    t0 = time.time()
    load_model_and_optim_state(a.ckpt, model)
    log(f"loaded {a.ckpt} in {time.time() - t0:.0f}s")
    stop = stop_ids_from_rows(rows, ids)
    arm = ARMS[a.arm]
    model.enable_pooled_soft_tokens(ids.doc_start, ids.doc_end, ids.eos, placeholder_id=ids.landmark,
                                    keep_prob=(arm or {}).get("keep_prob", 0.0), keep_seed=a.seed, detach_soft_kv=True,
                                    keep_token_rule=(arm or {}).get("rule", "none"), keep_token_k=(arm or {}).get("k", 0.0))
    pst = model._pooled_soft_tokens  # slot construction is driven by these keys (as the trainer does)
    pst.update({"slot_mode": "cent_cmean", "slot_stop_ids": stop, "slot_stop_mask": None,
                "header_stop_id": None, "header_stop_count": 1, "header_cap": 32, "header_extra_tokens": 0,
                "keep_token_log_every": 0})
    model = model.cuda().to(torch.bfloat16)
    for mod in model.modules():
        rope = getattr(mod, "rope", None)
        if rope is not None and hasattr(rope, "warmup_cache"):
            rope.warmup_cache(len(rows[0]) + 8, torch.device("cuda"))
    res = {"full_ce": [], "tail20_ce": [], "own_ce": [], "own_tail20_ce": [], "own_compaction": []}
    t_start = time.time()
    for ri, (row, m) in enumerate(zip(rows, masks)):
        x = torch.tensor(row[None], device="cuda")
        # olmo-core labels are PRE-SHIFTED (labels[t] = input_ids[t+1], get_labels in train/common.py):
        # position t is scored on predicting token t+1, and the loss mask is the mask of the TARGET
        tgt_mask = torch.tensor(m[None], device="cuda")
        lab = torch.full_like(x, -100)
        lab[:, :-1] = torch.where(tgt_mask[:, 1:], x[:, 1:], torch.full_like(x[:, 1:], -100))
        # tail = the last tail_frac of the row's pseudo-documents
        cid = build_chunk_ids_from_tokens(x.cpu(), doc_start_id=ids.doc_start, doc_end_id=ids.doc_end, eos_id=ids.eos, mode="chunked")
        n_docs = int(cid.max()) + 1
        tail_docs = set(range(int(round(n_docs * (1 - a.tail_frac))), n_docs))
        # tail = predictions whose TARGET token lies in a tail pseudo-doc (shift by one like the labels)
        tail_tok = torch.tensor([c in tail_docs for c in cid[0].tolist()], device="cuda")
        tail_pos = torch.zeros_like(tail_tok)
        tail_pos[:-1] = tail_tok[1:]
        # (a)+(b): full attention
        model.eval(); model._pooled_keep_holder = None
        pt = per_token_loss(model, x, lab)
        res["full_ce"].append(float(torch.nanmean(pt)))
        res["tail20_ce"].append(float(torch.nanmean(pt[tail_pos])))
        # (c): the arm's own construction (gold-blind keep + rule from the run's flags)
        if arm is not None:
            keep = resolve_keep_docs(cid, n_docs, holder=None, keep_prob=float(arm["keep_prob"]), keep_seed=a.seed).cpu()
            model.train(); model._pooled_keep_holder = PooledDocKeepHolder(keep_docs=keep.clone())
            cb = model._compact_pooled_soft_tokens(x, lab, -100)[0]
            res["own_compaction"].append(cb.input_ids.shape[1] / x.shape[1])
            pt2 = per_token_loss(model, x, lab)
            res["own_ce"].append(float(torch.nanmean(pt2)))
            res["own_tail20_ce"].append(float(torch.nanmean(pt2[tail_pos])))
            model.eval(); model._pooled_keep_holder = None
        if ri + 1 in (1, 2, 5, 10) or (ri + 1) % 10 == 0:
            el = time.time() - t_start
            log(f"row {ri + 1}/{len(rows)} full {np.mean(res['full_ce']):.4f} tail20 {np.mean(res['tail20_ce']):.4f}"
                + (f" own {np.mean(res['own_ce']):.4f}/{np.mean(res['own_tail20_ce']):.4f} @x{np.mean(res['own_compaction']):.3f}" if arm else "")
                + f"  ({el:.0f}s, ETA {el / (ri + 1) * (len(rows) - ri - 1):.0f}s)")
    summ = {k: (float(np.mean(v)) if v else None) for k, v in res.items()}
    summ.update({k + "_se": (float(np.std(v, ddof=1) / np.sqrt(len(v))) if len(v) > 1 else None) for k, v in res.items()})
    out = {"ckpt": a.ckpt, "arm": a.arm, "dev": a.dev, "eval_size": len(rows), "summary": summ, "per_row": res,
           "git_commit": subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip(),
           "argv": os.sys.argv}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    log(f"SUMMARY {a.arm}: " + " ".join(f"{k}={v:.4f}" for k, v in summ.items() if v is not None and not k.endswith("_se")))
    log(f"wrote {a.out}")


if __name__ == "__main__":
    main()
