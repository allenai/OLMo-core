"""
Upper-bound sanity check for the mean-slot idea (Prasann, 2026-09-08): take the DENSE forward's own
per-layer keys/values, average them per pooled document, inject those at the slot columns of the
compacted forward at EVERY attention layer (+ the log L logit correction), and ask whether that
reproduces the dense model's answer. This isolates "is a mean-pooled slot enough?" from "does the
error accumulate when the slot has to be produced from a mean input embedding through the stack".

Per row: (1) full-attention forward, stashing every attention layer's post-RoPE K and V;
(2) compaction with the same keep policy as the soft-token probe; (3) per pooled doc and layer,
k* = mean of the doc's real post-RoPE keys, v* = mean of its real values (an exact "L copies of
the mean" slot with bias log L); (4) compacted forward with those slots overriding the slot
columns' K/V via the oracle-slot path (``soft_kv_override``, bias through the additive-bias SDPA
path). Variants: bias in {log L, 0, log L + c}. References: full attention; the mean-EMBEDDING soft
token with / without log L (the training-time construction).

On the Qwen3.5 hybrid only the 8 attention layers have K/V slots: the 24 GDN layers still see the
soft token's hidden state at the slot position, so this bounds the ATTENTION side of the error.

    python debug/pooled_kv/oracle_meankv_probe.py --task contradiction --rung 32k --rows 24 --out /results/oracle.json
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_side_slot_probe as P  # noqa: E402

from olmo_core.nn.attention import Attention  # noqa: E402
from olmo_core.nn.attention.chunked_mask import build_chunk_ids_from_tokens  # noqa: E402
from olmo_core.nn.attention.pooled_doc_kv import PooledDocKeepHolder, make_fingerprint_keep_docs_fn  # noqa: E402
from olmo_core.nn.pooled_soft_token import add_soft_len_bias, build_position_causal_bias  # noqa: E402

IDS = P.IDS


class KVStash:
    """Wrap Attention._prepare_qkv to stash each attention layer's post-RoPE k, v (detached)."""

    def __init__(self, model):
        self.layers = {}  # id(attn) -> layer index
        for li, blk in model.blocks.items():
            if type(blk.attention) is Attention:
                self.layers[id(blk.attention)] = int(li)
        self.kv = {}
        self.on = False
        orig = Attention._prepare_qkv
        stash = self

        def wrapper(attn, x, **kw):
            q, k, v = orig(attn, x, **kw)
            if stash.on and id(attn) in stash.layers:
                stash.kv[stash.layers[id(attn)]] = (k.detach(), v.detach())
            return q, k, v

        Attention._prepare_qkv = wrapper
        self.orig = orig

    def restore(self):
        Attention._prepare_qkv = self.orig


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="contradiction", choices=["contradiction", "oolong", "nq", "outlier"])
    ap.add_argument("--rung", default="32k")
    ap.add_argument("--rows", type=int, default=24)
    ap.add_argument("--keeps", default="0.3333,0.1667,0.0833,0")
    ap.add_argument("--extras", default="-2,-1,1,2")
    ap.add_argument("--work", default="/results/probe_work")
    ap.add_argument("--out", default="/results/oracle.json")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--jsonl", default=None)
    ap.add_argument("--shard", default=None)
    a = ap.parse_args()

    shard = a.shard or f"{a.work}/{P.FAMILY}_{a.task}_{a.rung}"
    if a.shard is None:
        P.convert(a.task, a.jsonl or P.EVAL_JSONL[a.task][a.rung], a.rows, shard)
    rows, masks = P.load_rows(shard, a.rows)
    P.log(f"{len(rows)} rows, lengths {[len(r) for r in rows[:6]]}...")

    from olmo_core.distributed.checkpoint import load_model_and_optim_state
    from olmo_core.nn.attention import AttentionBackendName
    from olmo_core.nn.lm_head import LMLossImplementation
    from olmo_core.nn.transformer import TransformerConfig

    cfg = P.build_cfg()
    cfg.lm_head.loss_implementation = LMLossImplementation.default
    model = cfg.build(init_device="cpu")
    ck = P.find_ckpt(a.ckpt) if a.ckpt else P.find_ckpt(P.CKPT[a.task])
    t0 = time.time(); load_model_and_optim_state(ck, model); P.log(f"loaded {ck} in {time.time() - t0:.0f}s")
    model.enable_pooled_soft_tokens(IDS.doc_start, IDS.doc_end, IDS.eos, placeholder_id=IDS.landmark, keep_prob=0.0,
                                    keep_seed=a.seed, detach_soft_kv=True)
    model.pooled_projector.reset_parameters()
    model = model.cuda().to(torch.bfloat16)
    pst = model._pooled_soft_tokens
    n_blocks = len(model.blocks)
    stash = KVStash(model)
    attn_layers = sorted(stash.layers.values())
    P.log(f"attention layers with K/V slots: {attn_layers}")

    gold_table = json.load(open(f"{shard}/gold_fingerprints.json")) if a.task in P.GOLD_TASKS else None
    keeps = [float(k) for k in a.keeps.split(",")]
    extras = [float(c) for c in a.extras.split(",")]
    variants = [("oracle meanKV +logL", "oracle", 1.0, 0.0), ("oracle meanKV no-bias", "oracle", 0.0, 0.0)]
    variants += [(f"oracle meanKV +logL{c:+.0f}", "oracle", 1.0, c) for c in extras]
    variants += [("soft meanEmb no-bias", "soft", 0.0, 0.0), ("soft meanEmb +logL", "soft", 1.0, 0.0)]
    names = ["full"] + [f"k={k:.3f} {v[0]}" for k in keeps for v in variants]
    res = {n: {"ce": [], "top1": [], "kl": [], "correct": [], "compaction": []} for n in names}

    for ri, (row, rmask) in enumerate(zip(rows, masks)):
        x = torch.tensor(row[None], device="cuda")
        ans_pos = torch.tensor(np.nonzero(rmask)[0], device="cuda")
        pred_pos = ans_pos - 1
        targets = x[0, ans_pos]
        cid = build_chunk_ids_from_tokens(x.cpu(), doc_start_id=IDS.doc_start, doc_end_id=IDS.doc_end, eos_id=IDS.eos)[0]

        # (1) dense forward, stash K/V
        model.eval(); stash.kv.clear(); stash.on = True
        lf = model(x, logits_to_keep=pred_pos[None])[0].float()
        stash.on = False
        r = res["full"]; r["ce"].append(float(F.cross_entropy(lf, targets))); r["top1"].append(1.0); r["kl"].append(0.0)
        r["correct"].append(float((lf.argmax(-1) == targets).all())); r["compaction"].append(1.0)

        for keep in keeps:
            model.train()
            if a.task in P.GOLD_TASKS:
                keep_fn = make_fingerprint_keep_docs_fn(gold_table, doc_start_id=IDS.doc_start, doc_end_id=IDS.doc_end,
                                                        eos_id=IDS.eos, n_random_frac=keep, mode="gold_plus_random", seed=a.seed)
                model._pooled_keep_holder = PooledDocKeepHolder(keep_docs=keep_fn(x.cpu()))
            else:
                model._pooled_keep_holder = None
                pst["keep_prob"] = keep
            pst["len_bias"] = False
            cb = model._compact_pooled_soft_tokens(x, None, -100)[0]
            posmap = {int(p): c for c, p in enumerate(cb.position_ids[0].tolist())}
            cols = torch.tensor([posmap[int(p)] for p in pred_pos.tolist()], device="cuda")
            comp = cb.input_ids.shape[1] / x.shape[1]
            n_slots = int(cb.soft_rows.numel())

            # (3) per-slot, per-layer mean of the REAL post-RoPE K and V from the dense forward.
            # The override rotates k by the given position with the layer's RoPE; passing pos=0
            # leaves the post-RoPE mean untouched, which is exactly the "L copies of the mean" slot.
            k0, v0 = stash.kv[attn_layers[0]]
            Hkv, D = k0.shape[2], k0.shape[3]
            slots = torch.zeros((n_slots, n_blocks, 2, Hkv, D), dtype=k0.dtype, device="cuda")
            log_len = torch.zeros(n_slots, device="cuda")
            doc_pos = {}
            for si in range(n_slots):
                d = int(cb.soft_docs[si])
                if d not in doc_pos:
                    doc_pos[d] = (cid == d).nonzero(as_tuple=True)[0].cuda()
                p = doc_pos[d]
                log_len[si] = math.log(max(1, p.numel()))
                for li in attn_layers:
                    k, v = stash.kv[li]
                    slots[si, li, 0] = k[0, p].float().mean(0).to(k.dtype)
                    slots[si, li, 1] = v[0, p].float().mean(0).to(v.dtype)
            base_bias = build_position_causal_bias(cb, dtype=torch.bfloat16, device=torch.device("cuda"))

            for vname, kind, scale, extra in variants:
                name = f"k={keep:.3f} {vname}"
                if kind == "oracle":
                    biases = torch.zeros((n_slots, n_blocks), device="cuda")
                    for li in attn_layers:
                        biases[:, li] = scale * log_len + extra
                    ovr = {"rows": cb.soft_rows.cuda(), "cols": cb.soft_cols.cuda(), "pos": torch.zeros(n_slots, dtype=torch.long, device="cuda"),
                           "slots": slots, "biases": biases}
                    pst["len_bias"] = False
                    lg = model(x, logits_to_keep=cols[None], soft_kv_override_layers=ovr, attn_bias=base_bias)[0].float()
                else:
                    pst["len_bias"], pst["len_bias_scale"], pst["len_bias_extra"] = (scale > 0), scale, extra
                    lg = model(x, logits_to_keep=cols[None])[0].float()
                r = res[name]
                r["ce"].append(float(F.cross_entropy(lg, targets)))
                r["top1"].append(float((lg.argmax(-1) == lf.argmax(-1)).float().mean()))
                r["kl"].append(float(F.kl_div(F.log_softmax(lg, -1), F.log_softmax(lf, -1), log_target=True, reduction="batchmean")))
                r["correct"].append(float((lg.argmax(-1) == targets).all()))
                r["compaction"].append(comp)
            model.eval()
        if ri + 1 in (1, 2, 5) or (ri + 1) % 8 == 0:
            P.log(f"row {ri + 1}/{len(rows)}")
            table(res, names)
    stash.restore()
    table(res, names)
    out = {"task": a.task, "rung": a.rung, "rows": len(rows), "ckpt": ck, "attn_layers": attn_layers,
           "configs": {n: {k: float(np.mean(v)) for k, v in res[n].items()} for n in names}}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    P.log(f"wrote {a.out}")


def table(res, names):
    print(f"{'config':40} {'answer CE':>9} {'top1=full':>9} {'KL':>7} {'correct':>8} {'compact':>8}", flush=True)
    for n in names:
        r = res[n]
        if r["ce"]:
            print(f"{n:40} {np.mean(r['ce']):9.3f} {np.mean(r['top1']):9.3f} {np.mean(r['kl']):7.3f} {np.mean(r['correct']):8.2f} {np.mean(r['compaction']):8.3f}", flush=True)


if __name__ == "__main__":
    main()
