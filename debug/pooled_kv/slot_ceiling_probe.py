"""
Slot CEILING probe (Prasann, 2026-09-08): on a dense-trained model and held-out rows, how close
can ONE slot per document get to full attention if the slot is (a) the mean-embedding soft token
(the training-time construction), (b) the dense forward's per-layer mean K/V (+log L), or (c) a
slot FITTED to the document's log-mass function -- and how much do G > 1 slots per document buy?

Multiple slots: each non-gold document is split into G contiguous pieces at the chunk-id level
(positions untouched, no extra tokens); every path then simply sees G documents. Gold documents
stay whole (they are always kept real anyway).

Fitted slot (per document, per attention layer): unknowns k*_h (one per KV head) and a scalar c
such that  scale * q_g . k*_{h(g)} + c  ~=  logsumexp_t( scale * q_g . k_{t,h(g)} )  over the
document's tokens t, for a query stash q = the row's answer-position queries plus random queries
from the tail of the row, ridge-regularised toward (mean key, log L). v* = the attention-weighted
mean value under the same queries. Because the query stash is shared by every document, the
normal-equation matrix is shared too and all documents are solved at once. This is fitted on the
row's own queries, so it is an optimistic ceiling for a single static slot.

    python debug/pooled_kv/slot_ceiling_probe.py --task contradiction --rung 32k --rows 24 --slots 1,2,4,8
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

import olmo_core.nn.transformer.model as MODEL_MOD  # noqa: E402
from olmo_core.nn.attention import Attention  # noqa: E402
from olmo_core.nn.attention.chunked_mask import build_chunk_ids_from_tokens  # noqa: E402
from olmo_core.nn.attention.pooled_doc_kv import PooledDocKeepHolder, make_fingerprint_keep_docs_fn  # noqa: E402
from olmo_core.nn.pooled_soft_token import build_position_causal_bias  # noqa: E402

IDS = P.IDS


class QKVStash:
    def __init__(self, model):
        self.layers = {id(b.attention): int(li) for li, b in model.blocks.items() if type(b.attention) is Attention}
        self.qkv, self.on = {}, False
        orig = Attention._prepare_qkv
        st = self

        def wrapper(attn, x, **kw):
            q, k, v = orig(attn, x, **kw)
            if st.on and id(attn) in st.layers:
                st.qkv[st.layers[id(attn)]] = (q.detach(), k.detach(), v.detach())
            return q, k, v

        Attention._prepare_qkv = wrapper
        self.orig = orig

    def restore(self):
        Attention._prepare_qkv = self.orig


def split_docs(cid, gold, G):
    """Sub-document ids: non-gold docs split into G contiguous pieces; returns (sub_cid, parent)."""
    sub = cid.clone()
    n_docs = int(cid.max()) + 1
    parent = []
    nxt = 0
    for d in range(n_docs):
        pos = (cid == d).nonzero(as_tuple=True)[0]
        if pos.numel() == 0:
            continue
        pieces = 1 if (gold[d] or G == 1) else min(G, max(1, pos.numel() // 3))
        bounds = torch.linspace(0, pos.numel(), pieces + 1).round().long()
        for j in range(pieces):
            sub[pos[bounds[j]:bounds[j + 1]]] = nxt
            parent.append(d)
            nxt += 1
    return sub, torch.tensor(parent)


@torch.no_grad()
def fit_slots(q_all, k, v, seg, n_seg, scale, lam=0.02, mu=0.02, log_len=None, kbar=None):
    """
    q_all: (Nq, Hq, D) queries; k, v: (T, Hkv, D); seg: (T,) sub-doc id per token (-1 = not a doc).
    Returns k* (n_seg, Hkv, D), v* (n_seg, Hkv, D), c (n_seg,).
    """
    Nq, Hq, D = q_all.shape
    T, Hkv, _ = k.shape
    n_rep = Hq // Hkv
    dev = q_all.device
    ctx = seg >= 0
    kc, vc, segc = k[ctx].float(), v[ctx].float(), seg[ctx]
    Tc = kc.shape[0]
    n_eq = Nq * n_rep
    # shared normal-equation matrix: block-diagonal over heads + the c column
    M = torch.zeros(Hkv * D + 1, Hkv * D + 1, device=dev)
    qf = q_all.float()
    rhs_k = torch.zeros(n_seg, Hkv, D, device=dev)
    rhs_c = torch.zeros(n_seg, device=dev)
    vstar = torch.zeros(n_seg, Hkv, D, device=dev)
    for h in range(Hkv):
        qh = qf[:, h * n_rep:(h + 1) * n_rep].reshape(-1, D) * scale  # (n_eq, D)
        M[h * D:(h + 1) * D, h * D:(h + 1) * D] = qh.t() @ qh + lam * n_eq * torch.eye(D, device=dev)
        M[h * D:(h + 1) * D, -1] = qh.sum(0)
        M[-1, h * D:(h + 1) * D] = qh.sum(0)
        s = qh @ kc[:, h].t()  # (n_eq, Tc) logits of every query against every doc token
        smax = torch.full((n_eq, n_seg), -1e30, device=dev).scatter_reduce(1, segc[None].expand(n_eq, -1), s, reduce="amax")
        ex = torch.exp(s - smax.gather(1, segc[None].expand(n_eq, -1)))
        ssum = torch.zeros(n_eq, n_seg, device=dev).index_add_(1, segc, ex)
        lse = smax + torch.log(ssum.clamp(min=1e-30))  # (n_eq, n_seg) log-mass of each sub-doc per query
        rhs_k[:, h] = (lse.t() @ qh) + lam * n_eq * (kbar[:, h] if kbar is not None else 0.0)
        rhs_c += lse.sum(0)
        p = ex / ssum.gather(1, segc[None].expand(n_eq, -1)).clamp(min=1e-30)  # softmax within each sub-doc
        w = p.sum(0)  # (Tc,) total attention weight on each token, summed over queries
        vstar[:, h] = torch.zeros(n_seg, D, device=dev).index_add_(0, segc, w[:, None] * vc[:, h]) / n_eq
    M[-1, -1] = Hkv * n_eq + mu * n_eq
    rhs = torch.cat([rhs_k.reshape(n_seg, -1), (rhs_c + mu * n_eq * (log_len if log_len is not None else 0.0))[:, None]], 1)
    z = torch.linalg.solve(M, rhs.t()).t()  # (n_seg, Hkv*D+1)
    return z[:, :-1].reshape(n_seg, Hkv, D), vstar, z[:, -1]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="contradiction", choices=["contradiction", "oolong", "nq", "outlier"])
    ap.add_argument("--rung", default="32k")
    ap.add_argument("--rows", type=int, default=24)
    ap.add_argument("--keeps", default="0.3333,0.0833")
    ap.add_argument("--slots", default="1,2,4,8")
    ap.add_argument("--n-random-q", type=int, default=128)
    ap.add_argument("--work", default="/results/probe_work")
    ap.add_argument("--out", default="/results/ceiling.json")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--jsonl", default=None)
    ap.add_argument("--shard", default=None)
    a = ap.parse_args()

    shard = a.shard or f"{a.work}/{a.task}_{a.rung}"
    if a.shard is None:
        P.convert(a.task, a.jsonl or P.EVAL_JSONL[a.task][a.rung], a.rows, shard)
    rows, masks = P.load_rows(shard, a.rows)
    P.log(f"{len(rows)} rows")

    from olmo_core.distributed.checkpoint import load_model_and_optim_state
    from olmo_core.nn.attention import AttentionBackendName
    from olmo_core.nn.lm_head import LMLossImplementation
    from olmo_core.nn.transformer import TransformerConfig

    cfg = TransformerConfig.qwen3_5_4B(vocab_size=P.VOCAB, attn_backend=AttentionBackendName.torch)
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
    stash = QKVStash(model)
    attn_layers = sorted(stash.layers.values())
    scale = model.blocks[str(attn_layers[0])].attention.head_dim ** -0.5

    # chunk-id override: the compaction path reads this module-level function
    current = {"cid": None}
    orig_build = MODEL_MOD.build_chunk_ids_from_tokens
    MODEL_MOD.build_chunk_ids_from_tokens = lambda *args, **kw: current["cid"] if current["cid"] is not None else orig_build(*args, **kw)

    gold_table = json.load(open(f"{shard}/gold_fingerprints.json")) if a.task in P.GOLD_TASKS else None
    keeps = [float(k) for k in a.keeps.split(",")]
    slots_G = [int(g) for g in a.slots.split(",")]
    extra_oolong = 2.0 if a.task == "oolong" else 0.0
    variants = [("soft meanEmb no-bias", "soft", 0.0, 0.0), (f"soft meanEmb +logL{extra_oolong:+.0f}", "soft", 1.0, extra_oolong),
                (f"oracle meanKV +logL{extra_oolong:+.0f}", "oracle", 1.0, extra_oolong), ("oracle meanKV no-bias", "oracle", 0.0, 0.0),
                ("fitted slot (k*, v*, c)", "fit", 0.0, 0.0)]
    names = ["full"] + [f"G={g} k={k:.3f} {v[0]}" for g in slots_G for k in keeps for v in variants]
    res = {n: {"ce": [], "top1": [], "kl": [], "correct": [], "compaction": [], "sec": []} for n in names}
    gen = torch.Generator().manual_seed(a.seed)

    for ri, (row, rmask) in enumerate(zip(rows, masks)):
        x = torch.tensor(row[None], device="cuda")
        T = x.shape[1]
        ans_pos = torch.tensor(np.nonzero(rmask)[0], device="cuda")
        pred_pos = ans_pos - 1
        targets = x[0, ans_pos]
        base_cid = build_chunk_ids_from_tokens(x.cpu(), doc_start_id=IDS.doc_start, doc_end_id=IDS.doc_end, eos_id=IDS.eos, mode="chunked")[0]
        n_docs = int(base_cid.max()) + 1
        if a.task in P.GOLD_TASKS:
            keep_fn = make_fingerprint_keep_docs_fn(gold_table, doc_start_id=IDS.doc_start, doc_end_id=IDS.doc_end, eos_id=IDS.eos,
                                                    n_random_frac=0.0, mode="gold_plus_random", seed=a.seed)
            gold = keep_fn(x.cpu())[0].bool()
        else:
            gold = torch.zeros(n_docs, dtype=torch.bool)
        rand_u = torch.rand(n_docs, generator=gen)  # one draw per doc per row -> nested keep sets across keeps

        model.eval(); stash.qkv.clear(); stash.on = True
        t0 = time.time()
        lf = model(x, logits_to_keep=pred_pos[None])[0].float()
        stash.on = False
        r = res["full"]; r["ce"].append(float(F.cross_entropy(lf, targets))); r["top1"].append(1.0); r["kl"].append(0.0)
        r["correct"].append(float((lf.argmax(-1) == targets).all())); r["compaction"].append(1.0); r["sec"].append(time.time() - t0)
        # query stash: answer-position queries + random queries from the last 15% of the row
        tail = torch.randint(int(0.85 * T), T, (a.n_random_q,), generator=gen).cuda()
        q_pos = torch.cat([pred_pos, tail])

        for G in slots_G:
            sub_cid, parent = split_docs(base_cid, gold, G)
            n_sub = int(parent.numel())
            current["cid"] = sub_cid[None].cuda()
            seg = sub_cid.cuda()
            # per-sub-doc slots from the dense forward: mean K/V, log L, fitted (k*, v*, c)
            lens = torch.bincount(seg[seg >= 0], minlength=n_sub).float().cuda()
            log_len = torch.log(lens.clamp(min=1))
            meanK, meanV, fitK, fitV, fitC = {}, {}, {}, {}, {}
            t_fit = time.time()
            for li in attn_layers:
                q, k, v = stash.qkv[li]
                k0, v0 = k[0].float(), v[0].float()
                segc = seg[seg >= 0]
                mk = torch.zeros(n_sub, k0.shape[1], k0.shape[2], device="cuda").index_add_(0, segc, k0[seg >= 0]) / lens[:, None, None]
                mv = torch.zeros(n_sub, v0.shape[1], v0.shape[2], device="cuda").index_add_(0, segc, v0[seg >= 0]) / lens[:, None, None]
                meanK[li], meanV[li] = mk, mv
                ks, vs, cs = fit_slots(q[0, q_pos], k[0], v[0], seg, n_sub, scale, log_len=log_len, kbar=mk)
                fitK[li], fitV[li], fitC[li] = ks, vs, cs
            fit_sec = time.time() - t_fit
            for keep in keeps:
                keep_parent = gold | (rand_u < keep)
                model._pooled_keep_holder = PooledDocKeepHolder(keep_docs=keep_parent[parent][None].cuda())
                model.train()
                pst["len_bias"] = False
                cb = model._compact_pooled_soft_tokens(x, None, -100)[0]
                posmap = {int(p): c for c, p in enumerate(cb.position_ids[0].tolist())}
                cols = torch.tensor([posmap[int(p)] for p in pred_pos.tolist()], device="cuda")
                comp = cb.input_ids.shape[1] / T
                sd = cb.soft_docs.cuda()  # sub-doc id per slot
                n_slots = int(sd.numel())
                base_bias = build_position_causal_bias(cb, dtype=torch.bfloat16, device=torch.device("cuda"))
                Hkv, D = meanK[attn_layers[0]].shape[1:]
                for vname, kind, sc, extra in variants:
                    name = f"G={G} k={keep:.3f} {vname}"
                    t_cfg = time.time()
                    if kind == "soft":
                        pst["len_bias"], pst["len_bias_scale"], pst["len_bias_extra"] = (sc > 0), sc, extra
                        lg = model(x, logits_to_keep=cols[None])[0].float()
                    else:
                        slots = torch.zeros((n_slots, n_blocks, 2, Hkv, D), dtype=torch.bfloat16, device="cuda")
                        biases = torch.zeros((n_slots, n_blocks), device="cuda")
                        for li in attn_layers:
                            if kind == "oracle":
                                slots[:, li, 0] = meanK[li][sd].to(torch.bfloat16); slots[:, li, 1] = meanV[li][sd].to(torch.bfloat16)
                                biases[:, li] = sc * log_len[sd] + extra
                            else:
                                slots[:, li, 0] = fitK[li][sd].to(torch.bfloat16); slots[:, li, 1] = fitV[li][sd].to(torch.bfloat16)
                                biases[:, li] = fitC[li][sd]
                        ovr = {"rows": cb.soft_rows.cuda(), "cols": cb.soft_cols.cuda(), "pos": torch.zeros(n_slots, dtype=torch.long, device="cuda"),
                               "slots": slots, "biases": biases}
                        pst["len_bias"] = False
                        lg = model(x, logits_to_keep=cols[None], soft_kv_override_layers=ovr, attn_bias=base_bias)[0].float()
                    r = res[name]
                    r["ce"].append(float(F.cross_entropy(lg, targets)))
                    r["top1"].append(float((lg.argmax(-1) == lf.argmax(-1)).float().mean()))
                    r["kl"].append(float(F.kl_div(F.log_softmax(lg, -1), F.log_softmax(lf, -1), log_target=True, reduction="batchmean")))
                    r["correct"].append(float((lg.argmax(-1) == targets).all()))
                    r["compaction"].append(comp); r["sec"].append(time.time() - t_cfg)
                model.eval()
            current["cid"] = None
            if ri == 0:
                P.log(f"G={G}: {n_sub} sub-docs, fit {fit_sec:.1f}s")
        if ri + 1 in (1, 2, 5) or (ri + 1) % 8 == 0:
            P.log(f"row {ri + 1}/{len(rows)}")
            table(res, names)
    stash.restore()
    MODEL_MOD.build_chunk_ids_from_tokens = orig_build
    table(res, names)
    out = {"task": a.task, "rung": a.rung, "rows": len(rows), "ckpt": ck,
           "configs": {n: {k: float(np.mean(v)) for k, v in res[n].items()} for n in names}}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    P.log(f"wrote {a.out}")


def table(res, names):
    print(f"{'config':44} {'answer CE':>9} {'top1=full':>9} {'KL':>7} {'correct':>8} {'compact':>8} {'s':>5}", flush=True)
    for n in names:
        r = res[n]
        if r["ce"]:
            print(f"{n:44} {np.mean(r['ce']):9.3f} {np.mean(r['top1']):9.3f} {np.mean(r['kl']):7.3f} {np.mean(r['correct']):8.2f} {np.mean(r['compaction']):8.3f} {np.mean(r['sec']):5.1f}", flush=True)


if __name__ == "__main__":
    main()
