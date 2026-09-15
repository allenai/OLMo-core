"""
Can a RICHER (but still cheap) slot make a pooled outlier readable?  (eval-only, 2026-09-14)

``records/outlier-slot-probe.md`` answered the prior question with a flat NO: with the training
construction's slot -- the document's **mean input embedding** -- a frozen dense outlier model
recovers pooled gold documents at exactly the ``k/n`` uniform-guess floor, and swapping the gold
documents' slot vectors with random non-gold ones is undetectable, while gold whose BODY stayed
real is recovered at 0.71-0.84.

This probe keeps everything about that setup (same checkpoint, same rung files, same header-real
construction, same metrics, same swap control) and changes only **how the slot vector is built**,
looking for a construction whose ``R@gold_pooled`` lifts off the ``k/n`` floor AND drops under the
swap control.  Outlier only needs the slot to carry TOPIC ("which 3 of n documents are odd"), so
the candidates are all cheap topical summaries rather than trained summarizers.

Constraints every candidate respects (FLOP-optimal + trainable at very long lengths):

* ``O(doc tokens)`` or less to build,
* at most a few extra tokens per document on the compacted row,
* **nothing** per-token-per-layer on the main stack.

Candidates (``--slots``)::

  mean        the training construction: mean input embedding over the doc's body   [baseline]
  cmean100    mean over CONTENT tokens only: drop punctuation/whitespace and the top-100 most
  cmean500    frequent token ids of the eval corpus (K = 100 / 500)
  centered    (mean - corpus mean embedding), renormalised to the typical real-token norm
  cent_cmean  the same centring applied to cmean100
  idf         tokens weighted by -log p(token) over the eval corpus, renormalised
  g2 / g4     G contiguous segment means (halves / quarters), each injected at its OWN centre
  g2cent      G = 2 segment means, centred + renormalised
  enc2 / enc4 mean of the doc's hidden states after a BLOCK-LOCAL forward through the frozen
              model's own first k layers (attention restricted to the doc), renormalised to the
              input-embedding scale because it is injected at layer 0.  Costs k/n_layers of a
              dense forward over the pooled tokens -- reported per condition.
  enc4late    the k = 4 block-local mean injected as the slot's RESIDUAL at layer 4 instead
              (layers 0..3 see the plain mean slot); no renormalisation (it is already a layer-4
              hidden state).

Each candidate runs at keep 0 (``gb00h``: every document pooled -> every gold document is a slot
read) and at gold-blind keep 1/6 (``gb17h``), always header-real (``--st-header-stop-id 5491
--st-header-stop-count 1``) so document ids are readable, and each gb00h candidate also runs its
``_swap`` control.  ``full`` is the dense reference.

    python debug/pooled_kv/outlier_probe/outlier_richer_slot_probe.py --rung 2k --rows 240 \
        --gen-rows 64 --ckpt-name ds64-outlier-dense-u64M
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, ".."))

import eval_side_slot_probe as P  # noqa: E402
import outlier_slot_probe as O  # noqa: E402  (shared: CKPTS/RUNGS/parse_ids/set_f1/generate)

from olmo_core.distributed.checkpoint import load_model_and_optim_state  # noqa: E402
from olmo_core.nn.attention.chunked_mask import (  # noqa: E402
    build_chunk_ids_from_tokens,
    mark_doc_headers_free,
)
from olmo_core.nn.attention.gold_grad_mask import content_fingerprint_from_row  # noqa: E402
from olmo_core.nn.attention.pooled_doc_kv import resolve_keep_docs  # noqa: E402
from olmo_core.nn.lm_head import LMLossImplementation  # noqa: E402
from olmo_core.nn.pooled_soft_token import compact_pooled_rows  # noqa: E402

IDS = O.IDS
W = O.W
HEADER_STOP_ID = O.HEADER_STOP_ID
parse_ids, set_f1, generate = O.parse_ids, O.set_f1, O.generate

# ---------------------------------------------------------------------------------------------
# slot constructions
# ---------------------------------------------------------------------------------------------
# G      : segment means per document (G tokens on the compacted row, each at its own centre)
# drop   : drop punctuation/whitespace + the top-`drop` corpus token ids from the mean
# idf    : weight tokens by -log p(token)
# center : subtract the (identically weighted) corpus mean embedding
# renorm : rescale the slot to the frequency-weighted mean norm of a real token embedding
# enc_k  : block-local forward through the frozen model's first k layers, mean of the hiddens
# late   : inject the enc vector as the slot RESIDUAL at layer enc_k (layers < k see `mean`)
SLOTS = {
    "mean": dict(),
    "cmean100": dict(drop=100),
    "cmean500": dict(drop=500),
    "centered": dict(center=True, renorm=True),
    "cent_cmean": dict(drop=100, center=True, renorm=True),
    "idf": dict(idf=True, renorm=True),
    "g2": dict(G=2),
    "g4": dict(G=4),
    "g2cent": dict(G=2, center=True, renorm=True),
    "enc2": dict(enc_k=2, renorm=True),
    "enc4": dict(enc_k=4, renorm=True),
    "enc4late": dict(enc_k=4, late=True),
}
DEFAULT_SLOTS = ",".join(SLOTS)

# per-condition state the patched compaction reads
ST = {
    "cond": None,  # cache key (row, name, phase)
    "params": {},
    "header": True,
    "keep": None,  # (1, n_docs_orig) bool, CPU
    "gold": None,  # (n_docs_orig,) bool, CPU
    "swap": False,
    "seed": 0,
    "corpus": None,
    "model": None,
    "enc_cache": {},  # (row, name) -> feats
    "stats": {},  # filled per forward: tokens pooled / compacted length
}
_CACHE = {"key": None, "out": None}
_LATE = {"on": False, "layer": 0, "rows": None, "cols": None, "feats": None}


def log(m):
    print(f"[richer-slot] {m}", flush=True)


# ---------------------------------------------------------------------------------------------
# corpus statistics (computed once over the loaded eval rows)
# ---------------------------------------------------------------------------------------------


def corpus_stats(rows, tok, model, drop_ks=(100, 500)):
    """Token frequencies over the CONTEXT tokens of the eval shard, the drop sets, the idf
    weights, and the frequency-weighted mean embedding / mean embedding norm."""
    cnt = Counter()
    for r in rows:
        cid = build_chunk_ids_from_tokens(
            torch.tensor(r[None]),
            doc_start_id=IDS.doc_start,
            doc_end_id=IDS.doc_end,
            eos_id=IDS.eos,
            mode="chunked",
        )[0].numpy()
        cnt.update(np.asarray(r)[cid >= 0].tolist())
    total = sum(cnt.values())
    uniq = np.array(sorted(cnt), dtype=np.int64)
    counts = np.array([cnt[int(t)] for t in uniq], dtype=np.float64)
    order = np.argsort(-counts)
    # punctuation / whitespace: decoded piece has no alphanumeric character
    punct = []
    for t in uniq.tolist():
        s = tok.decode([t])
        if not any(ch.isalnum() for ch in s):
            punct.append(t)
    punct = set(punct)
    V = int(model.embeddings.weight.shape[0])
    drops = {}
    for K in drop_ks:
        top = set(uniq[order[:K]].tolist())
        drops[K] = torch.zeros(V, dtype=torch.bool)
        for t in top | punct:
            drops[K][t] = True
    # idf weight, -log p, on the full vocab (unseen tokens get the max weight)
    p = np.full(V, 1.0 / (total + 1.0))
    p[uniq] = counts / total
    idf = torch.from_numpy(-np.log(p)).float()
    dev = model.embeddings.weight.device
    emb = model.embeddings.weight.detach()
    idx = torch.from_numpy(uniq).to(dev)
    wt = torch.from_numpy(counts / total).to(dev).to(emb.dtype)
    rows_e = emb[idx].float()
    mean_emb = (rows_e * wt[:, None].float()).sum(0)
    mean_norm = float((rows_e.norm(dim=-1) * wt.float()).sum())
    # content-weighted corpus mean (the cmean/cent_cmean centre)
    keepm = ~drops[100][idx.cpu()].to(dev)
    wt2 = wt.float() * keepm.float()
    mean_emb_c = (rows_e * wt2[:, None]).sum(0) / wt2.sum().clamp(min=1e-6)
    # idf-weighted corpus mean
    wi = wt.float() * idf.to(dev)[idx]
    mean_emb_i = (rows_e * wi[:, None]).sum(0) / wi.sum().clamp(min=1e-6)
    log(
        f"corpus: {total} ctx tokens, {len(uniq)} unique, {len(punct)} punct/ws ids, "
        f"mean |emb| = {mean_norm:.4f}, |plain corpus mean| = {float(mean_emb.norm()):.4f}"
    )
    return {
        "drop": {K: v.to(dev) for K, v in drops.items()},
        "idf": idf.to(dev),
        "mean_emb": mean_emb,
        "mean_emb_c": mean_emb_c,
        "mean_emb_i": mean_emb_i,
        "mean_norm": mean_norm,
    }


# ---------------------------------------------------------------------------------------------
# slot feature builders
# ---------------------------------------------------------------------------------------------


def _seg_split(cid, n_docs, G):
    """Split every document's body into G contiguous segments; segment id = doc * G + g."""
    if G == 1:
        return cid, n_docs
    out = cid.clone().to(torch.long)
    B, T = cid.shape
    for b in range(B):
        c = cid[b].to(torch.long)
        ctx = c >= 0
        if not bool(ctx.any()):
            continue
        pos = ctx.nonzero(as_tuple=True)[0]
        d = c[ctx]
        first = torch.full((n_docs,), T, dtype=torch.long, device=c.device)
        first.scatter_reduce_(0, d, pos, reduce="amin", include_self=True)
        cntd = torch.bincount(d, minlength=n_docs).clamp(min=1)
        rank = pos - first[d]
        g = torch.div(rank * G, cntd[d], rounding_mode="floor").clamp(max=G - 1)
        out[b, pos] = d * G + g
    return out, n_docs * G


def _weighted_doc_means(emb, cid, n_seg, w=None):
    """(B, n_seg, D) weighted mean of ``emb`` over each segment's tokens (w = None -> plain mean),
    plus the (B, n_seg) weight mass so callers can fall back where a segment lost every token."""
    B, T, D = emb.shape
    c = cid.to(torch.long)
    is_ctx = c >= 0
    flat = (torch.arange(B, device=emb.device)[:, None] * n_seg + c.clamp(min=0)).reshape(-1)[
        is_ctx.reshape(-1)
    ]
    e = emb.reshape(B * T, D)[is_ctx.reshape(-1)].float()
    ww = (
        torch.ones(e.shape[0], device=emb.device)
        if w is None
        else w.reshape(-1)[is_ctx.reshape(-1)].float()
    )
    sums = torch.zeros(B * n_seg, D, device=emb.device).index_add(0, flat, e * ww[:, None])
    mass = torch.zeros(B * n_seg, device=emb.device).index_add(0, flat, ww)
    return (sums / mass.clamp(min=1e-6)[:, None]).reshape(B, n_seg, D), mass.reshape(B, n_seg)


@torch.no_grad()
def _enc_feats(model, input_ids, cid, n_seg, want_docs, k):
    """Mean hidden state at layer ``k`` of a BLOCK-LOCAL forward: each wanted document is its own
    sequence (right-padded), attention/recurrence never crosses a document, positions restart at 0.
    Cost = k / n_layers of a dense forward over those documents' tokens."""
    dev = input_ids.device
    c = cid[0].to(torch.long)
    sel = torch.zeros(n_seg, dtype=torch.bool, device=dev)
    sel[want_docs] = True
    ctx = (c >= 0) & sel[c.clamp(min=0)]
    pos = ctx.nonzero(as_tuple=True)[0]
    d = c[ctx]
    # compact the wanted doc ids to 0..m-1
    remap = torch.full((n_seg,), -1, dtype=torch.long, device=dev)
    remap[want_docs] = torch.arange(len(want_docs), device=dev)
    dd = remap[d]
    m = len(want_docs)
    first = torch.full((m,), input_ids.shape[1], dtype=torch.long, device=dev)
    first.scatter_reduce_(0, dd, pos, reduce="amin", include_self=True)
    lens = torch.bincount(dd, minlength=m)
    Lmax = int(lens.max().clamp(min=1))
    rank = pos - first[dd]
    padded = torch.full((m, Lmax), IDS.eos, dtype=input_ids.dtype, device=dev)
    padded[dd, rank] = input_ids[0][ctx]
    real = torch.zeros((m, Lmax), dtype=torch.bool, device=dev)
    real[dd, rank] = True

    h = model.embeddings(padded)
    if model.embed_scale is not None:
        h = h * model.embed_scale
    if model.embedding_norm is not None:
        h = model.embedding_norm(h)
    for i in range(k):
        h = model.blocks[str(i)](h)
    rm = real[..., None].to(h.dtype)
    feats = (h.float() * rm.float()).sum(1) / rm.float().sum(1).clamp(min=1.0)
    return feats, int(ctx.sum()), Lmax


def _enc_cached(model, input_ids, cid_seg, n_seg, inj_docs, k, row_key):
    """The documents do not change between a row's CE forward and its generation steps, so the
    block-local encode is done once per (row, k, document set)."""
    key = (row_key, int(k), int(inj_docs.numel()), int(inj_docs.sum()))
    hit = ST["enc_cache"].get(key)
    if hit is None:
        feats, ntok, _ = _enc_feats(model, input_ids, cid_seg, n_seg, inj_docs, k)
        ST["enc_cache"] = {key: (feats, ntok)}
        hit = (feats, ntok)
    return hit


def _renorm(v, target):
    n = v.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    return v / n * target


@torch.no_grad()
def build_slot_feats(model, input_ids, cid_seg, n_seg, inj_docs, params, row_key):
    """(len(inj_docs), D) slot vectors for the requested segments."""
    cs = ST["corpus"]
    emb = model.embeddings(input_ids)
    G = params.get("G", 1)
    enc_k = params.get("enc_k")
    extra = {"enc_tokens": 0, "enc_layers": 0}

    if enc_k and not params.get("late"):
        feats, ntok = _enc_cached(model, input_ids, cid_seg, n_seg, inj_docs, enc_k, row_key)
        extra = {"enc_tokens": ntok, "enc_layers": enc_k}
        return _renorm(feats, cs["mean_norm"]), extra

    w = None
    centre = cs["mean_emb"]
    if params.get("drop"):
        w = (~cs["drop"][params["drop"]][input_ids]).float()
        centre = cs["mean_emb_c"]
    elif params.get("idf"):
        w = cs["idf"][input_ids]
        centre = cs["mean_emb_i"]

    plain, _ = _weighted_doc_means(emb, cid_seg, n_seg)
    if w is None:
        feats = plain[0, inj_docs]
    else:
        wm, mass = _weighted_doc_means(emb, cid_seg, n_seg, w)
        feats = torch.where(mass[0, inj_docs, None] > 1e-5, wm[0, inj_docs], plain[0, inj_docs])
    if params.get("center"):
        feats = feats - centre[None, :]
    if params.get("renorm"):
        feats = _renorm(feats, cs["mean_norm"])
    del G
    return feats, extra


# ---------------------------------------------------------------------------------------------
# the patched compaction: chunk ids -> (optional) segment split -> keep -> slot feats
# ---------------------------------------------------------------------------------------------


def install_patch(model):
    cls = type(model)

    def patched(self, input_ids, labels, ignore_index):
        key = (ST["cond"], int(input_ids.data_ptr()), int(input_ids.shape[1]))
        if _CACHE["key"] == key:
            return _CACHE["out"]
        out = _build(self, input_ids, labels, ignore_index)
        _CACHE["key"], _CACHE["out"] = key, out
        return out

    cls._compact_pooled_soft_tokens = patched

    def make_hook(idx):
        def hook(module, args, output):
            if (
                _LATE["on"]
                and idx == _LATE["layer"] - 1
                and _LATE["rows"] is not None
                and output.shape[1] == _LATE["T"]
            ):
                out = output.clone()
                out[_LATE["rows"], _LATE["cols"]] = _LATE["feats"].to(out.dtype)
                return out
            return output

        return hook

    for bk, blk in model.blocks.items():
        blk.register_forward_hook(make_hook(int(bk)))


def _build(model, input_ids, labels, ignore_index):
    cfg = model._pooled_soft_tokens
    params = ST["params"]
    G = params.get("G", 1)
    cid = build_chunk_ids_from_tokens(
        input_ids,
        doc_start_id=IDS.doc_start,
        doc_end_id=IDS.doc_end,
        eos_id=IDS.eos,
        mode="chunked",
    )
    n_docs = int(cid.max().item()) + 1
    if n_docs <= 0:
        return None
    if ST["header"]:
        cid = mark_doc_headers_free(
            cid,
            input_ids,
            doc_start_id=IDS.doc_start,
            doc_end_id=IDS.doc_end,
            stop_id=HEADER_STOP_ID,
            stop_count=1,
            cap=32,
        )
    cid_seg, n_seg = _seg_split(cid, n_docs, G)
    keep = ST["keep"][:, :n_docs].to(input_ids.device)
    keep_seg = keep.repeat_interleave(G, dim=1) if G > 1 else keep

    cb = compact_pooled_rows(
        input_ids,
        labels,
        cid_seg,
        keep_seg,
        placeholder_id=cfg["placeholder_id"],
        pad_token_id=cfg["eos_id"],
        ignore_index=ignore_index,
        add_shadows=False,
    )
    inj_rows, inj_cols, inj_docs = cb.soft_rows, cb.soft_cols, cb.soft_docs
    if inj_docs.numel() == 0:
        ST["stats"] = {"t_in": input_ids.shape[1], "t_out": cb.input_ids.shape[1],
                       "pooled_tokens": 0, "enc_tokens": 0, "enc_layers": 0, "slot_norm": 0.0}
        return cb, (inj_rows, inj_cols, torch.zeros(0, model.embeddings.weight.shape[1],
                                                    device=input_ids.device)), None

    feats, extra = build_slot_feats(
        model, input_ids, cid_seg, n_seg, inj_docs, params, ST["cond"][0] if ST["cond"] else 0
    )
    if ST["swap"]:
        feats = _swap_feats(feats, inj_docs, G)

    # enc4late: layers < k see the plain mean slot; the enc vector replaces the slot residual at k
    _LATE["on"] = False
    if params.get("late"):
        enc, ntok, _ = _enc_feats(model, input_ids, cid_seg, n_seg, inj_docs, params["enc_k"])
        if ST["swap"]:
            enc = _swap_feats(enc, inj_docs, G)
        extra = {"enc_tokens": ntok, "enc_layers": params["enc_k"]}
        _LATE.update(
            on=True,
            layer=params["enc_k"],
            rows=inj_rows,
            cols=inj_cols,
            feats=enc,
            T=cb.input_ids.shape[1],
        )

    pooled_tokens = int(((cid_seg >= 0) & ~keep_seg[0][cid_seg.clamp(min=0)]).sum())
    ST["stats"] = {
        "t_in": int(input_ids.shape[1]),
        "t_out": int(cb.input_ids.shape[1]),
        "pooled_tokens": pooled_tokens,
        "enc_tokens": extra["enc_tokens"],
        "enc_layers": extra["enc_layers"],
        "slot_norm": float(feats.norm(dim=-1).mean()),
    }
    return cb, (inj_rows, inj_cols, feats.to(model.embeddings.weight.dtype)), None


def _swap_feats(feats, inj_docs, G):
    """CONTROL: exchange the gold documents' slot vectors with an equal number of random non-gold
    documents' (segment-wise when G > 1). Positions, ids and lengths are untouched."""
    gold = ST["gold"]
    dl = [int(d) // G for d in inj_docs.tolist()]
    gi = [i for i, d in enumerate(dl) if d < len(gold) and bool(gold[d])]
    ni = [i for i, d in enumerate(dl) if not (d < len(gold) and bool(gold[d]))]
    if not gi or len(ni) < len(gi):
        return feats
    g = torch.Generator().manual_seed(ST["seed"] + len(dl))
    pick = [ni[j] for j in torch.randperm(len(ni), generator=g)[: len(gi)].tolist()]
    out = feats.clone()
    a = out[gi].clone()
    out[gi] = out[pick]
    out[pick] = a
    return out


# ---------------------------------------------------------------------------------------------


def make_conditions(slot_names, keeps, swaps):
    conds = [("full", None, False, "-", False)]
    for kname, kp in keeps:
        for s in slot_names:
            conds.append((f"{kname}_{s}", kp, True, s, False))
            if kname in swaps:
                conds.append((f"{kname}_{s}_swap", kp, True, s, True))
    return conds


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rung", default="2k")
    ap.add_argument("--rows", type=int, default=240)
    ap.add_argument("--gen-rows", type=int, default=64)
    ap.add_argument("--gen-max-new", type=int, default=64)
    ap.add_argument("--ckpt-name", default="ds64-outlier-dense-u64M")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--jsonl", default=None)
    ap.add_argument("--shard", default=None)
    ap.add_argument("--slots", default=DEFAULT_SLOTS)
    ap.add_argument("--keeps", default="gb00h:0,gb17h:0.1667")
    ap.add_argument("--swaps", default="gb00h", help="comma list of keep names that also get _swap controls")
    ap.add_argument("--conditions", default="", help="explicit condition subset (comma list)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--work", default="/results/richer_slot_work")
    ap.add_argument("--out", default="/results/outlier_richer_slot_probe.json")
    ap.add_argument("--weka-out", default=f"{W}/_eval_results/outlier_slot_probe")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    if a.tokenizer:
        P.TOKENIZER = a.tokenizer
    slot_names = [s for s in a.slots.split(",") if s]
    bad = [s for s in slot_names if s not in SLOTS]
    if bad:
        raise SystemExit(f"unknown slots {bad}; known: {list(SLOTS)}")
    keeps = [(kv.split(":")[0], float(kv.split(":")[1])) for kv in a.keeps.split(",") if kv]
    conds = make_conditions(slot_names, keeps, set(a.swaps.split(",")))
    if a.conditions:
        want = set(a.conditions.split(",")) | {"full"}
        conds = [c for c in conds if c[0] in want]
    log(f"{len(conds)} conditions: {[c[0] for c in conds]}")

    shard = a.shard or f"{a.work}/outlier_{a.rung}"
    if a.shard is None:
        P.convert("outlier", a.jsonl or O.RUNGS[a.rung], a.rows, shard)
    rows, masks = P.load_rows(shard, a.rows)
    log(f"{len(rows)} rows; lengths {[len(r) for r in rows[:6]]}")
    if len(rows) < 500:
        log(f"WARNING eval_size={len(rows)} (<500): binomial SE at f1~0.5 is "
            f"{0.5 / max(1, len(rows)) ** 0.5:.3f}")

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(P.TOKENIZER)

    cfg = P.build_cfg()
    cfg.lm_head.loss_implementation = LMLossImplementation.default
    model = cfg.build(init_device="cpu")
    ck = P.find_ckpt(a.ckpt or O.CKPTS[a.ckpt_name])
    t0 = time.time()
    load_model_and_optim_state(ck, model)
    log(f"loaded {ck} in {time.time() - t0:.0f}s")
    model.enable_pooled_soft_tokens(
        IDS.doc_start,
        IDS.doc_end,
        IDS.eos,
        placeholder_id=IDS.landmark,
        keep_prob=0.0,
        keep_seed=a.seed,
        detach_soft_kv=True,
    )
    model.pooled_projector.reset_parameters()  # identity projector: the slot IS the feature
    model = model.cuda().to(torch.bfloat16)
    model._pooled_keep_holder = None
    n_layers = len(model.blocks)
    install_patch(model)
    ST["seed"] = a.seed

    # RoPE: rope.forward sizes its absolute-position sin/cos buffer from the CACHE, not from the
    # positions it is handed, and a compacted row is short while carrying ORIGINAL positions that
    # free generation walks past the row's length -> silent out-of-bounds gather. Warm every cache.
    max_pos = max(len(r) for r in rows) + a.gen_max_new + 8
    n_warm = 0
    for mod in model.modules():
        rope = getattr(mod, "rope", None)
        if rope is not None and hasattr(rope, "warmup_cache"):
            rope.warmup_cache(max_pos, torch.device("cuda"))
            n_warm += 1
    log(f"warmed {n_warm} RoPE caches to {max_pos} positions")

    ST["corpus"] = corpus_stats(rows, tok, model)

    gold_table = json.load(open(f"{shard}/gold_fingerprints.json"))
    gold_table = {
        fp: sorted({int(i) for v in val for i in (v if isinstance(v, (list, tuple)) else [v])})
        for fp, val in gold_table.items()
    }
    log(f"gold sidecar: {len(gold_table)} fingerprints")

    keys = ["ce", "ce_digit", "top1", "kl", "tf_em", "tf_f1", "tf_id1", "gen_f1", "gen_em",
            "hit_gold_pooled", "hit_gold_real", "pred_is_pooled", "base_pooled", "compaction",
            "slot_norm", "enc_frac_dense", "enc_frac_compact", "sec", "rowsplit"]
    acc = {c[0]: {k: [] for k in keys} for c in conds}
    id_offset_votes = Counter()
    n_fp_miss = 0

    for ri, (row, rmask) in enumerate(zip(rows, masks)):
        x = torch.tensor(row[None], device="cuda")
        ans_pos = torch.tensor(np.nonzero(rmask)[0], device="cuda")
        pred_pos = ans_pos - 1
        targets = x[0, ans_pos]
        true_text = tok.decode(targets.tolist())
        true_ids = parse_ids(true_text)
        pieces = [tok.decode([int(t)]) for t in targets.tolist()]
        digit_sel = torch.tensor(
            [i for i, q in enumerate(pieces) if any(ch.isdigit() for ch in q)],
            device="cuda", dtype=torch.long,
        )
        ans_start = int(ans_pos[0])

        fp = content_fingerprint_from_row(row.tolist(), IDS.eos)
        gold_docs = gold_table.get(fp)
        if gold_docs is None:
            n_fp_miss += 1
            log(f"row {ri}: gold fingerprint MISS -- skipped")
            continue
        cid_plain = build_chunk_ids_from_tokens(
            x.cpu(), doc_start_id=IDS.doc_start, doc_end_id=IDS.doc_end, eos_id=IDS.eos,
            mode="chunked",
        )
        n_docs_row = int(cid_plain.max()) + 1
        gold_row = torch.zeros(n_docs_row, dtype=torch.bool)
        gold_row[[d for d in gold_docs if 0 <= d < n_docs_row]] = True
        ST["gold"] = gold_row
        off = 1
        if true_ids and gold_docs and len(true_ids) == len(gold_docs):
            diffs = {t - g for t, g in zip(sorted(true_ids), sorted(gold_docs))}
            if len(diffs) == 1:
                off = diffs.pop()
        id_offset_votes[off] += 1

        # header-real chunk ids (all soft conditions use them) -> which documents are present
        cid_h = mark_doc_headers_free(
            cid_plain, x.cpu(), doc_start_id=IDS.doc_start, doc_end_id=IDS.doc_end,
            stop_id=HEADER_STOP_ID, stop_count=1, cap=32,
        )
        present = [int(v) for v in torch.unique(cid_h[0]).tolist() if v >= 0]

        # one keep draw per keep level, shared by every slot candidate at that level
        keep_by_level = {}
        for kname, kp in keeps:
            if kp <= 0:
                keep_by_level[kname] = torch.zeros(1, n_docs_row, dtype=torch.bool)
            else:
                keep_by_level[kname] = resolve_keep_docs(
                    cid_h, n_docs_row, holder=None, keep_prob=float(kp), keep_seed=a.seed
                )[0:1].cpu()

        full_lg = None
        do_gen = ri < a.gen_rows
        for name, keep_spec, header, slot, swap in conds:
            t_cfg = time.time()
            ST["cond"] = (ri, name, "ce")
            _CACHE["key"] = None
            ST["header"] = header
            ST["swap"] = swap
            ST["params"] = SLOTS.get(slot, {})
            _LATE["on"] = False

            if keep_spec is None:  # FULL
                model.eval()
                lg = model(x, logits_to_keep=pred_pos[None])[0].float()
                full_lg = lg
                keep_mask = None
                st = {"t_in": x.shape[1], "t_out": x.shape[1], "pooled_tokens": 0,
                      "enc_tokens": 0, "enc_layers": 0, "slot_norm": float("nan")}
            else:
                kname = name.split("_")[0]
                keep_mask = keep_by_level[kname][0]
                ST["keep"] = keep_by_level[kname]
                model.train()
                cb = model._compact_pooled_soft_tokens(x, None, -100)[0]
                posmap = {int(p): c for c, p in enumerate(cb.position_ids[0].tolist())}
                cols = torch.tensor([posmap[int(p)] for p in pred_pos.tolist()], device="cuda")
                assert int(cols.max()) < cb.input_ids.shape[1], "compaction/column mismatch"
                lg = model(x, logits_to_keep=cols[None])[0].float()
                st = dict(ST["stats"])

            r = acc[name]
            r["ce"].append(float(F.cross_entropy(lg, targets)))
            if digit_sel.numel():
                r["ce_digit"].append(float(F.cross_entropy(lg[digit_sel], targets[digit_sel])))
            r["top1"].append(float((lg.argmax(-1) == full_lg.argmax(-1)).float().mean()))
            r["kl"].append(float(F.kl_div(F.log_softmax(lg, -1), F.log_softmax(full_lg, -1),
                                          log_target=True, reduction="batchmean")))
            r["tf_em"].append(float((lg.argmax(-1) == targets).all()))
            tf_ids = parse_ids(tok.decode(lg.argmax(-1).tolist()))
            r["tf_f1"].append(set_f1(tf_ids, true_ids))
            if true_ids:
                r["tf_id1"].append(float(bool(tf_ids) and tf_ids[0] == true_ids[0]))
            r["compaction"].append(st["t_out"] / max(1, st["t_in"]))
            r["slot_norm"].append(st["slot_norm"])
            # extra FLOPs of the slot builder, as a fraction of (a) a DENSE forward over the row
            # and (b) the compacted forward this construction actually runs. Everything that is
            # not `enc*` is O(doc tokens) embedding arithmetic: no matmuls, charged 0.
            r["enc_frac_dense"].append(
                st["enc_layers"] * st["enc_tokens"] / max(1, n_layers * st["t_in"])
            )
            r["enc_frac_compact"].append(
                st["enc_layers"] * st["enc_tokens"] / max(1, n_layers * st["t_out"])
            )

            if keep_mask is None:
                pooled_gold, real_gold, base_pooled = [], list(gold_docs), 0.0
            else:
                pooled_gold = [d for d in gold_docs if d < len(keep_mask) and not bool(keep_mask[d])]
                real_gold = [d for d in gold_docs if d < len(keep_mask) and bool(keep_mask[d])]
                base_pooled = (
                    float(np.mean([0.0 if bool(keep_mask[d]) else 1.0 for d in present]))
                    if present else 0.0
                )
            r["base_pooled"].append(base_pooled)
            r["rowsplit"].append(len(real_gold))

            if do_gen:
                ST["cond"] = (ri, name, "gen")
                _CACHE["key"] = None
                model.train() if keep_spec is not None else model.eval()
                gen = generate(model, x[:, :ans_start].clone(), a.gen_max_new, len(true_ids) or 3, tok)
                gtext = tok.decode(gen)
                gen_ids = parse_ids(gtext)
                r["gen_f1"].append(set_f1(gen_ids, true_ids))
                r["gen_em"].append(
                    float(set(gen_ids) == set(true_ids) and len(gen_ids) == len(true_ids))
                )
                pred_docs = [i - off for i in gen_ids]
                if pooled_gold:
                    r["hit_gold_pooled"] += [1.0 if d in pred_docs else 0.0 for d in pooled_gold]
                if real_gold:
                    r["hit_gold_real"] += [1.0 if d in pred_docs else 0.0 for d in real_gold]
                if keep_mask is not None:
                    for d in pred_docs:
                        if 0 <= d < len(keep_mask):
                            r["pred_is_pooled"].append(0.0 if bool(keep_mask[d]) else 1.0)
            r["sec"].append(time.time() - t_cfg)
            model.eval()
            _LATE["on"] = False
        ST["enc_cache"] = {}

        if ri + 1 in (1, 2, 5, 10) or (ri + 1) % 10 == 0:
            done = ri + 1
            el = sum(sum(acc[c[0]]["sec"]) for c in conds)
            log(f"row {done}/{len(rows)}  elapsed {el / 60:.1f} min  "
                f"ETA {el / done * (len(rows) - done) / 60:.0f} min")
            table(acc, conds)

    table(acc, conds)
    log(f"answer-id offset votes: {dict(id_offset_votes)}")
    out = {
        "task": "outlier", "rung": a.rung, "eval_size": len(acc["full"]["ce"]),
        "rows_loaded": len(rows), "ckpt": ck, "ckpt_name": a.ckpt_name,
        "gen_rows": min(a.gen_rows, len(rows)), "header_stop_id": HEADER_STOP_ID,
        "seed": a.seed, "n_layers": n_layers,
        "slots": {s: SLOTS[s] for s in slot_names},
        "fingerprint_misses": n_fp_miss,
        "id_offset_votes": {str(k): v for k, v in id_offset_votes.items()},
        "conditions": {n: summarize(acc[n]) for n, *_ in conds},
    }
    sfx = f"_{a.tag}" if a.tag else ""
    weka = None if a.weka_out in ("", "none") else (
        f"{a.weka_out}/outlier_richer_slot_{a.ckpt_name}_{a.rung}{sfx}.json"
    )
    for path in [a.out] + ([weka] if weka else []):
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            json.dump(out, open(path, "w"), indent=1)
            log(f"wrote {path}")
        except Exception as e:
            log(f"could not write {path}: {e}")


def summarize(r):
    d = {}
    for k, v in r.items():
        if k == "rowsplit":
            d["rows_allgoldpooled"] = int(sum(1 for z in v if z == 0))
            d["rows_somegoldreal"] = int(sum(1 for z in v if z > 0))
            continue
        vv = [z for z in v if z == z]
        d[k] = float(np.mean(vv)) if vv else float("nan")
        d[f"{k}_count"] = len(vv)
        if k in ("hit_gold_pooled", "hit_gold_real") and vv:
            p = float(np.mean(vv))
            d[f"{k}_se"] = float((p * (1 - p) / len(vv)) ** 0.5)
    return d


def table(acc, conds):
    print(
        f"{'condition':22} {'CE':>7} {'CEdig':>7} {'top1':>6} {'tfID1':>6} {'genF1':>6} "
        f"{'R@gold_pooled':>18} {'R@gold_real':>14} {'predP':>6} {'baseP':>6} "
        f"{'|slot|':>7} {'compact':>8} {'encFr':>6} {'s/row':>6}",
        flush=True,
    )
    for name, *_ in conds:
        r = acc[name]
        if not r["ce"]:
            continue
        m = lambda k: (np.mean([z for z in r[k] if z == z]) if any(z == z for z in r[k]) else float("nan"))  # noqa: E731
        hp, hr = r["hit_gold_pooled"], r["hit_gold_real"]
        se = (lambda v: (np.mean(v) * (1 - np.mean(v)) / len(v)) ** 0.5 if v else float("nan"))
        print(
            f"{name:22} {m('ce'):7.3f} {m('ce_digit'):7.3f} {m('top1'):6.3f} {m('tf_id1'):6.3f} "
            f"{m('gen_f1'):6.3f} {m('hit_gold_pooled'):7.3f}±{se(hp):.3f}[{len(hp):4d}] "
            f"{m('hit_gold_real'):7.3f}[{len(hr):3d}] {m('pred_is_pooled'):6.3f} "
            f"{m('base_pooled'):6.3f} {m('slot_norm'):7.3f} {m('compaction'):8.3f} "
            f"{m('enc_frac_dense'):6.3f} {m('sec'):6.2f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
