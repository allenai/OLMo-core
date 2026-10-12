"""
Layer-wise soft-token probe (eval only): does the pooled-doc soft-token construction have to be
applied at EVERY layer?

Background: :mod:`debug.pooled_kv.eval_side_slot_probe` holds a dense-trained model fixed and
compares FULL attention against SOFT (every pooled document collapsed to one slot token, sequence
compacted) on held-out rows. That is an all-or-nothing choice. Here the choice is made PER LAYER:

* a **dense** layer runs on the full sequence exactly as the FULL baseline does;
* a **soft-only** layer runs on the COMPACTED sequence (kept real tokens + one slot per pooled
  doc at the doc's centre RoPE position, ``position_ids`` carrying original positions). The real
  tokens of the pooled documents are not computed at all in that layer -- their hidden state
  passes through by the residual identity -- and nothing in that layer attends to them.

With no soft layers this reduces exactly to FULL; with every layer soft and
``--slot-source embed_mean`` it reduces exactly to the SOFT condition of the existing probe (the
slot state is initialised from the mean input embedding and then carried through the stack, which
is what the ordinary compacted forward does). Both reductions are asserted by ``--selftest``.

Slot sources (``--slot-source``):

* ``embed_mean`` -- slot initialised once from the pooled doc's mean INPUT embedding (identity
  projector, i.e. the training construction) and carried forward; frozen across dense layers.
* ``layer_input_mean`` -- at the first soft layer of every soft RUN the slot is (re)built as the
  mean of the doc's real hidden states at that layer's input, then carried through the rest of the
  run. Combined with a ``topK`` pattern this is the "read-then-compress" idea: the pooled documents
  are really read for the first L layers and summarised once.

Costs are reported as the fraction of block FLOPs removed, both token-linear only and including
the quadratic attention term at the row's actual length.

    python research/token_routing/pooled_kv/layer_probe/layer_soft_probe.py --task contradiction \
        --shard /scratch/users/prasann/slot_probe/contra_iid_8192 --rows 200 --keep 0.0833 \
        --patterns none,all,top50,bottom50,alt,gdn,attn --out /data/prasann/layer_probe/x.json
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from eval_side_slot_probe import (  # noqa: E402  (reused helpers; this module is never edited here)
    _PREFIX_REAL,
    CKPT,
    EVAL_JSONL,
    FAMILY,
    GOLD_TASKS,
    IDS,
    build_cfg,
    convert,
    find_ckpt,
    install_prefix_real_patch,
    load_rows,
    log,
)

from olmo_core.distributed.checkpoint import load_model_and_optim_state  # noqa: E402
from olmo_core.nn.attention.pooled_doc_kv import (  # noqa: E402
    PooledDocKeepHolder,
    make_fingerprint_keep_docs_fn,
)
from olmo_core.nn.lm_head import LMLossImplementation  # noqa: E402

# ---------------------------------------------------------------------------------------------
# layer-set patterns
# ---------------------------------------------------------------------------------------------


def layer_set(spec: str, n_layers: int, gdn_layers, attn_layers):
    """Resolve a pattern spec to the sorted set of SOFT-ONLY layer indices.

    Specs: ``none`` | ``all`` | ``topP`` / ``bottomP`` (P = percent, e.g. ``top50``) |
    ``alt`` / ``alt0`` / ``alt1`` (every other layer) | ``everyK`` (every K-th layer) |
    ``gdn`` / ``attn`` (all layers of one mixer type) | ``gdn_topP`` / ``attn_topP`` |
    ``fromL`` (layers >= L, the read-then-compress cut) | ``list:0,1,2``.
    """
    s = spec.strip()
    if s in ("none", "", "droppooled"):
        return []
    if s == "all":
        return list(range(n_layers))
    if s.startswith("list:"):
        return sorted({int(i) for i in s[len("list:") :].split(",") if i != ""})
    if s.startswith("from"):
        return list(range(int(s[len("from") :]), n_layers))
    if s.startswith("every"):
        k = int(s[len("every") :])
        return [i for i in range(n_layers) if i % k == 0]
    if s in ("alt", "alt0"):
        return [i for i in range(n_layers) if i % 2 == 0]
    if s == "alt1":
        return [i for i in range(n_layers) if i % 2 == 1]
    pool = list(range(n_layers))
    if s.startswith("gdn"):
        pool, s = list(gdn_layers), s[3:].lstrip("_")
    elif s.startswith("attn"):
        pool, s = list(attn_layers), s[4:].lstrip("_")
    if s in ("", "all"):
        return sorted(pool)
    if s.startswith("top"):
        k = int(round(float(s[len("top") :]) / 100.0 * len(pool)))
        return sorted(pool[len(pool) - k :]) if k else []
    if s.startswith("bottom"):
        k = int(round(float(s[len("bottom") :]) / 100.0 * len(pool)))
        return sorted(pool[:k])
    raise ValueError(f"bad layer pattern: {spec!r}")


# ---------------------------------------------------------------------------------------------
# per-pattern FLOP accounting
# ---------------------------------------------------------------------------------------------


def block_costs(model, attn_layers):
    """Per-block cost coefficients: ``lin[i]`` FLOPs per token (2 * matmul params of the block)
    and ``quad[i]`` FLOPs per token^2 (attention scores + AV, softmax layers only)."""
    lin, quad = [], []
    n_heads = None
    head_dim = None
    for key, block in model.blocks.items():
        li = int(key)
        p = sum(
            m.weight.numel()
            for m in block.modules()
            if isinstance(m, torch.nn.Linear) and m.weight is not None
        )
        lin.append(2.0 * p)
        if li in attn_layers:
            a = block.attention
            n_heads = int(getattr(a, "n_heads", 16))
            head_dim = int(getattr(a, "head_dim", 256))
            quad.append(2.0 * 2.0 * n_heads * head_dim)  # QK^T and AV, causal factor cancels
        else:
            quad.append(0.0)
    return np.asarray(lin), np.asarray(quad)


def costs(lin, quad, length, T, T2):
    """Cost of running each block on ``length[i]`` tokens, as fractions SAVED against two
    references: FULL (every layer at ``T``) and SOFT (every layer at the compacted ``T2``).

    Returns ``(flop_vs_full, flop_vs_soft, mem_vs_full, mem_vs_soft, flop_linear_vs_full)``.
    ``mem`` is the activation / KV footprint proxy -- tokens present, summed over layers -- which
    is the quantity the LAYER axis buys on top of the token axis.
    """
    length = np.asarray(length, dtype=float)
    n = len(lin)

    def cost(L):
        L = np.asarray(L, dtype=float)
        return float((lin * L).sum() + (quad * L * L).sum() / 2.0)

    c = cost(length)
    return (
        1.0 - c / cost(np.full(n, float(T))),
        1.0 - c / cost(np.full(n, float(T2))),
        1.0 - length.sum() / (T * n),
        1.0 - length.sum() / (T2 * n),
        1.0 - (lin * length).sum() / (lin.sum() * T),
    )


def layer_lengths(n_layers, sel, base_len, sel_len, stale_mask=False, stale_len=None):
    """Per-layer token count: ``sel_len`` in the selected layers, ``base_len`` elsewhere. Under
    ``stale_mask`` the tokens dropped by the first selected layer are never read again, so every
    LATER unselected layer can run at ``stale_len`` too."""
    length = np.full(n_layers, float(base_len))
    length[list(sel)] = float(sel_len)
    if stale_mask and len(sel):
        first = int(min(sel))
        for i in range(first + 1, n_layers):
            if i not in set(sel):
                length[i] = float(stale_len)
    return length


# ---------------------------------------------------------------------------------------------
# the hybrid forward
# ---------------------------------------------------------------------------------------------


def embed(model, ids):
    h = model.embeddings(ids)
    if model.embed_scale is not None:
        h = h * model.embed_scale
    if model.embedding_norm is not None:
        h = model.embedding_norm(h)
    return h


class RowPlan:
    """Everything about one (row, keep-set) needed to run any layer pattern."""

    def __init__(self, model, x, keep_holder):
        model._pooled_keep_holder = keep_holder
        cb, soft_inject, _ = model._compact_pooled_soft_tokens(x, None, -100)
        self.cb = cb
        T2 = cb.input_ids.shape[1]
        assert cb.input_ids.shape[0] == 1
        assert int(cb.row_lens[0]) == T2, f"padded compacted row ({int(cb.row_lens[0])} != {T2})"
        assert cb.shadow_rows.numel() == 0, "shadow tokens are not supported here"
        is_slot = torch.zeros(T2, dtype=torch.bool, device=cb.input_ids.device)
        is_slot[cb.soft_cols] = True
        self.slot_cols = cb.soft_cols
        self.nonslot_cols = (~is_slot).nonzero(as_tuple=True)[0]
        self.orig_pos = cb.position_ids[0, self.nonslot_cols].long()
        self.position_ids = cb.position_ids
        self.n_slots = int(self.slot_cols.numel())
        self.T2 = T2
        # slot init from the mean input embedding (identity projector => the training construction)
        inj_rows, inj_cols, mean_feats = soft_inject
        slot_vecs = model.pooled_projector(mean_feats)
        h_c = model.embeddings(cb.input_ids).clone()
        h_c[inj_rows, inj_cols] = slot_vecs.to(h_c.dtype)
        if model.embed_scale is not None:
            h_c = h_c * model.embed_scale
        if model.embedding_norm is not None:
            h_c = model.embedding_norm(h_c)
        self.slot_embed = h_c[:, self.slot_cols]  # (1, n_slots, D)
        self.base_embed = h_c  # (1, T2, D) layer-0 input of the SOFT compacted row
        # doc membership of the pooled docs, for slot-source=layer_input_mean
        self.soft_docs = cb.soft_docs
        # (1, T) True = token that is present in the compacted row; False = a pooled doc's real
        # token, whose hidden state goes stale the moment a soft layer runs (--stale-keys mask).
        self.block_keep = torch.zeros((1, x.shape[1]), dtype=torch.bool, device=x.device)
        self.block_keep[0, self.orig_pos] = True


class UltraPlan:
    """The SECOND compaction level ("soft layers"): on top of the SOFT compacted row (``plan``),
    the KEPT documents are pooled too, so an ultra layer sees only the free tokens (question,
    answer, any header tokens the keep plan left free), one slot per pooled document -- carried
    from the base row -- and one NEW slot per kept document.

    Built by compacting the original row with ``keep_docs`` all false, then joining to the base
    row on ORIGINAL POSITION: a free token and a pooled doc's slot exist in both rows, a kept
    doc's slot exists only here.
    """

    def __init__(self, model, x, plan, keep_docs, ultra_keep=None):
        # ``ultra_keep``: documents that stay REAL even in an ultra layer (e.g. the gold docs,
        # ``--ultra-keep-gold`` -- an ORACLE condition, flag it as such wherever it is quoted).
        uk = torch.zeros_like(keep_docs) if ultra_keep is None else ultra_keep.to(keep_docs.dtype)
        holder = PooledDocKeepHolder(keep_docs=uk)
        model._pooled_keep_holder = holder
        cb_u, soft_inject, _ = model._compact_pooled_soft_tokens(x, None, -100)
        T3 = cb_u.input_ids.shape[1]
        dev = cb_u.input_ids.device
        self.T3 = T3
        self.position_ids = cb_u.position_ids
        kept = keep_docs[0].to(dev).bool()
        is_slot = torch.zeros(T3, dtype=torch.bool, device=dev)
        is_slot[cb_u.soft_cols] = True
        slot_is_kept = kept[cb_u.soft_docs]
        # columns of the ultra row that are CARRIED from the base row (free tokens + pooled slots)
        carried = (~is_slot).clone()
        carried[cb_u.soft_cols[~slot_is_kept]] = True
        self.carried_cols = carried.nonzero(as_tuple=True)[0]
        self.kept_slot_cols = cb_u.soft_cols[slot_is_kept]
        self.kept_slot_docs = cb_u.soft_docs[slot_is_kept]
        # join on original position: base column index for every carried ultra column
        base_pos = plan.position_ids[0].long()
        pos2base = torch.full((int(x.shape[1]) + 1,), -1, dtype=torch.long, device=dev)
        pos2base[base_pos] = torch.arange(base_pos.numel(), device=dev)
        self.parent_idx = pos2base[cb_u.position_ids[0, self.carried_cols].long()]
        assert (
            int((self.parent_idx < 0).sum()) == 0
        ), "ultra row has a column absent from the base row"
        # embed-mean slot for the kept docs (same recipe as RowPlan)
        inj_rows, inj_cols, mean_feats = soft_inject
        h_c = model.embeddings(cb_u.input_ids).clone()
        h_c[inj_rows, inj_cols] = model.pooled_projector(mean_feats).to(h_c.dtype)
        if model.embed_scale is not None:
            h_c = h_c * model.embed_scale
        if model.embedding_norm is not None:
            h_c = model.embedding_norm(h_c)
        self.slot_embed = h_c[:, self.kept_slot_cols]
        self.n_kept_slots = int(self.kept_slot_cols.numel())
        # base-row columns of each kept doc's REAL tokens (for layer_input_mean and stale masking)
        self.block_keep = torch.zeros((1, plan.T2), dtype=torch.bool, device=dev)
        self.block_keep[0, self.parent_idx] = True


def make_kept_doc_scatter(plan, ultra, cid_row, n_docs):
    """Closure h(base row) -> (1, n_kept_slots, D): mean of each KEPT doc's real hidden states."""
    dev = plan.position_ids.device
    cid = cid_row.to(dev).long()
    base_doc = cid[plan.position_ids[0].long()].clone()  # doc id of every base column
    base_doc[plan.slot_cols] = -1  # a base slot column is not a real token
    sel = (base_doc >= 0).nonzero(as_tuple=True)[0]
    idx = base_doc[sel]
    counts = (
        torch.zeros(n_docs, dtype=torch.float32, device=dev)
        .index_add(0, idx, torch.ones_like(idx, dtype=torch.float32))
        .clamp(min=1.0)
    )
    docs = ultra.kept_slot_docs.to(dev)

    def f(h):
        D = h.shape[-1]
        sums = torch.zeros(n_docs, D, dtype=torch.float32, device=dev).index_add(
            0, idx, h[0, sel].float()
        )
        return (sums / counts[:, None])[docs][None].to(h.dtype)

    return f


@torch.no_grad()
def run_two_level(
    model,
    x,
    plan,
    ultra,
    ultra_layers,
    pred_cols,
    slot_source,
    kept_scatter=None,
    stale_keys="keep",
):
    """SOFT compacted row in every layer, further compacted to the ULTRA row in ``ultra_layers``.

    ``pred_cols`` are BASE-row columns. With ``ultra_layers`` empty this is bitwise the ordinary
    soft forward (asserted by ``--selftest``).
    """
    u = set(int(i) for i in ultra_layers)
    h = plan.base_embed.clone()
    s = ultra.slot_embed.clone()
    prev_u, seen_u = False, False
    for key, block in model.blocks.items():
        li = int(key)
        if li not in u:
            kw = {"block_keep": ultra.block_keep} if (stale_keys == "mask" and seen_u) else {}
            h = block(h, position_ids=plan.position_ids, **kw)
            prev_u = False
            continue
        seen_u = True
        if slot_source == "layer_input_mean" and not prev_u:
            s = kept_scatter(h)
        h_c = torch.empty((1, ultra.T3, h.shape[-1]), dtype=h.dtype, device=h.device)
        h_c[0, ultra.carried_cols] = h[0, ultra.parent_idx]
        h_c[:, ultra.kept_slot_cols] = s.to(h.dtype)
        out = block(h_c, position_ids=ultra.position_ids)
        h = h.clone()
        h[0, ultra.parent_idx] = out[0, ultra.carried_cols]
        s = out[:, ultra.kept_slot_cols]
        prev_u = True
    return model.lm_head(h, logits_to_keep=pred_cols[None])[0]


@torch.no_grad()
def run_pattern(
    model,
    x,
    plan,
    soft_layers,
    pred_pos,
    slot_source,
    doc_scatter=None,
    stale_keys="keep",
    drop_all=False,
):
    """Run the stack with ``soft_layers`` computed on the compacted sequence; return logits at
    ``pred_pos`` (original positions).

    ``stale_keys='mask'``: once any soft layer has run, the pooled documents' real tokens carry a
    hidden state frozen at that layer's input, so every LATER dense layer excludes them as keys /
    recurrent writers via ``block_keep`` -- the same kwarg
    :mod:`olmo_core.nn.block_skip` uses for a skipped token. Masking only (no compaction), so it
    changes fidelity, not the FLOP accounting.
    """
    soft = set(int(i) for i in soft_layers)
    h = embed(model, x)
    if not soft:
        # drop_all: the pooled documents are simply ABSENT from every layer -- no slot anywhere.
        # The control for "is the slot doing anything, or is the win just removing the distractors?"
        kw = {"block_keep": plan.block_keep} if drop_all else {}
        for block in model.blocks.values():
            h = block(h, **kw)
        return model.lm_head(h, logits_to_keep=pred_pos[None])[0]
    s = plan.slot_embed.clone()  # (1, n_slots, D) carried slot state
    prev_soft = False
    seen_soft = False
    for key, block in model.blocks.items():
        li = int(key)
        if li not in soft:
            kw = {"block_keep": plan.block_keep} if (stale_keys == "mask" and seen_soft) else {}
            h = block(h, **kw)
            prev_soft = False
            continue
        seen_soft = True
        if slot_source == "layer_input_mean" and not prev_soft:
            s = doc_scatter(h)  # (1, n_slots, D) mean of each pooled doc's real hidden states
        h_c = torch.empty((1, plan.T2, h.shape[-1]), dtype=h.dtype, device=h.device)
        h_c[0, plan.nonslot_cols] = h[0, plan.orig_pos]
        h_c[:, plan.slot_cols] = s.to(h.dtype)
        out = block(h_c, position_ids=plan.position_ids)
        h = h.clone()
        h[0, plan.orig_pos] = out[0, plan.nonslot_cols]
        s = out[:, plan.slot_cols]
        prev_soft = True
    return model.lm_head(h, logits_to_keep=pred_pos[None])[0]


def make_doc_scatter(plan, cid_row, n_docs, device):
    """Closure h -> (1, n_slots, D): mean of each pooled doc's REAL token hidden states in h."""
    cid = cid_row.to(device).long()
    is_ctx = cid >= 0
    idx = cid.clamp(min=0)[is_ctx]
    counts = (
        torch.zeros(n_docs, dtype=torch.float32, device=device)
        .index_add(0, idx, torch.ones_like(idx, dtype=torch.float32))
        .clamp(min=1.0)
    )
    sel = is_ctx.nonzero(as_tuple=True)[0]
    docs = plan.soft_docs.to(device)

    def f(h):
        D = h.shape[-1]
        sums = torch.zeros(n_docs, D, dtype=torch.float32, device=device).index_add(
            0, idx, h[0, sel].float()
        )
        means = sums / counts[:, None]
        return means[docs][None].to(h.dtype)

    return f


# ---------------------------------------------------------------------------------------------


def mean_se(v):
    v = np.asarray(v, dtype=np.float64)
    if v.size == 0:
        return float("nan"), float("nan")
    return float(v.mean()), float(v.std(ddof=1) / np.sqrt(v.size)) if v.size > 1 else 0.0


def summary(res, names, two_level=False):
    ref = "SOFT" if two_level else "FULL"
    print(
        f"{'pattern':18} {'answer CE':>16} {'dCE vs SOFT':>11} {'top1=FULL':>16} {'KL|FULL':>8} "
        f"{'EM':>13} {'flop/FULL':>9} {'flop/SOFT':>9} {'mem/FULL':>8} {'mem/SOFT':>8} {'s/row':>6}",
        flush=True,
    )
    del ref
    for n in names:
        r = res[n]
        if not r["ce"]:
            continue
        ce, ce_se = mean_se(r["ce"])
        t1, t1_se = mean_se(r["top1"])
        em, em_se = mean_se(r["correct"])
        print(
            f"{n:18} {ce:8.3f}+-{ce_se:<6.3f} {np.mean(r['dce_soft']):11.3f} "
            f"{t1:8.3f}+-{t1_se:<6.3f} {np.mean(r['kl']):8.3f} {em:6.3f}+-{em_se:<5.3f} "
            f"{np.mean(r['flop_full']):9.3f} {np.mean(r['flop_soft']):9.3f} "
            f"{np.mean(r['mem_full']):8.3f} {np.mean(r['mem_soft']):8.3f} {np.mean(r['sec']):6.2f}",
            flush=True,
        )


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--task", default="contradiction", choices=["contradiction", "oolong", "nq", "outlier"]
    )
    ap.add_argument("--rung", default="8k")
    ap.add_argument("--rows", type=int, default=200)
    ap.add_argument(
        "--keep", type=float, default=0.0833, help="fraction of NON-gold docs kept real"
    )
    ap.add_argument(
        "--patterns",
        default="none,all,top50,bottom50,alt,alt1,top25,top75,every3,every4,gdn,attn",
        help="comma list of soft-layer patterns; see layer_set()",
    )
    ap.add_argument(
        "--prefix-real",
        default="none",
        help="HEADER-REAL pooling, the construction the 2026-09-08 eval-side PARITY result used "
        "(records/soft-kv-slot-probe-handoff.md): 'stopK' keeps each document's header -- the "
        "tokens after doc_start through the K-th --prefix-stop-id -- REAL, so the slot stands for "
        "the body only. K=1 for contradiction ('Claim N:'), K=3 for oolong "
        "('Date: .. || User: .. || Instance:'). 'none' = the plain mean slot, which is NOT at "
        "parity: the document id is unreadable from a body-only slot.",
    )
    ap.add_argument(
        "--prefix-stop-id", type=int, default=25, help="token id ending a header (':' = 25)"
    )
    ap.add_argument(
        "--slot-source", default="embed_mean", choices=["embed_mean", "layer_input_mean"]
    )
    ap.add_argument(
        "--ultra-keep-gold",
        action="store_true",
        help="two-level mode: the GOLD documents stay real even in ultra layers "
        "(ORACLE -- the gold sidecar is not available at inference)",
    )
    ap.add_argument(
        "--ultra-patterns",
        default=None,
        help="TWO-LEVEL ('soft layers') mode: same pattern vocabulary, but the per-layer default is "
        "the SOFT compacted row and the SELECTED layers run on the ULTRA row (the kept documents "
        "are pooled to one slot each as well). 'none' in this list is the SOFT baseline and 'all' "
        "is every layer ultra. Overrides --patterns.",
    )
    ap.add_argument(
        "--stale-keys",
        default="keep",
        choices=["keep", "mask"],
        help="mask: dense layers AFTER a soft layer drop the pooled docs' (now stale) tokens as keys",
    )
    ap.add_argument("--selftest", action="store_true", help="assert none==FULL and all==SOFT")
    ap.add_argument("--work", default="/data/prasann/layer_probe/work")
    ap.add_argument("--out", default="/data/prasann/layer_probe/out.json")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--jsonl", default=None)
    ap.add_argument("--shard", default=None)
    a = ap.parse_args()

    shard = a.shard or f"{a.work}/{FAMILY}_{a.task}_{a.rung}_r{a.rows}"
    if a.shard is None:
        convert(a.task, a.jsonl or EVAL_JSONL[a.task][a.rung], a.rows, shard)
    rows, masks = load_rows(shard, a.rows)
    log(f"{len(rows)} rows, lengths {[len(r) for r in rows[:6]]}...")

    cfg = build_cfg()
    cfg.lm_head.loss_implementation = LMLossImplementation.default
    model = cfg.build(init_device="cpu")
    ck = find_ckpt(a.ckpt) if a.ckpt else find_ckpt(CKPT[a.task])
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
    model.pooled_projector.reset_parameters()  # identity: soft token == mean input embedding
    model = model.cuda().to(torch.bfloat16)
    model.eval()

    from olmo_core.nn.attention.recurrent import GatedDeltaNet

    n_layers = len(model.blocks)
    gdn_layers = [int(k) for k, b in model.blocks.items() if isinstance(b.attention, GatedDeltaNet)]
    attn_layers = [int(k) for k in model.blocks.keys() if int(k) not in gdn_layers]
    log(
        f"{n_layers} layers: {len(gdn_layers)} GDN {gdn_layers[:6]}..., {len(attn_layers)} attention {attn_layers}"
    )
    lin, quad = block_costs(model, set(attn_layers))

    _PREFIX_REAL["stop_id"] = a.prefix_stop_id
    _PREFIX_REAL["seed"] = a.seed
    _PREFIX_REAL["spec"] = a.prefix_real
    if a.prefix_real != "none":
        install_prefix_real_patch()
        log(
            f"HEADER-REAL pooling on: spec={a.prefix_real} stop_id={a.prefix_stop_id} "
            f"(the slot stands for the document BODY only)"
        )

    two_level = a.ultra_patterns is not None
    spec = a.ultra_patterns if two_level else a.patterns
    names = [p.strip() for p in spec.split(",") if p.strip()]
    if two_level and "full" not in names:
        names = ["full"] + names
    sets = {
        n: (None if n == "full" else layer_set(n, n_layers, gdn_layers, attn_layers)) for n in names
    }
    kind = "ULTRA" if two_level else "soft"
    for n in names:
        log(
            f"pattern {n:14} -> "
            + (
                "FULL attention"
                if sets[n] is None
                else f"{len(sets[n]):2d} {kind} layers {sets[n]}"
            )
        )

    import olmo_core.nn.transformer.model as _tm  # patched binding when --prefix-real is on

    gold_table = (
        json.load(open(f"{shard}/gold_fingerprints.json")) if a.task in GOLD_TASKS else None
    )
    KEYS = (
        "ce",
        "dce_soft",
        "top1",
        "kl",
        "top1_soft",
        "kl_soft",
        "correct",
        "compaction",
        "flop_full",
        "flop_soft",
        "flop_lin_full",
        "mem_full",
        "mem_soft",
        "sec",
    )
    res = {n: {k: [] for k in KEYS} for n in names}
    for n in names:
        res[n]["tok_ce"] = []
        res[n]["tok_ids"] = []
    t_start = time.time()
    for ri, (row, rmask) in enumerate(zip(rows, masks)):
        x = torch.tensor(row[None], device="cuda")
        ans_pos = torch.tensor(np.nonzero(rmask)[0], device="cuda")
        pred_pos = ans_pos - 1
        targets = x[0, ans_pos]
        T = x.shape[1]
        cid_row = _tm.build_chunk_ids_from_tokens(
            x.cpu(),
            doc_start_id=IDS.doc_start,
            doc_end_id=IDS.doc_end,
            eos_id=IDS.eos,
            mode="chunked",
        )[0]
        n_docs = int(cid_row.max()) + 1
        if a.task in GOLD_TASKS:
            keep_fn = make_fingerprint_keep_docs_fn(
                gold_table,
                doc_start_id=IDS.doc_start,
                doc_end_id=IDS.doc_end,
                eos_id=IDS.eos,
                n_random_frac=a.keep,
                mode="gold_plus_random",
                seed=a.seed,
            )
            keep_docs = keep_fn(x.cpu())
        else:
            model._pooled_soft_tokens["keep_prob"] = a.keep
            keep_docs = None
        holder = PooledDocKeepHolder(keep_docs=keep_docs) if keep_docs is not None else None
        plan = RowPlan(model, x, holder)
        if keep_docs is None:  # gold-blind task: recover the realised keep set for the ultra level
            kd = torch.ones((1, n_docs), dtype=torch.bool)
            kd[0, plan.soft_docs.cpu()] = False
            keep_docs = kd
        # FULL reference: every top-1 agreement and KL in the table is against this.
        lf = run_pattern(model, x, plan, [], pred_pos, "embed_mean").float()
        if two_level:
            uk = None
            if a.ultra_keep_gold and a.task in GOLD_TASKS:
                gfn = make_fingerprint_keep_docs_fn(
                    gold_table,
                    doc_start_id=IDS.doc_start,
                    doc_end_id=IDS.doc_end,
                    eos_id=IDS.eos,
                    n_random_frac=0.0,
                    mode="gold_plus_random",
                    seed=a.seed,
                )
                uk = gfn(x.cpu()).bool()
            ultra = UltraPlan(model, x, plan, keep_docs, uk)
            model._pooled_keep_holder = holder  # UltraPlan swapped it out
            kept_scatter = make_kept_doc_scatter(plan, ultra, cid_row, n_docs)
            posmap = {int(pp): c for c, pp in enumerate(plan.position_ids[0].tolist())}
            pred_cols = torch.tensor([posmap[int(pp)] for pp in pred_pos.tolist()], device=x.device)
            ls = run_two_level(
                model, x, plan, ultra, [], pred_cols, a.slot_source, kept_scatter, a.stale_keys
            ).float()
            if ri == 0:
                log(
                    f"lengths: T={T} SOFT T2={plan.T2} ULTRA T3={ultra.T3} "
                    f"(pooled slots {plan.n_slots}, kept-doc slots {ultra.n_kept_slots})"
                )
        else:
            scatter = (
                make_doc_scatter(plan, cid_row, n_docs, x.device)
                if a.slot_source == "layer_input_mean"
                else None
            )
            ls = run_pattern(
                model,
                x,
                plan,
                list(range(n_layers)),
                pred_pos,
                a.slot_source,
                scatter,
                a.stale_keys,
            ).float()
        ce_soft = float(F.cross_entropy(ls, targets))
        for n in names:
            t_cfg = time.time()
            sel = sets[n]
            if sel is None:
                lg, length = lf, np.full(n_layers, float(T))
            elif two_level:
                lg = (
                    ls
                    if not sel
                    else run_two_level(
                        model,
                        x,
                        plan,
                        ultra,
                        sel,
                        pred_cols,
                        a.slot_source,
                        kept_scatter,
                        a.stale_keys,
                    ).float()
                )
                length = layer_lengths(
                    n_layers,
                    sel,
                    plan.T2,
                    ultra.T3,
                    a.stale_keys == "mask",
                    ultra.T3 - ultra.n_kept_slots,
                )
            elif n == "droppooled":
                lg = run_pattern(
                    model, x, plan, [], pred_pos, "embed_mean", None, "keep", drop_all=True
                ).float()
                length = np.full(n_layers, float(plan.T2 - plan.n_slots))
            else:
                lg = (
                    lf
                    if not sel
                    else run_pattern(
                        model, x, plan, sel, pred_pos, a.slot_source, scatter, a.stale_keys
                    ).float()
                )
                length = layer_lengths(
                    n_layers, sel, T, plan.T2, a.stale_keys == "mask", plan.T2 - plan.n_slots
                )
            r = res[n]
            r["ce"].append(float(F.cross_entropy(lg, targets)))
            r["dce_soft"].append(r["ce"][-1] - ce_soft)
            r["top1"].append(float((lg.argmax(-1) == lf.argmax(-1)).float().mean()))
            r["kl"].append(
                float(
                    F.kl_div(
                        F.log_softmax(lg, -1),
                        F.log_softmax(lf, -1),
                        log_target=True,
                        reduction="batchmean",
                    )
                )
            )
            r["top1_soft"].append(float((lg.argmax(-1) == ls.argmax(-1)).float().mean()))
            r["kl_soft"].append(
                float(
                    F.kl_div(
                        F.log_softmax(lg, -1),
                        F.log_softmax(ls, -1),
                        log_target=True,
                        reduction="batchmean",
                    )
                )
            )
            r["correct"].append(float((lg.argmax(-1) == targets).all()))
            r["compaction"].append(float(length.mean() / T))
            ff, fs, mf, ms, fl = costs(lin, quad, length, T, plan.T2)
            r["flop_full"].append(ff)
            r["flop_soft"].append(fs)
            r["mem_full"].append(mf)
            r["mem_soft"].append(ms)
            r["flop_lin_full"].append(fl)
            r["tok_ce"].append(F.cross_entropy(lg, targets, reduction="none").tolist())
            r["tok_ids"].append(targets.tolist())
            r["sec"].append(time.time() - t_cfg)
        if a.selftest and ri == 0:
            if two_level:
                _selftest_two_level(
                    model, x, plan, ultra, pred_cols, pred_pos, kept_scatter, a.slot_source
                )
            else:
                _selftest(model, x, plan, pred_pos, sets, names)
        done = ri + 1
        if done in (1, 2, 5, 10) or done % 25 == 0 or done == len(rows):
            el = time.time() - t_start
            log(
                f"row {done}/{len(rows)}  {el / done:.1f}s/row  ETA {(len(rows) - done) * el / done / 60:.1f} min"
            )
            summary(res, names, two_level)

    summary(res, names, two_level)
    stat_keys = (
        "ce",
        "dce_soft",
        "top1",
        "kl",
        "top1_soft",
        "kl_soft",
        "correct",
        "compaction",
        "flop_full",
        "flop_soft",
        "flop_lin_full",
        "mem_full",
        "mem_soft",
    )
    out = {
        "task": a.task,
        "rung": a.rung,
        "eval_size": len(rows),
        "keep": a.keep,
        "mode": (
            ("two_level_goldreal" if a.ultra_keep_gold else "two_level")
            if two_level
            else "layer_soft"
        ),
        "slot_source": a.slot_source,
        "stale_keys": a.stale_keys,
        "prefix_real": a.prefix_real,
        "ckpt": ck,
        "n_layers": n_layers,
        "gdn_layers": gdn_layers,
        "attn_layers": attn_layers,
        "patterns": {
            n: {
                "layers": sets[n],
                "n_soft_layers": (0 if sets[n] is None else len(sets[n])),
                **{f"{k}_mean": mean_se(res[n][k])[0] for k in stat_keys},
                **{f"{k}_se": mean_se(res[n][k])[1] for k in ("ce", "dce_soft", "top1", "correct")},
                "sec_per_row": float(np.mean(res[n]["sec"])),
            }
            for n in names
        },
        "per_row": {
            n: {
                k: res[n][k]
                for k in ("ce", "top1", "kl", "correct", "dce_soft", "tok_ce", "tok_ids")
            }
            for n in names
        },
    }
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    log(f"wrote {a.out}")


@torch.no_grad()
def _selftest(model, x, plan, pred_pos, sets, names):
    """none == plain eval forward; all == the compacted soft forward the existing probe runs."""

    def cmp(tag, a, b, targets=None):
        d = float((a - b).abs().max())
        kl = float(
            F.kl_div(
                F.log_softmax(b, -1), F.log_softmax(a, -1), log_target=True, reduction="batchmean"
            )
        )
        extra = ""
        if targets is not None:
            extra = f"; CE {float(F.cross_entropy(a, targets)):.4f} vs {float(F.cross_entropy(b, targets)):.4f}"
        log(
            f"selftest {tag}: max|dlogit| = {d:.4f} (max|logit| {float(a.abs().max()):.1f}); "
            f"top1 agree {float((a.argmax(-1) == b.argmax(-1)).float().mean()):.4f}; KL {kl:.5f}{extra}"
        )
        return kl

    targets = x[0, pred_pos + 1]
    ref = model(x, logits_to_keep=pred_pos[None])[0].float()
    mine = run_pattern(model, x, plan, [], pred_pos, "embed_mean").float()
    assert (
        cmp("none-vs-model()", ref, mine, targets) < 1e-4
    ), "dense path diverged from model.forward"
    # SOFT reference through model.forward's own compacted path (needs training mode).
    # NOTE: that path is NOT deterministic in bf16 (the GDN chunk kernel), so the acceptance
    # threshold is the measured call-to-call noise floor, not zero.
    model.train()
    posmap = {int(p): c for c, p in enumerate(plan.position_ids[0].tolist())}
    cols = torch.tensor([posmap[int(p)] for p in pred_pos.tolist()], device=x.device)
    ref_s = model(x, logits_to_keep=cols[None])[0].float()
    ref_s2 = model(x, logits_to_keep=cols[None])[0].float()  # same call twice: noise floor
    model.eval()
    mine_s = run_pattern(
        model, x, plan, list(range(len(model.blocks))), pred_pos, "embed_mean"
    ).float()
    floor = cmp("compacted-forward-vs-itself (noise floor)", ref_s, ref_s2, targets)
    kl = cmp("all(eval)-vs-compacted-forward", ref_s, mine_s, targets)
    tol = max(8.0 * floor, 0.01)
    assert kl < tol, f"all-soft path diverged from the compacted forward (KL {kl} > {tol})"


@torch.no_grad()
def _selftest_two_level(model, x, plan, ultra, pred_cols, pred_pos, kept_scatter, slot_source):
    """ULTRA on no layers == the ordinary SOFT compacted forward (bitwise, same kernels)."""
    model.train()
    ref = model(x, logits_to_keep=pred_cols[None])[0].float()
    model.eval()
    mine = run_two_level(model, x, plan, ultra, [], pred_cols, slot_source, kept_scatter).float()
    d = float((ref - mine).abs().max())
    kl = float(
        F.kl_div(
            F.log_softmax(mine, -1), F.log_softmax(ref, -1), log_target=True, reduction="batchmean"
        )
    )
    log(
        f"selftest two-level ultra=none vs compacted forward: max|dlogit| {d:.4f} KL {kl:.5f} "
        f"(bf16 GDN kernels are not bitwise deterministic; expect KL < 0.01)"
    )
    assert kl < 0.01, f"SOFT baseline of the two-level path diverged (KL {kl})"
    ualL = run_two_level(
        model, x, plan, ultra, list(range(len(model.blocks))), pred_cols, slot_source, kept_scatter
    ).float()
    log(
        f"selftest ultra=all: CE {float(F.cross_entropy(ualL, x[0, pred_pos + 1])):.4f} "
        f"(should resemble keep=0); T2 {plan.T2} T3 {ultra.T3}"
    )


if __name__ == "__main__":
    main()
