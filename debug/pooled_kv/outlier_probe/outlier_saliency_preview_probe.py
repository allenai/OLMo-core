"""
SALIENCY-SELECTED REAL TOKENS and the ONE-LAYER PREVIEW: two routes to FULL-attention parity on
outlier for a FROZEN dense model.  (eval-only, 2026-09-15)

Where this starts.  ``records/outlier-slot-probe.md`` and ``records/outlier-richer-slot-probe.md``
exhaust the *slot vector*: every construction (mean / cmean / cent_cmean / idf / g2 / g4 / enc2 /
enc4 ...) leaves the frozen dense model exactly on the ``k/n`` uniform-guess floor for
``R@gold_pooled``, with the slot-swap control undetectable.  ``records/layer-soft-probe.md`` adds
that, on contradiction with headers real, reading densely for L layers and compressing afterwards
(``fromL`` + ``layer_input_mean``) is at CE parity -- a fact never tested on outlier.

Two ideas, both judged the same way: **paired ΔCE against FULL on the frozen dense checkpoint**, at
2k / 8k / 32k, gold-blind (every document gets the same rule), ``\\n\\nDocument [N]:`` header real in
every construction, remainder of each document pooled into one ``cent_cmean`` slot unless stated.

IDEA 1 -- SALIENCY-SELECTED REAL TOKENS  (``--mode saliency``)
    (a) ORACLE.  From the FULL forward, score every token two ways and keep the top-k of EACH
        document real at its ORIGINAL position:
          * ``grad{k}``  -- input-gradient norm ``||d(answer CE)/d e_t||`` (one backward per row,
            gradient-checkpointed, w.r.t. the embedding output only);
          * ``attn{k}``  -- attention mass received from the answer / last-prompt positions, summed
            over heads and over ALL 8 softmax layers (``attnlast{k}``: the last 4 only).
        This is the upper bound for "if we knew which tokens matter".
    (b) TRANSFERABLE RULE.  A ridge predictor of the gradient saliency from six cheap token
        features (idf, relative position in document, first-sentence, capitalised, digit, token
        length) is fit on ``--fit-rows`` rows DISJOINT from the scored rows and applied as a fixed
        per-document top-k selector (``rule{k}``).
    Baselines computed on the SAME rows: ``first{k}`` (first k body tokens), ``idf{k}`` (highest
    -log p), ``rand{k}``, and ``cc00`` (k = 0, the known floor).  ``grad16_swap`` exchanges the kept
    tokens of the gold documents with random non-gold documents' -- recall must drop.

IDEA 2 -- ONE-LAYER PREVIEW  (``--mode preview``)
    (a) Run layers ``0..L-1`` DENSELY over the whole real context, then compact: each pooled
        document's slot is the mean of its layer-L hidden states (contextualised across ALL
        documents), injected at layer L; kept tokens (question, headers) continue from their own
        layer-L states; layers ``L..31`` run on the compacted row.  This is
        ``debug/pooled_kv/layer_probe/layer_soft_probe.py``'s ``fromL`` + ``layer_input_mean``, on
        outlier.  ``prev0`` (= the plain-mean slot, known at the floor) and ``full`` bracket it.
    (b) GRADIENT-FREE SALIENCY FROM THE PREVIEW.  Layer 3 (the first softmax layer; layers 0-2 are
        GatedDeltaNet and expose no attention matrix) scores doc tokens by the attention mass they
        receive from the last 32 prompt positions; the top-k per document stay REAL for layers
        4..31 (``prev4_k{k}``).  Deployable at training time: no gradients, no second network, and
        the preview is a forward the arm already pays for.

Cost.  ``flop_frac`` prices each construction against FULL with the model's own per-block
coefficients: ``sum_i lin_i * len_i + quad_i * len_i^2 / 2``, ``len_i = T`` for a preview layer and
``T2`` (the compacted length) for a compacted one; 8 of the 32 layers are softmax (quadratic), 24
are GDN (linear).  The ORACLE conditions' selection cost (a dense forward, plus a backward for
``grad*``) is NOT in ``flop_frac`` -- they are upper bounds, not recipes; ``rule*`` and ``prev4_k*``
are the deployable ones and their selection cost IS priced (free / the preview layers).

    python debug/pooled_kv/outlier_probe/outlier_saliency_preview_probe.py --mode saliency \\
        --rung 8k --rows 240 --gen-rows 48 --ckpt-name ds64-outlier-dense-u64M --work /results/w
    python debug/pooled_kv/outlier_probe/outlier_saliency_preview_probe.py --mode preview \\
        --rung 8k --rows 240 --gen-rows 48 --work /results/w
"""

from __future__ import annotations

import argparse
import glob as _glob
import json
import os
import re
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import eval_side_slot_probe as P  # noqa: E402  (shared harness: convert/load_rows/find_ckpt/build_cfg)

from olmo_core.distributed.checkpoint import load_model_and_optim_state  # noqa: E402
from olmo_core.nn.attention import chunked_mask as CM  # noqa: E402
from olmo_core.nn.attention.chunked_mask import (  # noqa: E402
    build_chunk_ids_from_tokens,
    mark_doc_headers_free,
)
from olmo_core.nn.attention.gold_grad_mask import content_fingerprint_from_row  # noqa: E402
from olmo_core.nn.attention.pooled_doc_kv import PooledDocKeepHolder  # noqa: E402
from olmo_core.nn.lm_head import LMLossImplementation  # noqa: E402

W = P.W
IDS = P.IDS
HEADER_STOP_ID = 5491  # Qwen3.5 ']:' ends `\n\nDocument [N]:` (id 25 does not occur in outlier)
HEADER_STOP_COUNT = 1

CKPTS = {
    "ds64-outlier-dense-u64M": f"{W}/ctc_suite/ckpts/ds64-outlier-dense-u64M/model_and_optim",
    "ds64-outlier-dense-u128M": f"{W}/ctc_suite/ckpts/ds64-outlier-dense-u128M/model_and_optim",
    "lmx-full-mixs160M-4b": f"{W}/*/ckpts/lmx-full-mixs160M-4b-2026",
}
CKPT_ROOT = f"{W}/ctc_suite/ckpts"
DS64_SHARDS = f"{W}/ds64/shards"
RUNGS = {r: f"{W}/outlier_lengthmix/eval_rungs/outlier/rung_{n}.jsonl"
         for r, n in (("2k", 2048), ("8k", 8192), ("16k", 16384), ("32k", 32768))}


def ckpt_path_for(name):
    return CKPTS.get(name) or f"{CKPT_ROOT}/{name}"


def log(m):
    print(f"[salprev] {m}", flush=True)


# =============================================================================================
# chunk-id override: how an arbitrary REAL-TOKEN SUBSET is expressed
# =============================================================================================
# ``Transformer._compact_pooled_soft_tokens`` builds chunk ids from the token stream and, when
# ``header_stop_id`` is set, calls ``mark_doc_headers_free`` to re-label the header FREE.  A FREE
# token survives compaction at its ORIGINAL position and is excluded from the pooled slot mean.  So
# "keep an arbitrary subset of a document's tokens real" IS "mark that subset FREE".  model.py
# imports the function INSIDE the forward, so patching the module attribute reaches it.
_OV = {"cid": None}
_ORIG_MARK = mark_doc_headers_free


def _patched_mark(chunk_ids, input_ids, **kw):
    ov = _OV["cid"]
    if ov is None:
        return _ORIG_MARK(chunk_ids, input_ids, **kw)
    S = int(chunk_ids.shape[1])
    o = ov
    if o.shape[1] < S:  # free greedy generation appends FREE tokens past the row
        pad = torch.full((o.shape[0], S - o.shape[1]), -1, dtype=o.dtype, device=o.device)
        o = torch.cat([o, pad], dim=1)
    elif o.shape[1] > S:  # the generation PREFIX is shorter than the scored row
        o = o[:, :S]
    return o.to(device=chunk_ids.device, dtype=chunk_ids.dtype)


CM.mark_doc_headers_free = _patched_mark

# The probe compacts once itself (to map answer positions -> compacted columns) and the forward
# compacts again; if the two disagree the gathered columns go out of bounds and CUDA reports it
# asynchronously inside an unrelated kernel.  Memoize on the exact input tensor + condition.
_CB = {"key": None, "sig": None, "out": None}


def install_compaction_cache(model):
    cls = type(model)
    orig = cls._compact_pooled_soft_tokens

    def patched(self, input_ids, labels, ignore_index):
        sig = (_CB["key"], int(input_ids.data_ptr()), int(input_ids.shape[1]))
        if _CB["out"] is not None and _CB["sig"] == sig:
            return _CB["out"]
        out = orig(self, input_ids, labels, ignore_index)
        _CB["sig"], _CB["out"] = sig, out
        return out

    cls._compact_pooled_soft_tokens = patched
    return orig


# =============================================================================================
# corpus statistics
# =============================================================================================
def _shard_head_ids(shard_dir, n_rows):
    """``token_ids_part_*.npy`` is a RAW HEADERLESS array despite the extension -- np.load dies."""
    parts = sorted(_glob.glob(f"{shard_dir}/token_ids_part_*.npy"))
    if not parts:
        raise SystemExit(f"{shard_dir} has no token_ids_part_*.npy")
    meta = json.load(open(f"{shard_dir}/metadata.json"))
    dtype = np.dtype(meta.get("dtype") or "uint32")
    n_total = os.path.getsize(parts[0]) // dtype.itemsize
    arr = np.memmap(parts[0], dtype=dtype, mode="r", shape=(n_total,))
    row_len = int(meta.get("max_example_len") or 65536)
    n_tok = min(n_total, max(1, n_rows) * max(1, row_len))
    return np.asarray(arr[:n_tok], dtype=np.int64), int(n_tok)


def build_idf(shard_dir, vocab, n_rows=512):
    """``-log p(token)`` over the head of the TRAINING shard (add-one smoothed)."""
    ids, n_tok = _shard_head_ids(shard_dir, n_rows)
    cnt = np.bincount(ids, minlength=vocab)[:vocab].astype(np.float64)
    p = (cnt + 1.0) / (cnt.sum() + float(vocab))
    return (-np.log(p)).astype(np.float32), n_tok


def build_piece_tables(tok, vocab):
    """Per-token-id feature tables: sentence-end, capitalised, has-digit, piece length."""
    pieces = tok.convert_ids_to_tokens(list(range(vocab)))
    sent_end = np.zeros(vocab, dtype=bool)
    is_cap = np.zeros(vocab, dtype=bool)
    is_dig = np.zeros(vocab, dtype=bool)
    plen = np.zeros(vocab, dtype=np.float32)
    for i, s in enumerate(pieces):
        if s is None:
            continue
        t = s.replace("Ġ", " ").replace("Ċ", "\n")
        if "." in t or "\n" in t:
            sent_end[i] = True
        st = t.strip()
        if st[:1].isupper():
            is_cap[i] = True
        if any(c.isdigit() for c in t):
            is_dig[i] = True
        plen[i] = float(len(st))
    return sent_end, is_cap, is_dig, plen


# =============================================================================================
# per-document real-token selection
# =============================================================================================
def topk_positions(body_pos, scores, k):
    """Positions of the k highest-scoring body tokens, returned in ORIGINAL order."""
    n = len(body_pos)
    if n == 0 or k <= 0:
        return []
    if n <= k:
        return [int(p) for p in body_pos]
    order = np.argsort(-np.asarray(scores, dtype=np.float64), kind="stable")[:k]
    return [int(body_pos[i]) for i in sorted(order.tolist())]


def select_positions(kind, k, body_pos, body_ids, ctx, rng=None):
    """``kind`` in {none, first, idf, rand, grad, attn, attnlast, rule, prevattn}.

    ``ctx`` carries the per-row score arrays (``sal_grad`` / ``sal_attn`` / ... indexed by ORIGINAL
    row position) and the corpus tables.
    """
    n = len(body_pos)
    if kind == "none" or n == 0 or k <= 0:
        return []
    if kind == "first":
        return [int(p) for p in body_pos[:k]]
    if kind == "idf":
        return topk_positions(body_pos, ctx["idf"][body_ids], k)
    if kind == "rand":
        if n <= k:
            return [int(p) for p in body_pos]
        idx = np.sort(rng.choice(n, size=k, replace=False))
        return [int(body_pos[i]) for i in idx]
    key = {"grad": "sal_grad", "attn": "sal_attn", "attnlast": "sal_attn_last",
           "rule": "sal_rule", "prevattn": "sal_prev"}[kind]
    sal = ctx[key]
    if sal is None:
        return []
    return topk_positions(body_pos, sal[body_pos], k)


def swap_kept(x_cpu, sel_by_doc, gold_row, n_docs, seed):
    """CONTROL: exchange the KEPT REAL tokens of each gold document with those of a random non-gold
    document.  Positions, ids, headers and slot means are untouched (kept tokens are FREE, so they
    never enter a slot mean), so only "what the model can read" changes."""
    gi = [d for d in range(n_docs) if bool(gold_row[d]) and len(sel_by_doc[d]) > 0]
    ni = [d for d in range(n_docs) if not bool(gold_row[d]) and len(sel_by_doc[d]) > 0]
    xs = x_cpu.clone()
    if not gi or len(ni) < len(gi):
        return xs
    g = torch.Generator().manual_seed(seed + n_docs)
    pick = [ni[j] for j in torch.randperm(len(ni), generator=g)[: len(gi)].tolist()]
    for da, db in zip(gi, pick):
        pa, pb = sel_by_doc[da], sel_by_doc[db]
        m = min(len(pa), len(pb))
        if m == 0:
            continue
        ia = torch.tensor(pa[:m], dtype=torch.long)
        ib = torch.tensor(pb[:m], dtype=torch.long)
        va, vb = x_cpu[0, ia].clone(), x_cpu[0, ib].clone()
        xs[0, ia] = vb
        xs[0, ib] = va
    return xs


# =============================================================================================
# saliency: input gradients, and attention mass received
# =============================================================================================
def embed_raw(model, ids):
    """The embedding lookup ONLY -- the tensor the input gradient is taken with respect to."""
    return model.embeddings(ids)


def embed_post(model, e):
    """The rest of the embedding path (scale + embedding norm), applied to a raw embedding."""
    h = e
    if model.embed_scale is not None:
        h = h * model.embed_scale
    if model.embedding_norm is not None:
        h = model.embedding_norm(h)
    return h


def grad_saliency(model, x, pred_pos, targets, ckpt_blocks=True):
    """``||d(mean answer CE)/d e_t||_2`` per token, from ONE gradient-checkpointed backward.

    Only the embedding output requires grad (every parameter is frozen), so autograd keeps the
    minimum needed for an input gradient; with ``ckpt_blocks`` each block is recomputed in the
    backward instead of storing its internals, which is what makes 32k rows fit.

    :returns: ``(T,)`` float32 CPU tensor.
    """
    from torch.utils.checkpoint import checkpoint

    was = model.training
    model.eval()
    model._pooled_keep_holder = None
    with torch.enable_grad():
        e = embed_raw(model, x).detach().requires_grad_(True)
        h = embed_post(model, e)
        for block in model.blocks.values():
            h = checkpoint(block, h, use_reentrant=False) if ckpt_blocks else block(h)
        lg = model.lm_head(h, logits_to_keep=pred_pos[None])[0].float()
        loss = F.cross_entropy(lg, targets)
        (g,) = torch.autograd.grad(loss, e)
    if was:
        model.train()
    return g[0].float().norm(dim=-1).detach().cpu()


class AttnCapture:
    """Attention mass RECEIVED by every key position from a chosen set of query positions.

    The softmax backend runs ``F.scaled_dot_product_attention``, which never materialises the
    probabilities, so this re-computes them for the (few) selected query rows only:
    ``softmax(q[qsel] @ k^T * scale)`` with the causal mask, summed over heads.  Memory is
    ``n_heads * |qsel| * T`` floats per layer -- 84 MB at 16 heads / 40 queries / 32k.

    Installed as a forward PRE-hook on each attention block's ``backend`` module, so it sees q and k
    AFTER RoPE and QK-norm, exactly as attention does.
    """

    def __init__(self, model):
        self.handles = []
        self.layers = []  # layer index of each softmax block, in order
        self.state = {"on": False, "qsel": None, "acc": {}, "only": None}
        for key, block in model.blocks.items():
            attn = getattr(block, "attention", None)
            if attn is None or not hasattr(attn, "backend"):
                continue
            li = int(key)
            self.layers.append(li)
            self.handles.append(
                attn.backend.register_forward_pre_hook(self._make(li, attn), with_kwargs=True)
            )

    def _make(self, li, attn):
        def hook(mod, args, kwargs):
            st = self.state
            if not st["on"] or (st["only"] is not None and li not in st["only"]):
                return None
            # the backend is called as ``self.backend((q, k, v), ...)`` -- one positional tuple
            qkv = kwargs.get("qkv", args[0] if args else None)
            if not isinstance(qkv, (tuple, list)) or len(qkv) != 3:
                return None
            q, k = qkv[0], qkv[1]
            if q is None or k is None or q.dim() != 4 or q.shape[0] != 1:
                return None
            qs = st["qsel"]
            T = k.shape[1]
            if qs is None or int(qs.max()) >= q.shape[1]:
                return None
            n_rep = max(1, q.shape[2] // k.shape[2])
            kk = k if n_rep == 1 else k.repeat_interleave(n_rep, dim=2)
            scale = getattr(mod, "scale", None) or float(q.shape[-1]) ** -0.5
            # bf16 matmul (fp32 accumulate on tensor cores) then cast: an fp32 copy of k is
            # ~0.5 GB at 32k and buys nothing for a RANKING signal
            sc = torch.einsum("qhd,thd->hqt", q[0, qs], kk[0]).float() * float(scale)
            causal = torch.arange(T, device=sc.device)[None, :] > qs[:, None]
            sc = sc.masked_fill(causal[None], float("-inf"))
            p = sc.softmax(-1).sum(dim=(0, 1))  # (T,) summed over heads and query rows
            st["acc"][li] = st["acc"].get(li, 0.0) + p.detach().float()
            return None

        return hook

    def run(self, model, x, qsel, only=None):
        """One dense forward with capture on; returns ``{layer: (T,) tensor}``."""
        self.state.update(on=True, qsel=qsel, acc={}, only=only)
        was = model.training
        model.eval()
        model._pooled_keep_holder = None
        with torch.no_grad():
            model(x, logits_to_keep=1)
        self.state["on"] = False
        if was:
            model.train()
        return {li: v.cpu() for li, v in self.state["acc"].items()}

    def arm(self, qsel, only):
        self.state.update(on=True, qsel=qsel, acc={}, only=only)

    def disarm(self):
        self.state["on"] = False
        return {li: v.cpu() for li, v in self.state["acc"].items()}

    def remove(self):
        for h in self.handles:
            h.remove()


# =============================================================================================
# the one-layer preview (layer_soft_probe's `fromL` + `layer_input_mean`, self-contained)
# =============================================================================================
class RowPlan:
    """Compaction of one row: which compacted column is which original position, where the slots
    are, and each slot's document."""

    def __init__(self, model, x):
        cb = model._compact_pooled_soft_tokens(x, None, -100)[0]
        T2 = cb.input_ids.shape[1]
        assert cb.input_ids.shape[0] == 1
        assert int(cb.row_lens[0]) == T2, f"padded compacted row ({int(cb.row_lens[0])} != {T2})"
        assert cb.shadow_rows.numel() == 0, "shadow tokens are not supported here"
        is_slot = torch.zeros(T2, dtype=torch.bool, device=cb.input_ids.device)
        is_slot[cb.soft_cols] = True
        self.cb = cb
        self.slot_cols = cb.soft_cols
        self.soft_docs = cb.soft_docs
        self.nonslot_cols = (~is_slot).nonzero(as_tuple=True)[0]
        self.orig_pos = cb.position_ids[0, self.nonslot_cols].long()
        self.position_ids = cb.position_ids
        self.T2 = T2


def make_doc_scatter(plan, cid_row, n_docs, device):
    """Closure ``h -> (1, n_slots, D)``: mean of each pooled doc's REAL token hidden states in h.

    ``cid_row`` is the (already header-freed and override-applied) chunk-id row, so FREE tokens --
    headers and any kept real tokens -- are excluded from the mean automatically.
    """
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
        return (sums / counts[:, None])[docs][None].to(h.dtype)

    return f


@torch.no_grad()
def preview_forward(model, x, cid_row, n_docs, L, want_pos, capture=None, qsel=None):
    """Layers ``0..L-1`` dense over the full row, then compact and run ``L..31``.

    :param cid_row: ``(T,)`` chunk ids AFTER header-freeing and any real-token override.
    :param want_pos: original positions whose logits to return.
    :param capture: an :class:`AttnCapture` to arm over the DENSE preview layers only.
    :returns: ``(logits, plan, h_at_cut)`` -- ``h_at_cut`` is the layer-L input of the full row, so
        a caller can reuse one preview for several keep-sets.
    """
    plan = RowPlan(model, x)
    scatter = make_doc_scatter(plan, cid_row, n_docs, x.device)
    h = embed_post(model, embed_raw(model, x))
    blocks = list(model.blocks.items())
    if capture is not None:
        capture.arm(qsel, only=None)
    for key, block in blocks[:L]:
        h = block(h)
    if capture is not None:
        capture.disarm()
    h_cut = h
    s = scatter(h)
    for key, block in blocks[L:]:
        h_c = torch.empty((1, plan.T2, h.shape[-1]), dtype=h.dtype, device=h.device)
        h_c[0, plan.nonslot_cols] = h[0, plan.orig_pos]
        h_c[:, plan.slot_cols] = s.to(h.dtype)
        out = block(h_c, position_ids=plan.position_ids)
        h = h.clone()
        h[0, plan.orig_pos] = out[0, plan.nonslot_cols]
        s = out[:, plan.slot_cols]
    return model.lm_head(h, logits_to_keep=want_pos[None])[0], plan, h_cut


@torch.no_grad()
def preview_from_cut(model, x, h_cut, cid_row, n_docs, L, want_pos):
    """Continue a preview from a cached layer-L input ``h_cut`` with a DIFFERENT keep-set.

    The preview layers are identical across keep-sets (they are dense over the full real row), so
    the k-sweep of idea 2b costs one preview per row, not one per k.
    """
    plan = RowPlan(model, x)
    scatter = make_doc_scatter(plan, cid_row, n_docs, x.device)
    h = h_cut
    s = scatter(h)
    for key, block in list(model.blocks.items())[L:]:
        h_c = torch.empty((1, plan.T2, h.shape[-1]), dtype=h.dtype, device=h.device)
        h_c[0, plan.nonslot_cols] = h[0, plan.orig_pos]
        h_c[:, plan.slot_cols] = s.to(h.dtype)
        out = block(h_c, position_ids=plan.position_ids)
        h = h.clone()
        h[0, plan.orig_pos] = out[0, plan.nonslot_cols]
        s = out[:, plan.slot_cols]
    return model.lm_head(h, logits_to_keep=want_pos[None])[0], plan


# =============================================================================================
# FLOP accounting
# =============================================================================================
def block_costs(model):
    """``lin[i]`` FLOPs per token and ``quad[i]`` FLOPs per token^2 for every block."""
    lin, quad, attn_layers = [], [], []
    for key, block in model.blocks.items():
        li = int(key)
        p = sum(m.weight.numel() for m in block.modules()
                if isinstance(m, torch.nn.Linear) and m.weight is not None)
        lin.append(2.0 * p)
        a = getattr(block, "attention", None)
        if a is not None and hasattr(a, "backend"):
            attn_layers.append(li)
            quad.append(2.0 * 2.0 * int(getattr(a, "n_heads", 16)) * int(getattr(a, "head_dim", 256)))
        else:
            quad.append(0.0)
    return np.asarray(lin), np.asarray(quad), attn_layers


def flop_fraction(lin, quad, T, T2, L=0):
    """Cost of ``L`` dense preview layers + ``n-L`` compacted layers, as a fraction of FULL."""
    n = len(lin)
    length = np.full(n, float(T2))
    length[:L] = float(T)

    def c(v):
        v = np.asarray(v, dtype=float)
        return float((lin * v).sum() + (quad * v * v).sum() / 2.0)

    return c(length) / c(np.full(n, float(T)))


# =============================================================================================
# metrics helpers
# =============================================================================================
_ID_RE = re.compile(r"\[\s*(\d+)\s*\]")


def parse_ids(text):
    out = []
    for m in _ID_RE.finditer(text):
        v = int(m.group(1))
        if v not in out:
            out.append(v)
    return out


def set_f1(pred, true):
    if not pred or not true:
        return 0.0
    inter = len(set(pred) & set(true))
    if inter == 0:
        return 0.0
    p, r = inter / len(set(pred)), inter / len(set(true))
    return 2 * p * r / (p + r)


@torch.no_grad()
def generate(fwd, seq, max_new, k_ids, tok):
    """Free greedy continuation. ``fwd(seq) -> last-position logits``."""
    gen = []
    for _ in range(max_new):
        nxt = int(fwd(seq).argmax())
        if nxt == IDS.eos:
            break
        gen.append(nxt)
        seq = torch.cat([seq, torch.tensor([[nxt]], device=seq.device)], dim=1)
        txt = tok.decode(gen)
        if len(parse_ids(txt)) >= k_ids and txt.rstrip().endswith("]"):
            break
    return gen


def mean_se(v):
    if not v:
        return float("nan"), float("nan")
    a = np.asarray(v, dtype=np.float64)
    return float(a.mean()), (float(a.std(ddof=1) / len(a) ** 0.5) if len(a) > 1 else 0.0)


def paired(a, b):
    """mean and SE of the PAIRED difference a - b over the rows both have."""
    n = min(len(a), len(b))
    if n < 2:
        return float("nan"), float("nan")
    d = np.asarray(a[:n], dtype=np.float64) - np.asarray(b[:n], dtype=np.float64)
    return float(d.mean()), float(d.std(ddof=1) / n ** 0.5)


# =============================================================================================
# the transferable rule (idea 1b)
# =============================================================================================
FEATURES = ("idf", "relpos", "first_sent", "is_cap", "is_dig", "tok_len", "log_doclen")


def token_features(body_pos, body_ids, ctx):
    """``(n_body, 7)`` cheap features -- nothing that needs a model forward."""
    n = len(body_pos)
    if n == 0:
        return np.zeros((0, len(FEATURES)), dtype=np.float32)
    sent_end = ctx["sent_end"][body_ids]
    first_sent = np.zeros(n, dtype=np.float32)
    end = int(np.argmax(sent_end)) if sent_end.any() else n - 1
    first_sent[: end + 1] = 1.0
    return np.stack([
        ctx["idf"][body_ids],
        np.arange(n, dtype=np.float32) / max(1, n - 1),
        first_sent,
        ctx["is_cap"][body_ids].astype(np.float32),
        ctx["is_dig"][body_ids].astype(np.float32),
        ctx["plen"][body_ids],
        np.full(n, np.log(max(2, n)), dtype=np.float32),
    ], axis=1).astype(np.float32)


def fit_ridge(X, y, lam=1.0):
    """Closed-form ridge on standardised features; returns ``(mu, sd, w, b)``."""
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Z = (X - mu) / sd
    Z = np.concatenate([Z, np.ones((len(Z), 1), dtype=np.float64)], axis=1)
    A = Z.T @ Z + lam * np.eye(Z.shape[1])
    A[-1, -1] -= lam  # do not penalise the intercept
    w = np.linalg.solve(A, Z.T @ y)
    return mu, sd, w[:-1], float(w[-1])


def apply_ridge(X, fit):
    mu, sd, w, b = fit
    return ((X - mu) / sd) @ w + b


# =============================================================================================
# conditions
# =============================================================================================
def C(name, kind, k=0, mode="pool", L=0, swap=False, slot="cent_cmean"):
    """``mode``: full | pool (compacted, cent_cmean slot) | preview (dense prefix then compact)."""
    return dict(name=name, kind=kind, k=k, mode=mode, L=L, swap=swap, slot=slot)


def build_conditions(mode, ks):
    if mode == "saliency":
        cs = [C("full", "none", mode="full"), C("cc00", "none", 0)]
        for k in ks:
            cs.append(C(f"first{k}", "first", k))
        for k in ks:
            cs.append(C(f"idf{k}", "idf", k))
        for k in ks:
            cs.append(C(f"grad{k}", "grad", k))
        for k in ks:
            cs.append(C(f"attn{k}", "attn", k))
        cs.append(C(f"attnlast{ks[1]}", "attnlast", ks[1]))
        for k in ks:
            cs.append(C(f"rule{k}", "rule", k))
        cs.append(C(f"rand{ks[1]}", "rand", ks[1]))
        cs.append(C(f"grad{ks[-1]}_swap", "grad", ks[-1], swap=True))
        return cs
    if mode == "preview":
        cs = [C("full", "none", mode="full"), C("cc00", "none", 0)]
        for L in (0, 1, 2, 4):
            cs.append(C(f"prev{L}", "none", 0, mode="preview", L=L))
        for k in ks:
            cs.append(C(f"prev4_k{k}", "prevattn", k, mode="preview", L=4))
        cs.append(C(f"prev4_first{ks[1]}", "first", ks[1], mode="preview", L=4))
        cs.append(C(f"prev4_k{ks[1]}_swap", "prevattn", ks[1], mode="preview", L=4, swap=True))
        return cs
    raise SystemExit(f"unknown --mode {mode}")


# =============================================================================================
@torch.no_grad()
def main():
    global HEADER_STOP_ID
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="saliency", choices=["saliency", "preview"])
    ap.add_argument("--rung", default="8k")
    ap.add_argument("--rungs", default=None, help="comma list, scored in ONE process")
    ap.add_argument("--rows", type=int, default=240)
    ap.add_argument("--fit-rows", type=int, default=40,
                    help="rows AFTER the scored rows used to fit the transferable rule (disjoint)")
    ap.add_argument("--gen-rows", type=int, default=48)
    ap.add_argument("--gen-max-new", type=int, default=48)
    ap.add_argument("--ks", default="4,8,16")
    ap.add_argument("--conditions", default="all", help="comma list, or 'all'")
    ap.add_argument("--ckpt-name", default="ds64-outlier-dense-u64M")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--jsonl", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--work", default="/results/salprev_work")
    ap.add_argument("--out", default="/results/outlier_saliency_preview.json")
    ap.add_argument("--weka-out", default=f"{W}/_eval_results/outlier_slot_probe")
    ap.add_argument("--tag", default="")
    ap.add_argument("--idf-shard", default=None)
    ap.add_argument("--idf-rows", type=int, default=512)
    ap.add_argument("--header-stop-id", type=int, default=HEADER_STOP_ID)
    ap.add_argument("--n-query", type=int, default=32,
                    help="prompt positions used as attention queries for the attention saliency")
    ap.add_argument("--dump-rows", type=int, default=6)
    ap.add_argument("--no-ckpt-blocks", action="store_true",
                    help="do NOT gradient-checkpoint the backward (faster, much more memory)")
    a = ap.parse_args()
    HEADER_STOP_ID = int(a.header_stop_id)
    a.rungs = a.rungs or a.rung
    ks = [int(v) for v in a.ks.split(",")]
    if a.tokenizer:
        P.TOKENIZER = a.tokenizer

    all_conds = build_conditions(a.mode, ks)
    known = {c["name"]: c for c in all_conds}
    if a.conditions == "all":
        conds = all_conds
    else:
        want = [w for w in a.conditions.split(",") if w]
        if "full" not in want:
            want = ["full"] + want
        miss = [w for w in want if w not in known]
        if miss:
            raise SystemExit(f"unknown conditions {miss}; known: {sorted(known)}")
        conds = [known[w] for w in want]
    log(f"mode={a.mode} conditions ({len(conds)}): {[c['name'] for c in conds]}")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(P.TOKENIZER)
    vocab = int(P.VOCAB)

    idf_shard = a.idf_shard
    if idf_shard is None:
        bud = a.ckpt_name.rsplit("-u", 1)[-1] if "-u" in a.ckpt_name else "64M"
        idf_shard = f"{DS64_SHARDS}/outlier_u{bud}"
    t0 = time.time()
    idf, n_idf_tok = build_idf(idf_shard, vocab, n_rows=a.idf_rows)
    sent_end, is_cap, is_dig, plen = build_piece_tables(tok, vocab)
    log(f"corpus tables from {n_idf_tok} tokens of {idf_shard} in {time.time() - t0:.0f}s; "
        f"{int(sent_end.sum())} sentence-end pieces, {int(is_dig.sum())} digit pieces")

    cfg = P.build_cfg()
    cfg.lm_head.loss_implementation = LMLossImplementation.default
    model = cfg.build(init_device="cpu")
    ck = P.find_ckpt(a.ckpt or ckpt_path_for(a.ckpt_name))
    t0 = time.time()
    load_model_and_optim_state(ck, model)
    log(f"loaded {ck} in {time.time() - t0:.0f}s")
    a.resolved_ckpt = ck
    model.enable_pooled_soft_tokens(
        IDS.doc_start, IDS.doc_end, IDS.eos, placeholder_id=IDS.landmark,
        keep_prob=0.0, keep_seed=a.seed, detach_soft_kv=True,
    )
    model.pooled_projector.reset_parameters()  # identity: slot == (content) mean input embedding
    model = model.cuda().to(torch.bfloat16)
    model.requires_grad_(False)  # only the embedding OUTPUT carries a gradient
    pst = model._pooled_soft_tokens
    install_compaction_cache(model)
    lin, quad, attn_layers = block_costs(model)
    n_layers = len(lin)
    log(f"{n_layers} blocks, {len(attn_layers)} softmax layers at {attn_layers}")

    # the --st-slot-mode stop set, from the TRAINING shard, exactly as train_ctc_suite builds it
    from olmo_core.nn.pooled_soft_token import build_slot_stop_ids
    head_ids, _ = _shard_head_ids(idf_shard, a.idf_rows)
    stop, shown = build_slot_stop_ids(
        head_ids, top_k=100,
        extra_ids=(IDS.doc_start, IDS.doc_end, IDS.eos, IDS.landmark, IDS.pad),
        decode=lambda t: tok.decode([t]))
    pst["slot_stop_ids"] = [int(t) for t in stop]
    pst["slot_stop_mask"] = None
    log(f"slot stop set: {len(stop)} ids; most frequent dropped: {' '.join(shown[:12])}")

    capture = AttnCapture(model)
    ctx0 = {"idf": idf, "sent_end": sent_end, "is_cap": is_cap, "is_dig": is_dig, "plen": plen}

    summaries = {}
    for rung in [r for r in a.rungs.split(",") if r]:
        summaries[rung] = run_rung(a, rung, conds, model, pst, tok, ctx0, capture,
                                   lin, quad, attn_layers, n_layers)
    if len(summaries) > 1:
        for rung, s in summaries.items():
            print(f"\n### rung {rung}", flush=True)
            verdict(s)


DUMP_CONDS = ("cc00", "grad8", "grad16", "rule8", "prev4", "prev4_k8")


def row_layout(x_cpu):
    """Chunk ids with the header FREE, plus each document's body positions / ids."""
    cid0 = build_chunk_ids_from_tokens(
        x_cpu, doc_start_id=IDS.doc_start, doc_end_id=IDS.doc_end, eos_id=IDS.eos, mode="chunked")
    n_docs = int(cid0.max()) + 1
    cid_h = _ORIG_MARK(cid0, x_cpu, doc_start_id=IDS.doc_start, doc_end_id=IDS.doc_end,
                       stop_id=HEADER_STOP_ID, stop_count=HEADER_STOP_COUNT, cap=32)
    c_h = cid_h[0].numpy()
    xr = x_cpu[0].numpy()
    is_marker = (xr == IDS.doc_start) | (xr == IDS.doc_end)
    body_pos, body_ids = [], []
    for d in range(n_docs):
        sel = np.nonzero((c_h == d) & ~is_marker)[0]
        body_pos.append(sel)
        body_ids.append(xr[sel])
    n_free_nondoc = int((cid0[0].numpy() == -1).sum())
    return cid0, cid_h, n_docs, body_pos, body_ids, n_free_nondoc


def run_rung(a, rung, conds, model, pst, tok, ctx0, capture, lin, quad, attn_layers, n_layers):
    shard = f"{a.work}/outlier_{rung}"
    P.convert("outlier", a.jsonl or RUNGS[rung], a.rows + a.fit_rows, shard)
    rows, masks = P.load_rows(shard, a.rows + a.fit_rows)
    score_rows = rows[: a.rows]
    fit_rows = rows[a.rows: a.rows + a.fit_rows]
    log(f"=== rung {rung}: {len(score_rows)} scored rows + {len(fit_rows)} rule-fit rows "
        f"(disjoint); lengths {[len(r) for r in score_rows[:5]]}")
    if len(score_rows) < 500:
        log(f"WARNING eval_size={len(score_rows)} (<500): binomial SE at f1~0.5 is "
            f"{0.5 / max(1, len(score_rows)) ** 0.5:.3f}")

    # RoPE: rope.forward sizes its absolute-position sin/cos buffer from the CACHE, not from the
    # position_ids it is handed, so a short compacted row carrying ORIGINAL positions reads past the
    # end (reported asynchronously, far from the cause).  Warm every module first.
    max_pos = max(len(r) for r in rows) + a.gen_max_new + 8
    n_warm = 0
    for mod in model.modules():
        rope = getattr(mod, "rope", None)
        if rope is not None and hasattr(rope, "warmup_cache"):
            rope.warmup_cache(max_pos, torch.device("cuda"))
            n_warm += 1
    log(f"warmed {n_warm} RoPE caches to {max_pos} positions")

    gold_table = json.load(open(f"{shard}/gold_fingerprints.json"))
    gold_table = {fp: sorted({int(i) for v in val for i in (v if isinstance(v, (list, tuple)) else [v])})
                  for fp, val in gold_table.items()}
    log(f"gold sidecar: {len(gold_table)} fingerprints")

    need_grad = any(c["kind"] in ("grad", "rule") for c in conds)
    need_attn = any(c["kind"] in ("attn", "attnlast") for c in conds)
    last4 = attn_layers[-4:]

    # ---------------------------------------------------------------------------------------
    # (1b) fit the transferable rule on rows DISJOINT from the scored ones
    # ---------------------------------------------------------------------------------------
    rule_fit, rule_info = None, {}
    if any(c["kind"] == "rule" for c in conds) and fit_rows:
        Xs, ys = [], []
        t0 = time.time()
        for fi, row in enumerate(fit_rows):
            x = torch.tensor(row[None], device="cuda")
            m = masks[a.rows + fi]
            ans_pos = torch.tensor(np.nonzero(m)[0], device="cuda")
            sal = grad_saliency(model, x, ans_pos - 1, x[0, ans_pos],
                                ckpt_blocks=not a.no_ckpt_blocks).numpy()
            _, _, n_docs, bpos, bids, _ = row_layout(x.cpu())
            for d in range(n_docs):
                if len(bpos[d]) < 4:
                    continue
                X = token_features(bpos[d], bids[d], ctx0)
                s = sal[bpos[d]].astype(np.float64)
                s = (s - s.mean()) / (s.std() + 1e-9)   # rank the tokens WITHIN the document
                Xs.append(X)
                ys.append(s)
            if fi + 1 in (1, 2, 5) or (fi + 1) % 10 == 0:
                log(f"rule fit row {fi + 1}/{len(fit_rows)} ({time.time() - t0:.0f}s)")
        X = np.concatenate(Xs).astype(np.float64)
        y = np.concatenate(ys)
        rule_fit = fit_ridge(X, y, lam=10.0)
        pred = apply_ridge(X, rule_fit)
        rule_info = {
            "n_tokens": int(len(y)), "n_rows": len(fit_rows),
            "pearson_r_in_sample": float(np.corrcoef(pred, y)[0, 1]),
            "weights": {f: float(w) for f, w in zip(FEATURES, rule_fit[2])},
        }
        log(f"rule fit on {len(y)} tokens of {len(fit_rows)} rows: in-sample r="
            f"{rule_info['pearson_r_in_sample']:.3f}  weights={rule_info['weights']}")

    KEYS = ("ce", "ce_digit", "gen_f1", "gen_em", "compaction", "flop_frac", "tok_per_doc",
            "body_per_doc", "n_docs", "sec")
    acc = {c["name"]: {k: [] for k in KEYS} for c in conds}
    for c in conds:
        acc[c["name"]].update(hit_gold_pooled=[], hit_gold_real=[])
    sel_stats = {c["name"]: {"idf_pct": [], "relpos": [], "digit": [], "cap": [], "first_sent": [],
                             "overlap_grad": []} for c in conds}
    sel_samples = {}
    n_miss, dumps = 0, []
    rng = np.random.default_rng(a.seed)
    t_start = time.time()

    for ri, row in enumerate(score_rows):
        rmask = masks[ri]
        x = torch.tensor(row[None], device="cuda")
        x_cpu = x.cpu()
        T = x.shape[1]
        ans_pos = torch.tensor(np.nonzero(rmask)[0], device="cuda")
        pred_pos = ans_pos - 1
        targets = x[0, ans_pos]
        true_text = tok.decode(targets.tolist())
        true_ids = parse_ids(true_text)
        pieces = [tok.decode([int(t)]) for t in targets.tolist()]
        digit_sel = torch.tensor([i for i, q in enumerate(pieces) if any(ch.isdigit() for ch in q)],
                                 device="cuda", dtype=torch.long)
        ans_start = int(ans_pos[0])

        fp = content_fingerprint_from_row(row.tolist(), IDS.eos)
        gold_docs = gold_table.get(fp)
        if gold_docs is None:
            n_miss += 1
            log(f"row {ri}: gold fingerprint MISS -- skipped")
            continue

        cid0, cid_h, n_docs, bpos, bids, n_free_nondoc = row_layout(x_cpu)
        gold_row = torch.zeros(n_docs, dtype=torch.bool)
        gold_row[[d for d in gold_docs if 0 <= d < n_docs]] = True
        off = 1
        if true_ids and gold_docs and len(true_ids) == len(gold_docs):
            diffs = {t - g for t, g in zip(sorted(true_ids), sorted(gold_docs))}
            if len(diffs) == 1:
                off = diffs.pop()

        # -------- per-row saliency maps (computed once, shared by every k) -------------------
        ctx = dict(ctx0)
        ctx["sal_grad"] = ctx["sal_attn"] = ctx["sal_attn_last"] = None
        ctx["sal_rule"] = ctx["sal_prev"] = None
        if need_grad:
            ctx["sal_grad"] = grad_saliency(model, x, pred_pos, targets,
                                            ckpt_blocks=not a.no_ckpt_blocks).numpy()
        if need_attn:
            # ORACLE queries: the answer positions plus the last 16 prompt positions
            q_oracle = torch.cat([
                torch.arange(max(0, ans_start - 16), ans_start, device="cuda"), pred_pos]).unique()
            per_layer = capture.run(model, x, q_oracle)
            tot = sum(per_layer.values())
            ctx["sal_attn"] = tot.numpy()
            ctx["sal_attn_last"] = sum(per_layer[li] for li in last4 if li in per_layer).numpy()
        if rule_fit is not None:
            sal = np.zeros(T, dtype=np.float32)
            for d in range(n_docs):
                if len(bpos[d]) == 0:
                    continue
                sal[bpos[d]] = apply_ridge(token_features(bpos[d], bids[d], ctx0).astype(np.float64),
                                           rule_fit).astype(np.float32)
            ctx["sal_rule"] = sal

        do_gen = ri < a.gen_rows
        prev_cache = {}  # L -> (h_cut,) for the preview conditions, one per row

        for cond in conds:
            name = cond["name"]
            t_cfg = time.time()
            _CB["key"], _CB["sig"], _CB["out"] = (ri, name), None, None
            pst["header_stop_id"] = HEADER_STOP_ID
            pst["header_stop_count"] = HEADER_STOP_COUNT
            pst["header_cap"] = 32
            pst["slot_mode"] = cond["slot"]

            # ---------------- FULL ------------------------------------------------------
            if cond["mode"] == "full":
                _OV["cid"] = None
                model.eval()
                model._pooled_keep_holder = None
                lg = model(x, logits_to_keep=pred_pos[None])[0].float()
                comp, ff = 1.0, 1.0
                sel_by_doc, x_use = None, x
                tok_doc = float(T - n_free_nondoc) / max(1, n_docs)
                body_doc = float("nan")

                def fwd_full(seq):
                    return model(seq, logits_to_keep=1)[0][-1]
                fwd, prefix = fwd_full, x[:, :ans_start].clone()
            else:
                # -------- the dense preview (shared by every keep-set at this L) ----------
                if cond["mode"] == "preview" and not cond["swap"]:
                    L = cond["L"]
                    if ("hcut", L) not in prev_cache:
                        qsel = torch.arange(max(0, ans_start - a.n_query), ans_start, device="cuda")
                        cut_layers = [li for li in attn_layers if li < L]
                        _OV["cid"] = cid_h.cuda()
                        _CB["sig"], _CB["out"] = None, None
                        model.eval()
                        model._pooled_keep_holder = PooledDocKeepHolder(
                            keep_docs=torch.zeros(n_docs, dtype=torch.bool)[None])
                        if cut_layers:
                            capture.arm(qsel, only=cut_layers)
                        _, _, h_cut0 = preview_forward(model, x, cid_h[0], n_docs, L, pred_pos)
                        per_layer = capture.disarm() if cut_layers else {}
                        prev_cache[("hcut", L)] = h_cut0
                        sp = None
                        if per_layer:
                            v = sum(per_layer[li] for li in cut_layers if li in per_layer)
                            if isinstance(v, torch.Tensor):
                                sp = np.zeros(T, dtype=np.float32)
                                vv = v.numpy()
                                sp[: min(T, len(vv))] = vv[:T]
                        prev_cache[("sal", L)] = sp
                        # the memo now holds the HEADER-ONLY compaction under this condition's
                        # key; the real forward below uses a DIFFERENT chunk-id override, so it
                        # must not be served from it
                        _CB["sig"], _CB["out"] = None, None
                    ctx["sal_prev"] = prev_cache.get(("sal", cond["L"]))

                # ---------------- build the real-token subset ---------------------------
                base = cid_h.clone()
                sel_by_doc = []
                for d in range(n_docs):
                    s = select_positions(cond["kind"], cond["k"], bpos[d], bids[d], ctx, rng=rng)
                    sel_by_doc.append([int(p) for p in s])
                    if s:
                        base[0, torch.tensor(list(s), dtype=torch.long)] = -1
                _OV["cid"] = base.cuda()
                x_use = swap_kept(x_cpu, sel_by_doc, gold_row, n_docs, a.seed).cuda() \
                    if cond["swap"] else x
                body_doc = float(np.mean([len(s) for s in sel_by_doc])) if sel_by_doc else 0.0

                if cond["mode"] == "pool":
                    model.train()
                    model._pooled_keep_holder = PooledDocKeepHolder(
                        keep_docs=torch.zeros(n_docs, dtype=torch.bool)[None])
                    cb = model._compact_pooled_soft_tokens(x_use, None, -100)[0]
                    posmap = {int(p): c for c, p in enumerate(cb.position_ids[0].tolist())}
                    cols = torch.tensor([posmap[int(p)] for p in pred_pos.tolist()], device="cuda")
                    assert int(cols.max()) < cb.input_ids.shape[1], "compaction/column mismatch"
                    lg = model(x_use, logits_to_keep=cols[None])[0].float()
                    T2 = cb.input_ids.shape[1]
                    comp = T2 / T
                    ff = flop_fraction(lin, quad, T, T2, L=0)
                    tok_doc = float(T2 - n_free_nondoc) / max(1, n_docs)

                    def fwd_pool(seq):
                        return model(seq, logits_to_keep=1)[0][-1]
                    fwd, prefix = fwd_pool, x_use[:, :ans_start].clone()
                else:  # preview
                    model.eval()
                    model._pooled_keep_holder = PooledDocKeepHolder(
                        keep_docs=torch.zeros(n_docs, dtype=torch.bool)[None])
                    L = cond["L"]
                    cid_use = base[0]
                    if cond["swap"]:  # the swap changes the token ids, so the preview changes too
                        lg, plan, _ = preview_forward(model, x_use, cid_use, n_docs, L, pred_pos)
                    else:
                        lg, plan = preview_from_cut(model, x_use, prev_cache[("hcut", L)],
                                                    cid_use, n_docs, L, pred_pos)
                    lg = lg.float()
                    T2 = plan.T2
                    comp = T2 / T
                    ff = flop_fraction(lin, quad, T, T2, L=L)
                    tok_doc = float(T2 - n_free_nondoc) / max(1, n_docs)

                    def fwd_prev(seq, _L=L, _cid=cid_use, _nd=n_docs):
                        cu = _cid
                        if seq.shape[1] > cu.shape[0]:
                            cu = torch.cat([cu, torch.full((seq.shape[1] - cu.shape[0],), -1,
                                                           dtype=cu.dtype, device=cu.device)])
                        else:
                            cu = cu[: seq.shape[1]]
                        _OV["cid"] = cu[None].cuda()
                        _CB["sig"], _CB["out"] = None, None
                        return preview_forward(model, seq, cu, _nd, _L,
                                               torch.tensor([seq.shape[1] - 1], device=seq.device))[0][-1]
                    fwd, prefix = fwd_prev, x_use[:, :ans_start].clone()

            r = acc[name]
            r["ce"].append(float(F.cross_entropy(lg, targets)))
            if digit_sel.numel():
                r["ce_digit"].append(float(F.cross_entropy(lg[digit_sel], targets[digit_sel])))
            r["compaction"].append(comp)
            r["flop_frac"].append(ff)
            r["tok_per_doc"].append(tok_doc)
            r["body_per_doc"].append(body_doc)
            r["n_docs"].append(float(n_docs))

            # ---- what kind of token did the selector pick? -----------------------------
            if sel_by_doc and cond["k"] > 0 and ri < 32:
                st = sel_stats[name]
                xr_np = x_cpu[0].numpy()
                allsel = [p for sd in sel_by_doc for p in sd]
                if allsel:
                    ids_sel = xr_np[allsel]
                    st["digit"].append(float(ctx0["is_dig"][ids_sel].mean()))
                    st["cap"].append(float(ctx0["is_cap"][ids_sel].mean()))
                    pct, rel, fs, ov = [], [], [], []
                    for d in range(n_docs):
                        n = len(bpos[d])
                        sd = sel_by_doc[d]
                        if n == 0 or not sd:
                            continue
                        di = np.sort(ctx0["idf"][bids[d]])
                        si = ctx0["idf"][xr_np[sd]]
                        # idf percentile of the kept tokens WITHIN their own document
                        pct.append(float(np.mean(np.searchsorted(di, si) / float(n))))
                        se = ctx0["sent_end"][bids[d]]
                        end = int(np.argmax(se)) if se.any() else n - 1
                        first_sent = set(int(p) for p in bpos[d][: end + 1])
                        ordi = np.searchsorted(bpos[d], np.asarray(sd))
                        rel.append(float(np.mean(ordi / max(1, n - 1))))
                        fs.append(float(np.mean([p in first_sent for p in sd])))
                        if ctx.get("sal_grad") is not None and cond["kind"] != "grad" and n > cond["k"]:
                            go = set(topk_positions(bpos[d], ctx["sal_grad"][bpos[d]], cond["k"]))
                            ov.append(len(go & set(sd)) / float(cond["k"]))
                    if pct:
                        st["idf_pct"].append(float(np.mean(pct)))
                        st["relpos"].append(float(np.mean(rel)))
                        st["first_sent"].append(float(np.mean(fs)))
                    if ov:
                        st["overlap_grad"].append(float(np.mean(ov)))
                if name not in sel_samples and ri < 3:
                    ex = []
                    for d in range(min(3, n_docs)):
                        if sel_by_doc[d]:
                            ex.append(tok.decode([int(v) for v in xr_np[sel_by_doc[d]]]))
                    sel_samples[name] = ex

            keep_mask = torch.zeros(n_docs, dtype=torch.bool) if cond["mode"] != "full" else None
            if keep_mask is None:
                pooled_gold, real_gold = [], list(gold_docs)
            else:
                pooled_gold = [d for d in gold_docs if d < n_docs]
                real_gold = []

            if do_gen:
                _CB["sig"], _CB["out"] = None, None
                gen = generate(fwd, prefix, a.gen_max_new, len(true_ids) or 3, tok)
                gtext = tok.decode(gen)
                gen_ids = parse_ids(gtext)
                r["gen_f1"].append(set_f1(gen_ids, true_ids))
                r["gen_em"].append(float(set(gen_ids) == set(true_ids) and len(gen_ids) == len(true_ids)))
                pred_docs = [i - off for i in gen_ids]
                r["hit_gold_pooled"] += [1.0 if d in pred_docs else 0.0 for d in pooled_gold]
                r["hit_gold_real"] += [1.0 if d in pred_docs else 0.0 for d in real_gold]
                if len(dumps) < a.dump_rows and name in DUMP_CONDS:
                    dumps.append({"row": ri, "cond": name, "true_ids": true_ids,
                                  "gen_ids": gen_ids, "gen": gtext})
            r["sec"].append(time.time() - t_cfg)
            model.eval()
        _OV["cid"] = None

        if ri + 1 in (1, 2, 5, 10) or (ri + 1) % 20 == 0:
            el = time.time() - t_start
            eta = el / (ri + 1) * (len(score_rows) - ri - 1) / 60.0
            log(f"row {ri + 1}/{len(score_rows)}  full CE {acc['full']['ce'][-1]:.3f}  "
                f"elapsed {el / 60:.1f} min  ETA {eta:.0f} min")
            table(acc, conds)

    table(acc, conds)
    for d in dumps:
        print(f"  dump row {d['row']:3d} {d['cond']:12} true={d['true_ids']} pred={d['gen_ids']} "
              f"{d['gen']!r}", flush=True)
    print("\n=== what the selector picks (first 64 rows) ===", flush=True)
    print(f"{'condition':14} {'idf-pctile':>11} {'relpos':>7} {'digit':>7} {'capital':>8} "
          f"{'1st-sent':>9} {'overlap@k grad':>15}", flush=True)
    for c in conds:
        st = sel_stats[c["name"]]
        if not st["digit"]:
            continue
        m = lambda k: (float(np.mean(st[k])) if st[k] else float("nan"))  # noqa: E731
        print(f"{c['name']:14} {m('idf_pct'):11.3f} {m('relpos'):7.3f} {m('digit'):7.3f} "
              f"{m('cap'):8.3f} {m('first_sent'):9.3f} {m('overlap_grad'):15.3f}", flush=True)
    for n, ex in sel_samples.items():
        print(f"  [{n}] kept tokens, 3 docs of row <3: {ex}", flush=True)

    summ = {c["name"]: summarize(acc[c["name"]], acc["full"]) for c in conds}
    for c in conds:
        summ[c["name"]]["sel_stats"] = {k: (float(np.mean(v)) if v else None)
                                        for k, v in sel_stats[c["name"]].items()}
    verdict(summ)
    out = {
        "task": "outlier", "mode": a.mode, "rung": rung, "eval_size": len(acc["full"]["ce"]),
        "rows_loaded": len(score_rows), "gen_rows": min(a.gen_rows, len(score_rows)),
        "fit_rows": len(fit_rows), "rule_fit": rule_info,
        "ckpt": a.resolved_ckpt, "ckpt_name": a.ckpt_name, "seed": a.seed,
        "header_stop_id": HEADER_STOP_ID, "fingerprint_misses": n_miss,
        "n_layers": n_layers, "attn_layers": attn_layers,
        "conditions": summ,
        "construction": {c["name"]: {k: c[k] for k in ("kind", "k", "mode", "L", "swap", "slot")}
                         for c in conds},
        "per_row": {c["name"]: dict(acc[c["name"]]) for c in conds},
        "sel_samples": sel_samples, "dumps": dumps,
    }
    sfx = f"_{a.tag}" if a.tag else ""
    local = a.out if a.out.endswith(".json") else f"{a.out}/salprev_{rung}.json"
    if len(a.rungs.split(",")) > 1 and f"_{rung}" not in local:
        local = local[:-5] + f"_{rung}.json"
    weka = None if a.weka_out in ("", "none") else \
        f"{a.weka_out}/saliency_preview_{a.mode}_{a.ckpt_name}_{rung}{sfx}.json"
    for path in [local] + ([weka] if weka else []):
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            json.dump(out, open(path, "w"), indent=1)
            log(f"wrote {path}")
        except Exception as e:
            log(f"could not write {path}: {e}")
    return summ


def summarize(r, full):
    d = {}
    for k, v in r.items():
        m, se = mean_se(v)
        d[k], d[f"{k}_se"], d[f"{k}_count"] = m, se, len(v)
    d["dce"], d["dce_se"] = paired(r["ce"], full["ce"])
    d["dce_digit"], d["dce_digit_se"] = paired(r["ce_digit"], full["ce_digit"])
    d["dgen_f1"], d["dgen_f1_se"] = paired(r["gen_f1"], full["gen_f1"])
    return d


def table(acc, conds):
    print(f"{'condition':14} {'CE':>7} {'dCE':>8} {'CEdig':>7} {'dCEdig':>8} {'genF1':>6} "
          f"{'dF1':>7} {'R@gp':>11} {'tok/doc':>8} {'real':>6} {'compact':>8} {'FLOPfrac':>8} "
          f"{'s/row':>6}", flush=True)
    fu = acc.get("full")
    for c in conds:
        r = acc[c["name"]]
        if not r["ce"]:
            continue
        m = lambda k: (np.mean(r[k]) if r[k] else float("nan"))  # noqa: E731
        dce, _ = paired(r["ce"], fu["ce"])
        dcd, _ = paired(r["ce_digit"], fu["ce_digit"])
        df1, _ = paired(r["gen_f1"], fu["gen_f1"])
        print(f"{c['name']:14} {m('ce'):7.3f} {dce:+8.3f} {m('ce_digit'):7.3f} {dcd:+8.3f} "
              f"{m('gen_f1'):6.3f} {df1:+7.3f} "
              f"{m('hit_gold_pooled'):6.3f}[{len(r['hit_gold_pooled']):3d}] "
              f"{m('tok_per_doc'):8.1f} {m('body_per_doc'):6.1f} {m('compaction'):8.3f} "
              f"{m('flop_frac'):8.3f} {m('sec'):6.2f}", flush=True)


def verdict(summ):
    """The deliverable: the cheapest construction with dCE <= 1 paired SE and genF1 within noise."""
    print("\n=== PARITY CHECK (dCE <= 1 paired SE AND |dF1| <= 1 paired SE) ===", flush=True)
    print(f"{'condition':14} {'dCE':>8} {'SE':>6} {'dCEdig':>8} {'SE':>6} {'dF1':>7} {'SE':>6} "
          f"{'tok/doc':>8} {'FLOPfrac':>8}  verdict", flush=True)
    rows = []
    for name, s in summ.items():
        if name == "full":
            continue
        ok_ce = s["dce"] <= s["dce_se"] if s["dce_se"] == s["dce_se"] else False
        ok_f1 = abs(s["dgen_f1"]) <= s["dgen_f1_se"] if s["dgen_f1_se"] == s["dgen_f1_se"] else False
        v = "PARITY" if (ok_ce and ok_f1) else ("ce-ok" if ok_ce else ("f1-ok" if ok_f1 else ""))
        print(f"{name:14} {s['dce']:+8.3f} {s['dce_se']:6.3f} {s['dce_digit']:+8.3f} "
              f"{s['dce_digit_se']:6.3f} {s['dgen_f1']:+7.3f} {s['dgen_f1_se']:6.3f} "
              f"{s['tok_per_doc']:8.1f} {s['flop_frac']:8.3f}  {v}", flush=True)
        if v == "PARITY":
            rows.append((s["flop_frac"], name))
    if rows:
        rows.sort()
        print(f"CHEAPEST PARITY: {rows[0][1]} at FLOP fraction {rows[0][0]:.3f}", flush=True)
    else:
        print("NO construction reaches parity at this rung -- read the dCE-vs-cost curve above.",
              flush=True)


if __name__ == "__main__":
    main()
