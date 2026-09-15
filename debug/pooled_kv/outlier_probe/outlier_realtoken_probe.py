"""
REAL-TOKEN subsets per document: can a FROZEN dense model reach FULL-attention parity on outlier?
(eval-only, 2026-09-15)

Where this starts.  ``records/outlier-slot-probe.md`` and ``records/outlier-richer-slot-probe.md``
between them exhaust the *slot vector* idea: mean / meanrn / cmean100 / cmean500 / centered /
cent_cmean / idf / g2 / g4 / enc2 / enc4 / enc4late all leave the frozen dense model exactly on the
``k/n`` uniform-guess floor for ``R@gold_pooled``, with the slot-swap control undetectable.  One
vector per document is not a readable summary of "is this document the odd one out" for a model
that never trained on slots.

``records/soft-kv-slot-probe-handoff.md`` records the construction that DID reach parity on
contradiction and oolong: keep each document's **HEADER** real (exact-match tokens) and pool the
body -- ``--st-header-stop-id 25``, dCE ~ 0 on the frozen model.  That is a *real-token subset*, not
a richer vector.  This probe asks the same question for outlier with a bigger, still-O(k)-per-doc
real-token subset:

    keep k REAL body tokens per document at their ORIGINAL positions, pool (or drop) the rest.

Constructions (``--conditions``), all gold-blind unless stated (gold documents get exactly the same
treatment as non-gold), all with the ``\\n\\nDocument [N]:`` header real (stop id 5491, count 1) so
the document ids stay nameable:

  full            plain full attention over the real tokens (the parity reference)
  cc00            header real, keep 0, ``cent_cmean`` slot -- the known-at-the-floor control
  first{k}        first k body tokens real, remainder POOLED into one cent_cmean slot (k=4..64)
  first{k}d       the same subset, remainder DROPPED (no slot) -- does the slot add anything?
  idf{k}          the k body tokens with the highest -log(corpus freq), non-contiguous, remainder pooled
  idfspan{k}      the contiguous span of length k with the highest summed idf
  fl{k}           first k/2 + last k/2 body tokens
  sent1           the first sentence (to the first '.'/newline piece, capped at 32 tokens)
  <cond>_swap     CONTROL: the kept REAL tokens of the gold documents exchanged with those of
                  random non-gold documents (positions, ids, headers, slots all untouched).
                  Recall must DROP, or the model is not reading the kept tokens.
  goldonly        gold docs fully REAL, every non-gold doc pooled, NO header, ``mean`` slot
                  (gold-AWARE; the 2026-09-08 "parity" reference, not a training recipe)
  goldonly_cc     the same with a ``cent_cmean`` slot

Every candidate is ``O(k)`` tokens per document, computed from the document's own tokens plus a
precomputed corpus token-frequency table: nothing per-layer, nothing per-token-per-layer, no second
network.

Metrics per condition x rung: answer CE (and the paired dCE vs FULL with its SE), CE on the DIGIT
tokens of the answer (the retrieval decision -- mean answer CE on outlier is ~95% prose, see
``records/outlier-slot-probe.md`` Sec. 4), free-generation set-F1 over the k document ids, recall of
the gold documents split by whether that document's body was pooled or real, real tokens kept per
document, compaction, and the linear-term FLOP fraction (= compaction).

    python debug/pooled_kv/outlier_probe/outlier_realtoken_probe.py --rung 2k --rows 240 \\
        --gen-rows 64 --conditions all --ckpt-name ds64-outlier-dense-u64M --work /results/w
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
    print(f"[realtoken] {m}", flush=True)


# --------------------------------------------------------------------------------------------
# conditions
# --------------------------------------------------------------------------------------------
def C(name, sel, k, mode="pool", slot="cent_cmean", header=True, keep="none", swap=False,
      group=None, gk=0, decoy=0, catslot=False):
    """``sel`` in {none, first, firstlast, idf, idfspan, sent1, all}; ``mode`` in {full, pool, drop};
    ``keep`` in {none (every doc pooled), gold (gold docs real), gpr (gold + 1/3 random),
    all (no pooling)}.

    ``group`` selects a DOCUMENT-LEVEL keep rule computed from the per-document ``cent_cmean``
    vectors (gold-blind unless noted): ``smallcat`` (keep every doc in the ``gk`` smallest topical
    clusters whole), ``smallcatle`` (every cluster of size <= ``gk``), ``margin`` (the ``gk`` docs
    farthest from the row centroid PLUS the ``gk`` nearest the k-outlier decision boundary), or
    ``hardneg`` (ORACLE: gold pooled, the ``gk`` non-gold docs closest to gold kept real).
    ``decoy`` additionally keeps ``decoy`` random LARGE clusters whole, so "kept whole" no longer
    implies "small".  ``catslot`` gives each pooled CLUSTER one slot instead of each pooled doc.
    Documents chosen by a ``group`` rule are kept WHOLE; ``sel``/``k`` still applies to the rest."""
    return dict(name=name, sel=sel, k=k, mode=mode, slot=slot, header=header, keep=keep, swap=swap,
                group=group, gk=gk, decoy=decoy, catslot=catslot)


def build_conditions():
    cs = [
        C("full", "all", 0, mode="full", keep="all", header=False),
        C("goldonly", "none", 0, mode="pool", slot="mean", header=False, keep="gold"),
        C("goldonly_cc", "none", 0, mode="pool", slot="cent_cmean", header=False, keep="gold"),
        C("cc00", "none", 0),
    ]
    for k in (4, 8, 16, 32, 64):
        cs.append(C(f"first{k}", "first", k))
    for k in (4, 8, 16, 32, 64):
        cs.append(C(f"first{k}d", "first", k, mode="drop"))
    for k in (4, 8, 16):
        cs.append(C(f"idf{k}", "idf", k))
        cs.append(C(f"idfspan{k}", "idfspan", k))
    for k in (8, 16, 32):
        cs.append(C(f"fl{k}", "firstlast", k))
    cs.append(C("sent1", "sent1", 32))
    # --- document-level keep rules over the per-doc cent_cmean vectors --------------------------
    for cc in (1, 2, 3, 5):
        cs.append(C(f"smallcat{cc}", "none", 0, group="smallcat", gk=cc))
        cs.append(C(f"smallcat{cc}cat", "none", 0, group="smallcat", gk=cc, catslot=True))
        for kk in (8, 16):
            cs.append(C(f"smallcat{cc}f{kk}", "first", kk, group="smallcat", gk=cc))
        for dd in (1, 2):
            cs.append(C(f"smallcat{cc}d{dd}", "none", 0, group="smallcat", gk=cc, decoy=dd))
    for ss in (2, 3, 5):
        cs.append(C(f"smallcatle{ss}", "none", 0, group="smallcatle", gk=ss))
    for mm in (3, 6, 10):
        cs.append(C(f"margin{mm}", "none", 0, group="margin", gk=mm))
        cs.append(C(f"margin{mm}f16", "first", 16, group="margin", gk=mm))
    for mm in (6, 10):
        cs.append(C(f"hardneg{mm}", "none", 0, group="hardneg", gk=mm))  # ORACLE (uses gold)
    for kk in (2, 4):
        # GOLD-AWARE: the gold document's whole topical category real, plus K other whole
        # categories, everything else pooled -- the proposed training recipe.
        cs.append(C(f"goldcats{kk}", "none", 0, group="goldcats", gk=kk))
        cs.append(C(f"goldcats{kk}r", "none", 0, group="goldcatsr", gk=kk))
    # the old gold_plus_random construction, as the mechanistic diagnostic
    cs.append(C("gpr33", "none", 0, keep="gpr"))
    # swap controls (the kept REAL tokens of gold docs exchanged with random non-gold docs')
    for base in ("first8", "first16", "first32", "first64", "idfspan16", "sent1",
                 "smallcat3", "smallcat5"):
        b = next(c for c in cs if c["name"] == base)
        cs.append(dict(b, name=f"{base}_swap", swap=True))
    return cs


PRESETS = {
    "all": None,  # everything
    # the topical-category keep rules (the user's priority construction) + the cheapest
    # real-token subsets to compare them against + the mechanistic diagnostic
    "cat": ["full", "goldonly", "goldonly_cc", "gpr33", "cc00",
            "smallcat1", "smallcat2", "smallcat3", "smallcat5", "smallcat3cat",
            "smallcat5cat", "smallcat3f16",
            "smallcatle2", "smallcatle3", "smallcat3d1", "smallcat3d2",
            "margin3", "margin6", "margin10", "margin6f16", "hardneg6", "hardneg10",
            "goldcats2", "goldcats4", "goldcats2r",
            "first16", "first32", "first64", "smallcat3_swap"],
    "cat32k": ["full", "goldonly", "gpr33", "cc00",
               "smallcat1", "smallcat3", "smallcat5", "smallcat3cat", "smallcat3f16",
               "smallcatle3",
               "smallcat3d1", "margin6", "margin6f16", "hardneg6",
               "goldcats2", "goldcats4",
               "first32", "first64", "smallcat3_swap"],
    "core": ["full", "goldonly", "goldonly_cc", "cc00", "first8", "first16", "first32", "first64",
             "first16d", "first32d", "idf16", "idfspan16", "fl32", "sent1", "first32_swap"],
    "lean": ["full", "goldonly", "goldonly_cc", "cc00", "first16", "first32", "first64",
             "first32d", "idfspan16", "sent1", "first32_swap"],
}


# --------------------------------------------------------------------------------------------
# chunk-id override: the mechanism that makes a REAL-TOKEN SUBSET possible
# --------------------------------------------------------------------------------------------
# ``Transformer._compact_pooled_soft_tokens`` builds chunk ids from the token stream and then, when
# ``header_stop_id`` is set, calls ``mark_doc_headers_free`` to re-label the header FREE.  FREE
# tokens survive compaction at their ORIGINAL positions and are excluded from the pooled slot mean.
# So "keep an arbitrary subset of a document's tokens real" is exactly "mark that subset FREE".
# model.py imports the function INSIDE the forward, so patching the module attribute reaches it.
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
# compacts again; if the two ever disagree the gathered columns go out of bounds and CUDA reports it
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


# --------------------------------------------------------------------------------------------
# corpus statistics (idf) and sentence ends
# --------------------------------------------------------------------------------------------
def build_idf_from_rows(rows, vocab):
    """Fallback when the training shard is not mounted (local runs): estimate the corpus token
    frequencies from the EVAL rows themselves.  A rung file holds hundreds of documents drawn from
    the same corpus, so this is the same statistic, measured on the same distribution."""
    ids = np.concatenate([np.asarray(r, dtype=np.int64) for r in rows])
    ids = ids[(ids >= 0) & (ids < vocab)]
    cnt = np.bincount(ids, minlength=vocab)[:vocab].astype(np.float64)
    p = (cnt + 1.0) / (cnt.sum() + float(vocab))
    return (-np.log(p)).astype(np.float32), int(ids.size)


def build_idf(shard_dir, vocab, n_rows=512):
    """``-log p(token)`` over the head of the TRAINING shard (add-one smoothed).

    ``token_ids_part_*.npy`` is a RAW HEADERLESS array despite the extension -- ``np.load`` dies.
    """
    parts = sorted(_glob.glob(f"{shard_dir}/token_ids_part_*.npy"))
    if not parts:
        raise SystemExit(f"--idf-shard {shard_dir} has no token_ids_part_*.npy")
    meta = json.load(open(f"{shard_dir}/metadata.json"))
    dtype = np.dtype(meta.get("dtype") or "uint32")
    n_total = os.path.getsize(parts[0]) // dtype.itemsize
    arr = np.memmap(parts[0], dtype=dtype, mode="r", shape=(n_total,))
    row_len = int(meta.get("max_example_len") or 65536)
    n_tok = min(n_total, max(1, n_rows) * max(1, row_len))
    ids = np.asarray(arr[:n_tok], dtype=np.int64)
    cnt = np.bincount(ids, minlength=vocab)[:vocab].astype(np.float64)
    p = (cnt + 1.0) / (cnt.sum() + float(vocab))
    return (-np.log(p)).astype(np.float32), int(n_tok)


def build_sent_end(tok, vocab):
    """``(vocab,)`` bool: this token's piece ends a sentence ('.' or a newline)."""
    pieces = tok.convert_ids_to_tokens(list(range(vocab)))
    out = np.zeros(vocab, dtype=bool)
    for i, s in enumerate(pieces):
        if s is None:
            continue
        if "." in s or "\n" in s or "Ċ" in s:
            out[i] = True
    return out


# --------------------------------------------------------------------------------------------
# per-document real-token selection
# --------------------------------------------------------------------------------------------
def select_positions(kind, k, body_pos, body_ids, idf, sent_end):
    """Positions (original row indices) of the real-token subset for one document."""
    n = len(body_pos)
    if kind == "none" or n == 0 or k <= 0:
        return []
    if kind == "first":
        return list(body_pos[:k])
    if kind == "firstlast":
        return _firstlast(body_pos, k)
    if kind == "idf":
        if n <= k:
            return list(body_pos)
        sc = idf[body_ids]
        order = np.argsort(-sc, kind="stable")[:k]
        return [int(body_pos[i]) for i in sorted(order.tolist())]
    if kind == "idfspan":
        if n <= k:
            return list(body_pos)
        sc = idf[body_ids].astype(np.float64)
        cs = np.concatenate([[0.0], np.cumsum(sc)])
        sums = cs[k:] - cs[:-k]
        i = int(np.argmax(sums))
        return list(body_pos[i:i + k])
    if kind == "sent1":
        out = []
        for p, t in zip(body_pos, body_ids):
            out.append(int(p))
            if sent_end[t] or len(out) >= k:
                break
        return out
    raise ValueError(kind)


def _firstlast(body_pos, k):
    n = len(body_pos)
    if n <= k:
        return list(body_pos)
    a, b = k // 2, k - k // 2
    return list(body_pos[:a]) + list(body_pos[n - b:])




# --------------------------------------------------------------------------------------------
# document-level keep rules over the per-document cent_cmean vectors
# --------------------------------------------------------------------------------------------
N_OUTLIERS = 3  # the task constant k -- knowledge of the TASK, not of this row's gold


def doc_cent_vectors(model, x, cid, n_docs, stop_mask):
    """``(n_docs, D)`` L2-normalised ``cent_cmean`` vectors and a ``present`` mask.

    Same construction as ``olmo_core.nn.pooled_soft_token.apply_slot_mode(mode='cent_cmean')``:
    the mean input embedding over the document's CONTENT tokens (stop set dropped), minus the row
    centroid over the same tokens.  A document with no surviving content token falls back to its
    plain mean.  Normalised here because everything downstream is a cosine.
    """
    emb = model.embeddings(x)[0]
    cidl = cid.to(x.device).long()[0]
    is_ctx = cidl >= 0
    content = is_ctx & ~stop_mask[x[0]]
    D = emb.shape[-1]
    dev = x.device

    def acc(mask):
        idx = cidl[mask]
        sums = torch.zeros(n_docs, D, dtype=torch.float32, device=dev)
        cnts = torch.zeros(n_docs, dtype=torch.float32, device=dev)
        if idx.numel():
            sums.index_add_(0, idx, emb[mask].float())
            cnts.index_add_(0, idx, torch.ones(idx.numel(), device=dev))
        return sums, cnts

    sums, cnts = acc(content)
    s2, c2 = acc(is_ctx)
    empty = cnts == 0
    if bool(empty.any()):
        sums[empty], cnts[empty] = s2[empty], c2[empty]
    present = c2 > 0
    means = sums / cnts.clamp(min=1.0).unsqueeze(-1)
    centroid = sums.sum(0) / cnts.sum().clamp(min=1.0)
    v = means - centroid
    v = v / v.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    return v.float().cpu().numpy(), present.cpu().numpy()


def avg_linkage_clusters(V, present, kmax=12):
    """Average-linkage agglomerative clustering on COSINE distance, cut at the largest relative
    gap in the merge-distance sequence (parameter-free elbow) among cuts leaving 2..``kmax``
    clusters.

    :returns: ``(clusters, n_clusters, merge_gap)`` with ``clusters`` a list of doc-id lists
        sorted by size ASCENDING.
    """
    ids = [int(i) for i in np.nonzero(present)[0]]
    n = len(ids)
    if n <= 2:
        return [[i] for i in ids], n, 0.0
    X = V[ids]
    Dm = (1.0 - X @ X.T).astype(np.float64)
    np.fill_diagonal(Dm, np.inf)
    alive = np.ones(n, dtype=bool)
    size = np.ones(n)
    members = [[i] for i in range(n)]
    ds, parts = [], []
    for _ in range(n - 1):
        sub = np.nonzero(alive)[0]
        M = Dm[np.ix_(sub, sub)]
        f = int(np.argmin(M))
        i_, j_ = divmod(f, len(sub))
        i, j = int(sub[i_]), int(sub[j_])
        ds.append(float(M[i_, j_]))
        new = (size[i] * Dm[i, :] + size[j] * Dm[j, :]) / (size[i] + size[j])
        Dm[i, :] = new
        Dm[:, i] = new
        Dm[i, i] = np.inf
        Dm[j, :] = np.inf
        Dm[:, j] = np.inf
        alive[j] = False
        size[i] += size[j]
        members[i] = members[i] + members[j]
        parts.append([list(members[k]) for k in np.nonzero(alive)[0]])
    best_c, best_gap = 2, -1.0
    for c in range(2, min(kmax, n - 1) + 1):
        t = n - c          # merges applied to leave c clusters
        nxt = ds[t] if t < len(ds) else ds[-1]
        cur = ds[t - 1] if t >= 1 else 1e-9
        gap = nxt / max(cur, 1e-9)
        if gap > best_gap:
            best_c, best_gap = c, gap
    part = parts[n - best_c - 1] if best_c < n else [[k] for k in range(n)]
    clus = [sorted(ids[k] for k in grp) for grp in part]
    clus.sort(key=len)
    return clus, len(clus), float(best_gap)


def robust_centroid_cos(V, present):
    """Cosine of every document to a ROBUST majority centroid (mean of the 75% of documents
    closest to the plain mean, renormalised).  ``nan`` for absent documents."""
    ids = np.nonzero(present)[0]
    X = V[ids]
    c = X.mean(0)
    c /= max(np.linalg.norm(c), 1e-6)
    cs = X @ c
    keep = ids[np.argsort(-cs)[: max(2, int(round(0.75 * len(ids))))]]
    c = V[keep].mean(0)
    c /= max(np.linalg.norm(c), 1e-6)
    out = np.full(V.shape[0], np.nan, dtype=np.float64)
    out[ids] = V[ids] @ c
    return out


def group_keep(cond, clus, cos_c, V, present, gold_row, n_docs, seed):
    """The set of documents this condition keeps WHOLE, and (for ``catslot``) the pooled-cluster
    map.  Returns ``(keep_set, cluster_of_doc)``."""
    g = cond["group"]
    cl_of = {d: ci for ci, grp in enumerate(clus) for d in grp}
    keep = set()
    if g in ("smallcat", "smallcatle", "goldcats", "goldcatsr"):
        if g == "smallcat":
            chosen = list(range(min(cond["gk"], len(clus))))          # clus is size-ascending
        elif g == "smallcatle":
            chosen = [ci for ci, grp in enumerate(clus) if len(grp) <= cond["gk"]]
        else:
            gold_cl = sorted({cl_of[d] for d in range(n_docs) if bool(gold_row[d]) and d in cl_of})
            rest = [ci for ci in range(len(clus)) if ci not in set(gold_cl)]
            if g == "goldcats":
                extra = rest[: cond["gk"]]                             # the K smallest non-gold
            else:
                rng = np.random.RandomState(seed + n_docs)
                extra = list(rng.permutation(rest)[: cond["gk"]]) if rest else []
            chosen = gold_cl + [int(c) for c in extra]
        if cond["decoy"]:
            big = [ci for ci in range(len(clus)) if ci not in set(chosen)][::-1]  # largest first
            rng = np.random.RandomState(seed + 7 * n_docs)
            chosen = chosen + [int(c) for c in rng.permutation(big[: max(1, len(big))])[: cond["decoy"]]]
        for ci in chosen:
            keep |= set(clus[ci])
    elif g == "margin":
        ids = np.nonzero(present)[0]
        cs = cos_c[ids]
        order = ids[np.argsort(cs)]                                    # farthest from centroid first
        M = cond["gk"]
        far = list(order[:M])
        srt = np.sort(cs)
        tau = 0.5 * (srt[N_OUTLIERS - 1] + srt[N_OUTLIERS]) if len(srt) > N_OUTLIERS else srt.mean()
        amb = list(ids[np.argsort(np.abs(cs - tau))][:M])
        keep = {int(d) for d in far} | {int(d) for d in amb}
    elif g == "hardneg":
        gold = [d for d in range(n_docs) if bool(gold_row[d]) and present[d]]
        non = [d for d in range(n_docs) if not bool(gold_row[d]) and present[d]]
        if gold and non:
            sc = (V[non] @ V[gold].T).max(1)
            keep = {int(non[i]) for i in np.argsort(-sc)[: cond["gk"]]}
    else:
        raise ValueError(g)
    return keep, cl_of


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
    return float(a.mean()), float(a.std(ddof=1) / max(1, len(a) ** 0.5)) if len(a) > 1 else 0.0


def paired(a, b):
    """mean and SE of the PAIRED difference a - b over the rows both have."""
    n = min(len(a), len(b))
    if n < 2:
        return float("nan"), float("nan")
    d = np.asarray(a[:n], dtype=np.float64) - np.asarray(b[:n], dtype=np.float64)
    return float(d.mean()), float(d.std(ddof=1) / n ** 0.5)


# --------------------------------------------------------------------------------------------
@torch.no_grad()
def main():
    global HEADER_STOP_ID
    ap = argparse.ArgumentParser()
    ap.add_argument("--rung", default="2k")
    ap.add_argument("--rungs", default=None, help="comma list, scored in ONE process")
    ap.add_argument("--rows", type=int, default=240)
    ap.add_argument("--gen-rows", type=int, default=64)
    ap.add_argument("--gen-max-new", type=int, default=48)
    ap.add_argument("--ckpt-name", default="ds64-outlier-dense-u64M")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--jsonl", default=None, help="override eval JSONL (local runs)")
    ap.add_argument("--shard", default=None, help="already-tokenized shard dir (skips conversion)")
    ap.add_argument("--conditions", default="core",
                    help="a preset (all|core|lean) or a comma list of condition names")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--work", default="/results/realtoken_work")
    ap.add_argument("--out", default="/results/outlier_realtoken_probe.json")
    ap.add_argument("--weka-out", default=f"{W}/_eval_results/outlier_slot_probe")
    ap.add_argument("--tag", default="")
    ap.add_argument("--idf-shard", default=None,
                    help=f"training shard for the corpus token frequencies (default {DS64_SHARDS}/outlier_u<budget>)")
    ap.add_argument("--idf-rows", type=int, default=512)
    ap.add_argument("--header-stop-id", type=int, default=HEADER_STOP_ID)
    ap.add_argument("--dump-rows", type=int, default=8)
    a = ap.parse_args()
    HEADER_STOP_ID = int(a.header_stop_id)
    a.rungs = a.rungs or a.rung
    if a.tokenizer:
        P.TOKENIZER = a.tokenizer

    assert CM.mark_doc_headers_free is _patched_mark, "chunk-id override patch did not install"
    all_conds = build_conditions()
    known = {c["name"]: c for c in all_conds}
    if a.conditions in PRESETS:
        want = PRESETS[a.conditions] or [c["name"] for c in all_conds]
    else:
        want = [w for w in a.conditions.split(",") if w]
        if "full" not in want:
            want = ["full"] + want
    missing = [w for w in want if w not in known]
    if missing:
        raise SystemExit(f"unknown conditions {missing}; known: {sorted(known)}")
    conds = [known[w] for w in want]
    log(f"conditions ({len(conds)}): {[c['name'] for c in conds]}")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(P.TOKENIZER)
    vocab = int(P.VOCAB)

    # corpus statistics: idf from the training shard, sentence ends from the tokenizer
    idf_shard = a.idf_shard
    if idf_shard is None:
        bud = a.ckpt_name.rsplit("-u", 1)[-1] if "-u" in a.ckpt_name else "64M"
        idf_shard = f"{DS64_SHARDS}/outlier_u{bud}"
    t0 = time.time()
    try:
        idf, n_idf_tok = build_idf(idf_shard, vocab, n_rows=a.idf_rows)
        idf_src = idf_shard
    except (SystemExit, FileNotFoundError, OSError) as e:
        log(f"idf shard unavailable ({e}); falling back to the eval rows")
        idf, n_idf_tok, idf_src = None, 0, "eval-rows"
    sent_end = build_sent_end(tok, vocab)
    log(f"idf from {n_idf_tok} tokens of {idf_src} in {time.time() - t0:.0f}s; "
        f"{int(sent_end.sum())} sentence-end pieces")

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
    pst = model._pooled_soft_tokens
    install_compaction_cache(model)

    # the --st-slot-mode stop set, from the TRAINING shard, exactly as train_ctc_suite builds it
    if ({c["slot"] for c in conds} - {"mean"}) and idf is not None:
        from olmo_core.nn.pooled_soft_token import build_slot_stop_ids
        parts = sorted(_glob.glob(f"{idf_shard}/token_ids_part_*.npy"))
        meta = json.load(open(f"{idf_shard}/metadata.json"))
        dt = np.dtype(meta.get("dtype") or "uint32")
        n_total = os.path.getsize(parts[0]) // dt.itemsize
        arr = np.memmap(parts[0], dtype=dt, mode="r", shape=(n_total,))
        n_tok = min(n_total, a.idf_rows * int(meta.get("max_example_len") or 65536))
        stop, shown = build_slot_stop_ids(
            np.asarray(arr[:n_tok]), top_k=100,
            extra_ids=(IDS.doc_start, IDS.doc_end, IDS.eos, IDS.landmark, IDS.pad),
            decode=lambda t: tok.decode([t]))
        pst["slot_stop_ids"] = [int(t) for t in stop]
        pst["slot_stop_mask"] = None
        log(f"slot stop set: {len(stop)} ids; most frequent dropped: {' '.join(shown[:12])}")

    summaries = {}
    for rung in [r for r in a.rungs.split(",") if r]:
        summaries[rung] = run_rung(a, rung, conds, model, pst, tok, idf, sent_end)
    if len(summaries) > 1:
        for rung, s in summaries.items():
            print(f"\n### rung {rung}", flush=True)
            verdict(s)


@torch.no_grad()
def run_rung(a, rung, conds, model, pst, tok, idf, sent_end):  # noqa: C901
    shard = a.shard or f"{a.work}/outlier_{rung}"
    if a.shard is None:
        P.convert("outlier", a.jsonl or RUNGS[rung], a.rows, shard)
    rows, masks = P.load_rows(shard, a.rows)
    log(f"=== rung {rung}: {len(rows)} rows; lengths {[len(r) for r in rows[:6]]}")
    if idf is None:
        idf, n_t = build_idf_from_rows(rows, int(P.VOCAB))
        from olmo_core.nn.pooled_soft_token import build_slot_stop_ids
        stop, shown = build_slot_stop_ids(
            np.concatenate([np.asarray(r) for r in rows]), top_k=100,
            extra_ids=(IDS.doc_start, IDS.doc_end, IDS.eos, IDS.landmark, IDS.pad),
            decode=lambda t: tok.decode([int(t)]))
        pst["slot_stop_ids"] = [int(t) for t in stop]
        pst["slot_stop_mask"] = None
        log(f"idf + slot stop set ({len(stop)} ids) from the {n_t} eval-row tokens; "
            f"most frequent dropped: {' '.join(shown[:12])}")
    if len(rows) < 500:
        log(f"WARNING eval_size={len(rows)} (<500): binomial SE at f1~0.5 is "
            f"{0.5 / max(1, len(rows)) ** 0.5:.3f}")

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

    KEYS = ("ce", "ce_digit", "gen_f1", "gen_em", "compaction", "tok_per_doc", "body_per_doc",
            "n_docs", "sec", "kept_docs", "rule_recall", "n_cats_full", "gold_only_full")
    corpus = {k: [] for k in ("n_clusters", "smallest", "largest", "gold_in_smallest",
                              "gold_cat_purity", "cos_gold", "cos_other", "hn_rate",
                              "oracle_cosR")}
    need_vecs = any(c["group"] or c["catslot"] or c["keep"] == "gpr" for c in conds) or True
    stop_ids = pst.get("slot_stop_ids") or []
    stop_mask_t = torch.zeros(int(P.VOCAB), dtype=torch.bool, device="cuda")
    if stop_ids:
        _t = torch.tensor(stop_ids, dtype=torch.long, device="cuda")
        stop_mask_t[_t[_t < int(P.VOCAB)]] = True
    acc = {c["name"]: {k: [] for k in KEYS} for c in conds}
    for c in conds:
        acc[c["name"]]["hit_gold_pooled"] = []
        acc[c["name"]]["hit_gold_real"] = []
        acc[c["name"]]["gen"] = []
    n_miss, dumps = 0, []

    for ri, (row, rmask) in enumerate(zip(rows, masks)):
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

        # --- per-row document layout ---------------------------------------------------------
        cid0 = build_chunk_ids_from_tokens(
            x_cpu, doc_start_id=IDS.doc_start, doc_end_id=IDS.doc_end, eos_id=IDS.eos,
            mode="chunked")
        n_docs = int(cid0.max()) + 1
        cid_h = _ORIG_MARK(cid0, x_cpu, doc_start_id=IDS.doc_start, doc_end_id=IDS.doc_end,
                           stop_id=HEADER_STOP_ID, stop_count=HEADER_STOP_COUNT, cap=32)
        c_h = cid_h[0].numpy()
        xr = x_cpu[0].numpy()
        is_marker = (xr == IDS.doc_start) | (xr == IDS.doc_end)
        body_pos_by_doc, body_ids_by_doc = [], []
        for d in range(n_docs):
            sel = np.nonzero((c_h == d) & ~is_marker)[0]
            body_pos_by_doc.append(sel)
            body_ids_by_doc.append(xr[sel])
        n_header = int((c_h == -1).sum() - (cid0[0].numpy() == -1).sum())
        n_free_nondoc = int((cid0[0].numpy() == -1).sum())  # preamble + question + answer

        gold_row = torch.zeros(n_docs, dtype=torch.bool)
        gold_row[[d for d in gold_docs if 0 <= d < n_docs]] = True
        off = 1
        if true_ids and gold_docs and len(true_ids) == len(gold_docs):
            diffs = {t - g for t, g in zip(sorted(true_ids), sorted(gold_docs))}
            if len(diffs) == 1:
                off = diffs.pop()

        # --- per-document cent_cmean vectors, topical clusters, centroid cosines ---------------
        dv = clus = cos_c = cl_of = None
        present = np.zeros(n_docs, dtype=bool)
        if need_vecs:
            dv, present = doc_cent_vectors(model, x, cid_h, n_docs, stop_mask_t)
            clus, n_cl, gap = avg_linkage_clusters(dv, present)
            cos_c = robust_centroid_cos(dv, present)
            cl_of = {d: ci for ci, grp in enumerate(clus) for d in grp}
            gsz = sorted(len(g) for g in clus)
            gold_cl = sorted({cl_of[d] for d in gold_docs if d in cl_of})
            corpus["n_clusters"].append(float(n_cl))
            corpus["smallest"].append(float(gsz[0]))
            corpus["largest"].append(float(gsz[-1]))
            corpus["gold_in_smallest"].append(
                1.0 if (len(gold_cl) == 1 and gold_cl[0] == 0) else 0.0)
            corpus["gold_cat_purity"].append(
                float(np.mean([1.0 if (d in cl_of and cl_of[d] in gold_cl) else 0.0
                               for d in gold_docs])) if gold_docs else float("nan"))
            gm = [cos_c[d] for d in gold_docs if d < n_docs and present[d]]
            om = [cos_c[d] for d in range(n_docs) if present[d] and not bool(gold_row[d])]
            if gm and om:
                corpus["cos_gold"].append(float(np.mean(gm)))
                corpus["cos_other"].append(float(np.mean(om)))
                corpus["hn_rate"].append(1.0 if min(om) < max(gm) else 0.0)
                ids_p = [d for d in range(n_docs) if present[d]]
                low3 = set(np.asarray(ids_p)[np.argsort(cos_c[ids_p])][:len(gold_docs)].tolist())
                corpus["oracle_cosR"].append(
                    float(len(low3 & set(gold_docs)) / max(1, len(gold_docs))))

        do_gen = ri < a.gen_rows
        for cond in conds:
            name = cond["name"]
            t_cfg = time.time()
            _CB["key"], _CB["sig"], _CB["out"] = (ri, name), None, None

            # ---- build this condition's chunk-id override and (for _swap) the modified row ----
            sel_by_doc = None
            keep_real = torch.ones(n_docs, dtype=torch.bool)
            rule_rec, n_cats_full, gold_only_full = float("nan"), float("nan"), float("nan")
            if cond["mode"] == "full":
                _OV["cid"] = None
                x_use = x
            else:
                # 1. which documents are kept WHOLE
                keep_real = torch.zeros(n_docs, dtype=torch.bool)
                if cond["keep"] == "gold":
                    keep_real = gold_row.clone()
                elif cond["keep"] == "gpr":  # gold + a random 1/3 of the non-gold documents
                    keep_real = gold_row.clone()
                    non = [d for d in range(n_docs) if not bool(gold_row[d])]
                    rng = np.random.RandomState(a.seed + n_docs)
                    for d in rng.permutation(non)[: int(round(len(non) / 3.0))]:
                        keep_real[int(d)] = True
                elif cond["group"]:
                    ks, _ = group_keep(cond, clus, cos_c, dv, present, gold_row, n_docs, a.seed)
                    for d in ks:
                        keep_real[int(d)] = True
                if gold_docs:
                    rule_rec = float(np.mean([1.0 if (d < n_docs and bool(keep_real[d])) else 0.0
                                              for d in gold_docs]))
                if clus is not None:
                    comp = [float(np.mean([1.0 if bool(keep_real[d]) else 0.0 for d in g]))
                            for g in clus]
                    n_cats_full = float(sum(1 for c in comp if c >= 1.0))
                    gcl = sorted({cl_of[d] for d in gold_docs if d in cl_of})
                    gold_only_full = float(
                        bool(gcl) and all(comp[ci] >= 1.0 for ci in gcl)
                        and n_cats_full == len(gcl))
                # 2. the real-token subset of every POOLED document
                bn = (cid_h[0] if cond["header"] else cid0[0]).numpy().copy()
                sel_by_doc = []
                for d in range(n_docs):
                    if bool(keep_real[d]) or cond["sel"] == "none":
                        sel = []
                    elif cond["sel"] == "firstlast":
                        sel = _firstlast(body_pos_by_doc[d], cond["k"])
                    else:
                        sel = select_positions(cond["sel"], cond["k"], body_pos_by_doc[d],
                                               body_ids_by_doc[d], idf, sent_end)
                    sel = [int(q) for q in sel]
                    sel_by_doc.append(sel)
                    if sel:
                        bn[sel] = -1
                # 3. one slot per pooled CLUSTER instead of per pooled document
                if cond["catslot"] and clus is not None:
                    rep = {}
                    for grp in clus:
                        pooled = [d for d in grp if not bool(keep_real[d])]
                        if len(pooled) > 1:
                            r = min(pooled)
                            for d in pooled:
                                rep[d] = r
                    if rep:
                        m = np.arange(n_docs, dtype=bn.dtype)
                        for d, r in rep.items():
                            m[d] = r
                        ctx = bn >= 0
                        bn[ctx] = m[bn[ctx].astype(np.int64)]
                _OV["cid"] = torch.as_tensor(bn)[None].cuda()
                x_use = x
                if cond["swap"]:
                    sw = sel_by_doc
                    if not any(sw):  # a group rule keeps WHOLE documents: swap their bodies
                        sw = [[int(q) for q in body_pos_by_doc[d]] if bool(keep_real[d]) else []
                              for d in range(n_docs)]
                    x_use = swap_kept(x_cpu, sw, gold_row, n_docs, a.seed).cuda()

            pst["header_stop_id"] = HEADER_STOP_ID  # always on: the patch returns _OV["cid"]
            pst["header_stop_count"] = HEADER_STOP_COUNT
            pst["header_cap"] = 32
            pst["slot_mode"] = cond["slot"]

            # ---- forward -------------------------------------------------------------------
            keep_mask = None
            if cond["mode"] == "full":
                model.eval()
                model._pooled_keep_holder = None
                lg = model(x, logits_to_keep=pred_pos[None])[0].float()
                comp = 1.0
                tok_doc = float(T - n_free_nondoc) / max(1, n_docs)
                body_doc = float("nan")
                x_in, ans_start_in, base_pos = x, ans_start, None
            elif cond["mode"] == "pool":
                keep_mask = keep_real.clone()
                model.train()
                model._pooled_keep_holder = PooledDocKeepHolder(keep_docs=keep_mask[None].clone())
                cb = model._compact_pooled_soft_tokens(x_use, None, -100)[0]
                posmap = {int(p): c for c, p in enumerate(cb.position_ids[0].tolist())}
                cols = torch.tensor([posmap[int(p)] for p in pred_pos.tolist()], device="cuda")
                assert int(cols.max()) < cb.input_ids.shape[1], "compaction/column mismatch"
                lg = model(x_use, logits_to_keep=cols[None])[0].float()
                comp = cb.input_ids.shape[1] / T
                body_doc = float(np.mean([len(s) for s in sel_by_doc])) if sel_by_doc else 0.0
                tok_doc = float(cb.input_ids.shape[1] - n_free_nondoc) / max(1, n_docs)
                x_in, ans_start_in, base_pos = x_use, ans_start, None
            else:  # drop: the same real-token subset with NO slot, at ORIGINAL positions
                ovn = _OV["cid"][0]
                kr = keep_real.to(ovn.device)
                kmask = (ovn == -1) | ((ovn >= 0) & kr[ovn.clamp(min=0).long()])
                newidx = torch.cumsum(kmask.to(torch.long), 0) - 1
                x_in = x_use[:, kmask]
                base_pos = kmask.nonzero(as_tuple=True)[0]
                cols = newidx[pred_pos]
                model.eval()
                model._pooled_keep_holder = None
                lg = model(x_in, position_ids=base_pos[None], logits_to_keep=cols[None])[0].float()
                comp = x_in.shape[1] / T
                body_doc = float(np.mean([len(s) for s in sel_by_doc])) if sel_by_doc else 0.0
                tok_doc = float(x_in.shape[1] - n_free_nondoc) / max(1, n_docs)
                ans_start_in = int(newidx[ans_start])
                keep_mask = keep_real.clone()

            r = acc[name]
            r["ce"].append(float(F.cross_entropy(lg, targets)))
            if digit_sel.numel():
                r["ce_digit"].append(float(F.cross_entropy(lg[digit_sel], targets[digit_sel])))
            r["compaction"].append(comp)
            r["tok_per_doc"].append(tok_doc)
            r["body_per_doc"].append(body_doc)
            r["n_docs"].append(float(n_docs))
            r["kept_docs"].append(float(int(keep_real.sum())) if cond["mode"] != "full" else float(n_docs))
            if rule_rec == rule_rec:
                r["rule_recall"].append(rule_rec)
            if n_cats_full == n_cats_full:
                r["n_cats_full"].append(n_cats_full)
                r["gold_only_full"].append(gold_only_full)

            if keep_mask is None:
                pooled_gold, real_gold = [], list(gold_docs)
            else:
                pooled_gold = [d for d in gold_docs if d < n_docs and not bool(keep_mask[d])]
                real_gold = [d for d in gold_docs if d < n_docs and bool(keep_mask[d])]

            if do_gen:
                if cond["mode"] == "full":
                    model.eval()
                    model._pooled_keep_holder = None

                    def fwd(seq):
                        return model(seq, logits_to_keep=1)[0][-1]
                    prefix = x[:, :ans_start].clone()
                elif cond["mode"] == "pool":
                    model.train()

                    def fwd(seq):
                        return model(seq, logits_to_keep=1)[0][-1]
                    prefix = x_use[:, :ans_start].clone()
                else:
                    model.eval()
                    model._pooled_keep_holder = None
                    bp = base_pos[:ans_start_in]

                    def fwd(seq, _bp=bp):
                        n_new = seq.shape[1] - _bp.numel()
                        pos = torch.cat([_bp, torch.arange(ans_start, ans_start + n_new,
                                                           device=seq.device)])
                        return model(seq, position_ids=pos[None], logits_to_keep=1)[0][-1]
                    prefix = x_in[:, :ans_start_in].clone()
                _CB["sig"], _CB["out"] = None, None
                gen = generate(fwd, prefix, a.gen_max_new, len(true_ids) or 3, tok)
                gtext = tok.decode(gen)
                gen_ids = parse_ids(gtext)
                r["gen_f1"].append(set_f1(gen_ids, true_ids))
                r["gen_em"].append(float(set(gen_ids) == set(true_ids) and len(gen_ids) == len(true_ids)))
                pred_docs = [i - off for i in gen_ids]
                r["hit_gold_pooled"] += [1.0 if d in pred_docs else 0.0 for d in pooled_gold]
                r["hit_gold_real"] += [1.0 if d in pred_docs else 0.0 for d in real_gold]
                if len(dumps) < a.dump_rows and name in ("first16", "first32", "cc00"):
                    dumps.append({"row": ri, "cond": name, "true_ids": true_ids,
                                  "gen_ids": gen_ids, "gen": gtext})
            r["sec"].append(time.time() - t_cfg)
            model.eval()
        _OV["cid"] = None

        if ri + 1 in (1, 2, 5, 10) or (ri + 1) % 25 == 0:
            el = sum(sum(acc[c["name"]]["sec"]) for c in conds)
            eta = el / (ri + 1) * (len(rows) - ri - 1) / 60.0
            log(f"row {ri + 1}/{len(rows)}  full CE {acc['full']['ce'][-1]:.3f}  ETA {eta:.0f} min")
            table(acc, conds)

    table(acc, conds)
    for d in dumps:
        print(f"  dump row {d['row']:3d} {d['cond']:10} true={d['true_ids']} pred={d['gen_ids']} {d['gen']!r}",
              flush=True)

    print("\n=== CORPUS STRUCTURE (per-row means over the cent_cmean document vectors) ===",
          flush=True)
    for k, v in corpus.items():
        if v:
            mm, se = mean_se(v)
            print(f"  {k:18} {mm:8.3f} +- {se:.3f}   (n={len(v)})", flush=True)
    print("  gold_in_smallest = the gold documents are exactly the SMALLEST cluster; "
          "hn_rate = some non-gold document sits farther from the centroid than a gold one;\n"
          "  oracle_cosR = recall of a 'the k lowest-cosine documents are the outliers' rule.",
          flush=True)
    print("\n=== CATEGORY COMPLETENESS IN THE KEPT SET (the gold-forcing signature) ===", flush=True)
    print(f"{'condition':14} {'rule_R':>7} {'n_cats_full':>12} {'gold_only_full':>15}", flush=True)
    for c in conds:
        r = acc[c["name"]]
        if r["n_cats_full"]:
            print(f"{c['name']:14} {np.mean(r['rule_recall']) if r['rule_recall'] else float('nan'):7.3f} "
                  f"{np.mean(r['n_cats_full']):12.2f} {np.mean(r['gold_only_full']):15.3f}", flush=True)
    print("gold_only_full = the gold document's category is the ONLY fully-real category in the "
          "row (the shortcut signature).", flush=True)

    summ = {c["name"]: summarize(acc[c["name"]], acc["full"]) for c in conds}
    summ["_corpus"] = {k: mean_se(v)[0] for k, v in corpus.items() if v}
    verdict(summ)
    out = {
        "task": "outlier", "rung": rung, "eval_size": len(acc["full"]["ce"]),
        "rows_loaded": len(rows), "gen_rows": min(a.gen_rows, len(rows)),
        "ckpt": a.resolved_ckpt, "ckpt_name": a.ckpt_name, "seed": a.seed,
        "header_stop_id": HEADER_STOP_ID, "fingerprint_misses": n_miss,
        "conditions": {k: v for k, v in summ.items() if k != "_corpus"},
        "corpus_structure": summ.get("_corpus", {}),
        "construction": {c["name"]: {k: c[k] for k in ("sel", "k", "mode", "slot", "header", "keep", "swap")}
                         for c in conds},
        "per_row": {c["name"]: {k: v for k, v in acc[c["name"]].items() if k != "gen"} for c in conds},
        "dumps": dumps,
    }
    sfx = f"_{a.tag}" if a.tag else ""
    local = a.out if a.out.endswith(".json") else f"{a.out}/realtoken_{rung}.json"
    if len(a.rungs.split(",")) > 1 and f"_{rung}" not in local:
        local = local[:-5] + f"_{rung}.json"
    weka = None if a.weka_out in ("", "none") else \
        f"{a.weka_out}/realtoken_{a.ckpt_name}_{rung}{sfx}.json"
    for path in [local] + ([weka] if weka else []):
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            json.dump(out, open(path, "w"), indent=1)
            log(f"wrote {path}")
        except Exception as e:
            log(f"could not write {path}: {e}")
    return summ


def swap_kept(x_cpu, sel_by_doc, gold_row, n_docs, seed):
    """CONTROL: exchange the KEPT REAL tokens of each gold document with those of a random
    non-gold document.  Positions, ids, headers and slot means are untouched (the kept tokens are
    FREE, so they never enter a slot mean), so only "what the model can read" changes."""
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


def summarize(r, full):
    d = {}
    for k, v in r.items():
        if k == "gen":
            continue
        m, se = mean_se(v)
        d[k], d[f"{k}_se"], d[f"{k}_count"] = m, se, len(v)
    d["dce"], d["dce_se"] = paired(r["ce"], full["ce"])
    d["dce_digit"], d["dce_digit_se"] = paired(r["ce_digit"], full["ce_digit"])
    d["dgen_f1"], d["dgen_f1_se"] = paired(r["gen_f1"], full["gen_f1"])
    d["flop_lin"] = d["compaction"]
    return d


def table(acc, conds):
    print(f"{'condition':14} {'CE':>7} {'dCE':>8} {'CEdig':>7} {'dCEdig':>8} {'genF1':>6} "
          f"{'dF1':>7} {'R@gp':>11} {'R@gr':>11} {'rule_R':>7} {'keptD':>6} {'tok/doc':>8} "
          f"{'compact':>8} {'s/row':>6}", flush=True)
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
              f"{m('hit_gold_real'):6.3f}[{len(r['hit_gold_real']):3d}] "
              f"{m('rule_recall'):7.3f} {m('kept_docs'):6.1f} "
              f"{m('tok_per_doc'):8.1f} {m('compaction'):8.3f} {m('sec'):6.2f}", flush=True)


def verdict(summ):
    """The deliverable: the cheapest construction with dCE <= 1 SE and genF1 within noise of FULL."""
    print("\n=== PARITY CHECK (dCE <= 1 paired SE AND |dF1| <= 1 paired SE) ===", flush=True)
    print(f"{'condition':13} {'dCE':>8} {'SE':>6} {'dCEdig':>8} {'SE':>6} {'dF1':>7} {'SE':>6} "
          f"{'tok/doc':>8} {'FLOPfrac':>8}  verdict", flush=True)
    rows = []
    for name, s in summ.items():
        if name in ("full", "_corpus"):
            continue
        ok_ce = s["dce"] <= s["dce_se"] if s["dce_se"] == s["dce_se"] else False
        ok_f1 = abs(s["dgen_f1"]) <= s["dgen_f1_se"] if s["dgen_f1_se"] == s["dgen_f1_se"] else False
        v = "PARITY" if (ok_ce and ok_f1) else ("ce-ok" if ok_ce else ("f1-ok" if ok_f1 else ""))
        print(f"{name:13} {s['dce']:+8.3f} {s['dce_se']:6.3f} {s['dce_digit']:+8.3f} "
              f"{s['dce_digit_se']:6.3f} {s['dgen_f1']:+7.3f} {s['dgen_f1_se']:6.3f} "
              f"{s['tok_per_doc']:8.1f} {s['flop_lin']:8.3f}  {v}", flush=True)
        if v == "PARITY":
            rows.append((s["tok_per_doc"], name))
    if rows:
        rows.sort()
        print(f"CHEAPEST PARITY: {rows[0][1]} at {rows[0][0]:.1f} real tokens/doc", flush=True)
    else:
        print("NO construction reaches parity at this rung -- read the dCE-vs-tokens curve above.",
              flush=True)


if __name__ == "__main__":
    main()
