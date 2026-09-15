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
def C(name, sel, k, mode="pool", slot="cent_cmean", header=True, keep="none", swap=False):
    """``sel`` in {none, first, firstlast, idf, idfspan, sent1, all}; ``mode`` in {full, pool, drop};
    ``keep`` in {none (every doc pooled), gold (gold docs real), all (no pooling)}."""
    return dict(name=name, sel=sel, k=k, mode=mode, slot=slot, header=header, keep=keep, swap=swap)


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
    # swap controls (the kept REAL tokens of gold docs exchanged with random non-gold docs')
    for base in ("first8", "first16", "first32", "first64", "idfspan16", "sent1"):
        b = next(c for c in cs if c["name"] == base)
        cs.append(dict(b, name=f"{base}_swap", swap=True))
    return cs


PRESETS = {
    "all": None,  # everything
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
    ap.add_argument("--jsonl", default=None)
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
    idf, n_idf_tok = build_idf(idf_shard, vocab, n_rows=a.idf_rows)
    sent_end = build_sent_end(tok, vocab)
    log(f"idf from {n_idf_tok} tokens of {idf_shard} in {time.time() - t0:.0f}s; "
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
    if {c["slot"] for c in conds} - {"mean"}:
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
def run_rung(a, rung, conds, model, pst, tok, idf, sent_end):
    shard = f"{a.work}/outlier_{rung}"
    P.convert("outlier", a.jsonl or RUNGS[rung], a.rows, shard)
    rows, masks = P.load_rows(shard, a.rows)
    log(f"=== rung {rung}: {len(rows)} rows; lengths {[len(r) for r in rows[:6]]}")
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
            "n_docs", "sec")
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

        do_gen = ri < a.gen_rows
        for cond in conds:
            name = cond["name"]
            t_cfg = time.time()
            _CB["key"], _CB["sig"], _CB["out"] = (ri, name), None, None

            # ---- build this condition's chunk-id override and (for _swap) the modified row ----
            sel_by_doc = None
            if cond["mode"] == "full":
                _OV["cid"] = None
                x_use = x
            else:
                base = cid_h.clone() if cond["header"] else cid0.clone()
                sel_by_doc = []
                for d in range(n_docs):
                    if cond["keep"] == "gold" or cond["sel"] == "none":
                        s = []
                    elif cond["sel"] == "firstlast":
                        s = _firstlast(body_pos_by_doc[d], cond["k"])
                    else:
                        s = select_positions(cond["sel"], cond["k"], body_pos_by_doc[d],
                                             body_ids_by_doc[d], idf, sent_end)
                    sel_by_doc.append([int(p) for p in s])
                    if s:
                        base[0, torch.tensor(list(s), dtype=torch.long)] = -1
                _OV["cid"] = base.cuda()
                x_use = x
                if cond["swap"]:
                    x_use = swap_kept(x_cpu, sel_by_doc, gold_row, n_docs, a.seed).cuda()

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
                keep_mask = gold_row.clone() if cond["keep"] == "gold" else torch.zeros(n_docs, dtype=torch.bool)
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
                kmask = (_OV["cid"][0] == -1)
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
                keep_mask = torch.zeros(n_docs, dtype=torch.bool)

            r = acc[name]
            r["ce"].append(float(F.cross_entropy(lg, targets)))
            if digit_sel.numel():
                r["ce_digit"].append(float(F.cross_entropy(lg[digit_sel], targets[digit_sel])))
            r["compaction"].append(comp)
            r["tok_per_doc"].append(tok_doc)
            r["body_per_doc"].append(body_doc)
            r["n_docs"].append(float(n_docs))

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

    summ = {c["name"]: summarize(acc[c["name"]], acc["full"]) for c in conds}
    verdict(summ)
    out = {
        "task": "outlier", "rung": rung, "eval_size": len(acc["full"]["ce"]),
        "rows_loaded": len(rows), "gen_rows": min(a.gen_rows, len(rows)),
        "ckpt": a.resolved_ckpt, "ckpt_name": a.ckpt_name, "seed": a.seed,
        "header_stop_id": HEADER_STOP_ID, "fingerprint_misses": n_miss,
        "conditions": summ,
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
    print(f"{'condition':13} {'CE':>7} {'dCE':>8} {'CEdig':>7} {'dCEdig':>8} {'genF1':>6} "
          f"{'dF1':>7} {'R@gp':>11} {'R@gr':>11} {'tok/doc':>8} {'body':>6} {'compact':>8} {'s/row':>6}",
          flush=True)
    fu = acc.get("full")
    for c in conds:
        r = acc[c["name"]]
        if not r["ce"]:
            continue
        m = lambda k: (np.mean(r[k]) if r[k] else float("nan"))  # noqa: E731
        dce, _ = paired(r["ce"], fu["ce"])
        dcd, _ = paired(r["ce_digit"], fu["ce_digit"])
        df1, _ = paired(r["gen_f1"], fu["gen_f1"])
        print(f"{c['name']:13} {m('ce'):7.3f} {dce:+8.3f} {m('ce_digit'):7.3f} {dcd:+8.3f} "
              f"{m('gen_f1'):6.3f} {df1:+7.3f} "
              f"{m('hit_gold_pooled'):6.3f}[{len(r['hit_gold_pooled']):3d}] "
              f"{m('hit_gold_real'):6.3f}[{len(r['hit_gold_real']):3d}] "
              f"{m('tok_per_doc'):8.1f} {m('body_per_doc'):6.1f} {m('compaction'):8.3f} "
              f"{m('sec'):6.2f}", flush=True)


def verdict(summ):
    """The deliverable: the cheapest construction with dCE <= 1 SE and genF1 within noise of FULL."""
    print("\n=== PARITY CHECK (dCE <= 1 paired SE AND |dF1| <= 1 paired SE) ===", flush=True)
    print(f"{'condition':13} {'dCE':>8} {'SE':>6} {'dCEdig':>8} {'SE':>6} {'dF1':>7} {'SE':>6} "
          f"{'tok/doc':>8} {'FLOPfrac':>8}  verdict", flush=True)
    rows = []
    for name, s in summ.items():
        if name == "full":
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
