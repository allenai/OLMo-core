"""
Is a pooled outlier document READABLE from its soft-token slot?  (eval-only, 2026-09-14)

Background.  The gold-blind soft-token outlier arms split in two:

  * ``kvgb50`` (gold-blind, keep 1/2, NO header) reaches dense parity,
  * ``xhdr00/17/33`` (gold-blind + ``Document [N]:`` headers real) sit EXACTLY on the
    ``k/n`` uniform-guess floor at every rung and every keep.

``debug/ds64/xhdr_collapse_diagnosis.md`` shows the collapse is not a code bug and argues it is a
*shortcut*: header-real makes every document id copyable, so "emit three ids from the visible list"
is already loss-optimal and the gradient never has to read the slots.  That argument leaves one
question open, and it decides what to do next:

  (A) the slot DOES carry enough signal for a dense model to name the odd document out -- the
      training collapse is an optimisation failure and the fix is training-side (gold-forcing,
      id blinding, renumbering, curriculum); or
  (B) the slot does NOT carry that signal -- the header-real target is genuinely unjustifiable and
      the arm family is a hard limit, not a shortcut.

This probe answers it WITHOUT training anything: a DENSE-trained outlier checkpoint is held fixed
and fed the same slot constructions the training arms use.  If the frozen dense model can still
name a *pooled* outlier from its mean-embedding slot, the information is there (A).

Constructions (``--conditions``)::

  full        plain full attention over the real tokens (reference)
  goldonly    gold docs real, every other doc pooled, no header  (the 2026-09-08 "parity" point)
  gb50        gold-blind keep 1/2, no header                     (= kvgb50)
  gb50h       gold-blind keep 1/2 + header real                  (= the would-be xhdr50 / xh2k50)
  gb17h       gold-blind keep 1/6 + header real                  (= xhdr17)
  gb00h       keep 0 + header real                               (= xhdr00)
  gb00        keep 0, no header (pure slots)
  gb00h_swap  CONTROL: gb00h with the gold docs' slot vectors swapped with random non-gold slots'.
              If F1 does not drop, the model is not reading slot CONTENT at all.
  gb50h_swap  the same control on gb50h, where half the documents are still real: it separates
              "found the gold because its BODY was real" from "found it from the slot".

Everything gold-blind uses the SAME keep draw the trainer uses (``resolve_keep_docs`` with
``holder=None``), and the header path is the trainer's own ``mark_doc_headers_free``
(``--st-header-stop-id 5491 --st-header-stop-count 1``), reached by setting
``header_stop_id`` on the model's pooled-soft-token config -- not the probe's ``--prefix-real``
patch -- so the construction is bit-for-bit the training one.

Metrics, at the answer positions (aligned by ORIGINAL position): mean CE of the true answer tokens,
top-1 agreement with FULL's argmax, KL(FULL || SOFT), teacher-forced exact match, and -- from FREE
greedy generation on a subsample -- exact set match and set-F1 over the k document ids.  Every
per-document metric is SPLIT by whether that gold document was POOLED or REAL in the row, which is
the decisive number: ``recall_gold_pooled`` in ``gb00h`` / ``gb50h`` is "can the frozen dense model
pick out an outlier it can only see as one mean-embedding vector?".

    python debug/pooled_kv/outlier_probe/outlier_slot_probe.py --rung 2k --rows 200 --out /results/x.json

----------------------------------------------------------------------------------------------
EVAL-TIME CE PARITY OF A *TRAINED* SOFT ARM  (``--trained-parity``, 2026-09-15)
----------------------------------------------------------------------------------------------

The probe above holds a DENSE checkpoint fixed and asks what a slot can carry.  The complementary
question -- the one a ladder eval silently answers wrong -- is about a checkpoint that WAS trained
on slots: when we score it on the ladder we feed it FULL REAL TEXT, an input distribution it may
never have seen.  ``cc00`` is the case that forced this: trained with keep 0 (every document body
pooled), it scores f1 0.80 at the 2k rung but 0.17 at 8k and 0.05 at 16k under full attention.
That is either "the slot readout does not scale with document count" or "the model reads slots
fine and the ladder is measuring distribution shift", and the two call for opposite fixes.

``--trained-parity`` is the one-command check, and it should be run on EVERY new soft checkpoint
before its ladder number is believed.  It scores the SAME checkpoint twice per rung -- once on
FULL real text (what the ladder feeds it) and once on its own TRAINING construction -- and prints
CE, CE on the answer's DIGIT tokens, and free-generation set-F1 both ways with the deltas::

    python debug/pooled_kv/outlier_probe/outlier_slot_probe.py --trained-parity \
      --ckpt-name ds64-outlier-cc00-b128f3-u128M --rungs 8k,32k --rows 240 --gen-rows 32

The construction comes from the run name's arm field (``cc00``) via ``ARMS``.  For an arm that is
not registered, pass it explicitly as the trainer's own flags with commas for spaces (so the
string survives a launcher that splits argv on whitespace)::

      --construction --st-gold-blind,--st-keep-prob,0.0,--st-header-stop-id,5491,\
                     --st-header-stop-count,1,--st-slot-mode,cent_cmean

Notes that make the check faithful rather than approximate:

* the ``cmean``/``cent_cmean`` stop set is rebuilt from token frequencies over the head of the
  arm's own TRAINING shard, exactly as ``train_ctc_suite.build_slot_stop_set`` does -- not from
  the eval rows;
* ``--trained-parity`` implies ``--no-reset-projector``: a trained checkpoint keeps its own
  ``pooled_projector`` (with ``detach_soft_kv`` it is the identity, and the log prints
  ``max|w_out|`` so you can see that);
* all rungs are scored in ONE process, so the 4B checkpoint is loaded once.

Reading it: parity is ``dCEdig ~ 0`` AND ``dF1 ~ 0``.  A large ``dCEdig`` with the soft side
BETTER means the ladder is measuring distribution shift (fix = exposure: mix real-body rows in,
or anneal keep up at the end).  Soft side also weak means the readout itself does not scale (fix
= the slot or the readout).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import eval_side_slot_probe as P  # noqa: E402  (shared harness: convert/load_rows/find_ckpt/CKPT/IDS)

from olmo_core.distributed.checkpoint import load_model_and_optim_state  # noqa: E402
from olmo_core.nn.attention.chunked_mask import (  # noqa: E402
    build_chunk_ids_from_tokens,
    mark_doc_headers_free,
)
from olmo_core.nn.attention.gold_grad_mask import content_fingerprint_from_row  # noqa: E402
from olmo_core.nn.attention.pooled_doc_kv import (  # noqa: E402
    PooledDocKeepHolder,
    resolve_keep_docs,
)
from olmo_core.nn.lm_head import LMLossImplementation  # noqa: E402

W = P.W
IDS = P.IDS
HEADER_STOP_ID = 5491  # Qwen3.5 ']:' -- ends `\n\nDocument [N]:` (stop id 25 does NOT occur in outlier)
HEADER_STOP_COUNT = 1

# ds64 dense (full-attention-trained) outlier arms; `find_ckpt` takes the run dir or model_and_optim.
CKPTS = {
    "ds64-outlier-dense-u64M": f"{W}/ctc_suite/ckpts/ds64-outlier-dense-u64M/model_and_optim",
    "ds64-outlier-dense-u128M": f"{W}/ctc_suite/ckpts/ds64-outlier-dense-u128M/model_and_optim",
    "ds64-outlier-dense-u32M": f"{W}/ctc_suite/ckpts/ds64-outlier-dense-u32M/model_and_optim",
    "ds64-outlier-dense-u16M": f"{W}/ctc_suite/ckpts/ds64-outlier-dense-u16M/model_and_optim",
    # the checkpoint the 2026-09-08 "gold-only parity" number was measured on
    "lmx-full-mixs160M-4b": f"{W}/*/ckpts/lmx-full-mixs160M-4b-2026",
}
# Any other name is resolved as a ds64 run name under ctc_suite/ckpts/ (trained SOFT arms live
# there too -- ds64-outlier-cc00-b128f3-u128M, ...).  ``find_ckpt`` accepts either the run dir or
# its model_and_optim subdir, and also finds a mid-run step*/model_and_optim.
CKPT_ROOT = f"{W}/ctc_suite/ckpts"


def ckpt_path_for(name):
    return CKPTS.get(name) or f"{CKPT_ROOT}/{name}"


def arm_for_ckpt_name(name):
    """``ds64-outlier-cc00-b128f3-u128M`` -> ``cc00``.  Returns None if no field matches ARMS."""
    for f in name.split("-"):
        if f in ARMS:
            return f
    return None


# The ds64 training shards -- the stop set for ``cmean``/``cent_cmean`` is built from token
# FREQUENCIES over the head of the training shard, exactly as train_ctc_suite.build_slot_stop_set
# does, so the eval-side slot is bit-for-bit the one the arm trained with.
DS64_SHARDS = f"{W}/ds64/shards"
# the rung files the ds64 outlier arms are evaluated on (EVAL_JSONL in the shared harness has no 2k)
RUNGS = {r: f"{W}/outlier_lengthmix/eval_rungs/outlier/rung_{n}.jsonl"
         for r, n in (("2k", 2048), ("8k", 8192), ("16k", 16384), ("32k", 32768))}

# A condition is (name, keep-spec, header?, slot-mode, body-truncation).
#   keep-spec : None = FULL real text | "gold" = gold docs real | float p = gold-blind Bernoulli
#   header    : keep every document's `\n\nDocument [N]:` header real (the trainer's
#               ``--st-header-stop-id 5491 --st-header-stop-count 1`` path)
#   slot      : "mean" (the historical plain mean input embedding) | "cmean" | "cent_cmean" --
#               the trainer's own ``--st-slot-mode`` (olmo_core.nn.pooled_soft_token.apply_slot_mode)
#   trunc     : only meaningful with keep=None -- FULL attention over real text whose document
#               BODIES are cut to their first N tokens (headers kept). A cheap probe of
#               "too many real tokens" vs "real text per se".
# ``_swap`` in a name turns on the slot-swap control.
def C(name, keep, header, slot="mean", trunc=None):
    return {"name": name, "keep": keep, "header": header, "slot": slot, "trunc": trunc}


CONDITIONS = [
    C("full", None, False),
    # --- plain-mean slot (every arm trained before 2026-09-15) ---------------------------------
    C("goldonly", "gold", False),
    C("gb50", 0.5, False),
    C("gb50h", 0.5, True),
    C("gb50h_swap", 0.5, True),
    C("gb17h", 1.0 / 6.0, True),
    C("gb00h", 0.0, True),
    C("gb00", 0.0, False),
    C("gb00h_swap", 0.0, True),
    # --- content-only / row-centred slot (--st-slot-mode cent_cmean, the cc00 family) ----------
    C("cc00", 0.0, True, "cent_cmean"),  # == the cc00/occ00 TRAINING construction
    C("cc00_swap", 0.0, True, "cent_cmean"),
    C("cc17", 1.0 / 6.0, True, "cent_cmean"),
    C("cc50", 0.5, True, "cent_cmean"),
    C("goldonly_cc", "gold", False, "cent_cmean"),
    C("ccgold", "gold", True, "cent_cmean"),  # ORACLE: gold bodies real, everything else a slot
    # --- real text, less of it ------------------------------------------------------------------
    C("trunc8", None, False, "mean", 8),
    C("trunc32", None, False, "mean", 32),
]

# Trainer ``--extra-args`` of the registered ds64 soft arms, so ``--arm <name>`` (or the arm
# segment of a run name like ``ds64-outlier-cc00-b128f3-u128M``) reproduces an arm's EXACT
# training construction eval-side.  Keep in sync with ``debug/ds64/launch_ds64.py:ARM_EXTRA``.
ARMS = {
    #  name  -> (keep_prob, header_stop_id, header_stop_count, slot_mode)
    "cc00": (0.0, 5491, 1, "cent_cmean"),
    "cc03": (0.0278, 5491, 1, "cent_cmean"),
    "cc08": (0.0833, 5491, 1, "cent_cmean"),
    "cc17": (0.1667, 5491, 1, "cent_cmean"),
    "cc50": (0.5, 5491, 1, "cent_cmean"),
    "xhdr00": (0.0, 5491, 1, "mean"),
    "xhdr17": (0.1667, 5491, 1, "mean"),
    "xh2k50": (0.5, 5491, 1, "mean"),
    "kvgb50": (0.5, None, 1, "mean"),
    "occ00": (0.0, 25, 3, "cent_cmean"),
    "occ08": (0.0833, 25, 3, "cent_cmean"),
    "ohdr08": (0.0833, 25, 3, "mean"),
}

_SWAP = {"on": False, "gold": None, "seed": 0}
# The probe compacts once itself (to map answer positions -> compacted columns) and the model's
# forward compacts again. Those two MUST agree: if they ever disagree about the compacted length,
# the column indices go out of bounds and torch reports it as an async device-side assert in
# whatever kernel runs next ("vectorized gather kernel index out of bounds"), far from the cause.
# So memoize the compaction on the exact input tensor + config so the forward reuses the probe's.
_CBCACHE = {"key": None, "sig": None, "out": None}


def log(m):
    print(f"[outlier-probe] {m}", flush=True)


def install_swap_patch(model):
    """CONTROL: exchange the slot feature vectors of the GOLD documents with those of an equal
    number of random non-gold pooled documents, leaving every position / id / length untouched.
    A construction whose F1 survives this is not reading slot content."""
    cls = type(model)
    orig = cls._compact_pooled_soft_tokens

    def patched(self, input_ids, labels, ignore_index):
        sig = (_CBCACHE["key"], int(input_ids.data_ptr()), int(input_ids.shape[1]))
        if _CBCACHE["out"] is not None and _CBCACHE["sig"] == sig:
            return _CBCACHE["out"]
        out = _apply_swap(orig(self, input_ids, labels, ignore_index))
        _CBCACHE["sig"], _CBCACHE["out"] = sig, out
        return out

    def _apply_swap(out):
        if out is None or not _SWAP["on"]:
            return out
        cb, inj, ovr = out
        if inj is None:
            return out
        rows, cols, feats = inj
        docs = cb.soft_docs
        assert docs.numel() == rows.numel(), "shadows are on; the swap control assumes none"
        gold = _SWAP["gold"]
        dl = docs.tolist()
        gi = [i for i, d in enumerate(dl) if bool(gold[d])]
        ni = [i for i, d in enumerate(dl) if not bool(gold[d])]
        if not gi or len(ni) < len(gi):
            return out
        g = torch.Generator().manual_seed(_SWAP["seed"] + len(dl))
        pick = [ni[j] for j in torch.randperm(len(ni), generator=g)[: len(gi)].tolist()]
        feats = feats.clone()
        a = feats[gi].clone()
        feats[gi] = feats[pick]
        feats[pick] = a
        return cb, (rows, cols, feats), ovr

    cls._compact_pooled_soft_tokens = patched


def parse_construction(spec):
    """Parse an arm's TRAINING construction from the trainer's own ``--extra-args`` flags, with
    spaces written as commas so the string survives a launcher that splits argv on whitespace::

        --st-gold-blind,--st-keep-prob,0.0,--st-header-stop-id,5491,--st-header-stop-count,1,--st-slot-mode,cent_cmean

    :returns: ``(keep_prob, header_stop_id, header_stop_count, slot_mode)``.
    :raises SystemExit: If a keep policy this probe cannot reproduce is requested.
    """
    toks = [t for t in re.split(r"[,\s]+", spec) if t]
    kv = {}
    i = 0
    while i < len(toks):
        t = toks[i]
        if t.startswith("--"):
            if i + 1 < len(toks) and not toks[i + 1].startswith("--"):
                kv[t] = toks[i + 1]
                i += 2
                continue
            kv[t] = "1"
        i += 1
    if "--st-keep-frac" in kv:
        raise SystemExit(
            "--construction: --st-keep-frac (gold_plus_random) is not reproducible here; this "
            "probe implements the gold-blind --st-keep-prob path and the gold-only keep"
        )
    keep = float(kv.get("--st-keep-prob", 0.0))
    hid = kv.get("--st-header-stop-id")
    hcnt = int(kv.get("--st-header-stop-count", 1))
    slot = kv.get("--st-slot-mode", "mean")
    return keep, (None if hid is None else int(hid)), hcnt, slot


def parity_report(a, summaries, arm_name):
    """The deliverable of ``--trained-parity``: for each rung, the SAME checkpoint's answer loss
    under FULL real text (what the ladder eval feeds it) against its own TRAINING construction."""
    print("\n=== EVAL-TIME CE PARITY  (one checkpoint, two inputs) ===", flush=True)
    print(f"ckpt {a.ckpt_name}   arm {arm_name}   rows/rung {a.rows}", flush=True)
    hdr = (f"{'rung':>5} {'eval_size':>9} | {'CE_full':>8} {'CE_soft':>8} {'dCE':>8} | "
           f"{'CEdig_full':>10} {'CEdig_soft':>10} {'dCEdig':>8} | "
           f"{'F1_full':>8} {'F1_soft':>8} {'dF1':>7} | {'compact':>7}")
    print(hdr, flush=True)
    for rung, cs in summaries.items():
        f, g = cs.get("full"), cs.get("arm")
        if not f or not g:
            continue
        print(f"{rung:>5} {int(f['ce_count']):>9} | {f['ce']:8.3f} {g['ce']:8.3f} {g['ce'] - f['ce']:+8.3f} | "
              f"{f['ce_digit']:10.3f} {g['ce_digit']:10.3f} {g['ce_digit'] - f['ce_digit']:+8.3f} | "
              f"{f['gen_f1']:8.3f} {g['gen_f1']:8.3f} {g['gen_f1'] - f['gen_f1']:+7.3f} | "
              f"{g['compaction']:7.3f}", flush=True)
    print("PARITY = dCEdig ~ 0 AND dF1 ~ 0: the soft construction costs the checkpoint nothing at "
          "this rung.\nA LARGE dCEdig with a small |dF1| on a WEAK full-text F1 is not parity -- "
          "it is two ways of being wrong.", flush=True)


def chunk_ids_for(x, header: bool):
    """The chunk ids the model's own compaction will see (post header-freeing)."""
    cid = build_chunk_ids_from_tokens(
        x, doc_start_id=IDS.doc_start, doc_end_id=IDS.doc_end, eos_id=IDS.eos, mode="chunked"
    )
    if header:
        cid = mark_doc_headers_free(
            cid, x, doc_start_id=IDS.doc_start, doc_end_id=IDS.doc_end,
            stop_id=HEADER_STOP_ID, stop_count=HEADER_STOP_COUNT, cap=32,
        )
    return cid


def build_stop_ids(shard_dir, tok, topk=100, n_rows=512, seq_len=65536):
    """Reproduce ``train_ctc_suite.build_slot_stop_set``: the ``--st-slot-mode`` stop set, taken
    from token frequencies over the head of the TRAINING shard (top-``topk`` ids + markers/pad +
    every id whose decoded piece has no alphanumeric character).

    Building it from the training shard rather than the eval rows is what makes the eval-side
    ``cent_cmean`` slot bit-for-bit the one the arm trained with.  ``token_ids_part_*.npy`` is a
    RAW HEADERLESS array despite the extension -- ``np.load`` on one dies.

    :returns: ``(sorted stop ids, decoded sample, n_tokens_scanned)``.
    """
    import glob as _glob

    from olmo_core.nn.pooled_soft_token import build_slot_stop_ids

    parts = sorted(_glob.glob(f"{shard_dir}/token_ids_part_*.npy"))
    if not parts:
        raise SystemExit(f"--slot-stop-shard {shard_dir} has no token_ids_part_*.npy")
    meta = json.load(open(f"{shard_dir}/metadata.json"))
    dtype = np.dtype(meta.get("dtype") or "uint32")
    n_total = os.path.getsize(parts[0]) // dtype.itemsize
    arr = np.memmap(parts[0], dtype=dtype, mode="r", shape=(n_total,))
    row_len = int(meta.get("max_example_len") or seq_len)
    n_tok = min(n_total, max(1, n_rows) * max(1, row_len))
    stop, shown = build_slot_stop_ids(
        np.asarray(arr[:n_tok]),
        top_k=topk,
        extra_ids=(IDS.doc_start, IDS.doc_end, IDS.eos, IDS.landmark, IDS.pad),
        decode=lambda t: tok.decode([t]),
    )
    return stop, shown, int(n_tok)


def truncate_bodies(x_cpu, n_keep, header_stop_id):
    """FULL real text with every document BODY cut to its first ``n_keep`` tokens.

    Each document keeps its ``<|doc_start|>`` marker, its whole ``\n\nDocument [N]:`` header
    (found with the trainer's own :func:`mark_doc_headers_free`, which turns header tokens FREE),
    the first ``n_keep`` body tokens and the document's final token (its ``<|doc_end|>``).  Free
    tokens -- preamble, question, answer -- are untouched.

    :returns: ``(kept-position bool mask over the row,)`` as a 1-D CPU bool tensor.
    """
    cid = build_chunk_ids_from_tokens(
        x_cpu, doc_start_id=IDS.doc_start, doc_end_id=IDS.doc_end, eos_id=IDS.eos, mode="chunked"
    )
    cid = mark_doc_headers_free(
        cid, x_cpu, doc_start_id=IDS.doc_start, doc_end_id=IDS.doc_end,
        stop_id=header_stop_id, stop_count=HEADER_STOP_COUNT, cap=32,
    )[0]
    n_docs = int(cid.max()) + 1
    keep = torch.ones(cid.numel(), dtype=torch.bool)
    if n_docs <= 0:
        return keep
    pos = (cid >= 0).nonzero(as_tuple=True)[0]
    d = cid[pos].to(torch.long)
    order = torch.argsort(d, stable=True)
    lens = torch.bincount(d, minlength=n_docs)
    offs = torch.cumsum(lens, 0) - lens
    rank = torch.empty_like(d)
    rank[order] = torch.arange(d.numel()) - offs[d[order]]
    keep[pos] = (rank <= n_keep) | (rank == (lens[d] - 1))
    return keep


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
def generate(model, seq, max_new, k_ids, tok):
    """Free greedy continuation of the answer region (B=1, so the compacted row has no pad and the
    last logit is the last content token). Stops at EOS, or once ``k_ids`` bracketed ids have been
    emitted and the last one is closed."""
    gen = []
    for _ in range(max_new):
        lg = model(seq, logits_to_keep=1)[0][-1]
        nxt = int(lg.argmax())
        if nxt == IDS.eos:
            break
        gen.append(nxt)
        seq = torch.cat([seq, torch.tensor([[nxt]], device=seq.device)], dim=1)
        txt = tok.decode(gen)
        if len(parse_ids(txt)) >= k_ids and txt.rstrip().endswith("]"):
            break
    return gen


@torch.no_grad()
def main():
    global HEADER_STOP_ID, HEADER_STOP_COUNT
    ap = argparse.ArgumentParser()
    ap.add_argument("--rung", default="2k", help="single rung (kept for the 2026-09-14 command line)")
    ap.add_argument("--rungs", default=None,
                    help="comma list of rungs to score in ONE process, reusing the loaded "
                         "checkpoint (e.g. 8k,32k). Overrides --rung.")
    ap.add_argument("--rows", type=int, default=200)
    ap.add_argument("--gen-rows", type=int, default=32, help="rows that also get FREE greedy generation (0 = none)")
    ap.add_argument("--gen-max-new", type=int, default=64)
    ap.add_argument("--ckpt-name", default="ds64-outlier-dense-u64M",
                    help="a key of CKPTS, or any ds64 run name under ctc_suite/ckpts/ "
                         "(e.g. ds64-outlier-cc00-b128f3-u128M)")
    ap.add_argument("--ckpt", default=None, help="explicit override path")
    ap.add_argument("--jsonl", default=None, help="override eval JSONL (local runs)")
    ap.add_argument("--shard", default=None, help="already-tokenized shard dir")
    ap.add_argument("--conditions", default=",".join(c["name"] for c in CONDITIONS[:9]),
                    help="comma list; default = the nine plain-mean constructions of the 2026-09-14 "
                         f"run. All: {[c['name'] for c in CONDITIONS]} (+ 'arm', see --arm)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tokenizer", default=None, help="local tokenizer dir (default: the shared harness's HF id)")
    ap.add_argument("--work", default="/results/outlier_probe_work")
    ap.add_argument("--out", default="/results/outlier_slot_probe.json")
    ap.add_argument("--weka-out", default=f"{W}/_eval_results/outlier_slot_probe")
    ap.add_argument("--dump-rows", type=int, default=12, help="pooled-gold rows to dump greedy id sets for")
    ap.add_argument("--tag", default="", help="suffix for the weka JSON filename (e.g. 'smoke')")
    # --- eval-time CE parity of a TRAINED soft arm ---------------------------------------------
    ap.add_argument("--arm", default="auto",
                    help=f"the arm whose TRAINING construction to reproduce as condition 'arm'. "
                         f"'auto' = read it off --ckpt-name; 'none' = off. Known: {sorted(ARMS)}")
    ap.add_argument("--trained-parity", dest="parity", action="store_true",
                    help="alias of --parity: the one-command eval-time CE-parity check for a "
                         "TRAINED soft checkpoint")
    ap.add_argument("--construction", default=None,
                    help="the arm's TRAINING construction, as the trainer's own flags with "
                         "spaces replaced by commas, e.g. "
                         "'--st-gold-blind,--st-keep-prob,0.0,--st-header-stop-id,5491,"
                         "--st-header-stop-count,1,--st-slot-mode,cent_cmean'. Overrides --arm.")
    ap.add_argument("--parity", action="store_true",
                    help="eval-time CE-parity check: run ONLY `full` (real text, what the ladder "
                         "eval does) and `arm` (the checkpoint's own training construction) and "
                         "print the CE / CE(digits) / genF1 gap between them")
    ap.add_argument("--slot-stop-shard", default=None,
                    help="training shard whose token frequencies build the --st-slot-mode stop "
                         f"set (default: {DS64_SHARDS}/outlier_u<budget of --ckpt-name>)")
    ap.add_argument("--slot-stop-topk", type=int, default=100)
    ap.add_argument("--slot-stop-rows", type=int, default=512)
    ap.add_argument("--header-stop-id", type=int, default=HEADER_STOP_ID,
                    help="token that ends a document header (outlier ']:' = 5491; oolong ':' = 25)")
    ap.add_argument("--no-reset-projector", action="store_true",
                    help="keep the CHECKPOINT's pooled_projector weights instead of re-initialising "
                         "it to the identity. Use on a TRAINED soft checkpoint.")
    a = ap.parse_args()
    HEADER_STOP_ID = int(a.header_stop_id)
    a.rungs = a.rungs or a.rung
    if a.parity and a.no_reset_projector is False:
        # a trained soft arm's projector is whatever the checkpoint holds; never overwrite it
        a.no_reset_projector = True

    if a.tokenizer:
        P.TOKENIZER = a.tokenizer
    arm_name = arm_for_ckpt_name(a.ckpt_name) if a.arm == "auto" else (None if a.arm == "none" else a.arm)
    arm_spec = None
    if a.construction:
        arm_spec = parse_construction(a.construction)
        arm_name = arm_name or "custom"
        log(f"construction from --construction: {arm_spec}")
    arm_cond = None
    if arm_spec is not None:
        kp, hid, hcnt, smode = arm_spec
    elif arm_name is not None:
        if arm_name not in ARMS:
            raise SystemExit(f"unknown --arm {arm_name!r}; known arms: {sorted(ARMS)}")
        kp, hid, hcnt, smode = ARMS[arm_name]
    if arm_name is not None:
        if hid is not None:
            HEADER_STOP_ID = int(hid)
        HEADER_STOP_COUNT = int(hcnt)
        arm_cond = C("arm", kp, hid is not None, smode)
        log(f"arm {arm_name}: keep_prob={kp} header_stop_id={hid} (count {hcnt}) slot_mode={smode} "
            f"-> condition 'arm'")
    table_all = list(CONDITIONS) + ([arm_cond] if arm_cond else [])
    if a.parity:
        if arm_cond is None:
            raise SystemExit("--parity needs an arm (--arm <name>, or a --ckpt-name carrying one)")
        want = {"full", "arm"}
    else:
        want = set(a.conditions.split(",")) | {"full"}  # FULL is the reference every metric needs
    conds = [c for c in table_all if c["name"] in want]
    missing = want - {c["name"] for c in conds}
    if missing:
        raise SystemExit(f"unknown conditions {sorted(missing)}; known: {[c['name'] for c in table_all]}")
    log(f"conditions: {[c['name'] for c in conds]}")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(P.TOKENIZER)

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
    wout = float(model.pooled_projector.w_out.weight.abs().max())
    if a.no_reset_projector:
        log(f"keeping the checkpoint's pooled_projector (max|w_out| = {wout:.3e}; "
            f"0 => it is the identity, which is what detach_soft_kv training leaves behind)")
    else:
        model.pooled_projector.reset_parameters()  # identity: soft token == mean input embedding
        log(f"reset pooled_projector to the identity (checkpoint had max|w_out| = {wout:.3e})")
    model = model.cuda().to(torch.bfloat16)
    pst = model._pooled_soft_tokens
    install_swap_patch(model)

    # --- --st-slot-mode stop set, from the TRAINING shard, exactly as the trainer builds it -----
    slot_modes = {c["slot"] for c in conds}
    if slot_modes != {"mean"}:
        shard_dir = a.slot_stop_shard
        if shard_dir is None:
            bud = a.ckpt_name.rsplit("-u", 1)[-1] if "-u" in a.ckpt_name else "16M"
            shard_dir = f"{DS64_SHARDS}/outlier_u{bud}"
        stop_ids, shown, n_scanned = build_stop_ids(
            shard_dir, tok, topk=a.slot_stop_topk, n_rows=a.slot_stop_rows)
        pst["slot_stop_ids"] = [int(t) for t in stop_ids]
        pst["slot_stop_mask"] = None
        pst["_stop_shard"] = shard_dir
        log(f"slot stop set: {len(stop_ids)} ids from {n_scanned} tokens of {shard_dir} "
            f"(top-{a.slot_stop_topk} + markers + punctuation/whitespace); most frequent dropped: "
            + " ".join(shown[:16]))
    else:
        stop_ids, shard_dir = None, None

    summaries = {}
    for rung in [r for r in a.rungs.split(",") if r]:
        summaries[rung] = run_rung(a, rung, conds, model, pst, tok, arm_name)
    if a.parity or len(summaries) > 1:
        parity_report(a, summaries, arm_name)


@torch.no_grad()
def run_rung(a, rung, conds, model, pst, tok, arm_name):
    """Score every condition on one eval rung.  Returns the per-condition summary dict."""
    shard = a.shard or f"{a.work}/outlier_{rung}"
    if a.shard is None:
        P.convert("outlier", a.jsonl or RUNGS[rung], a.rows, shard)
    rows, masks = P.load_rows(shard, a.rows)
    log(f"=== rung {rung}: {len(rows)} rows; lengths {[len(r) for r in rows[:6]]}")
    if len(rows) < 500:
        log(f"WARNING eval_size={len(rows)} (<500): binomial SE on a right/wrong metric at f1~0.5 is "
            f"{0.5 / max(1, len(rows)) ** 0.5:.3f}")

    # RoPE cache: on the compacted path rope.forward indexes an absolute-position sin/cos buffer by
    # position_ids, and it sizes that buffer from the CACHE, never from the positions it is asked
    # for -- so a compacted row (short) carrying original positions (long) silently reads past the
    # end. Free generation makes it worse: it walks positions past the row's own length. Warm every
    # RoPE module to the longest position any construction can ask for. (No RoPE scaling here, so a
    # larger buffer is the same values, just longer.)
    max_pos = max(len(r) for r in rows) + a.gen_max_new + 8
    n_warm = 0
    for mod in model.modules():
        rope = getattr(mod, "rope", None)
        if rope is not None and hasattr(rope, "warmup_cache"):
            rope.warmup_cache(max_pos, torch.device("cuda"))
            n_warm += 1
    log(f"warmed {n_warm} RoPE caches to {max_pos} positions")

    # Gold ids straight from the sidecar. NOT make_fingerprint_keep_docs_fn: with
    # n_random_frac=0.0 that helper still forces `max(1, ...)` == ONE random non-gold document
    # real, so the 2026-09-08 "gold-only" probe point was really gold + 1 random. Here `goldonly`
    # is exactly the gold documents.
    gold_table = json.load(open(f"{shard}/gold_fingerprints.json"))
    gold_table = {fp: sorted({int(i) for v in val for i in (v if isinstance(v, (list, tuple)) else [v])})
                  for fp, val in gold_table.items()}
    log(f"gold sidecar: {len(gold_table)} fingerprints")
    n_fp_miss = 0

    acc = {c["name"]: {"ce": [], "top1": [], "kl": [], "tf_em": [], "compaction": [], "sec": [],
                  "gen_em": [], "gen_f1": [], "tf_f1": [], "tf_id1": [], "gen_ids": [], "ce_digit": [],
                  "hit_gold_pooled": [], "hit_gold_real": [],
                  "pred_is_pooled": [], "base_pooled": [],
                  "rowsplit": []} for c in conds}
    dumps = []
    id_offset_votes = Counter()

    for ri, (row, rmask) in enumerate(zip(rows, masks)):
        x = torch.tensor(row[None], device="cuda")
        ans_pos = torch.tensor(np.nonzero(rmask)[0], device="cuda")
        pred_pos = ans_pos - 1
        targets = x[0, ans_pos]
        true_text = tok.decode(targets.tolist())
        true_ids = parse_ids(true_text)
        # Outlier answers can wrap the ids in a prose sentence ("Most passages are about X and the
        # outliers are [2], [3], [20]"), and that prose dominates the mean answer CE. Score the
        # DIGIT tokens separately -- those are the ones that carry the retrieval decision.
        pieces = [tok.decode([int(t)]) for t in targets.tolist()]
        digit_sel = torch.tensor([i for i, q in enumerate(pieces) if any(ch.isdigit() for ch in q)],
                                 device="cuda", dtype=torch.long)
        ans_start = int(ans_pos[0])

        fp = content_fingerprint_from_row(row.tolist(), IDS.eos)
        gold_docs = gold_table.get(fp)
        if gold_docs is None:
            n_fp_miss += 1
            log(f"row {ri}: gold fingerprint MISS -- skipped")
            continue
        n_docs_row = int(build_chunk_ids_from_tokens(
            x.cpu(), doc_start_id=IDS.doc_start, doc_end_id=IDS.doc_end, eos_id=IDS.eos,
            mode="chunked").max()) + 1
        gold_row = torch.zeros(n_docs_row, dtype=torch.bool)
        gold_row[[d for d in gold_docs if 0 <= d < n_docs_row]] = True
        # answer ids are 1-indexed document indices; derive the offset from THIS row rather than
        # assuming it (the gold-sidecar index-base bug of 2026-09-08 was exactly this assumption)
        off = 1
        if true_ids and gold_docs and len(true_ids) == len(gold_docs):
            diffs = {t - g for t, g in zip(sorted(true_ids), sorted(gold_docs))}
            if len(diffs) == 1:
                off = diffs.pop()
        id_offset_votes[off] += 1

        cid_cache, present_docs = {}, {}
        for hh in sorted({c["header"] for c in conds}):
            cc = chunk_ids_for(x.cpu(), hh)
            cid_cache[hh] = cc
            present_docs[hh] = [int(v) for v in torch.unique(cc[0]).tolist() if v >= 0]

        full_lg = None
        do_gen = ri < a.gen_rows
        for cond in conds:
            name, keep_spec, header = cond["name"], cond["keep"], cond["header"]
            t_cfg = time.time()
            _CBCACHE["key"], _CBCACHE["sig"], _CBCACHE["out"] = (ri, name), None, None
            pst["header_stop_id"] = HEADER_STOP_ID if header else None
            pst["header_stop_count"] = HEADER_STOP_COUNT
            pst["header_cap"] = 32
            pst["slot_mode"] = cond["slot"]
            _SWAP["on"] = name.endswith("_swap")
            _SWAP["gold"] = gold_row
            _SWAP["seed"] = a.seed
            x_in, ans_start_in, pred_pos_in = x, ans_start, pred_pos

            if keep_spec is None and cond["trunc"]:  # FULL attention, SHORTER real text
                kmask = truncate_bodies(x.cpu(), int(cond["trunc"]), HEADER_STOP_ID)
                newidx = torch.cumsum(kmask.to(torch.long), 0) - 1
                x_in = x[:, kmask.to(x.device)]
                pred_pos_in = newidx[pred_pos.cpu()].to(x.device)
                ans_start_in = int(newidx[ans_start])
                model.eval()
                model._pooled_keep_holder = None
                lg = model(x_in, logits_to_keep=pred_pos_in[None])[0].float()
                comp, keep_mask = x_in.shape[1] / x.shape[1], None
            elif keep_spec is None:  # FULL
                model.eval()
                model._pooled_keep_holder = None
                lg = model(x, logits_to_keep=pred_pos[None])[0].float()
                full_lg = lg
                comp, keep_mask = 1.0, None
            else:
                cid = cid_cache[header]
                n_docs = int(cid.max()) + 1
                if keep_spec == "gold":
                    keep_mask = gold_row[:n_docs].clone()
                else:  # gold-blind: the trainer's own seeded-hash draw (holder=None path)
                    keep_mask = resolve_keep_docs(cid, n_docs, holder=None,
                                                  keep_prob=float(keep_spec), keep_seed=a.seed)[0].cpu()
                model.train()
                model._pooled_keep_holder = PooledDocKeepHolder(keep_docs=keep_mask[None].clone())
                cb = model._compact_pooled_soft_tokens(x, None, -100)[0]
                posmap = {int(p): c for c, p in enumerate(cb.position_ids[0].tolist())}
                cols = torch.tensor([posmap[int(p)] for p in pred_pos.tolist()], device="cuda")
                assert int(cols.max()) < cb.input_ids.shape[1], "compaction/column mismatch"
                lg = model(x, logits_to_keep=cols[None])[0].float()
                comp = cb.input_ids.shape[1] / x.shape[1]

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
            # tf_f1 is LEAKY (teacher forcing puts the earlier true ids in the prefix); the FIRST
            # id is predicted from a prefix that contains no id at all, so tf_id1 is honest.
            if true_ids:
                r["tf_id1"].append(float(bool(tf_ids) and tf_ids[0] == true_ids[0]))
            r["compaction"].append(comp)

            # --- pooled / real accounting for this row's gold documents -------------------------
            if keep_mask is None:
                pooled_gold, real_gold, base_pooled = [], list(gold_docs), 0.0
            else:
                present = present_docs[header]
                pooled_gold = [d for d in gold_docs if d < len(keep_mask) and not bool(keep_mask[d])]
                real_gold = [d for d in gold_docs if d < len(keep_mask) and bool(keep_mask[d])]
                base_pooled = float(np.mean([0.0 if bool(keep_mask[d]) else 1.0 for d in present])) if present else 0.0
            r["base_pooled"].append(base_pooled)
            r["rowsplit"].append(len(real_gold))

            gen_ids = None
            if do_gen:
                model_mode_train = keep_spec is not None
                if model_mode_train:
                    model.train()
                else:
                    model.eval()
                _CBCACHE["sig"], _CBCACHE["out"] = None, None
                gen = generate(model, x_in[:, :ans_start_in].clone(), a.gen_max_new, len(true_ids) or 3, tok)
                gtext = tok.decode(gen)
                gen_ids = parse_ids(gtext)
                r["gen_ids"].append({"row": ri, "ids": gen_ids, "text": gtext})
                r["gen_f1"].append(set_f1(gen_ids, true_ids))
                r["gen_em"].append(float(set(gen_ids) == set(true_ids) and len(gen_ids) == len(true_ids)))
                pred_docs = [i - off for i in gen_ids]
                if pooled_gold:
                    r["hit_gold_pooled"] += [1.0 if d in pred_docs else 0.0 for d in pooled_gold]
                if real_gold:
                    r["hit_gold_real"] += [1.0 if d in pred_docs else 0.0 for d in real_gold]
                if keep_mask is not None:
                    for d in pred_docs:
                        if 0 <= d < len(keep_mask):
                            r["pred_is_pooled"].append(0.0 if bool(keep_mask[d]) else 1.0)
                if name in ("gb00h", "cc00", "arm") and not real_gold and len(dumps) < a.dump_rows:
                    dumps.append({"row": ri, "true_ids": true_ids, "gold_docs": gold_docs,
                                  "n_docs": int(len(keep_mask)), "gen": gtext, "gen_ids": gen_ids})
            r["sec"].append(time.time() - t_cfg)
            model.eval()

        if ri + 1 in (1, 2, 5, 10) or (ri + 1) % 25 == 0:
            log(f"row {ri + 1}/{len(rows)}  (full CE {acc['full']['ce'][-1]:.3f})" if "full" in acc else f"row {ri+1}")
            table(acc, conds)

    table(acc, conds)
    if dumps:
        print("\n=== greedy id sets on pooled-gold rows (every document pooled; ids visible in headers) ===", flush=True)
        for d in dumps:
            print(f"  row {d['row']:3d} n_docs={d['n_docs']:4d} true={d['true_ids']} pred={d['gen_ids']}  {d['gen']!r}", flush=True)
    log(f"answer-id offset votes (answer_id - doc_index): {dict(id_offset_votes)}")

    out = {
        "task": "outlier", "rung": rung, "eval_size": len(acc["full"]["ce"]), "rows_loaded": len(rows), "ckpt": a.resolved_ckpt,
        "ckpt_name": a.ckpt_name, "gen_rows": min(a.gen_rows, len(rows)),
        "header_stop_id": HEADER_STOP_ID, "seed": a.seed,
        "id_offset_votes": {str(k): v for k, v in id_offset_votes.items()},
        "fingerprint_misses": n_fp_miss,
        "conditions": {c["name"]: summarize(acc[c["name"]]) for c in conds},
        "per_row": {c["name"]: dict(acc[c["name"]]) for c in conds},
        "dumps": dumps,
        "construction": {c["name"]: {k: c[k] for k in ("keep", "header", "slot", "trunc")} for c in conds},
        "arm": arm_name,
        "slot_stop_shard": pst.get("_stop_shard"),
        "slot_stop_ids_n": len(pst.get("slot_stop_ids") or []),
    }
    sfx = f"_{a.tag}" if a.tag else ""
    weka = None if a.weka_out in ("", "none") else f"{a.weka_out}/outlier_slot_probe_{a.ckpt_name}_{rung}{sfx}.json"
    local = a.out if a.out.endswith(".json") else f"{a.out}/probe_{rung}.json"
    if len(a.rungs.split(",")) > 1 and local.endswith(".json") and f"_{rung}" not in local:
        local = local[:-5] + f"_{rung}.json"
    for path in [local] + ([weka] if weka else []):
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            json.dump(out, open(path, "w"), indent=1)
            log(f"wrote {path}")
        except Exception as e:  # weka may not be mounted on a local run
            log(f"could not write {path}: {e}")
    return out["conditions"]


def summarize(r):
    d = {}
    for k, v in r.items():
        if k == "gen_ids":
            continue
        if k == "rowsplit":
            d["rows_allgoldpooled"] = int(sum(1 for z in v if z == 0))
            d["rows_somegoldreal"] = int(sum(1 for z in v if z > 0))
            continue
        d[k] = float(np.mean(v)) if v else float("nan")
        d[f"{k}_count"] = len(v)
    return d


def table(acc, conds):
    hdr = (f"{'condition':12} {'CE':>7} {'CEdig':>7} {'top1':>6} {'KL':>7} {'tfEM':>6} {'tfF1':>6} "
           f"{'tfID1':>6} {'genF1':>6} {'genEM':>6} {'R@gold_pooled':>14} {'R@gold_real':>12} "
           f"{'pred_pooled':>11} {'base_pooled':>11} {'compact':>8} {'s/row':>6}")
    print(hdr, flush=True)
    for cond in conds:
        name = cond["name"]
        r = acc[name]
        if not r["ce"]:
            continue
        m = lambda k: (np.mean(r[k]) if r[k] else float("nan"))  # noqa: E731
        print(f"{name:12} {m('ce'):7.3f} {m('ce_digit'):7.3f} {m('top1'):6.3f} {m('kl'):7.3f} {m('tf_em'):6.2f} "
              f"{m('tf_f1'):6.3f} {m('tf_id1'):6.3f} {m('gen_f1'):6.3f} {m('gen_em'):6.2f} "
              f"{m('hit_gold_pooled'):9.3f}[{len(r['hit_gold_pooled']):4d}] "
              f"{m('hit_gold_real'):7.3f}[{len(r['hit_gold_real']):4d}] "
              f"{m('pred_is_pooled'):11.3f} {m('base_pooled'):11.3f} "
              f"{m('compaction'):8.3f} {m('sec'):6.2f}", flush=True)


if __name__ == "__main__":
    main()
