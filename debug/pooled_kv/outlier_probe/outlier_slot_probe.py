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

# ds64 dense (full-attention-trained) outlier arms; `find_ckpt` takes the run dir or model_and_optim.
CKPTS = {
    "ds64-outlier-dense-u64M": f"{W}/ctc_suite/ckpts/ds64-outlier-dense-u64M/model_and_optim",
    "ds64-outlier-dense-u128M": f"{W}/ctc_suite/ckpts/ds64-outlier-dense-u128M/model_and_optim",
    "ds64-outlier-dense-u32M": f"{W}/ctc_suite/ckpts/ds64-outlier-dense-u32M/model_and_optim",
    "ds64-outlier-dense-u16M": f"{W}/ctc_suite/ckpts/ds64-outlier-dense-u16M/model_and_optim",
    # the checkpoint the 2026-09-08 "gold-only parity" number was measured on
    "lmx-full-mixs160M-4b": f"{W}/*/ckpts/lmx-full-mixs160M-4b-2026",
}
# the rung files the ds64 outlier arms are evaluated on (EVAL_JSONL in the shared harness has no 2k)
RUNGS = {r: f"{W}/outlier_lengthmix/eval_rungs/outlier/rung_{n}.jsonl"
         for r, n in (("2k", 2048), ("8k", 8192), ("16k", 16384), ("32k", 32768))}

# (name, keep-spec, header?) -- keep-spec: "gold" | float p (gold-blind Bernoulli) ; "swap" via name
CONDITIONS = [
    ("full", None, False),
    ("goldonly", "gold", False),
    ("gb50", 0.5, False),
    ("gb50h", 0.5, True),
    ("gb17h", 1.0 / 6.0, True),
    ("gb00h", 0.0, True),
    ("gb00", 0.0, False),
    ("gb00h_swap", 0.0, True),
]

_SWAP = {"on": False, "gold": None, "seed": 0}


def log(m):
    print(f"[outlier-probe] {m}", flush=True)


def install_swap_patch(model):
    """CONTROL: exchange the slot feature vectors of the GOLD documents with those of an equal
    number of random non-gold pooled documents, leaving every position / id / length untouched.
    A construction whose F1 survives this is not reading slot content."""
    cls = type(model)
    orig = cls._compact_pooled_soft_tokens

    def patched(self, input_ids, labels, ignore_index):
        out = orig(self, input_ids, labels, ignore_index)
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


def chunk_ids_for(x, header: bool):
    """The chunk ids the model's own compaction will see (post header-freeing)."""
    cid = build_chunk_ids_from_tokens(
        x, doc_start_id=IDS.doc_start, doc_end_id=IDS.doc_end, eos_id=IDS.eos, mode="chunked"
    )
    if header:
        cid = mark_doc_headers_free(
            cid, x, doc_start_id=IDS.doc_start, doc_end_id=IDS.doc_end,
            stop_id=HEADER_STOP_ID, stop_count=1, cap=32,
        )
    return cid


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
    ap = argparse.ArgumentParser()
    ap.add_argument("--rung", default="2k")
    ap.add_argument("--rows", type=int, default=200)
    ap.add_argument("--gen-rows", type=int, default=32, help="rows that also get FREE greedy generation (0 = none)")
    ap.add_argument("--gen-max-new", type=int, default=28)
    ap.add_argument("--ckpt-name", default="ds64-outlier-dense-u64M", choices=sorted(CKPTS))
    ap.add_argument("--ckpt", default=None, help="explicit override path")
    ap.add_argument("--jsonl", default=None, help="override eval JSONL (local runs)")
    ap.add_argument("--shard", default=None, help="already-tokenized shard dir")
    ap.add_argument("--conditions", default=",".join(c[0] for c in CONDITIONS))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tokenizer", default=None, help="local tokenizer dir (default: the shared harness's HF id)")
    ap.add_argument("--work", default="/results/outlier_probe_work")
    ap.add_argument("--out", default="/results/outlier_slot_probe.json")
    ap.add_argument("--weka-out", default=f"{W}/_eval_results/outlier_slot_probe")
    ap.add_argument("--dump-rows", type=int, default=12, help="pooled-gold rows to dump greedy id sets for")
    a = ap.parse_args()

    if a.tokenizer:
        P.TOKENIZER = a.tokenizer
    want = set(a.conditions.split(",")) | {"full"}  # FULL is the reference every metric needs
    conds = [c for c in CONDITIONS if c[0] in want]
    log(f"conditions: {[c[0] for c in conds]}")

    shard = a.shard or f"{a.work}/outlier_{a.rung}"
    if a.shard is None:
        P.convert("outlier", a.jsonl or RUNGS[a.rung], a.rows, shard)
    rows, masks = P.load_rows(shard, a.rows)
    log(f"{len(rows)} rows; lengths {[len(r) for r in rows[:6]]}")
    if len(rows) < 500:
        log(f"WARNING eval_size={len(rows)} (<500): binomial SE on a right/wrong metric at f1~0.5 is "
            f"{0.5 / max(1, len(rows)) ** 0.5:.3f}")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(P.TOKENIZER)

    cfg = P.build_cfg()
    cfg.lm_head.loss_implementation = LMLossImplementation.default
    model = cfg.build(init_device="cpu")
    ck = P.find_ckpt(a.ckpt or CKPTS[a.ckpt_name])
    t0 = time.time()
    load_model_and_optim_state(ck, model)
    log(f"loaded {ck} in {time.time() - t0:.0f}s")
    model.enable_pooled_soft_tokens(
        IDS.doc_start, IDS.doc_end, IDS.eos, placeholder_id=IDS.landmark,
        keep_prob=0.0, keep_seed=a.seed, detach_soft_kv=True,
    )
    model.pooled_projector.reset_parameters()  # identity: soft token == mean input embedding
    model = model.cuda().to(torch.bfloat16)
    pst = model._pooled_soft_tokens
    install_swap_patch(model)

    # Gold ids straight from the sidecar. NOT make_fingerprint_keep_docs_fn: with
    # n_random_frac=0.0 that helper still forces `max(1, ...)` == ONE random non-gold document
    # real, so the 2026-09-08 "gold-only" probe point was really gold + 1 random. Here `goldonly`
    # is exactly the gold documents.
    gold_table = json.load(open(f"{shard}/gold_fingerprints.json"))
    gold_table = {fp: sorted({int(i) for v in val for i in (v if isinstance(v, (list, tuple)) else [v])})
                  for fp, val in gold_table.items()}
    log(f"gold sidecar: {len(gold_table)} fingerprints")
    n_fp_miss = 0

    acc = {c[0]: {"ce": [], "top1": [], "kl": [], "tf_em": [], "compaction": [], "sec": [],
                  "gen_em": [], "gen_f1": [], "tf_f1": [],
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
        for hh in sorted({h for _, _, h in conds}):
            cc = chunk_ids_for(x.cpu(), hh)
            cid_cache[hh] = cc
            present_docs[hh] = [int(v) for v in torch.unique(cc[0]).tolist() if v >= 0]

        full_lg = None
        do_gen = ri < a.gen_rows
        for name, keep_spec, header in conds:
            t_cfg = time.time()
            pst["header_stop_id"] = HEADER_STOP_ID if header else None
            pst["header_stop_count"] = 1
            pst["header_cap"] = 32
            _SWAP["on"] = name.endswith("_swap")
            _SWAP["gold"] = gold_row
            _SWAP["seed"] = a.seed

            if keep_spec is None:  # FULL
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
                lg = model(x, logits_to_keep=cols[None])[0].float()
                comp = cb.input_ids.shape[1] / x.shape[1]

            r = acc[name]
            r["ce"].append(float(F.cross_entropy(lg, targets)))
            r["top1"].append(float((lg.argmax(-1) == full_lg.argmax(-1)).float().mean()))
            r["kl"].append(float(F.kl_div(F.log_softmax(lg, -1), F.log_softmax(full_lg, -1),
                                          log_target=True, reduction="batchmean")))
            r["tf_em"].append(float((lg.argmax(-1) == targets).all()))
            r["tf_f1"].append(set_f1(parse_ids(tok.decode(lg.argmax(-1).tolist())), true_ids))
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
                gen = generate(model, x[:, :ans_start].clone(), a.gen_max_new, len(true_ids) or 3, tok)
                gtext = tok.decode(gen)
                gen_ids = parse_ids(gtext)
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
                if name == "gb00h" and not real_gold and len(dumps) < a.dump_rows:
                    dumps.append({"row": ri, "true_ids": true_ids, "gold_docs": gold_docs,
                                  "n_docs": int(len(keep_mask)), "gen": gtext, "gen_ids": gen_ids})
            r["sec"].append(time.time() - t_cfg)
            model.eval()

        if ri + 1 in (1, 2, 5, 10) or (ri + 1) % 25 == 0:
            log(f"row {ri + 1}/{len(rows)}  (full CE {acc['full']['ce'][-1]:.3f})" if "full" in acc else f"row {ri+1}")
            table(acc, conds)

    table(acc, conds)
    if dumps:
        print("\n=== gb00h greedy id sets on pooled-gold rows (every document pooled; ids visible in headers) ===", flush=True)
        for d in dumps:
            print(f"  row {d['row']:3d} n_docs={d['n_docs']:4d} true={d['true_ids']} pred={d['gen_ids']}  {d['gen']!r}", flush=True)
    log(f"answer-id offset votes (answer_id - doc_index): {dict(id_offset_votes)}")

    out = {
        "task": "outlier", "rung": a.rung, "eval_size": len(acc["full"]["ce"]), "rows_loaded": len(rows), "ckpt": ck,
        "ckpt_name": a.ckpt_name, "gen_rows": min(a.gen_rows, len(rows)),
        "header_stop_id": HEADER_STOP_ID, "seed": a.seed,
        "id_offset_votes": {str(k): v for k, v in id_offset_votes.items()},
        "fingerprint_misses": n_fp_miss,
        "conditions": {n: summarize(acc[n]) for n, _, _ in conds},
        "per_row": {n: {k: v for k, v in acc[n].items()} for n, _, _ in conds},
        "dumps": dumps,
    }
    for path in [a.out] + ([f"{a.weka_out}/outlier_slot_probe_{a.ckpt_name}_{a.rung}.json"] if a.weka_out else []):
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            json.dump(out, open(path, "w"), indent=1)
            log(f"wrote {path}")
        except Exception as e:  # weka may not be mounted on a local run
            log(f"could not write {path}: {e}")


def summarize(r):
    d = {}
    for k, v in r.items():
        if k == "rowsplit":
            d["rows_allgoldpooled"] = int(sum(1 for z in v if z == 0))
            d["rows_somegoldreal"] = int(sum(1 for z in v if z > 0))
            continue
        d[k] = float(np.mean(v)) if v else float("nan")
        d[f"{k}_count"] = len(v)
    return d


def table(acc, conds):
    hdr = (f"{'condition':12} {'CE':>7} {'top1':>6} {'KL':>7} {'tfEM':>6} {'tfF1':>6} "
           f"{'genF1':>6} {'genEM':>6} {'R@gold_pooled':>14} {'R@gold_real':>12} "
           f"{'pred_pooled':>11} {'base_pooled':>11} {'compact':>8} {'s/row':>6}")
    print(hdr, flush=True)
    for name, _, _ in conds:
        r = acc[name]
        if not r["ce"]:
            continue
        m = lambda k: (np.mean(r[k]) if r[k] else float("nan"))  # noqa: E731
        print(f"{name:12} {m('ce'):7.3f} {m('top1'):6.3f} {m('kl'):7.3f} {m('tf_em'):6.2f} "
              f"{m('tf_f1'):6.3f} {m('gen_f1'):6.3f} {m('gen_em'):6.2f} "
              f"{m('hit_gold_pooled'):9.3f}[{len(r['hit_gold_pooled']):4d}] "
              f"{m('hit_gold_real'):7.3f}[{len(r['hit_gold_real']):4d}] "
              f"{m('pred_is_pooled'):11.3f} {m('base_pooled'):11.3f} "
              f"{m('compaction'):8.3f} {m('sec'):6.2f}", flush=True)


if __name__ == "__main__":
    main()
