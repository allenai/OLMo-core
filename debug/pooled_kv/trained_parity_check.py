"""
Task-general eval-time CE parity check for a TRAINED soft-token checkpoint (2026-09-15).

``debug/pooled_kv/outlier_probe/outlier_slot_probe.py --trained-parity`` answered this for the
outlier ``cc00`` family (records/outlier-cc00-evaltime-probe.md): score the SAME checkpoint on
FULL real text (what a ladder eval feeds it) and on its own TRAINING construction, and compare CE /
CE(digits) / free-generation F1. That probe is outlier-only in a way that matters here:
``parse_construction`` explicitly refuses ``--st-keep-frac`` (``raise SystemExit``, "not
reproducible here"), and ``--st-keep-frac --st-keep-mode gold_plus_random`` is exactly what
contradiction's ``hdr33`` and nq's ``kv33`` train with -- so it cannot check the campaign's other
two winning arms at all, and its free-generation grader (bracket doc-id set-F1) is outlier's own.

This driver generalizes both, WITHOUT editing either shared file:
  * adds the ``gold_plus_random`` keep mode (``make_fingerprint_keep_docs_fn``, the trainer's own
    path for contradiction/nq), alongside the ``gold_blind`` mode (``resolve_keep_docs``) the
    outlier probe already has, and
  * grades free generation with each task's OWN answer format, derived from the row's own
    teacher-forced answer span (never from the gold sidecar's index convention, which differs by
    task -- see ``gold_index_base_for_task`` -- so comparing model output against the row's own
    decoded answer text sidesteps that landmine entirely, the same trick the outlier probe uses for
    ``true_ids``).

Reuses ``debug/pooled_kv/eval_side_slot_probe.py`` (P) for the shared harness (checkpoint loading,
shard conversion incl. ``--emit-gold-sidecar``, IDS, EVAL_JSONL) and
``debug/pooled_kv/outlier_probe/outlier_slot_probe.py`` (O) for the ``--st-slot-mode cent_cmean``
stop-id-set builder (``build_stop_ids``, unchanged from the outlier probe's own bit-for-bit-matched
training-shard reproduction).  Neither file is modified.

    python debug/pooled_kv/trained_parity_check.py \\
      --ckpt-name ds64-oolong-occ00-b128f3-u16M --rungs 2k,8k,32k \\
      --rows 240,240,120 --gen-rows 48,48,24 --out /results/tp.json

The construction is looked up from the run name's arm field (``occ00``) via ``ARMS``, which
mirrors ``debug/ds64/launch_ds64.py:ARM_EXTRA`` for the four arms this check targets (plus a few
siblings). Reading it: parity is ``dCEdig ~ 0`` AND ``dF1 ~ 0`` -- see the outlier record's
"Reading it" note for what a large gap in either direction implies.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "outlier_probe"))
# ctc_eval's graders: same package the ladder eval itself uses (_eval_contradiction /
# _eval_retrieval are the exact functions `evaluate.py --task contradiction/retrieval` calls).
# NOTE: add src/scripts (the PARENT of the ctc_eval package), not ctc_eval itself -- evaluate.py's
# own imports are absolute (`from ctc_eval.lib... import ...`) and need `ctc_eval` importable as a
# top-level package.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(_HERE)), "src", "scripts"))

import eval_side_slot_probe as P  # noqa: E402  (shared harness)
import outlier_slot_probe as O  # noqa: E402  (only for build_stop_ids: the cent_cmean stop set)
from ctc_eval.eval.evaluate import _eval_contradiction, _oolong_extract, _oolong_norm  # noqa: E402
from ctc_eval.lib.metrics import retrieval_exact_match, retrieval_f1  # noqa: E402

from olmo_core.distributed.checkpoint import load_model_and_optim_state  # noqa: E402
from olmo_core.nn.attention.chunked_mask import build_chunk_ids_from_tokens, mark_doc_headers_free  # noqa: E402
from olmo_core.nn.attention.pooled_doc_kv import (  # noqa: E402
    PooledDocKeepHolder,
    make_fingerprint_keep_docs_fn,
    resolve_keep_docs,
)
from olmo_core.nn.lm_head import LMLossImplementation  # noqa: E402

W = P.W
IDS = P.IDS
DS64_SHARDS = f"{W}/ds64/shards"
CKPT_ROOT = f"{W}/ctc_suite/ckpts"

# arm -> (task, keep_mode, keep_value, header_stop_id, header_stop_count, slot_mode)
#   keep_mode "gold_blind"      -> resolve_keep_docs(holder=None, keep_prob=keep_value)   (--st-keep-prob)
#   keep_mode "gold_plus_random" -> make_fingerprint_keep_docs_fn(..., n_random_frac=keep_value, mode="gold_plus_random")  (--st-keep-frac)
# Keep in sync with debug/ds64/launch_ds64.py:ARM_EXTRA. header_stop_id 25 = ':' (oolong/contradiction
# headers); None = no header freeing (nq's kv33).
ARMS = {
    "occ00":  ("oolong", "gold_blind", 0.0, 25, 3, "cent_cmean"),
    "occ08":  ("oolong", "gold_blind", 0.0833, 25, 3, "cent_cmean"),
    "ohdr08": ("oolong", "gold_blind", 0.0833, 25, 3, "mean"),
    "ohdr17": ("oolong", "gold_blind", 0.1667, 25, 3, "mean"),
    "ohdr33": ("oolong", "gold_blind", 0.3333, 25, 3, "mean"),
    "hdr03":  ("contradiction", "gold_plus_random", 0.0278, 25, 1, "mean"),
    "hdr08":  ("contradiction", "gold_plus_random", 0.0833, 25, 1, "mean"),
    "hdr17":  ("contradiction", "gold_plus_random", 0.1667, 25, 1, "mean"),
    "hdr33":  ("contradiction", "gold_plus_random", 0.3333, 25, 1, "mean"),
    "kv17":   ("nq", "gold_plus_random", 0.1667, None, 1, "mean"),
    "kv33":   ("nq", "gold_plus_random", 0.3333, None, 1, "mean"),
}


def log(m):
    print(f"[trained-parity] {m}", flush=True)


def arm_for_ckpt_name(name):
    for f in name.split("-"):
        if f in ARMS:
            return f
    return None


def budget_for_ckpt_name(name):
    return name.rsplit("-u", 1)[-1] if "-u" in name else "16M"


def ckpt_path_for(name):
    return f"{CKPT_ROOT}/{name}"


_ID_RE = re.compile(r"\[\s*(\d+)\s*\]")
_PAIR_RE = re.compile(r"[\[\(]\s*(\d+)\s*,\s*(\d+)\s*[\]\)]")


def parse_ids(text):
    out = []
    for m in _ID_RE.finditer(text):
        v = int(m.group(1))
        if v not in out:
            out.append(v)
    return out


def parse_pairs_text(text):
    """Same convention as ctc_eval's parse_pairs, applied to a decoded token span (JSON list of
    pairs, or a fallback ``(a, b)``/``[a, b]`` regex scan)."""
    text = text.strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [sorted([int(p[0]), int(p[1])]) for p in parsed if isinstance(p, list) and len(p) == 2]
    except (json.JSONDecodeError, ValueError, TypeError):
        pass
    m = _PAIR_RE.findall(text)
    if m:
        return [sorted([int(a), int(b)]) for a, b in m]
    return []


def set_f1(pred, true):
    if not pred or not true:
        return 0.0
    inter = len(set(pred) & set(true))
    if inter == 0:
        return 0.0
    p, r = inter / len(set(pred)), inter / len(set(true))
    return 2 * p * r / (p + r)


def grade_oolong(gold_text, pred_text):
    """Minimal reproduction of ctc_eval.eval.evaluate._eval_oolong's three branches (numeric
    partial credit / set-overlap F1 / exact match), driven off the row's own decoded answer span
    instead of the raw example's ``_meta`` (which this probe never loads -- see module docstring).
    Both gold and pred go through the SAME ``_oolong_extract`` (strip an 'Answer:'/'Label:'/...
    prefix) so a raw teacher-forced gold span compares fairly against free-generated text.
    Returns (score, exact_match)."""
    gold = _oolong_extract(gold_text)
    pred = _oolong_extract(pred_text)
    gold_items = [g.strip() for g in re.split(r"[;,]", gold) if g.strip()]
    try:
        gval = float(gold.strip())
        nums = re.findall(r"-?\d+\.?\d*", pred)
        pval = float(nums[-1])
        err = abs(gval - pval)
        return 0.75**err, float(err == 0)
    except (ValueError, IndexError):
        pass
    if len(gold_items) > 1:
        pset = {_oolong_norm(x) for x in re.split(r"[;,]", pred)}
        gset = {_oolong_norm(x) for x in gold_items}
        tp = len(pset & gset)
        p = tp / len(pset) if pset else 0.0
        r = tp / len(gset) if gset else 0.0
        score = 2 * p * r / (p + r) if (p + r) else 0.0
        return score, float(pset == gset)
    em = float(_oolong_norm(pred) == _oolong_norm(gold))
    return em, em


def grade(task, gold_text, pred_text):
    """Returns (score, exact_match) with SCORE the number this driver's genF1 column reports:
    contradiction pair-F1, nq retrieval-F1, oolong's own score (EM / numeric partial credit /
    set-F1 depending on the row -- see grade_oolong)."""
    if task == "contradiction":
        gold_pairs = parse_pairs_text(gold_text)  # ctc_eval's own parse_pairs re-derives predicted
        m, _ = _eval_contradiction([{"gold_doc_indices": gold_pairs}], [pred_text])
        return m["f1"], m["exact_match"]
    if task == "nq":
        # NOTE: use retrieval_f1 directly on the ids as PARSED FROM TEXT, not
        # ctc_eval's compute_retrieval_metrics_single -- that helper adds +1 to its
        # gold_doc_indices arg because it expects the RAW 0-indexed JSONL field, but both gold_text
        # and pred_text here are already decoded in the answer's own 1-indexed DISPLAY convention
        # (see module docstring: grading never touches the sidecar's index base). Applying that +1
        # here would double-shift.
        gold_ids, pred_ids = set(parse_ids(gold_text)), set(parse_ids(pred_text))
        return retrieval_f1(pred_ids, gold_ids), float(retrieval_exact_match(pred_ids, gold_ids))
    if task == "oolong":
        return grade_oolong(gold_text, pred_text)
    raise ValueError(task)


def chunk_ids_for(x_cpu, header, header_stop_id, header_stop_count):
    cid = build_chunk_ids_from_tokens(x_cpu, doc_start_id=IDS.doc_start, doc_end_id=IDS.doc_end, eos_id=IDS.eos, mode="chunked")
    if header and header_stop_id is not None:
        cid = mark_doc_headers_free(cid, x_cpu, doc_start_id=IDS.doc_start, doc_end_id=IDS.doc_end,
                                     stop_id=header_stop_id, stop_count=header_stop_count, cap=32)
    return cid


@torch.no_grad()
def generate(model, seq, max_new, tok):
    """Free greedy continuation until EOS or max_new (no bracket-count early stop: unlike outlier,
    not every task's answer is a bracketed id list)."""
    gen = []
    for _ in range(max_new):
        lg = model(seq, logits_to_keep=1)[0][-1]
        nxt = int(lg.argmax())
        if nxt == IDS.eos:
            break
        gen.append(nxt)
        seq = torch.cat([seq, torch.tensor([[nxt]], device=seq.device)], dim=1)
    return gen


def parity_report(ckpt_name, arm_name, rows_by_rung, summaries):
    print("\n=== EVAL-TIME CE PARITY  (one checkpoint, two inputs) ===", flush=True)
    print(f"ckpt {ckpt_name}   arm {arm_name}   task {ARMS[arm_name][0]}", flush=True)
    hdr = (f"{'rung':>5} {'eval_size':>9} | {'CE_full':>8} {'CE_soft':>8} {'dCE':>8} | "
           f"{'CEdig_full':>10} {'CEdig_soft':>10} {'dCEdig':>8} | "
           f"{'F1_full':>8} {'F1_soft':>8} {'dF1':>7} | {'compact':>7}")
    print(hdr, flush=True)
    for rung, s in summaries.items():
        f, g = s.get("full"), s.get("arm")
        if not f or not g:
            continue
        print(f"{rung:>5} {int(f['ce_count']):>9} | {f['ce']:8.3f} {g['ce']:8.3f} {g['ce'] - f['ce']:+8.3f} | "
              f"{f['ce_digit']:10.3f} {g['ce_digit']:10.3f} {g['ce_digit'] - f['ce_digit']:+8.3f} | "
              f"{f['gen_f1']:8.3f} {g['gen_f1']:8.3f} {g['gen_f1'] - f['gen_f1']:+7.3f} | "
              f"{g['compaction']:7.3f}", flush=True)
    print("PARITY = dCEdig ~ 0 AND dF1 ~ 0. A large dCEdig with a WEAK full-text F1 is not parity "
          "-- it is two ways of being wrong.", flush=True)


def summarize(r):
    d = {}
    for k, v in r.items():
        if k == "gen_texts":
            continue
        d[k] = float(np.mean(v)) if v else float("nan")
        d[f"{k}_count"] = len(v)
    return d


def table(acc):
    hdr = f"{'condition':10} {'CE':>7} {'CEdig':>7} {'top1':>6} {'KL':>7} {'genF1':>6} {'genEM':>6} {'compact':>8} {'s/row':>6}"
    print(hdr, flush=True)
    for name, r in acc.items():
        if not r["ce"]:
            continue
        m = lambda k: (np.mean(r[k]) if r[k] else float("nan"))  # noqa: E731
        print(f"{name:10} {m('ce'):7.3f} {m('ce_digit'):7.3f} {m('top1'):6.3f} {m('kl'):7.3f} "
              f"{m('gen_f1'):6.3f} {m('gen_em'):6.2f} {m('compaction'):8.3f} {m('sec'):6.2f}", flush=True)


@torch.no_grad()
def run_rung(a, task, arm_name, rung, rows_n, gen_rows_n, model, pst, tok, keep_mode, keep_val,
             header_stop_id, header_stop_count, slot_mode):
    shard = f"{a.work}/{task}_{rung}"
    if a.jsonl:
        jsonl = a.jsonl
    elif a.eval_dir:
        # local-cluster mirror: same basename as the weka rung file, flat under <eval-dir>/<task>/
        jsonl = os.path.join(a.eval_dir, task, os.path.basename(P.EVAL_JSONL[task][rung]))
    else:
        jsonl = P.EVAL_JSONL[task][rung]
    P.convert(task, jsonl, rows_n, shard)
    rows, masks = P.load_rows(shard, rows_n)
    log(f"=== rung {rung}: {len(rows)} rows; lengths {[len(r) for r in rows[:6]]}")
    if len(rows) < 200:
        log(f"WARNING eval_size={len(rows)} (<200 requested): binomial SE on a right/wrong metric "
            f"at f1~0.5 is {0.5 / max(1, len(rows)) ** 0.5:.3f}")

    max_pos = max(len(r) for r in rows) + a.gen_max_new + 8
    n_warm = 0
    for mod in model.modules():
        rope = getattr(mod, "rope", None)
        if rope is not None and hasattr(rope, "warmup_cache"):
            rope.warmup_cache(max_pos, torch.device("cuda"))
            n_warm += 1
    log(f"warmed {n_warm} RoPE caches to {max_pos} positions")

    gold_table = None
    if keep_mode == "gold_plus_random":
        gold_table = json.load(open(f"{shard}/gold_fingerprints.json"))
        log(f"gold sidecar: {len(gold_table)} fingerprints")

    acc = {c: {"ce": [], "ce_digit": [], "top1": [], "kl": [], "gen_f1": [], "gen_em": [],
               "compaction": [], "sec": [], "gen_texts": []} for c in ("full", "arm")}

    header = header_stop_id is not None
    for ri, (row, rmask) in enumerate(zip(rows, masks)):
        x = torch.tensor(row[None], device="cuda")
        ans_pos = torch.tensor(np.nonzero(rmask)[0], device="cuda")
        pred_pos = ans_pos - 1
        targets = x[0, ans_pos]
        true_text = tok.decode(targets.tolist())
        pieces = [tok.decode([int(t)]) for t in targets.tolist()]
        digit_sel = torch.tensor([i for i, q in enumerate(pieces) if any(ch.isdigit() for ch in q)],
                                  device="cuda", dtype=torch.long)
        ans_start = int(ans_pos[0])
        do_gen = ri < gen_rows_n

        full_lg = None
        for name in ("full", "arm"):
            t_cfg = time.time()
            r = acc[name]
            if name == "full":
                model.eval()
                model._pooled_keep_holder = None
                lg = model(x, logits_to_keep=pred_pos[None])[0].float()
                full_lg = lg
                comp = 1.0
            else:
                pst["header_stop_id"] = header_stop_id if header else None
                pst["header_stop_count"] = header_stop_count
                pst["header_cap"] = 32
                pst["slot_mode"] = slot_mode
                cid = chunk_ids_for(x.cpu(), header, header_stop_id, header_stop_count)
                n_docs = int(cid.max()) + 1
                if keep_mode == "gold_blind":
                    keep_mask = resolve_keep_docs(cid, n_docs, holder=None, keep_prob=float(keep_val),
                                                   keep_seed=a.seed).cpu()
                else:  # gold_plus_random -- the trainer's own fingerprint-sidecar path
                    keep_fn = make_fingerprint_keep_docs_fn(
                        gold_table, doc_start_id=IDS.doc_start, doc_end_id=IDS.doc_end, eos_id=IDS.eos,
                        n_random_frac=float(keep_val), mode="gold_plus_random", seed=a.seed)
                    keep_mask = keep_fn(x.cpu())
                model.train()
                model._pooled_keep_holder = PooledDocKeepHolder(keep_docs=keep_mask.clone())
                cb = model._compact_pooled_soft_tokens(x, None, -100)[0]
                posmap = {int(p): c for c, p in enumerate(cb.position_ids[0].tolist())}
                cols = torch.tensor([posmap[int(p)] for p in pred_pos.tolist()], device="cuda")
                assert int(cols.max()) < cb.input_ids.shape[1], "compaction/column mismatch"
                lg = model(x, logits_to_keep=cols[None])[0].float()
                comp = cb.input_ids.shape[1] / x.shape[1]

            r["ce"].append(float(F.cross_entropy(lg, targets)))
            if digit_sel.numel():
                r["ce_digit"].append(float(F.cross_entropy(lg[digit_sel], targets[digit_sel])))
            r["top1"].append(float((lg.argmax(-1) == full_lg.argmax(-1)).float().mean()))
            r["kl"].append(float(F.kl_div(F.log_softmax(lg, -1), F.log_softmax(full_lg, -1),
                                           log_target=True, reduction="batchmean")))
            r["compaction"].append(comp)

            if do_gen:
                model.train() if name == "arm" else model.eval()
                gen = generate(model, x[:, :ans_start].clone(), a.gen_max_new, tok)
                gtext = tok.decode(gen)
                score, em = grade(task, true_text, gtext)
                r["gen_f1"].append(score)
                r["gen_em"].append(em)
                r["gen_texts"].append({"row": ri, "true": true_text, "pred": gtext})
            r["sec"].append(time.time() - t_cfg)
            model.eval()

        if ri + 1 in (1, 2, 5, 10) or (ri + 1) % 25 == 0:
            log(f"row {ri + 1}/{len(rows)}  (full CE {acc['full']['ce'][-1]:.3f})")
            table(acc)

    table(acc)
    if acc["arm"]["gen_texts"]:
        print("\n=== sample generations (arm construction) ===", flush=True)
        for d in acc["arm"]["gen_texts"][:8]:
            print(f"  row {d['row']:3d} true={d['true']!r} pred={d['pred']!r}", flush=True)

    out = {
        "task": task, "arm": arm_name, "rung": rung, "eval_size": len(acc["full"]["ce"]),
        "rows_loaded": len(rows), "gen_rows": min(gen_rows_n, len(rows)), "seed": a.seed,
        "keep_mode": keep_mode, "keep_value": keep_val, "header_stop_id": header_stop_id,
        "header_stop_count": header_stop_count, "slot_mode": slot_mode,
        "conditions": {c: summarize(acc[c]) for c in acc},
        "per_row": {c: dict(acc[c]) for c in acc},
    }
    sfx = f"_{a.tag}" if a.tag else ""
    weka = None if a.weka_out in ("", "none") else f"{a.weka_out}/trained_parity_{a.ckpt_name}_{rung}{sfx}.json"
    local = a.out if a.out.endswith(".json") else f"{a.out}/tp_{rung}.json"
    if local.endswith(".json") and f"_{rung}" not in local:
        local = local[:-5] + f"_{rung}.json"
    for path in [local] + ([weka] if weka else []):
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            json.dump(out, open(path, "w"), indent=1)
            log(f"wrote {path}")
        except Exception as e:  # weka may not be mounted on a local run
            log(f"could not write {path}: {e}")
    return out["conditions"]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-name", required=True)
    ap.add_argument("--ckpt", default=None, help="explicit override path")
    ap.add_argument("--arm", default="auto", help="'auto' = parsed off --ckpt-name; else a key of ARMS")
    ap.add_argument("--task", default="auto", help="'auto' = ARMS[arm][0]")
    ap.add_argument("--rungs", default="2k,8k,32k")
    ap.add_argument("--rows", default="240,240,120", help="single int, or one per rung")
    ap.add_argument("--gen-rows", default="48,48,24", help="single int, or one per rung")
    ap.add_argument("--gen-max-new", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--jsonl", default=None, help="override eval JSONL for EVERY rung (single-rung local runs only)")
    ap.add_argument("--eval-dir", default=None,
                     help="local-cluster mirror of the eval rung files: <eval-dir>/<task>/<basename of the "
                          "weka EVAL_JSONL path>, e.g. /data/prasann/ds64_eval (varies correctly per rung, "
                          "unlike --jsonl)")
    ap.add_argument("--work", default="/results/trained_parity_work")
    ap.add_argument("--out", default="/results/trained_parity.json")
    ap.add_argument("--weka-out", default=f"{W}/_eval_results/trained_parity",
                     help="set to 'none' when weka isn't mounted (local-cluster runs)")
    ap.add_argument("--tag", default="")
    ap.add_argument("--slot-stop-topk", type=int, default=100)
    ap.add_argument("--slot-stop-rows", type=int, default=512)
    ap.add_argument("--slot-stop-shard-root", default=None,
                     help="override DS64_SHARDS (weka) for the --st-slot-mode cent_cmean stop-id-set "
                          "training shard, e.g. a local-cluster mirror root holding <task>_u<budget>/")
    a = ap.parse_args()

    arm_name = arm_for_ckpt_name(a.ckpt_name) if a.arm == "auto" else a.arm
    if arm_name is None or arm_name not in ARMS:
        raise SystemExit(f"could not resolve an arm from --ckpt-name {a.ckpt_name!r} / --arm {a.arm!r}; "
                          f"known arms: {sorted(ARMS)}")
    task, keep_mode, keep_val, header_stop_id, header_stop_count, slot_mode = ARMS[arm_name]
    if a.task != "auto" and a.task != task:
        raise SystemExit(f"--task {a.task} disagrees with arm {arm_name}'s task {task}")
    log(f"ckpt {a.ckpt_name}  arm {arm_name}  task {task}  keep_mode={keep_mode} keep={keep_val} "
        f"header_stop_id={header_stop_id} (count {header_stop_count}) slot_mode={slot_mode}")

    rungs = [r for r in a.rungs.split(",") if r]
    rows_l = [int(x) for x in a.rows.split(",")]
    rows_l = rows_l * len(rungs) if len(rows_l) == 1 else rows_l
    gen_l = [int(x) for x in a.gen_rows.split(",")]
    gen_l = gen_l * len(rungs) if len(gen_l) == 1 else gen_l
    assert len(rows_l) == len(rungs) and len(gen_l) == len(rungs), "--rows/--gen-rows must match --rungs"

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(P.TOKENIZER)

    cfg = P.build_cfg()
    cfg.lm_head.loss_implementation = LMLossImplementation.default
    model = cfg.build(init_device="cpu")
    ck = P.find_ckpt(a.ckpt or ckpt_path_for(a.ckpt_name))
    t0 = time.time()
    load_model_and_optim_state(ck, model)
    log(f"loaded {ck} in {time.time() - t0:.0f}s")
    model.enable_pooled_soft_tokens(IDS.doc_start, IDS.doc_end, IDS.eos, placeholder_id=IDS.landmark,
                                     keep_prob=0.0, keep_seed=a.seed, detach_soft_kv=True)
    wout = float(model.pooled_projector.w_out.weight.abs().max())
    log(f"keeping the checkpoint's own pooled_projector (max|w_out| = {wout:.3e}; 0 => identity, "
        f"which is what detach_soft_kv training leaves behind)")
    model = model.cuda().to(torch.bfloat16)
    pst = model._pooled_soft_tokens

    if slot_mode != "mean":
        shards_root = a.slot_stop_shard_root or DS64_SHARDS
        shard_dir = f"{shards_root}/{task}_u{budget_for_ckpt_name(a.ckpt_name)}"
        stop_ids, shown, n_scanned = O.build_stop_ids(shard_dir, tok, topk=a.slot_stop_topk, n_rows=a.slot_stop_rows)
        pst["slot_stop_ids"] = [int(t) for t in stop_ids]
        pst["slot_stop_mask"] = None
        log(f"slot stop set: {len(stop_ids)} ids from {n_scanned} tokens of {shard_dir}; "
            f"most frequent dropped: " + " ".join(shown[:16]))

    summaries = {}
    for rung, rows_n, gen_rows_n in zip(rungs, rows_l, gen_l):
        summaries[rung] = run_rung(a, task, arm_name, rung, rows_n, gen_rows_n, model, pst, tok,
                                    keep_mode, keep_val, header_stop_id, header_stop_count, slot_mode)
    parity_report(a.ckpt_name, arm_name, dict(zip(rungs, rows_l)), summaries)


if __name__ == "__main__":
    main()
