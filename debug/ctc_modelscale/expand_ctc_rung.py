#!/usr/bin/env python3
"""Grow a CTC-suite eval rung to a longer context, for the length-generalization study.

WHY NOT REUSE THE EXISTING xlong FILES. `contradiction/rung_131072.jsonl` already exists, and its
gold pairs are a proper nested superset of `rung_32768.jsonl` -- but its distractors average
**47.1 tokens/doc against 15.6 for every rung from 2k to 32k**. It was built from a filler glob
that also matched the FEVER and wiki_mix corpora (the leak documented in
`src/corpus_reasoning/data/build_xlong_rungs.py` and the `contra-fever-filler-leak` record), so it
is a different document distribution, not a longer version of the same ladder. Scoring it next to
the 2k-32k rungs would read as a length effect when part of it is a corpus change. This script
therefore builds long rungs from **each task's own rung file**, so the filler pool is by
construction the same corpus at the same document length.

WHY THE TARGET IS MEASURED, NOT LABELLED. Rung labels are build-time targets that several tasks
missed: measured against the real tokenizer, contradiction's rungs carry ~1.5x fewer document
tokens than their label and niah's ~2.9x fewer (`measure_rung_tokens.py`). Rungs built here are
calibrated on MEASURED tokens/doc from the source file, and the output name records the measured
budget, so the new points sit on a true token axis. That means a `64k` file built here is NOT the
same x-position as a hypothetical `rung_65536` built by the old pipeline -- report the measured
value.

Construction (mirrors the "keep_all + self_nongold" mode of build_xlong_rungs.py):
  * every document of the source example is KEPT, so each output example is a strict nested
    superset of the source example -- same gold, same original hard negatives, length is the only
    variable;
  * distractors are drawn from the pool of documents that are non-gold in EVERY example of the
    source file (a doc that is gold anywhere is never injected, which would plant a stray answer);
  * documents are shuffled with a per-example seed and every index field is remapped.

    python debug/ctc_modelscale/expand_ctc_rung.py --task contradiction \\
        --src /scratch/.../eval_rungs/contradiction/rung_32768.jsonl \\
        --targets 65536,131072 --out-dir /scratch/.../eval_rungs_xlong/contradiction
"""
import argparse
import json
import os
import random
import statistics

TOKENIZER = "Qwen/Qwen3.5-0.8B-Base"

#: Per-task gold-index conventions. ``base`` is the index origin of ``gold_doc_indices``
#: (contradiction is 1-indexed, the retrieval-family tasks are 0-indexed -- the
#: `gold-doc-indices-per-task-base` record); ``pairs`` marks gold stored as [[a,b],...] rather
#: than a flat list. Both are asserted against the data at load time, never trusted blindly.
TASKS = {
    "contradiction": {"base": 1, "pairs": True, "index_fields": []},
    "niah": {"base": 1, "pairs": False, "index_fields": []},
    "nq": {"base": 0, "pairs": False, "index_fields": ["hard_neg_indices"]},
    "hotpotqa": {"base": 0, "pairs": False, "index_fields": ["hard_neg_indices"]},
    "msmarco": {"base": 0, "pairs": False, "index_fields": ["hard_neg_indices"]},
}


def doc_text(d):
    if isinstance(d, dict):
        t = d.get("title") or ""
        return (t + " " + d.get("text", "")).strip() if t else d.get("text", "")
    return str(d)


def flat_gold(row, cfg):
    """Gold indices of one row as a flat list of 0-based positions into ``documents``."""
    g = row.get("gold_doc_indices") or []
    flat = []
    for x in g:
        flat.extend(x if isinstance(x, (list, tuple)) else [x])
    return [i - cfg["base"] for i in flat]


def detect_base(rows, cfg):
    """Verify the configured index base: every gold index must be in range for all rows.

    A wrong base silently shifts every gold by one, which grades as a near-total failure that
    looks like a model collapse. Checked here rather than assumed.
    """
    for base in (cfg["base"], 1 - cfg["base"]):
        probe = dict(cfg, base=base)
        if all(all(0 <= i < len(r["documents"]) for i in flat_gold(r, probe)) for r in rows):
            return base
    raise SystemExit("FATAL: neither index base 0 nor 1 keeps every gold index in range")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=sorted(TASKS))
    ap.add_argument("--src", required=True, help="source rung JSONL to grow")
    ap.add_argument("--targets", required=True, help="comma-separated MEASURED token budgets")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--calib", type=int, default=25, help="examples tokenized for calibration")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    cfg = dict(TASKS[args.task])
    rows = [json.loads(l) for l in open(args.src)]
    print(f"[expand] {args.task}: {len(rows)} examples from {args.src}", flush=True)

    cfg["base"] = detect_base(rows, cfg)
    print(f"[expand] gold index base verified = {cfg['base']}", flush=True)

    # --- distractor pool: docs that are gold in NO example ---
    gold_texts = set()
    for r in rows:
        for i in flat_gold(r, cfg):
            gold_texts.add(doc_text(r["documents"][i]))
    pool, seen = [], set()
    for r in rows:
        for d in r["documents"]:
            t = doc_text(d)
            if t in gold_texts or t in seen:
                continue
            seen.add(t)
            pool.append(d)
    print(f"[expand] pool={len(pool)} distinct non-gold docs "
          f"({len(gold_texts)} gold texts excluded)", flush=True)

    # --- calibrate tokens/doc on the source file, through the real tokenizer ---
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(TOKENIZER, trust_remote_code=True)
    tot, nd = [], []
    for r in rows[: args.calib]:
        tot.append(len(tok("\n".join(doc_text(d) for d in r["documents"]),
                           add_special_tokens=False)["input_ids"]))
        nd.append(len(r["documents"]))
    tok_per_doc = statistics.median(tot) / statistics.median(nd)
    src_docs = int(statistics.median(nd))
    print(f"[expand] measured {statistics.median(tot):.0f} doc-tokens over {src_docs} docs "
          f"-> {tok_per_doc:.1f} tok/doc", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    for target in [int(t) for t in args.targets.split(",")]:
        n_target = int(round(target / tok_per_doc))
        need = n_target - src_docs
        if need <= 0:
            print(f"[expand] {target}: source already has {src_docs} docs >= {n_target}, skip")
            continue
        if need > len(pool):
            print(f"[expand] {target}: SKIP -- needs {need} fillers/example, pool has {len(pool)}")
            continue
        out = os.path.join(args.out_dir, f"rung_{target}.jsonl")
        rng_master = random.Random(args.seed)
        with open(out, "w") as fh:
            for ei, r in enumerate(rows):
                rng = random.Random(args.seed * 1_000_003 + ei)
                docs = list(r["documents"])
                extra = rng.sample(pool, need)
                # One position can carry SEVERAL tags: the same document may appear in two gold
                # pairs, or be both gold and a listed hard negative. A dict keyed by position
                # would silently drop all but the last tag and then KeyError on remap, so tags
                # accumulate in a list per position.
                marks = [[] for _ in docs]
                for k, i in enumerate(flat_gold(r, cfg)):
                    marks[i].append(("gold", k))
                for f in cfg["index_fields"]:
                    for k, i in enumerate(r.get(f) or []):
                        marks[i].append((f, k))
                tagged = [(d, marks[i]) for i, d in enumerate(docs)]
                tagged += [(d, []) for d in extra]
                rng.shuffle(tagged)
                new_docs = [d for d, _ in tagged]
                pos = {tag: j for j, (_, tags) in enumerate(tagged) for tag in tags}

                new = dict(r)
                new["documents"] = new_docs
                g = r.get("gold_doc_indices") or []
                if cfg["pairs"]:
                    k = 0
                    remapped = []
                    for pair in g:
                        remapped.append([pos[("gold", k + j)] + cfg["base"]
                                         for j in range(len(pair))])
                        k += len(pair)
                    new["gold_doc_indices"] = remapped
                else:
                    new["gold_doc_indices"] = [pos[("gold", k)] + cfg["base"]
                                               for k in range(len(g))]
                for f in cfg["index_fields"]:
                    if r.get(f):
                        new[f] = [pos[(f, k)] for k in range(len(r[f]))]
                new["source"] = r.get("source", "")
                new["_expanded_from"] = os.path.basename(args.src)
                new["_measured_target_tokens"] = target
                fh.write(json.dumps(new) + "\n")
        # verify the written file round-trips
        chk = [json.loads(l) for l in open(out)]
        assert len(chk) == len(rows), f"{out}: wrote {len(chk)} of {len(rows)}"
        for a, b in zip(rows[:50], chk[:50]):
            ga = sorted(doc_text(a["documents"][i]) for i in flat_gold(a, cfg))
            gb = sorted(doc_text(b["documents"][i]) for i in flat_gold(b, cfg))
            assert ga == gb, f"{out}: gold text changed after remap"
        print(f"[expand] wrote {out}  n_docs={n_target} (+{need} fillers/ex)  "
              f"gold-preservation verified on 50 examples", flush=True)


if __name__ == "__main__":
    main()
