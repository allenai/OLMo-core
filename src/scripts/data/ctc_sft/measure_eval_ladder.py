"""
Measure the CTC eval's own ladder, so training data can be built to match it rung by rung.

``ctc``'s training builder sizes every rung from ``ladders.docs_for_rung``. The eval data that
olmo-eval actually grades (``PrasannSinghal/ctc-suite-eval``) was not built from that table for
every task -- strmatch's eval "2k" holds 38 documents where the table says 72, rerank's 23 where it
says 15 -- so a train set built from the table is not IID with its eval, however well each row is
formatted. The eval is the gold standard, so this reads it and writes the calibration the builder
consumes (``build_ctc_sft.py --calibration``):

* ``n_docs``: the median document count of the eval's rows at that rung (it is constant per rung
  for every ladder measured; min/max are recorded so a spread would show).
* ``set``: extra ``ctc-data build -C`` overrides. oolong's rung is a token budget that the
  generator draws UNIFORMLY in ``[300, rung]`` (median = half the rung); the eval's rows sit at the
  full rung, so its budget is pinned with ``min_tokens`` = rung.
* 256k: the eval's own rows there run 251k-278k rendered tokens, past the 262,144 training window
  for some tasks. The count is scaled down just enough for its longest measured row to fit, and
  the cell is marked ``capped`` with the ratio so the deviation is on record.

    python src/scripts/data/ctc_sft/measure_eval_ladder.py --out src/scripts/data/ctc_sft/eval_calibration.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, *[os.pardir] * 4))
sys.path.insert(0, os.path.join(REPO, "debug", "ctc_sft_setA"))
sys.path.insert(0, os.path.join(REPO, "src"))

import audit_train_eval_iid as A  # noqa: E402  (row loading, ROSTER mapping)

BUCKET_TOKENS = {b: int(b[:-1]) * 1024 for b in A.BUCKETS}
#: rendered tokens a training row may use: the 262,144 window minus a margin for the chat wrapper
#: and document markers the converter adds on top of the rendered prompt
#: ``-C`` overrides that hold at every rung, read off the eval rows' own metadata.
CONSTANT_SET = {
    # every eval row's _meta is {num_hardneg, relation, span_len, str_len}: the pre-migration
    # construction, which ctc documents as exactly num_scattered=0 (training defaulted to 6)
    "strmatch": ["num_scattered=0"],
}
WINDOW = 262_144
FIT = WINDOW - 4_096


def _rendered_tokens(tok, ex: dict, spec: str) -> int:
    from olmo_core.data.corpus_reasoning_prompts import build_prompt

    p, a = build_prompt(ex, task=spec, query_position="both", use_alpaca=False, cot_mode="none",
                        use_titles=False)
    s = tok.apply_chat_template([{"role": "user", "content": p},
                                 {"role": "assistant", "content": a}], tokenize=False)
    # count documents' boundary markers too (2 per document in the dense shards)
    return len(tok(s, add_special_tokens=False)["input_ids"]) + 2 * len(ex.get("documents") or [])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--roster", default=os.path.join(REPO, "debug", "ctc_sft_setA",
                                                     "olmo_eval_roster.json"))
    ap.add_argument("--sample", type=int, default=40)
    ap.add_argument("--tokenizer", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--tasks", nargs="*", default=list(A.TASK_TO_ROW))
    ap.add_argument("--out", default=os.path.join(HERE, "eval_calibration.json"))
    ap.add_argument("--merge", action="store_true",
                    help="update only --tasks in an existing --out, keeping every other task")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    R = json.load(open(args.roster))
    sys.path.insert(0, os.path.join(HERE))
    import build_ctc_sft as B

    render_spec = {t.name: t.spec for t in B.SET_A}
    out = {"source": R["hf_dataset"], "olmo_eval_commit": R["olmo_eval_commit"],
           "window": WINDOW, "tasks": {}}
    for task in args.tasks:
        row = R["roster"][A.TASK_TO_ROW[task]]
        cells = {}
        for b in A.BUCKETS:
            if "r" + b not in row["rungs"]:
                continue
            rows = A._eval_rows(R["hf_dataset"], row["subset"], "r" + b, args.sample)
            if not rows:
                continue
            nd = [len(r.get("documents") or []) for r in rows]
            cell = {"eval_n_docs_min": min(nd), "eval_n_docs_med": int(statistics.median(nd)),
                    "eval_n_docs_max": max(nd), "eval_rows_measured": len(rows)}
            if task == "oolong":
                # the scaling param IS the token budget (``docs_for_rung`` returns it), so the
                # budget goes through n_docs like every other task; min_tokens pins the draw to it
                t = BUCKET_TOKENS[b]
                cell["n_docs"] = t
                cell["set"] = [f"min_tokens={t}", f"max_context={max(131_072, t)}"]
            elif task == "rerank":
                # the eval keeps a fixed candidate set of CE-scored documents at every rung and
                # pads to the rung with unscored foreign passages (ce_scores None), which the
                # scorer excludes from the target. ctc builds the scored set; the builder fills.
                scored = [sum(x is not None for x in (r.get("ce_scores") or [])) for r in rows]
                cell["n_docs"] = int(statistics.median(scored))
                cell["fill_to"] = cell["eval_n_docs_med"]
            else:
                cell["n_docs"] = cell["eval_n_docs_med"]
            if b == "256k":
                longest = max(_rendered_tokens(tok, r, render_spec[task]) for r in rows[:10])
                cell["eval_rendered_tokens_max"] = longest
                if task == "oolong":
                    # the budget is tokens of CONTENT; the eval's own longest 256k row is the
                    # size to match, and pinning the budget to the full 262,144 overflows the
                    # window once the chat wrapper is added
                    t = min(cell["n_docs"], longest, FIT)
                    cell["n_docs"] = t
                    cell["set"] = [f"min_tokens={t}", f"max_context={max(131_072, t)}"]
                elif longest > FIT:
                    ratio = FIT / longest
                    cell["capped"] = round(ratio, 4)
                    if "fill_to" in cell:
                        cell["fill_to"] = int(cell["fill_to"] * ratio)
                    else:
                        cell["n_docs"] = int(cell["n_docs"] * ratio)
                    if task == "oolong":
                        t = cell["n_docs"]
                        cell["set"] = [f"min_tokens={t}", f"max_context={max(131_072, t)}"]
            if task in CONSTANT_SET:
                cell["set"] = cell.get("set", []) + CONSTANT_SET[task]
            cells[b] = cell
            print(f"{task:<17}{b:>5}  eval n_docs {cell['eval_n_docs_min']}/"
                  f"{cell['eval_n_docs_med']}/{cell['eval_n_docs_max']}"
                  + (f"  -> n_docs {cell.get('n_docs')}" if "n_docs" in cell else "")
                  + (f"  fill_to {cell['fill_to']}" if "fill_to" in cell else "")
                  + (f"  set {cell['set']}" if "set" in cell else "")
                  + (f"  CAPPED x{cell['capped']} (eval row {cell['eval_rendered_tokens_max']:,} tok)"
                     if "capped" in cell else ""), flush=True)
        out["tasks"][task] = cells
    if args.merge and os.path.exists(args.out):
        prev = json.load(open(args.out))
        prev["tasks"].update(out["tasks"])
        out = prev
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
