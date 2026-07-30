"""Port the Oolong synthetic benchmark into our unified `oolong` task.

Oolong (arXiv:2511.02817) gives a long context of N labeled classification items
(each tagged with a date + user-id) and asks distributional questions —
counting (most/least common label), user (most frequent user), timeline (when
did label A overtake B). The model must analyze EVERY item and aggregate, not
retrieve. `oolong-synth` ships a built-in length ladder (context_len ∈
1K..128K), which we keep as the corpus-size scaling axis.

We use `context_window_text` (items WITHOUT gold labels — the model classifies +
aggregates, the hard regime). Each row → one example:
    documents = [{"text": context_window_text}]   (shown verbatim, no wrapper)
    queries   = [question]                         (carries its own format spec)
    answers   = [primary gold value]
    _meta     = {answer_type, task_group, gold_list, context_len}

Scored by `--task oolong`: exact-match for label/user/comparison/date, numeric
partial credit 0.75^|err|, set-overlap F1 for list answers.

Needs HuggingFace `datasets` → run via sbatch (jobs/gen_oolong.sh).

Usage:
    python scripts/data/generate_oolong_data.py --split validation --output-dir data
"""

import argparse
import ast
import re
from collections import defaultdict


def parse_gold(answer, answer_type):
    """Return a list of clean gold value strings."""
    val = answer
    try:
        lit = ast.literal_eval(answer)
        if isinstance(lit, (list, tuple)):
            val = list(lit)
        else:
            val = [lit]
    except (ValueError, SyntaxError):
        val = [answer]
    if "NUMERIC" in (answer_type or ""):
        nums = re.findall(r'-?\d+\.?\d*', str(val[0]))
        return [nums[0]] if nums else [str(val[0]).strip()]
    return [str(v).strip() for v in val]


def main():
    import json
    from datasets import load_dataset
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="validation")
    ap.add_argument("--config", default="oolong-synth")
    ap.add_argument("--max-ctx", type=int, default=0,
                    help="if >0, only write buckets with context_len <= this "
                         "(skip the huge long-context buckets)")
    ap.add_argument("--output-dir", default="data")
    args = ap.parse_args()

    name = "oolongbench/oolong-synth"
    ds = load_dataset(name)
    split = args.split if args.split in ds else list(ds.keys())[0]
    rows = ds[split]
    print(f"{name} [{split}]: {len(rows)} rows")

    by_len = defaultdict(list)
    for r in rows:
        gold_list = parse_gold(r["answer"], r.get("answer_type"))
        ex = {
            "documents": [{"text": r["context_window_text"]}],
            "queries": [r["question"]],
            "answers": [gold_list[0]],
            "gold_doc_indices": [],
            "source": f"oolong_{r.get('task_group', 'synth')}",
            "_meta": {
                "answer_type": r.get("answer_type"),
                "task_group": r.get("task_group"),
                "gold_list": gold_list,
                "context_len": r.get("context_len"),
                "dataset": r.get("dataset"),
            },
        }
        by_len[r.get("context_len")].append(ex)

    for clen, exs in sorted(by_len.items(), key=lambda kv: (kv[0] or 0)):
        if args.max_ctx and (clen or 0) > args.max_ctx:
            continue
        path = f"{args.output_dir}/oolong_{split}_synth_ctx{clen}.jsonl"
        with open(path, "w") as f:
            for ex in exs:
                f.write(json.dumps(ex) + "\n")
        groups = defaultdict(int)
        for ex in exs:
            groups[ex["_meta"]["task_group"]] += 1
        print(f"  ctx={clen}: {len(exs)} ex {dict(groups)} -> {path}")


if __name__ == "__main__":
    main()
