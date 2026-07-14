"""Port HELMET long-document summarization into our `summarization` task.

Multi-LexSum (default): a set of legal-case documents + a professional reference
summary (HELMET uses the "long" granularity). ∞Bench Sum (`--source infbench`):
a novel + reference summary. Each → :
    documents = [{"text": source_doc}, ...]   (concatenated, end-truncated to L)
    queries   = ["Summarize the document(s) above."]
    answers   = [reference summary]
Scored by `--task summarization` (ROUGE-1/L F1; the faithful LLM-judge
atomic-claim F1 is scripts/eval/score_summarization_claims.py via local Qwen).

Caches → /data/prasann (set in jobs/gen_helmet_summ.sh).

Usage:
    python scripts/data/generate_helmet_summ_data.py --source multilexsum \\
        --split validation --num-examples 200 --lengths 16000,64000
"""

import argparse
import json
import random


def _truncate(text, target_tokens, cpt=4):
    return text[: target_tokens * cpt]


def load_multilexsum(split, n, rng):
    from datasets import load_dataset
    ds = load_dataset("allenai/multi_lexsum", name="v20230518", split=split)
    idx = list(range(len(ds)))
    rng.shuffle(idx)
    out = []
    for i in idx:
        r = ds[i]
        summ = r.get("summary/long")
        sources = r.get("sources") or []
        if not summ or not sources:
            continue
        out.append({"sources": sources, "summary": summ})
        if len(out) >= n:
            break
    return out


def load_govreport(split, n, rng):
    """GovReport long-document summarization (parquet; no loading script).
    A government report -> a professional summary. Long single docs."""
    from datasets import load_dataset
    ds = load_dataset("ccdv/govreport-summarization", split=split)
    cols = ds.column_names
    doc_col = "report" if "report" in cols else cols[0]
    sum_col = "summary" if "summary" in cols else cols[1]
    idx = list(range(len(ds)))
    rng.shuffle(idx)
    out = []
    for i in idx[:n]:
        r = ds[i]
        if r[doc_col] and r[sum_col]:
            out.append({"sources": [r[doc_col]], "summary": r[sum_col]})
    return out


def load_infbench_sum(split, n, rng):
    from datasets import load_dataset
    ds = load_dataset("xinrongzhang2022/InfiniteBench", "longbook_sum_eng",
                      split=split)
    idx = list(range(len(ds)))
    rng.shuffle(idx)
    out = []
    for i in idx[:n]:
        r = ds[i]
        out.append({"sources": [r["context"]], "summary": r["answer"]
                    if isinstance(r["answer"], str) else r["answer"][0]})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["govreport", "multilexsum", "infbench"],
                    default="govreport")
    ap.add_argument("--split", default="validation")
    ap.add_argument("--num-examples", type=int, default=200)
    ap.add_argument("--lengths", default="16000,64000")
    ap.add_argument("--output-dir", default="data")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    loader = {"govreport": load_govreport, "multilexsum": load_multilexsum,
              "infbench": load_infbench_sum}[args.source]
    items = loader(args.split, args.num_examples, rng)
    print(f"{args.source}: loaded {len(items)} items")

    for L in [int(x) for x in args.lengths.split(",")]:
        exs = []
        for it in items:
            combined = "\n\n".join(it["sources"])
            exs.append({
                "documents": [{"title": None, "text": _truncate(combined, L)}],
                "queries": ["Summarize the document(s) above."],
                "answers": [it["summary"]],
                "gold_doc_indices": [0],
                "source": f"helmet_summ_{args.source}",
                "_meta": {"target_len": L},
            })
        path = f"{args.output_dir}/helmet_summ_{args.source}_{args.split}_L{L}.jsonl"
        with open(path, "w") as f:
            for ex in exs:
                f.write(json.dumps(ex) + "\n")
        avg_d = sum(len(e["documents"][0]["text"]) for e in exs) / max(1, len(exs))
        avg_s = sum(len(e["answers"][0]) for e in exs) / max(1, len(exs))
        print(f"  L={L}: {len(exs)} ex, doc avg {avg_d:.0f} chars, "
              f"summary avg {avg_s:.0f} chars -> {path}")


if __name__ == "__main__":
    main()
