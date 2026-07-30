"""Port HELMET-style long-document QA into our `qa` task.

NarrativeQA (the PDF's named example): single long document (book / movie
script) + a question with ~2 free-text reference answers. We map each to:
    documents = [{"title": None, "text": <story, end-truncated to target len>}]
    queries   = [question]
    answers   = [ref1, ref2]
and bucket by target length (HELMET truncates the source doc to L tokens; we
approximate with ~4 chars/token, keeping the document head). Scored by the
existing `--task qa` (token-F1 / substring / EM over the references) — a no-LLM
proxy for HELMET's GPT-4o judge; the local Qwen judge can be swapped in later.

Also supports ∞Bench multiple-choice (`--source infbench_mc`) where the answer
is one of given options (scored by substring/EM via `qa`).

Caches go to /data/prasann (scratch quota) — set in jobs/gen_helmet_qa.sh.

Usage:
    python scripts/data/generate_helmet_qa_data.py --source narrativeqa \\
        --split validation --num-examples 300 --lengths 8000,32000,131072
"""

import argparse
import json
import random


def _truncate_chars(text, target_tokens, chars_per_tok=4):
    budget = target_tokens * chars_per_tok
    return text[:budget]


def load_narrativeqa(split, n, rng):
    from datasets import load_dataset
    ds = load_dataset("deepmind/narrativeqa", split=split)
    idx = list(range(len(ds)))
    rng.shuffle(idx)
    out = []
    seen_q = set()
    for i in idx:
        r = ds[i]
        q = r["question"]["text"]
        story = r["document"]["text"]
        key = (r["document"]["id"], q)
        if key in seen_q:
            continue
        seen_q.add(key)
        out.append({"story": story, "question": q,
                    "answers": [a["text"] for a in r["answers"]]})
        if len(out) >= n:
            break
    return out


def load_infbench_mc(split, n, rng):
    from datasets import load_dataset
    ds = load_dataset("xinrongzhang2022/InfiniteBench", "longbook_choice_eng",
                      split=split)
    idx = list(range(len(ds)))
    rng.shuffle(idx)
    out = []
    for i in idx[:n]:
        r = ds[i]
        opts = r.get("options") or []
        ans = r["answer"]
        ans = ans[0] if isinstance(ans, list) else ans
        q = (r["input"] + "\n\nOptions:\n"
             + "\n".join(f"{chr(65+j)}. {o}" for j, o in enumerate(opts)))
        out.append({"story": r["context"], "question": q, "answers": [str(ans)]})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["narrativeqa", "infbench_mc"],
                    default="narrativeqa")
    ap.add_argument("--split", default="validation")
    ap.add_argument("--num-examples", type=int, default=300)
    ap.add_argument("--lengths", default="8000,32000,131072",
                    help="comma-separated target token lengths (the scaling axis)")
    ap.add_argument("--output-dir", default="data")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    loader = {"narrativeqa": load_narrativeqa,
              "infbench_mc": load_infbench_mc}[args.source]
    items = loader(args.split, args.num_examples, rng)
    print(f"{args.source}: loaded {len(items)} items")

    lengths = [int(x) for x in args.lengths.split(",")]
    for L in lengths:
        exs = []
        for it in items:
            exs.append({
                "documents": [{"title": None,
                               "text": _truncate_chars(it["story"], L)}],
                "queries": [it["question"]],
                "answers": it["answers"],
                "gold_doc_indices": [0],
                "source": f"helmet_{args.source}",
                "_meta": {"target_len": L},
            })
        path = f"{args.output_dir}/helmet_{args.source}_{args.split}_L{L}.jsonl"
        with open(path, "w") as f:
            for ex in exs:
                f.write(json.dumps(ex) + "\n")
        avg = sum(len(e["documents"][0]["text"]) for e in exs) / max(1, len(exs))
        print(f"  L={L}: {len(exs)} ex, avg {avg:.0f} chars -> {path}")


if __name__ == "__main__":
    main()
