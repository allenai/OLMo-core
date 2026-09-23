"""
Per-(task, context-bucket) TOKEN counts for a setA SFT build -- exactly what the shards hold.

Every row is tokenized through the converter's own ``tokenize_example`` (the function that writes
the shards: chat template, document boundary markers, EOS, ``--emit dense``), with each task's spec
and ``chunk_by`` taken from the builder's roster. So a count here is a count of training tokens, not
of a re-rendering that could drift (an earlier version of this script rendered through the Alpaca
wrapper and mapped hotpotqa to ``cot_retrieval``, neither of which the shards use).

Rows longer than ``--seq-len`` are reported as dropped, as the converter drops them.

    python debug/ctc_sft_setA/count_setA_tokens.py \\
        --root /weka/oe-training-default/ai2-llm/checkpoints/prasanns/ctc_sft_sets/setA_max20_evaliid \\
        --out .../setA_token_counts.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "src", "scripts", "data"))
sys.path.insert(0, os.path.join(REPO, "src", "scripts", "data", "ctc_sft"))

BUCKETS = ["2k", "4k", "8k", "16k", "32k", "64k", "128k", "256k"]
_W: dict = {}


def _init(tokenizer: str, marker_set: str, seq_len: int, query_position: str) -> None:
    from transformers import AutoTokenizer

    from olmo_core.data.document_chunk_landmark import reserved_ids

    _W.update(tok=AutoTokenizer.from_pretrained(tokenizer), ids=reserved_ids(marker_set),
              seq_len=seq_len, qpos=query_position)


def _count(job):
    """One (task, bucket) file -> row / token / drop counts."""
    from convert_unified_to_document_landmark import tokenize_example

    task, bucket, spec, chunk_by, path = job
    rows = tokens = dropped = loss = 0
    longest = 0
    with open(path) as f:
        for line in f:
            ex = json.loads(line)
            ex = ex["ex"] if "ex" in ex and "documents" not in ex else ex
            rows += 1
            out = tokenize_example(
                _W["tok"], ex, spec, emit="dense", query_position=_W["qpos"], cot_mode="none",
                mem_freq=63, seq_len=10**9, chunk_by=chunk_by, item_regex=r"\|\|",
                use_titles=False, ids_set=_W["ids"])
            if out is None:
                dropped += 1
                continue
            n = int(out[0].shape[0])
            if n > _W["seq_len"]:
                dropped += 1
                continue
            tokens += n
            loss += int(out[1].sum())
            longest = max(longest, n)
    return task, bucket, {"rows": rows, "kept": rows - dropped, "dropped": dropped,
                          "tokens": tokens, "loss_tokens": loss, "max_len": longest}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="set root holding per_task/_b<bucket>/<task>/")
    ap.add_argument("--tokenizer", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--marker-set", default="qwen3_5")
    ap.add_argument("--seq-len", type=int, default=262_144)
    ap.add_argument("--query-position", default="both")
    ap.add_argument("--workers", type=int, default=min(64, os.cpu_count() or 8))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import build_ctc_sft as B

    jobs = []
    for t in B.SET_A:
        for b in BUCKETS:
            p = os.path.join(args.root, "per_task", f"_b{b}", t.name, "train.jsonl")
            if os.path.exists(p) and os.path.getsize(p):
                jobs.append((t.name, b, t.spec, t.chunk_by, p))
    # longest files first so the pool does not finish on one straggler
    jobs.sort(key=lambda j: -os.path.getsize(j[4]))
    print(f"counting {len(jobs)} (task, bucket) files with {args.workers} workers", flush=True)

    t0, res = time.time(), {}
    with ProcessPoolExecutor(args.workers, initializer=_init,
                             initargs=(args.tokenizer, args.marker_set, args.seq_len,
                                       args.query_position)) as pool:
        for i, (task, bucket, r) in enumerate(pool.map(_count, jobs), 1):
            res.setdefault(task, {})[bucket] = r
            if i <= 2 or i % 10 == 0 or i == len(jobs):
                print(f"  [{i}/{len(jobs)}] {task}@{bucket} {r['tokens']:,} tok "
                      f"({time.time() - t0:.0f}s)", flush=True)

    json.dump({"root": args.root, "tokenizer": args.tokenizer, "seq_len": args.seq_len,
               "counts": res}, open(args.out, "w"), indent=1)

    def m(x):
        return f"{x / 1e6:.1f}M" if x else "-"

    print(f"\n{'task':<17}" + "".join(f"{b:>8}" for b in BUCKETS) + f"{'TOTAL':>9}{'rows':>9}")
    col = {b: 0 for b in BUCKETS}
    for t in B.SET_A:
        row = res.get(t.name, {})
        tot = sum(v["tokens"] for v in row.values())
        for b in BUCKETS:
            col[b] += row.get(b, {}).get("tokens", 0)
        print(f"{t.name:<17}" + "".join(f"{m(row.get(b, {}).get('tokens', 0)):>8}" for b in BUCKETS)
              + f"{m(tot):>9}{sum(v['kept'] for v in row.values()):>9,}")
    print(f"{'TOTAL':<17}" + "".join(f"{m(col[b]):>8}" for b in BUCKETS)
          + f"{m(sum(col.values())):>9}")
    drops = {f"{t}@{b}": v["dropped"] for t, r in res.items() for b, v in r.items() if v["dropped"]}
    print(f"dropped (> seq-len {args.seq_len}): {drops or 'none'}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
