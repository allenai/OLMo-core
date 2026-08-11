"""How much prefill does a fast rung actually save, under each ``query_position``?

Shared documents are not shared tokens. The reuse a fast rung delivers is the longest common
*token* prefix within a corpus group, and that depends on where the question is rendered:

``both``   ``{questions}\\n\\n{documents}\\n\\n{questions}`` -- the per-query question comes first, so
           a byte-identical document block is not a prefix and there is almost nothing to reuse.
``after``  ``{documents}\\n\\n{questions}`` -- the corpus *is* the prefix.

This measures both on real built files, so the payoff of training with ``after`` is a number rather
than an expectation. No GPU: it only tokenizes.

    PYTHONPATH=ctc/src python debug/fast_bundle/measure_reuse.py debug/fast_bundle/out/*/*.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

TASK_OF = {"contradiction": "contradiction", "nq": "nq", "rerank": "rerank", "oolong": "oolong"}


def task_from(path: Path) -> str:
    for part in path.parts:
        if part in TASK_OF:
            return TASK_OF[part]
    raise SystemExit(f"cannot tell which task {path} belongs to")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-4B")
    ap.add_argument("--limit", type=int, default=60, help="rows per file (tokenizing is the cost)")
    args = ap.parse_args()

    import ctc.tasks
    from ctc.eval.prefix_cache import group_by_corpus, longest_common_token_prefix
    from ctc.format import registry
    from transformers import AutoTokenizer

    ctc.tasks.load_all()
    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    print(f"{'file':<34} {'position':<8} {'groups':>6} {'shared tok':>11} {'of prompt':>10} "
          f"{'prefill left':>13}")
    for path in [Path(p) for p in args.files]:
        spec = registry.get(task_from(path))
        rows: List[dict] = []
        with open(path) as f:
            for line in f:
                rows.append(json.loads(line))
                if len(rows) >= args.limit:
                    break

        groups: Dict[str, List[int]] = group_by_corpus(rows)
        for position in ("both", "after"):
            prompts = [spec.build_prompt(r, query_position=position) for r in rows]
            ids = [tok(p, add_special_tokens=False)["input_ids"] for p in prompts]

            total = sum(len(i) for i in ids)
            # What a cache-reusing runner feeds: each group's shared prefix once, plus every row's
            # own remainder.
            fed = 0
            shared_tokens = 0
            for members in groups.values():
                seqs = [ids[i] for i in members]
                common = longest_common_token_prefix(seqs)
                shared_tokens += common
                fed += common + sum(len(s) - common for s in seqs)
            mean_common = shared_tokens / max(1, len(groups))
            mean_len = total / max(1, len(ids))
            print(f"{path.name:<34} {position:<8} {len(groups):>6} {mean_common:>11,.0f} "
                  f"{mean_common / mean_len:>9.1%} {fed / max(1, total):>12.1%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
