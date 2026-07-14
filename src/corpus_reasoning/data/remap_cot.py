"""Remap chain-of-thought from a source CoT file onto a target file at a
different k size.

The CoT for a given (question, gold-doc-set) is identical across k sizes —
only the [N] doc-id references shift, since gold docs land at different
positions in different document lists. This script matches by question,
then rebuilds [N] references by looking up each source doc's text in the
target's document list.

Usage:
    python scripts/data/remap_cot.py \\
        --source data/hotpotqa_train_k100_bridge_unified_hn98_2000_cot.jsonl \\
        --target data/hotpotqa_train_k20_bridge_unified_hn18_2000.jsonl \\
        --output data/hotpotqa_train_k20_bridge_unified_hn18_2000_cot.jsonl
"""
import argparse
import json
import re
import sys
from pathlib import Path

from corpus_reasoning.lib.io import load_jsonl, save_jsonl


def doc_key(doc):
    return (doc.get("title") or "", doc["text"])


def remap_cot(cot_text, src_docs, tgt_docs):
    """Rewrite [N] references in cot_text from source 1-indexed positions to
    target 1-indexed positions, by matching documents via (title, text).

    References pointing at docs absent from the target are left untouched.
    Returns (new_cot, n_remapped, n_unresolved).
    """
    if not cot_text:
        return "", 0, 0
    tgt_key_to_pos = {doc_key(d): i + 1 for i, d in enumerate(tgt_docs)}
    src_to_tgt = {}
    for i, d in enumerate(src_docs):
        pos = tgt_key_to_pos.get(doc_key(d))
        if pos is not None:
            src_to_tgt[i + 1] = pos
    counters = {"remap": 0, "miss": 0}

    def sub(m):
        idx = int(m.group(1))
        new = src_to_tgt.get(idx)
        if new is None:
            counters["miss"] += 1
            return m.group(0)
        counters["remap"] += 1
        return f"[{new}]"

    new_cot = re.sub(r"\[(\d+)\]", sub, cot_text)
    return new_cot, counters["remap"], counters["miss"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True,
                   help="JSONL with chain_of_thought populated")
    p.add_argument("--target", required=True,
                   help="JSONL to receive remapped CoT")
    p.add_argument("--output", required=True,
                   help="Output path (target with chain_of_thought field added)")
    args = p.parse_args()

    src_by_q = {}
    for ex in load_jsonl(args.source):
        if ex.get("chain_of_thought"):
            src_by_q[ex["queries"][0]] = ex
    print(f"loaded {len(src_by_q)} source examples with CoT")

    out = []
    n_match = n_unmatched = n_no_src_cot = 0
    n_total_remap = n_total_miss = 0
    for ex in load_jsonl(args.target):
        q = ex["queries"][0]
        src = src_by_q.get(q)
        if src is None:
            n_unmatched += 1
            ex["chain_of_thought"] = ""
            out.append(ex)
            continue
        src_cot = src.get("chain_of_thought", "")
        if not src_cot:
            n_no_src_cot += 1
            ex["chain_of_thought"] = ""
            out.append(ex)
            continue
        new_cot, n_r, n_m = remap_cot(src_cot, src["documents"], ex["documents"])
        ex["chain_of_thought"] = new_cot
        n_total_remap += n_r
        n_total_miss += n_m
        n_match += 1
        out.append(ex)

    save_jsonl(args.output, out)
    print(f"wrote {args.output}")
    print(f"  matched + remapped: {n_match}")
    print(f"  unmatched (question not in source): {n_unmatched}")
    print(f"  matched but source had no CoT: {n_no_src_cot}")
    print(f"  total [N] tokens remapped: {n_total_remap}, "
          f"unresolved (doc missing in target): {n_total_miss}")


if __name__ == "__main__":
    sys.exit(main())
