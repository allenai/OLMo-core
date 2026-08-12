"""Decide whether the xlong5 (q35-4b) training shards are in-distribution for the v2 evals.

The q35-4b-dense / q35-4b-fastcomplm checkpoints in results-hub train on `xlong5_2k256k_qwen35`,
NOT on `single_task_ladders_v2` where the contradiction and outlier mismatches were established.
Those shards record NO source jsonl, so provenance has to come from the DATA. This decodes the
tokenized shard itself -- the ground truth of what was trained on -- and measures the same two
signatures used everywhere else in this investigation:

  contradiction: gold-pair word-Jaccard.
      realistic -> median ~0.31, p99/max 0.500, 0.0% above 0.5   (matches the v3 eval)
      both      -> median ~0.39, mean ~0.50, ~38% above 0.5      (matches the v2 eval)

  outlier: the category count K.
      The target enumerates every majority topic ("Most passages are about A, B, C ... and the
      outliers are about X"), so K is recoverable from the ANSWER even though document titles are
      hidden in the prompt. The v2 eval ladder is scale-K: K = 3/7/13/25 at n = 22/55/110/220.
      The single_task_ladders_v2 training file was K = 2-10 (median 6.5 at n~220) -- OOD.

Instances are EOS-separated; the answer is exactly the span where labels_mask is True.
"""

import argparse
import json
import re
import statistics
from collections import Counter

import numpy as np

EOS = 248044  # qwen3_5 marker set, per the shard metadata


def iter_instances(tokens, mask, limit):
    """Yield (prompt_ids, answer_ids) per EOS-separated instance."""
    starts = np.flatnonzero(tokens == EOS)
    prev = 0
    n = 0
    for e in starts:
        if n >= limit:
            return
        seg_t, seg_m = tokens[prev:e], mask[prev:e]
        prev = e + 1
        if seg_t.size < 32:
            continue
        ans_idx = np.flatnonzero(seg_m)
        if ans_idx.size == 0:
            continue
        a0 = ans_idx[0]
        yield seg_t[:a0], seg_t[a0:]
        n += 1


def toks(s):
    return set(re.findall(r"[a-z0-9]+", s.lower()))


def jaccard(a, b):
    A, B = toks(a), toks(b)
    return len(A & B) / max(1, len(A | B))


def parse_docs(prompt_text):
    """Documents render as `[i] text`; return {id: text}."""
    out = {}
    for m in re.finditer(r"\[(\d+)\]\s*(.*?)(?=\n\s*\[\d+\]|\Z)", prompt_text, re.S):
        out[int(m.group(1))] = m.group(2).strip()
    return out


def audit_contradiction(tok, tokens, mask, limit):
    ovl = []
    for p_ids, a_ids in iter_instances(tokens, mask, limit):
        docs = parse_docs(tok.decode(p_ids, skip_special_tokens=True))
        ans = tok.decode(a_ids, skip_special_tokens=True)
        pairs = re.findall(r"\[\s*(\d+)\s*,\s*(\d+)\s*\]", ans)
        for a, b in pairs:
            a, b = int(a), int(b)
            if a in docs and b in docs:
                ovl.append(jaccard(docs[a], docs[b]))
    if not ovl:
        return None
    ovl.sort()
    return {
        "gold_pairs": len(ovl),
        "median": round(statistics.median(ovl), 4),
        "mean": round(statistics.mean(ovl), 4),
        "p90": round(ovl[int(0.90 * len(ovl))], 4),
        "p99": round(ovl[int(0.99 * len(ovl))], 4),
        "max": round(ovl[-1], 4),
        "frac_gt_0.5": round(sum(1 for x in ovl if x > 0.5) / len(ovl), 4),
    }


def audit_outlier(tok, tokens, mask, limit):
    """K from the answer's topic list; n from the prompt's document count."""
    rows = []
    for p_ids, a_ids in iter_instances(tokens, mask, limit):
        docs = parse_docs(tok.decode(p_ids, skip_special_tokens=True))
        ans = tok.decode(a_ids, skip_special_tokens=True)
        m = re.search(r"most passages are about (.*?) and the outliers are about",
                      ans, re.I | re.S)
        if not m or not docs:
            continue
        majors = [t.strip() for t in re.split(r",| and ", m.group(1)) if t.strip()]
        rows.append((len(docs), len(majors) + 1))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["contradiction", "outlier"])
    ap.add_argument("--tokens", required=True)
    ap.add_argument("--mask", required=True)
    ap.add_argument("--tokenizer", default="/scratch/users/prasann/hf_models/Qwen3.5-4B-Base")
    ap.add_argument("--limit", type=int, default=120)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    tokens = np.load(args.tokens, mmap_mode="r")
    mask = np.load(args.mask, mmap_mode="r")
    print(f"[audit] {args.task}: {tokens.shape[0]:,} tokens in this part", flush=True)

    if args.task == "contradiction":
        res = audit_contradiction(tok, tokens, mask, args.limit)
        print(json.dumps(res, indent=2))
        if res:
            both_like = res["frac_gt_0.5"] > 0.15
            print(f"\nVERDICT: looks like **{'BOTH' if both_like else 'REALISTIC'}** mode")
            print("  realistic ref: median 0.306 mean 0.307 p99 0.500 max 0.500 frac>0.5 0.000")
            print("  both ref     : median 0.388 mean 0.501 p90 0.943 max 1.000 frac>0.5 0.381")
    else:
        rows = audit_outlier(tok, tokens, mask, args.limit)
        print(f"[audit] parsed {len(rows)} outlier answers")
        print("\n  n band      train K (med, min-max)     eval K (v2 ladder)")
        for lo, hi, ev in [(14, 30, "3 (3-5)"), (45, 70, "7 (5-11)"),
                           (95, 125, "13 (10-17)"), (190, 260, "25 (23-28)")]:
            ks = [k for n, k in rows if lo <= n <= hi]
            if ks:
                print(f"  n={lo}-{hi:<5} {statistics.median(ks):<6} ({min(ks)}-{max(ks)})"
                      f"{'':<12} {ev}")
            else:
                print(f"  n={lo}-{hi:<5} (no samples){'':<12} {ev}")
        allk = [k for _, k in rows]
        if allk:
            print(f"\n  overall K: median {statistics.median(allk)}, "
                  f"range {min(allk)}-{max(allk)}")
            print(f"  n_docs   : {Counter(n // 50 * 50 for n, _ in rows).most_common(6)}")


if __name__ == "__main__":
    main()
