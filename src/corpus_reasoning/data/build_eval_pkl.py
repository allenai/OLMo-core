"""Build per-query mdata/qdata eval pkls from existing task JSONLs.

Output layout (per eval set <name>, one pair of files per query):
  data/eval_pkls/<name>/
      mdata_000.pkl   list[str]   - this query's full corpus
      qdata_000.pkl   list[dict]  - length-1 list with one entry:
          { "query": str,
            "answer": str,
            "reference_list": list[str] }   (gold passages, all present in mdata)
      mdata_001.pkl   ...
      qdata_001.pkl
      ...

Each (mdata_<id>, qdata_<id>) pair is a self-contained eval instance.

Subcommands:
  contradiction-from-jsonl
      One eval instance per gold contradiction pair in the source JSONL.
      reference_list = [claim_A, claim_B]; corpus = the full source row's docs.

  contradiction-stitch
      One eval instance per gold pair; corpus is the seed row's docs padded
      with fillers pooled from other rows until ~--target-tokens is reached.
      Used for the 1M-token scale.

  grouping-from-jsonl
      One eval instance per source row; reference_list = first abstract per
      gold cluster in corpus order; answer = {label: [1-idx positions]}.

  grouping-generate-from-compact
      Naturalistic grouping: sample n_docs abstracts uniformly, let k emerge
      from the distinct L<level> values present. Used for the 1M-token scale.
"""

import argparse
import json
import pickle
import random
from collections import defaultdict
from pathlib import Path

from corpus_reasoning.lib.io import load_jsonl


def _est_tokens(text: str) -> int:
    return int(len(text.split()) * 1.3)


def _doc_to_str(doc: dict) -> str:
    title = doc.get("title", "").strip()
    text = doc["text"].strip()
    return f"{title}\n{text}" if title else text


# ─────────────────────────── contradiction-from-jsonl ───────────────────────


def cmd_contradiction_from_jsonl(args):
    """One instance per source row. gold_doc_indices typically has 2-3 pairs
    per row (k=3 perturbation injects multiple contradictions) — all are
    valid answers, so reference_list = union of all gold-pair doc texts and
    the prompt asks for any contradicting pair."""
    rows = load_jsonl(args.input)
    instances: list[tuple[list[str], dict]] = []

    for row in rows:
        if len(instances) >= args.num_queries:
            break
        docs = [d["text"] for d in row["documents"]]
        pairs_idx = [(p[0] - 1, p[1] - 1) for p in row["gold_doc_indices"]]
        gold_doc_idx: list[int] = sorted({i for pair in pairs_idx for i in pair})
        reference_list = [docs[i] for i in gold_doc_idx]
        # gold_pairs stored as lists-of-texts so the scorer can resolve each
        # pair back to corpus indices via doc_to_idx[text] (mirrors how
        # MSA's benchmark.py resolves reference_list).
        gold_pair_groups = [[docs[a], docs[b]] for a, b in pairs_idx]
        pretty_pairs = "\n".join(
            f"  - [{docs[a][:60]}...]  CONTRADICTS  [{docs[b][:60]}...]"
            for a, b in pairs_idx
        )
        query = (
            f"This corpus contains {len(pairs_idx)} pair(s) of documents whose "
            f"claims directly contradict each other. Identify any one valid "
            f"contradicting pair by citing both document IDs in the form "
            f"`[id_a] [id_b]`. The documents you cite must be the contradicting "
            f"pair — do not cite supporting evidence."
        )
        entry = {
            "query": query,
            "answer": pretty_pairs,
            "reference_list": reference_list,
            "gold_pair_groups": gold_pair_groups,
        }
        instances.append((list(docs), entry))

    _write_per_query(args.output_dir, args.name, instances)


# ─────────────────────────── contradiction-stitch ───────────────────────────


def cmd_contradiction_stitch(args):
    rng = random.Random(args.seed)

    seed_rows = load_jsonl(args.input)
    filler_rows = list(seed_rows)
    for path in args.filler_pools or []:
        filler_rows.extend(load_jsonl(path))

    # Pre-flatten filler claims (only non-gold sentences from each row).
    filler_pool: list[str] = []
    for row in filler_rows:
        gold = {idx - 1 for pair in row["gold_doc_indices"] for idx in pair}
        for i, d in enumerate(row["documents"]):
            if i not in gold:
                filler_pool.append(d["text"])
    rng.shuffle(filler_pool)
    print(f"[stitch] filler pool: {len(filler_pool)} non-gold sentences")

    instances: list[tuple[list[str], dict]] = []
    used_seed_rows = rng.sample(range(len(seed_rows)), k=args.num_queries)

    cursor = 0  # walk filler_pool without replacement across instances
    for row_i in used_seed_rows:
        row = seed_rows[row_i]
        seed_docs = [d["text"] for d in row["documents"]]
        a_idx, b_idx = rng.choice(row["gold_doc_indices"])
        claim_a, claim_b = seed_docs[a_idx - 1], seed_docs[b_idx - 1]

        corpus_strs: list[str] = list(seed_docs)
        seen = set(corpus_strs)
        cur_toks = sum(_est_tokens(s) for s in corpus_strs)
        while cur_toks < args.target_tokens and cursor < len(filler_pool):
            cand = filler_pool[cursor]
            cursor += 1
            if cand in seen:
                continue
            seen.add(cand)
            corpus_strs.append(cand)
            cur_toks += _est_tokens(cand)
        if cur_toks < args.target_tokens:
            print(
                f"[stitch] warning: ran out of fillers at {cur_toks} tokens "
                f"(< target {args.target_tokens})"
            )

        rng.shuffle(corpus_strs)
        entry = {
            "query": "Find a pair of contradicting claims in the corpus.",
            "answer": f"{claim_a}\nCONTRADICTS\n{claim_b}",
            "reference_list": [claim_a, claim_b],
        }
        instances.append((corpus_strs, entry))
        print(
            f"[stitch] example {len(instances)}/{args.num_queries}: "
            f"{len(corpus_strs)} docs, ~{cur_toks} tok"
        )

    _write_per_query(args.output_dir, args.name, instances)


# ─────────────────────────── grouping-from-jsonl ────────────────────────────


def cmd_grouping_from_jsonl(args):
    instances: list[tuple[list[str], dict]] = []
    for path in args.inputs:
        if len(instances) >= args.num_queries:
            break
        for row in load_jsonl(path):
            if len(instances) >= args.num_queries:
                break
            docs = [_doc_to_str(d) for d in row["documents"]]
            # gold_doc_indices is 0-indexed in this JSONL; answer convention
            # in the source's `answers` field is 1-indexed — match that.
            gold = row["gold_doc_indices"]
            labels = row["cluster_labels"]
            answer = json.dumps(
                {lbl: sorted(i + 1 for i in c) for lbl, c in zip(labels, gold)}
            )
            ref_list = [docs[min(c)] for c in gold]
            entry = {
                "query": row["queries"][0],
                "answer": answer,
                "reference_list": ref_list,
            }
            instances.append((list(docs), entry))
    _write_per_query(args.output_dir, args.name, instances)


# ─────────────────────── grouping-generate-from-compact ────────────────────


def cmd_grouping_generate_from_compact(args):
    """Naturalistic grouping examples: sample n_docs abstracts uniformly,
    let k emerge from the distinct L<level> values that show up."""
    from corpus_reasoning.data.generate_arxiv_grouping_data import (
        has_all_levels, load_compact,
    )

    rng = random.Random(args.seed)
    papers = load_compact(args.compact_in)
    eval_pool = [p for p in papers
                 if p["year"] >= args.eval_year_min and has_all_levels(p)]
    print(f"eval pool (year>={args.eval_year_min}, all L0-L3 set): "
          f"{len(eval_pool):,} papers")

    instances: list[tuple[list[str], dict]] = []
    for ex_i in range(args.num_queries):
        level = rng.choice(args.levels)
        level_key = f"concept_L{level}"

        picks = rng.sample(eval_pool, args.docs_per_example)
        perm = list(range(args.docs_per_example))
        rng.shuffle(perm)
        permuted = [picks[old] for old in perm]

        groups: dict[str, list[int]] = defaultdict(list)
        for i, p in enumerate(permuted):
            groups[p[level_key]].append(i + 1)  # 1-indexed
        labels = sorted(groups.keys())
        k = len(labels)

        docs_str = [_doc_to_str({"title": p["title"], "text": p["abstract"]})
                    for p in permuted]
        query = f"Cluster these {args.docs_per_example} abstracts into {k} groups."
        answer = json.dumps({lbl: sorted(groups[lbl]) for lbl in labels})
        ref_list = [docs_str[min(groups[lbl]) - 1] for lbl in labels]

        entry = {
            "query": query,
            "answer": answer,
            "reference_list": ref_list,
        }
        instances.append((docs_str, entry))
        sizes = sorted((len(groups[lbl]) for lbl in labels), reverse=True)
        print(f"[gen] {ex_i+1}/{args.num_queries}: level=L{level}, "
              f"n_docs={args.docs_per_example}, k={k}, "
              f"cluster sizes (top 10)={sizes[:10]}")

    _write_per_query(args.output_dir, args.name, instances)


# ─────────────────────────── per-query writer ───────────────────────────────


def _write_per_query(
    output_dir: str,
    name: str,
    instances: list[tuple[list[str], dict]],
):
    out = Path(output_dir) / name
    out.mkdir(parents=True, exist_ok=True)
    width = max(3, len(str(len(instances) - 1)))
    for i, (corpus, entry) in enumerate(instances):
        sid = f"{i:0{width}d}"
        with open(out / f"mdata_{sid}.pkl", "wb") as f:
            pickle.dump(corpus, f)
        with open(out / f"qdata_{sid}.pkl", "wb") as f:
            pickle.dump([entry], f)

    if instances:
        c0, _ = instances[0]
        toks_0 = sum(_est_tokens(s) for s in c0)
        print(
            f"wrote {len(instances)} (mdata, qdata) pairs to {out}/  "
            f"(corpus 0: {len(c0)} docs, ~{toks_0} tok)"
        )


# ─────────────────────────── CLI ────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("contradiction-from-jsonl")
    p.add_argument("--input", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--num-queries", type=int, default=500)
    p.add_argument("--output-dir", default="data/eval_pkls")
    p.set_defaults(fn=cmd_contradiction_from_jsonl)

    p = sub.add_parser("contradiction-stitch")
    p.add_argument("--input", required=True, help="JSONL of seed rows (each contributes one gold pair).")
    p.add_argument("--filler-pools", nargs="*", default=[],
                   help="Additional JSONLs whose non-gold sentences are pooled as fillers.")
    p.add_argument("--name", required=True)
    p.add_argument("--num-queries", type=int, default=10)
    p.add_argument("--target-tokens", type=int, default=1_000_000)
    p.add_argument("--output-dir", default="data/eval_pkls")
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(fn=cmd_contradiction_stitch)

    p = sub.add_parser("grouping-generate-from-compact")
    p.add_argument("--compact-in", default="data/openalex_compact.jsonl")
    p.add_argument("--name", required=True)
    p.add_argument("--num-queries", type=int, default=10)
    p.add_argument("--docs-per-example", type=int, default=6000)
    p.add_argument("--levels", nargs="+", type=int, default=[0, 1, 2],
                   help="Concept levels to sample from per example. k is not "
                        "specified — it emerges as the number of distinct "
                        "L<level> values appearing in each random sample.")
    p.add_argument("--eval-year-min", type=int, default=2024)
    p.add_argument("--output-dir", default="data/eval_pkls")
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(fn=cmd_grouping_generate_from_compact)

    p = sub.add_parser("grouping-from-jsonl")
    p.add_argument("--inputs", nargs="+", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--num-queries", type=int, default=500)
    p.add_argument("--output-dir", default="data/eval_pkls")
    p.set_defaults(fn=cmd_grouping_from_jsonl)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
