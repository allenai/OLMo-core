"""Unified-corpus MuSiQue data generation.

Same recipe as `generate_hotpotqa_unified_corpus.py`, but for MuSiQue (2-4
hop). Each example has 2-4 gold paragraphs (one per hop, indexed via
`paragraph_support_idx` in `question_decomposition`); each is replaced with
the closest passage from `wikipedia-dpr-100w` (title-match + Jaccard) so
golds and distractors share one surface format. Hard negatives come from the
same corpus, mined with the full multi-hop question.

An example survives only if every hop's gold can be successfully replaced
AND each replacement still contains that hop's answer. Drop rate scales with
hop count, so 4-hop survival is lower than 2-hop.

Hard-neg answer-leak filter excludes ALL hop answers (intermediate +
final) so no bridge entity leaks into the distractor pool. Bumps the
multi-hop rejection rate above HPQA, so consider raising
`--num-hard-negatives` cautiously — fewer candidates survive the filter.

Usage:
    # Prototype on 20 examples
    python scripts/data/generate_musique_unified_corpus.py --prototype 20

    # Full generation, all hop counts
    python scripts/data/generate_musique_unified_corpus.py \
        --num-train 10000 --num-eval 500 --num-docs 100 --num-hard-negatives 10

    # Filter to specific hop counts (id prefix matches `2hop`, `3hop1`,
    # `3hop2`, `4hop1`, `4hop2`, `4hop3`)
    python scripts/data/generate_musique_unified_corpus.py \
        --num-train 5000 --hop-prefixes 3hop1,3hop2,4hop1,4hop2,4hop3 \
        --num-docs 100 --num-hard-negatives 10
"""

import argparse
import random
from collections import Counter

from datasets import load_dataset
from tqdm import tqdm

from corpus_reasoning.lib.bm25 import (
    BM25Searcher,
    hit_contains_answer,
    word_jaccard,
    _word_set,
    compute_search_depth,
    pick_hard_negs_from_hits,
    sample_from_pool,
)
from corpus_reasoning.lib.io import save_jsonl, print_dataset_stats
from corpus_reasoning.data.generate_hotpotqa_unified_corpus import (
    chunk_title,
    title_matches,
)


def get_musique_golds(example):
    """Per-hop list of {title, text, hop_answer} from a MuSiQue example.

    Resolves each hop's `paragraph_support_idx` (the value of the `idx` field
    on the supporting paragraph, NOT a list position — paragraphs may be
    sparse / shuffled in the bundled list). Returns None if any hop's
    support paragraph is missing (shouldn't happen for `answerable=True`).
    """
    paragraphs = example["paragraphs"]
    decomp = example["question_decomposition"]
    out = []
    for hop in decomp:
        idx = hop["paragraph_support_idx"]
        para = next((p for p in paragraphs if p["idx"] == idx), None)
        if para is None:
            return None
        out.append({
            "title": para["title"],
            "text": para["paragraph_text"],
            "hop_answer": hop["answer"],
        })
    return out


def hop_prefix(example):
    return example["id"].split("__")[0]


def batch_find_musique_gold_replacements(searcher, examples, k=20, threads=8):
    """Per-hop gold replacement (BM25 → title filter → best Jaccard).

    Mirrors `batch_find_gold_replacements` in the HPQA pipeline but flattens
    over hops instead of two-paragraph supporting facts. Returns three lists
    aligned to `examples`: gold_paras_per_ex (or None if extraction failed),
    replacements (list of {text,docid,score} or None per hop), jaccards.
    """
    gold_paras_per_ex = [get_musique_golds(ex) for ex in examples]

    queries, owners = [], []
    for i, gps in enumerate(gold_paras_per_ex):
        if gps is None:
            continue
        for j, gp in enumerate(gps):
            queries.append(gp["text"])
            owners.append((i, j))

    hits_batch = searcher.batch_search(queries, k=k, threads=threads)

    replacements = [
        [None] * len(gps) if gps else [] for gps in gold_paras_per_ex
    ]
    jaccards = [
        [0.0] * len(gps) if gps else [] for gps in gold_paras_per_ex
    ]
    for (ex_idx, hop_idx), hits in zip(owners, hits_batch):
        gp = gold_paras_per_ex[ex_idx][hop_idx]
        # Coverage 0.5 (vs HPQA default 1.0) admits article extensions like
        # 'Al-Qubeir' → 'Al-Qubeir massacre' or 'Calhoun, West Virginia' →
        # 'Calhoun County, West Virginia'. The answer-presence check below
        # backstops against admitting the wrong article.
        title_ok = [
            h for h in hits
            if title_matches(gp["title"], chunk_title(h["text"]), min_coverage=0.5)
        ]
        if not title_ok:
            continue
        gold_words = _word_set(gp["text"])
        best, best_jac = None, 0.0
        for h in title_ok:
            jac = word_jaccard(_word_set(h["text"]), gold_words)
            if jac > best_jac:
                best_jac, best = jac, h
        replacements[ex_idx][hop_idx] = best
        jaccards[ex_idx][hop_idx] = best_jac
    return gold_paras_per_ex, replacements, jaccards


def build_unified_musique(ex, gold_paras, gold_replacements, num_docs,
                          num_hard_negatives, rng, prefetched_hits, random_pool):
    """Tag, shuffle, emit one unified-format MuSiQue record.

    Excludes all hop-level answers + the final answer + any answer aliases
    from the hard-neg pool so no bridge entity leaks into a distractor.

    Random distractors are drawn from `random_pool` (pre-fetched once via
    `BM25Searcher.prefetch_random_pool`) — pure Python, no per-example
    Lucene fetches.
    """
    final_answer = ex["answer"]
    aliases = ex.get("answer_aliases") or []
    all_answers = [final_answer] + [g["hop_answer"] for g in gold_paras] + list(aliases)

    gold_texts = [g["text"] for g in gold_replacements]
    exclude_texts = set(gold_texts)

    n_needed = num_docs - len(gold_replacements)
    n_hard = min(num_hard_negatives, n_needed)
    n_rand = n_needed - n_hard

    hard = pick_hard_negs_from_hits(
        prefetched_hits, n_hard,
        exclude_answers=all_answers,
        exclude_docids=None,
        max_pair_overlap=0.5,
    ) if prefetched_hits else []
    used_texts = exclude_texts | {d["text"] for d in hard}

    random_docs = sample_from_pool(
        random_pool, n_rand,
        exclude_answers=all_answers,
        exclude_texts=used_texts,
        rng=rng,
    ) if n_rand > 0 else []

    distractors = hard + random_docs

    tagged = (
        [({"title": "", "text": t}, "gold") for t in gold_texts]
        + [(d, "hard") for d in distractors[:n_hard]]
        + [(d, "pool") for d in distractors[n_hard:n_hard + n_rand]]
    )
    rng.shuffle(tagged)
    all_paragraphs = [{"title": "", "text": d["text"]} for d, _ in tagged]
    gold_indices = [i for i, (_, tag) in enumerate(tagged) if tag == "gold"]
    hard_neg_indices = [i for i, (_, tag) in enumerate(tagged) if tag == "hard"]

    return {
        "documents": all_paragraphs,
        "queries": [ex["question"]],
        "answers": [final_answer],
        "gold_doc_indices": gold_indices,
        "hard_neg_indices": hard_neg_indices,
        "source": "musique_unified_corpus",
        "musique_id": ex["id"],
        "hop_count": len(gold_paras),
    }


def _example_survives(gps, reps, jacs, min_jaccard):
    if gps is None:
        return False
    if not all(r is not None and j >= min_jaccard for r, j in zip(reps, jacs)):
        return False
    return all(
        hit_contains_answer(r["text"], [gp["hop_answer"]])
        for r, gp in zip(reps, gps)
    )


def generate_split(searcher, split_name, num_wanted, num_docs,
                   num_hard_negatives, min_jaccard,
                   candidate_multiplier, seed, hop_prefixes,
                   pool_size, batch_threads=8):
    rng = random.Random(seed)
    print(f"\nLoading dgslibisey/MuSiQue [{split_name}]...")
    ds = load_dataset("dgslibisey/MuSiQue", split=split_name)
    if hop_prefixes:
        ds = ds.filter(lambda ex: hop_prefix(ex) in hop_prefixes)
        print(f"  {len(ds)} examples after hop filter {sorted(hop_prefixes)}")
    else:
        print(f"  {len(ds)} examples")

    indices = list(range(len(ds)))
    rng.shuffle(indices)

    n_probe = min(len(ds), max(num_wanted * candidate_multiplier, num_wanted))
    probe_examples = [ds[i] for i in indices[:n_probe]]
    print(f"  Probing {len(probe_examples)} candidates for gold replacement...")

    gold_paras_per_ex, replacements, jaccards = batch_find_musique_gold_replacements(
        searcher, probe_examples, k=20, threads=batch_threads,
    )

    survivors = []  # (ex, gold_paras, gold_replacements_as_doc_dicts)
    for ex, gps, reps, jacs in zip(
        probe_examples, gold_paras_per_ex, replacements, jaccards
    ):
        if not _example_survives(gps, reps, jacs, min_jaccard):
            continue
        survivors.append((
            ex, gps, [{"title": "", "text": r["text"]} for r in reps],
        ))
        if len(survivors) >= num_wanted:
            break

    n_probed = len(probe_examples)
    print(f"  Survivors after gold replacement + per-hop answer check: "
          f"{len(survivors)}/{n_probed} ({100*len(survivors)/max(n_probed,1):.1f}%)")
    if survivors:
        surv_hop_dist = Counter(hop_prefix(ex) for ex, _, _ in survivors)
        print(f"  Survivor hop distribution: {dict(sorted(surv_hop_dist.items()))}")
    if len(survivors) < num_wanted:
        print(f"  WARNING: only {len(survivors)} survived (requested {num_wanted})")

    # Hard-neg mining via full multi-hop question.
    if num_hard_negatives > 0 and survivors:
        search_depth = compute_search_depth(num_hard_negatives, max_pair_overlap=0.5)
        queries = [ex["question"] for ex, _, _ in survivors]
        print(f"  Batch BM25 mining: {len(queries)} queries × depth {search_depth} "
              f"× threads {batch_threads}...")
        hits_batch = searcher.batch_search(queries, k=search_depth, threads=batch_threads)
    else:
        hits_batch = [None] * len(survivors)

    # Pre-fetch random pool once. Per-example sampling then runs in pure Python
    # over `random_pool` — avoids ~88 serial Lucene `doc()` lookups per example.
    print(f"  Pre-fetching random pool: {pool_size} passages × threads {batch_threads}...")
    random_pool = searcher.prefetch_random_pool(
        pool_size, threads=batch_threads, seed=seed,
    )
    print(f"    Pool size after fetch: {len(random_pool)}")

    examples = []
    for (ex, gold_paras, gold_reps), hits in tqdm(
        zip(survivors, hits_batch), total=len(survivors),
        desc=f"Building {split_name}",
    ):
        examples.append(build_unified_musique(
            ex, gold_paras, gold_reps, num_docs, num_hard_negatives, rng,
            prefetched_hits=hits, random_pool=random_pool,
        ))
    return examples


def prototype(searcher, num, hop_prefixes, seed=42, min_jaccard=0.05):
    rng = random.Random(seed)
    ds = load_dataset("dgslibisey/MuSiQue", split="train")
    if hop_prefixes:
        ds = ds.filter(lambda ex: hop_prefix(ex) in hop_prefixes)
    indices = list(range(len(ds)))
    rng.shuffle(indices)
    probe = [ds[i] for i in indices[:num]]

    gold_paras_per_ex, replacements, jaccards = batch_find_musique_gold_replacements(
        searcher, probe, k=20,
    )

    n_gold_total = sum(len(gps) for gps in gold_paras_per_ex if gps)
    n_gold_swapped = sum(
        1 for reps in replacements for r in reps if r is not None
    )
    n_above_floor = sum(
        1 for reps, jacs in zip(replacements, jaccards)
        for r, j in zip(reps, jacs) if r is not None and j >= min_jaccard
    )
    n_examples_survive = sum(
        1 for gps, reps, jacs in zip(gold_paras_per_ex, replacements, jaccards)
        if _example_survives(gps, reps, jacs, min_jaccard)
    )

    hop_dist = Counter(hop_prefix(ex) for ex in probe)
    print("\n" + "=" * 60)
    print(f"Prototype results on {num} MuSiQue examples")
    print("=" * 60)
    print(f"Hop distribution:               {dict(sorted(hop_dist.items()))}")
    print(f"Gold paragraphs total:          {n_gold_total}")
    print(f"Replaced with 100w chunk:       {n_gold_swapped}/{n_gold_total} "
          f"({100*n_gold_swapped/max(n_gold_total,1):.1f}%)")
    print(f"Above Jaccard floor {min_jaccard:.2f}:       {n_above_floor}/{n_gold_total} "
          f"({100*n_above_floor/max(n_gold_total,1):.1f}%)")
    print(f"Examples where ALL hops pass:   {n_examples_survive}/{num} "
          f"({100*n_examples_survive/num:.1f}%)")

    # Survival rate broken down by hop count — this is what scales drop rate.
    surv_by_hop = Counter()
    total_by_hop = Counter()
    for ex, gps, reps, jacs in zip(probe, gold_paras_per_ex, replacements, jaccards):
        prefix = hop_prefix(ex)
        total_by_hop[prefix] += 1
        if _example_survives(gps, reps, jacs, min_jaccard):
            surv_by_hop[prefix] += 1
    print("Survival by hop type:")
    for prefix in sorted(total_by_hop):
        s, t = surv_by_hop[prefix], total_by_hop[prefix]
        print(f"  {prefix}: {s}/{t} ({100*s/t:.1f}%)")

    # Side-by-side: 2 examples per hop type
    print("\n--- Side-by-side samples (up to 2 per hop type) ---")
    by_hop = {}
    for ex, gps, reps, jacs in zip(probe, gold_paras_per_ex, replacements, jaccards):
        if not gps:
            continue
        by_hop.setdefault(hop_prefix(ex), []).append((ex, gps, reps, jacs))

    for prefix in sorted(by_hop):
        print(f"\n=== {prefix} ===")
        for ex, gps, reps, jacs in by_hop[prefix][:2]:
            print(f"\n[Q] {ex['question']}")
            print(f"[A] {ex['answer']}")
            for i, (gp, r, j) in enumerate(zip(gps, reps, jacs)):
                print(f"  Hop {i+1}: title={gp['title']!r}, hop_answer={gp['hop_answer']!r}, jac={j:.2f}")
                print(f"    GOLD: {gp['text'][:240]}")
                if r is not None:
                    print(f"    100W: {r['text'][:240]}")
                else:
                    print(f"    100W: <no title-matching hit>")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prototype", type=int, default=0,
                   help="Run prototype analysis on N examples then exit")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min-jaccard", type=float, default=0.35,
                   help="Reject replacements below this Jaccard with gold")
    p.add_argument("--num-train", type=int, default=0)
    p.add_argument("--num-eval", type=int, default=0)
    p.add_argument("--num-docs", type=int, default=100,
                   help="Total documents per example (golds + hard + random)")
    p.add_argument("--num-hard-negatives", type=int, default=10,
                   help="BM25 hard negatives from 100w per example")
    p.add_argument("--candidate-multiplier", type=int, default=4,
                   help="Probe this many × target to allow for drop-outs "
                        "(higher hop counts may need more)")
    p.add_argument("--hop-prefixes", type=str, default="",
                   help="Comma-separated id prefixes to keep "
                        "(e.g. '2hop' or '3hop1,3hop2,4hop1,4hop2,4hop3'). "
                        "Empty = keep all.")
    p.add_argument("--output-dir", type=str, default="data")
    p.add_argument("--batch-threads", type=int, default=8)
    p.add_argument("--pool-size", type=int, default=200000,
                   help="Random pool pre-fetched once per split. Sampled "
                        "from in Python per-example (no per-example Lucene "
                        "calls). Should comfortably exceed "
                        "(num_docs - num_hard_negatives) so the per-example "
                        "answer filter has slack.")
    args = p.parse_args()

    hop_prefixes = (
        set(s.strip() for s in args.hop_prefixes.split(",") if s.strip())
        if args.hop_prefixes else set()
    )

    searcher = BM25Searcher()

    if args.prototype > 0:
        prototype(searcher, args.prototype, hop_prefixes,
                  seed=args.seed, min_jaccard=args.min_jaccard)
        return

    if args.num_train == 0 and args.num_eval == 0:
        raise SystemExit("Set --num-train and/or --num-eval (or --prototype)")

    specs = []
    if args.num_train > 0:
        specs.append(("train", args.num_train, "train"))
    if args.num_eval > 0:
        specs.append(("validation", args.num_eval, "eval"))

    for split_name, num_wanted, label in specs:
        examples = generate_split(
            searcher, split_name, num_wanted,
            num_docs=args.num_docs,
            num_hard_negatives=args.num_hard_negatives,
            min_jaccard=args.min_jaccard,
            candidate_multiplier=args.candidate_multiplier,
            seed=args.seed + (0 if split_name == "train" else 1),
            hop_prefixes=hop_prefixes,
            pool_size=args.pool_size,
            batch_threads=args.batch_threads,
        )
        hop_tag = ("_" + "+".join(sorted(hop_prefixes))) if hop_prefixes else ""
        tag = f"k{args.num_docs}_unified_hn{args.num_hard_negatives}{hop_tag}"
        path = f"{args.output_dir}/musique_{label}_{tag}_{num_wanted}.jsonl"
        save_jsonl(path, examples)
        print_dataset_stats(examples, split_name.capitalize(), path)


if __name__ == "__main__":
    main()
