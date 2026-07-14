"""Unified-corpus HotpotQA data generation (prototype).

Motivation
----------
The existing HPQA pipeline (with BM25 hard negatives from wikipedia-dpr-100w)
is stylistically mismatched: gold paragraphs come from HotpotQA's own
Wikipedia snapshot (natural paragraphs, titled) while hard negatives come from
wikipedia-dpr-100w (100w chunks, no titles). We suspect the model learns to
key on chunk-level surface features rather than semantic relevance, inflating
recall on the BM25-heavy setup (hn98 2k hit 97% vs. HPQA-native-format 76.5%).

Fix: replace each HPQA gold paragraph with the *most similar passage in
wikipedia-dpr-100w* (retrieved by BM25), then mine hard + random negatives
from the same 100w corpus. Everything comes from one distribution.

Pipeline per example
--------------------
1. For each gold paragraph:
   - BM25-search 100w with `title + first-sentence`.
   - Rank top-k candidates by word-overlap Jaccard with gold text.
   - Require that the top candidate (a) is above a min Jaccard floor and
     (b) still contains the answer (lowercased substring or strict
     token-coverage rule from lib/bm25.py `hit_contains_answer`).
2. Gemini-flash gate (optional): ask "does this passage support the answer
   to this question?" for each replacement gold chunk. Drop the example if
   any gold fails.
3. Mine BM25 hard negs from 100w (same corpus, same style) via existing
   `batch_mine_hard_negs`. Fill remaining slots with random 100w passages.
4. Shuffle and emit in unified JSONL format.

Prototype mode
--------------
`--prototype N` runs on N examples, reports:
  - Replacement success per gold (top Jaccard, answer-present rate)
  - Gemini gate pass rate (if --use-gate)
  - 3 side-by-side (original gold ↔ retrieved chunk) for manual inspection

Usage
-----
    # Prototype on 20 examples (BM25 only, no Gemini yet)
    python scripts/data/generate_hotpotqa_unified_corpus.py --prototype 20

    # With Gemini gate
    python scripts/data/generate_hotpotqa_unified_corpus.py --prototype 20 --use-gate
"""

import argparse
import random
import re

from datasets import load_dataset
from tqdm import tqdm

from corpus_reasoning.lib.bm25 import (
    BM25Searcher,
    hit_contains_answer,
    word_jaccard,
    _word_set,
    compute_search_depth,
    pick_hard_negs_from_hits,
)
from corpus_reasoning.lib.io import save_jsonl, print_dataset_stats


GATE_PROMPT = (
    "You are judging whether a Wikipedia passage still supports the answer "
    "to a question that was originally answered by a different (but related) "
    "passage.\n\n"
    "Question: {question}\n"
    "Answer: {answer}\n"
    "Passage: {passage}\n\n"
    "Does this passage contain enough information to support the given "
    "answer? Respond with exactly one word: YES or NO."
)


def first_sentence(text: str) -> str:
    m = re.match(r"(.*?[.!?])\s", text)
    return m.group(1) if m else text[:200]


# dpr-100w chunks are formatted as `"Title"\n<content>`. Extract the leading
# quoted title so we can require an article-level match against the gold title.
_CHUNK_TITLE_RE = re.compile(r'^\s*"([^"]+)"')


def chunk_title(chunk_text: str) -> str:
    m = _CHUNK_TITLE_RE.match(chunk_text)
    return m.group(1) if m else ""


def _title_sig_tokens(title: str) -> set[str]:
    """Significant tokens of a title: lowercased, len >= 3, strips
    parenthetical disambiguators (e.g. '(song)', '(film)') that would
    otherwise block cross-edition matches."""
    stripped = re.sub(r"\([^)]*\)", "", title)
    return {t for t in re.findall(r"\w+", stripped.lower()) if len(t) >= 3}


def title_matches(gold_title: str, candidate_title: str,
                  min_coverage: float = 1.0) -> bool:
    """True if all significant gold-title tokens appear in candidate title.

    Default `min_coverage=1.0` means every significant gold token must
    appear; lower values relax to require that fraction. Parentheticals
    are stripped on both sides first.
    """
    g = _title_sig_tokens(gold_title)
    c = _title_sig_tokens(candidate_title)
    if not g:
        return False
    matched = sum(1 for t in g if t in c)
    return matched / len(g) >= min_coverage


def find_best_chunk(searcher, gold_title: str, gold_text: str,
                    k: int = 20) -> tuple[dict | None, float]:
    """Return (best_candidate_dict_or_None, jaccard_with_gold).

    Searches 100w with the full gold paragraph text, filters to hits whose
    leading `"Title"` matches the gold title (all significant gold tokens
    must appear in the candidate title), then picks the surviving hit with
    highest word-level Jaccard vs. gold. The title filter rejects
    same-topic-wrong-article matches (e.g. 'Letters from Iwo Jima' vs
    'Flags of Our Fathers (film)').
    """
    query = gold_text
    hits = searcher.search(query, k=k)
    if not hits:
        return None, 0.0

    # Title-match filter first — cheap, rejects same-topic-wrong-article.
    title_ok = [h for h in hits if title_matches(gold_title, chunk_title(h["text"]))]
    if not title_ok:
        return None, 0.0

    gold_words = _word_set(gold_text)
    best, best_jac = None, 0.0
    for h in title_ok:
        jac = word_jaccard(_word_set(h["text"]), gold_words)
        if jac > best_jac:
            best_jac, best = jac, h
    return best, best_jac


def paragraphs_from_context(context):
    titles = context["title"]
    sentences_list = context["sentences"]
    docs = []
    for title, sentences in zip(titles, sentences_list):
        text = " ".join(s.strip() for s in sentences)
        docs.append({"title": title, "text": text})
    return docs


def get_supporting(example):
    all_docs = paragraphs_from_context(example["context"])
    sf_titles = list(dict.fromkeys(example["supporting_facts"]["title"]))
    out = []
    for t in sf_titles:
        for d in all_docs:
            if d["title"] == t and d not in out:
                out.append(d)
                break
    return out


def prototype(searcher, num, use_gate=False, seed=42, min_jaccard=0.05):
    rng = random.Random(seed)
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="train")
    ds = ds.filter(lambda ex: ex["type"] == "bridge")
    indices = list(range(len(ds)))
    rng.shuffle(indices)
    selected = indices[:num]

    records = []  # (example, gold_paras, replacements, jaccards)
    for idx in tqdm(selected, desc="BM25 gold-swap"):
        ex = ds[idx]
        gold_paras = get_supporting(ex)

        replacements, jaccards = [], []
        for gp in gold_paras:
            best, jac = find_best_chunk(searcher, gp["title"], gp["text"])
            replacements.append(best)
            jaccards.append(jac)
        records.append((ex, gold_paras, replacements, jaccards))

    n_gold_total = sum(len(r[1]) for r in records)
    n_gold_swapped = sum(1 for r in records for b in r[2] if b is not None)
    n_above_floor = sum(
        1 for r in records for b, j in zip(r[2], r[3])
        if b is not None and j >= min_jaccard
    )
    # Example survives if: all golds swapped + above floor + at least one
    # replacement chunk (across the pair) still contains the answer.
    def _example_survives(ex, reps, jacs):
        if not all(b is not None and j >= min_jaccard for b, j in zip(reps, jacs)):
            return False
        return any(hit_contains_answer(b["text"], [ex["answer"]]) for b in reps)
    n_examples_survive = sum(
        1 for r in records if _example_survives(r[0], r[2], r[3])
    )

    print("\n" + "=" * 60)
    print(f"Prototype results on {num} bridge examples")
    print("=" * 60)
    print(f"Gold paragraphs total:          {n_gold_total}")
    print(f"Replaced with 100w chunk:       {n_gold_swapped}/{n_gold_total} "
          f"({100 * n_gold_swapped / n_gold_total:.1f}%)")
    print(f"Above Jaccard floor {min_jaccard:.2f}:       {n_above_floor}/{n_gold_total} "
          f"({100 * n_above_floor / n_gold_total:.1f}%)")
    print(f"Examples where BOTH golds pass: {n_examples_survive}/{num} "
          f"({100 * n_examples_survive / num:.1f}%)")

    # Jaccard distribution
    all_jac = [j for r in records for j in r[3]]
    if all_jac:
        all_jac_sorted = sorted(all_jac)
        q = lambda p: all_jac_sorted[min(int(p * len(all_jac_sorted)), len(all_jac_sorted) - 1)]
        print(f"Jaccard p10/p50/p90:            {q(0.1):.2f} / {q(0.5):.2f} / {q(0.9):.2f}")

    # Samples bucketed by Jaccard so user can judge threshold
    buckets = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)]
    by_bucket = {b: [] for b in buckets}
    for ex, gold_paras, reps, jacs in records:
        for gp, rep, jac in zip(gold_paras, reps, jacs):
            if rep is None:
                continue
            for b in buckets:
                if b[0] <= jac < b[1]:
                    by_bucket[b].append((ex, gp, rep, jac))
                    break

    print("\n--- Bucketed samples (2 per bucket) ---")
    for lo, hi in buckets:
        samples = by_bucket[(lo, hi)]
        print(f"\n=== Jaccard [{lo:.1f}, {hi:.2f})   ({len(samples)} chunks) ===")
        for ex, gp, rep, jac in samples[:2]:
            print(f"\n[Q] {ex['question'][:120]}")
            print(f"[A] {ex['answer']}   [Jaccard] {jac:.2f}   [Title] {gp['title']}")
            print(f"[GOLD ] {gp['text'][:280]}")
            print(f"[100W ] {rep['text'][:280]}")

    if use_gate:
        from corpus_reasoning.lib.llm_request_client import ParallelResponsesClient
        print("\n--- Gemini-flash gate ---")
        prompts, meta = [], []
        for ex, gold_paras, reps, _ in records:
            for gp, rep in zip(gold_paras, reps):
                if rep is None:
                    continue
                prompts.append(GATE_PROMPT.format(
                    question=ex["question"],
                    answer=ex["answer"],
                    passage=rep["text"],
                ))
                meta.append((ex["question"], ex["answer"], gp, rep))
        print(f"Sending {len(prompts)} gate requests...")
        client = ParallelResponsesClient(max_concurrent=10, use_cache=True)
        responses = client.run(
            model="gemini-2.5-flash",
            prompts=prompts,
            temperature=0.0,
            max_output_tokens=5,
        )
        pass_count = sum(
            1 for r in responses
            if r.get("response", "").strip().upper().startswith("YES")
        )
        print(f"Gate pass rate: {pass_count}/{len(prompts)} "
              f"({100 * pass_count / max(len(prompts), 1):.1f}%)")
        # Show any failures
        print("\n--- Up to 3 gate failures ---")
        shown = 0
        for resp, (q, a, gp, rep) in zip(responses, meta):
            verdict = resp.get("response", "").strip().upper()
            if verdict.startswith("YES") or shown >= 3:
                continue
            print(f"\n[Verdict] {verdict}")
            print(f"[Q] {q}   [A] {a}")
            print(f"[GOLD ] {gp['text'][:250]}")
            print(f"[100W ] {rep['text'][:250]}")
            shown += 1


def batch_find_gold_replacements(searcher, examples, k=20, threads=8):
    """Batched gold replacement for many HPQA examples.

    For each example, emits (gold_paras, replacements, jaccards) where
    replacements[i] is either a dict (100w chunk) or None if no title-
    matching hit was found. Uses batch_search to amortize Lucene startup.
    """
    # Flatten: one query per gold paragraph
    queries, owners = [], []  # owners[q] = (example_idx, gold_idx)
    gold_paras_per_ex = []
    for i, ex in enumerate(examples):
        gps = get_supporting(ex)
        gold_paras_per_ex.append(gps)
        for j, gp in enumerate(gps):
            queries.append(gp["text"])
            owners.append((i, j))

    hits_batch = searcher.batch_search(queries, k=k, threads=threads)

    # Reduce each query's hits to best-Jaccard title-match, then scatter back.
    replacements = [[None] * len(gps) for gps in gold_paras_per_ex]
    jaccards = [[0.0] * len(gps) for gps in gold_paras_per_ex]
    for (ex_idx, gold_idx), hits in zip(owners, hits_batch):
        gp = gold_paras_per_ex[ex_idx][gold_idx]
        title_ok = [h for h in hits if title_matches(gp["title"], chunk_title(h["text"]))]
        if not title_ok:
            continue
        gold_words = _word_set(gp["text"])
        best, best_jac = None, 0.0
        for h in title_ok:
            jac = word_jaccard(_word_set(h["text"]), gold_words)
            if jac > best_jac:
                best_jac, best = jac, h
        replacements[ex_idx][gold_idx] = best
        jaccards[ex_idx][gold_idx] = best_jac

    return gold_paras_per_ex, replacements, jaccards


def build_unified_example(ex, gold_replacements, searcher, num_docs,
                          num_hard_negatives, rng, prefetched_hits=None):
    """Assemble one unified-JSONL record from gold replacements + BM25 hard +
    random 100w fill.

    If `prefetched_hits` is provided (a list of BM25 hits already retrieved at
    sufficient depth — see `compute_search_depth`), the per-query BM25 search
    is skipped and hard-neg selection is done directly via
    `pick_hard_negs_from_hits`. This is the fast path used by the batched
    main loop in `generate_split`. When None, falls back to per-example
    `searcher.mine_distractors` (slower; preserved for back-compat).
    """
    question = ex["question"]
    answer = ex["answer"]
    exclude_texts = {g["text"] for g in gold_replacements}

    n_needed = num_docs - len(gold_replacements)
    n_hard = min(num_hard_negatives, n_needed)
    n_rand = n_needed - n_hard

    if prefetched_hits is not None:
        hard = pick_hard_negs_from_hits(
            prefetched_hits, n_hard,
            exclude_answers=[answer],
            exclude_docids=None,
            max_pair_overlap=0.5,
        )
        used_texts = {d["text"] for d in hard}
        random_docs = []
        if n_rand > 0:
            random_docs = searcher.sample_random_passages(
                n_rand,
                exclude_docids=None,
                exclude_answers=[answer],
                rng=rng,
            )
            random_docs = [d for d in random_docs if d["text"] not in used_texts]
        distractors = hard + random_docs
    else:
        distractors = searcher.mine_distractors(
            query=question, num_hard=n_hard, num_random=n_rand,
            exclude_answers=[answer],
            exclude_docids=None,  # hard-neg dedup is text-based inside mine_distractors
            max_pair_overlap=0.5,
            rng=rng,
        )
    # Drop any distractor that collides with our gold replacements by text
    distractors = [d for d in distractors if d["text"] not in exclude_texts]
    # Pad if we came up short (rare)
    while len(distractors) < n_needed:
        extra = searcher.sample_random_passages(
            1, exclude_answers=[answer], rng=rng,
        )
        if extra and extra[0]["text"] not in exclude_texts:
            distractors.append(extra[0])

    tagged = ([(g, "gold") for g in gold_replacements]
              + [(d, "hard") for d in distractors[:n_hard]]
              + [(d, "pool") for d in distractors[n_hard:n_hard + n_rand]])
    rng.shuffle(tagged)
    all_paragraphs = [{"title": "", "text": d["text"]} for d, _ in tagged]
    gold_indices = [i for i, (_, tag) in enumerate(tagged) if tag == "gold"]
    hard_neg_indices = [i for i, (_, tag) in enumerate(tagged) if tag == "hard"]

    return {
        "documents": all_paragraphs,
        "queries": [question],
        "answers": [answer],
        "gold_doc_indices": gold_indices,
        "hard_neg_indices": hard_neg_indices,
        "source": "hotpotqa_unified_corpus",
    }


def generate_split(searcher, split_name, num_wanted, num_docs,
                   num_hard_negatives, min_jaccard,
                   candidate_multiplier, seed, batch_threads=8):
    """Process one HPQA split (train or validation) and return up to
    `num_wanted` unified examples that survive gold replacement + pair
    answer check."""
    rng = random.Random(seed)
    print(f"\nLoading hotpot_qa distractor [{split_name}]...")
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split=split_name)
    ds = ds.filter(lambda ex: ex["type"] == "bridge")
    print(f"  {len(ds)} bridge examples available")

    indices = list(range(len(ds)))
    rng.shuffle(indices)

    n_probe = min(len(ds), max(num_wanted * candidate_multiplier, num_wanted))
    probe_examples = [ds[i] for i in indices[:n_probe]]
    print(f"  Probing {len(probe_examples)} candidates for gold replacement...")

    # Phase 1: batch gold replacement
    gold_paras_per_ex, replacements, jaccards = batch_find_gold_replacements(
        searcher, probe_examples, k=20, threads=batch_threads,
    )

    # Phase 2: filter survivors
    survivors = []  # (ex, gold_replacements)
    for ex, reps, jacs in zip(probe_examples, replacements, jaccards):
        if not all(r is not None and j >= min_jaccard for r, j in zip(reps, jacs)):
            continue
        if not any(hit_contains_answer(r["text"], [ex["answer"]]) for r in reps):
            continue
        # Strip title prefix from 100w text for storage — we stored raw chunk
        survivors.append((ex, [{"title": "", "text": r["text"]} for r in reps]))
        if len(survivors) >= num_wanted:
            break

    print(f"  Survivors after gold replacement + answer check: {len(survivors)}"
          f"/{len(probe_examples)} ({100*len(survivors)/max(len(probe_examples),1):.1f}%)")
    if len(survivors) < num_wanted:
        print(f"  WARNING: only {len(survivors)} survived (requested {num_wanted})")

    # Phase 3: batch BM25 mining, then per-example filter+pad.
    #
    # The original per-example `searcher.mine_distractors(...)` was sequential:
    # one Lucene `searcher.search(query, k=depth)` call per survivor. At
    # high `num_hard_negatives` the depth grows linearly (depth = 5 * num_hard
    # under pair-overlap filtering), so per-query latency dominates wall time.
    # Batching all queries into a single `batch_search` lets Lucene's thread
    # pool parallelize across queries, then we walk the prefetched hits per
    # example to apply the answer + pair-overlap filters single-threaded.
    n_hard_per_ex = min(num_hard_negatives, num_docs - 2)
    search_depth = compute_search_depth(n_hard_per_ex, max_pair_overlap=0.5)
    queries = [ex["question"] for ex, _ in survivors]
    print(f"  Batch BM25 mining: {len(queries)} queries × depth {search_depth} "
          f"× threads {batch_threads}...")
    hits_batch = searcher.batch_search(queries, k=search_depth, threads=batch_threads)

    examples = []
    for (ex, gold_reps), hits in tqdm(
        zip(survivors, hits_batch), total=len(survivors),
        desc=f"Building {split_name}",
    ):
        examples.append(build_unified_example(
            ex, gold_reps, searcher, num_docs, num_hard_negatives, rng,
            prefetched_hits=hits,
        ))
    return examples


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prototype", type=int, default=0,
                   help="Run prototype analysis on N examples then exit")
    p.add_argument("--use-gate", action="store_true",
                   help="Prototype mode only: call gemini-2.5-flash on replacements")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min-jaccard", type=float, default=0.35,
                   help="Reject replacements below this Jaccard with gold")
    # Full-generation args
    p.add_argument("--num-train", type=int, default=0,
                   help="Number of train examples to generate (0 = skip)")
    p.add_argument("--num-eval", type=int, default=0,
                   help="Number of eval examples to generate (0 = skip)")
    p.add_argument("--num-docs", type=int, default=100,
                   help="Total documents per example (gold + hard + random)")
    p.add_argument("--num-hard-negatives", type=int, default=10,
                   help="BM25 hard negatives from 100w per example")
    p.add_argument("--candidate-multiplier", type=int, default=4,
                   help="Probe this many × target to allow for drop-outs")
    p.add_argument("--output-dir", type=str, default="data")
    p.add_argument("--batch-threads", type=int, default=8,
                   help="Lucene thread pool size for batch BM25 search")
    args = p.parse_args()

    searcher = BM25Searcher()

    if args.prototype > 0:
        prototype(
            searcher, args.prototype, use_gate=args.use_gate,
            seed=args.seed, min_jaccard=args.min_jaccard,
        )
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
            batch_threads=args.batch_threads,
        )
        # Filename uses requested count (num_wanted) for predictability in
        # downstream configs; actual len(examples) may be lower if survival
        # rate is below what candidate_multiplier anticipated.
        tag = f"k{args.num_docs}_bridge_unified_hn{args.num_hard_negatives}"
        path = f"{args.output_dir}/hotpotqa_{label}_{tag}_{num_wanted}.jsonl"
        save_jsonl(path, examples)
        print_dataset_stats(examples, split_name.capitalize(), path)


if __name__ == "__main__":
    main()
