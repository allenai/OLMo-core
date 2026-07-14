"""Generate Wikipedia chunk-level contradiction data from the 100w corpus.

Wikipedia analog of generate_pubmed_contradiction_data.py, redesigned for
chunk-level rather than sentence-level documents:

  Each example is N=num_docs Wikipedia 100w chunks, with K=num_contradictions
  hidden gold pairs. Each gold pair is:

      (chunk A, chunk B')   where chunk A is unmodified and chunk B' is a
                            real Wikipedia chunk in which one sentence has
                            been LLM-rewritten to factually contradict a
                            claim in chunk A.

  Chunk B is picked by BM25-searching the corpus with chunk A's claim as
  query, so B is topically related to A — a plausible distractor — rather
  than a random chunk that gives the contradiction away by topic alone.

  Filler chunks are random unrelated Wikipedia passages (excluding any
  article that A or B came from in this example).

Two entity modes via --entity-mode:
  same    — B's rewritten sentence makes a contradicting claim about the
            SAME entity that A's claim is about (high overlap, easy to
            spot).
  related — B's rewritten sentence makes a contradicting claim about a
            different entity that's topically related (subtler — the
            contradiction is logical, not lexical).
  mix     — half each.

`gold_doc_indices` stores contradiction pairs as 1-indexed chunk-position
pairs `[[a, b], ...]`, the same format as the PubMed task — so the existing
`scripts/lib/data_format.py:_build_output(task="contradiction")` consumes it
unchanged.

Usage:
    python scripts/data/generate_wiki_contradiction_data.py \\
        --num-docs 20 --num-contradictions 3 --entity-mode mix \\
        --num-train 2000 --num-eval 300
"""

import argparse
import json
import random
import re
from pathlib import Path

from tqdm import tqdm

from corpus_reasoning.lib.bm25 import BM25Searcher
from corpus_reasoning.lib.io import save_jsonl, print_dataset_stats
from corpus_reasoning.lib.wiki100w_sample import (
    pick_chunk_with_claim,
    find_related_chunks,
    sample_random_chunks,
    sample_clustered_fillers,
    fetch_article_neighbors,
    fetch_article_neighbors_by_title,
    split_sentences,
)


# ─────────────────────────── prompts ──────────────────────────────────────

SAME_ENTITY_PROMPT = """You will be given two Wikipedia passages and one specific sentence in passage B that should be rewritten so it factually contradicts a specific claim in passage A.

Goal: rewrite the chosen sentence in passage B so that it asserts something about the SAME entity that the claim in A is about, but in a way that is factually incompatible with A — both statements cannot be true at once.

Requirements:
- The rewritten sentence must mention the same entity as the claim in A and make a directly conflicting factual assertion (different date, different location, different role, opposite attribute, opposite outcome, etc.).
- It should still read like a Wikipedia sentence: neutral tone, no first person, no editorial language.
- Length similar to the original sentence in B (±50%).
- Stand alone — do not start with "However", "But", "It", "This", "These".
- Do not preserve the rest of B's sentence verbatim if it is now incoherent with the new fact.

Return ONLY the single rewritten sentence — no preamble, no quotes, no explanation.

Passage A:
{passage_a}

Claim in A to contradict:
{claim_a}

Passage B (where you will rewrite one sentence):
{passage_b}

Original sentence in B to rewrite:
{sentence_b}

Rewritten sentence:"""


RELATED_ENTITY_PROMPT = """You will be given two Wikipedia passages and one specific sentence in passage B that should be rewritten so it factually contradicts a specific claim in passage A — but indirectly, via a related entity rather than the same one.

Goal: rewrite the chosen sentence in passage B so that it asserts something about a RELATED entity (one already mentioned somewhere in B), and that assertion is logically incompatible with the claim in A. A reader has to combine information from A and B to spot the contradiction; surface lexical overlap should be low.

Examples of "related entity" contradictions:
  A: "Smith founded the company in 1850."   (entity: company, fact: founded 1850)
  B: "Smith was born in 1885 in Liverpool." (related entity: Smith — can't found a company 35 years before being born)

  A: "The X war ended with treaty Y in 1763."   (entity: war, fact: ended 1763)
  B: "Treaty Y, signed in 1781, was the first major peace agreement of the era." (related entity: treaty Y — different signing year)

Requirements:
- The new sentence should be about an entity that is mentioned in passage B but is DIFFERENT from the primary subject of the claim in A.
- The contradiction should require a small inference step (a date, role, sequence, or attribute that A and B together cannot both be true about).
- Wikipedia style: neutral, no first person.
- Length similar to the original sentence in B (±50%).
- Stand alone — do not start with "However", "But", "It", "This", "These".

Return ONLY the single rewritten sentence — no preamble, no quotes, no explanation.

Passage A:
{passage_a}

Claim in A to contradict:
{claim_a}

Passage B (where you will rewrite one sentence):
{passage_b}

Original sentence in B to rewrite:
{sentence_b}

Rewritten sentence:"""


# ───────────────────────── LLM-response cleanup ────────────────────────────

def clean_response(text):
    """Strip quotes / preamble / multi-line junk; reject too-short outputs."""
    if not text:
        return None
    text = text.strip()
    if text.startswith(('"', "'")) and text.endswith(('"', "'")):
        text = text[1:-1].strip()
    text = re.sub(r"^(Rewritten sentence|Rewritten|Modified|Contradicting):\s*",
                  "", text, flags=re.IGNORECASE)
    if text:
        text = text.splitlines()[0].strip()
    if not text or len(text.split()) < 4:
        return None
    return text


# ─────────────── pair assembly: pick (A, B, S_A, S_B) tuples ───────────────

def _pick_target_sentence_in_b(claims: list[str], all_sentences: list[str]) -> str | None:
    """Choose which sentence in B to rewrite.

    Prefer claim-bearing sentences (more semantic content for the LLM to
    anchor a contradiction). Fall back to any sentence ≥ 8 words. Pick the
    LONGEST such sentence — usually the most fact-dense.
    """
    pool = claims if claims else [s for s in all_sentences if len(s.split()) >= 8]
    if not pool:
        return None
    return max(pool, key=lambda s: len(s.split()))


def _pick_anchor_claim(claims: list[str]) -> str:
    """Of A's claim sentences, prefer the longest non-junky one.

    The claim filter occasionally lets through table-like residue (e.g.
    'C - ABA Champions 1969 ABA All-Star Game...'). We can spot these by
    abnormal punctuation density; preferring the longest natural-prose
    sentence weeds them out.
    """
    def score(s):
        # Penalize sentences with unusual punctuation density (signals
        # table/list residue).
        n_words = len(s.split())
        n_special = sum(1 for c in s if c in '|/\\[]{}()<>=*')
        density = n_special / max(n_words, 1)
        return n_words * (1.0 if density < 0.3 else 0.3)
    return max(claims, key=score)


def _b_mentions_a_entity(b_body: str, a_title: str) -> bool:
    """Cheap test for 'B is about A's entity': A's title appears in B's body.

    Case-insensitive substring. Avoids hits where the title is just a common
    word by requiring at least 4 characters.
    """
    if len(a_title) < 4:
        return False
    return a_title.lower() in b_body.lower()


def assemble_pair(searcher: BM25Searcher, rng: random.Random,
                  mode: str, bm25_k: int = 30,
                  max_attempts: int = 10) -> dict | None:
    """Pick one (A, B, S_A, S_B) tuple ready for LLM rewriting.

    `mode` is "same" or "related" — controls how chunk B is filtered:
      same    — B must mention A's primary entity (article title) in its
                body, so the LLM-rewritten sentence stays topically anchored
                in B rather than reading as a random factoid splice.
      related — B must NOT mention A's title (different entity), but is
                topically close per BM25 rank. The contradiction will rely
                on cross-chunk inference rather than direct entity overlap.

    Retries up to `max_attempts` times with fresh chunk-A picks if the BM25
    filter yields no candidate B. Returns None only if every attempt fails.
    """
    for _ in range(max_attempts):
        a = pick_chunk_with_claim(searcher, rng=rng)
        if a is None:
            continue
        s_a = _pick_anchor_claim(a["claims"])

        # Bias the BM25 query toward A's primary entity by prepending A's
        # title. Without this, generic-claim queries (numbers + filler
        # words) drift to off-topic chunks that share tokens.
        query = f"{a['title']} {s_a}"
        related = find_related_chunks(
            searcher, query, k=bm25_k,
            exclude_titles={a["title"]},
            require_claim=True,
        )
        if not related:
            continue

        # Apply mode-specific filter to candidate B chunks.
        if mode == "same":
            cands = [b for b in related if _b_mentions_a_entity(b["body"], a["title"])]
        else:  # related
            cands = [b for b in related if not _b_mentions_a_entity(b["body"], a["title"])]
        if not cands:
            continue
        b = cands[0]
        s_b = _pick_target_sentence_in_b(b["claims"], split_sentences(b["body"]))
        if s_b is None:
            continue

        return {
            "chunk_a": a,
            "chunk_b": b,
            "s_a": s_a,
            "s_b": s_b,
            "mode": mode,
        }
    return None


# ─────────────────── batched LLM rewrite + splice into B ───────────────────

def generate_rewrites(pairs: list[dict], modes: list[str], model: str,
                      max_concurrent: int) -> list[str | None]:
    """Batch-call the LLM to produce S_B' for each (pair, mode).

    Returns list of rewritten sentences (or None on failure / unusable
    output) aligned with `pairs`.
    """
    from corpus_reasoning.lib.llm_request_client import ParallelResponsesClient
    client = ParallelResponsesClient(max_concurrent=max_concurrent, use_cache=True)

    prompts = []
    for p, m in zip(pairs, modes):
        tmpl = SAME_ENTITY_PROMPT if m == "same" else RELATED_ENTITY_PROMPT
        prompts.append(tmpl.format(
            passage_a=p["chunk_a"]["body"],
            claim_a=p["s_a"],
            passage_b=p["chunk_b"]["body"],
            sentence_b=p["s_b"],
        ))

    print(f"Rewriting {len(prompts)} sentences with {model}...")
    responses = client.run(model=model, prompts=prompts,
                           temperature=0.7, max_output_tokens=300)
    out: list[str | None] = []
    n_fail = 0
    for p, r in zip(pairs, responses):
        if not r.get("success", True):
            out.append(None); n_fail += 1; continue
        cleaned = clean_response(r.get("response") or "")
        if not cleaned or cleaned.strip() == p["s_b"].strip():
            out.append(None); n_fail += 1
        else:
            out.append(cleaned)
    print(f"  {len(out) - n_fail} usable rewrites, {n_fail} failures/dupes")
    return out


def splice_rewrite(body: str, s_b: str, s_b_prime: str) -> str:
    """Replace the target sentence inside chunk B's body with the rewrite.

    Tries exact substring replacement; falls back to fuzzier matching by
    normalizing whitespace. If nothing matches, prepend the rewrite at the
    start of the body — this keeps the contradiction in the chunk even when
    splicing fails, but the result will read slightly awkwardly. Caller
    can drop those examples by checking the `splice_status` field.
    """
    if s_b in body:
        return body.replace(s_b, s_b_prime, 1)
    # Whitespace-normalize and retry.
    norm = re.compile(r"\s+")
    body_norm = norm.sub(" ", body)
    s_b_norm = norm.sub(" ", s_b)
    if s_b_norm in body_norm:
        # Find position in normalized space, then map back roughly.
        idx = body_norm.find(s_b_norm)
        # Roughly: walk through original body counting non-whitespace chars.
        target = idx
        orig_start = 0; seen = 0
        for i, ch in enumerate(body):
            if seen >= target:
                orig_start = i; break
            if not ch.isspace() or (orig_start > 0 and not body[i - 1].isspace()):
                seen += 1
        # Just do a normalized-space replace and reconstruct — simpler.
        return norm.sub(" ", body).replace(s_b_norm, s_b_prime, 1)
    # Fallback: prepend.
    return s_b_prime + " " + body


# ─────────────────────── example construction ─────────────────────────────

def build_example(gold_pairs: list[tuple[dict, dict]],
                  gold_distractors: list[dict],
                  fillers_iter, num_docs: int, rng: random.Random,
                  exclude_titles: set[str]) -> dict | None:
    """Assemble one example.

    gold_pairs: list of (chunk_a_dict, chunk_b_modified_dict). chunk_a_dict
        and chunk_b_modified_dict are {title, body} — only the body is
        emitted; title is used for filler dedup.
    gold_distractors: extra chunks from the gold-pair articles (A's article
        and B's article, neighbors of the gold chunks). They share titles
        with gold chunks and don't enter `gold_doc_indices`.
    fillers_iter: iterator yielding clustered filler chunk dicts.
    exclude_titles: titles to keep out of fillers (gold-pair article titles
        already excluded automatically).
    """
    statements = []
    pair_indices: list[tuple[int, int]] = []
    for a, b in gold_pairs:
        idx_a = len(statements); statements.append(a)
        idx_b = len(statements); statements.append(b)
        pair_indices.append((idx_a, idx_b))

    # Gold-pair distractors are unmodified neighbors from A's and B's
    # articles. They share titles with gold chunks; that's intentional —
    # the model can no longer shortcut by "find the only same-topic pair."
    gold_titles = {c["title"] for c in statements}
    for d in gold_distractors:
        statements.append(d)

    # Fillers must come from articles outside any gold title.
    seen_titles = set(exclude_titles) | gold_titles
    need = num_docs - len(statements)
    while need > 0:
        try:
            f = next(fillers_iter)
        except StopIteration:
            return None
        if f["title"] in seen_titles:
            continue
        statements.append(f)
        # Don't add to seen_titles: clustered fillers from the same article
        # are explicitly allowed (the iterator already groups them).
        need -= 1

    order = list(range(len(statements)))
    rng.shuffle(order)
    old_to_new = {old: new + 1 for new, old in enumerate(order)}

    documents = [{"text": statements[order[i]]["body"]} for i in range(len(order))]
    gold_pairs_remapped = sorted(
        sorted([old_to_new[a], old_to_new[b]]) for a, b in pair_indices
    )

    return {
        "documents": documents,
        "queries": [],
        "answers": [],
        "gold_doc_indices": gold_pairs_remapped,
        "source": "wiki_contradiction",
    }


# ──────────────────────────────── main ────────────────────────────────────

def _modes_for_pair(entity_mode: str, n_pairs: int, rng: random.Random) -> list[str]:
    if entity_mode == "same":
        return ["same"] * n_pairs
    if entity_mode == "related":
        return ["related"] * n_pairs
    # mix: half each, plus parity coin-flip
    half = n_pairs // 2
    modes = ["same"] * half + ["related"] * (n_pairs - half)
    rng.shuffle(modes)
    return modes


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--num-docs", type=int, default=20,
                    help="Total chunks per example. Default 20 keeps the "
                         "example length manageable (~3000 tokens).")
    ap.add_argument("--num-contradictions", type=int, default=3,
                    help="Gold pairs per example.")
    ap.add_argument("--num-train", type=int, default=2000)
    ap.add_argument("--num-eval", type=int, default=300)
    ap.add_argument("--entity-mode", choices=["same", "related", "mix"],
                    default="mix")
    ap.add_argument("--bm25-k", type=int, default=30,
                    help="BM25 search depth when picking chunk B.")
    ap.add_argument("--gold-article-distractors", type=int, default=1,
                    help="Per gold pair, fetch this many extra chunks from "
                         "A's article AND from B's article as distractors. "
                         "Defeats the 'only-pair-with-similar-topic' shortcut. "
                         "Default 1 = 1 extra chunk per side per pair.")
    ap.add_argument("--filler-cluster-size", type=int, default=2,
                    help="Filler chunks are drawn in runs of this size from "
                         "a single article (so non-gold chunks also cluster "
                         "topically). 1 = singleton random fillers.")
    ap.add_argument("--filler-pool-size", type=int, default=20000,
                    help="How many random Wikipedia chunks to pre-fetch as "
                         "the filler pool. Must cover "
                         "~num_docs * (num_train + num_eval) chunks.")
    ap.add_argument("--model", type=str, default="gemini-2.5-flash")
    ap.add_argument("--max-concurrent", type=int, default=25)
    ap.add_argument("--output-dir", type=str, default="data")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    n_examples = args.num_train + args.num_eval

    print("Loading wikipedia-dpr-100w index...")
    searcher = BM25Searcher()

    # ── Pass 1: assemble (A, B, S_A, S_B) tuples for every gold pair. ──────
    print(f"Assembling {n_examples} examples × {args.num_contradictions} "
          f"pairs = {n_examples * args.num_contradictions} pair candidates...")
    per_example_pairs: list[list[dict]] = []
    flat_pairs: list[dict] = []
    flat_modes: list[str] = []
    for _ in tqdm(range(n_examples), desc="assemble pairs"):
        modes = _modes_for_pair(args.entity_mode,
                                args.num_contradictions, rng)
        ex_pairs = []
        for m in modes:
            p = assemble_pair(searcher, rng, mode=m, bm25_k=args.bm25_k)
            if p is None:
                # Skip this slot; example will get fewer than K pairs.
                continue
            ex_pairs.append(p)
            flat_pairs.append(p)
            flat_modes.append(m)
        per_example_pairs.append(ex_pairs)

    # ── Pass 2: batch-call the LLM for all pairs at once. ─────────────────
    rewrites = generate_rewrites(flat_pairs, flat_modes,
                                 args.model, args.max_concurrent)

    # Distribute rewrites back per example; build the modified chunk B and
    # fetch in-article distractors for both A and B (unmodified neighbors).
    pair_iter = iter(rewrites)
    final_pairs_per_example: list[list[tuple[dict, dict]]] = []
    distractors_per_example: list[list[dict]] = []
    n_dropped_pairs = 0
    for ex_pairs in tqdm(per_example_pairs, desc="splice + neighbors"):
        out_pairs = []
        out_distractors: list[dict] = []
        for p in ex_pairs:
            r = next(pair_iter)
            if r is None:
                n_dropped_pairs += 1
                continue
            modified_b_body = splice_rewrite(p["chunk_b"]["body"], p["s_b"], r)
            out_pairs.append((
                {"title": p["chunk_a"]["title"], "body": p["chunk_a"]["body"]},
                {"title": p["chunk_b"]["title"], "body": modified_b_body},
            ))
            if args.gold_article_distractors > 0:
                a_neighbors = fetch_article_neighbors(
                    searcher,
                    center_lid=p["chunk_a"]["lid"],
                    center_title=p["chunk_a"]["title"],
                    n_extra=args.gold_article_distractors,
                )
                b_neighbors = fetch_article_neighbors_by_title(
                    searcher,
                    title=p["chunk_b"]["title"],
                    n_extra=args.gold_article_distractors,
                    exclude_docids={p["chunk_b"]["docid"]},
                )
                for n in a_neighbors + b_neighbors:
                    out_distractors.append({"title": n["title"], "body": n["body"]})
        final_pairs_per_example.append(out_pairs)
        distractors_per_example.append(out_distractors)
    print(f"  {n_dropped_pairs} pair slots dropped (LLM unusable).")

    # ── Pass 3: pre-fetch a clustered filler pool, build examples. ────────
    if args.filler_cluster_size <= 1:
        print(f"Pre-fetching {args.filler_pool_size} singleton filler chunks...")
        filler_pool = sample_random_chunks(
            searcher, args.filler_pool_size, rng=rng,
        )
    else:
        print(f"Pre-fetching {args.filler_pool_size} clustered filler chunks "
              f"(cluster_size={args.filler_cluster_size})...")
        filler_pool = sample_clustered_fillers(
            searcher, args.filler_pool_size, args.filler_cluster_size, rng=rng,
        ) or []
    # Don't rng.shuffle: keeping clusters contiguous in the iterator means
    # each example gets coherent same-article runs as fillers.

    examples: list[dict] = []
    fillers_iter = iter(filler_pool)
    n_dropped_examples = 0
    for ex_pairs, ex_distractors in tqdm(
        zip(final_pairs_per_example, distractors_per_example),
        total=len(final_pairs_per_example), desc="build examples",
    ):
        if not ex_pairs:
            n_dropped_examples += 1
            continue
        exclude_titles = {c["title"] for pair in ex_pairs for c in pair}
        ex = build_example(ex_pairs, ex_distractors, fillers_iter,
                           args.num_docs, rng, exclude_titles)
        if ex is None:
            n_dropped_examples += 1
            continue
        examples.append(ex)
    print(f"  built {len(examples)} examples; dropped {n_dropped_examples}.")

    # ── Split, save. ──────────────────────────────────────────────────────
    rng.shuffle(examples)
    train = examples[:args.num_train]
    evald = examples[args.num_train:args.num_train + args.num_eval]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"wiki_{args.entity_mode}_n{args.num_docs}_k{args.num_contradictions}"
    train_path = out_dir / f"contradiction_train_{tag}.jsonl"
    eval_path = out_dir / f"contradiction_eval_{tag}.jsonl"
    save_jsonl(str(train_path), train)
    save_jsonl(str(eval_path), evald)
    print_dataset_stats(train, "train", str(train_path))
    print_dataset_stats(evald, "eval", str(eval_path))


if __name__ == "__main__":
    main()
