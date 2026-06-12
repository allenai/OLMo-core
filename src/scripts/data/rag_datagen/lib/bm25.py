"""BM25 search utility using pyserini's pre-built wikipedia-dpr-100w index.

Provides hard negative mining (and gold passage retrieval for NQ) for both
single-query and batched data generation. The wikipedia-dpr-100w index
contains ~21M Wikipedia passages, each ~100 words.

Requires: pyserini (pip install pyserini) in the corpus-reasoning-eval env.

Usage:
    from lib.bm25 import BM25Searcher

    searcher = BM25Searcher()  # downloads index on first use

    # Single-query hard-negative mining (HotpotQA path)
    hard_negs = searcher.mine_distractors(
        query="who is the president", num_hard=5,
        exclude_answers=["Biden"], max_pair_overlap=0.5,
    )

    # Single-query fused gold + hard-neg retrieval (NQ path)
    gold, gold_docid, hard_negs = searcher.find_gold_and_hard_negs(
        query="who is the president", answers=["Biden"],
        num_hard=5, max_pair_overlap=0.5,
    )

    # Batched variants use pyserini's internal Java thread pool for a big speedup.
    gold_batch, hard_batch = searcher.batch_find_gold_and_hard_negs(
        questions=[...], answers_list=[[...], ...],
        num_hard=5, max_pair_overlap=0.5, threads=8,
    )
"""

import json
import random
import re
from concurrent.futures import ThreadPoolExecutor

from pyserini.search.lucene import LuceneSearcher

INDEX_NAME = "wikipedia-dpr-100w"

_WORD_RE = re.compile(r"\w+")

# Minimum token length to count as "significant" during answer-exclusion.
# Drops articles, pronouns, and other short function words so that e.g. the
# answer "Big Ben" isn't diluted by the token "big" being matched to unrelated
# hits. 3 is a reasonable balance — it keeps "Ben", "USA", "FBI" but drops
# "a", "of", "the", "in".
_MIN_ANSWER_TOKEN_LEN = 3


def _word_set(text: str) -> set[str]:
    return set(_WORD_RE.findall(text.lower()))


def _significant_answer_tokens(answer: str) -> list[str]:
    return [t for t in _WORD_RE.findall(answer.lower()) if len(t) >= _MIN_ANSWER_TOKEN_LEN]


def word_jaccard(a_words: set[str], b_words: set[str]) -> float:
    """Jaccard similarity between two sets of lowercased words."""
    if not a_words and not b_words:
        return 0.0
    return len(a_words & b_words) / len(a_words | b_words)


def hit_contains_answer(hit_text: str, answers, token_coverage_threshold: float = 0.6) -> bool:
    """True if a hit text should be excluded because it covers an answer.

    Stricter than a plain substring check: also catches partial forms of
    multi-token answers. The classic failure mode: answer =
    "Marvin John Heemayer", hit = "... Marvin Heemayer ..." — the middle
    name is missing so the exact substring doesn't match, but a reader would
    still say the hit reveals the answer.

    Two rules; triggering either filters the hit:
    1. **Exact substring**: the full answer (lowercased) appears in the hit.
    2. **Token coverage**: for each answer, ≥ `token_coverage_threshold` of
       the answer's significant tokens (len ≥ 3) appear somewhere in the
       hit. 0.6 means e.g. 2-of-3 tokens must match, catching
       "Marvin Heemayer" for the 3-token "Marvin John Heemayer".

    Rule (1) is a superset of rule (2) for single-token answers, so
    short answers still work as before. For single-significant-token
    answers, rule (2) requires that one token to appear — same as the
    substring check.
    """
    text_lower = hit_text.lower()
    hit_token_set: set[str] | None = None  # built lazily

    for ans in answers:
        if not ans:
            continue
        if ans.lower() in text_lower:
            return True
        ans_tokens = _significant_answer_tokens(ans)
        if not ans_tokens:
            continue
        if hit_token_set is None:
            hit_token_set = _word_set(hit_text)
        matched = sum(1 for t in ans_tokens if t in hit_token_set)
        if matched / len(ans_tokens) >= token_coverage_threshold:
            return True
    return False


def _hit_to_dict(hit) -> dict:
    raw = json.loads(hit.lucene_document.get("raw"))
    return {
        "text": raw.get("contents", raw.get("body", "")),
        "docid": hit.docid,
        "score": hit.score,
    }


# Minimum search depth for any BM25 query; below this Lucene's warmup cost
# dominates and we might as well retrieve more candidates anyway.
_MIN_SEARCH_DEPTH = 50


def compute_search_depth(num_hard: int, max_pair_overlap) -> int:
    """How deep into BM25 results to scan.

    Overshoots num_hard so answer-containing / near-duplicate rejections don't
    starve the pool. The `*5` multiplier under pair-overlap filtering was
    validated empirically against `num_hard=10`: rejection rate stays under
    ~20%% and the scan terminates early once num_hard is hit.
    """
    multiplier = 5 if max_pair_overlap is not None else 3
    return max(num_hard * multiplier, _MIN_SEARCH_DEPTH)


def pick_gold_from_hits(hits: list[dict], answers) -> tuple[dict | None, str | None]:
    """First BM25 hit whose text contains any answer string (lowercased substring)."""
    for hit in hits:
        text_lower = hit["text"].lower()
        if any(ans.lower() in text_lower for ans in answers):
            return {"title": None, "text": hit["text"]}, hit["docid"]
    return None, None


def sample_from_pool(pool, num, exclude_answers=None, exclude_texts=None, rng=None):
    """Sample `num` distinct passages from a pre-fetched pool, in Python.

    Companion to `BM25Searcher.prefetch_random_pool`. Walks a shuffled view
    of the pool, skipping passages in `exclude_texts` or that contain any
    `exclude_answers` (per `hit_contains_answer`). Cheap when num << len(pool).
    """
    if num <= 0:
        return []
    if rng is None:
        rng = random.Random()
    exclude_texts = set(exclude_texts or [])
    indices = list(range(len(pool)))
    rng.shuffle(indices)
    picked = []
    for i in indices:
        if len(picked) >= num:
            break
        d = pool[i]
        if d["text"] in exclude_texts:
            continue
        if exclude_answers and hit_contains_answer(d["text"], exclude_answers):
            continue
        picked.append(d)
        exclude_texts.add(d["text"])
    return picked


def pick_hard_negs_from_hits(
    hits: list[dict],
    num_hard: int,
    exclude_answers=None,
    exclude_docids=None,
    max_pair_overlap=None,
) -> list[dict]:
    """Walk BM25 hits and pick up to `num_hard` hard negatives.

    Drops hits that: (1) are in `exclude_docids`, (2) appear to reveal an
    answer (strict token-coverage filter, see `hit_contains_answer`), or
    (3) exceed `max_pair_overlap` Jaccard with an already-picked hard neg.
    """
    exclude_docids = set(exclude_docids or [])
    picked: list[dict] = []
    picked_word_sets: list[set[str]] = []
    filter_pair = max_pair_overlap is not None

    for hit in hits:
        if hit["docid"] in exclude_docids:
            continue
        if exclude_answers and hit_contains_answer(hit["text"], exclude_answers):
            continue
        if filter_pair:
            hit_words = _word_set(hit["text"])
            if any(
                word_jaccard(hit_words, prev) > max_pair_overlap
                for prev in picked_word_sets
            ):
                continue
            picked_word_sets.append(hit_words)
        picked.append({"title": None, "text": hit["text"]})
        exclude_docids.add(hit["docid"])
        if len(picked) >= num_hard:
            break
    return picked


class BM25Searcher:
    """Wrapper around pyserini's LuceneSearcher for BM25 distractor mining."""

    def __init__(self, index_name=INDEX_NAME, k1=0.9, b=0.4, no_titles=False):
        print(f"Loading pyserini index '{index_name}'... (downloads on first use)")
        self.searcher = LuceneSearcher.from_prebuilt_index(index_name)
        self.searcher.set_bm25(k1=k1, b=b)
        self.num_docs = self.searcher.num_docs
        self.no_titles = no_titles
        print(f"  Index loaded: {self.num_docs:,} passages")

    def _make_doc(self, text):
        return {"title": None, "text": text}

    # ---------- Low-level search ----------

    def search(self, query, k=100) -> list[dict]:
        """Search and return list of {text, docid, score} dicts."""
        return [_hit_to_dict(h) for h in self.searcher.search(query, k=k)]

    def batch_search(self, queries: list[str], k: int = 100, threads: int = 8) -> list[list[dict]]:
        """Run BM25 search over many queries in parallel using Lucene's thread pool.

        Returns a list of hit-lists, one per input query, preserving input order.
        """
        qids = [str(i) for i in range(len(queries))]
        raw = self.searcher.batch_search(queries, qids, k=k, threads=threads)
        return [[_hit_to_dict(h) for h in raw.get(qid, [])] for qid in qids]

    def sample_random_passages(self, num, exclude_docids=None, exclude_answers=None, rng=None) -> list[dict]:
        """Uniform-random passages from the corpus, honoring exclusions."""
        if num <= 0:
            return []
        if rng is None:
            rng = random.Random()
        exclude_docids = set(exclude_docids or [])
        picked: list[dict] = []
        attempts = 0
        max_attempts = num * 5
        while len(picked) < num and attempts < max_attempts:
            lucene_id = rng.randint(0, self.num_docs - 1)
            doc = self.searcher.doc(lucene_id)
            if doc is None:
                attempts += 1
                continue
            docid = doc.docid()
            if docid in exclude_docids:
                attempts += 1
                continue
            raw = json.loads(doc.raw())
            text = raw.get("contents", raw.get("body", ""))
            if exclude_answers and hit_contains_answer(text, exclude_answers):
                exclude_docids.add(docid)
                attempts += 1
                continue
            picked.append(self._make_doc(text))
            exclude_docids.add(docid)
            attempts += 1
        return picked

    def get_passage_by_docid(self, docid):
        """Fetch a single passage by its document ID."""
        doc = self.searcher.doc(str(docid))
        if doc is None:
            return None
        raw = json.loads(doc.raw())
        return self._make_doc(raw.get("contents", raw.get("body", "")))

    def prefetch_random_pool(self, pool_size, threads=16, seed=None):
        """Pre-fetch a pool of random passages in parallel.

        Replaces per-example `sample_random_passages` for high-volume data
        gen: sampling 100+ random distractors per example × thousands of
        examples means hundreds of thousands of serial Lucene `doc()`
        lookups, which dominate wall time. Pre-fetching once with a thread
        pool (Lucene `IndexReader.document()` is thread-safe) and then doing
        per-example Python sampling is ~30-50× faster end-to-end.

        Returns a list of {title, text} dicts. Caller filters by
        per-example answers using `sample_from_pool`.
        """
        rng = random.Random(seed)
        lucene_ids = rng.sample(range(self.num_docs), pool_size)

        def _fetch(lid):
            doc = self.searcher.doc(lid)
            if doc is None:
                return None
            raw = json.loads(doc.raw())
            return raw.get("contents", raw.get("body", ""))

        with ThreadPoolExecutor(max_workers=threads) as ex:
            texts = list(ex.map(_fetch, lucene_ids))
        return [self._make_doc(t) for t in texts if t]

    # ---------- Single-query mining (HotpotQA path) ----------

    def mine_distractors(self, query, num_hard, num_random=0,
                         exclude_answers=None, exclude_docids=None,
                         search_depth=None, rng=None, max_pair_overlap=None):
        """BM25 hard negatives + optional random corpus passages for one query.

        Used by HotpotQA generation (gold is annotated, not retrieved).
        Hard negs first, then random, returned as a single list.
        """
        if search_depth is None:
            search_depth = compute_search_depth(num_hard, max_pair_overlap)

        hard = []
        if num_hard > 0:
            hits = self.search(query, k=search_depth)
            hard = pick_hard_negs_from_hits(
                hits, num_hard,
                exclude_answers=exclude_answers,
                exclude_docids=exclude_docids,
                max_pair_overlap=max_pair_overlap,
            )

        used = set(exclude_docids or [])
        for d in hard:
            # hard negs don't expose docid; approximate dedup via text match.
            used.add(d["text"])

        random_docs = self.sample_random_passages(
            num_random,
            exclude_docids=used,
            exclude_answers=exclude_answers,
            rng=rng,
        )
        return hard + random_docs

    # ---------- Fused single-query (NQ path) ----------

    def find_gold_and_hard_negs(self, query, answers, num_hard,
                                max_pair_overlap=None, search_depth=None):
        """One BM25 search → (gold_doc, gold_docid, hard_negs).

        Replaces the old `find_gold_passage` + `mine_distractors` pair, which
        ran two separate searches over the same query. Returns (None, None, [])
        if no answer-containing passage is found within `search_depth`.
        """
        if search_depth is None:
            # Overshoot for gold search (answer match may be deep) and for
            # hard-neg filtering.
            search_depth = max(200, compute_search_depth(num_hard, max_pair_overlap))

        hits = self.search(query, k=search_depth)
        gold, gold_docid = pick_gold_from_hits(hits, answers)
        if gold is None:
            return None, None, []
        hard_negs = pick_hard_negs_from_hits(
            hits, num_hard,
            exclude_answers=answers,
            exclude_docids={gold_docid},
            max_pair_overlap=max_pair_overlap,
        )
        return gold, gold_docid, hard_negs

    # ---------- Batched (both paths) ----------

    def batch_find_gold_and_hard_negs(
        self, questions, answers_list, num_hard,
        max_pair_overlap=None, search_depth=None, threads=8,
    ):
        """Batched version of `find_gold_and_hard_negs`.

        Uses Lucene's internal thread pool for ~4-8× speedup over per-example
        calls. Returns two parallel lists: `golds` (each element is
        `(gold_doc, gold_docid)` or `(None, None)`) and `hard_neg_lists`.
        """
        if search_depth is None:
            search_depth = max(200, compute_search_depth(num_hard, max_pair_overlap))

        hits_batch = self.batch_search(questions, k=search_depth, threads=threads)

        golds: list[tuple[dict | None, str | None]] = []
        hard_neg_lists: list[list[dict]] = []
        for hits, answers in zip(hits_batch, answers_list):
            gold, gold_docid = pick_gold_from_hits(hits, answers)
            if gold is None:
                golds.append((None, None))
                hard_neg_lists.append([])
                continue
            hard_negs = pick_hard_negs_from_hits(
                hits, num_hard,
                exclude_answers=answers,
                exclude_docids={gold_docid},
                max_pair_overlap=max_pair_overlap,
            )
            golds.append((gold, gold_docid))
            hard_neg_lists.append(hard_negs)
        return golds, hard_neg_lists

    def batch_mine_hard_negs(
        self, queries, num_hard,
        exclude_answers_list=None, max_pair_overlap=None,
        search_depth=None, threads=8,
    ):
        """Batched hard-neg mining (no gold retrieval). HotpotQA path.

        `exclude_answers_list` is an optional list (one entry per query) of
        answer strings to filter from the hard-neg pool. Returns a list of
        hard-neg lists, one per input query.
        """
        if search_depth is None:
            search_depth = compute_search_depth(num_hard, max_pair_overlap)

        if exclude_answers_list is None:
            exclude_answers_list = [None] * len(queries)

        hits_batch = self.batch_search(queries, k=search_depth, threads=threads)

        out: list[list[dict]] = []
        for hits, exclude_answers in zip(hits_batch, exclude_answers_list):
            out.append(pick_hard_negs_from_hits(
                hits, num_hard,
                exclude_answers=exclude_answers,
                max_pair_overlap=max_pair_overlap,
            ))
        return out

    # ---------- Legacy single-query gold lookup ----------

    def find_gold_passage(self, query, answers, search_depth=200):
        """Search for the gold passage that contains an answer.

        Legacy API — prefer `find_gold_and_hard_negs` which fuses the two
        searches. Kept for callers that only need the gold and not the hard
        negatives.
        """
        hits = self.search(query, k=search_depth)
        return pick_gold_from_hits(hits, answers)
