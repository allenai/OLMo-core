"""
BM25, behind one interface. Lazily imported, never at module scope.

Two backends, because the suite genuinely needs both: a **prebuilt** index for the corpora
retrieval questions are asked against (``wikipedia-dpr-100w``, ``msmarco-v1-passage``) and a
**local** index built once over a caller-supplied corpus (BEIR, which ships its own). The
pre-migration tree had these two wrappers plus three sites that opened
``LuceneSearcher.from_prebuilt_index`` directly and re-implemented the hit conversion each time.

Answer exclusion is stricter than a substring check, and deliberately so -- see
:func:`reveals_answer`.
"""

from __future__ import annotations

import json
import random
import re
from typing import Dict, Iterable, List, Optional, Sequence, Set

__all__ = [
    "WIKI_INDEX",
    "reveals_answer",
    "word_jaccard",
    "pick_hard_negatives",
    "PrebuiltBM25Searcher",
    "LocalBM25Searcher",
]

#: The prebuilt pyserini index every Wikipedia-backed task retrieves from. ~21M 100-word passages.
WIKI_INDEX = "wikipedia-dpr-100w"

_WORD_RE = re.compile(r"\w+")

#: Tokens shorter than this do not count toward answer coverage. Drops articles and pronouns, so
#: the answer "Big Ben" is not diluted by "big" matching an unrelated hit.
_MIN_ANSWER_TOKEN_LEN = 3


def _words(text: str) -> Set[str]:
    return set(_WORD_RE.findall(text.lower()))


def word_jaccard(a: Set[str], b: Set[str]) -> float:
    """
    :param a: Word set.
    :param b: Word set.

    :returns: Jaccard similarity, 0.0 when both are empty.
    """
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def reveals_answer(text: str, answers: Iterable[str], token_coverage: float = 0.6) -> bool:
    """
    Whether a passage gives the answer away, and so cannot be used as a distractor.

    Stricter than a substring test on purpose. The classic miss: answer ``"Marvin John Heemayer"``,
    passage ``"... Marvin Heemayer ..."`` -- the middle name is absent so the substring does not
    match, and a reader would still say the passage reveals the answer. Either rule fires:

    1. the full answer appears verbatim, or
    2. at least ``token_coverage`` of the answer's significant tokens (length >= 3) appear.

    :param text: The candidate passage.
    :param answers: Answer aliases.
    :param token_coverage: Fraction of significant tokens that must appear for rule 2. 0.6 means
        2-of-3.

    :returns: True when the passage must be excluded.
    """
    lowered = text.lower()
    hit_tokens: Optional[Set[str]] = None
    for answer in answers:
        if not answer:
            continue
        if answer.lower() in lowered:
            return True
        tokens = [t for t in _WORD_RE.findall(answer.lower()) if len(t) >= _MIN_ANSWER_TOKEN_LEN]
        if not tokens:
            continue
        if hit_tokens is None:
            hit_tokens = _words(text)
        if sum(1 for t in tokens if t in hit_tokens) / len(tokens) >= token_coverage:
            return True
    return False


def pick_hard_negatives(
    hits: Sequence[Dict],
    num_hard: int,
    *,
    exclude_answers: Optional[Sequence[str]] = None,
    exclude_docids: Optional[Set[str]] = None,
    max_pair_overlap: Optional[float] = 0.5,
) -> List[Dict]:
    """
    Walk BM25 hits and take the hardest ones that are actually negatives.

    :param hits: ``{docid, text, score}`` dicts in BM25 rank order.
    :param num_hard: How many to take.
    :param exclude_answers: Answer aliases; a hit revealing one is not a negative at all.
    :param exclude_docids: Ids already used (the gold).
    :param max_pair_overlap: Reject a candidate whose word-Jaccard with an already-picked negative
        exceeds this. Near-duplicate Wikipedia chunks otherwise fill the pool with one passage
        wearing several hats. ``None`` disables.

    :returns: Up to ``num_hard`` ``{docid, text}`` dicts, hardest first.
    """
    used = set(exclude_docids or ())
    picked: List[Dict] = []
    picked_words: List[Set[str]] = []
    for hit in hits:
        if hit["docid"] in used:
            continue
        if exclude_answers and reveals_answer(hit["text"], exclude_answers):
            continue
        if max_pair_overlap is not None:
            words = _words(hit["text"])
            if any(word_jaccard(words, prev) > max_pair_overlap for prev in picked_words):
                continue
            picked_words.append(words)
        picked.append({"docid": hit["docid"], "text": hit["text"]})
        used.add(hit["docid"])
        if len(picked) >= num_hard:
            break
    return picked


def _hit_to_dict(hit) -> Dict:
    raw = json.loads(hit.lucene_document.get("raw"))
    return {
        "docid": hit.docid,
        "text": raw.get("contents", raw.get("body", "")),
        "score": hit.score,
    }


class PrebuiltBM25Searcher:
    """BM25 over one of pyserini's prebuilt indexes."""

    def __init__(self, index_name: str = WIKI_INDEX, k1: float = 0.9, b: float = 0.4) -> None:
        """
        :param index_name: Prebuilt index id; downloads on first use.
        :param k1: BM25 term-frequency saturation.
        :param b: BM25 length normalisation.

        :raises ImportError: If pyserini is missing; install ``ctc[sources]`` plus pyserini.
        """
        try:
            from pyserini.search.lucene import LuceneSearcher
        except ImportError as e:  # pragma: no cover - depends on the install
            raise ImportError("BM25 mining needs pyserini: pip install pyserini") from e
        self.searcher = LuceneSearcher.from_prebuilt_index(index_name)
        self.searcher.set_bm25(k1=k1, b=b)
        self.num_docs = self.searcher.num_docs

    def batch_search(self, queries: Sequence[str], k: int = 200, threads: int = 8) -> List[List]:
        """
        :param queries: Query strings.
        :param k: Search depth.
        :param threads: Lucene's own thread pool size -- 4-8x faster than per-query calls.

        :returns: One hit list per query, in input order.
        """
        qids = [str(i) for i in range(len(queries))]
        raw = self.searcher.batch_search(list(queries), qids, k=k, threads=threads)
        return [[_hit_to_dict(h) for h in raw.get(qid, [])] for qid in qids]

    def passage(self, docid: str) -> Optional[str]:
        """
        :param docid: Document id, or a Lucene ordinal.

        :returns: Its text, or ``None`` when the id is absent.
        """
        doc = self.searcher.doc(docid)
        if doc is None:
            return None
        raw = doc.raw()
        try:
            obj = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return raw
        return obj.get("contents") or obj.get("text") or raw

    def sample_pool(self, size: int, *, seed: int = 42, threads: int = 16) -> List[str]:
        """
        Pre-fetch random passages once, in parallel, to sample distractors from in pure Python.

        Sampling 100+ distractors per example over thousands of examples means hundreds of
        thousands of serial Lucene ``doc()`` lookups, which dominate wall time. One threaded
        pre-fetch is ~30-50x faster end to end.

        :param size: Passages to fetch.
        :param seed: Which Lucene ordinals to draw.
        :param threads: Fetch threads; ``IndexReader.document()`` is thread-safe.

        :returns: Passage texts.
        """
        from concurrent.futures import ThreadPoolExecutor

        rng = random.Random(seed)
        ordinals = rng.sample(range(self.num_docs), min(size, self.num_docs))
        with ThreadPoolExecutor(max_workers=threads) as pool:
            texts = list(pool.map(self.passage, ordinals))
        return [t for t in texts if t]


class LocalBM25Searcher:
    """BM25 over a caller-supplied corpus, indexed once and cached on disk."""

    def __init__(self, corpus, index_dir: str, k1: float = 0.9, b: float = 0.4, threads: int = 8):
        """
        :param corpus: :class:`~ctc.data.sources.retrieval.Candidate` objects to index.
        :param index_dir: Cache location; the build is skipped when ``_SUCCESS`` is present.
        :param k1: BM25 term-frequency saturation.
        :param b: BM25 length normalisation.
        :param threads: Indexing threads.

        :raises ImportError: If pyserini is missing.
        """
        try:
            from pyserini.search.lucene import LuceneSearcher
        except ImportError as e:  # pragma: no cover - depends on the install
            raise ImportError("BM25 mining needs pyserini: pip install pyserini") from e
        path = self._build_index(corpus, index_dir, threads)
        self.searcher = LuceneSearcher(path)
        self.searcher.set_bm25(k1=k1, b=b)
        self.num_docs = self.searcher.num_docs

    @staticmethod
    def _build_index(corpus, index_dir: str, threads: int) -> str:
        import subprocess
        import sys
        import tempfile
        from pathlib import Path

        directory = Path(index_dir)
        marker = directory / "_SUCCESS"
        if marker.exists():
            return str(directory)
        directory.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory() as tmp:
            with open(Path(tmp) / "corpus.jsonl", "w", encoding="utf-8") as f:
                for doc in corpus:
                    f.write(json.dumps({"id": str(doc.id), "contents": doc.text}) + "\n")
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pyserini.index.lucene",
                    "--collection",
                    "JsonCollection",
                    "--input",
                    tmp,
                    "--index",
                    str(directory),
                    "--generator",
                    "DefaultLuceneDocumentGenerator",
                    "--threads",
                    str(threads),
                    "--storePositions",
                    "--storeDocvectors",
                    "--storeRaw",
                ],
                check=True,
            )
        marker.write_text("done\n", encoding="utf-8")
        return str(directory)

    def batch_search(self, queries: Sequence[str], k: int = 100, threads: int = 8) -> List[List]:
        """
        :param queries: Query strings.
        :param k: Search depth.
        :param threads: Lucene thread pool size.

        :returns: One ``{docid, score}`` list per query, in input order.
        """
        qids = [str(i) for i in range(len(queries))]
        raw = self.searcher.batch_search(list(queries), qids, k=k, threads=threads)
        return [[{"docid": h.docid, "score": h.score} for h in raw.get(qid, [])] for qid in qids]
