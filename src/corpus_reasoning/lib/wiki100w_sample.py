"""Sampling primitives over the wikipedia-dpr-100w Lucene index.

Used by the Wikipedia variants of the reorder, outlier, and contradiction
data generators. Primitives:

  sample_article_run(n_chunks)
      Consecutive in-article chunks (for reorder). Exploits the index's
      property that adjacent Lucene ids belong to the same article — verified
      empirically in scripts/debug/probe_wiki100w_docids.py.

  sample_two_articles(n_majority, n_outlier)
      Article run of n_majority chunks + a separate run of n_outlier chunks
      from a different article (for outlier).

  pick_chunk_with_claim()
      One random chunk that contains ≥1 claim-bearing sentence (for
      contradiction; this is chunk A). Returns the chunk plus its candidate
      claim sentences.

  find_related_chunks(query, k)
      BM25 topical search returning parsed-title/body chunks (for picking
      chunk B given chunk A in the contradiction task).

  sample_random_chunks(n)
      Bulk random chunks parsed into {title, body} (for filler passages).

Chunk text format in the index (verified by the probe):
    line 1: "Article Title"   (with literal double-quotes; sometimes unquoted)
    line 2..: <chunk text, often with title repeated as a prefix on line 2>

`extract_title_and_body` splits these and strips the title-prefix-on-line-2
where present.
"""

import json
import random
import re

from corpus_reasoning.lib.bm25 import BM25Searcher, INDEX_NAME


# ───────────────────────────── text parsing ────────────────────────────────

_QUOTED_TITLE = re.compile(r'^"(.+)"\s*$')


def extract_title_and_body(raw_text: str) -> tuple[str, str]:
    """Parse the `contents` field into (title, body).

    Format: line 1 is the title (usually quoted, sometimes not); line 2+ is
    the body. The body often re-prefixes the title verbatim — strip that.
    Returns ("", body) if parsing fails (defensive — caller can still use the
    body).
    """
    if "\n" not in raw_text:
        return "", raw_text.strip()
    line1, rest = raw_text.split("\n", 1)
    line1 = line1.strip()
    m = _QUOTED_TITLE.match(line1)
    title = m.group(1) if m else line1
    body = rest.strip()
    if title and body.startswith(title):
        body = body[len(title):].lstrip(" .,:;-")
    return title, body


def _doc_at(searcher, lid: int) -> dict | None:
    """Fetch a doc by Lucene id. Returns {lid, docid, title, body, raw_text} or None."""
    doc = searcher.doc(lid)
    if doc is None:
        return None
    raw = json.loads(doc.raw())
    raw_text = raw.get("contents", raw.get("body", ""))
    title, body = extract_title_and_body(raw_text)
    return {
        "lid": lid,
        "docid": doc.docid(),
        "title": title,
        "body": body,
        "raw_text": raw_text,
    }


# ───────────────────────── article-run sampling ────────────────────────────

def _walk_run(searcher, start_lid: int, n_chunks: int, num_docs: int) -> list[dict] | None:
    """Try to assemble n_chunks consecutive same-title chunks around start_lid.

    Walks both directions from start_lid; returns the ordered list (by lid)
    or None if no contiguous run of length n_chunks exists in this article.
    """
    base = _doc_at(searcher, start_lid)
    if base is None or not base["title"]:
        return None
    title = base["title"]
    run = [base]

    # Forward
    lid = start_lid + 1
    while len(run) < n_chunks and lid < num_docs:
        d = _doc_at(searcher, lid)
        if d is None or d["title"] != title:
            break
        run.append(d)
        lid += 1
    # Backward
    lid = start_lid - 1
    while len(run) < n_chunks and lid >= 0:
        d = _doc_at(searcher, lid)
        if d is None or d["title"] != title:
            break
        run.insert(0, d)
        lid -= 1
    if len(run) < n_chunks:
        return None
    return run


def sample_article_run(
    searcher: BM25Searcher,
    n_chunks: int,
    rng: random.Random | None = None,
    max_attempts: int = 50,
    exclude_titles: set[str] | None = None,
) -> list[dict] | None:
    """Sample one article run of `n_chunks` consecutive chunks.

    Returns None if no such run can be found within `max_attempts` tries.
    Each chunk dict has: {lid, docid, title, body, raw_text}.
    """
    if rng is None:
        rng = random.Random()
    exclude_titles = exclude_titles or set()
    n = searcher.num_docs
    for _ in range(max_attempts):
        start = rng.randint(0, n - 1)
        run = _walk_run(searcher.searcher, start, n_chunks, n)
        if run is None:
            continue
        if run[0]["title"] in exclude_titles:
            continue
        return run
    return None


def sample_two_articles(
    searcher: BM25Searcher,
    n_majority: int,
    n_outlier: int,
    rng: random.Random | None = None,
    max_attempts: int = 50,
) -> tuple[list[dict], list[dict]] | None:
    """Sample (majority_run, outlier_run) from two distinct articles."""
    if rng is None:
        rng = random.Random()
    majority = sample_article_run(searcher, n_majority, rng=rng, max_attempts=max_attempts)
    if majority is None:
        return None
    outlier = sample_article_run(
        searcher, n_outlier, rng=rng, max_attempts=max_attempts,
        exclude_titles={majority[0]["title"]},
    )
    if outlier is None:
        return None
    return majority, outlier


# ───────────────────────── claim-sentence sampling ─────────────────────────

# Wikipedia-flavored claim regex. Broader than the biomedical PubMed filter:
# adds factual/biographical verbs ("born", "founded", "located", "called",
# "named"), state verbs ("is", "was", "were", "has", "had"), plus all the
# numeric/date patterns that already worked for PubMed. Tuning is empirical;
# expect ~5-15% of Wikipedia sentences to pass.
WIKI_CLAIM_TOKEN = re.compile(
    r"\b(is|was|were|are|has|had|have|"
    r"born|died|founded|established|located|situated|named|called|"
    r"became|begat|begun|built|created|built|invented|introduced|"
    r"discovered|developed|won|defeated|elected|appointed|"
    r"published|released|recorded|directed|composed|wrote|written|"
    r"served|fought|killed|murdered|signed|formed|joined|"
    r"increased?|decreased?|reduced?|elevated|higher|lower|greater|"
    r"associated|correlated|"
    r"\d{4}|"  # 4-digit years
    r"\d+(?:\.\d+)?%?|\d+-fold)\b",
    re.IGNORECASE,
)

LEADING_BAD = re.compile(
    r"^(It|This|These|Those|They|He|She|We|Our|Their|Such|However|"
    r"Moreover|Furthermore|Also|Additionally|Thus|Therefore|Hence|"
    r"Consequently|Although|Though|Instead|Meanwhile|Finally|First|"
    r"Second|Third|In\s+contrast|In\s+addition|In\s+summary|"
    r"In\s+conclusion)\b",
    re.IGNORECASE,
)

DANGLING = re.compile(r"\(Fig\.?|\(Table|\[\d+\]|\bet\s+al\.|supplement", re.IGNORECASE)

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


def split_sentences(body: str) -> list[str]:
    """Same regex split as the PubMed pipeline."""
    return [s.strip() for s in _SENT_SPLIT.split(body.strip()) if s.strip()]


def is_wiki_claim_sentence(s: str) -> bool:
    """Wikipedia version of is_claim_sentence (PubMed pipeline)."""
    s = s.strip()
    if not s.endswith("."):
        return False
    if "?" in s:
        return False
    words = s.split()
    if not (8 <= len(words) <= 40):
        return False
    if LEADING_BAD.match(s):
        return False
    if DANGLING.search(s):
        return False
    if not WIKI_CLAIM_TOKEN.search(s):
        return False
    if not s[0].isupper():
        return False
    return True


def claim_sentences(body: str) -> list[str]:
    """All claim-bearing sentences in a chunk body."""
    return [s for s in split_sentences(body) if is_wiki_claim_sentence(s)]


def pick_chunk_with_claim(
    searcher: BM25Searcher,
    rng: random.Random | None = None,
    max_attempts: int = 50,
    exclude_lids: set[int] | None = None,
    exclude_titles: set[str] | None = None,
) -> dict | None:
    """Sample a random chunk that contains ≥1 claim-bearing sentence.

    Returns {lid, docid, title, body, raw_text, claims:[str]} or None if no
    suitable chunk found within `max_attempts`.
    """
    if rng is None:
        rng = random.Random()
    exclude_lids = exclude_lids or set()
    exclude_titles = exclude_titles or set()
    n = searcher.num_docs
    for _ in range(max_attempts):
        lid = rng.randint(0, n - 1)
        if lid in exclude_lids:
            continue
        d = _doc_at(searcher.searcher, lid)
        if d is None or not d["body"] or d["title"] in exclude_titles:
            continue
        claims = claim_sentences(d["body"])
        if not claims:
            continue
        d["claims"] = claims
        return d
    return None


def find_related_chunks(
    searcher: BM25Searcher,
    query: str,
    k: int = 50,
    exclude_titles: set[str] | None = None,
    require_claim: bool = True,
) -> list[dict]:
    """Topical neighbours of `query` via BM25 over the 100w corpus.

    Returns parsed `{lid, docid, title, body, raw_text, claims}` dicts in
    BM25-rank order. Drops hits whose title is in `exclude_titles` (use for
    skipping chunk A's own article). When `require_claim` is True, also
    filters to chunks that contain ≥1 claim-bearing sentence (so chunk B has
    something rewritable).
    """
    exclude_titles = exclude_titles or set()
    hits = searcher.searcher.search(query, k=k)
    out: list[dict] = []
    for h in hits:
        raw = json.loads(h.lucene_document.get("raw"))
        raw_text = raw.get("contents", raw.get("body", ""))
        title, body = extract_title_and_body(raw_text)
        if title in exclude_titles or not body:
            continue
        claims = claim_sentences(body) if require_claim else []
        if require_claim and not claims:
            continue
        out.append({
            "lid": None,  # Lucene id not directly exposed by hit; use docid for identity
            "docid": h.docid,
            "title": title,
            "body": body,
            "raw_text": raw_text,
            "claims": claims,
        })
    return out


def fetch_article_neighbors(
    searcher: BM25Searcher,
    center_lid: int,
    center_title: str,
    n_extra: int,
    exclude_lids: set[int] | None = None,
) -> list[dict]:
    """Fetch up to `n_extra` chunks adjacent to `center_lid` in the same article.

    Walks forward first, then backward; returns whatever it finds in the
    order encountered (may be fewer than n_extra at article boundaries).
    Adjacent chunks are detected by title equality (cheap; same trick used
    in `sample_article_run`).
    """
    exclude_lids = exclude_lids or set()
    results: list[dict] = []
    n = searcher.num_docs
    lid = center_lid + 1
    while len(results) < n_extra and lid < n:
        if lid not in exclude_lids:
            d = _doc_at(searcher.searcher, lid)
            if d is None or d["title"] != center_title:
                break
            results.append(d)
        lid += 1
    lid = center_lid - 1
    while len(results) < n_extra and lid >= 0:
        if lid not in exclude_lids:
            d = _doc_at(searcher.searcher, lid)
            if d is None or d["title"] != center_title:
                break
            results.append(d)
        lid -= 1
    return results


def fetch_article_neighbors_by_title(
    searcher: BM25Searcher,
    title: str,
    n_extra: int,
    exclude_docids: set[str] | None = None,
    bm25_k: int = 30,
) -> list[dict]:
    """Find up to n_extra chunks from the article with the given title.

    Uses BM25 over the title as a query and filters hits to exact-title
    match. Slower than `fetch_article_neighbors` (which walks Lucene ids),
    but doesn't require knowing the lid — useful when the anchor came from
    a BM25 hit (e.g. chunk B in the contradiction pipeline).
    """
    exclude_docids = exclude_docids or set()
    hits = searcher.searcher.search(title, k=bm25_k)
    out: list[dict] = []
    for h in hits:
        if len(out) >= n_extra:
            break
        if h.docid in exclude_docids:
            continue
        raw = json.loads(h.lucene_document.get("raw"))
        raw_text = raw.get("contents", raw.get("body", ""))
        h_title, body = extract_title_and_body(raw_text)
        if h_title != title or not body:
            continue
        out.append({
            "lid": None,
            "docid": h.docid,
            "title": h_title,
            "body": body,
            "raw_text": raw_text,
        })
    return out


def sample_clustered_fillers(
    searcher: BM25Searcher,
    n_chunks: int,
    cluster_size: int,
    rng: random.Random | None = None,
    exclude_titles: set[str] | None = None,
    max_attempts_per_cluster: int = 50,
) -> list[dict] | None:
    """Sample `n_chunks` filler chunks in runs of `cluster_size` per article.

    Each cluster comes from a single article (consecutive Lucene ids); each
    article appears at most once across the result. Returns None if too few
    eligible articles. With cluster_size=1 this reduces to random-singleton
    sampling (slower than `sample_random_chunks` due to per-article retries,
    so prefer `sample_random_chunks` when cluster_size==1).
    """
    if rng is None:
        rng = random.Random()
    exclude_titles = set(exclude_titles or [])
    out: list[dict] = []
    while len(out) < n_chunks:
        run = sample_article_run(
            searcher, cluster_size, rng=rng,
            max_attempts=max_attempts_per_cluster,
            exclude_titles=exclude_titles,
        )
        if run is None:
            return None
        exclude_titles.add(run[0]["title"])
        out.extend(run)
    return out[:n_chunks]


def sample_random_chunks(
    searcher: BM25Searcher,
    n: int,
    threads: int = 16,
    rng: random.Random | None = None,
    seed: int | None = None,
    exclude_titles: set[str] | None = None,
) -> list[dict]:
    """Bulk random chunks parsed into {lid, docid, title, body, raw_text}.

    Like `BM25Searcher.prefetch_random_pool` but returns the parsed form
    (including title) rather than just text. Used for filler passages.
    """
    if rng is None:
        rng = random.Random(seed)
    exclude_titles = exclude_titles or set()
    n_total = searcher.num_docs
    # Oversample to leave room for title-exclusion drops.
    pool_size = min(n_total, max(n * 2, n + 200))
    lids = rng.sample(range(n_total), pool_size)

    from concurrent.futures import ThreadPoolExecutor

    def _fetch(lid):
        return _doc_at(searcher.searcher, lid)

    with ThreadPoolExecutor(max_workers=threads) as ex:
        docs = list(ex.map(_fetch, lids))

    out: list[dict] = []
    for d in docs:
        if d is None or not d["body"]:
            continue
        if d["title"] in exclude_titles:
            continue
        out.append(d)
        if len(out) >= n:
            break
    return out


__all__ = [
    "INDEX_NAME",
    "BM25Searcher",
    "extract_title_and_body",
    "split_sentences",
    "is_wiki_claim_sentence",
    "claim_sentences",
    "sample_article_run",
    "sample_two_articles",
    "pick_chunk_with_claim",
    "find_related_chunks",
    "sample_random_chunks",
    "fetch_article_neighbors",
    "fetch_article_neighbors_by_title",
    "sample_clustered_fillers",
]
