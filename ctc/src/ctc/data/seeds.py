"""
Seed pools: the expensive half of data generation, serialized to one portable file.

Building a pool is the only step of ``ctc-data build`` that can need a GPU cross-encoder, a Lucene
index, an LLM endpoint or a multi-gigabyte HF download. Everything *after* the pool -- placing
gold, drawing distractors, laddering rungs, auditing -- is pure Python over plain data, and fast.
So the pool is the natural unit of reuse: run the expensive part once, serialize the pool, and
every later build (any rung ladder, any train size, any seed, up to 10M+ tokens per example)
assembles from the file in seconds with ``pip install ./ctc`` and nothing else.

::

    ctc-data pool export --task nq --out seeds/            # the expensive part, once
    ctc-data build --task nq --pool seeds/nq.seed.jsonl.gz --out DIR      # fast forever after
    ctc-data build --task nq --pool auto --out DIR         # fetch the published pool from HF

A seed pool is a gzipped two-line JSONL file: a header line (format tag, ladder, pool type,
provenance) and a payload line. The codecs here are **explicit per pool type** -- no pickled
objects, no class paths resolved from file content -- so loading a pool downloaded from the Hub
executes nothing but ``json.loads`` and whitelisted dataclass constructors.

Two normalizations are inherent to the JSON round trip and deliberate: tuples inside a pool's
``provenance`` dict come back as lists (provenance is copied into ``_meta``, never compared), and
dict ordering is preserved. Everything the generators actually *read* -- pair order, hard-negative
order, per-document scores -- round-trips exactly, and the tests assert pool equality, not just
schema equality.
"""

from __future__ import annotations

import gzip
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

__all__ = [
    "FORMAT",
    "DEFAULT_REPO",
    "REPO_ENV",
    "LADDER_TAGS",
    "filename_for",
    "save",
    "load",
    "read_header",
    "resolve",
]

#: The format tag every seed-pool header carries. Bumped only on an incompatible payload change;
#: a mismatch is a refusal, not a warning, because a half-understood pool builds plausible data.
FORMAT = "ctc-seed-pool-v1"

#: Where ``--pool auto`` looks: one HF dataset repo holding ``<ladder>.seed.jsonl.gz`` per ladder.
DEFAULT_REPO = "PrasannSinghal/ctc-seed-pools"

#: Environment variable overriding :data:`DEFAULT_REPO`, so a fork can publish its own pools
#: without patching code.
REPO_ENV = "CTC_SEED_POOL_REPO"

#: Ladder name -> codec tag. Several ladders share one pool type (``nq``, ``hotpotqa``, ``fiqa``,
#: ``scifact`` and ``rerank`` are all :class:`~ctc.data.sources.retrieval.RetrievalPool`), but a
#: seed file records the LADDER it was exported for and ``load`` enforces the match: an nq pool
#: fed to the fiqa ladder would build data that grades, plausibly, as the wrong ladder.
LADDER_TAGS: Dict[str, str] = {
    "contradiction": "pubmed",
    "redundancy": "redundancy",
    "contra_fever": "fever",
    "nq": "retrieval",
    "hotpotqa": "retrieval",
    "fiqa": "retrieval",
    "scifact": "retrieval",
    "rerank": "retrieval",
    "outlier": "article",
    "outlier_review": "review",
    "oolong": "oolong",
    "absence": "book",
    "xabsence": "paraphrase",
    "reorder": "passage",
    "grouping_labeled": "openalex",
    "qdmatch_nq": "unit",
    "qdmatch_hpqa": "unit",
}


def filename_for(ladder: str) -> str:
    """
    :param ladder: Ladder name.

    :returns: The canonical seed-pool file name, which is also the file name ``--pool auto``
        fetches from the Hub repo.
    """
    return f"{ladder}.seed.jsonl.gz"


# ── codecs ────────────────────────────────────────────────────────────────────────────────────
#
# One (encode, decode) pair per pool type. Imports are lazy and per-codec: ``ctc.data.seeds`` must
# import on a bare install, and two pool classes live in task modules that pull in the format
# registry. Every decode goes through the pool type's own constructor, so derived state
# (``ArticlePool``'s length buckets, the lazy ``_neighbours``/``_indices`` caches) is rebuilt
# rather than restored -- caches are per-split working state, not identity.


def _encode_pubmed(pool: Any) -> Dict[str, Any]:
    return {
        "pairs": [
            {
                "claim": p.claim,
                "contradiction": p.contradiction,
                "abstract_id": p.abstract_id,
                "mode": p.mode,
            }
            for p in pool.pairs
        ],
        "fillers": {k: list(v) for k, v in pool.fillers.items()},
        "provenance": pool.provenance,
    }


def _decode_pubmed(payload: Dict[str, Any]) -> Any:
    from .sources import pubmed

    return pubmed.PubMedPool(
        pairs=tuple(pubmed.ClaimPair(**p) for p in payload["pairs"]),
        fillers={k: tuple(v) for k, v in payload["fillers"].items()},
        provenance=payload.get("provenance", {}),
    )


def _encode_redundancy(pool: Any) -> Dict[str, Any]:
    return {
        "pairs": [
            {
                "claim": p.claim,
                "paraphrase": p.paraphrase,
                "abstract_id": p.abstract_id,
                "mode": p.mode,
            }
            for p in pool.pairs
        ],
        "hardnegs": [
            {
                "first": h.first,
                "second": h.second,
                "abstract_id": h.abstract_id,
                "overlap": h.overlap,
            }
            for h in pool.hardnegs
        ],
        "fillers": {k: list(v) for k, v in pool.fillers.items()},
        "provenance": pool.provenance,
    }


def _decode_redundancy(payload: Dict[str, Any]) -> Any:
    from .sources import pubmed_redundancy as red

    return red.RedundancyPool(
        pairs=tuple(red.RedundantPair(**p) for p in payload["pairs"]),
        hardnegs=tuple(red.HardNegativePair(**h) for h in payload["hardnegs"]),
        fillers={k: tuple(v) for k, v in payload["fillers"].items()},
        provenance=payload.get("provenance", {}),
    )


def _encode_fever(pool: Any) -> Dict[str, Any]:
    return {
        "pairs": [list(p) for p in pool.pairs],
        "nei_by_page": {k: list(v) for k, v in pool.nei_by_page.items()},
        "support_pairs_by_page": {
            k: [list(p) for p in v] for k, v in pool.support_pairs_by_page.items()
        },
        "fillers": list(pool.fillers),
        "pages": list(pool.pages),
    }


def _decode_fever(payload: Dict[str, Any]) -> Any:
    from .sources import fever

    return fever.FeverPool(
        pairs=tuple((p[0], p[1], p[2]) for p in payload["pairs"]),
        nei_by_page={k: tuple(v) for k, v in payload["nei_by_page"].items()},
        support_pairs_by_page={
            k: tuple((p[0], p[1]) for p in v) for k, v in payload["support_pairs_by_page"].items()
        },
        fillers=tuple(payload["fillers"]),
        pages=tuple(payload["pages"]),
    )


def _encode_retrieval(pool: Any) -> Dict[str, Any]:
    queries = []
    for q in pool.queries:
        row: Dict[str, Any] = {
            "query": q.query,
            "gold": [{"id": c.id, "text": c.text} for c in q.gold],
        }
        # The optional fields default empty on most ladders; omitting them keeps an nq/hotpotqa
        # pool from paying for rerank's per-query fill and scores.
        if q.hard:
            row["hard"] = [{"id": c.id, "text": c.text} for c in q.hard]
        if q.answers:
            row["answers"] = list(q.answers)
        if q.fill:
            row["fill"] = [{"id": c.id, "text": c.text} for c in q.fill]
        if q.scores:
            row["scores"] = q.scores
        queries.append(row)
    return {
        "source": pool.source,
        "queries": queries,
        "corpus": [{"id": c.id, "text": c.text} for c in pool.corpus],
    }


def _decode_retrieval(payload: Dict[str, Any]) -> Any:
    from .sources.retrieval import Candidate, QueryPool, RetrievalPool

    def candidates(rows: Any) -> Tuple[Any, ...]:
        return tuple(Candidate(id=r["id"], text=r["text"]) for r in rows)

    return RetrievalPool(
        queries=tuple(
            QueryPool(
                query=q["query"],
                gold=candidates(q["gold"]),
                hard=candidates(q.get("hard", ())),
                answers=tuple(q.get("answers", ())),
                fill=candidates(q.get("fill", ())),
                scores=q.get("scores", {}),
            )
            for q in payload["queries"]
        ),
        corpus=candidates(payload["corpus"]),
        source=payload["source"],
    )


def _encode_article(pool: Any) -> Dict[str, Any]:
    return {
        "articles": [[title, list(bodies)] for title, bodies in pool.articles],
        "min_chunks": pool.min_chunks,
    }


def _decode_article(payload: Dict[str, Any]) -> Any:
    from .sources import wiki100w

    return wiki100w.ArticlePool(
        [(title, bodies) for title, bodies in payload["articles"]],
        min_chunks=payload["min_chunks"],
    )


def _encode_review(pool: Any) -> Dict[str, Any]:
    return {
        "by_category": {
            cat: [
                {"title": r.title, "text": r.text, "rating": r.rating, "category": r.category}
                for r in rows
            ]
            for cat, rows in pool.by_category.items()
        }
    }


def _decode_review(payload: Dict[str, Any]) -> Any:
    from .sources import amazon

    return amazon.ReviewPool(
        by_category={
            cat: tuple(amazon.Review(**r) for r in rows)
            for cat, rows in payload["by_category"].items()
        }
    )


def _encode_oolong(pool: Any) -> Dict[str, Any]:
    return {
        "items": {
            sub: [
                {
                    "line": i.line,
                    "user": i.user,
                    "label": i.label,
                    "date": i.date,
                    "month": i.month,
                    "tokens": i.tokens,
                }
                for i in rows
            ]
            for sub, rows in pool.items.items()
        },
        "labels": {sub: list(v) for sub, v in pool.labels.items()},
        "preamble": pool.preamble,
        "preamble_tokens": pool.preamble_tokens,
    }


def _decode_oolong(payload: Dict[str, Any]) -> Any:
    from .sources import oolong

    return oolong.OolongPool(
        items={
            sub: tuple(oolong.Item(**i) for i in rows) for sub, rows in payload["items"].items()
        },
        labels={sub: tuple(v) for sub, v in payload["labels"].items()},
        preamble=payload["preamble"],
        preamble_tokens=payload["preamble_tokens"],
    )


def _encode_book(pool: Any) -> Dict[str, Any]:
    return {
        "runs": [{"book": r.book, "sentences": list(r.sentences)} for r in pool.runs],
        "provenance": pool.provenance,
    }


def _decode_book(payload: Dict[str, Any]) -> Any:
    from .sources import gutenberg

    return gutenberg.BookPool(
        runs=tuple(
            gutenberg.ProseRun(book=r["book"], sentences=tuple(r["sentences"]))
            for r in payload["runs"]
        ),
        provenance=payload.get("provenance", {}),
    )


def _encode_paraphrase(pool: Any) -> Dict[str, Any]:
    # ``_neighbours`` is a lazy per-split decoy cache, excluded from the dataclass's own equality;
    # it is dropped here and recomputed on first use.
    return {
        "pairs": [{"original": p.original, "paraphrase": p.paraphrase} for p in pool.pairs],
        "provenance": pool.provenance,
    }


def _decode_paraphrase(payload: Dict[str, Any]) -> Any:
    from .sources import paraphrase

    return paraphrase.ParaphrasePool(
        pairs=tuple(paraphrase.ParaphrasePair(**p) for p in payload["pairs"]),
        provenance=payload.get("provenance", {}),
    )


def _encode_passage(pool: Any) -> Dict[str, Any]:
    return {
        "books": [
            {"book": b.book, "passages": list(b.passages), "seams": sorted(b.seams)}
            for b in pool.books
        ],
        "provenance": pool.provenance,
    }


def _decode_passage(payload: Dict[str, Any]) -> Any:
    from ctc.tasks.reorder.sources.gutenberg import BookPassages, PassagePool

    return PassagePool(
        books=tuple(
            BookPassages(book=b["book"], passages=tuple(b["passages"]), seams=frozenset(b["seams"]))
            for b in payload["books"]
        ),
        provenance=payload.get("provenance", {}),
    )


def _encode_openalex(pool: Any) -> Dict[str, Any]:
    def papers(rows: Any) -> Any:
        return [
            {
                "id": p.id,
                "title": p.title,
                "abstract": p.abstract,
                "year": p.year,
                "concepts": {str(level): value for level, value in p.concepts.items()},
            }
            for p in rows
        ]

    return {
        "papers": papers(pool.papers),
        "eval_papers": None if pool.eval_papers is None else papers(pool.eval_papers),
        "eval_year_min": pool.eval_year_min,
        "provenance": pool.provenance,
    }


def _decode_openalex(payload: Dict[str, Any]) -> Any:
    from .sources import openalex

    def papers(rows: Any) -> Tuple[Any, ...]:
        # JSON object keys are strings; concept levels are ints in the dataclass.
        return tuple(
            openalex.Paper(
                id=p["id"],
                title=p["title"],
                abstract=p["abstract"],
                year=p["year"],
                concepts={int(level): value for level, value in p["concepts"].items()},
            )
            for p in rows
        )

    return openalex.OpenAlexPool(
        papers=papers(payload["papers"]),
        eval_papers=None if payload["eval_papers"] is None else papers(payload["eval_papers"]),
        eval_year_min=payload["eval_year_min"],
        provenance=payload.get("provenance", {}),
    )


def _encode_unit(pool: Any) -> Dict[str, Any]:
    return {
        "units": [{"query": u.query, "gold": list(u.gold)} for u in pool.units],
        "source": pool.source,
        "provenance": pool.provenance,
    }


def _decode_unit(payload: Dict[str, Any]) -> Any:
    from ctc.tasks.qdmatch.generate import QueryUnit, UnitPool

    return UnitPool(
        units=tuple(QueryUnit(query=u["query"], gold=tuple(u["gold"])) for u in payload["units"]),
        source=payload["source"],
        provenance=payload.get("provenance", {}),
    )


_CODECS: Dict[str, Tuple[Callable[[Any], Dict[str, Any]], Callable[[Dict[str, Any]], Any]]] = {
    "pubmed": (_encode_pubmed, _decode_pubmed),
    "redundancy": (_encode_redundancy, _decode_redundancy),
    "fever": (_encode_fever, _decode_fever),
    "retrieval": (_encode_retrieval, _decode_retrieval),
    "article": (_encode_article, _decode_article),
    "review": (_encode_review, _decode_review),
    "oolong": (_encode_oolong, _decode_oolong),
    "book": (_encode_book, _decode_book),
    "paraphrase": (_encode_paraphrase, _decode_paraphrase),
    "passage": (_encode_passage, _decode_passage),
    "openalex": (_encode_openalex, _decode_openalex),
    "unit": (_encode_unit, _decode_unit),
}


def _tag_for(ladder: str) -> str:
    if ladder not in LADDER_TAGS:
        raise ValueError(
            f"{ladder!r} has no seed-pool codec: it is either synthetic (no corpus to seed) or "
            f"unknown. Seedable ladders: {', '.join(sorted(LADDER_TAGS))}"
        )
    return LADDER_TAGS[ladder]


# ── files ─────────────────────────────────────────────────────────────────────────────────────


def save(
    path: Path, ladder: str, pool: Any, *, corpus_config: Optional[Dict[str, Any]] = None
) -> Path:
    """
    Serialize a pool to a seed file.

    :param path: Destination file. Convention: ``<dir>/<ladder>.seed.jsonl.gz``
        (:func:`filename_for`), which is the name ``--pool auto`` resolves on the Hub.
    :param ladder: The ladder the pool was built for. Recorded in the header and enforced on load.
    :param pool: The pool object, as returned by the ladder's corpus loader.
    :param corpus_config: The corpus parameters the pool was built with, for the header's
        provenance record.

    :returns: ``path``.

    :raises ValueError: If the ladder has no codec (synthetic or unknown).
    """
    tag = _tag_for(ladder)
    encode, _ = _CODECS[tag]
    header = {
        "format": FORMAT,
        "ladder": ladder,
        "pool": tag,
        "created": {
            "corpus_config": corpus_config or {},
            "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write(json.dumps(header, ensure_ascii=False) + "\n")
        handle.write(json.dumps(encode(pool), ensure_ascii=False) + "\n")
    return path


def read_header(path: Path) -> Dict[str, Any]:
    """
    Read a seed file's header line without decoding the payload.

    :param path: A seed-pool file.

    :returns: The header dict.

    :raises ValueError: If the file does not carry the expected format tag.
    """
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        header = json.loads(handle.readline())
    if not isinstance(header, dict) or header.get("format") != FORMAT:
        raise ValueError(
            f"{path} is not a {FORMAT} file (header says format={header.get('format')!r} "
            "if it is a dict at all). Refusing to guess at a payload layout."
        )
    return header


def load(path: Path, ladder: str) -> Any:
    """
    Load a seed pool, verifying it was exported for this ladder.

    :param path: A seed-pool file.
    :param ladder: The ladder about to be built.

    :returns: The pool object, exactly as the ladder's generator expects it.

    :raises ValueError: On a format mismatch, a ladder mismatch, or an unknown pool tag. All three
        refuse rather than warn: a pool fed to the wrong ladder builds plausible, wrong data.
    """
    header = read_header(path)
    if header.get("ladder") != ladder:
        raise ValueError(
            f"{path} was exported for ladder {header.get('ladder')!r}, not {ladder!r}. "
            "Export a pool per ladder; sharing one across ladders is exactly the mixup the "
            "header exists to catch."
        )
    tag = header.get("pool")
    expected = _tag_for(ladder)
    if tag != expected:
        raise ValueError(f"{path} holds a {tag!r} pool but {ladder} builds from {expected!r}")
    _, decode = _CODECS[expected]
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        handle.readline()  # header, already validated
        payload = json.loads(handle.readline())
    return decode(payload)


def resolve(spec: str, ladder: str) -> Path:
    """
    Turn a ``--pool`` argument into a local file path.

    :param spec: A local path, ``"auto"`` (fetch ``<ladder>.seed.jsonl.gz`` from
        :data:`DEFAULT_REPO`, or from ``$CTC_SEED_POOL_REPO`` if set), or ``"hf://<repo-id>"``
        (fetch from that HF dataset repo).
    :param ladder: The ladder being built; names the file fetched from a repo.

    :returns: A local path to the seed file. Hub fetches land in (and re-serve from) the standard
        ``huggingface_hub`` cache, so repeat builds are offline.

    :raises FileNotFoundError: If a local ``spec`` does not exist.
    """
    if spec == "auto":
        return _fetch(os.environ.get(REPO_ENV, DEFAULT_REPO), ladder)
    if spec.startswith("hf://"):
        return _fetch(spec[len("hf://") :], ladder)
    path = Path(spec)
    if not path.exists():
        raise FileNotFoundError(
            f"seed pool {path} does not exist (pass a file, 'auto', or hf://<repo-id>)"
        )
    return path


def _fetch(repo_id: str, ladder: str) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(repo_id=repo_id, filename=filename_for(ladder), repo_type="dataset")
    )
