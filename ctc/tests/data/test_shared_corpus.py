"""
The shared-corpus ("fast") construction, on synthetic rows.

These are the properties the fast bundle's speedup and its fairness rest on, and neither shows up
as an error when it breaks: a prefix that is not byte-identical silently disables KV reuse, and a
gold pair that lands entirely in the per-query tail turns an all-pairs search into a lookup at the
end of the context. Both produce a perfectly plausible score.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import pytest

from ctc.data import shared_corpus

NDOCS = 40
NROWS = 20


def _rows() -> List[dict]:
    """:returns: Contradiction-shaped rows, one gold pair each, no document shared between rows."""
    rows = []
    for i in range(NROWS):
        docs = [{"title": None, "text": f"row {i} document {j}"} for j in range(NDOCS)]
        rows.append(
            {
                "documents": docs,
                # this task is 1-indexed on disk
                "gold_doc_indices": [[1, 2]],
                "queries": [],
                "answers": [f"1; 2 (row {i})"],
                "meta": {"row": i},
            }
        )
    return rows


@pytest.fixture()
def source(tmp_path: Path) -> Path:
    path = tmp_path / "rung_test.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in _rows()))
    return path


def _build(tmp_path: Path, source: Path, tail_frac: float = 0.1) -> List[dict]:
    out, _ = shared_corpus.build_contradiction_shared(
        task="contradiction",
        rung=4096,
        out_root=str(tmp_path / "fast"),
        tail_frac=tail_frac,
        source=str(source),
    )
    return [json.loads(line) for line in Path(out).read_text().splitlines() if line.strip()]


def _by_corpus(rows: List[dict]) -> Dict[str, List[dict]]:
    groups: Dict[str, List[dict]] = {}
    for r in rows:
        groups.setdefault(r["corpus_id"], []).append(r)
    return groups


def test_every_question_survives_the_rebuild(tmp_path, source):
    """A fast rung has to grade the same questions as its reliable twin, or a rung-to-rung
    comparison is measuring the eval set instead of the model."""
    built = _build(tmp_path, source)
    assert len(built) == NROWS
    assert {r["answers"][0] for r in built} == {f"1; 2 (row {i})" for i in range(NROWS)}


def test_the_shared_prefix_is_byte_identical_within_a_corpus(tmp_path, source):
    built = _build(tmp_path, source)
    for cid, group in _by_corpus(built).items():
        plen = group[0]["shared_prefix_len"]
        first = [shared_corpus.doc_key(d) for d in group[0]["documents"][:plen]]
        for r in group[1:]:
            assert r["shared_prefix_len"] == plen, cid
            assert [shared_corpus.doc_key(d) for d in r["documents"][:plen]] == first, cid
        assert len({r["shared_prefix_sha1"] for r in group}) == 1, cid


def test_every_gold_pair_straddles_the_prefix_boundary(tmp_path, source):
    """One member in the shared prefix, its partner in the per-query tail. Both in the tail would
    let recency hand over a whole answer -- the failure this construction exists to avoid."""
    built = _build(tmp_path, source)
    for r in built:
        plen = r["shared_prefix_len"]
        for pair in r["gold_doc_indices"]:
            positions = [i - 1 for i in pair]
            assert sum(p < plen for p in positions) == 1, (pair, plen)


def test_gold_indices_point_at_the_documents_they_claim(tmp_path, source):
    """The rebuild moves every document, so the indices are rewritten. If that rewrite is off by
    one the eval grades the wrong pair and simply scores lower."""
    built = _build(tmp_path, source)
    for r in built:
        row = r["meta"]["row"]
        want = {f"row {row} document 0", f"row {row} document 1"}
        got = {r["documents"][i - 1]["text"] for pair in r["gold_doc_indices"] for i in pair}
        assert got == want


def test_the_tail_fraction_is_honoured(tmp_path, source):
    built = _build(tmp_path, source, tail_frac=0.25)
    ndocs = len(built[0]["documents"])
    plen = built[0]["shared_prefix_len"]
    assert ndocs == NDOCS
    assert plen / ndocs == pytest.approx(0.75, abs=0.03)


def test_queries_per_corpus_is_capped_instead_of_raising(tmp_path, source):
    """Each query parks one half of each gold pair in the shared prefix, so a corpus can host at
    most ``prefix_len // pairs`` of them. The pre-migration script took a flat 125 and raised
    SystemExit whenever that did not fit -- which is every small rung once the tail is large."""
    built = _build(tmp_path, source, tail_frac=0.9)
    sizes = {len(g) for g in _by_corpus(built).values()}
    assert max(sizes) <= built[0]["shared_prefix_len"]
    assert len(built) == NROWS


# ── the planted outlier construction ─────────────────────────────────────────


@pytest.fixture()
def pool_pickle(tmp_path: Path) -> str:
    """A synthetic article pool, every chunk text unique.

    Big enough that ``ArticlePool.for_split("eval")`` -- which keeps a tenth of the articles, so
    that a topic never straddles train and eval -- still leaves enough topics to build from.
    """
    import pickle

    articles = []
    for a in range(3000):
        n = 4 + (a % 12)
        articles.append((f"article {a}", [f"article {a} chunk {c}" for c in range(n)]))
    path = tmp_path / "pool.pkl"
    path.write_bytes(pickle.dumps({"articles": articles, "min_article_chunks": 4}))
    return str(path)


def _planted(tmp_path: Path, pool_pickle: str, **kwargs):
    opts = dict(ndocs=60, pool_path=pool_pickle, n_rows=40, tail_frac=0.2)
    opts.update(kwargs)
    out, manifest = shared_corpus.build_outlier_planted(
        task="outlier", rung=4096, out_root=str(tmp_path / "fast"), **opts
    )
    rows = [json.loads(line) for line in Path(out).read_text().splitlines() if line.strip()]
    return rows, manifest


def _topics(row: dict) -> Dict[str, int]:
    return {t: c for t, c in row["meta"]["category_distribution"]}


def test_planted_answer_lives_in_the_shared_prefix(tmp_path, pool_pickle):
    """The whole point of the inversion. The +0.215 the tail construction cost came from putting
    the golds at the end of the context; here they are never in the tail."""
    rows, _ = _planted(tmp_path, pool_pickle)
    assert rows
    for r in rows:
        assert len(r["gold_doc_indices"]) == 3
        for i in r["gold_doc_indices"]:
            assert i < r["shared_prefix_len"], "a gold document landed in the per-query tail"


def test_planted_answer_is_the_unique_minimum_at_gap_one(tmp_path, pool_pickle):
    """Answer at 3, nearest competitor at 4 -- the same structure as the reliable rung, where the
    smallest majority topic is 4 in 484/500 rows. A wider gap is a different, much easier task:
    widening it once took outlier from 0.320 to 0.887."""
    rows, manifest = _planted(tmp_path, pool_pickle)
    for r in rows:
        counts = _topics(r)
        answer = r["meta"]["minority_label"]
        others = [c for t, c in counts.items() if t != answer]
        assert counts[answer] == 3
        assert min(others) == 4, "the floor moved; the task got easier"
        assert sorted(counts.values())[:2] == [3, 4]
    assert manifest["gap_above_answer"] == 1


def test_planted_decoys_are_topped_past_the_boundary(tmp_path, pool_pickle):
    """Decoys go 3 -> 5, not 3 -> 4. Topping to 4 would park every decoy on the discrimination
    boundary and multiply the near-misses; 5 is the modal majority size, so they are camouflaged."""
    rows, _ = _planted(tmp_path, pool_pickle)
    by_corpus: Dict[str, List[dict]] = {}
    for r in rows:
        by_corpus.setdefault(r["corpus_id"], []).append(r)
    for group in by_corpus.values():
        answers = {r["meta"]["minority_label"] for r in group}
        for r in group:
            counts = _topics(r)
            for other in answers - {r["meta"]["minority_label"]}:
                # >= rather than ==: when majority camouflage runs short the tail is padded with
                # decoy spare, which pushes a decoy further above the boundary. Harmless -- what
                # must never happen is a decoy sitting AT the boundary, or below it.
                assert counts[other] >= 5, "a decoy was not topped up past the boundary"


def test_planted_prefix_is_byte_identical_within_a_corpus(tmp_path, pool_pickle):
    rows, _ = _planted(tmp_path, pool_pickle)
    by_corpus: Dict[str, List[dict]] = {}
    for r in rows:
        by_corpus.setdefault(r["corpus_id"], []).append(r)
    for cid, group in by_corpus.items():
        plen = group[0]["shared_prefix_len"]
        first = [d["text"] for d in group[0]["documents"][:plen]]
        for r in group[1:]:
            assert [d["text"] for d in r["documents"][:plen]] == first, cid
        assert len({r["shared_prefix_sha1"] for r in group}) == 1


def test_planted_rows_hold_no_duplicate_documents(tmp_path, pool_pickle):
    """Camouflage is drawn without replacement. A repeated passage is one a reader would notice,
    and it would double-count its topic -- which is how the floor moves by accident."""
    rows, _ = _planted(tmp_path, pool_pickle)
    for r in rows:
        texts = [d["text"] for d in r["documents"]]
        assert len(set(texts)) == len(texts)


def test_planted_tail_absence_is_not_a_shortcut(tmp_path, pool_pickle):
    """The answer's topic is the one candidate the tail does not top up, so absence has to be
    checked rather than assumed harmless. Every majority topic donating no camouflage is absent
    too, so guessing among absent topics is near chance."""
    _, manifest = _planted(tmp_path, pool_pickle)
    assert manifest["topics_absent_from_tail_mean"] > 3
    assert manifest["shortcut_absent_from_tail_acc"] < 0.35


def test_planted_emits_every_requested_row(tmp_path, pool_pickle):
    rows, manifest = _planted(tmp_path, pool_pickle, n_rows=40)
    assert len(rows) == 40 == manifest["rows"]
    assert manifest["shared_token_fraction"] == pytest.approx(0.8, abs=0.02)
