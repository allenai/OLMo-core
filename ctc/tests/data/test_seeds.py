"""
Seed pools: the expensive corpus half, serialized once and rebuilt from file.

The property that matters is not "the file parses" but "a build from the file is THE SAME BUILD":
the pool is everything a generator reads, so if one field, one ordering or one score fails to
round-trip, the seeded build produces different -- plausible, ungraded-against -- data. So the
tests here assert pool *equality* and example *equality*, per ladder, against the same fixture
pools the generator tests run on.

Everything runs with no network, no GPU and no Hub: ``--pool`` takes a local file, and the one
codepath that would touch the Hub (``--pool auto``) resolves through ``seeds.resolve``, which is
exercised only on its local-path branch.
"""

from __future__ import annotations

import gzip
import json
import random

import pytest
from fixtures import pools

from ctc.data import cli, ladders, seeds
from ctc.data.generators import base as generators
from ctc.tasks import load_all

#: The same ladder -> fixture map the generator tests use. Duplicated deliberately: this file must
#: not import another test module, and the coverage test below fails if either map falls behind
#: the registry.
POOLS = {
    "contradiction": pools.pubmed_pool,
    "nq": pools.retrieval_pool,
    "hotpotqa": lambda: pools.retrieval_pool(source="hotpotqa", gold=2, hard=8),
    "fiqa": lambda: pools.retrieval_pool(source="beir_fiqa"),
    "scifact": lambda: pools.retrieval_pool(source="beir_scifact"),
    "rerank": pools.rerank_pool,
    "outlier": pools.article_pool,
    "outlier_review": pools.review_pool,
    "oolong": pools.oolong_pool,
    "absence": pools.book_pool,
    "xabsence": pools.paraphrase_pool,
    "reorder": pools.reorder_pool,
    "qdmatch_nq": pools.unit_pool,
    "qdmatch_hpqa": lambda: pools.unit_pool(gold=2, source="hotpotqa"),
    "grouping_labeled": pools.openalex_pool,
}


@pytest.fixture(scope="module", autouse=True)
def _tasks():
    load_all()


def roundtrip(task, tmp_path):
    pool = POOLS[task]()
    path = seeds.save(
        tmp_path / seeds.filename_for(task), task, pool, corpus_config={"kind": "fixture"}
    )
    return pool, seeds.load(path, task)


# ── coverage ────────────────────────────────────────────────────────────────────────────────────


def test_every_corpus_backed_ladder_has_a_codec_and_only_those():
    """A ladder without a codec silently keeps its GPU/index/LLM requirement; a codec for a
    ladder that does not exist is a file the Hub repo would serve to nobody."""
    corpus_backed = {n for n in generators.names() if generators.get(n).corpus is not None}
    assert set(seeds.LADDER_TAGS) == corpus_backed


def test_every_seedable_ladder_is_round_trip_tested_here():
    assert set(POOLS) == set(seeds.LADDER_TAGS)


def test_a_synthetic_ladder_is_refused_by_name():
    with pytest.raises(ValueError, match="synthetic"):
        seeds.save("unused.seed.jsonl.gz", "strmatch", None)


# ── the round trip itself ───────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("task", sorted(POOLS))
def test_a_pool_survives_the_round_trip_exactly(task, tmp_path):
    pool, loaded = roundtrip(task, tmp_path)
    if task == "outlier":
        # ArticlePool is not a dataclass; its identity is the article list and the filter it was
        # built with, and its buckets are derived in __init__.
        assert loaded.articles == pool.articles
        assert loaded.min_chunks == pool.min_chunks
        assert loaded.max_chunks == pool.max_chunks
    else:
        assert loaded == pool


@pytest.mark.parametrize("task", sorted(POOLS))
def test_a_build_from_a_seeded_pool_is_the_same_build(task, tmp_path):
    """Same seed, same config, same example -- the only difference a seed pool may make is where
    the pool came from."""
    pool, loaded = roundtrip(task, tmp_path)
    gen = generators.get(task)
    config = gen.config()
    if gen.indexed:
        config["index"] = 0
    live = gen.build_example(random.Random(3), corpus=pool, **config)
    seeded = gen.build_example(random.Random(3), corpus=loaded, **config)
    assert live is not None
    assert seeded == live


# ── refusals ────────────────────────────────────────────────────────────────────────────────────


def test_a_pool_for_another_ladder_is_refused(tmp_path):
    """An nq pool fed to the fiqa ladder would build plausible data for the wrong ladder, and the
    mistake survives every downstream audit -- the header check is the only place to catch it."""
    path = seeds.save(tmp_path / "x.seed.jsonl.gz", "nq", pools.retrieval_pool())
    with pytest.raises(ValueError, match="exported for ladder 'nq'"):
        seeds.load(path, "fiqa")


def test_a_file_without_the_format_tag_is_refused(tmp_path):
    path = tmp_path / "bogus.seed.jsonl.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write(json.dumps({"format": "something-else"}) + "\n")
        handle.write("{}\n")
    with pytest.raises(ValueError, match="not a ctc-seed-pool-v1 file"):
        seeds.read_header(path)


def test_resolve_refuses_a_missing_local_path(tmp_path):
    with pytest.raises(FileNotFoundError):
        seeds.resolve(str(tmp_path / "absent.seed.jsonl.gz"), "nq")


# ── the CLI surface ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def _small_ladders(monkeypatch):
    monkeypatch.setitem(ladders.LADDERS, "nq", {"2k": 11, "4k": 23})


def test_build_from_a_seed_pool_needs_no_corpus_loader(tmp_path, _small_ladders):
    """The whole point: this build runs with the generator's real loader UNPATCHED, so if the seed
    path fell through to it, the test would be reaching for pyserini and a live Hub."""
    seed = seeds.save(tmp_path / seeds.filename_for("nq"), "nq", pools.retrieval_pool(queries=600))
    out = tmp_path / "out"
    assert (
        cli.main(["build", "--task", "nq", "--pool", str(seed), "--train", "20", "--out", str(out)])
        == 0
    )
    rows = [json.loads(line) for line in (out / "nq" / "eval_2k.jsonl").read_text().splitlines()]
    assert len(rows) == 500
    assert all(r["source"] == "nq" and r["gold_doc_indices"] for r in rows)


def test_a_corpus_override_alongside_a_pool_is_refused(tmp_path):
    """The corpus parameters were consumed at export time; accepting one here would label the
    output as built with a setting that had no effect."""
    seed = seeds.save(tmp_path / seeds.filename_for("nq"), "nq", pools.retrieval_pool())
    with pytest.raises(SystemExit, match="baked into the pool"):
        cli.main(
            [
                "build",
                "--task",
                "nq",
                "--pool",
                str(seed),
                "-C",
                "ce_filter=false",
                "--out",
                str(tmp_path),
            ]
        )


def test_a_pool_on_a_synthetic_task_is_refused(tmp_path):
    with pytest.raises(SystemExit, match="synthetic"):
        cli.main(["build", "--task", "strmatch", "--pool", "anything", "--out", str(tmp_path)])


def test_pool_export_then_info_then_build(tmp_path, monkeypatch, capsys):
    """The full loop a publisher runs: export (with the loader swapped for a fixture, standing in
    for the expensive real load), inspect, then build from the file."""
    import dataclasses

    module = __import__("ctc.tasks.retrieval.sources.nq", fromlist=["GENERATOR"])
    monkeypatch.setattr(
        module,
        "GENERATOR",
        dataclasses.replace(
            module.GENERATOR, corpus=lambda **kw: pools.retrieval_pool(queries=600, source="nq")
        ),
    )
    monkeypatch.setitem(ladders.LADDERS, "nq", {"2k": 11, "4k": 23})

    assert cli.main(["pool", "export", "--task", "nq", "--out", str(tmp_path / "seeds")]) == 0
    seed = tmp_path / "seeds" / seeds.filename_for("nq")
    assert seed.exists()

    assert cli.main(["pool", "info", str(seed)]) == 0
    printed = capsys.readouterr().out
    assert '"ladder": "nq"' in printed

    out = tmp_path / "out"
    assert (
        cli.main(["build", "--task", "nq", "--pool", str(seed), "--train", "20", "--out", str(out)])
        == 0
    )
    assert (out / "nq" / "train.jsonl").exists()


def test_pool_export_refuses_build_parameters(tmp_path, capsys):
    assert (
        cli.main(["pool", "export", "--task", "nq", "--out", str(tmp_path), "-C", "num_docs=5"])
        == 1
    )
    assert "build parameters" in capsys.readouterr().err


def test_pool_export_refuses_a_synthetic_task(tmp_path, capsys):
    assert cli.main(["pool", "export", "--task", "strmatch", "--out", str(tmp_path)]) == 1
    assert "no corpus" in capsys.readouterr().err
