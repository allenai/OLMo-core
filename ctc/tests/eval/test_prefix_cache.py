"""
Prefill reuse: does the model see exactly the tokens it would have seen without the cache?

That is the whole safety claim, so it is checked here rather than assumed. The stub model records
what it consumes, with the same rewind semantics the real hybrid has -- the KV cursor truncates the
consumed sequence, and the recurrent state is a separate object that must be restored explicitly.
The interesting failure is not "the cache crashes", it is "the cache works and query 2 onward were
conditioned on query 1's suffix", which produces plausible text and a slightly wrong score.

The GPU-level byte-identity check was run against the real model in the pre-migration work
(``records/shared-corpus-eval-results.md``); this pins the logic that can be checked on CPU.
"""

from __future__ import annotations

import pytest

from ctc.eval import prefix_cache as pc

# ── the pure half ───────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "sequences,expected",
    [
        ([], 0),
        ([[1, 2, 3]], 3),
        ([[1, 2, 3], [1, 2, 9]], 2),
        ([[1, 2], [1, 2, 3]], 2),
        ([[5], [6]], 0),
    ],
)
def test_longest_common_token_prefix(sequences, expected):
    assert pc.longest_common_token_prefix(sequences) == expected


def test_rows_without_a_corpus_id_each_become_their_own_group():
    """So an ordinary independent eval file runs through this path unchanged, sharing nothing."""
    groups = pc.group_by_corpus([{"a": 1}, {"a": 2}])
    assert len(groups) == 2
    assert all(len(rows) == 1 for rows in groups.values())


def test_grouping_preserves_file_order():
    examples = [{"corpus_id": "a"}, {"corpus_id": "b"}, {"corpus_id": "a"}]
    assert pc.group_by_corpus(examples) == {"a": [0, 2], "b": [1]}


def test_measure_reuse_reports_the_token_prefix_not_the_shared_documents():
    """
    The distinction that made the whole prior effort fall over. Both groups below "share a corpus";
    only the first shares tokens. With query_position="both" the per-query question is emitted
    before the documents, and the measured reusable prefix came out at 0.5-0.9% on the multiplexed
    tasks against 95% on the ones whose question block is empty.
    """
    corpus = list(range(100))
    prefix_shared = [corpus + [900 + i] for i in range(4)]
    question_first = [[900 + i] + corpus for i in range(4)]

    good = pc.measure_reuse([prefix_shared])
    bad = pc.measure_reuse([question_first])

    assert good.fraction > 0.9 and good.speedup > 9
    assert bad.fraction == 0.0, "documents shared but not as a token prefix: nothing is reusable"
    assert "NOT worth caching" in str(bad)


def test_measure_reuse_handles_an_empty_input():
    assert pc.measure_reuse([]).fraction == 0.0


# ── the equivalence property ────────────────────────────────────────────────────────────────────


class StubMixer:
    """One block: a KV cursor plus a recurrent state, mirroring the hybrid's two mechanisms."""

    def __init__(self):
        self.consumed = []
        self.state_cache = None  # set by StubModel for the recurrent blocks

    class _KV:
        def __init__(self, owner):
            self.owner = owner

        class _Cursor:
            def __init__(self, owner):
                self.owner = owner

            def fill_(self, length):
                # Truncating the consumed record IS the KV rewind: positions [0, L) keep the
                # prefix's own K/V and everything after is abandoned.
                del self.owner.consumed[length:]

        @property
        def cache_seqlens(self):
            return self._Cursor(self.owner)

    @property
    def kv_cache_manager(self):
        return self._KV(self)


class StubRecurrentCache:
    """A state every token mutates, so forgetting to restore it is observable."""

    def __init__(self):
        self.recurrent_state = None
        self.has_state = False
        self.absorbed = []

    class _T(list):
        def clone(self):
            return StubRecurrentCache._T(self)

        def copy_(self, other):
            self[:] = other

    def __getattr__(self, name):
        if name.startswith("conv_state_"):
            value = self._T()
            object.__setattr__(self, name, value)
            return value
        raise AttributeError(name)


class StubModel:
    def __init__(self, n_blocks=2):
        self.mixers = [StubMixer() for _ in range(n_blocks)]
        for mixer in self.mixers:
            mixer.state_cache = StubRecurrentCache()
            for axis in "qkv":
                setattr(mixer.state_cache, f"conv_state_{axis}", StubRecurrentCache._T())
        self.blocks = {str(i): type("B", (), {"attention": m})() for i, m in enumerate(self.mixers)}

    def __call__(self, tensor, logits_to_keep=1, cache_leftpad=None):
        ids = tensor[0]
        for mixer in self.mixers:
            mixer.consumed.extend(ids)
            # conv_state_q stands in for the GDN state: every token mutates it, and unlike the KV
            # cursor nothing truncates it -- only an explicit restore can wind it back.
            mixer.state_cache.conv_state_q.extend(ids)
        return list(self.mixers[0].consumed)  # "logits" = what the model has now seen


class StubGM:
    def __init__(self):
        self.model = StubModel()

    def prepare_inference_cache(self, batch, max_length):
        for mixer in self.model.mixers:
            mixer.consumed.clear()
            mixer.state_cache.absorbed.clear()


@pytest.fixture
def torch_stub(monkeypatch):
    """A `torch` stand-in: this module needs only `tensor`, `zeros` and `no_grad`."""
    import sys
    import types

    module = types.ModuleType("torch")
    module.tensor = lambda data, device=None: list(data)
    module.zeros = lambda *a, **k: []
    module.int32 = "int32"

    class _NoGrad:
        def __enter__(self):
            return None

        def __exit__(self, *exc):
            return False

    module.no_grad = _NoGrad
    monkeypatch.setitem(sys.modules, "torch", module)
    monkeypatch.setitem(sys.modules, "olmo_core.nn.attention", types.SimpleNamespace(Attention=()))
    return module


def run_group(prefills, min_prefix=4):
    gm = StubGM()
    seen = []
    return (
        pc.generate_group_with_shared_prefix(
            gm,
            prefills,
            device=None,
            max_length=99,
            decode_fn=lambda logits: seen.append(list(logits)) or "",
            min_prefix=min_prefix,
        ),
        seen,
    )


def test_each_query_is_conditioned_on_its_own_full_prompt(torch_stub):
    """
    The equivalence claim, stated as a test: with the cache on, the token sequence the model has
    consumed when a query decodes must equal that query's whole prompt -- not the prefix plus some
    earlier query's suffix.
    """
    corpus = list(range(50))
    prefills = [corpus + [900 + i, 910 + i] for i in range(4)]
    (_, stats), seen = run_group(prefills)

    assert stats["prefix_len"] == 50
    assert seen == [list(p) for p in prefills], "a query saw a prompt that was not its own"


def test_the_recurrent_state_is_rewound_between_queries(torch_stub):
    """
    Rewinding the KV cursor but not the GDN state is silent: every query after the first is
    conditioned on its predecessor's suffix, and the output is merely wrong rather than an error.

    After the group, the recurrent state must hold the prefix plus ONLY the last query's suffix. If
    it holds every suffix, the restore never happened.
    """
    corpus = list(range(50))
    prefills = [corpus + [900 + i] for i in range(3)]
    gm = StubGM()
    pc.generate_group_with_shared_prefix(
        gm, prefills, device=None, max_length=99, decode_fn=lambda logits: "", min_prefix=4
    )
    for mixer in gm.model.mixers:
        assert list(mixer.state_cache.conv_state_q) == corpus + [
            902
        ], "the recurrent state accumulated earlier queries' suffixes"


def test_it_falls_back_to_plain_prefills_when_there_is_nothing_to_reuse(torch_stub):
    """The multiplexed-task case: rows share a corpus, but the question comes first, so there is no
    token prefix and the cache must not pretend otherwise."""
    corpus = list(range(50))
    prefills = [[900 + i] + corpus for i in range(3)]
    (_, stats), seen = run_group(prefills)

    assert stats["prefix_len"] == 0
    assert stats["shared"] == 0.0
    assert seen == [list(p) for p in prefills]


def test_a_single_row_group_takes_the_plain_path(torch_stub):
    (_, stats), seen = run_group([list(range(80))])
    assert stats["prefix_len"] == 0
    assert len(seen) == 1


def test_the_reported_saving_matches_the_tokens_actually_prefilled(torch_stub):
    """A speedup number that is computed rather than measured is how 'shared documents' got quoted
    as 'shared tokens' in the first place."""
    corpus = list(range(90))
    prefills = [corpus + [900 + i] for i in range(5)]
    (_, stats), _ = run_group(prefills)

    naive = sum(len(p) for p in prefills)
    actual = stats["prefill_tokens"] + stats["suffix_tokens"]
    assert stats["shared"] == pytest.approx(1 - actual / naive)
    assert actual == 90 + 5


def test_a_prefix_shorter_than_the_threshold_takes_the_plain_path(torch_stub):
    """
    Not a correctness property -- reusing a 3-token prefix still produces the right answer -- but a
    cost one: below the threshold the snapshot/restore bookkeeping costs more than the prefill it
    saves, and a cache that reports itself active while saving nothing is how a speedup gets quoted
    that does not exist.
    """
    prefills = [[1, 2, 3] + [900 + i] * 40 for i in range(3)]
    (_, stats), seen = run_group(prefills, min_prefix=10)

    assert pc.longest_common_token_prefix(prefills) == 3
    assert stats["prefix_len"] == 0, "took the caching path for a 3-token prefix"
    assert seen == [list(p) for p in prefills]
