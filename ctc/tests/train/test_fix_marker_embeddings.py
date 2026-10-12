"""
The marker-embedding audit and repair.

This gates *every* document-chunked and landmark run: a base whose reserved marker rows are
bit-identical trains to chance, and the failure reads as a modeling result rather than a data bug.
The script itself can only be exercised end to end against a real distcp checkpoint, so what is
tested here is the tensor logic underneath it, on a synthetic embedding matrix small enough to
reason about.

**Both halves are tested, deliberately.** The first version of the repair fixed the cosine and left
the markers at ~1/3.6 the norm of a trained row, which RMSNorm amplifies into a full-strength noise
vector at every marker position -- that flatlines training at CE ~0.79 for every mask, *including*
plain causal, and cost a set of runs to diagnose. So there is a test for the state that has good
cosines and bad norms, asserting it is still rejected. A test that only checked distinguishability
would have called that base healthy. See ``records/n100-chunked-marker-position-bug.md``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch", reason="needs torch")
pytest.importorskip("olmo_core", reason="needs olmo-core")

from olmo_core.data.document_chunk_landmark import ReservedIds  # noqa: E402

SCRIPTS = Path(__file__).parents[3] / "src" / "scripts" / "ctc"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

fixmark = pytest.importorskip("fix_marker_embeddings", reason="script not importable")


D_MODEL = 16
REAL_VOCAB = 40

#: A miniature stand-in for a real family's reserved ids: 40 trained rows, then four reserved rows
#: (two inside the "real" range like Qwen3's ``<|box_*|>`` reserved specials, two past it like the
#: landmark/pad rows in the padded region).
IDS = ReservedIds(
    doc_start=30,
    doc_end=31,
    eos=29,
    landmark=44,
    pad=45,
    real_vocab_size=REAL_VOCAB,
)


def trained_matrix(seed: int = 0) -> torch.Tensor:
    """An embedding matrix whose trained rows have a well-defined median norm of ~1."""
    g = torch.Generator().manual_seed(seed)
    return torch.nn.functional.normalize(torch.randn(48, D_MODEL, generator=g), dim=-1)


def poisoned_matrix() -> torch.Tensor:
    """The raw-Qwen3 state: all four marker rows are one vector, at a fraction of a trained norm."""
    emb = trained_matrix()
    g = torch.Generator().manual_seed(7)
    row = torch.nn.functional.normalize(torch.randn(D_MODEL, generator=g), dim=-1) * 0.28
    for tid in fixmark.marker_ids(IDS).values():
        emb[tid] = row.clone()
    return emb


def donors() -> dict:
    """Four distinct trained rows to seed from, none of them the EOS row."""
    return {"doc_start": 3, "doc_end": 4, "landmark": 5, "pad": 6}


# ── the audit ───────────────────────────────────────────────────────────────────────────────────


def test_the_raw_base_is_reported_as_both_indistinguishable_and_wrong_norm():
    emb = poisoned_matrix()

    cosines = fixmark.marker_cosines(emb, IDS)
    assert len(cosines) == 6
    assert all(c == pytest.approx(1.0, abs=1e-5) for c in cosines.values())

    ratios = fixmark.marker_norm_ratios(emb, IDS)
    assert all(r < fixmark.MIN_NORM_RATIO for r in ratios.values())

    issues = fixmark.problems(emb, IDS)
    assert any("indistinguishable" in p for p in issues)
    assert any("out of\ndistribution" in p or "out of distribution" in p for p in issues)


def test_a_healthy_matrix_reports_no_problems():
    # Every row is an independent unit vector here, so the markers are both distinguishable from
    # each other and exactly at the trained-row median norm.
    assert fixmark.problems(trained_matrix(), IDS) == []


def test_cosine_fixed_but_norm_still_small_is_STILL_rejected():
    """The historical half-repair: distinguishable markers at ~1/3.6 of a trained row's norm.

    This is the state the first version of the script produced and called success.
    """
    emb = trained_matrix()
    g = torch.Generator().manual_seed(11)
    for tid in fixmark.marker_ids(IDS).values():
        emb[tid] = torch.nn.functional.normalize(torch.randn(D_MODEL, generator=g), dim=-1) / 3.6

    assert all(abs(c) < fixmark.MAX_MARKER_COS for c in fixmark.marker_cosines(emb, IDS).values())
    issues = fixmark.problems(emb, IDS)
    assert issues, "a norm-broken base must not pass just because its cosines are fine"
    assert all("norm" in p for p in issues)


# ── the repair ──────────────────────────────────────────────────────────────────────────────────


def test_repair_fixes_BOTH_the_cosine_and_the_norm():
    emb = poisoned_matrix()
    fixmark.repair_markers(emb, IDS, donors(), verbose=False)

    for (a, b), cos in fixmark.marker_cosines(emb, IDS).items():
        assert abs(cos) < fixmark.MAX_MARKER_COS, f"{a}/{b} still indistinguishable (cos={cos:+.4f})"
    for name, ratio in fixmark.marker_norm_ratios(emb, IDS).items():
        assert fixmark.MIN_NORM_RATIO < ratio < fixmark.MAX_NORM_RATIO, (
            f"{name} norm ratio {ratio:.3f} is out of distribution"
        )
    assert fixmark.problems(emb, IDS) == []


def test_repair_leaves_the_trained_rows_untouched():
    before = poisoned_matrix()
    after = before.clone()
    fixmark.repair_markers(after, IDS, donors(), verbose=False)
    keep = [i for i in range(before.shape[0]) if i not in set(fixmark.marker_ids(IDS).values())]
    assert torch.equal(before[keep], after[keep])


def test_repair_is_reproducible_from_the_seed():
    a, b = poisoned_matrix(), poisoned_matrix()
    fixmark.repair_markers(a, IDS, donors(), seed=123, verbose=False)
    fixmark.repair_markers(b, IDS, donors(), seed=123, verbose=False)
    assert torch.equal(a, b)


def test_repair_stays_close_to_its_donor_row():
    """Each marker should inherit its donor's direction -- that is where the in-distribution norm
    comes from, and it is why delimiters (which already mean "a boundary is here") are the donors."""
    emb = poisoned_matrix()
    donor_of = donors()
    original = emb.clone()
    fixmark.repair_markers(emb, IDS, donor_of, verbose=False)
    for name, tid in fixmark.marker_ids(IDS).items():
        cos = torch.nn.functional.cosine_similarity(
            emb[tid][None], original[donor_of[name]][None]
        ).item()
        assert cos > 0.9, f"{name} drifted from its donor (cos={cos:.4f})"


@pytest.mark.parametrize(
    "bad,why",
    [
        (IDS.eos, "the EOS row is load-bearing and must never be a donor"),
        (REAL_VOCAB + 1, "a row past real_vocab_size is untrained -- exactly what we are repairing"),
        (-1, "out of range"),
    ],
)
def test_an_untrainable_donor_is_refused(bad, why):
    emb = poisoned_matrix()
    d = donors()
    d["doc_start"] = bad
    with pytest.raises(ValueError):
        fixmark.repair_markers(emb, IDS, d, verbose=False)


def test_a_missing_donor_is_refused():
    emb = poisoned_matrix()
    d = donors()
    del d["pad"]
    with pytest.raises(ValueError, match="pad"):
        fixmark.repair_markers(emb, IDS, d, verbose=False)


def test_every_marker_has_a_donor_string_defined():
    """The CLI resolves donors by name from DONOR_STRINGS; a missing entry would KeyError on a
    2 GB checkpoint that had already been loaded."""
    assert set(fixmark.DONOR_STRINGS) == set(fixmark.marker_ids(IDS))
