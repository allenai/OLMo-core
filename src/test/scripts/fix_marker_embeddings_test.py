"""
Tests for the marker-embedding audit in ``src/scripts/data/fix_marker_embeddings.py``.

Both gates matter, and the second one is the one that has actually cost experiments. An untrained
reserved row is *bit-identical* to its neighbours (so the model cannot tell an open marker from a
close one) **and** far below a trained row's norm -- and because RMSNorm rescales every position to
the same RMS, a low-norm row is *amplified* into a full-strength meaningless vector rather than being
ignored. On marker-dense data that flatlines training at CE ~0.79 for every mask, including plain
causal, which reads as "the mask is too restrictive" instead of as an embedding bug.

The repair half needs the model's kernels (Qwen3.5 is a GDN hybrid and needs triton), so these tests
cover the audit and the repair *arithmetic* on a synthetic matrix rather than a real checkpoint.
"""

import importlib.util
from pathlib import Path

import pytest
import torch

from olmo_core.data.document_chunk_landmark import RESERVED_IDS, reserved_ids

SCRIPT = Path(__file__).parent.parent.parent / "scripts" / "data" / "fix_marker_embeddings.py"

TRAINED_NORM = 1.415
SUMMARY_MARKERS = ["doc_start", "doc_end", "summary", "pad"]


@pytest.fixture(scope="module")
def script():
    spec = importlib.util.spec_from_file_location("fix_marker_embeddings", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _synthetic_embedding(ids_set, marker_ids, *, dim: int = 64, poison: bool = True):
    """A matrix whose trained rows look trained and whose reserved rows look untrained."""
    torch.manual_seed(0)
    emb = torch.randn(max(marker_ids.values()) + 8, dim)
    real = emb[: ids_set.real_vocab_size]
    emb[: ids_set.real_vocab_size] = real * (TRAINED_NORM / real.norm(dim=-1, keepdim=True))
    if poison:
        # This is what an untrained padded row actually looks like: one shared small vector.
        shared = torch.randn(dim) * 0.02
        for tid in marker_ids.values():
            emb[tid] = shared.clone()
    return emb


def test_marker_set_resolves_named_markers(script):
    ids_set = reserved_ids("qwen3_5")
    markers = script._marker_ids(ids_set, SUMMARY_MARKERS)
    assert markers["summary"] == ids_set.summary
    assert markers["doc_start"] == ids_set.doc_start
    assert len(set(markers.values())) == len(SUMMARY_MARKERS)


def test_unknown_marker_is_rejected(script):
    with pytest.raises(SystemExit, match="unknown marker"):
        script._marker_ids(reserved_ids("qwen3_5"), ["not_a_marker"])


def test_audit_flags_an_unrepaired_base(script):
    """The signature of an untrained row: cosine 1.0 and a norm far below the trained median."""
    ids_set = reserved_ids("qwen3_5")
    markers = script._marker_ids(ids_set, SUMMARY_MARKERS)
    report = script.audit(_synthetic_embedding(ids_set, markers), ids_set, markers)

    assert not report["audit_pass"]
    assert not report["cosine_gate_pass"]
    assert not report["norm_gate_pass"]
    assert all(abs(c - 1.0) < 1e-5 for c in report["pairwise_cosine"].values())
    assert all(report["bit_identical"].values())
    assert all(r < 0.5 for r in report["marker_norm_ratios"].values())


def test_repair_clears_both_gates(script):
    ids_set = reserved_ids("qwen3_5")
    markers = script._marker_ids(ids_set, SUMMARY_MARKERS)
    emb = _synthetic_embedding(ids_set, markers)

    # The script's arithmetic: a trained donor row plus a small jitter.
    g = torch.Generator().manual_seed(34521)
    std = emb[: ids_set.real_vocab_size].float().std()
    for offset, name in enumerate(SUMMARY_MARKERS):
        donor = 10 + offset
        emb[markers[name]] = emb[donor].float() + torch.randn(emb.shape[1], generator=g) * (
            std * 0.1
        )

    report = script.audit(emb, ids_set, markers)
    assert report["audit_pass"], report
    for ratio in report["marker_norm_ratios"].values():
        assert script.NORM_RATIO_MIN < ratio < script.NORM_RATIO_MAX


def test_norm_gate_catches_a_cosine_only_repair(script):
    """
    The historical near-miss: markers made mutually distinguishable but left at ~1/3.6 of a trained
    row's norm. The cosine gate passes and the norm gate must not.
    """
    ids_set = reserved_ids("qwen3_5")
    markers = script._marker_ids(ids_set, SUMMARY_MARKERS)
    emb = _synthetic_embedding(ids_set, markers)

    torch.manual_seed(1)
    for tid in markers.values():
        vec = torch.randn(emb.shape[1])
        emb[tid] = vec * (0.396 / vec.norm())  # the measured norm of the first, broken repair

    report = script.audit(emb, ids_set, markers)
    assert report["cosine_gate_pass"], "distinct directions should clear the cosine gate"
    assert not report["norm_gate_pass"], "a 0.396-norm marker must NOT pass"
    assert not report["audit_pass"]


@pytest.mark.parametrize("family", sorted(RESERVED_IDS))
def test_every_family_can_name_a_donor_for_its_summary_row(script, family):
    """A family with a summary id but no donor could not be repaired for the summary-token path."""
    ids_set = reserved_ids(family)
    assert ids_set.summary > 0
    assert "summary" in script.DONOR_TOKENS
    script._marker_ids(ids_set, ["summary"])


def test_qwen35_uses_its_own_tokenizer_for_donors(script):
    """
    Qwen3.5's vocabulary is a different vocabulary from Qwen3's, so resolving donors against the
    wrong tokenizer would seed the markers from unrelated rows -- silently, since every id resolves.
    """
    assert script.DEFAULT_TOKENIZERS["qwen3_5"].startswith("Qwen/Qwen3.5")
    assert script.DEFAULT_TOKENIZERS["qwen3"].startswith("Qwen/Qwen3-")
    assert ("qwen3_5", "4B") in script.MODEL_BUILDERS
