"""Unit tests for scripts/lib/olmo_flex_attention.py.

Pure-torch (no CUDA, no flex kernel, no olmo-core import): we materialize each
FlexAttention ``mask_mod`` over the full (q, kv) grid and compare against an
independent dense boolean reference, then assert the concrete chunked / modified-SWA
semantics on a small fixture (instruction prefix + 2 docs + query/answer + eos + pad).

Run on a GPU node (login nodes have no python env):
    sbatch jobs/test_olmo_flex_attention.sh      # or: python -m pytest scripts/tests/test_olmo_flex_attention.py
"""

import torch

from corpus_reasoning.lib.chunked_attention import (
    FREE_CHUNK_ID,
    PAD_CHUNK_ID,
    AttentionPattern,
    build_dense_bool_mask,
)
from corpus_reasoning.lib.olmo_flex_attention import (
    SINK_CHUNK_ID,
    build_roles,
    make_mask_mod,
    modified_swa_dense_bool,
)

# Fixture marker / sentinel token ids (arbitrary, distinct).
EOS = 1000
DS = 1001  # doc_start
DE = 1002  # doc_end

# ids: [pref pref] [DS d d DE] [DS d d DE] [q q] eos pad
IDS = [50, 51, DS, 60, 61, DE, DS, 70, 71, DE, 80, 81, EOS, EOS]
#       0   1   2   3   4   5   6   7   8   9   10  11  12   13
INPUT = torch.tensor([IDS], dtype=torch.long)
S = len(IDS)

# A version with NO document markers (mask-mixing "present_as_standard" case).
IDS_NOMARK = [50, 51, 60, 61, 70, 71, 80, 81, EOS, EOS]
INPUT_NOMARK = torch.tensor([IDS_NOMARK], dtype=torch.long)
S_NOMARK = len(IDS_NOMARK)


def _materialize(mask_mod, B, S):
    """Evaluate a mask_mod over the full grid -> (B, S, S) bool (True = attend)."""
    q = torch.arange(S).view(S, 1).expand(S, S)
    kv = torch.arange(S).view(1, S).expand(S, S)
    out = torch.zeros((B, S, S), dtype=torch.bool)
    for b in range(B):
        out[b] = mask_mod(b, 0, q, kv).bool()
    return out


def test_roles_chunked():
    roles = build_roles(INPUT, DS, DE, EOS, mode="chunked")[0].tolist()
    assert roles[0] == FREE_CHUNK_ID and roles[1] == FREE_CHUNK_ID   # prefix is FREE
    assert roles[2:6] == [0, 0, 0, 0]                                # doc 0 (incl markers)
    assert roles[6:10] == [1, 1, 1, 1]                               # doc 1
    assert roles[10] == FREE_CHUNK_ID and roles[11] == FREE_CHUNK_ID  # query/answer
    assert roles[12] == FREE_CHUNK_ID                                # real eos is FREE
    assert roles[13] == PAD_CHUNK_ID                                 # trailing pad


def test_roles_modified_swa_marks_sink():
    roles = build_roles(INPUT, DS, DE, EOS, mode="modified_swa")[0].tolist()
    assert roles[0] == SINK_CHUNK_ID and roles[1] == SINK_CHUNK_ID   # prefix is SINK
    assert roles[2:6] == [0, 0, 0, 0]
    assert roles[10] == FREE_CHUNK_ID and roles[11] == FREE_CHUNK_ID  # query stays FREE
    assert roles[13] == PAD_CHUNK_ID


def test_chunked_matches_dense_reference():
    roles = build_roles(INPUT, DS, DE, EOS, mode="chunked")
    mm = make_mask_mod("chunked", roles, window=0)
    flex = _materialize(mm, 1, S)
    dense = build_dense_bool_mask(AttentionPattern(name="chunked"), roles)
    assert torch.equal(flex, dense)


def test_modified_swa_matches_dense_reference():
    for w in (0, 1, 3, 100):
        roles = build_roles(INPUT, DS, DE, EOS, mode="modified_swa")
        mm = make_mask_mod("modified_swa", roles, window=w)
        flex = _materialize(mm, 1, S)
        dense = modified_swa_dense_bool(roles, w)
        assert torch.equal(flex, dense), f"mismatch at W={w}"


def test_chunked_semantics():
    roles = build_roles(INPUT, DS, DE, EOS, mode="chunked")
    m = _materialize(make_mask_mod("chunked", roles, 0), 1, S)[0]
    # within-doc attends; cross-doc does not.
    assert m[8, 7]      # doc1 token -> doc1 token (same chunk)
    assert not m[8, 4]  # doc1 token -> doc0 token (cross chunk) blocked
    # query (free) sees all prior non-pad; docs see the prefix (free).
    assert m[11, 4]     # query -> doc0 token
    assert m[4, 0]      # doc0 token -> prefix (free)
    # nobody attends to pad except the diagonal.
    assert not m[:, 13].any() or m[13, 13]
    assert not m[11, 13]


def test_modified_swa_semantics():
    roles = build_roles(INPUT, DS, DE, EOS, mode="modified_swa")
    w = 1
    m = _materialize(make_mask_mod("modified_swa", roles, w), 1, S)[0]
    # doc token attends to the prefix sink globally...
    assert m[7, 0] and m[7, 1]
    # ...and to within-window neighbours (diff <= W=1)...
    assert m[7, 6]
    # ...but NOT to far tokens outside the window / sink / free.
    assert not m[7, 3]
    # query/answer (free) attends to everything prior.
    assert m[11, 3] and m[11, 7]
    # causality holds and pad is never attended to (except diagonal NaN-guard).
    assert not m[3, 7]
    assert not m[11, 13]


def _causal_nonpad_reference(input_ids, eos_id, add_diagonal):
    """Plain causal mask with pad (after first eos) blocked — what an all-FREE
    (mask-mixed-out, marker-free) example should reduce to.

    `add_diagonal` toggles the diagonal NaN-guard: only `modified_swa` keeps the
    diagonal always-on (so fully-masked pad rows don't NaN); `chunked` has no such
    guard, so its pad rows are fully False.
    """
    ids = input_ids[0].tolist()
    n = len(ids)
    try:
        pad_from = ids.index(eos_id) + 1
    except ValueError:
        pad_from = n
    ref = torch.zeros((n, n), dtype=torch.bool)
    for qi in range(n):
        for ki in range(n):
            nonpad = qi < pad_from and ki < pad_from
            ref[qi, ki] = (qi >= ki) and nonpad
    if add_diagonal:
        ref |= torch.eye(n, dtype=torch.bool)
    return ref


def test_mixed_standard_collapses_to_causal():
    """A marker-free example (the mask-mixing "present_as_standard" case) has no doc
    spans -> every non-pad token is FREE -> both chunked and modified_swa collapse to
    plain causal. This is what makes tokenize-time mixing equivalent to standard
    attention for the mixed-out fraction. (modified_swa additionally keeps the diagonal
    NaN-guard, chunked does not — the only difference between the two here.)"""
    for mode in ("chunked", "modified_swa"):
        roles = build_roles(INPUT_NOMARK, DS, DE, EOS, mode=mode)
        r = roles[0].tolist()
        # No doc indices; everything up to eos is FREE, trailing is PAD. No SINK
        # (sink only exists relative to a first document, which is absent here).
        assert all(x == FREE_CHUNK_ID for x in r[:8]), (mode, r)
        assert r[8] == FREE_CHUNK_ID, (mode, r)   # real eos is FREE
        assert r[9] == PAD_CHUNK_ID, (mode, r)    # trailing pad
        m = _materialize(make_mask_mod(mode, roles, 4), 1, S_NOMARK)[0]
        ref = _causal_nonpad_reference(INPUT_NOMARK, EOS, add_diagonal=(mode == "modified_swa"))
        assert torch.equal(m, ref), f"{mode}: marker-free example did not collapse to causal"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\nall {len(fns)} tests passed")
