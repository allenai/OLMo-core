"""
Tests for the summary-token mask when an instance holds **several packed SFT examples**
(``PackingInstanceSource``) rather than exactly one padded example (``PadToLengthInstanceSource``).

The organizing property is :func:`test_packed_mask_is_the_direct_sum_of_standalone_masks`: packing is
correct exactly when the mask over a packed instance is the **block diagonal** of the masks each
example would have received on its own, and nothing else. Everything below it is a named
decomposition of that one property, kept separate because each names a distinct way the old
single-example role builder failed:

* every packed example after the first was classified ``PAD`` outright (padding was "everything after
  the *first* EOS"), so its tokens attended nothing and its labels trained through a dead row;
* ``doc_id`` was a running count over the whole window, so "an earlier document" reached across the
  pack boundary into a previous example's summary runs;
* only the last example's trailing region was ``QUERY``; earlier examples' query/answer spans fell
  through to ``INSTRUCTION``, which is globally readable -- turning the answers of earlier examples
  into free context for later ones;
* the causal arm was one flag per instance, so a packed pair could not hold one masked and one causal
  example, and a causal example could read its neighbours.

``regime`` is parametrized throughout because telling a terminator from padding depends entirely on
the tokenizer's id assignment, and the regime these runs actually use -- ``production`` -- is the
least obvious of the three. See :data:`REGIMES`.
"""

from typing import List, Optional

import pytest
import torch

from olmo_core.nn.attention.summary_mask import (
    N_ROLE_FIELDS,
    ROLE_DOC_ID,
    ROLE_EXAMPLE_ID,
    ROLE_KIND,
    SummaryMaskSpec,
    TokenKind,
    build_summary_mask_mod,
    build_summary_roles,
    summary_mask_allowed,
)
from olmo_core.nn.attention.summary_token import build_summary_block_mask

DOC_START, DOC_END, SUMM = 900, 901, 902
EOS = 903
#: A reserved pad id, distinct from EOS.
PAD_DISTINCT = 904

#: The three regimes that differ in how padding can be told apart from a terminator. ``production``
#: is the one ``sft_summtoken`` actually runs and is easy to overlook: Qwen3.5 ties
#: ``pad_token_id == eos_token_id``, so the data loader fills the tail with **EOS-valued** tokens,
#: while the ``pad_id`` handed to the roles builder is the reserved marker id (248203) which never
#: appears in the stream at all. A rule that leans on ``pad_id`` to find the tail therefore finds
#: nothing, and one that leans on "the first EOS" finds the first example's terminator.
#: name -> (token written into the padded tail, pad_id passed to build_summary_roles)
REGIMES = {
    "reserved_pad": (PAD_DISTINCT, PAD_DISTINCT),
    "eos_as_pad": (EOS, EOS),
    "production": (EOS, PAD_DISTINCT),
}
ALL_REGIMES = list(REGIMES)

N_SUMMARY = 3
BLOCK = 128

SPECS = [
    SummaryMaskSpec(n_summary_tokens=N_SUMMARY),
    SummaryMaskSpec(n_summary_tokens=N_SUMMARY, summary_visible_tokens=1),
    SummaryMaskSpec(n_summary_tokens=N_SUMMARY, summary_visible_tokens=0),
    SummaryMaskSpec(n_summary_tokens=N_SUMMARY, summaries_read_own_document=False),
    SummaryMaskSpec(n_summary_tokens=N_SUMMARY, summaries_read_earlier_summaries=False),
    SummaryMaskSpec(n_summary_tokens=N_SUMMARY, query_reads_documents=True),
]


def _ids_kw(regime: str) -> dict:
    return dict(
        doc_start_id=DOC_START,
        doc_end_id=DOC_END,
        summary_token_id=SUMM,
        eos_id=EOS,
        pad_id=REGIMES[regime][1],
    )


def _terminator_is_pad(regime: str) -> bool:
    """Whether an example's own EOS ends up classified PAD (only when pad_id *is* the EOS id)."""
    return REGIMES[regime][1] == EOS


def _example_tokens(
    n_docs: int, doc_len: int, *, instr_len: int = 3, query_len: int = 4
) -> List[int]:
    """One complete EOS-terminated example: ``[instruction][<doc><summ>]*[<query>][eos]``."""
    ids = [10 + i for i in range(instr_len)]
    for d in range(n_docs):
        ids += [DOC_START] + [20 + d * 10 + i for i in range(doc_len)] + [DOC_END]
        ids += [SUMM] * N_SUMMARY
    ids += [DOC_START] + [50 + i for i in range(query_len)] + [DOC_END] + [EOS]
    return ids


#: Deliberately ragged: different document counts and lengths, so a bug that happens to align the
#: examples cannot pass. Documents are non-repeating across examples too.
SHAPES = [(3, 4), (2, 6), (4, 3)]


def _pack(shapes, *, regime: str, total_len: Optional[int] = None):
    """Pack ``shapes`` into one instance with tail padding, as ``PackingInstanceSource`` emits.

    :returns: ``(ids (1, T), [slice per example])``.
    """
    ids: List[int] = []
    spans = []
    for n_docs, doc_len in shapes:
        tokens = _example_tokens(n_docs, doc_len)
        spans.append(slice(len(ids), len(ids) + len(tokens)))
        ids += tokens
    pad_id = REGIMES[regime][0]
    if total_len is None:
        total_len = len(ids) + 7  # a non-trivial amount of tail padding
    assert total_len >= len(ids)
    ids += [pad_id] * (total_len - len(ids))
    return torch.tensor([ids]), spans


# ---------------------------------------------------------------------------------------------
# Roles
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("regime", ALL_REGIMES)
def test_packed_examples_are_numbered_and_doc_ids_restart(regime):
    """``example_id`` separates the examples; ``doc_id`` is example-local, so it restarts at 0."""
    ids, spans = _pack(SHAPES, regime=regime)
    roles = build_summary_roles(ids, **_ids_kw(regime))
    assert roles.shape[1] == N_ROLE_FIELDS

    ex, doc, kind = roles[0, ROLE_EXAMPLE_ID], roles[0, ROLE_DOC_ID], roles[0, ROLE_KIND]
    for i, span in enumerate(spans):
        live = [p for p in range(span.start, span.stop) if int(kind[p]) != TokenKind.PAD]
        assert live, "an example was classified entirely as padding"
        assert {int(ex[p]) for p in live} == {i}
        # Document indices are local: every example counts its own documents from 0, and the
        # trailing query lands one past the last of them.
        n_docs = SHAPES[i][0]
        content_docs = {int(doc[p]) for p in live if int(kind[p]) == TokenKind.DOC_CONTENT}
        assert content_docs == set(range(n_docs))
        query_docs = {int(doc[p]) for p in live if int(kind[p]) == TokenKind.QUERY}
        assert query_docs == {n_docs}


@pytest.mark.parametrize("regime", ALL_REGIMES)
def test_every_packed_example_gets_its_own_query_region(regime):
    """
    Each example's trailing span must be ``QUERY``, not ``INSTRUCTION``.

    Under a single global ``n_docs`` only the *last* example was trailing, and the earlier examples'
    query/answer spans fell through to ``INSTRUCTION`` -- which every token may read. That is worse
    than a leak: it promotes earlier answers to globally-readable context.
    """
    ids, spans = _pack(SHAPES, regime=regime)
    roles = build_summary_roles(ids, **_ids_kw(regime))
    kind = roles[0, ROLE_KIND]

    for i, span in enumerate(spans):
        kinds = [int(kind[p]) for p in range(span.start, span.stop)]
        assert TokenKind.QUERY in kinds, f"example {i} has no query region"
        # The instruction prefix is the only instruction text in an example: three tokens.
        assert kinds.count(int(TokenKind.INSTRUCTION)) == 3, f"example {i} leaked extra INSTRUCTION"


@pytest.mark.parametrize("regime", ALL_REGIMES)
def test_padding_is_only_the_tail(regime):
    """
    Content past the first EOS is content, not padding.

    "Everything after the first EOS is PAD" is the single-example rule; applied to a pack it wipes
    out every example after the first, and does so silently because padding is legal input.
    """
    ids, spans = _pack(SHAPES, regime=regime)
    roles = build_summary_roles(ids, **_ids_kw(regime))
    kind = roles[0, ROLE_KIND]
    T = ids.shape[1]

    body_end = spans[-1].stop
    for i, span in enumerate(spans):
        n_live = sum(1 for p in range(span.start, span.stop) if int(kind[p]) != int(TokenKind.PAD))
        # Every example keeps essentially all its tokens. When pad ties eos the terminator itself is
        # PAD (it carries no content), so allow exactly that one.
        expected_pad = 1 if _terminator_is_pad(regime) else 0
        assert (
            n_live == (span.stop - span.start) - expected_pad
        ), f"example {i} was eaten by padding"

    assert all(int(kind[p]) == int(TokenKind.PAD) for p in range(body_end, T))
    assert roles[0, ROLE_EXAMPLE_ID][body_end:].eq(-1).all()


# ---------------------------------------------------------------------------------------------
# The organizing property
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("regime", ALL_REGIMES)
@pytest.mark.parametrize("spec", SPECS)
@pytest.mark.parametrize("causal", [False, True])
def test_packed_mask_is_the_direct_sum_of_standalone_masks(regime, spec, causal):
    """
    The mask over a pack must be the block diagonal of the standalone masks, and nothing else.

    This is the whole correctness claim in one assertion: on the diagonal, packing must not change
    what an example sees; off it, packing must not create anything. Both halves matter -- a rule that
    merely blocked cross-example edges could still corrupt the within-example mask by renumbering
    documents, and a rule that got each example right in isolation could still leak across.
    """
    ids_kw = _ids_kw(regime)
    packed_ids, spans = _pack(SHAPES, regime=regime)
    packed = summary_mask_allowed(
        build_summary_roles(packed_ids, **ids_kw),
        spec,
        causal_example=torch.tensor([causal]),
    )[0]

    for i, span in enumerate(spans):
        standalone_ids = torch.tensor([_example_tokens(*SHAPES[i])])
        standalone = summary_mask_allowed(
            build_summary_roles(standalone_ids, **ids_kw),
            spec,
            causal_example=torch.tensor([causal]),
        )[0]
        assert torch.equal(packed[span, span], standalone), f"example {i} changed under packing"

    # Off-diagonal: no edge between any two distinct examples, in either direction. Guarded on the
    # examples being live, so this cannot be satisfied by an all-PAD block.
    roles = build_summary_roles(packed_ids, **ids_kw)
    for span in spans:
        assert (roles[0, ROLE_KIND][span] != int(TokenKind.PAD)).any()
    for i, qs in enumerate(spans):
        for j, ks in enumerate(spans):
            if i != j:
                assert not packed[qs, ks].any(), f"example {i} can read example {j}"

    # And the tail padding neither attends nor is attended (bar the self-diagonal NaN guard).
    T = packed_ids.shape[1]
    tail = range(spans[-1].stop, T)
    for p in tail:
        assert not any(packed[p, c] for c in range(T) if c != p)
        assert not any(packed[q, p] for q in range(T) if q != p)


@pytest.mark.parametrize("regime", ALL_REGIMES)
def test_no_cross_example_edges_by_role(regime):
    """The direct-sum property, named per role, so a failure says which edge came back."""
    ids, spans = _pack(SHAPES, regime=regime)
    roles = build_summary_roles(ids, **_ids_kw(regime))
    allowed = summary_mask_allowed(roles, SummaryMaskSpec(n_summary_tokens=N_SUMMARY))[0]
    kind = roles[0, ROLE_KIND]

    def positions(example: int, want) -> List[int]:
        span = spans[example]
        found = [p for p in range(span.start, span.stop) if int(kind[p]) == int(want)]
        # Without this, every assertion below is vacuously true whenever the roles are broken in a
        # way that empties a probe set -- which is exactly what the old builder did, classifying
        # every example after the first as PAD. "Nothing reaches across" must not be satisfiable by
        # there being nothing there.
        assert found, f"no {want.name} tokens in example {example}; the probe set is empty"
        return found

    # A later example's query must not reach an earlier example's summary runs. Under a global
    # doc_id these were "earlier documents" and every one of these edges was open.
    assert not any(
        allowed[q, s]
        for q in positions(2, TokenKind.QUERY)
        for s in positions(0, TokenKind.SUMMARY)
    )
    # ...nor its raw content, instruction prefix, or answer span.
    for role in (TokenKind.DOC_CONTENT, TokenKind.INSTRUCTION, TokenKind.QUERY):
        assert not any(
            allowed[q, c] for q in positions(2, TokenKind.QUERY) for c in positions(0, role)
        ), f"query of example 2 reads {role.name} of example 0"

    # A later example's documents and summary runs are likewise sealed off.
    assert not any(
        allowed[q, s]
        for q in positions(1, TokenKind.SUMMARY)
        for s in positions(0, TokenKind.SUMMARY)
    )
    assert not any(
        allowed[q, c]
        for q in positions(1, TokenKind.DOC_CONTENT)
        for c in positions(0, TokenKind.DOC_CONTENT)
    )
    # Sanity: the *within*-example edges those mirror are open, so the assertions above are not
    # passing because everything is masked.
    assert all(
        allowed[q, s]
        for q in positions(2, TokenKind.QUERY)
        for s in positions(2, TokenKind.SUMMARY)
    )


# ---------------------------------------------------------------------------------------------
# The causal arm
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("regime", ALL_REGIMES)
def test_causal_arm_stays_inside_its_example(regime):
    """
    A causal example is plain causal **within itself**, not across the pack.

    Otherwise the causal control would not be the control: an arm trained on packed instances would
    see cross-example context that the same arm on unpacked instances never sees.
    """
    ids, spans = _pack(SHAPES, regime=regime)
    roles = build_summary_roles(ids, **_ids_kw(regime))
    allowed = summary_mask_allowed(
        roles, SummaryMaskSpec(n_summary_tokens=N_SUMMARY), causal_example=torch.tensor([True])
    )[0]

    kind = roles[0, ROLE_KIND]
    later, earlier = spans[2], spans[0]
    # Both spans must actually hold live tokens, or "reads nothing across the boundary" is satisfied
    # by an all-PAD row rather than by the rule -- the way the pre-packing builder satisfied it.
    for name, span in (("later", later), ("earlier", earlier)):
        assert any(
            int(kind[p]) != int(TokenKind.PAD) for p in range(span.start, span.stop)
        ), f"the {name} example is entirely PAD; the assertion below would be vacuous"

    assert not allowed[later, earlier].any()
    # Within the example it really is plain causal: the query now reaches raw document content.
    q = [p for p in range(later.start, later.stop) if int(kind[p]) == int(TokenKind.QUERY)]
    c = [p for p in range(later.start, later.stop) if int(kind[p]) == int(TokenKind.DOC_CONTENT)]
    assert q and c
    assert all(allowed[i, j] for i in q for j in c)


@pytest.mark.parametrize("regime", ALL_REGIMES)
def test_per_example_causal_arms_within_one_pack(regime):
    """
    Each packed example draws its own arm, so one instance can hold both.

    With a single ``(B,)`` flag the mixture's unit is the *instance*; the realized causal fraction
    would then depend on how the packer binned examples, and ``standard_mix_prob=0.5`` would no
    longer mean half the examples.
    """
    ids, spans = _pack(SHAPES, regime=regime)
    roles = build_summary_roles(ids, **_ids_kw(regime))
    kind = roles[0, ROLE_KIND]

    # Examples 0 and 2 causal, example 1 masked -- expressed per token, as the model builds it.
    ce = torch.zeros(1, ids.shape[1], dtype=torch.bool)
    for i in (0, 2):
        ce[0, spans[i]] = True
    allowed = summary_mask_allowed(
        roles, SummaryMaskSpec(n_summary_tokens=N_SUMMARY), causal_example=ce
    )[0]

    def q_and_content(i):
        span = spans[i]
        return (
            [p for p in range(span.start, span.stop) if int(kind[p]) == int(TokenKind.QUERY)],
            [p for p in range(span.start, span.stop) if int(kind[p]) == int(TokenKind.DOC_CONTENT)],
        )

    for i in (0, 2):
        q, c = q_and_content(i)
        assert all(allowed[a, b] for a in q for b in c), f"example {i} should be causal"
    q, c = q_and_content(1)
    assert not any(allowed[a, b] for a in q for b in c), "example 1 should be masked"


def test_per_token_causal_flag_shape_is_validated():
    ids, _ = _pack(SHAPES, regime="production")
    roles = build_summary_roles(ids, **_ids_kw("production"))
    spec = SummaryMaskSpec(n_summary_tokens=N_SUMMARY)
    with pytest.raises(ValueError, match="per-token"):
        summary_mask_allowed(roles, spec, causal_example=torch.zeros(1, 5, dtype=torch.bool))


# ---------------------------------------------------------------------------------------------
# The two renderings, and the analytic block mask
# ---------------------------------------------------------------------------------------------


def _dense_mask_mod(roles, spec, causal_example=None):
    T = roles.shape[-1]
    mod = build_summary_mask_mod(roles, spec, causal_example=causal_example)
    b = torch.zeros(T, T, dtype=torch.long)
    q = torch.arange(T).view(-1, 1).expand(T, T)
    k = torch.arange(T).view(1, -1).expand(T, T)
    return mod(b, b, q, k)


@pytest.mark.parametrize("spec", SPECS)
@pytest.mark.parametrize("mode", ["masked", "causal", "mixed"])
def test_dense_matches_mask_mod_when_packed(spec, mode):
    """The dense and FlexAttention renderings must not drift apart under packing either."""
    ids, spans = _pack(SHAPES, regime="production")
    roles = build_summary_roles(ids, **_ids_kw("production"))
    if mode == "mixed":
        ce = torch.zeros(1, ids.shape[1], dtype=torch.bool)
        ce[0, spans[1]] = True
    else:
        ce = torch.tensor([mode == "causal"])
    dense = summary_mask_allowed(roles, spec, causal_example=ce)[0]
    assert torch.equal(_dense_mask_mod(roles, spec, causal_example=ce), dense)


def _packed_roles_for_blocks(regime: str, seq_len: int = 1024):
    """A pack sized to an exact multiple of the flex block size, with several examples per block."""
    shapes = [(3, 30), (2, 45), (4, 22), (3, 18)]
    ids, spans = _pack(shapes, regime=regime, total_len=seq_len)
    return build_summary_roles(ids, **_ids_kw(regime)), spans


def _long_doc_packed_roles(regime: str, seq_len: int = 4096):
    """A pack whose documents are **longer than the block size**.

    Required to reach the full-block shortcut at all: a block is only marked full when it lies wholly
    inside one document, so with documents shorter than ``BLOCK`` the shortcut never fires and any
    test built on short documents cannot see a full-block bug. Each example here numbers its
    documents 0 and 1, so the local ``doc_id`` collides across examples -- which is the specific
    collision the same-example condition on ``full`` exists to break.
    """
    ids, spans = _pack([(2, 400), (2, 400), (2, 400)], regime=regime, total_len=seq_len)
    return build_summary_roles(ids, **_ids_kw(regime)), spans


def _token_mask_implied_by(block_mask, mask_mod, seq_len: int, block: int = BLOCK):
    """Reconstruct exactly what the kernel computes: full blocks unconditionally, partial via mod."""
    out = torch.zeros(seq_len, seq_len, dtype=torch.bool)
    kv_num, kv_idx = block_mask.kv_num_blocks[0, 0], block_mask.kv_indices[0, 0]
    full_num, full_idx = block_mask.full_kv_num_blocks[0, 0], block_mask.full_kv_indices[0, 0]
    q_off = torch.arange(block).view(-1, 1).expand(block, block)
    k_off = torch.arange(block).view(1, -1).expand(block, block)
    zeros = torch.zeros(block, block, dtype=torch.long)
    for i in range(seq_len // block):
        q0 = i * block
        for j in full_idx[i, : full_num[i]].tolist():
            out[q0 : q0 + block, j * block : (j + 1) * block] = True
        for j in kv_idx[i, : kv_num[i]].tolist():
            out[q0 : q0 + block, j * block : (j + 1) * block] = mask_mod(
                zeros, zeros, q_off + q0, k_off + j * block
            )
    return out


@pytest.mark.parametrize("regime", ALL_REGIMES)
@pytest.mark.parametrize("spec", SPECS)
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("fixture", ["short_docs", "long_docs"])
def test_block_mask_implies_exact_token_mask_when_packed(regime, spec, causal, fixture):
    """
    What the block-sparse kernel would compute must equal the dense rule, full blocks included.

    Packing is where a *full* block is most dangerous. ``doc_id`` restarts per example, so two blocks
    lying in different examples' document 2 look identical to the block-level test that declares a
    block full -- and a block wrongly marked full skips ``mask_mod`` entirely, so no amount of
    correctness in the predicate would catch it.

    Both fixtures are needed: ``long_docs`` is the only one that reaches the full-block path at all
    (see :func:`_long_doc_packed_roles`), and ``short_docs`` is the one where examples share blocks
    and the partial-block bookkeeping is dense.
    """
    seq_len, roles = (
        (1024, _packed_roles_for_blocks(regime, 1024)[0])
        if fixture == "short_docs"
        else (4096, _long_doc_packed_roles(regime, 4096)[0])
    )
    ce = torch.tensor([causal])
    mask_mod = build_summary_mask_mod(roles, spec, causal_example=ce)
    block_mask = build_summary_block_mask(roles, spec, causal_example=ce, block_size=BLOCK)
    assert block_mask is not None
    implied = _token_mask_implied_by(block_mask, mask_mod, seq_len)
    expected = summary_mask_allowed(roles, spec, causal_example=ce)[0]
    assert torch.equal(implied, expected)


@pytest.mark.parametrize("regime", ALL_REGIMES)
def test_full_blocks_lie_within_a_single_example(regime):
    """
    Directly pin the full-block shortcut: no block declared full may span two examples.

    The exactness test above would catch this too, but only through its end result. This one names
    the condition, and -- crucially -- asserts the fixture *produces* full blocks, so it cannot pass
    by never reaching the shortcut. (Dropping the same-example condition from ``full`` takes this
    fixture from 8 full blocks to 36 and corrupts ~459k token pairs, with the predicate never run.)
    """
    seq_len = 4096
    roles, _ = _long_doc_packed_roles(regime, seq_len)
    block_mask = build_summary_block_mask(roles, SPECS[0], block_size=BLOCK)
    assert int(block_mask.full_kv_num_blocks.sum()) > 0, "fixture never reaches the full-block path"

    example_id = roles[0, ROLE_EXAMPLE_ID]
    full_num = block_mask.full_kv_num_blocks[0, 0]
    full_idx = block_mask.full_kv_indices[0, 0]

    def examples_in(block: int) -> set:
        ids = example_id[block * BLOCK : (block + 1) * BLOCK]
        return {int(v) for v in ids if int(v) >= 0}

    for q in range(seq_len // BLOCK):
        for k in full_idx[q, : full_num[q]].tolist():
            q_ex, k_ex = examples_in(q), examples_in(k)
            assert len(q_ex) == 1 and q_ex == k_ex, (
                f"full block ({q}, {k}) spans examples q={q_ex} k={k_ex}; the predicate that would "
                "have separated them is skipped"
            )


def test_block_mask_handles_a_per_token_causal_flag():
    """A mixed pack must still round-trip through the block builder's per-block causal reduction."""
    seq_len = 1024
    roles, spans = _packed_roles_for_blocks("production", seq_len)
    spec = SPECS[0]
    ce = torch.zeros(1, seq_len, dtype=torch.bool)
    ce[0, spans[1]] = True
    ce[0, spans[3]] = True
    mask_mod = build_summary_mask_mod(roles, spec, causal_example=ce)
    block_mask = build_summary_block_mask(roles, spec, causal_example=ce, block_size=BLOCK)
    implied = _token_mask_implied_by(block_mask, mask_mod, seq_len)
    expected = summary_mask_allowed(roles, spec, causal_example=ce)[0]
    assert torch.equal(implied, expected)


def test_analytic_block_set_is_a_superset_of_the_reference_when_packed():
    """Dropping a block the predicate needs is silent data loss; extra blocks are only slower."""
    from torch.nn.attention.flex_attention import create_block_mask

    seq_len = 1024
    roles, _ = _packed_roles_for_blocks("production", seq_len)
    spec = SPECS[0]
    mask_mod = build_summary_mask_mod(roles, spec)
    analytic = build_summary_block_mask(roles, spec, block_size=BLOCK)
    reference = create_block_mask(
        mask_mod, 1, None, seq_len, seq_len, device="cpu", BLOCK_SIZE=(BLOCK, BLOCK)
    )
    assert bool((reference.to_dense()[0, 0].bool() <= analytic.to_dense()[0, 0].bool()).all())


def test_a_packed_window_stays_block_sparse():
    """
    Packing must not spend on attention what it saves on padding.

    A pack is block-diagonal by construction, so filling a window with examples instead of padding
    cannot approach the density of plain causal attention over that window (~0.5 of the blocks).
    This is the cost side of the trade the packing option exists to make; it is a loose bound, not a
    tuned one, so it fails only on a real regression in the analytic block set.
    """
    seq_len = 2048
    nb = seq_len // BLOCK
    packed_roles, _ = _packed_roles_for_blocks("production", seq_len)
    bm = build_summary_block_mask(packed_roles, SPECS[0], block_size=BLOCK)
    density = (int(bm.kv_num_blocks.sum()) + int(bm.full_kv_num_blocks.sum())) / (nb * nb)

    causal_density = (nb * (nb + 1) // 2) / (nb * nb)
    assert (
        density < 0.5 * causal_density
    ), f"packed window is not block-sparse (density {density:.3f})"


# ---------------------------------------------------------------------------------------------
# Numerics
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("spec", SPECS)
@pytest.mark.parametrize("mode", ["masked", "mixed"])
def test_flex_output_matches_the_dense_mask_when_packed(spec, mode):
    """
    End to end on the values, not just the mask: the block-sparse kernel and the materialized mask
    must agree numerically on a packed instance, including a pack holding both arms.
    """
    from torch.nn.attention.flex_attention import flex_attention

    torch.manual_seed(0)
    seq_len, n_heads, head_dim = 1024, 2, 32
    roles, spans = _packed_roles_for_blocks("production", seq_len)
    if mode == "mixed":
        ce = torch.zeros(1, seq_len, dtype=torch.bool)
        ce[0, spans[1]] = True
        ce[0, spans[2]] = True
    else:
        ce = torch.tensor([False])

    q, k, v = (torch.randn(1, n_heads, seq_len, head_dim, dtype=torch.float64) for _ in range(3))
    scale = head_dim**-0.5
    block_mask = build_summary_block_mask(roles, spec, causal_example=ce, block_size=BLOCK)
    got = flex_attention(q, k, v, block_mask=block_mask, scale=scale)

    allowed = summary_mask_allowed(roles, spec, causal_example=ce)
    bias = torch.where(
        allowed.unsqueeze(1),
        torch.zeros((), dtype=q.dtype),
        torch.full((), torch.finfo(q.dtype).min, dtype=q.dtype),
    )
    want = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, attn_mask=bias, is_causal=False, scale=scale
    )
    torch.testing.assert_close(got, want, atol=1e-9, rtol=1e-7)


def test_a_packed_examples_output_does_not_depend_on_its_neighbours():
    """
    The strongest numerical statement available: changing the *content* of one packed example must
    not change another example's output values at all.

    Masks can be right while values leak through a kernel that ignores them, and this is the property
    an experiment built on packing actually depends on.
    """
    from torch.nn.attention.flex_attention import flex_attention

    torch.manual_seed(0)
    seq_len, n_heads, head_dim = 1024, 2, 32
    spec = SPECS[0]
    roles, spans = _packed_roles_for_blocks("production", seq_len)
    q, k, v = (torch.randn(1, n_heads, seq_len, head_dim, dtype=torch.float64) for _ in range(3))
    scale = head_dim**-0.5
    block_mask = build_summary_block_mask(roles, spec, block_size=BLOCK)

    base = flex_attention(q, k, v, block_mask=block_mask, scale=scale)
    # Perturb example 0's activations only. Roles are unchanged, so the mask is identical.
    q2, k2, v2 = q.clone(), k.clone(), v.clone()
    for t in (q2, k2, v2):
        t[:, :, spans[0], :] += 3.0
    perturbed = flex_attention(q2, k2, v2, block_mask=block_mask, scale=scale)

    later = spans[2]
    torch.testing.assert_close(base[:, :, later, :], perturbed[:, :, later, :], atol=1e-12, rtol=0)
    # The perturbation must actually have done something, or this passes for the wrong reason.
    assert not torch.allclose(base[:, :, spans[0], :], perturbed[:, :, spans[0], :])
