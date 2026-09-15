"""Tests for the soft-token KEEP-TOKEN RULE (``--st-keep-token-rule {first,rule}``).

Which body tokens of a pooled document stay real is the whole effect: the eval-side probe
(``records/outlier-saliency-preview-probe.md`` 5d) measures, at 8 tokens per document on outlier,
random selection at CE 0.558 -- *worse* than keeping nothing (0.470) -- against 0.263 for the first
8 and 0.071 for a cheap-feature rule, which is the gradient oracle's 0.072. This file pins the
training-side port: ``none`` changes nothing, ``first`` is bit-identical to
``--st-header-extra-tokens``, and ``rule`` keeps exactly K per document, non-contiguously, with the
kept tokens excluded from the slot.
"""

import pytest
import torch

from olmo_core.config import DType
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.pooled_soft_token import KeepTokenTables
from olmo_core.nn.transformer import TransformerConfig

DOC_START, DOC_END, EOS, PLACEHOLDER = 900, 901, 999, 902
STOP = 25  # the header terminator (":" on Qwen3.5)
DIGITS = (30, 31, 32, 33)
IGN = -100


def _row(docs):
    """``[free] + <doc_start> tokens <doc_end> ... + [q, EOS]``."""
    ids = [11, 12]  # free preamble; ids here never collide with a body token
    for toks in docs:
        ids += [DOC_START, *toks, DOC_END]
    ids += [51, 52, EOS]
    return ids


# header = "<id> :" ; body = 8 tokens, every other one a "digit"
DOCS = [
    [200 + i, STOP, 20, DIGITS[0], 21, DIGITS[1], 22, DIGITS[2], 23, DIGITS[3]] for i in range(3)
]


def _tables(vocab=1000):
    """Hand-made vocab tables: only the DIGITS ids are digits, nothing else is set."""
    is_dig = torch.zeros(vocab, dtype=torch.bool)
    for i in DIGITS:
        is_dig[i] = True
    return KeepTokenTables(
        idf=torch.zeros(vocab),
        sent_end=torch.zeros(vocab, dtype=torch.bool),
        is_cap=torch.zeros(vocab, dtype=torch.bool),
        is_dig=is_dig,
        tok_len=torch.zeros(vocab),
    )


def _model(**kwargs):
    cfg = TransformerConfig.olmo2_190M(
        vocab_size=1000, n_layers=2, fused_ops=False, dtype=DType.float32
    )
    model = cfg.build(init_device="cpu")
    model.enable_pooled_soft_tokens(
        DOC_START,
        DOC_END,
        EOS,
        placeholder_id=PLACEHOLDER,
        keep_prob=0.0,  # gold-blind, every document pooled
        header_stop_id=STOP,
        header_stop_count=1,
        **kwargs,
    )
    model.train()
    return model


def _compact(model, ids):
    x = torch.tensor([ids])
    lab = torch.full_like(x, IGN)
    lab[0, -3:-1] = x[0, -2:]
    cb, _inject, _oracle = model._compact_pooled_soft_tokens(x, lab, IGN)
    return cb


def _same(a, b):
    return (
        torch.equal(a.input_ids, b.input_ids)
        and torch.equal(a.position_ids, b.position_ids)
        and torch.equal(a.labels, b.labels)
        and torch.equal(a.row_lens, b.row_lens)
        and torch.equal(a.soft_cols, b.soft_cols)
        and torch.allclose(a.soft_log_len, b.soft_log_len)
    )


def test_keep_token_rule_off_is_bit_identical():
    """The default (``"none"``) must reproduce the header-only compaction exactly."""
    ids = _row(DOCS)
    base = _compact(_model(), ids)
    off = _compact(_model(keep_token_rule="none", keep_token_k=0), ids)
    assert _same(base, off)


@pytest.mark.parametrize("k", [1, 4, 8, 32])
def test_keep_token_rule_first_equals_header_extra_tokens(k):
    """``first`` is ``--st-header-extra-tokens`` -- bit-identical, including the K-exceeds-body
    case (k=32) where the whole document is kept real and produces no slot at all."""
    ids = _row(DOCS)
    extra = _compact(_model(header_extra_tokens=k), ids)
    first = _compact(_model(keep_token_rule="first", keep_token_k=k), ids)
    assert _same(extra, first)
    if k == 32:
        assert first.soft_rows.numel() == 0


@pytest.mark.parametrize("k", [1, 2, 3])
def test_keep_token_rule_rule_keeps_k_digits_per_doc(k):
    """With a weight vector that scores only ``is_dig``, exactly the first k digit tokens of each
    pooled document survive into the compacted row -- non-contiguously -- and each document still
    contributes exactly one slot."""
    ids = _row(DOCS)
    cb = _compact(
        _model(
            keep_token_rule="rule",
            keep_token_k=k,
            keep_token_weights={"is_dig": 1.0},
            keep_token_tables=_tables(),
        ),
        ids,
    )
    row = cb.input_ids[0, : cb.row_lens[0]].tolist()
    assert cb.soft_rows.numel() == len(DOCS)  # one slot per pooled document
    assert row.count(PLACEHOLDER) == len(DOCS)
    # Every document contributes: <doc_start> <id> : <k digits> <SLOT> <doc_end>
    assert row.count(DIGITS[0]) == len(DOCS) * (1 if k >= 1 else 0)
    assert row.count(DIGITS[1]) == len(DOCS) * (1 if k >= 2 else 0)
    assert row.count(DIGITS[2]) == len(DOCS) * (1 if k >= 3 else 0)
    assert row.count(DIGITS[3]) == 0  # the 4th digit is never inside a k<=3 budget
    # and no NON-digit body token survives (20/21/22/23 are pooled away)
    assert all(t not in row for t in (20, 21, 22, 23))
    # header + markers stay real for every document
    assert row.count(STOP) == len(DOCS)
    # slot length: 10 doc tokens - 2 header - k kept = 8 - k, + the two markers
    expected = torch.full((len(DOCS),), float(torch.tensor(float(10 - k)).log()))
    assert torch.allclose(cb.soft_log_len, expected)


def test_keep_token_rule_rule_beats_first_at_picking_digits():
    """The rule and ``first`` spend the same budget on DIFFERENT tokens -- the point of the knob."""
    ids = _row(DOCS)
    first = _compact(_model(keep_token_rule="first", keep_token_k=2), ids)
    rule = _compact(
        _model(
            keep_token_rule="rule",
            keep_token_k=2,
            keep_token_weights={"is_dig": 1.0},
            keep_token_tables=_tables(),
        ),
        ids,
    )
    assert first.row_lens.tolist() == rule.row_lens.tolist()  # identical cost
    assert not torch.equal(first.input_ids, rule.input_ids)  # different tokens
    r_first = first.input_ids[0, : first.row_lens[0]].tolist()
    r_rule = rule.input_ids[0, : rule.row_lens[0]].tolist()
    assert r_first.count(DIGITS[0]) == len(DOCS) and r_first.count(20) == len(DOCS)
    assert r_rule.count(DIGITS[0]) == len(DOCS) and r_rule.count(20) == 0


def test_keep_token_rule_composes_with_gold_blind_and_slot_mode():
    """Gold-blind keeping (``keep_prob``) and ``--st-slot-mode cent_cmean`` are orthogonal to the
    rule: a kept document is untouched, a pooled one gets its k rule tokens, and the slot feature
    is built from the surviving tokens only."""
    from olmo_core.nn.attention.pooled_doc_kv import PooledDocKeepHolder

    ids = _row(DOCS)
    model = _model(
        keep_token_rule="rule",
        keep_token_k=2,
        keep_token_weights={"is_dig": 1.0},
        keep_token_tables=_tables(),
        slot_mode="cent_cmean",
        slot_stop_ids=[DOC_START, DOC_END, EOS, STOP],
    )
    model._pooled_keep_holder = PooledDocKeepHolder(keep_docs=torch.tensor([[True, False, False]]))
    cb = _compact(model, ids)
    row = cb.input_ids[0, : cb.row_lens[0]].tolist()
    assert cb.soft_rows.numel() == 2  # only the two pooled docs get slots
    assert row.count(20) == 1  # the kept document keeps its whole body
    assert row.count(DIGITS[3]) == 1  # ... including the digits outside any budget


def test_keep_token_rule_config_validation():
    with pytest.raises(OLMoConfigurationError):  # unknown rule
        _model(keep_token_rule="salience", keep_token_k=4)
    with pytest.raises(OLMoConfigurationError):  # k must be > 0
        _model(keep_token_rule="first", keep_token_k=0)
    with pytest.raises(OLMoConfigurationError):  # mutually exclusive with header_extra_tokens
        _model(keep_token_rule="first", keep_token_k=4, header_extra_tokens=4)
    with pytest.raises(OLMoConfigurationError):  # 'rule' needs the feature tables
        _model(keep_token_rule="rule", keep_token_k=4)
