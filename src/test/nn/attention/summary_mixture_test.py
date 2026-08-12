"""
Tests for the causal/masked **mixture** that drives SummTokenSFT: the schedule
(:func:`~olmo_core.nn.attention.chunked_mask.mask_mix_standard_prob`), the per-example arm
(:func:`~olmo_core.nn.attention.chunked_mask.causal_example_flags`), and the wiring that gets both
from :class:`~olmo_core.nn.transformer.Transformer` down to the attention layers.

Two things here are guards rather than behaviour checks:

* ``mix_schedule="linear"`` must stay **bit-identical** to the pre-existing behaviour, because live
  document-chunked arms depend on it;
* the model must record how many layers the mask actually covers. On a hybrid that is a minority,
  and a run whose config claims otherwise would misreport what the experiment showed.
"""

import logging

import pytest
import torch

from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.attention import AttentionType, AttentionTypePatternConfig
from olmo_core.nn.attention.chunked_mask import (
    causal_example_flags,
    collapse_roles_to_causal,
    mask_mix_standard_prob,
)
from olmo_core.nn.attention.summary_token import SummaryTokenAttention
from olmo_core.nn.transformer import TransformerConfig

DOC_START, DOC_END, SUMM, EOS, PAD = 900, 901, 902, 903, 904
VOCAB = 1024


# ---------------------------------------------------------------------------------------------
# Schedule
# ---------------------------------------------------------------------------------------------


def test_linear_schedule_is_unchanged():
    """Regression guard: live chunked arms depend on this exact curve."""
    kw = dict(mix_start_p=0.8, mix_end_p=0.0, mix_total_forwards=100)
    assert mask_mix_standard_prob(0, **kw) == pytest.approx(0.8)
    assert mask_mix_standard_prob(50, **kw) == pytest.approx(0.4)
    assert mask_mix_standard_prob(100, **kw) == pytest.approx(0.0)
    assert mask_mix_standard_prob(500, **kw) == pytest.approx(0.0)
    # Static wins over the curriculum, and no mixing configured means no mixing.
    assert mask_mix_standard_prob(50, standard_mix_prob=0.3, **kw) == pytest.approx(0.3)
    assert mask_mix_standard_prob(50) == 0.0


@pytest.mark.parametrize(
    "forward_idx,expected", [(0, 0.0), (49, 0.0), (50, 1.0), (99, 1.0), (500, 1.0)]
)
def test_step_schedule_switches_once(forward_idx, expected):
    assert mask_mix_standard_prob(
        forward_idx,
        mix_start_p=0.0,
        mix_end_p=1.0,
        mix_total_forwards=100,
        mix_schedule="step",
        mix_step_frac=0.5,
    ) == pytest.approx(expected)


def test_step_frac_moves_the_switch():
    kw = dict(mix_start_p=0.0, mix_end_p=1.0, mix_total_forwards=100, mix_schedule="step")
    assert mask_mix_standard_prob(24, mix_step_frac=0.25, **kw) == pytest.approx(0.0)
    assert mask_mix_standard_prob(25, mix_step_frac=0.25, **kw) == pytest.approx(1.0)


def test_unknown_schedule_raises():
    with pytest.raises(ValueError, match="mix_schedule"):
        mask_mix_standard_prob(
            0, mix_start_p=0.0, mix_end_p=1.0, mix_total_forwards=10, mix_schedule="cosine"
        )


# ---------------------------------------------------------------------------------------------
# Per-example arm
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("p", [0.0, 0.25, 0.5, 1.0])
def test_flags_match_the_legacy_collapse_choices(p):
    """
    The explicit flag and the legacy roles-to-FREE collapse must pick the *same* examples.

    They are two expressions of one schedule; if they diverged, a summary-token arm and a
    document-chunked arm run at the same seed would silently not be comparable.
    """
    batch = 16
    forward_idx, seed = 7, 42
    chunk_ids = torch.arange(batch).view(batch, 1).expand(batch, 8).contiguous().to(torch.int32)
    collapsed = collapse_roles_to_causal(chunk_ids, p, forward_idx=forward_idx, mix_seed=seed)
    # An example was collapsed iff its roles became all-FREE (-1).
    legacy = torch.tensor([bool((collapsed[b] == -1).all()) for b in range(batch)])
    flags = causal_example_flags(batch, p, forward_idx=forward_idx, mix_seed=seed)
    assert torch.equal(flags, legacy)


def test_flags_are_deterministic_and_respond_to_forward_index():
    a = causal_example_flags(32, 0.5, forward_idx=3, mix_seed=42)
    b = causal_example_flags(32, 0.5, forward_idx=3, mix_seed=42)
    c = causal_example_flags(32, 0.5, forward_idx=4, mix_seed=42)
    assert torch.equal(a, b)
    assert not torch.equal(a, c)


def test_zero_probability_means_no_causal_examples():
    assert not causal_example_flags(8, 0.0, forward_idx=1).any()


# ---------------------------------------------------------------------------------------------
# Model wiring
# ---------------------------------------------------------------------------------------------


def _example_ids(n_docs: int = 4, n_summary: int = 3):
    ids = [10, 11, 12]
    for d in range(n_docs):
        ids += [DOC_START] + [20 + d] * 10 + [DOC_END] + [SUMM] * n_summary
    return ids + [DOC_START] + [50] * 6 + [DOC_END] + [EOS] + [PAD] * 5


def _build_model(layer_types=None, **enable_kwargs):
    cfg = TransformerConfig.olmo2_190M(vocab_size=VOCAB)
    mixer = cfg.block["sequence_mixer"] if isinstance(cfg.block, dict) else cfg.block.sequence_mixer
    if layer_types is None:
        mixer.name = AttentionType.summary_token
    else:
        mixer.layer_types = AttentionTypePatternConfig(pattern=layer_types)
    mixer.n_summary_tokens = 3
    cfg.summary_token_attention = dict(
        doc_start_id=DOC_START,
        doc_end_id=DOC_END,
        summary_token_id=SUMM,
        eos_id=EOS,
        pad_id=PAD,
        **enable_kwargs,
    )
    return cfg.build(init_device="cpu")


def test_roles_reach_the_attention_layers_and_change_the_output():
    model = _build_model(standard_mix_prob=0.0)
    x = torch.tensor([_example_ids()])
    model.eval()
    with torch.no_grad():
        masked = model(x)
    # Disabling role construction leaves plain causal attention, which must differ.
    model._summary_token_attention = None
    with torch.no_grad():
        causal = model(x)
    assert not torch.allclose(masked, causal)


def test_mixture_is_training_only_and_advances_the_counter():
    model = _build_model(standard_mix_prob=0.5, mix_seed=7)
    x = torch.tensor([_example_ids(), _example_ids()])

    model.train()
    with torch.no_grad():
        first, second = model(x), model(x)
    assert model._summary_token_attention["mix"]["forward_idx"] == 2
    assert not torch.allclose(first, second), "the causal-arm draw should differ between forwards"

    model.eval()
    with torch.no_grad():
        a, b = model(x), model(x)
    assert model._summary_token_attention["mix"]["forward_idx"] == 2, "eval must not advance"
    assert torch.allclose(a, b), "eval must be deterministic"


def test_records_how_many_layers_are_actually_masked(caplog):
    """
    The R2 disclosure. On a hybrid only a minority of layers carry the mask, and that has to be
    recorded rather than remembered -- a run whose config implies full coverage would misdescribe
    what the experiment showed. Exercised here with a mixed attention-type pattern, which reaches the
    same code path as a GDN/attention hybrid without needing its kernels.
    """
    pattern = [AttentionType.summary_token, AttentionType.default]
    with caplog.at_level(logging.INFO, logger="olmo_core.nn.transformer.model"):
        model = _build_model(layer_types=pattern)

    recorded = model._summary_token_attention
    assert recorded["n_sequence_mixers"] == 12
    assert recorded["n_summary_layers"] == 6
    n_masked = sum(
        1
        for b in model.blocks.values()
        if isinstance(getattr(b, "attention", None), SummaryTokenAttention)
    )
    assert n_masked == recorded["n_summary_layers"]
    assert any("UNRESTRICTED cross-document channel" in r.message for r in caplog.records)


def test_enabling_without_any_summary_layer_raises():
    """A mask that covers nothing is a silent no-op, which is the failure mode to avoid."""
    cfg = TransformerConfig.olmo2_190M(vocab_size=VOCAB)
    cfg.summary_token_attention = dict(
        doc_start_id=DOC_START, doc_end_id=DOC_END, summary_token_id=SUMM, eos_id=EOS
    )
    with pytest.raises(OLMoConfigurationError, match="silent no-op"):
        cfg.build(init_device="cpu")


@pytest.mark.parametrize(
    "kwargs,match",
    [
        (
            dict(standard_mix_prob=0.3, mix_start_p=0.8, mix_end_p=0.0, mix_total_forwards=10),
            "mutually exclusive",
        ),
        (dict(mix_start_p=0.8, mix_end_p=0.0), "mix_total_forwards"),
        (
            dict(mix_start_p=0.0, mix_end_p=1.0, mix_total_forwards=10, mix_schedule="cosine"),
            "mix_schedule",
        ),
    ],
)
def test_enable_validates_the_mixture(kwargs, match):
    with pytest.raises(OLMoConfigurationError, match=match):
        _build_model(**kwargs)
