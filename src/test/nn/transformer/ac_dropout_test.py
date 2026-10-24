"""Activation checkpointing must replay dropout masks, attention dropout included.

Checkpointing runs the forward twice: once to get the output, once during backward to rebuild the
activations it discarded. If the recomputed pass draws *fresh* dropout masks, the gradients are
taken with respect to a different sample than the loss was. Attention dropout is easy to miss here
because it is not an ``nn.Dropout`` module: it is the ``dropout_p`` of the attention backend, and
the attention kernel draws its mask itself.

Attention dropout must also be off in eval mode, like ``nn.Dropout``.
"""

import pytest
import torch
import torch.nn.functional as F

from olmo_core.nn.attention import AttentionBackendName, AttentionConfig
from olmo_core.nn.transformer import TransformerActivationCheckpointingMode
from olmo_core.nn.transformer.config import TransformerBlockConfig, TransformerConfig

VOCAB, SEQ_LEN = 64, 16


def _model(attn_dropout: float = 0.0, dropout: float = 0.0, ac: bool = False, seed: int = 0):
    config = TransformerConfig.llama_like(
        d_model=32,
        vocab_size=VOCAB,
        n_layers=2,
        n_heads=2,
        attn_backend=AttentionBackendName.torch,
    )
    assert isinstance(config.block, TransformerBlockConfig)
    assert isinstance(config.block.sequence_mixer, AttentionConfig)
    config.block.sequence_mixer.dropout = attn_dropout
    config.block.dropout = dropout
    torch.manual_seed(seed)
    model = config.build(init_device="cpu")
    model.init_weights(device=torch.device("cpu"))
    if ac:
        model.apply_activation_checkpointing(TransformerActivationCheckpointingMode.full)
    return model


def _batch() -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(1234)
    input_ids = torch.randint(0, VOCAB, (2, SEQ_LEN), generator=generator)
    labels = torch.randint(0, VOCAB, (2, SEQ_LEN), generator=generator)
    return input_ids, labels


@pytest.mark.parametrize(
    "attn_dropout, dropout, expected",
    [
        pytest.param(0.0, 0.0, False, id="no-dropout"),
        pytest.param(0.1, 0.0, True, id="attention-dropout"),
        pytest.param(0.0, 0.1, True, id="block-dropout"),
    ],
)
def test_preserve_rng_state_tracks_whether_dropout_is_active(
    attn_dropout: float, dropout: float, expected: bool
):
    model = _model(attn_dropout=attn_dropout, dropout=dropout)
    assert model._dropout_is_active() is expected

    model.apply_activation_checkpointing(TransformerActivationCheckpointingMode.full)
    for block in model.blocks.values():
        assert block.checkpoint_fn.keywords["preserve_rng_state"] is expected  # type: ignore


def _grads(attn_dropout: float, dropout: float, ac: bool) -> list[torch.Tensor]:
    model = _model(attn_dropout=attn_dropout, dropout=dropout, ac=ac)
    input_ids, labels = _batch()
    torch.manual_seed(1)  # the same dropout draws for the checkpointed and the plain model
    logits = model(input_ids)
    F.cross_entropy(logits.float().reshape(-1, VOCAB), labels.reshape(-1)).backward()
    return [p.grad.clone() for p in model.parameters() if p.grad is not None]


@pytest.mark.parametrize(
    "attn_dropout, dropout",
    [
        pytest.param(0.1, 0.0, id="attention-dropout"),
        pytest.param(0.0, 0.1, id="block-dropout"),
        pytest.param(0.0, 0.0, id="no-dropout"),
    ],
)
def test_checkpointed_grads_match_uncheckpointed_grads(attn_dropout: float, dropout: float):
    without_ac = _grads(attn_dropout, dropout, ac=False)
    with_ac = _grads(attn_dropout, dropout, ac=True)

    assert len(without_ac) == len(with_ac)
    for expected, actual in zip(without_ac, with_ac):
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def _logits(model, seed: int) -> torch.Tensor:
    input_ids, _ = _batch()
    torch.manual_seed(seed)
    with torch.no_grad():
        return model(input_ids)


def test_attention_dropout_is_off_in_eval():
    model = _model(attn_dropout=0.1).eval()
    reference = _model(attn_dropout=0.0).eval()

    torch.testing.assert_close(_logits(model, seed=1), _logits(reference, seed=1), rtol=0, atol=0)
    torch.testing.assert_close(_logits(model, seed=2), _logits(reference, seed=2), rtol=0, atol=0)


def test_attention_dropout_is_on_in_train():
    model = _model(attn_dropout=0.1).train()

    assert not torch.equal(_logits(model, seed=1), _logits(model, seed=2))
