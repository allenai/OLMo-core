"""
The tokenizer/checkpoint vocabulary guard.

Grading a Qwen3.5 checkpoint (vocab 248,320) with the default Qwen3 tokenizer (151,936) does not
fail -- it completes and reports f1 = 0.000 at parse_rate 0.000, which reads as a dead model. That
is the single most expensive shape of bug in this project: a plausible number from a mis-set flag.
Measured on ``ctc-s5-contra-full-4b`` before the guard existed.

The guard logic is tested directly rather than through a model load, so these run without a GPU.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from ctc.eval.backends.native import NativeBackend

QWEN3 = 151_936
QWEN35 = 248_320


def _backend(tok_vocab: int, model_vocab: int) -> NativeBackend:
    """A bare object carrying only what the guard reads -- no checkpoint, no torch."""
    backend = object.__new__(NativeBackend)
    backend.tok = SimpleNamespace(vocab_size=tok_vocab)
    backend.gm = SimpleNamespace(
        model=SimpleNamespace(
            embeddings=SimpleNamespace(weight=SimpleNamespace(shape=(model_vocab,)))
        )
    )
    return backend


def test_qwen3_tokenizer_on_a_qwen35_checkpoint_is_rejected():
    with pytest.raises(ValueError, match="tokenizer/checkpoint mismatch"):
        _backend(QWEN3, QWEN35)._check_tokenizer_matches_model()


def test_the_error_names_both_tokenizers():
    """The message has to say what to pass instead; an error that only reports a mismatch sends
    the reader back to the checkpoint config to work out the family."""
    with pytest.raises(ValueError) as e:
        _backend(QWEN3, QWEN35)._check_tokenizer_matches_model()
    assert "Qwen/Qwen3-4B" in str(e.value)
    assert "Qwen/Qwen3.5-0.8B-Base" in str(e.value)


@pytest.mark.parametrize("vocab", [QWEN3, QWEN35])
def test_matching_vocabularies_pass(vocab):
    _backend(vocab, vocab)._check_tokenizer_matches_model()


def test_a_padded_model_vocabulary_passes():
    """Models round the embedding up for kernel alignment, so a slightly larger model vocab is
    normal and must not be flagged."""
    _backend(QWEN3, QWEN3 + 128)._check_tokenizer_matches_model()


def test_a_tokenizer_larger_than_the_model_is_rejected():
    """Ids past the embedding would index out of range, so this direction is never padding."""
    with pytest.raises(ValueError, match="different tokenizers"):
        _backend(QWEN35, QWEN3)._check_tokenizer_matches_model()


def test_guard_is_skipped_when_the_model_exposes_no_embedding():
    backend = object.__new__(NativeBackend)
    backend.tok = SimpleNamespace(vocab_size=QWEN3)
    backend.gm = SimpleNamespace(model=SimpleNamespace())
    backend._check_tokenizer_matches_model()
