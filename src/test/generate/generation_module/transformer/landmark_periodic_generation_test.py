"""SFT-aligned insertion, content-only outputs, and cached periodic-block generation."""

from types import SimpleNamespace

import pytest
import torch

from olmo_core.data.composable.landmark_packing_instance_source import LandmarkPackingInstanceSource
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.generate.generation_module.transformer.generation_module import (
    _periodic_landmark_budget,
)

from .landmark_generation_test import MEM_ID, _build_module

GEN_MODULE = "olmo_core.generate.generation_module.transformer.generation_module"


def _sft_ids(prompt, output, mem_freq):
    """Use the real SFT packer's document transform with an input/output loss-mask boundary."""
    source = SimpleNamespace(
        mem_freq=mem_freq,
        mem_id=MEM_ID,
        num_landmarks=1,
        pad_id=0,
        block_size=mem_freq + 1,
        exclude_landmark_predictors=False,
    )
    ids, _ = LandmarkPackingInstanceSource._emit_document(
        source, prompt + output, [False] * len(prompt) + [True] * len(output)
    )
    return ids


def _record_forward(monkeypatch, gm):
    seen = []

    def forward(ids, **kwargs):
        seen.append(ids.clone())
        logits = torch.zeros(ids.shape[0], 1, 512)
        # The model would prefer to emit a marker at EVERY content position. The mode must suppress
        # that, and choose the best actual content token instead; forced slots bypass sampling.
        logits[..., MEM_ID] = 10
        logits[..., 7] = 5
        return logits

    monkeypatch.setattr(gm, "_prefill_forward", forward)
    return seen


@pytest.mark.parametrize("prompt_len", [1, 3, 4, 5, 6, 8])
@pytest.mark.parametrize("top_k", [None, 1])
def test_periodic_layout_matches_sft_and_preserves_budget(monkeypatch, prompt_len, top_k):
    monkeypatch.setenv("LM_SPARSE_KERNEL", "0")
    gm = _build_module(
        "periodic", max_new_tokens=12, landmark_top_k_blocks=top_k, landmark_top_k_fraction=None
    )
    prompt = torch.full((2, prompt_len), 9, dtype=torch.long)
    seen = _record_forward(monkeypatch, gm)
    out, logits, logprobs = gm.generate_batch(
        prompt, return_logits=True, return_logprobs=True, log_timing=False
    )
    assert out.tolist() == [[9] * prompt_len + [7] * 12] * 2
    assert logits.shape == (2, 12, 512)
    assert logprobs.shape == (2, 12)
    assert torch.isfinite(logprobs).all()
    # The final sampled token has not yet been fed through the model, as in normal autoregression.
    physical = torch.cat(seen + [out[:, -1:]], dim=1)
    gold = _sft_ids([9] * prompt_len, [7] * 12, 4)
    assert physical.tolist() == [gold[: physical.shape[1]]] * 2
    assert all(
        a._eval_prompt_len is None and a._eval_top_k is None
        for a in gm._landmark_attention_layers()
    )


def test_periodic_user_example_23_prompt_tokens_then_40_output_tokens(monkeypatch):
    gm = _build_module("periodic", mem_freq=63, max_new_tokens=110)
    seen = _record_forward(monkeypatch, gm)
    prompt = torch.full((1, 63 + 23), 9, dtype=torch.long)
    completion, _, _ = gm.generate_batch(prompt, completions_only=True, log_timing=False)
    physical = torch.cat(seen + [completion[:, -1:]], dim=1)[0].tolist()
    assert physical == _sft_ids(prompt[0].tolist(), [7] * 110, 63)[: len(physical)]
    assert [i for i, token in enumerate(physical) if token == MEM_ID] == [63, 127, 191]
    physical_output = physical[87:]
    assert physical_output[:40] == [7] * 40
    assert physical_output[40] == MEM_ID
    assert completion.tolist() == [[7] * 110]


@pytest.mark.parametrize("prompt_len", [3, 4, 6])
@pytest.mark.parametrize("chunk_size", [None, 5])
def test_periodic_cached_logits_match_full_sft_forward(monkeypatch, prompt_len, chunk_size):
    """Check real model/cache/RoPE execution against a full teacher-forced SFT sequence."""
    monkeypatch.setenv("LM_SPARSE_KERNEL", "0")
    gm = _build_module(
        "periodic", max_new_tokens=9, prefill_chunk_size=chunk_size, landmark_top_k_fraction=None
    )
    output = list(range(20, 29))
    choices = iter(output)
    monkeypatch.setattr(
        GEN_MODULE + ".select_next_token",
        lambda logits, **kw: logits.new_full((1,), next(choices), dtype=torch.long),
    )
    prompt = torch.full((1, prompt_len), 9, dtype=torch.long)
    completion, logits, _ = gm.generate_batch(
        prompt, completions_only=True, return_logits=True, log_timing=False
    )
    assert completion.tolist() == [output]
    gm.free_inference_cache()
    gold = torch.tensor([_sft_ids([9] * prompt_len, output, 4)])
    with torch.inference_mode():
        full_logits = gm.model_forward(gold)
    # A content target's predictor may be a forced landmark; selecting the previous content
    # position instead would lose that landmark's recurrent/cache contribution.
    targets = [i for i, token in enumerate(gold[0].tolist()) if token in output]
    ref = full_logits[:, [i - 1 for i in targets]].clone()
    ref[..., MEM_ID] = float("-inf")
    torch.testing.assert_close(logits, ref, atol=2e-5, rtol=2e-5)


def test_periodic_strip_before_stop_string_decode(monkeypatch):
    gm = _build_module("periodic", max_new_tokens=20)
    seen = _record_forward(monkeypatch, gm)
    tokens = iter([2, 3, 4, 5, 6])
    monkeypatch.setattr(
        GEN_MODULE + ".select_next_token",
        lambda logits, **kw: logits.new_full((1,), next(tokens), dtype=torch.long),
    )

    class Tokenizer:
        def decode(self, ids, **kwargs):
            assert MEM_ID not in ids
            return "".join({2: "Ans", 3: "wer:", 4: " ", 5: "x", 6: "\n"}[i] for i in ids)

    # Prompt ends one content token before the landmark: "Ans<landmark>wer: x\n" must stop.
    out, _, _ = gm.generate_batch(
        torch.tensor([[9, 9, 9]]),
        completions_only=True,
        stop_strings=["Answer:"],
        stop_string_tokenizer=Tokenizer(),
        log_timing=False,
    )
    assert out.tolist() == [[2, 3, 4, 5, 6]]
    assert MEM_ID in torch.cat(seen, dim=1)[0].tolist()


@pytest.mark.parametrize("stop_token", [1, 8])
def test_periodic_eos_and_stop_tokens_after_forced_landmark(monkeypatch, stop_token):
    gm = _build_module("periodic", max_new_tokens=20, stop_token_ids=[8, MEM_ID])
    _record_forward(monkeypatch, gm)
    tokens = iter([7, stop_token])
    monkeypatch.setattr(
        GEN_MODULE + ".select_next_token",
        lambda logits, **kw: logits.new_full((1,), next(tokens), dtype=torch.long),
    )
    out, _, _ = gm.generate_batch(
        torch.tensor([[9, 9, 9]]), completions_only=True, log_timing=False
    )
    # The forced landmark neither consumes the two-token visible completion nor acts as a stop.
    assert out.tolist() == [[7, stop_token]]


def test_periodic_physical_context_cap(monkeypatch):
    gm = _build_module("periodic", max_new_tokens=10, max_length=29)
    seen = _record_forward(monkeypatch, gm)
    prompt = torch.full((1, 14), 9, dtype=torch.long)
    out, _, _ = gm.generate_batch(prompt, completions_only=True, log_timing=False)
    assert out.shape == (1, 10)
    assert sum(x.shape[1] for x in seen) + 1 == 29
    with pytest.raises(OLMoConfigurationError, match="physical token positions"):
        gm.generate_batch(prompt, max_length=28, log_timing=False)
    # With only a physical cap, derive the visible budget, ignoring a trailing structural slot.
    out, _, _ = gm.generate_batch(
        prompt, max_new_tokens=None, max_length=28, completions_only=True, log_timing=False
    )
    assert out.shape == (1, 9)
    assert _periodic_landmark_budget(14, 4, 0, 17) == (17, 0)
    with pytest.raises(OLMoConfigurationError, match="physical token positions"):
        _periodic_landmark_budget(14, 4, None, 16)


def test_periodic_mode_rejects_premarked_prompt_and_ragged_api():
    gm = _build_module("periodic")
    with pytest.raises(OLMoConfigurationError, match="content-only prompt"):
        gm.generate_batch(torch.tensor([[9, MEM_ID]]), log_timing=False)
    with pytest.raises(OLMoConfigurationError, match="generate_batch"):
        gm.generate_landmark_batch([[9, 9]], max_new_tokens=3)


def test_periodic_finished_rows_keep_structural_slots(monkeypatch):
    gm = _build_module("periodic", max_new_tokens=20)
    seen = _record_forward(monkeypatch, gm)
    tokens = iter([[1, 7], [9, 7], [9, 1]])
    monkeypatch.setattr(
        GEN_MODULE + ".select_next_token",
        lambda logits, **kw: torch.tensor(next(tokens), dtype=torch.long),
    )
    out, _, _ = gm.generate_batch(
        torch.full((2, 3), 9, dtype=torch.long), completions_only=True, log_timing=False
    )
    assert out.tolist() == [[1, 1, 1], [7, 7, 1]]
    # The first row is finished before this slot, but both cache streams remain periodic.
    assert torch.cat(seen, dim=1)[:, 4].tolist() == [MEM_ID, MEM_ID]
