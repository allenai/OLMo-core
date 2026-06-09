"""Model-side landmark handling for HELMET/RULER-style generation.

The generation module inserts landmark tokens into the *prompt* (so prefill sees the trained block
structure) and decodes plain content tokens as "one long local block", so the eval harness only ever
deals with content tokens. These run on CPU using the eager sparse-landmark path.
"""

import os

import pytest
import torch

from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.generate.generation_module import TransformerGenerationModule
from olmo_core.generate.generation_module.config import GenerationConfig
from olmo_core.generate.generation_module.transformer.generation_module import (
    _build_landmark_prompt,
    _insert_landmark_tokens,
)
from olmo_core.nn.attention import AttentionConfig, AttentionType
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.utils import seed_all

MEM_FREQ = 4
MEM_ID = 500


def test_insert_landmark_tokens():
    # Content length a multiple of mem_freq -> a landmark after every block (ends with a landmark).
    x = torch.arange(1, 13).view(1, 12)
    out = _insert_landmark_tokens(x, mem_freq=4, mem_id=999)
    assert out.tolist() == [[1, 2, 3, 4, 999, 5, 6, 7, 8, 999, 9, 10, 11, 12, 999]]

    # Trailing partial block (< mem_freq tokens) is left without a landmark.
    xx = torch.arange(1, 11).view(1, 10)
    out2 = _insert_landmark_tokens(xx, mem_freq=4, mem_id=999)
    assert out2.tolist() == [[1, 2, 3, 4, 999, 5, 6, 7, 8, 999, 9, 10]]

    # Fewer than mem_freq content tokens -> unchanged.
    short = torch.arange(1, 4).view(1, 3)
    assert torch.equal(_insert_landmark_tokens(short, mem_freq=4, mem_id=999), short)


def test_build_landmark_prompt_generation_only_ends_with_landmark():
    # Partial final block (14 not a multiple of mem_freq=4): generation_only pads with pad_id up to
    # the next landmark position, so the prompt ends with a landmark.
    content = torch.arange(1, 15).view(1, 14)
    out = _build_landmark_prompt(content, mem_freq=4, mem_id=999, mode="generation_only", pad_id=7)
    assert out[0, -1].item() == 999  # ends with a landmark
    assert out.shape[1] % (4 + 1) == 0  # whole blocks only
    assert out.tolist() == [
        [1, 2, 3, 4, 999, 5, 6, 7, 8, 999, 9, 10, 11, 12, 999, 13, 14, 7, 7, 999]
    ]

    # extend_last_block leaves the trailing partial block (ends with content, not a landmark).
    ext = _build_landmark_prompt(
        content, mem_freq=4, mem_id=999, mode="extend_last_block", pad_id=7
    )
    assert ext[0, -1].item() != 999
    assert ext.tolist() == [[1, 2, 3, 4, 999, 5, 6, 7, 8, 999, 9, 10, 11, 12, 999, 13, 14]]

    # Content already a multiple of mem_freq: no padding, ends with a landmark in both modes.
    full = torch.arange(1, 13).view(1, 12)
    for mode in ("generation_only", "extend_last_block"):
        o = _build_landmark_prompt(full, mem_freq=4, mem_id=999, mode=mode, pad_id=7)
        assert o[0, -1].item() == 999


def _build_module(decode_mode="extend_last_block", **gen_overrides):
    seed_all(0)
    cfg = TransformerConfig.llama_like(d_model=128, n_heads=4, n_layers=2, vocab_size=512)
    assert not isinstance(cfg.block, dict)
    sm = cfg.block.sequence_mixer
    assert isinstance(sm, AttentionConfig)
    cfg.block.sequence_mixer = AttentionConfig(
        name=AttentionType.sparse_landmark,
        n_heads=4,
        head_dim=32,
        mem_freq=MEM_FREQ,
        num_landmarks=1,
        rope=sm.rope,
    )
    gen_kwargs = dict(
        max_new_tokens=10,
        pad_token_id=0,
        eos_token_id=1,
        do_sample=False,
        use_cache=True,
        landmark_mem_id=MEM_ID,
        landmark_decode_mode=decode_mode,
    )
    gen_kwargs.update(gen_overrides)
    gen_cfg = GenerationConfig(**gen_kwargs)
    return TransformerGenerationModule(
        model=cfg.build(), generation_config=gen_cfg, device=torch.device("cpu")
    )


@pytest.mark.parametrize("decode_mode", ["extend_last_block", "generation_only"])
def test_sparse_landmark_generation_end_to_end(decode_mode):
    os.environ["LM_SPARSE_KERNEL"] = "0"
    gm = _build_module(decode_mode=decode_mode)
    prompt = torch.randint(2, 400, (1, 14))

    out, _, _ = gm.generate_batch(prompt, completions_only=False, log_timing=False)
    # Returned in content space: prompt preserved verbatim (no inserted landmarks), 10 new tokens.
    assert out.shape == (1, 24)
    assert torch.equal(out[:, :14], prompt)
    # The landmark token never appears in the returned sequence (decoded as plain content).
    assert (out == MEM_ID).sum().item() == 0
    # Eval-decode state is cleared after the call.
    assert all(a._eval_prompt_len is None for a in gm._landmark_attention_layers())

    comp, _, _ = gm.generate_batch(prompt, completions_only=True, log_timing=False)
    assert comp.shape == (1, 10)
    assert (comp == MEM_ID).sum().item() == 0


def test_landmark_generation_max_length_budget_accounts_for_inserted_tokens():
    # Eval harnesses (HELMET/RULER) pass an absolute ``max_length`` computed as
    # ``len(content_prompt) + max_gen_toks``, unaware that landmark generation inserts memory tokens
    # into the prompt. Those inserted tokens count toward ``generated.shape[1]``, so the budget must
    # be extended by the inserted-token count or the model under-generates (and zeroes out when the
    # inserted count reaches max_gen_toks). Here a 14-token prompt with mem_freq=4 gets 3 landmarks
    # inserted (prompt_len 14 -> 17); without compensation only 7 of the requested 10 tokens would
    # be produced.
    os.environ["LM_SPARSE_KERNEL"] = "0"
    budget = 10
    prompt = torch.randint(2, 400, (1, 14))
    gm = _build_module(max_new_tokens=None, max_length=prompt.shape[1] + budget)

    out, _, _ = gm.generate_batch(prompt, completions_only=False, log_timing=False)
    # Full content-token budget honored despite the 3 inserted landmark tokens.
    assert out.shape == (1, prompt.shape[1] + budget)
    assert torch.equal(out[:, : prompt.shape[1]], prompt)
    assert (out == MEM_ID).sum().item() == 0

    comp, _, _ = gm.generate_batch(prompt, completions_only=True, log_timing=False)
    assert comp.shape == (1, budget)


def test_landmark_generation_requires_mem_id():
    os.environ["LM_SPARSE_KERNEL"] = "0"
    gm = _build_module()
    gm._generation_config = gm._generation_config.replace(landmark_mem_id=None)
    with pytest.raises(OLMoConfigurationError, match="landmark_mem_id"):
        gm.generate_batch(torch.randint(2, 400, (1, 14)), log_timing=False)


def test_landmark_generation_requires_cache():
    os.environ["LM_SPARSE_KERNEL"] = "0"
    gm = _build_module()
    with pytest.raises(OLMoConfigurationError, match="use_cache"):
        gm.generate_batch(torch.randint(2, 400, (1, 14)), use_cache=False, log_timing=False)
