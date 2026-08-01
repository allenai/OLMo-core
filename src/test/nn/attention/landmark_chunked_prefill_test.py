"""
GPU tests for **chunked prefill** on the landmark attention variants.

A one-shot prefill materializes every intermediate at the full prompt length, which is what makes
ultra-long eval rungs (512k/1M) exceed an 80GB GPU. Feeding the prompt in slices bounds that
transient, but only if each slice, attending over the cached prefix, computes exactly what the
single-shot prefill would have.

That is a sharper requirement here than for plain attention: landmark blocks are tied to *absolute*
position, so a chunk boundary that lands mid-block shifts the whole block structure relative to the
``is_mem`` pattern the kernel is handed. These tests pin down both halves -- that aligned chunking
is equivalent, and that misaligned chunking is refused rather than silently wrong.
"""

import pytest
import torch

from olmo_core.nn.attention import AttentionConfig, AttentionType
from olmo_core.nn.attention.landmark_kernel import has_landmark_kernel
from olmo_core.nn.layer_norm import LayerNormConfig
from olmo_core.testing import requires_gpu

MEM_FREQ = 15
BLOCK = MEM_FREQ + 1  # 16

requires_landmark_kernel = pytest.mark.skipif(
    not has_landmark_kernel(), reason="requires triton landmark kernel"
)


def _build(name: AttentionType, *, d_model: int = 64, device: str = "cuda"):
    attn = AttentionConfig(
        name=name,
        n_heads=4,
        n_kv_heads=4,
        head_dim=16,
        bias=False,
        mem_freq=MEM_FREQ,
        qk_norm=LayerNormConfig(name="rms", eps=1e-6, bias=False),
        use_head_qk_norm=True,
    ).build(d_model, layer_idx=0, n_layers=1, init_device=device)
    attn.eval()
    return attn.to(device)


def _prefill(attn, x, *, chunk_size, max_seq_len):
    """Run ``x`` through the cached prefill path, optionally in ``chunk_size`` slices.

    :returns: ``(output, cache_position)`` -- the concatenated prefill output and the KV cache's
        final position, which must also agree between the chunked and one-shot runs.
    """
    attn.init_kv_cache_manager(1, max_seq_len)
    with torch.no_grad():
        if chunk_size is None:
            out = attn(x)
        else:
            outs = [
                attn(x[:, start : start + chunk_size]) for start in range(0, x.shape[1], chunk_size)
            ]
            out = torch.cat(outs, dim=1)
    pos = int(attn.kv_cache_manager.current_position())
    return out, pos


@requires_gpu
@requires_landmark_kernel
@pytest.mark.parametrize(
    "attn_type",
    [AttentionType.fast_landmark, AttentionType.fast_compressive_landmark],
)
@pytest.mark.parametrize("chunk_blocks", [1, 2, 3])
def test_landmark_chunked_prefill_matches_one_shot(attn_type: AttentionType, chunk_blocks: int):
    """
    Block-aligned chunked prefill must reproduce the single-shot prefill exactly.

    ``chunk_blocks`` sweeps chunk sizes of 1, 2 and 3 blocks against a 5-block prompt, so the sweep
    covers chunk counts that both divide the prompt (1, 2 blocks -> ragged) and do not, and the
    ragged final chunk is exercised in every case where it can occur.
    """
    torch.manual_seed(0)
    device = "cuda"
    attn = _build(attn_type, device=device)
    seq = 5 * BLOCK
    x = torch.randn(1, seq, 64, device=device)

    one_shot, pos_one_shot = _prefill(attn, x, chunk_size=None, max_seq_len=seq + 1)
    chunked, pos_chunked = _prefill(attn, x, chunk_size=chunk_blocks * BLOCK, max_seq_len=seq + 1)

    assert pos_chunked == pos_one_shot == seq
    torch.testing.assert_close(chunked, one_shot, atol=1e-4, rtol=1e-4)


@requires_gpu
@requires_landmark_kernel
@pytest.mark.parametrize(
    "attn_type",
    [AttentionType.fast_landmark, AttentionType.fast_compressive_landmark],
)
def test_landmark_chunked_prefill_rejects_misaligned_chunks(attn_type: AttentionType):
    """
    A chunk that does not start on a block boundary must raise, not silently mis-attend.

    The kernel takes its history implicitly as ``len(k) - len(q)`` and asserts it is a whole number
    of blocks. Without this guard a misaligned chunk either trips that assert somewhere far from the
    cause, or -- worse -- lines up by accident and shifts every landmark position.
    """
    torch.manual_seed(0)
    device = "cuda"
    attn = _build(attn_type, device=device)
    seq = 4 * BLOCK
    x = torch.randn(1, seq, 64, device=device)
    attn.init_kv_cache_manager(1, seq + 1)

    misaligned = BLOCK // 2
    with torch.no_grad():
        attn(x[:, :misaligned])  # first chunk starts at 0, which is fine
        with pytest.raises(NotImplementedError, match="block boundary"):
            attn(x[:, misaligned : misaligned + BLOCK])


@requires_gpu
@requires_landmark_kernel
def test_landmark_chunked_prefill_then_decode_matches_one_shot():
    """
    Chunked prefill must leave the KV cache in the same state a one-shot prefill would, so the
    decode steps that follow are unaffected.

    Prefill correctness alone is not enough for the eval: the generated tokens all come from decode
    steps reading that cache. A prefill that looked right but wrote keys at the wrong offsets would
    pass the test above and still corrupt every generation.
    """
    torch.manual_seed(0)
    device = "cuda"
    attn = _build(AttentionType.fast_compressive_landmark, device=device)
    seq = 4 * BLOCK
    x = torch.randn(1, seq, 64, device=device)
    nxt = torch.randn(1, 1, 64, device=device)

    outs = []
    for chunk_size in (None, BLOCK):
        attn.init_kv_cache_manager(1, seq + 4)
        with torch.no_grad():
            if chunk_size is None:
                attn(x)
            else:
                for start in range(0, seq, chunk_size):
                    attn(x[:, start : start + chunk_size])
            outs.append(attn(nxt))

    torch.testing.assert_close(outs[1], outs[0], atol=1e-4, rtol=1e-4)
