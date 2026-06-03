import math

import pytest
import torch

from olmo_core.data.composable.concat_and_chunk_instance_source import ConcatAndChunkInstanceSource
from olmo_core.data.composable.landmark_instance_source import LandmarkInstanceSource
from olmo_core.data.composable.token_source import InMemoryTokenSource
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.attention import AttentionConfig, AttentionType, LandmarkAttention
from olmo_core.nn.attention.landmark import landmark_grouped_softmax
from olmo_core.nn.attention.landmark_kernel import fused_landmark_attention, has_landmark_kernel
from olmo_core.nn.layer_norm import LayerNormConfig
from olmo_core.nn.rope import RoPEConfig, RoPEType
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.testing import requires_gpu


def _landmark_attention(
    *,
    d_model: int = 64,
    n_heads: int = 8,
    n_kv_heads: int = 2,
    head_dim: int = 8,
    mem_freq: int = 3,
) -> LandmarkAttention:
    config = AttentionConfig(
        name=AttentionType.landmark,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        bias=False,
        mem_freq=mem_freq,
        qk_norm=LayerNormConfig(name="rms", eps=1e-6, bias=False),
        use_head_qk_norm=True,
        rope=RoPEConfig(name=RoPEType.default, theta=10_000),
    )
    attn = config.build(d_model, layer_idx=0, n_layers=2)
    assert isinstance(attn, LandmarkAttention)
    return attn


def test_landmark_config_builds():
    attn = _landmark_attention(mem_freq=3)
    assert attn.mem_freq == 3
    assert attn.block_size == 4


def test_landmark_mem_freq_rejected_on_non_landmark():
    with pytest.raises(OLMoConfigurationError):
        AttentionConfig(name=AttentionType.default, n_heads=8, mem_freq=3).build(
            64, layer_idx=0, n_layers=1
        )


def test_landmark_eager_forward_shape():
    attn = _landmark_attention()
    attn.eval()
    B, T, d_model = 2, 12, 64  # T multiple of block_size (4)
    x = torch.randn(B, T, d_model)
    with torch.no_grad():
        out = attn(x)
    assert out.shape == (B, T, d_model)
    assert torch.isfinite(out).all()


def test_landmark_eager_training_backward():
    # The default (eager) path must be fully differentiable so training works without the
    # fused kernel, on plain CPU.
    attn = _landmark_attention()
    attn.train()
    assert attn.use_kernel is False
    B, T, d_model = 2, 12, 64
    x = torch.randn(B, T, d_model, requires_grad=True)
    out = attn(x)
    (out**2).sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    grads = [p.grad for p in attn.parameters()]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)


def test_landmark_use_kernel_rejected_on_non_landmark():
    with pytest.raises(OLMoConfigurationError):
        AttentionConfig(name=AttentionType.default, n_heads=8, landmark_use_kernel=True).build(
            64, layer_idx=0, n_layers=1
        )


def test_landmark_requires_seq_len_multiple_of_block_size():
    attn = _landmark_attention(mem_freq=3)  # block_size 4
    attn.eval()
    with pytest.raises(OLMoConfigurationError):
        attn(torch.randn(1, 10, 64))


def test_landmark_rejects_intra_document_masking():
    attn = _landmark_attention()
    attn.eval()
    with pytest.raises(NotImplementedError):
        attn(torch.randn(1, 12, 64), cu_doc_lens=torch.tensor([0, 6, 12], dtype=torch.int32))


def test_landmark_grouped_softmax_rows_sum_to_one():
    attn = _landmark_attention(mem_freq=3)
    B, n_heads, T = 2, 4, 12
    attn_mask, is_mem, last_section_mask = attn._landmark_masks(
        T, torch.device("cpu"), torch.float32
    )
    logits = torch.randn(B, n_heads, T, T)
    logits = logits + attn_mask
    logits = torch.maximum(logits, torch.tensor(torch.finfo(logits.dtype).min))
    probs = landmark_grouped_softmax(
        logits,
        dim=-1,
        is_mem=is_mem.expand(B, n_heads, T, T),
        last_section_mask=last_section_mask.expand(B, 1, T, T),
    )
    # Every query attends to at least itself, so all rows should normalize to 1.
    assert torch.allclose(probs.sum(-1), torch.ones(B, n_heads, T), atol=1e-5)


def _eager_landmark_reference(q, k, v, block_size):
    """Dense eager landmark attention over ``(B, H, T, d)`` tensors (full-context, causal)."""
    B, H, T, d = q.shape
    att = (q @ k.transpose(-1, -2)) / math.sqrt(d)
    att_mask = torch.tril(torch.ones((1, 1, T, T), device=q.device), diagonal=0) == 1.0
    sec = torch.arange(T, device=q.device) // block_size
    last_section_mask = (sec[None, :] == sec[:, None]).unsqueeze(0).unsqueeze(1)
    is_mem = ((torch.arange(T, device=q.device) % block_size) == (block_size - 1)).view(1, 1, 1, T)
    mask = att_mask & ~(last_section_mask & is_mem)
    last_section_mask = (last_section_mask & mask).expand(B, H, T, T)
    is_mem_ = (is_mem & mask).expand(B, H, T, T)
    att = att.masked_fill(~mask, float("-inf"))
    att = landmark_grouped_softmax(att, -1, is_mem_, last_section_mask).to(q.dtype)
    att = att.masked_fill(~mask, 0.0)
    return att @ v


@requires_gpu
@pytest.mark.skipif(not has_landmark_kernel(), reason="requires triton landmark kernel")
@pytest.mark.parametrize("mem_freq", [15, 63])
def test_landmark_kernel_matches_eager(mem_freq: int):
    # The fused kernel's tl.dot requires tile dims >= 16, hence mem_freq >= 15.
    torch.manual_seed(0)
    block_size = mem_freq + 1
    B, n_heads, d = 2, 4, 64
    T = block_size * 4
    attn = _landmark_attention(
        d_model=n_heads * d, n_heads=n_heads, n_kv_heads=n_heads, head_dim=d, mem_freq=mem_freq
    ).cuda()

    q = torch.rand(B, n_heads, T, d, device="cuda", dtype=torch.bfloat16)
    k = torch.rand(B, n_heads, T, d, device="cuda", dtype=torch.bfloat16)
    v = torch.rand(B, n_heads, T, d, device="cuda", dtype=torch.bfloat16)
    is_mem = (torch.arange(T, device="cuda") % block_size) == (block_size - 1)

    out_kernel = fused_landmark_attention(
        q, k, v, is_mem, sm_scale=attn.softmax_scale, block_size=block_size
    )
    out_eager = attn._eager_forward(q, k, v)

    torch.testing.assert_close(out_kernel, out_eager, rtol=1e-2, atol=1e-2)


@requires_gpu
@pytest.mark.skipif(not has_landmark_kernel(), reason="requires triton landmark kernel")
@pytest.mark.parametrize("mem_freq", [15, 63])
def test_landmark_kernel_backward_matches_eager(mem_freq: int):
    # Validate the fused kernel's gradients against the eager autograd reference. Run in fp32 so the
    # comparison is exact (bf16 differs only by accumulation noise).
    torch.manual_seed(0)
    block_size = mem_freq + 1
    B, n_heads, d = 2, 4, 64
    T = block_size * 4
    scale = d**-0.5
    is_mem = (torch.arange(T, device="cuda") % block_size) == (block_size - 1)
    base = torch.rand(B, n_heads, T, d, device="cuda", dtype=torch.float32)
    grad_out = torch.rand_like(base)

    def grads(use_kernel):
        q, k, v = (base.clone().requires_grad_(True) for _ in range(3))
        if use_kernel:
            out = fused_landmark_attention(q, k, v, is_mem, sm_scale=scale, block_size=block_size)
        else:
            out = _eager_landmark_reference(q, k, v, block_size)
        out.backward(grad_out)
        return out, q.grad, k.grad, v.grad

    out_k, dq_k, dk_k, dv_k = grads(True)
    out_e, dq_e, dk_e, dv_e = grads(False)

    torch.testing.assert_close(out_k, out_e, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(dq_k, dq_e, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(dk_k, dk_e, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(dv_k, dv_e, rtol=1e-3, atol=1e-3)


def test_landmark_instance_source(tmp_path):
    mem_freq, mem_id = 3, 999
    # Content tokens 0..23 -> two content instances of length 12 (a multiple of mem_freq).
    tokens = InMemoryTokenSource(tokens=list(range(24)), work_dir=tmp_path)
    content = ConcatAndChunkInstanceSource(tokens, sequence_length=12, work_dir=tmp_path)
    source = LandmarkInstanceSource(content, mem_freq=mem_freq, mem_id=mem_id, work_dir=tmp_path)

    block_size = mem_freq + 1
    assert source.sequence_length == 12 // mem_freq * block_size  # 16
    assert len(source) == len(content) == 2

    inst = source[0]
    ids = list(inst["input_ids"])
    mask = list(inst["label_mask"])
    assert len(ids) == source.sequence_length == 16
    assert len(mask) == 16
    # landmark tokens at every block_size-th position (the last of each block)
    landmark_positions = [i for i in range(16) if (i % block_size) == (block_size - 1)]
    assert landmark_positions == [3, 7, 11, 15]
    for i in range(16):
        if i in landmark_positions:
            assert ids[i] == mem_id
            assert mask[i] is False
        else:
            assert ids[i] != mem_id
            assert mask[i] is True
    # content tokens preserved in order
    assert [t for t in ids if t != mem_id] == list(range(12))


def test_landmark_instance_source_requires_multiple_of_mem_freq(tmp_path):
    tokens = InMemoryTokenSource(tokens=list(range(20)), work_dir=tmp_path)
    content = ConcatAndChunkInstanceSource(tokens, sequence_length=10, work_dir=tmp_path)
    with pytest.raises(OLMoConfigurationError):
        LandmarkInstanceSource(content, mem_freq=3, mem_id=999, work_dir=tmp_path)


def test_landmark_factory_wiring():
    config = TransformerConfig.qwen3_4B(vocab_size=1000, landmark=True, mem_freq=63)
    mixer = config.block.sequence_mixer
    assert mixer.name == AttentionType.landmark
    assert mixer.mem_freq == 63

    # mem_freq without landmark is rejected.
    with pytest.raises(OLMoConfigurationError):
        TransformerConfig.llama_like(d_model=64, vocab_size=256, n_layers=1, n_heads=8, mem_freq=3)

    # A small landmark model actually builds.
    small = TransformerConfig.llama_like(
        d_model=64,
        vocab_size=256,
        n_layers=2,
        n_heads=8,
        n_kv_heads=2,
        qk_norm=True,
        landmark=True,
        mem_freq=3,
    )
    model = small.build()
    n_landmark = sum(1 for m in model.modules() if isinstance(m, LandmarkAttention))
    assert n_landmark == 2
