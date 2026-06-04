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
from olmo_core.nn.attention.ring import UlyssesContextParallelStyle
from olmo_core.nn.layer_norm import LayerNormConfig
from olmo_core.nn.rope import RoPEConfig, RoPEType
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.testing import requires_gpu, run_distributed_test


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


def test_landmark_instance_source_exclude_landmark_predictors(tmp_path):
    from olmo_core.data.utils import get_labels

    mem_freq, mem_id = 3, 999  # block_size 4
    block_size = mem_freq + 1
    tokens = InMemoryTokenSource(tokens=list(range(100, 124)), work_dir=tmp_path)
    content = ConcatAndChunkInstanceSource(tokens, sequence_length=12, work_dir=tmp_path)

    default = LandmarkInstanceSource(content, mem_freq=mem_freq, mem_id=mem_id, work_dir=tmp_path)
    excluded = LandmarkInstanceSource(
        content,
        mem_freq=mem_freq,
        mem_id=mem_id,
        exclude_landmark_predictors=True,
        work_dir=tmp_path,
    )

    # Changing the option must change the fingerprint (so cached artifacts aren't reused).
    assert default.fingerprint != excluded.fingerprint

    # input_ids are identical; only label_mask differs.
    d_inst, e_inst = default[0], excluded[0]
    assert list(d_inst["input_ids"]) == list(e_inst["input_ids"])

    landmark_positions = [i for i in range(16) if (i % block_size) == (block_size - 1)]  # 3,7,11,15

    def loss_positions(inst):
        batch = {
            "input_ids": torch.tensor([list(map(int, inst["input_ids"]))]),
            "label_mask": torch.tensor([list(map(bool, inst["label_mask"]))]),
        }
        labels = get_labels(batch, label_ignore_index=-100)[0]
        return {i for i in range(labels.numel()) if labels[i].item() != -100}

    default_loss = loss_positions(d_inst)
    excluded_loss = loss_positions(e_inst)

    # By default, interior landmark positions contribute as predictors.
    assert default_loss & set(landmark_positions) == {3, 7, 11}  # 15 is the final position
    # With the option on, no landmark position contributes, and that's the only difference.
    assert excluded_loss & set(landmark_positions) == set()
    assert default_loss - excluded_loss == {3, 7, 11}


def _landmark_transformer_config(seq_len: int) -> TransformerConfig:
    # Small landmark transformer (eager path) used by the context-parallel test. n_heads and
    # n_kv_heads must be divisible by the CP degree (world_size=2).
    return TransformerConfig.llama_like(
        d_model=64,
        vocab_size=256,
        n_layers=2,
        n_heads=8,
        n_kv_heads=2,
        qk_norm=True,
        rope_theta=10_000,
        landmark=True,
        mem_freq=3,  # block_size 4
    )


def _run_landmark_ulysses_cp(
    checkpoint_dir: str, inputs_path: str, outputs_path: str, seq_len: int
):
    from torch.distributed.tensor import DTensor, Shard, init_device_mesh

    from olmo_core.distributed.checkpoint import load_model_and_optim_state
    from olmo_core.distributed.utils import get_full_tensor, get_world_size

    mesh = init_device_mesh("cpu", (get_world_size(),), mesh_dim_names=("cp",))

    model = _landmark_transformer_config(seq_len).build()
    model.apply_cp(mesh["cp"], uly=UlyssesContextParallelStyle())
    model.init_weights(device=torch.device("cpu"), max_seq_len=seq_len)
    load_model_and_optim_state(checkpoint_dir, model)
    model.eval()

    # The model shards input_ids and the RoPE buffers internally via the Ulysses load balancer, so
    # we pass the full (replicated) input; each rank returns its sequence shard of the logits.
    input_ids = torch.load(inputs_path, map_location="cpu")
    with torch.no_grad():
        local_logits = model(input_ids=input_ids)
    logits = DTensor.from_local(local_logits, mesh, (Shard(1),))

    expected = torch.load(outputs_path, map_location="cpu")
    torch.testing.assert_close(get_full_tensor(logits), expected, rtol=1e-4, atol=1e-4)


def test_landmark_ulysses_cp_matches_full(tmp_path):
    # Ulysses CP must reproduce a single-rank full-sequence forward: LandmarkAttention.forward
    # gathers the complete sequence (with n_heads / cp_degree heads) via all-to-all, so the grouped
    # softmax still sees every preceding block's landmark. The model itself shards the input and the
    # RoPE buffers, so this also covers RoPE under CP.
    from olmo_core.distributed.checkpoint import save_model_and_optim_state

    torch.manual_seed(0)
    seq_len = 16  # 4 landmark blocks; divisible by block_size (4) and world_size (2)

    model = _landmark_transformer_config(seq_len).build()
    model.init_weights(device=torch.device("cpu"), max_seq_len=seq_len)
    model.eval()

    input_ids = torch.randint(0, 256, (2, seq_len))
    with torch.no_grad():
        expected = model(input_ids=input_ids)

    inputs_path = tmp_path / "x.pt"
    outputs_path = tmp_path / "y.pt"
    checkpoint_dir = tmp_path / "checkpoint"
    torch.save(input_ids, inputs_path)
    torch.save(expected, outputs_path)
    save_model_and_optim_state(checkpoint_dir, model)

    run_distributed_test(
        _run_landmark_ulysses_cp,
        backend="gloo",
        world_size=2,
        func_args=(str(checkpoint_dir), str(inputs_path), str(outputs_path), seq_len),
    )


def test_landmark_rejects_ring_cp():
    from torch.distributed.device_mesh import DeviceMesh

    from olmo_core.nn.attention.ring import (
        RingAttentionZigZagLoadBalancer,
        RingContextParallelStyle,
    )

    attn = _landmark_attention(mem_freq=3)
    # An empty DeviceMesh shell is enough to exercise the guard, which fires before any collective
    # is created.
    fake_mesh = DeviceMesh.__new__(DeviceMesh)
    with pytest.raises(OLMoConfigurationError, match="only supports Ulysses"):
        attn.apply_cp(
            fake_mesh,
            ring=RingContextParallelStyle(load_balancer=RingAttentionZigZagLoadBalancer),
        )


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
