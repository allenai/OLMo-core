from functools import partial

import pytest
import torch
import torch.distributed as dist

from olmo_core.distributed.parallel.context_parallel import (
    all_to_all_cp2hp,
    all_to_all_hp2cp,
    all_to_all_qkv_cp2hp,
    all_to_all_single_cp2hp,
    all_to_all_single_hp2cp,
)
from olmo_core.testing import BACKENDS, run_distributed_test
from olmo_core.utils import get_default_device


def _test_cp2hp_scatter(input_ndim: int = 4):
    """Test cp2hp: [B, T/CP, H, D] -> [B, T, H/CP, D] or [B, T/CP, H] -> [B, T, H/CP]"""
    device = get_default_device()
    dtype = torch.float32
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    group = dist.new_group()
    assert group is not None

    B, T, H, D = 2, 8, 4, 16
    t_local = T // world_size
    h_local = H // world_size

    expected_shape: tuple[int, ...]
    if input_ndim == 4:
        input_tensor = torch.full((B, t_local, H, D), float(rank), device=device, dtype=dtype)
        expected_shape = (B, T, h_local, D)
        expected_first_chunk = torch.full((B, t_local, h_local, D), 0.0, device=device, dtype=dtype)
        expected_second_chunk = torch.full(
            (B, t_local, h_local, D), 1.0, device=device, dtype=dtype
        )
    else:
        input_tensor = torch.full((B, t_local, H), float(rank), device=device, dtype=dtype)
        expected_shape = (B, T, h_local)
        expected_first_chunk = torch.full((B, t_local, h_local), 0.0, device=device, dtype=dtype)
        expected_second_chunk = torch.full((B, t_local, h_local), 1.0, device=device, dtype=dtype)

    output = all_to_all_single_cp2hp(input_tensor, group)

    # Output shape should match expected
    assert output.shape == expected_shape, f"Expected shape {expected_shape}, got {output.shape}"

    # Verify output values: each batch item should have full sequence [T] but only a chunk of H
    expected = torch.cat([expected_first_chunk, expected_second_chunk], dim=1)

    assert torch.allclose(output, expected), (
        f"Rank {rank}: Output values don't match expected. "
        f"Max diff = {(output - expected).abs().max()}, "
        f"Expected first {t_local} timesteps from rank 0, next {t_local} from rank 1"
    )


def _test_hp2cp_gather(input_ndim: int = 4):
    """Test hp2cp: [B, T, H/CP, D] -> [B, T/CP, H, D] or [B, T, H/CP] -> [B, T/CP, H]"""
    device = get_default_device()
    dtype = torch.float32
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    group = dist.new_group()
    assert group is not None

    B, T, H, D = 2, 8, 4, 16
    h_local = H // world_size
    t_local = T // world_size

    expected_shape: tuple[int, ...]
    if input_ndim == 4:
        input_tensor = torch.full((B, T, h_local, D), float(rank), device=device, dtype=dtype)
        expected_shape = (B, t_local, H, D)
        expected_h_chunk_0 = torch.full((B, t_local, h_local, D), 0.0, device=device, dtype=dtype)
        expected_h_chunk_1 = torch.full((B, t_local, h_local, D), 1.0, device=device, dtype=dtype)
    else:
        input_tensor = torch.full((B, T, h_local), float(rank), device=device, dtype=dtype)
        expected_shape = (B, t_local, H)
        expected_h_chunk_0 = torch.full((B, t_local, h_local), 0.0, device=device, dtype=dtype)
        expected_h_chunk_1 = torch.full((B, t_local, h_local), 1.0, device=device, dtype=dtype)

    output = all_to_all_single_hp2cp(input_tensor, group)

    # Output shape should match expected
    assert output.shape == expected_shape, f"Expected shape {expected_shape}, got {output.shape}"

    # Verify output values: each batch item should get a chunk of sequence [T/CP] but full H
    expected = torch.cat([expected_h_chunk_0, expected_h_chunk_1], dim=2)

    assert torch.allclose(output, expected), (
        f"Rank {rank}: Output values don't match expected. "
        f"Max diff = {(output - expected).abs().max()}, "
        f"Expected {t_local} timesteps with H values gathered from all ranks"
    )


def _test_roundtrip(input_ndim: int = 4):
    """Test that cp2hp -> hp2cp roundtrips correctly."""
    device = get_default_device()
    dtype = torch.float32
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    group = dist.new_group()
    assert group is not None

    B, T, H, D = 2, 8, 4, 16
    t_local = T // world_size

    # Start with CP layout: each rank has [B, T/CP, H, D] or [B, T/CP, H]
    if input_ndim == 4:
        original = torch.randn(B, t_local, H, D, device=device, dtype=dtype)
    else:
        original = torch.randn(B, t_local, H, device=device, dtype=dtype)

    # CP -> HP -> CP should give back the original
    hp = all_to_all_single_cp2hp(original, group)
    recovered = all_to_all_single_hp2cp(hp, group)

    assert (
        original.shape == recovered.shape
    ), f"Rank {rank}: shape mismatch, original {original.shape} vs recovered {recovered.shape}"
    assert torch.allclose(
        original, recovered
    ), f"Rank {rank}: roundtrip failed, max diff = {(original - recovered).abs().max()}"


def _test_batched_cp2hp(input_ndim: int = 4):
    """Test batched cp2hp: list of [B, T/CP, H, D] -> list of [B, T, H/CP, D]"""
    device = get_default_device()
    dtype = torch.float32
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    group = dist.new_group()
    assert group is not None

    B, T, H, D = 2, 8, 4, 16
    t_local = T // world_size
    h_local = H // world_size

    # Create multiple input tensors with different values
    expected_shape: tuple[int, ...]
    if input_ndim == 4:
        input1 = torch.full((B, t_local, H, D), float(rank), device=device, dtype=dtype)
        input2 = torch.full((B, t_local, H, D), float(rank) + 10.0, device=device, dtype=dtype)
        expected_shape = (B, T, h_local, D)
        expected1_first = torch.full((B, t_local, h_local, D), 0.0, device=device, dtype=dtype)
        expected1_second = torch.full((B, t_local, h_local, D), 1.0, device=device, dtype=dtype)
        expected2_first = torch.full((B, t_local, h_local, D), 10.0, device=device, dtype=dtype)
        expected2_second = torch.full((B, t_local, h_local, D), 11.0, device=device, dtype=dtype)
    else:
        input1 = torch.full((B, t_local, H), float(rank), device=device, dtype=dtype)
        input2 = torch.full((B, t_local, H), float(rank) + 10.0, device=device, dtype=dtype)
        expected_shape = (B, T, h_local)
        expected1_first = torch.full((B, t_local, h_local), 0.0, device=device, dtype=dtype)
        expected1_second = torch.full((B, t_local, h_local), 1.0, device=device, dtype=dtype)
        expected2_first = torch.full((B, t_local, h_local), 10.0, device=device, dtype=dtype)
        expected2_second = torch.full((B, t_local, h_local), 11.0, device=device, dtype=dtype)

    outputs = all_to_all_cp2hp([input1, input2], group)

    assert len(outputs) == 2

    # Both outputs should have expected shape
    for output in outputs:
        assert (
            output.shape == expected_shape
        ), f"Expected shape {expected_shape}, got {output.shape}"

    # Verify output values match what single version would produce
    expected1 = torch.cat([expected1_first, expected1_second], dim=1)
    expected2 = torch.cat([expected2_first, expected2_second], dim=1)

    assert torch.allclose(outputs[0], expected1), (
        f"Rank {rank}: Output 0 values don't match expected. "
        f"Max diff = {(outputs[0] - expected1).abs().max()}"
    )
    assert torch.allclose(outputs[1], expected2), (
        f"Rank {rank}: Output 1 values don't match expected. "
        f"Max diff = {(outputs[1] - expected2).abs().max()}"
    )


def _test_batched_matches_single(input_ndim: int = 4):
    """Test that batched version produces same results as single version."""
    device = get_default_device()
    dtype = torch.float32
    rank = dist.get_rank()
    group = dist.new_group()
    assert group is not None

    B, T, H, D = 2, 8, 4, 16
    world_size = dist.get_world_size()
    t_local = T // world_size

    # Create input tensors
    if input_ndim == 4:
        input1 = torch.randn(B, t_local, H, D, device=device, dtype=dtype)
        input2 = torch.randn(B, t_local, H, D, device=device, dtype=dtype)
    else:
        input1 = torch.randn(B, t_local, H, device=device, dtype=dtype)
        input2 = torch.randn(B, t_local, H, device=device, dtype=dtype)

    # Get results from batched version
    batched_outputs = all_to_all_cp2hp([input1.clone(), input2.clone()], group)

    # Get results from single version (need new group for each call due to collective semantics)
    group2 = dist.new_group()
    single_output1 = all_to_all_single_cp2hp(input1.clone(), group2)
    group3 = dist.new_group()
    single_output2 = all_to_all_single_cp2hp(input2.clone(), group3)

    assert torch.allclose(batched_outputs[0], single_output1), (
        f"Rank {rank}: Batched output 0 doesn't match single output. "
        f"Max diff = {(batched_outputs[0] - single_output1).abs().max()}"
    )
    assert torch.allclose(batched_outputs[1], single_output2), (
        f"Rank {rank}: Batched output 1 doesn't match single output. "
        f"Max diff = {(batched_outputs[1] - single_output2).abs().max()}"
    )


def _test_batched_hp2cp(input_ndim: int = 4):
    """Test batched hp2cp: list of [B, T, H/CP, D] -> list of [B, T/CP, H, D]"""
    device = get_default_device()
    dtype = torch.float32
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    group = dist.new_group()
    assert group is not None

    B, T, H, D = 2, 8, 4, 16
    t_local = T // world_size
    h_local = H // world_size

    # Create multiple input tensors with different values
    expected_shape: tuple[int, ...]
    if input_ndim == 4:
        input1 = torch.full((B, T, h_local, D), float(rank), device=device, dtype=dtype)
        input2 = torch.full((B, T, h_local, D), float(rank) + 10.0, device=device, dtype=dtype)
        expected_shape = (B, t_local, H, D)
        expected1_h0 = torch.full((B, t_local, h_local, D), 0.0, device=device, dtype=dtype)
        expected1_h1 = torch.full((B, t_local, h_local, D), 1.0, device=device, dtype=dtype)
        expected2_h0 = torch.full((B, t_local, h_local, D), 10.0, device=device, dtype=dtype)
        expected2_h1 = torch.full((B, t_local, h_local, D), 11.0, device=device, dtype=dtype)
    else:
        input1 = torch.full((B, T, h_local), float(rank), device=device, dtype=dtype)
        input2 = torch.full((B, T, h_local), float(rank) + 10.0, device=device, dtype=dtype)
        expected_shape = (B, t_local, H)
        expected1_h0 = torch.full((B, t_local, h_local), 0.0, device=device, dtype=dtype)
        expected1_h1 = torch.full((B, t_local, h_local), 1.0, device=device, dtype=dtype)
        expected2_h0 = torch.full((B, t_local, h_local), 10.0, device=device, dtype=dtype)
        expected2_h1 = torch.full((B, t_local, h_local), 11.0, device=device, dtype=dtype)

    outputs = all_to_all_hp2cp([input1, input2], group)

    assert len(outputs) == 2

    # Both outputs should have expected shape
    for output in outputs:
        assert (
            output.shape == expected_shape
        ), f"Expected shape {expected_shape}, got {output.shape}"

    # Verify output values: H chunks gathered from all ranks
    expected1 = torch.cat([expected1_h0, expected1_h1], dim=2)
    expected2 = torch.cat([expected2_h0, expected2_h1], dim=2)

    assert torch.allclose(outputs[0], expected1), (
        f"Rank {rank}: Output 0 values don't match expected. "
        f"Max diff = {(outputs[0] - expected1).abs().max()}"
    )
    assert torch.allclose(outputs[1], expected2), (
        f"Rank {rank}: Output 1 values don't match expected. "
        f"Max diff = {(outputs[1] - expected2).abs().max()}"
    )


def _test_qkv_cp2hp_replicates_kv_heads(n_heads: int, n_kv_heads: int):
    """
    ``all_to_all_qkv_cp2hp`` must hand each rank the KV heads its query heads actually attend to,
    replicating them when the CP degree doesn't divide ``n_kv_heads``.
    """
    device = get_default_device()
    dtype = torch.float32
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    group = dist.new_group()
    assert group is not None

    B, T, D = 2, 8, 16
    t_local = T // world_size
    n_rep = n_heads // n_kv_heads

    # Fill each head with its own global head index so we can check the pairing by value. The
    # tensors are identical on every rank, so a value tells us which head it came from.
    q = (
        torch.arange(n_heads, device=device, dtype=dtype)
        .view(1, 1, n_heads, 1)
        .expand(B, t_local, n_heads, D)
        .contiguous()
    )
    kv = (
        torch.arange(n_kv_heads, device=device, dtype=dtype)
        .view(1, 1, n_kv_heads, 1)
        .expand(B, t_local, n_kv_heads, D)
        .contiguous()
    )

    q_out, k_out, v_out = all_to_all_qkv_cp2hp(q, kv.clone(), kv.clone(), group)

    h_local = n_heads // world_size
    assert q_out.shape == (B, T, h_local, D), f"got q shape {q_out.shape}"
    if n_kv_heads % world_size == 0:
        # No replication needed: the plain split still applies.
        assert k_out.shape == (B, T, n_kv_heads // world_size, D), f"got k shape {k_out.shape}"
        return

    assert k_out.shape == (B, T, h_local, D), f"got k shape {k_out.shape}"
    assert v_out.shape == (B, T, h_local, D), f"got v shape {v_out.shape}"

    # Rank r holds query heads [r*h_local, (r+1)*h_local); each must be paired with KV head
    # ``head // n_rep``, which is the GQA grouping the non-CP path uses.
    expected_q = (
        torch.arange(rank * h_local, (rank + 1) * h_local, device=device, dtype=dtype)
        .view(1, 1, h_local, 1)
        .expand(B, T, h_local, D)
    )
    torch.testing.assert_close(q_out, expected_q)
    torch.testing.assert_close(k_out, torch.div(expected_q, n_rep, rounding_mode="floor"))
    torch.testing.assert_close(v_out, torch.div(expected_q, n_rep, rounding_mode="floor"))


def _test_qkv_cp2hp_kv_grads_sum_over_replicas(n_heads: int, n_kv_heads: int):
    """
    The whole point of replicating with ``expand``: autograd must sum ``dK``/``dV`` back over the
    replicas, so a KV head used by ``n_rep`` query heads receives ``n_rep`` times the gradient.
    """
    device = get_default_device()
    world_size = dist.get_world_size()
    group = dist.new_group()
    assert group is not None

    B, T, D = 1, 8, 4
    t_local = T // world_size
    n_rep = n_heads // n_kv_heads

    q = torch.zeros(B, t_local, n_heads, D, device=device, requires_grad=True)
    k = torch.zeros(B, t_local, n_kv_heads, D, device=device, requires_grad=True)
    v = torch.zeros(B, t_local, n_kv_heads, D, device=device, requires_grad=True)

    q_out, k_out, v_out = all_to_all_qkv_cp2hp(q, k, v, group)
    (q_out.sum() + k_out.sum() + v_out.sum()).backward()

    assert q.grad is not None and k.grad is not None and v.grad is not None
    torch.testing.assert_close(q.grad, torch.ones_like(q))
    # Each KV head was replicated n_rep times, so its gradient is n_rep, not 1.
    expected_kv_grad = float(n_rep) if n_kv_heads % world_size != 0 else 1.0
    torch.testing.assert_close(k.grad, torch.full_like(k, expected_kv_grad))
    torch.testing.assert_close(v.grad, torch.full_like(v, expected_kv_grad))


_KV_REPLICATION_CASES = [
    pytest.param(2, 4, 1, id="cp2-mqa"),
    # n_kv_heads > 1 and smaller than the CP degree: the only shape that pins down the replicated
    # head *order*. Mirrors Qwen3.5-4B (n_heads=16, n_kv_heads=4) at cp=16.
    pytest.param(4, 8, 2, id="cp4-gqa-replicated"),
    pytest.param(2, 4, 2, id="cp2-gqa-no-replication"),
    pytest.param(2, 4, 4, id="cp2-mha-no-replication"),
]


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("world_size, n_heads, n_kv_heads", _KV_REPLICATION_CASES)
def test_qkv_cp2hp_replicates_kv_heads(
    backend: str, world_size: int, n_heads: int, n_kv_heads: int
):
    run_distributed_test(
        partial(_test_qkv_cp2hp_replicates_kv_heads, n_heads=n_heads, n_kv_heads=n_kv_heads),
        backend=backend,
        start_method="spawn",
        world_size=world_size,
    )


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("world_size, n_heads, n_kv_heads", _KV_REPLICATION_CASES)
def test_qkv_cp2hp_kv_grads_sum_over_replicas(
    backend: str, world_size: int, n_heads: int, n_kv_heads: int
):
    run_distributed_test(
        partial(_test_qkv_cp2hp_kv_grads_sum_over_replicas, n_heads=n_heads, n_kv_heads=n_kv_heads),
        backend=backend,
        start_method="spawn",
        world_size=world_size,
    )


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("input_ndim", [3, 4])
def test_cp2hp_scatter(backend: str, input_ndim: int):
    run_distributed_test(
        partial(_test_cp2hp_scatter, input_ndim=input_ndim),
        backend=backend,
        start_method="spawn",
        world_size=2,
    )


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("input_ndim", [3, 4])
def test_hp2cp_gather(backend: str, input_ndim: int):
    run_distributed_test(
        partial(_test_hp2cp_gather, input_ndim=input_ndim),
        backend=backend,
        start_method="spawn",
        world_size=2,
    )


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("input_ndim", [3, 4])
def test_roundtrip(backend: str, input_ndim: int):
    run_distributed_test(
        partial(_test_roundtrip, input_ndim=input_ndim),
        backend=backend,
        start_method="spawn",
        world_size=2,
    )


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("input_ndim", [3, 4])
def test_batched_cp2hp(backend: str, input_ndim: int):
    run_distributed_test(
        partial(_test_batched_cp2hp, input_ndim=input_ndim),
        backend=backend,
        start_method="spawn",
        world_size=2,
    )


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("input_ndim", [3, 4])
def test_batched_hp2cp(backend: str, input_ndim: int):
    run_distributed_test(
        partial(_test_batched_hp2cp, input_ndim=input_ndim),
        backend=backend,
        start_method="spawn",
        world_size=2,
    )


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("input_ndim", [3, 4])
def test_batched_matches_single(backend: str, input_ndim: int):
    run_distributed_test(
        partial(_test_batched_matches_single, input_ndim=input_ndim),
        backend=backend,
        start_method="spawn",
        world_size=2,
    )
