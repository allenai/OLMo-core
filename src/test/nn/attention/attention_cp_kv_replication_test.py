"""
Parity tests for Ulysses context parallelism when the CP degree exceeds ``n_kv_heads``.

These run on CPU/gloo with the ``torch`` attention backend, so they gate the KV-replication path
without needing GPUs. The gradient half is the important half: a wrong ``dK``/``dV`` reduction over
the replicas still produces a plausible forward pass and would otherwise train silently wrong.
"""

import copy
from functools import partial

import pytest
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

from olmo_core.nn.attention import Attention, AttentionBackendName
from olmo_core.nn.attention.ring import UlyssesContextParallelStyle
from olmo_core.testing import BACKENDS, run_distributed_test
from olmo_core.utils import get_default_device, seed_all


def _build_attention(d_model: int, n_heads: int, n_kv_heads: int, device: torch.device):
    seed_all(0)
    return Attention(
        d_model=d_model,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        bias=False,
        backend=AttentionBackendName.torch,
        init_device=device.type,
    )


def _test_ulysses_kv_replication_parity(n_heads: int, n_kv_heads: int):
    device = get_default_device()
    rank, world_size = dist.get_rank(), dist.get_world_size()
    mesh = init_device_mesh(device.type, (world_size,), mesh_dim_names=("cp",))

    B, T, head_dim = 2, 16, 8
    d_model = n_heads * head_dim

    # Two identical modules: one runs the full sequence with no CP as the reference, the other
    # runs this rank's shard under Ulysses CP.
    att_ref = _build_attention(d_model, n_heads, n_kv_heads, device)
    att_cp = copy.deepcopy(att_ref)
    att_cp.apply_cp(mesh["cp"], uly=UlyssesContextParallelStyle())

    seed_all(1234)
    x = torch.randn(B, T, d_model, device=device)

    # Reference: full sequence, no CP. Identical on every rank.
    out_ref = att_ref(x)
    out_ref.sum().backward()

    # CP: this rank owns a contiguous slice of the sequence.
    t_local = T // world_size
    x_local = x[:, rank * t_local : (rank + 1) * t_local, :]
    out_cp = att_cp(x_local)
    out_cp.sum().backward()

    torch.testing.assert_close(
        out_cp,
        out_ref[:, rank * t_local : (rank + 1) * t_local, :],
        msg=lambda m: f"rank {rank}: forward mismatch under CP\n{m}",
    )

    # Under CP each rank holds only a partial gradient; the real training step sums them across the
    # CP group (CP is flattened into the DP mesh for exactly this reason). Do that sum here, then
    # the result must match the single-process reference.
    #
    # Every rank must run the same collectives, so reduce everything first and only then assert --
    # otherwise a rank that fails early leaves its peers blocked in all_reduce and the test hangs
    # instead of failing.
    reduced = []
    for (name, p_ref), (_, p_cp) in zip(
        att_ref.named_parameters(), att_cp.named_parameters(), strict=True
    ):
        had_grad = p_cp.grad is not None
        grad_cp = p_cp.grad.clone() if p_cp.grad is not None else torch.zeros_like(p_cp)
        dist.all_reduce(grad_cp)
        reduced.append((name, p_ref.grad, grad_cp, had_grad))

    for name, grad_ref, grad_cp, had_grad in reduced:
        assert had_grad, f"rank {rank}: no gradient reached '{name}' under CP"
        assert grad_ref is not None, f"rank {rank}: reference gradient missing for '{name}'"
        torch.testing.assert_close(
            grad_cp,
            grad_ref,
            msg=lambda m, name=name: f"rank {rank}: gradient mismatch for '{name}' under CP\n{m}",
        )


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    "world_size, n_heads, n_kv_heads",
    [
        # cp > n_kv_heads: the replication path.
        pytest.param(2, 4, 1, id="cp2-mqa"),
        # The case that actually pins down the replicated head *order*: n_kv_heads > 1 and still
        # smaller than the CP degree, so a wrong layout pairs a query head with the wrong KV head.
        # This mirrors Qwen3.5-4B (n_heads=16, n_kv_heads=4) at cp=16.
        pytest.param(4, 8, 2, id="cp4-gqa-replicated"),
        # cp divides n_kv_heads: the pre-existing path, guarded against regression.
        pytest.param(2, 4, 2, id="cp2-gqa-no-replication"),
        pytest.param(2, 4, 4, id="cp2-mha-no-replication"),
    ],
)
def test_ulysses_kv_replication_parity(
    backend: str, world_size: int, n_heads: int, n_kv_heads: int
):
    run_distributed_test(
        partial(_test_ulysses_kv_replication_parity, n_heads=n_heads, n_kv_heads=n_kv_heads),
        backend=backend,
        start_method="spawn",
        world_size=world_size,
    )
