import pytest
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

from olmo_core.nn.attention import TorchAttentionBackend, UlyssesContextParallelStyle
from olmo_core.testing import requires_multi_gpu, run_distributed_test
from olmo_core.utils import get_default_device


@pytest.mark.parametrize("local_len", [1, 4])
@pytest.mark.parametrize("window_size", [(-1, -1), (0, 0), (1, 0), (7, 0)])
@pytest.mark.parametrize("n_kv_heads", [2, 4])
@pytest.mark.parametrize("head_rank", [0, 1])
def test_torch_ulysses_mask_after_exchange(
    monkeypatch, local_len, window_size, n_kv_heads, head_rank
):
    """Exercise real SDPA/masking and gradients, emulating only Ulysses transport."""
    torch.manual_seed(0)
    qkv = [
        torch.randn(1, 2 * local_len, heads, 4, requires_grad=True)
        for heads in (4, n_kv_heads, n_kv_heads)
    ]
    ref_qkv = [x.detach().clone().requires_grad_() for x in qkv]
    head_slice = slice(head_rank * 2, (head_rank + 1) * 2)
    kv_slice = slice(head_rank * (n_kv_heads // 2), (head_rank + 1) * (n_kv_heads // 2))

    # Supply the exact full-sequence/head-sharded results of the two exchanges.
    # Returning identity at the final exchange lets us compare this rank's heads
    # directly. Real collectives and their backward are covered by the GPU test below.
    monkeypatch.setattr(
        "olmo_core.nn.attention.backend.all_to_all_single_cp2hp",
        lambda q, pg: qkv[0][:, :, head_slice],
    )
    monkeypatch.setattr(
        "olmo_core.nn.attention.backend.all_to_all_cp2hp",
        lambda kv, pg: [x[:, :, kv_slice] for x in qkv[1:]],
    )
    monkeypatch.setattr(
        "olmo_core.nn.attention.backend.all_to_all_single_hp2cp", lambda out, pg: out
    )
    backend = TorchAttentionBackend(
        head_dim=4, n_heads=4, n_kv_heads=n_kv_heads, window_size=window_size
    )
    backend.cp_enabled = True
    backend.uly = UlyssesContextParallelStyle()
    monkeypatch.setattr(backend, "cp_pg", object())
    output = backend(tuple(x[:, :local_len] for x in qkv))
    reference = TorchAttentionBackend(
        head_dim=4, n_heads=4, n_kv_heads=n_kv_heads, window_size=window_size
    )(tuple(ref_qkv))[:, :, head_slice]
    torch.testing.assert_close(output, reference)
    weights = torch.randn_like(output)
    (output * weights).sum().backward()
    (reference * weights).sum().backward()
    for actual, expected in zip(qkv, ref_qkv):
        torch.testing.assert_close(actual.grad, expected.grad)


def _run_torch_ulysses_sliding_window(local_len, n_kv_heads):
    device = get_default_device()
    mesh = init_device_mesh(device.type, (2,), mesh_dim_names=("cp",))
    torch.manual_seed(0)
    full = [
        torch.randn(1, local_len * 2, heads, 4, device=device, requires_grad=True)
        for heads in (4, n_kv_heads, n_kv_heads)
    ]
    seq_slice = slice(dist.get_rank() * local_len, (dist.get_rank() + 1) * local_len)
    local = [x.detach()[:, seq_slice].clone().requires_grad_() for x in full]
    kwargs = dict(head_dim=4, n_heads=4, n_kv_heads=n_kv_heads, window_size=(1, 0))
    reference = TorchAttentionBackend(**kwargs)(tuple(full))
    backend = TorchAttentionBackend(**kwargs)
    backend.apply_cp(mesh, uly=UlyssesContextParallelStyle())
    output = backend(tuple(local))
    torch.testing.assert_close(output, reference[:, seq_slice])
    output.sum().backward()
    reference.sum().backward()
    for actual, expected in zip(local, full):
        assert expected.grad is not None
        torch.testing.assert_close(actual.grad, expected.grad[:, seq_slice])


@requires_multi_gpu
@pytest.mark.parametrize("local_len", [1, 4])
@pytest.mark.parametrize("n_kv_heads", [2, 4])
def test_torch_ulysses_sliding_window_distributed(local_len, n_kv_heads):
    run_distributed_test(
        _run_torch_ulysses_sliding_window,
        world_size=2,
        backend="nccl",
        start_method="spawn",
        func_args=(local_len, n_kv_heads),
    )
