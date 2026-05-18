from typing import Any, Dict

import pytest
import torch
from torch.distributed.tensor import Shard, init_device_mesh

from olmo_core.distributed.checkpoint import (
    load_model_and_optim_state,
    save_model_and_optim_state,
)
from olmo_core.distributed.utils import get_rank, get_world_size
from olmo_core.nn.feed_forward import ActivationFunction, FeedForward
from olmo_core.testing import BACKENDS, run_distributed_test
from olmo_core.utils import get_default_device, record_flops, seed_all


def _run_tensor_parallel_feed_forward(
    checkpoint_dir: str, inputs_path: str, outputs_path: str, ff_kwargs: Dict[str, Any]
):
    device = get_default_device()
    mesh = init_device_mesh(device.type, (get_world_size(),), mesh_dim_names=("tp",))

    ff = FeedForward(init_device=device.type, **ff_kwargs)

    ff.apply_tp(mesh["tp"], output_layout=Shard(1), use_local_output=False)
    load_model_and_optim_state(checkpoint_dir, ff)

    # Input x is replicated across ranks, output y is sharded on the sequence dimension.
    x = torch.load(inputs_path, map_location=device)
    y_local = ff(x).to_local()

    # Backward to exercise graph in TP mode.
    y_local.sum().backward()

    # Check the local shard of the output is the same as the corresponding shard of the reference output
    y_ref = torch.load(outputs_path, map_location=device)
    rank, world_size = get_rank(), get_world_size()
    chunk = x.size(1) // world_size
    y_ref_local = y_ref[:, rank * chunk : (rank + 1) * chunk, :]
    torch.testing.assert_close(y_ref_local, y_local)


@pytest.mark.parametrize("backend", BACKENDS)
def test_tensor_parallel_feed_forward(backend: str, tmp_path):
    device = torch.device("cuda") if "nccl" in backend else torch.device("cpu")

    seed_all(0)
    d_model = 128
    hidden = 4 * d_model
    ff_kwargs: Dict[str, Any] = {"d_model": d_model, "hidden_size": hidden}
    ff = FeedForward(init_device=device.type, **ff_kwargs)

    bs, seq_len = 2, 64
    x = torch.randn(bs, seq_len, d_model, device=device)
    y = ff(x)

    outputs_path = tmp_path / "ff_y.pt"
    torch.save(y, outputs_path)
    inputs_path = tmp_path / "ff_x.pt"
    torch.save(x, inputs_path)
    checkpoint_dir = tmp_path / "checkpoint"
    save_model_and_optim_state(checkpoint_dir, ff)

    run_distributed_test(
        _run_tensor_parallel_feed_forward,
        backend=backend,
        start_method="spawn",
        func_args=(checkpoint_dir, inputs_path, outputs_path, ff_kwargs),
    )


def test_feed_forward_num_flops_per_token():
    seed_all(0)

    d_model = 128
    hidden_size = 4 * d_model
    seq_len = 32
    batch_size = 1

    ff = FeedForward(d_model=d_model, hidden_size=hidden_size, init_device="cpu")

    x = torch.randn(batch_size, seq_len, d_model, requires_grad=True)

    actual_flops = record_flops(ff, x, with_backward=True)
    actual_flops_per_token = actual_flops // seq_len

    estimated_flops_per_token = ff.num_flops_per_token(seq_len)

    tolerance = 0.02
    relative_error = (
        abs(estimated_flops_per_token - actual_flops_per_token) / actual_flops_per_token
    )
    assert relative_error < tolerance, (
        f"Estimated FLOPs ({estimated_flops_per_token}) differs too much from actual ({actual_flops_per_token}), "
        f"{relative_error=:.2%}, {tolerance=:.2%}"
    )


@pytest.mark.parametrize("activation", [ActivationFunction.silu, ActivationFunction.gelu_tanh])
def test_feed_forward_activations(activation: ActivationFunction):
    seed_all(0)
    d_model = 128
    hidden_size = 4 * d_model
    batch_size = 2
    seq_len = 32

    ff = FeedForward(
        d_model=d_model, hidden_size=hidden_size, init_device="cpu", activation=activation
    )

    x = torch.randn(batch_size, seq_len, d_model)
    y = ff(x)

    assert y.shape == (batch_size, seq_len, d_model)
    assert not torch.isnan(y).any()
    assert not torch.isinf(y).any()

    y.sum().backward()
    assert ff.w1.weight.grad is not None
