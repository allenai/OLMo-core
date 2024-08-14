import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed._tensor import init_device_mesh
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    parallelize_module,
)

from olmo_core.distributed.checkpoint import (
    async_save_model_and_optim_state,
    load_model_and_optim_state,
    save_model_and_optim_state,
)

from .utils import (
    BACKENDS,
    get_default_device,
    requires_multi_gpu,
    run_distributed_test,
)


def run_save_and_load_torch_fsdp_model(dir, model_factory, model_data_factory, use_orig_params):
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    fsdp_model = FSDP(model_factory().cuda(), use_orig_params=use_orig_params)
    optim = torch.optim.AdamW(fsdp_model.parameters())

    # Take a train step to initialize optimizer state.
    fsdp_model(model_data_factory().cuda()).sum().backward()
    optim.step()

    # Save checkpoint.
    save_model_and_optim_state(dir, fsdp_model, optim)
    dist.barrier()

    # Now create a new fsdp model and load that state.
    fsdp_model2 = FSDP(model_factory().cuda(), use_orig_params=use_orig_params)
    optim2 = torch.optim.AdamW(fsdp_model2.parameters())
    load_model_and_optim_state(dir, fsdp_model2, optim2)

    # Check model parameters.
    with FSDP.summon_full_params(fsdp_model, recurse=True), FSDP.summon_full_params(
        fsdp_model2, recurse=True
    ):
        torch.testing.assert_close(fsdp_model.state_dict(), fsdp_model2.state_dict())

    # Check optimizer state.
    for (p1_name, p1), (p2_name, p2) in zip(
        fsdp_model.named_parameters(), fsdp_model2.named_parameters()
    ):
        assert p1_name == p2_name
        torch.testing.assert_close(
            optim.state[p1],
            optim2.state[p2],
            msg=lambda m: f"State for '{p1_name}' does not match. {m}",
        )


@requires_multi_gpu
@pytest.mark.parametrize(
    "use_orig_params",
    (
        pytest.param(True, id="use_orig_params=True"),
        pytest.param(False, id="use_orig_params=False"),
    ),
)
def test_save_and_load_torch_fsdp_model(
    tmp_path,
    tiny_model_factory,
    tiny_model_data_factory,
    use_orig_params,
):
    run_distributed_test(
        run_save_and_load_torch_fsdp_model,
        backend="nccl",
        start_method="spawn",
        func_args=(
            tmp_path,
            tiny_model_factory,
            tiny_model_data_factory,
            use_orig_params,
        ),
    )


def run_save_and_load_tensor_parallel_model(dir, take_step_before_checkpoint, run_async):
    tp_mesh = init_device_mesh(get_default_device().type, (dist.get_world_size(),))

    class FeedForward(nn.Module):
        def __init__(self, dim: int = 16):
            super().__init__()
            self.dim = dim
            self.w1 = nn.Linear(dim, dim)
            self.w2 = nn.Linear(dim, dim)
            self.w3 = nn.Linear(dim, dim)

        def forward(self, x):
            return self.w2(F.silu(self.w1(x)) * self.w3(x))

    feed_forward = FeedForward().to(get_default_device())
    parallelize_module(
        feed_forward,
        tp_mesh,
        {
            # by default ColwiseParallel input layouts is replicated
            # and RowwiseParallel output layouts is replicated
            "w1": ColwiseParallel(),
            "w2": RowwiseParallel(),
            "w3": ColwiseParallel(),
        },
    )
    optim = torch.optim.AdamW(feed_forward.parameters())

    if take_step_before_checkpoint:
        # Take a forward and backward pass.
        feed_forward(
            torch.rand((2, feed_forward.dim), device=get_default_device())
        ).sum().backward()

        # Take an optimizer step.
        optim.step()
        optim.zero_grad(set_to_none=True)

    # Save checkpoint.
    if run_async:
        async_save_model_and_optim_state(dir, feed_forward, optim).result()
    else:
        save_model_and_optim_state(dir, feed_forward, optim)

    # Create another sharded model, load the checkpoint and make sure the state matches.
    feed_forward2 = FeedForward().to(get_default_device())
    parallelize_module(
        feed_forward2,
        tp_mesh,
        {
            # by default ColwiseParallel input layouts is replicated
            # and RowwiseParallel output layouts is replicated
            "w1": ColwiseParallel(),
            "w2": RowwiseParallel(),
            "w3": ColwiseParallel(),
        },
    )
    optim2 = torch.optim.AdamW(feed_forward2.parameters())
    load_model_and_optim_state(dir, feed_forward2, optim2)
    torch.testing.assert_close(feed_forward.state_dict(), feed_forward2.state_dict())
    torch.testing.assert_close(optim.state_dict(), optim2.state_dict())

    # Now load the checkpoint with a different topology, in this case an unsharded model.
    unsharded_feed_forward = FeedForward().to(get_default_device())
    unsharded_optim = torch.optim.AdamW(unsharded_feed_forward.parameters())
    load_model_and_optim_state(dir, unsharded_feed_forward, unsharded_optim)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    "take_step_before_checkpoint",
    [pytest.param(True, id="after-step"), pytest.param(False, id="pre-step")],
)
@pytest.mark.parametrize(
    "run_async",
    [pytest.param(True, id="async"), pytest.param(False, id="sync")],
)
def test_save_and_load_tensor_parallel_model(
    backend, tmp_path, take_step_before_checkpoint, run_async
):
    run_distributed_test(
        run_save_and_load_tensor_parallel_model,
        backend=backend,
        start_method="spawn",
        func_args=(tmp_path, take_step_before_checkpoint, run_async),
    )
