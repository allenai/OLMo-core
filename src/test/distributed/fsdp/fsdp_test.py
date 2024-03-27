import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from olmo_core.distributed.fsdp import FSDP, FSDPDebugConfig
from olmo_core.distributed.sharded_flat_parameter import ShardedFlatParameter

from ..utils import (
    BACKENDS,
    FSDP_MIXED_PRECISION,
    get_default_device,
    run_distributed_test,
)


def run_fsdp_against_non_distributed_model(model_factory, model_data_factory):
    """
    Compare outputs from forward pass and gradients to those from a non-distributed model.
    """
    model_data = model_data_factory().to(get_default_device())

    model = model_factory()
    fsdp1 = FSDP(model_factory(), _debug_config=FSDPDebugConfig(no_reduce_grads=True))
    fsdp2 = FSDP(model_factory())

    # Ensure params for all models on all ranks match.
    for param in model.parameters():
        with torch.no_grad():
            dist.broadcast(param.data, 0)

    for fsdp in (fsdp1, fsdp2):
        with fsdp.summon_full_params():
            fsdp.module.load_state_dict(model.state_dict())

        for name, param in fsdp.module.named_parameters():
            assert isinstance(param, ShardedFlatParameter)
            assert param.is_sharded
            assert param.grad is None
            with torch.no_grad():
                torch.testing.assert_close(param.data, param.sharded_chunk(model.state_dict()[name]))

    # Run forward/backward pass on non-distributed model and collect grads for comparison.
    expected_grads = {}
    loss = model(model_data).sum()
    loss.backward()
    for param_name, param in model.named_parameters():
        expected_grads[param_name] = param.grad.detach()

    # Run forward pass on FSDP models.
    fsdp1_loss = fsdp1(model_data).sum()
    fsdp2_loss = fsdp2(model_data).sum()

    with torch.no_grad():
        torch.testing.assert_close(loss, fsdp1_loss)
        torch.testing.assert_close(loss, fsdp2_loss)

    # Models should be in a sharded state again.
    for fsdp in (fsdp1, fsdp2):
        for param in fsdp.parameters():
            assert isinstance(param, ShardedFlatParameter)
            assert param.is_sharded

    # Trigger backward pass on FSDP model.
    fsdp1_loss.backward()
    fsdp2_loss.backward()

    # Check gradients and ensure model is in a sharded state again.
    for name, param in fsdp1.module.named_parameters():
        assert isinstance(param, ShardedFlatParameter)
        assert param.is_sharded
        assert param.grad is not None
        with torch.no_grad():
            torch.testing.assert_close(
                param.grad, expected_grads[name], msg=lambda m: f"On gradient for '{name}'. {m}"
            )

    # Now manually reduce grads for the 1st FSDP model to compare to the 2nd FSDP model.
    for (name, param1), param2 in zip(fsdp1.module.named_parameters(), fsdp2.module.parameters()):
        with torch.no_grad():
            dist.all_reduce(param1.grad, group=fsdp1.process_group)
            torch.testing.assert_close(
                param2.sharded_chunk(param1.grad), param2.grad, msg=lambda m: f"On gradient for '{name}'. {m}"
            )


@pytest.mark.parametrize("backend", BACKENDS)
def test_fsdp_against_non_distributed_model(backend, tiny_model_factory, tiny_model_data_factory):
    run_distributed_test(
        run_fsdp_against_non_distributed_model,
        backend=backend,
        func_args=(tiny_model_factory, tiny_model_data_factory),
        start_method="spawn",
    )


def run_fsdp_against_ddp(model_factory, model_data_factory):
    """
    Compare outputs from forward pass and gradients against those from a DDP model.
    """
    model_data = model_data_factory().to(get_default_device())

    ddp_model = DDP(model_factory().to(get_default_device()))
    fsdp_model = FSDP(model_factory())

    with fsdp_model.summon_full_params():
        fsdp_model.module.load_state_dict(ddp_model.module.state_dict())

    for name, param in fsdp_model.module.named_parameters():
        assert isinstance(param, ShardedFlatParameter)
        assert param.is_sharded
        assert param.grad is None
        with torch.no_grad():
            torch.testing.assert_close(param.data, param.sharded_chunk(ddp_model.module.state_dict()[name]))

    # Run forward/backward pass on DDP model and collect grads for comparison.
    ddp_loss = ddp_model(model_data).sum()
    ddp_loss.backward()
    expected_grads = {}
    for param_name, param in ddp_model.module.named_parameters():
        expected_grads[param_name] = param.grad.detach()

    optim = torch.optim.AdamW(fsdp_model.parameters())

    # Run forward pass on FSDP model.
    fsdp_loss = fsdp_model(model_data).sum()

    with torch.no_grad():
        torch.testing.assert_close(ddp_loss, fsdp_loss)

    # Model should be in a sharded state again.
    for param in fsdp_model.parameters():
        assert isinstance(param, ShardedFlatParameter)
        assert param.is_sharded

    # Trigger backward pass on FSDP model.
    fsdp_loss.backward()

    # Model should be in a sharded state again.
    for name, param in fsdp_model.module.named_parameters():
        assert isinstance(param, ShardedFlatParameter)
        assert param.is_sharded
        assert param.grad is not None
        with torch.no_grad():
            # NOTE: DDP *averages* gradients over ranks, FSDP just takes the sum.
            torch.testing.assert_close(
                param.grad,
                param.sharded_chunk(expected_grads[name] * dist.get_world_size()),
                msg=lambda m: f"On gradient for '{name}'. {m}",
            )

    # Since we've only done a single backwards pass (no grad accumulation), there shouldn't
    # be any cached gradients.
    assert not fsdp_model.state.sharded_grad_cache

    # Run optimizer step.
    optim.step()


@pytest.mark.parametrize("backend", BACKENDS)
def test_fsdp_against_ddp(backend, tiny_model_factory, tiny_model_data_factory):
    run_distributed_test(
        run_fsdp_against_ddp,
        backend=backend,
        func_args=(tiny_model_factory, tiny_model_data_factory),
        start_method="spawn",
    )


def run_fsdp_with_gradient_accumulation(model_factory, model_data_factory):
    """
    Compare outputs from forward pass and gradients to those from a non-distributed model with
    gradient accumulation.
    """
    model_data1 = model_data_factory().to(get_default_device())
    model_data2 = model_data_factory().to(get_default_device())

    ddp = DDP(model_factory().to(get_default_device()))
    fsdp = FSDP(model_factory())

    # Ensure params for all models on all ranks match.
    with fsdp.summon_full_params():
        fsdp.module.load_state_dict(ddp.module.state_dict())

    for name, param in fsdp.module.named_parameters():
        assert isinstance(param, ShardedFlatParameter)
        assert param.is_sharded
        assert param.grad is None
        with torch.no_grad():
            torch.testing.assert_close(param.data, param.sharded_chunk(ddp.module.state_dict()[name]))

    # Run forward/backward pass on non-distributed model and collect grads for comparison.
    ddp(model_data1).sum().backward()
    ddp(model_data2).sum().backward()

    expected_grads = {}
    for param_name, param in ddp.module.named_parameters():
        expected_grads[param_name] = param.grad.detach()

    # Run forward/backward pass on FSDP model.
    fsdp(model_data1).sum().backward()
    fsdp(model_data2).sum().backward()

    # Check gradients and ensure model is in a sharded state again.
    for name, param in fsdp.module.named_parameters():
        assert isinstance(param, ShardedFlatParameter)
        assert param.is_sharded
        assert param.grad is not None
        with torch.no_grad():
            torch.testing.assert_close(
                param.grad,
                param.sharded_chunk(expected_grads[name] * dist.get_world_size()),
                msg=lambda m: f"On gradient for '{name}'. {m}",
            )


@pytest.mark.parametrize("backend", BACKENDS)
def test_fsdp_with_gradient_accumulation(backend, tiny_model_factory, tiny_model_data_factory):
    run_distributed_test(
        run_fsdp_with_gradient_accumulation,
        backend=backend,
        func_args=(tiny_model_factory, tiny_model_data_factory),
        start_method="spawn",
    )


def run_nested_fsdp_api(model_factory, model_data_factory):
    class NestedModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.inner = FSDP(model_factory())
            self.out = nn.Linear(8, 8)

        def forward(self, x):
            x = self.inner(x)
            return self.out(x)

    fsdp = FSDP(NestedModel())
    assert list(fsdp._fsdp_children()) == [fsdp.module.inner]
    assert list(fsdp._managed_named_parameters()) == [
        ("out.weight", fsdp.module.out.weight),
        ("out.bias", fsdp.module.out.bias),
    ]
    assert fsdp.is_root
    assert not fsdp.module.inner.is_root

    inner_weight = fsdp.module.inner.module.fc[0].weight
    assert isinstance(inner_weight, ShardedFlatParameter)
    outer_weight = fsdp.module.out.weight
    assert isinstance(outer_weight, ShardedFlatParameter)

    assert inner_weight.unsharded_shape == (16, 8)

    with fsdp.summon_full_params(recurse=False, writeback=False):
        # Only directly managed params should be sharded.
        assert inner_weight.is_sharded
        assert not outer_weight.is_sharded
        assert outer_weight.shape == outer_weight.unsharded_shape == (8, 8)

        # State dict should never contain "_fsdp_wrapped_module" prefix.
        state_dict = fsdp.state_dict()
        assert set(state_dict.keys()) == {
            "out.weight",
            "out.bias",
            "inner.fc.0.weight",
            "inner.fc.0.bias",
            "inner.fc.2.weight",
            "inner.fc.2.bias",
            "inner.fc.4.weight",
            "inner.fc.4.bias",
        }
        fsdp.load_state_dict(state_dict)

    with fsdp.summon_full_params(recurse=True, writeback=False):
        # All FSDP-managed params should be sharded, including those owned by child FSDP instances.
        assert not inner_weight.is_sharded
        assert inner_weight.shape == inner_weight.unsharded_shape == (16, 8)
        assert not outer_weight.is_sharded
        assert outer_weight.shape == outer_weight.unsharded_shape == (8, 8)

        # State dict should never contain "_fsdp_wrapped_module" prefix.
        state_dict = fsdp.state_dict()
        assert set(state_dict.keys()) == {
            "out.weight",
            "out.bias",
            "inner.fc.0.weight",
            "inner.fc.0.bias",
            "inner.fc.2.weight",
            "inner.fc.2.bias",
            "inner.fc.4.weight",
            "inner.fc.4.bias",
        }
        fsdp.load_state_dict(state_dict)

    with fsdp.summon_full_params(recurse=True, writeback=True, rank0_only=True):
        if dist.get_rank() == 0:
            with torch.no_grad():
                outer_weight.fill_(torch.tensor(1.0, device=fsdp.device))
                inner_weight.fill_(torch.tensor(1.0, device=fsdp.device))

    assert (outer_weight == 1.0).all()
    assert (inner_weight == 1.0).all()

    # Now complete a forward pass and make sure lazy initialization did its job.
    loss = fsdp(model_data_factory().to(fsdp.device)).sum()
    assert fsdp.state.forward_execution_order_finalized
    assert fsdp.state.forward_execution_order == [fsdp, fsdp.module.inner]
    assert fsdp.state.forward_execution_order is fsdp.module.inner.state.forward_execution_order
    assert fsdp.state.forward_prefetch_queue is fsdp.module.inner.state.forward_prefetch_queue

    assert not fsdp.state.backward_execution_order_finalized
    assert fsdp.state.backward_execution_order is fsdp.module.inner.state.backward_execution_order
    assert fsdp.state.backward_prefetch_queue is fsdp.module.inner.state.backward_prefetch_queue

    # Trigger backward pass.
    loss.backward()

    assert fsdp.state.backward_execution_order_finalized
    assert fsdp.state.backward_execution_order == [fsdp, fsdp.module.inner]

    # Let's do a 2nd forward pass now that the execution order is finalized.
    loss = fsdp(model_data_factory().to(fsdp.device)).sum()
    loss.backward()


@pytest.mark.parametrize("backend", BACKENDS)
def test_nested_fsdp_api(backend, tiny_model_factory, tiny_model_data_factory):
    run_distributed_test(
        run_nested_fsdp_api,
        backend=backend,
        func_args=(tiny_model_factory, tiny_model_data_factory),
        start_method="spawn",
    )


def run_fsdp_with_mixed_precision(model_factory, model_data_factory, precision):
    fsdp = FSDP(model_factory(), precision=precision)

    model_data = model_data_factory().to(fsdp.device)
    if precision.param_dtype is not None:
        model_data = model_data.to(dtype=precision.param_dtype)

    # Forward pass.
    loss = fsdp(model_data).sum()

    # Loss dtype should match precision.param_dtype (the datatype we gather full params in).
    if precision.param_dtype is not None:
        assert loss.dtype == precision.param_dtype

    # Trigger backward pass.
    loss.backward()

    # Make sure grads are now in the correct type.
    for param in fsdp.parameters():
        assert param.grad is not None
        if precision.keep_low_precision_grads and precision.param_dtype is not None:
            assert param.grad.dtype == precision.param_dtype
        else:
            assert param.grad.dtype == param.dtype


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("precision", FSDP_MIXED_PRECISION)
def test_fsdp_with_mixed_precision(backend, tiny_model_factory, tiny_model_data_factory, precision):
    run_distributed_test(
        run_fsdp_with_mixed_precision,
        backend=backend,
        func_args=(tiny_model_factory, tiny_model_data_factory, precision),
        start_method="spawn",
    )
