from functools import partial

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from olmo_core.distributed.fsdp import FSDP, FSDPDebugConfig
from olmo_core.distributed.tensors import ShardedFlatParameter

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

    model = model_factory().to(get_default_device())
    fsdp1 = FSDP(
        model_factory(), free_root_after_forward=True, _debug_config=FSDPDebugConfig(no_reduce_grads=True)
    )
    fsdp2 = FSDP(model_factory(), free_root_after_forward=True)

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

        with fsdp.summon_full_params():
            for name, param in fsdp.module.named_parameters():
                with torch.no_grad():
                    torch.testing.assert_close(param.data, model.state_dict()[name])

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
                param2.sharded_chunk(param1.grad) / dist.get_world_size(),
                param2.grad,
                msg=lambda m: f"On gradient for '{name}'. {m}",
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
    fsdp_model = FSDP(model_factory(), free_root_after_forward=True)

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
            torch.testing.assert_close(
                param.grad,
                param.sharded_chunk(expected_grads[name]),
                msg=lambda m: f"On gradient for '{name}'. {m}",
            )

    assert fsdp_model.state.flat_param_handles[0].params_sharded_grad is not None
    assert fsdp_model.state.flat_param_handles[0].params_unsharded_grad is None

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
                param.sharded_chunk(expected_grads[name]),
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
            self.register_buffer("buf", torch.tensor([1.0, 2.0]), persistent=True)

        def forward(self, x):
            x = self.inner(x)
            return self.out(x)

    fsdp = FSDP(NestedModel())
    assert isinstance(fsdp.module, NestedModel)

    # FSDP should forward getattr to wrapped module.
    assert isinstance(fsdp.out, nn.Linear)

    assert list(fsdp._fsdp_children()) == [fsdp.module.inner]
    assert list(fsdp._managed_named_parameters()) == [
        ("out.weight", fsdp.module.out.weight),
        ("out.bias", fsdp.module.out.bias),
    ]
    assert fsdp.is_root
    assert not fsdp.module.inner.is_root

    param_names = set(n for n, _ in fsdp.named_parameters())
    assert param_names == {
        "out.weight",
        "out.bias",
        "inner.fc.0.weight",
        "inner.fc.0.bias",
        "inner.fc.2.weight",
        "inner.fc.2.bias",
        "inner.fc.4.weight",
        "inner.fc.4.bias",
    }, param_names

    assert set(fsdp.state.flat_param_handles[0].param_fqns) == {
        "out.weight",
        "out.bias",
    }

    assert set(fsdp.module.inner.state.flat_param_handles[0].param_fqns) == {
        "fc.0.weight",
        "fc.0.bias",
        "fc.2.weight",
        "fc.2.bias",
        "fc.4.weight",
        "fc.4.bias",
    }

    buf_names = set(n for n, _ in fsdp.named_buffers())
    assert buf_names == {"buf"}

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
            "buf",
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
            "buf",
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


def run_fsdp_with_mix_of_frozen_and_non_frozen_params(case: int):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.ff1 = nn.Linear(8, 8)
            self.ff2 = nn.Linear(8, 8)
            if case == 1:
                self.ff1.weight.requires_grad = False
                self.ff1.bias.requires_grad = False
            elif case == 2:
                self.ff2.weight.requires_grad = False
                self.ff2.bias.requires_grad = False
            else:
                raise NotImplementedError

        def forward(self, x):
            return self.ff2(self.ff1(x))

    fsdp = FSDP(Model())

    # Check handles.
    assert len(fsdp.state.flat_param_handles) == 2
    assert fsdp.state.flat_param_handles[0].requires_grad
    assert not fsdp.state.flat_param_handles[1].requires_grad

    # Check params.
    for name, param in fsdp.named_parameters():
        assert param.grad is None, f"param {param} already has a grad!"

    # Run forward pass
    loss = fsdp(torch.rand(2, 8, device=fsdp.device)).sum()

    # Trigger backward pass.
    loss.backward()

    # Check grads.
    if case == 1:
        assert fsdp.module.ff1.weight.grad is None
        assert fsdp.module.ff1.bias.grad is None
        assert fsdp.module.ff2.weight.grad is not None
        assert fsdp.module.ff2.bias.grad is not None
    elif case == 2:
        assert fsdp.module.ff1.weight.grad is not None
        assert fsdp.module.ff1.bias.grad is not None
        assert fsdp.module.ff2.weight.grad is None
        assert fsdp.module.ff2.bias.grad is None
    else:
        raise NotImplementedError

    # Make sure every param has been resharded.
    for name, param in fsdp.named_parameters():
        assert isinstance(param, ShardedFlatParameter)
        assert param.is_sharded, f"param {name} has not been resharded!"


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("case", (1, 2))
def test_fsdp_with_mix_of_frozen_and_non_frozen_params(backend, case):
    run_distributed_test(
        run_fsdp_with_mix_of_frozen_and_non_frozen_params,
        backend=backend,
        start_method="spawn",
        func_args=(case,),
    )


def run_fsdp_with_frozen_fsdp_child(case: int):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            ff1 = nn.Linear(8, 8)
            ff2 = nn.Linear(8, 8)

            if case == 1:
                ff1.weight.requires_grad = False
                ff1.bias.requires_grad = False
                ff1 = FSDP(ff1)
            elif case == 2:
                ff2.weight.requires_grad = False
                ff2.bias.requires_grad = False
                ff2 = FSDP(ff2)
            else:
                raise NotImplementedError

            self.ff1 = ff1
            self.ff2 = ff2

        def forward(self, x):
            return self.ff2(self.ff1(x))

    fsdp = FSDP(Model())

    # Run forward pass
    loss = fsdp(torch.rand(2, 8, device=fsdp.device)).sum()

    # Trigger backward pass.
    loss.backward()

    # Check grads.
    if case == 1:
        assert fsdp.module.ff1.weight.grad is None
        assert fsdp.module.ff1.bias.grad is None
        assert fsdp.module.ff2.weight.grad is not None
        assert fsdp.module.ff2.bias.grad is not None
    elif case == 2:
        assert fsdp.module.ff1.weight.grad is not None
        assert fsdp.module.ff1.bias.grad is not None
        assert fsdp.module.ff2.weight.grad is None
        assert fsdp.module.ff2.bias.grad is None
    else:
        raise NotImplementedError

    # Make sure every param has been resharded.
    for name, param in fsdp.named_parameters():
        assert isinstance(param, ShardedFlatParameter)
        assert param.is_sharded, f"param {name} has not been resharded!"


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("case", (1, 2))
def test_fsdp_with_frozen_fsdp_child(backend, case):
    run_distributed_test(
        run_fsdp_with_frozen_fsdp_child,
        backend=backend,
        start_method="spawn",
        func_args=(case,),
    )


def run_fsdp_with_node_activation_checkpointing():
    from torch.utils.checkpoint import checkpoint

    checkpoint_fn = partial(checkpoint, use_reentrant=False)

    class Model(nn.Module):
        def __init__(self, do_activation_checkpointing: bool = True):
            super().__init__()
            self.ff1 = FSDP(nn.Linear(4, 8))
            self.ff2 = FSDP(nn.Linear(8, 8))
            self.ff3 = FSDP(nn.Linear(8, 4))
            self.do_activation_checkpointing = do_activation_checkpointing

        def forward(self, x):
            x = self.ff1(x)
            if self.do_activation_checkpointing:
                x = checkpoint_fn(self.ff2, x)
            else:
                x = self.ff2(x)
            x = self.ff3(x)
            return x

    fsdp_ckpt = FSDP(Model(do_activation_checkpointing=True))
    fsdp = FSDP(Model(do_activation_checkpointing=True))

    # Synchronize weights.
    fsdp.load_state_dict(fsdp_ckpt.state_dict())

    # Run forward pass
    inputs = torch.rand(2, 4, device=fsdp.device)
    loss_ckpt = fsdp_ckpt(inputs).sum()
    loss = fsdp(inputs).sum()
    torch.testing.assert_close(loss_ckpt, loss)

    # Run backward pass.
    loss_ckpt.backward()
    loss.backward()
    for p1, p2 in zip(fsdp_ckpt.parameters(), fsdp.parameters()):
        assert p1.grad is not None
        assert p2.grad is not None
        assert p1.grad.shape == p2.grad.shape
        torch.testing.assert_close(p1.grad, p2.grad)


@pytest.mark.parametrize("backend", BACKENDS)
def test_fsdp_with_node_activation_checkpointing(backend):
    run_distributed_test(
        run_fsdp_with_node_activation_checkpointing,
        backend=backend,
        start_method="spawn",
    )


def run_fsdp_with_intra_node_activation_checkpointing():
    from torch.utils.checkpoint import checkpoint

    checkpoint_fn = partial(checkpoint, use_reentrant=False)

    class SubModel(nn.Module):
        def __init__(self, do_activation_checkpointing: bool = True):
            super().__init__()
            self.ff1 = nn.Linear(8, 4)
            self.ff2 = nn.Linear(4, 8)
            self.do_activation_checkpointing = do_activation_checkpointing

        def forward(self, x):
            x = self.ff1(x)
            if self.do_activation_checkpointing:
                x = checkpoint_fn(self.ff2, x)
            else:
                x = self.ff2(x)
            return x

    class Model(nn.Module):
        def __init__(self, do_activation_checkpointing: bool = True):
            super().__init__()
            self.ff1 = nn.Linear(4, 8)
            self.ff2 = FSDP(SubModel(do_activation_checkpointing=do_activation_checkpointing))

        def forward(self, x):
            x = self.ff1(x)
            x = self.ff2(x)
            return x

    fsdp_ckpt = FSDP(Model(do_activation_checkpointing=True))
    fsdp = FSDP(Model(do_activation_checkpointing=True))

    # Synchronize weights.
    fsdp.load_state_dict(fsdp_ckpt.state_dict())

    # Run forward pass
    inputs = torch.rand(2, 4, device=fsdp.device)
    loss_ckpt = fsdp_ckpt(inputs).sum()
    loss = fsdp(inputs).sum()
    torch.testing.assert_close(loss_ckpt, loss)

    # Run backward pass.
    loss_ckpt.backward()
    loss.backward()
    for p1, p2 in zip(fsdp_ckpt.parameters(), fsdp.parameters()):
        assert p1.grad is not None
        assert p2.grad is not None
        assert p1.grad.shape == p2.grad.shape
        torch.testing.assert_close(p1.grad, p2.grad)


@pytest.mark.parametrize("backend", BACKENDS)
def test_fsdp_with_intra_node_activation_checkpointing(backend):
    run_distributed_test(
        run_fsdp_with_intra_node_activation_checkpointing,
        backend=backend,
        start_method="spawn",
    )


def run_fsdp_with_mixed_precision(model_factory, model_data_factory, precision):
    fsdp = FSDP(model_factory(), precision=precision)

    fsdp._unshard(cast=True, recurse=True)
    for param in fsdp.parameters():
        assert param.data.dtype == precision.param_dtype
    fsdp._reshard(recurse=True)

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
        assert isinstance(param, ShardedFlatParameter)
        assert param.is_sharded
        assert param.grad is not None
        assert param.grad.dtype == param.dtype


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("precision", FSDP_MIXED_PRECISION)
def test_fsdp_with_mixed_precision(backend, tiny_model_factory, tiny_model_data_factory, precision):
    if backend == "gloo" and (precision.param_dtype == torch.bfloat16 or precision.reduce_dtype == torch.bfloat16):
        pytest.skip("Weird combination")

    run_distributed_test(
        run_fsdp_with_mixed_precision,
        backend=backend,
        func_args=(tiny_model_factory, tiny_model_data_factory, precision),
        start_method="spawn",
    )


def run_auto_wrap():
    class ComplexNestedModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Sequential(nn.Linear(8, 8), nn.Linear(8, 8), nn.LayerNorm(8))
            self.fc2 = nn.Linear(8, 8)
            self.ln1 = nn.LayerNorm(8)
            self.fc3 = nn.ModuleDict(
                {
                    "in_proj": nn.Linear(8, 8),
                    "in_ln": nn.LayerNorm(8),
                    "out_proj": nn.Linear(8, 8),
                }
            )

    model = ComplexNestedModule()
    fsdp = FSDP.auto_wrap(model, ["fc1.*", nn.LayerNorm, model.fc3.out_proj], max_prefetch_count=3)

    assert isinstance(fsdp, FSDP)
    assert fsdp.is_root
    assert not isinstance(fsdp.module.fc1, FSDP)
    assert isinstance(fsdp.module.fc1[0], FSDP)
    assert not fsdp.module.fc1[0].is_root
    assert isinstance(fsdp.module.fc1[1], FSDP)
    assert isinstance(fsdp.module.fc1[2], FSDP)
    assert not isinstance(fsdp.module.fc2, FSDP)
    assert isinstance(fsdp.module.ln1, FSDP)
    assert not isinstance(fsdp.module.fc3, FSDP)
    assert not isinstance(fsdp.module.fc3.in_proj, FSDP)
    assert isinstance(fsdp.module.fc3.in_ln, FSDP)
    assert isinstance(fsdp.module.fc3.out_proj, FSDP)
    assert fsdp.module.fc3.out_proj.max_prefetch_count == 3


@pytest.mark.parametrize("backend", BACKENDS)
def test_auto_wrap(backend):
    run_distributed_test(
        run_auto_wrap,
        backend=backend,
    )


def run_apply():
    class ComplexNestedModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(8, 8)
            self.ln1 = nn.LayerNorm(8)
            self.fc2 = nn.ModuleDict(
                {
                    "fc1": nn.Linear(8, 8),
                    "ln1": nn.LayerNorm(8),
                    "fc2": nn.Linear(8, 8),
                }
            )

    model = ComplexNestedModule()
    fsdp = FSDP.auto_wrap(model, ["fc2", "fc2.fc1"])

    assert isinstance(fsdp, FSDP)
    assert fsdp.is_root
    assert not isinstance(fsdp.module.fc1, FSDP)
    assert not isinstance(fsdp.module.ln1, FSDP)
    assert isinstance(fsdp.module.fc2, FSDP)
    assert not fsdp.module.fc2.is_root
    assert isinstance(fsdp.module.fc2.module.fc1, FSDP)
    assert not fsdp.module.fc2.module.fc1.is_root
    assert not isinstance(fsdp.module.fc2.module.ln1, FSDP)
    assert not isinstance(fsdp.module.fc2.module.fc2, FSDP)

    def initialize_and_check(m: nn.Module):
        if isinstance(m, FSDP):
            # All managed params should be unsharded.
            for _, param in m._managed_named_parameters():
                if isinstance(param, ShardedFlatParameter):
                    assert not param.is_sharded

            # But all params in child instances should still be sharded.
            for child in m._fsdp_children(recurse=True):
                for _, param in child._managed_named_parameters():
                    if isinstance(param, ShardedFlatParameter):
                        assert param.is_sharded

        with torch.no_grad():
            for param in m.parameters(recurse=False):
                param.data.fill_(1.1)

    fsdp.apply(initialize_and_check)

    # Now validate that the param changes were written back.
    for param in fsdp.parameters():
        assert (param.data.detach() == 1.1).all()


@pytest.mark.parametrize("backend", BACKENDS)
def test_apply(backend):
    run_distributed_test(
        run_apply,
        backend=backend,
    )
