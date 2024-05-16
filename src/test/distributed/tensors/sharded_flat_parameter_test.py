from typing import Optional

import pytest
import torch
import torch.distributed as dist

from olmo_core.distributed.tensors.sharded_flat_parameter import ShardedFlatParameter
from olmo_core.distributed.tensors.sharded_flat_tensor import (
    ShardedFlatTensor,
    ShardingSpec,
)

from ..utils import BACKENDS, INIT_DEVICES, get_default_device, run_distributed_test


def test_init_empty_sharded_parameter():
    sp = ShardedFlatParameter()
    assert isinstance(sp, ShardedFlatParameter)
    assert isinstance(sp, torch.nn.Parameter)
    assert isinstance(sp, ShardedFlatTensor)
    assert isinstance(sp, torch.Tensor)
    assert repr(sp) == "ShardedFlatParameter(local_tensor=None, requires_grad=True)"


def test_init_sharded_parameter_from_tensor():
    sp = ShardedFlatParameter(torch.rand(6))
    assert isinstance(sp, ShardedFlatParameter)
    assert sp.shape == (6,)


def test_init_sharded_parameter_from_param():
    sp = ShardedFlatParameter(torch.nn.Parameter(torch.rand(6)))
    assert isinstance(sp, ShardedFlatParameter)
    assert sp.shape == (6,)


def test_init_sharded_parameter_from_sharded_param():
    sp = ShardedFlatParameter(ShardedFlatParameter(torch.rand(6)))
    assert isinstance(sp, ShardedFlatParameter)
    assert sp.shape == (6,)


def shard_and_gather(init_device: torch.device):
    assert dist.get_world_size() == 2

    unsharded_shape = (2, 3)

    for unsharded_flattened_offsets in [
        (((0, 3),), ((3, 6),)),  # balanced sharding
        (((0, 2),), ((2, 6),)),  # unbalanced sharding
        (((2, 6),), ((0, 2),)),  # unordered, unbalanced sharding
        (((0, 6),), ((6, 6),)),  # some ranks empty
        None,  # let ShardedFlatParameter decide
    ]:
        tensor = torch.rand(*unsharded_shape, device=init_device)

        sharding_spec: Optional[ShardingSpec] = None
        if unsharded_flattened_offsets is not None:
            sharding_spec = ShardingSpec(
                unsharded_shape=unsharded_shape, unsharded_flattened_offsets=unsharded_flattened_offsets
            )

        # Shard tensor.
        sharded_param = ShardedFlatParameter.shard(tensor, sharding_spec, device=get_default_device())

        # Get unsharded version of the parameter.
        param = sharded_param.gather()
        assert param.shape == tensor.shape

        # Check that unsharded parameter matches original data.
        if init_device == param.device:
            torch.testing.assert_close(tensor, param)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("init_device", INIT_DEVICES)
def test_shard_and_gather(backend, init_device: torch.device):
    if backend == "gloo" and init_device.type == "cuda":
        pytest.skip("Weird combination")
    run_distributed_test(shard_and_gather, backend=backend, func_args=(init_device,))


def unshard_reshard_in_place_rank0_only():
    flat_param = ShardedFlatParameter.shard(torch.rand(2, 3), requires_grad=False, device=get_default_device())
    param = flat_param.gather()
    assert flat_param.grad is None
    assert flat_param.is_sharded

    # Unshard in place from rank0 only.
    flat_param.unshard_(rank0_only=True)
    assert not flat_param.is_sharded
    if dist.get_rank() == 0:
        assert flat_param.shape == flat_param.unsharded_shape == param.shape
        torch.testing.assert_close(flat_param, param)
    else:
        assert flat_param.numel() == 0

    # Reshard in place.
    flat_param.reshard_()
    assert flat_param.shape == flat_param.sharded_shape
    assert flat_param.is_sharded

    # Unshard in place again, this time using a different dtype and modifying the data.
    flat_param.unshard_(rank0_only=True, dtype=torch.float16)
    assert flat_param.dtype == torch.float16
    assert not flat_param.is_sharded
    if dist.get_rank() == 0:
        assert flat_param.shape == flat_param.unsharded_shape == param.shape
        torch.testing.assert_close(flat_param, param.to(torch.float16))
        flat_param.fill_(torch.tensor(0.0, dtype=torch.float16))
    else:
        assert flat_param.numel() == 0

    flat_param.reshard_(writeback=True)
    assert flat_param.is_sharded
    assert flat_param.shape == flat_param.sharded_shape
    assert flat_param.dtype == torch.float32
    torch.testing.assert_close(flat_param, torch.zeros_like(flat_param))


@pytest.mark.parametrize("backend", BACKENDS)
def test_unshard_reshard_in_place_rank0_only(backend):
    run_distributed_test(unshard_reshard_in_place_rank0_only, backend=backend)


def unshard_reshard_in_place_with_grads():
    flat_param = ShardedFlatParameter.shard(torch.rand(2, 3), requires_grad=True, device=get_default_device())
    assert flat_param.grad is None
    assert flat_param.is_sharded

    # Unshard in place and compute a gradient.
    flat_param.unshard_()
    flat_param.sum().backward()
    assert flat_param.grad is not None
    assert flat_param.grad.shape == flat_param.unsharded_shape
    assert not flat_param.is_sharded

    # Reshard in place. Gradient should remain untouched.
    flat_param.reshard_()
    assert flat_param.grad is not None
    assert flat_param.grad.shape == flat_param.unsharded_shape
    assert flat_param.is_sharded


@pytest.mark.parametrize("backend", BACKENDS)
def test_unshard_reshard_in_place_with_grads(backend):
    run_distributed_test(unshard_reshard_in_place_with_grads, backend=backend, start_method="spawn")
