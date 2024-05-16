from typing import Optional

import pytest
import torch
import torch.distributed as dist

from olmo_core.distributed.tensors.sharded_flat_tensor import (
    ShardedFlatTensor,
    ShardingSpec,
)

from ..utils import BACKENDS, INIT_DEVICES, get_default_device, run_distributed_test


def test_init_sharded():
    tensor = ShardedFlatTensor(torch.tensor([0]))
    assert isinstance(tensor, ShardedFlatTensor)
    assert isinstance(tensor, ShardedFlatTensor)
    assert isinstance(tensor, torch.Tensor)
    assert tensor.metadata_set
    assert repr(tensor) == "ShardedFlatTensor(local_tensor=tensor([0]))"


def test_not_has_metadata():
    tensor = torch.Tensor._make_subclass(ShardedFlatTensor, torch.rand(3), False)
    assert isinstance(tensor, ShardedFlatTensor)
    assert not tensor.metadata_set


def test_init_sharded_tensor_from_tensor():
    tensor = ShardedFlatTensor(torch.rand(6))
    assert isinstance(tensor, torch.Tensor)
    assert isinstance(tensor, ShardedFlatTensor)
    assert tensor.shape == (6,)


def test_init_new_tensor_from_sharded_tensor():
    x = ShardedFlatTensor(torch.rand(6))
    x.mark_as_sharded(ShardingSpec(unsharded_shape=(2, 6), unsharded_flattened_offsets=(((0, 6),), ((6, 12),))))

    y1 = torch.empty_like(x)
    assert isinstance(y1, ShardedFlatTensor)
    assert y1.is_sharded

    y2 = torch.zeros_like(x)
    assert isinstance(y2, ShardedFlatTensor)
    assert y2.is_sharded

    y3 = torch.ones_like(x)
    assert isinstance(y3, ShardedFlatTensor)
    assert y3.is_sharded


def test_init_sharded_tensor_from_param():
    tensor = ShardedFlatTensor(torch.nn.Parameter(torch.rand(6)))
    assert isinstance(tensor, ShardedFlatTensor)
    assert tensor.shape == (6,)


def test_init_sharded_tensor_from_sharded_tensor():
    tensor = ShardedFlatTensor(ShardedFlatTensor(torch.rand(6)))
    assert isinstance(tensor, ShardedFlatTensor)
    assert tensor.shape == (6,)


def shard_and_gather(init_device: torch.device):
    assert dist.get_world_size() == 2

    unsharded_shape = (2, 3)

    for unsharded_flattened_offsets in [
        (((0, 3),), ((3, 6),)),  # balanced sharding
        (((0, 2),), ((2, 6),)),  # unbalanced sharding
        (((2, 6),), ((0, 2),)),  # unordered, unbalanced sharding
        (((0, 6),), ((6, 6),)),  # some ranks empty
        (((0, 2), (4, 6)), ((2, 4),)),  # more than one chunk on a rank
        None,  # let ShardedFlatTensor decide
    ]:
        tensor = torch.rand(*unsharded_shape, device=init_device)

        sharding_spec: Optional[ShardingSpec] = None
        if unsharded_flattened_offsets is not None:
            sharding_spec = ShardingSpec(
                unsharded_shape=unsharded_shape, unsharded_flattened_offsets=unsharded_flattened_offsets
            )

        # Shard tensor.
        sharded_tensor = ShardedFlatTensor.shard(tensor, sharding_spec, device=get_default_device())

        # Get unsharded version of the tensor.
        param = sharded_tensor.gather()
        assert param.shape == tensor.shape

        # Check that unsharded tensor matches original data.
        if init_device == param.device:
            torch.testing.assert_close(tensor, param)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("init_device", INIT_DEVICES)
def test_shard_and_gather(backend, init_device: torch.device):
    if backend == "gloo" and init_device.type == "cuda":
        pytest.skip("Weird combination")
    run_distributed_test(shard_and_gather, backend=backend, func_args=(init_device,))


def unshard_reshard_in_place_rank0_only():
    flat_tensor = ShardedFlatTensor.shard(torch.rand(2, 3), requires_grad=False, device=get_default_device())
    tensor = flat_tensor.gather()
    assert flat_tensor.grad is None
    assert flat_tensor.is_sharded

    # Unshard in place from rank0 only.
    flat_tensor.unshard_(rank0_only=True)
    assert not flat_tensor.is_sharded
    if dist.get_rank() == 0:
        assert flat_tensor.shape == flat_tensor.unsharded_shape == tensor.shape
        torch.testing.assert_close(flat_tensor, tensor)
    else:
        assert flat_tensor.numel() == 0

    # Reshard in place.
    flat_tensor.reshard_()
    assert flat_tensor.shape == flat_tensor.sharded_shape
    assert flat_tensor.is_sharded

    # Unshard in place again, this time using a different dtype and modifying the data.
    flat_tensor.unshard_(rank0_only=True, dtype=torch.float16)
    assert flat_tensor.dtype == torch.float16
    assert not flat_tensor.is_sharded
    if dist.get_rank() == 0:
        assert flat_tensor.shape == flat_tensor.unsharded_shape == tensor.shape
        torch.testing.assert_close(flat_tensor, tensor.to(torch.float16))
        flat_tensor.fill_(torch.tensor(0.0, dtype=torch.float16))
    else:
        assert flat_tensor.numel() == 0

    flat_tensor.reshard_(writeback=True)
    assert flat_tensor.is_sharded
    assert flat_tensor.shape == flat_tensor.sharded_shape
    assert flat_tensor.dtype == torch.float32
    torch.testing.assert_close(flat_tensor, torch.zeros_like(flat_tensor))


@pytest.mark.parametrize("backend", BACKENDS)
def test_unshard_reshard_in_place_rank0_only(backend):
    run_distributed_test(unshard_reshard_in_place_rank0_only, backend=backend)


def unshard_reshard_in_place_with_grads():
    flat_tensor = ShardedFlatTensor.shard(torch.rand(2, 3), requires_grad=True, device=get_default_device())
    assert flat_tensor.grad is None
    assert flat_tensor.is_sharded

    # Unshard in place and compute a gradient.
    flat_tensor.unshard_()
    flat_tensor.sum().backward()
    assert flat_tensor.grad is not None
    assert flat_tensor.grad.shape == flat_tensor.unsharded_shape
    assert not flat_tensor.is_sharded

    # Reshard in place. Gradient should remain untouched.
    flat_tensor.reshard_()
    assert flat_tensor.grad is not None
    assert flat_tensor.grad.shape == flat_tensor.unsharded_shape
    assert flat_tensor.is_sharded


@pytest.mark.parametrize("backend", BACKENDS)
def test_unshard_reshard_in_place_with_grads(backend):
    run_distributed_test(unshard_reshard_in_place_with_grads, backend=backend, start_method="spawn")
