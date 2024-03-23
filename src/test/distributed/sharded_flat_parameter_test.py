from typing import Optional

import pytest
import torch
import torch.distributed as dist

from olmo_core.distributed.sharded_flat_parameter import (
    ShardedFlatParameter,
    ShardingSpec,
)

from .utils import BACKENDS, get_default_device, run_distributed_test


def test_init_empty_sharded_parameter():
    sp = ShardedFlatParameter()
    assert isinstance(sp, ShardedFlatParameter)
    assert isinstance(sp, torch.nn.Parameter)
    assert repr(sp) == "ShardedFlatParameter([], requires_grad=True)"
    assert not sp.is_sharded  # hasn't been marked sharded yet


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


def shard_and_gather(init_device: Optional[str] = None):
    assert dist.get_world_size() == 2

    unsharded_shape = (2, 3)

    for unsharded_flattened_offsets in [
        ((0, 3), (3, 6)),  # balanced sharding
        ((0, 2), (2, 6)),  # unbalanced sharding
        ((2, 6), (0, 2)),  # unordered, unbalanced sharding
        ((0, 6), (6, 6)),  # some ranks empty
        None,  # let ShardedFlatParameter decide
    ]:
        tensor = torch.rand(*unsharded_shape, device=init_device)

        sharding_spec: Optional[ShardingSpec] = None
        if unsharded_flattened_offsets is not None:
            sharding_spec = ShardingSpec(
                unsharded_shape=unsharded_shape, unsharded_flattened_offsets=unsharded_flattened_offsets
            )

        sharded_param = ShardedFlatParameter.shard(tensor, sharding_spec, device=get_default_device())

        param = sharded_param.gather()
        assert param.shape == tensor.shape

        if tensor.device != torch.device("meta"):
            torch.testing.assert_close(tensor, param)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("init_device", ("meta", None))
def test_shard_and_gather(backend, init_device: Optional[str]):
    run_distributed_test(shard_and_gather, backend=backend, func_kwargs=dict(init_device=init_device))
