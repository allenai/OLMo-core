import pytest
import torch
import torch.distributed as dist
import torch.nn as nn

from olmo_core.distributed.fsdp.flat_param_handle import FlatParamHandle
from olmo_core.utils import same_storage

from ..utils import BACKENDS, get_default_device, run_distributed_test


def run_flat_param_handle_case1():
    og_params = [
        nn.Parameter(torch.rand(2, 3, device=get_default_device())),
        nn.Parameter(torch.rand(4, 8, device=get_default_device())),
        nn.Parameter(torch.rand(7, device=get_default_device())),
    ]
    for og_param in og_params:
        dist.all_reduce(og_param.data)

    handle = FlatParamHandle.shard_params(
        [nn.Parameter(p.data.clone()) for p in og_params], ["x", "y", "z"], device=get_default_device()
    )
    assert handle.params_data.is_sharded
    for og_param, flat_param in zip(og_params, handle.params):
        assert same_storage(flat_param, handle.params_data)
        assert flat_param.is_sharded
        assert flat_param.unsharded_shape == og_param.shape

    # Unshard all params.
    handle.unshard_()
    for i, (og_param, flat_param) in enumerate(zip(og_params, handle.params)):
        assert not flat_param.is_sharded
        torch.testing.assert_close(
            flat_param.data, og_param.data, msg=lambda msg: f"Mismatch for param {i} - {msg}"
        )
        assert same_storage(flat_param, handle.params_data)

    # Reshard all params.
    handle.reshard_()
    for flat_param in handle.params:
        assert flat_param.is_sharded
        assert same_storage(flat_param, handle.params_data)

    # Updated the data in a param should update the data in the handle, since the data in the
    # param is just a view into the data in the handle.
    with torch.no_grad():
        handle.params[0].fill_(torch.tensor(0.0, device=get_default_device()))
        assert (handle.params_data[0 : handle.params[0].numel()] == 0).all()
        handle.unshard_()
        assert (handle.params[0] == 0).all()
        handle.params[0].fill_(torch.tensor(1.0, device=get_default_device()))
        handle.reshard_(writeback=True)
        assert (handle.params[0] == 1).all()


@pytest.mark.parametrize("backend", BACKENDS)
def test_flat_param_handle_case1(backend):
    run_distributed_test(
        run_flat_param_handle_case1,
        backend=backend,
    )


def run_flat_param_handle_case2():
    og_params = [
        nn.Parameter(torch.rand(2, 2, device=get_default_device())),
        nn.Parameter(torch.rand(3, 4, device=get_default_device())),
        nn.Parameter(torch.rand(2, 1, device=get_default_device())),
        nn.Parameter(torch.rand(2, 1, device=get_default_device())),
    ]
    for og_param in og_params:
        dist.all_reduce(og_param.data)

    handle = FlatParamHandle.shard_params(
        [nn.Parameter(p.data.clone()) for p in og_params], ["x", "y", "z"], device=get_default_device()
    )
    assert handle.params_data.is_sharded
    for og_param, flat_param in zip(og_params, handle.params):
        assert same_storage(flat_param, handle.params_data)
        assert flat_param.is_sharded
        assert flat_param.unsharded_shape == og_param.shape

    # Unshard all params.
    handle.unshard_()
    for i, (og_param, flat_param) in enumerate(zip(og_params, handle.params)):
        assert not flat_param.is_sharded
        torch.testing.assert_close(
            flat_param.data, og_param.data, msg=lambda msg: f"Mismatch for param {i} - {msg}"
        )
        assert same_storage(flat_param, handle.params_data)

    # Reshard all params.
    handle.reshard_()
    for flat_param in handle.params:
        assert flat_param.is_sharded
        assert same_storage(flat_param, handle.params_data)


def test_flat_param_handle_case2():
    run_distributed_test(
        run_flat_param_handle_case2,
        backend="gloo",
        world_size=4,
    )
