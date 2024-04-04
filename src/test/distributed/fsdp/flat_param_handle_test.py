import pytest
import torch
import torch.distributed as dist

from olmo_core.distributed.fsdp.flat_param_handle import FlatParamHandle
from olmo_core.distributed.tensors import ShardedFlatParameter
from olmo_core.utils import same_storage

from ..utils import BACKENDS, get_default_device, run_distributed_test


def run_flat_param_handle_collate_flat_params():
    all_og_data = [
        torch.rand(2, 3, device=get_default_device()),
        torch.rand(4, 8, device=get_default_device()),
        torch.rand(7, device=get_default_device()),
    ]
    for og_data in all_og_data:
        dist.all_reduce(og_data)

    flat_params = [ShardedFlatParameter.shard(og_data) for og_data in all_og_data]
    handle = FlatParamHandle.collate_flat_params(flat_params, ["x", "y", "z"], device=get_default_device())
    for param in handle.params:
        assert same_storage(param, handle.params_data)

    # Unshard all params.
    handle.unshard_()
    for og_data, param in zip(all_og_data, handle.params):
        assert not param.is_sharded
        torch.testing.assert_close(param.data, og_data)

    # Reshard all params.
    handle.reshard_()
    for param in handle.params:
        assert param.is_sharded
        assert same_storage(param, handle.params_data)

    # Updated the data in a param should update the data in the handle, since the data in the
    # param is just a view into the data in the handle.
    handle.params[0].fill_(torch.tensor(0.0, device=get_default_device()))
    assert (handle.params_data[0 : handle.params[0].numel()] == 0).all()
    handle.unshard_()
    assert (handle.params[0] == 0).all()
    handle.params[0].fill_(torch.tensor(1.0, device=get_default_device()))
    handle.reshard_(writeback=True)
    assert (handle.params[0] == 1).all()


@pytest.mark.parametrize("backend", BACKENDS)
def test_flat_param_handle_collate_flat_params(backend):
    run_distributed_test(
        run_flat_param_handle_collate_flat_params,
        backend=backend,
    )
