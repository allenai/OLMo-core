import pytest
import torch

from olmo_core.distributed.fsdp import FSDP
from olmo_core.distributed.sharded_flat_parameter import ShardedFlatParameter

from ..utils import BACKENDS, get_default_device, run_distributed_test


def wrap_module(model_factory, model_data_factory):
    fsdp_model = FSDP(model_factory())
    model_data = model_data_factory().to(get_default_device())
    for name, param in fsdp_model.named_parameters():
        assert isinstance(param, ShardedFlatParameter), f"param {name}: {param}"
        assert param.is_sharded
        assert param.grad is None

    optim = torch.optim.AdamW(fsdp_model.parameters())

    # Run forward pass.
    loss = fsdp_model(model_data).sum()

    # Model should be in a sharded state again.
    for name, param in fsdp_model.named_parameters():
        assert isinstance(param, ShardedFlatParameter)
        assert param.is_sharded

    # Trigger backward pass.
    loss.backward()

    # Model should be in a sharded state again.
    for name, param in fsdp_model.named_parameters():
        assert isinstance(param, ShardedFlatParameter)
        assert param.is_sharded
        assert param.grad is not None

    # Run optimizer step.
    optim.step()


@pytest.mark.parametrize("backend", BACKENDS)
def test_wrap_module(backend, tiny_model_factory, tiny_model_data_factory):
    run_distributed_test(
        wrap_module, backend=backend, func_args=(tiny_model_factory, tiny_model_data_factory), start_method="spawn"
    )
