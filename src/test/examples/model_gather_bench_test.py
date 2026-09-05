"""Prototype-only gather layout/dtype/ownership parity, using real optimizer metadata."""

import pytest
import torch
import torch.distributed as dist

from examples.olmo_ddp.olmoe3_model_gather_bench import direct_large_gather, make_owner
from olmo_core.optim.moe_optimizer import OLMoDDPOptimizer
from olmo_core.testing import BACKENDS, run_distributed_test


def _run_gather_parity(dtype, thresholds):
    from olmo_core.distributed.utils import backend_supports_cuda

    rank = dist.get_rank()
    device = torch.device(f"cuda:{rank}" if backend_supports_cuda() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    solo = [dist.new_group([r]) for r in range(dist.get_world_size())]
    specs = [
        ("dp", dist.group.WORLD, [(16, True), (128, True), (12, False), (24, True), (256, True)]),
        ("solo", solo[rank], [(32, True), (8, False)]),
        ("unsharded", None, [(12, False)]),
    ]
    owner = make_owner(specs, device, dtype, varying_elements=True)
    pointers = {
        name: group.flat_buffer.data_ptr() for name, group in owner._flat_model_sync_groups.items()
    }
    masters = {key: value.to_local().clone() for key, value in owner.states.items()}
    # Repeated calls include an unchanged-master (skipped-update-like) iteration.
    for step in range(3):
        if step == 1:
            for key, value in owner.states.items():
                value.to_local().add_(0.12345)
                masters[key].add_(0.12345)
        OLMoDDPOptimizer._copy_main_params_to_flat_model_buffers(owner)
        expected = {
            name: group.flat_buffer.clone() for name, group in owner._flat_model_sync_groups.items()
        }
        for group in owner._flat_model_sync_groups.values():
            group.flat_buffer.fill_(float("nan"))
        direct_large_gather(owner, *thresholds)
        for name, group in owner._flat_model_sync_groups.items():
            torch.testing.assert_close(group.flat_buffer, expected[name], rtol=0, atol=0)
            assert group.flat_buffer.data_ptr() == pointers[name]
            for entry in group.sharded_entries + group.replicated_entries:
                assert entry.param.data_ptr() == entry.flat_slice.data_ptr()
        for key, value in owner.states.items():
            torch.testing.assert_close(value.to_local(), masters[key], rtol=0, atol=0)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("thresholds", [(1, 1), (200, 100), (10000, 10000)])
def test_model_gather_prototype(backend, dtype, thresholds):
    run_distributed_test(
        _run_gather_parity,
        backend=backend,
        start_method="spawn",
        func_args=(dtype, thresholds),
    )
