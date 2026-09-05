"""Exercise packing layouts, precision, group routing, and asynchronous input ownership."""

import pytest
import torch
import torch.distributed as dist
from torch import nn

from olmo_core.nn.parallel import MultiGroupDistributedDataParallel
from olmo_core.testing import BACKENDS, run_distributed_test


def _run_packing_parity(single_param, param_dtype, accumulate_fp32, split_groups):
    from olmo_core.distributed.utils import backend_supports_cuda

    rank = dist.get_rank()
    device = torch.device(f"cuda:{rank}" if backend_supports_cuda() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    solo_groups = [dist.new_group([r]) for r in range(dist.get_world_size())]
    for fast_path in (False, True):
        model = nn.ParameterDict(
            {
                name: nn.Parameter(torch.zeros(8, 8, device=device, dtype=param_dtype))
                for name in ("a", "b", "c")
            }
        )
        ddp = MultiGroupDistributedDataParallel(
            model,
            init_sync=False,
            bucket_cap_mb=0.0001 if single_param else 1,
            accumulate_grads_in_fp32=accumulate_fp32,
            reduce_grads_in_fp32=True,
            use_reduce_scatter=True,
            param_process_group_fn=lambda name, _: solo_groups[rank]
            if split_groups and name == "b"
            else None,
        )
        ddp._reduce_scatter_single_param_fast_path = fast_path
        ddp.configure_reduce_scatter_params(set(model.values()))
        # Repeated steps with several outstanding buckets catch shared-scratch aliasing.
        for step in range(3):
            expected = {}
            for index, param in enumerate(model.values()):
                view = ddp._param_to_bucket_view[param]
                values = torch.arange(param.numel(), device=device).reshape_as(param)
                view.copy_((values * 0.25 + rank + index * 4 + step * 0.5).to(view.dtype))
                expected[param] = view.float().clone()
                bucket_idx = ddp._param_to_bucket_idx[param]
                ddp._param_grad_ready[param] = True
                ddp._bucket_ready_count[bucket_idx] += 1
            ddp._maybe_kick_start_grad_reduce()
            ddp.finalize_grad_reduce()
            for param, reference in expected.items():
                pg = ddp.param_to_process_group[param]
                world_size, group_rank = dist.get_world_size(pg), dist.get_rank(pg)
                reference.div_(world_size)
                dist.all_reduce(reference, group=pg)
                shard = reference.flatten().chunk(world_size)[group_rank]
                actual = param._olmo_ddp_reduced_grad_shard
                torch.testing.assert_close(actual, shard.to(actual.dtype), rtol=0, atol=0)
        all_single = all(len(bucket.params) == 1 for bucket in ddp._grad_buckets)
        assert bool(ddp._reduce_scatter_pack_scratch) == (not fast_path or not all_single)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    "single_param,split_groups", [(True, False), (False, False), (False, True)]
)
@pytest.mark.parametrize(
    "param_dtype,accumulate_fp32",
    [(torch.float32, False), (torch.bfloat16, False), (torch.bfloat16, True)],
)
def test_reduce_scatter_single_param_fast_path(
    backend, single_param, split_groups, param_dtype, accumulate_fp32
):
    run_distributed_test(
        _run_packing_parity,
        backend=backend,
        start_method="spawn",
        func_args=(single_param, param_dtype, accumulate_fp32, split_groups),
    )
