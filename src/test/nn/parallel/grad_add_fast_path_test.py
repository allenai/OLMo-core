"""Validate mixed-dtype addition through real DDP accumulation and sharded Adam."""

from contextlib import nullcontext
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import init_device_mesh

from olmo_core.nn.parallel import MultiGroupDistributedDataParallel
from olmo_core.optim.moe_optimizer import OLMoDDPOptimizer
from olmo_core.testing import BACKENDS, run_distributed_test


def _run_optimizer_parity():
    from olmo_core.distributed.utils import backend_supports_cuda
    from olmo_core.ops.grad_accum import gradient_add

    rank, world = dist.get_rank(), dist.get_world_size()
    cuda = backend_supports_cuda()
    device = torch.device(f"cuda:{rank}" if cuda else "cpu")
    if cuda:
        torch.cuda.set_device(device)
    dtype = torch.bfloat16 if cuda else torch.float32  # Optimizer's BF16 masters are CUDA-only.
    width = 8192 if cuda else 128  # CUDA reaches the actual >=64Mi-element fast-path gate.
    mesh = init_device_mesh(device.type, (world,), mesh_dim_names=("dp",))
    stacks = []
    for enabled in (False, True):
        torch.manual_seed(731)
        model = nn.Linear(width, width, bias=False, dtype=dtype, device=device)
        ddp = MultiGroupDistributedDataParallel(
            model, init_sync=False, accumulate_grads_in_fp32=True, reduce_grads_in_fp32=True
        )
        ddp._vectorized_fp32_grad_add = enabled
        optim = OLMoDDPOptimizer(
            [{"named_params": dict(ddp.named_parameters()), "pg": "dp"}],
            world_mesh={"dense": mesh, "moe": None},
            dp_group=dist.group.WORLD,
            model_has_grad_accum_fp32_buffer=True,
            use_distributed=cuda,  # CPU fallback uses replicated FP32 parameters/states.
            lr=1e-3,
            betas=(0.9, 0.95),
            max_grad_norm=1.0,
        )
        stacks.append((ddp, optim))
    with patch("olmo_core.ops.grad_accum.gradient_add", wraps=gradient_add) as fast_call:
        for step in range(3):
            torch.manual_seed(900 + step + rank)
            inputs = [torch.randn(4, width, dtype=dtype, device=device) for _ in range(8)]
            for ddp, _ in stacks:
                for index, x in enumerate(inputs):
                    with ddp.no_sync() if index < 7 else nullcontext():
                        (ddp(x).float().square().mean() / 8).backward()
                ddp.finalize_grad_reduce()
            before, after = [next(ddp.parameters()) for ddp, _ in stacks]
            assert before.grad is after.grad is None
            torch.testing.assert_close(
                before._main_grad_fp32, after._main_grad_fp32, rtol=0, atol=0
            )
            for _, optim in stacks:
                optim.step()
                assert not bool(optim._step_skipped.item())
            torch.testing.assert_close(before, after, rtol=0, atol=0)
            optim_old, optim_new = [optim for _, optim in stacks]
            assert optim_old.states.keys() == optim_new.states.keys()
            for name in optim_old.states:
                torch.testing.assert_close(
                    optim_old.states[name].to_local(),
                    optim_new.states[name].to_local(),
                    rtol=0,
                    atol=0,
                )
            # Exercise both ordinary zeroing and next-forward view rebinding.
            for ddp, _ in stacks:
                ddp.zero_grad(set_to_none=(step == 1))
        assert fast_call.call_count == (24 if cuda else 0)


@pytest.mark.parametrize("backend", BACKENDS)
def test_vectorized_grad_add_sharded_adam_parity(backend):
    run_distributed_test(_run_optimizer_parity, backend=backend, start_method="spawn")
