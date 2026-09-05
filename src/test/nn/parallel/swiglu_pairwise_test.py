"""Qualify compiled activation integration and actual routed experts with sharded Adam."""

import os
from contextlib import nullcontext
from functools import partial

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.device_mesh import init_device_mesh

from olmo_core.config import DType
from olmo_core.nn.moe.v2.routed_experts import RoutedExperts
from olmo_core.nn.parallel import MultiGroupDistributedDataParallel
from olmo_core.optim.moe_optimizer import OLMoDDPOptimizer
from olmo_core.testing import run_distributed_test


def _activation(x):
    up, gate = x.chunk(2, dim=-1)
    return up * F.silu(gate)


@pytest.mark.gpu
def test_compiled_pairwise_activation():
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    from torch._inductor.utils import run_and_get_code

    from olmo_core.ops.swiglu_pairwise import pairwise_swiglu

    torch.manual_seed(132)
    source = torch.randn(1024, 2048, device="cuda", dtype=torch.bfloat16)
    grad = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
    outputs = []
    for fn in (_activation, pairwise_swiglu):
        compiled = torch.compile(fn, fullgraph=True, dynamic=False)
        x = source.detach().clone().requires_grad_(True)

        def execute():
            y = compiled(x)
            y.backward(grad)
            return y

        y, code = run_and_get_code(execute)
        outputs.append((y.detach(), x.grad))
        if fn is pairwise_swiglu:
            assert any("_swiglu_backward_pair" in source for source in code)
    torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)


def _run_routed_adam_parity(candidate="activation"):
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    mesh = init_device_mesh("cuda", (world,), mesh_dim_names=("dp",))
    stacks = []
    for enabled in (False, True):
        torch.manual_seed(731)
        if candidate == "rounded-wgrad":
            os.environ["OLMO_PROFILE_ROUNDED_WGRAD"] = "1" if enabled else "0"
        model = RoutedExperts(
            d_model=512,
            hidden_size=1024,
            num_experts=8,
            bias=False,
            dtype=DType.bfloat16,
            init_device=str(device),
        )
        model._profile_pairwise_swiglu = enabled if candidate == "activation" else True
        with torch.no_grad():
            for param in model.parameters():
                param.normal_(std=0.02)
        model.compile(dynamic=False)
        ddp = MultiGroupDistributedDataParallel(
            model, init_sync=False, accumulate_grads_in_fp32=True, reduce_grads_in_fp32=True
        )
        optim = OLMoDDPOptimizer(
            [{"named_params": dict(ddp.named_parameters()), "pg": "dp"}],
            world_mesh={"dense": mesh, "moe": None},
            dp_group=dist.group.WORLD,
            model_has_grad_accum_fp32_buffer=True,
            use_distributed=True,
            lr=1e-3,
            betas=(0.9, 0.95),
            max_grad_norm=1.0,
        )
        stacks.append((ddp, optim))
    os.environ.pop("OLMO_PROFILE_ROUNDED_WGRAD", None)
    counts = torch.tensor([16, 16, 32, 32, 32, 32, 48, 48], device=device, dtype=torch.int32)
    for step in range(3):
        torch.manual_seed(199 + step + rank)
        inputs = [torch.randn(256, 512, device=device, dtype=torch.bfloat16) for _ in range(8)]
        losses = []
        for ddp, _ in stacks:
            values = []
            for i, x in enumerate(inputs):
                with ddp.no_sync() if i < 7 else nullcontext():
                    loss = ddp(x, counts).float().square().mean() / 8
                    loss.backward()
                    values.append(loss.detach())
            ddp.finalize_grad_reduce()
            losses.append(torch.stack(values))
        torch.testing.assert_close(losses[0], losses[1], rtol=0, atol=0)
        for old, new in zip(stacks[0][0].parameters(), stacks[1][0].parameters()):
            torch.testing.assert_close(old._main_grad_fp32, new._main_grad_fp32, rtol=0, atol=0)
        for _, optim in stacks:
            optim.step()
            assert not bool(optim._step_skipped.item())
        for old, new in zip(stacks[0][0].parameters(), stacks[1][0].parameters()):
            torch.testing.assert_close(old, new, rtol=0, atol=0)
        old_opt, new_opt = (s[1] for s in stacks)
        assert old_opt.states.keys() == new_opt.states.keys()
        for name in old_opt.states:
            torch.testing.assert_close(
                old_opt.states[name].to_local(), new_opt.states[name].to_local(), rtol=0, atol=0
            )
        for ddp, _ in stacks:
            ddp.zero_grad(set_to_none=(step == 1))


@pytest.mark.gpu
def test_pairwise_routed_experts_sharded_adam():
    if torch.cuda.device_count() < 2:
        pytest.skip("requires two CUDA devices")
    run_distributed_test(_run_routed_adam_parity, backend="nccl", start_method="spawn")


@pytest.mark.gpu
def test_rounded_wgrad_routed_experts_sharded_adam():
    if torch.cuda.device_count() < 2:
        pytest.skip("requires two CUDA devices")
    run_distributed_test(
        partial(_run_routed_adam_parity, "rounded-wgrad"), backend="nccl", start_method="spawn"
    )
