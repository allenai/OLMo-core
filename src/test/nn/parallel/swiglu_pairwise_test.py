"""Qualify compiled activation integration and actual routed experts with sharded Adam."""

import io
import os
from contextlib import nullcontext
from functools import partial
from typing import Callable

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.device_mesh import init_device_mesh
from torch.utils.checkpoint import checkpoint

from olmo_core.config import DType
from olmo_core.nn.moe.v2.routed_experts import RoutedExperts
from olmo_core.nn.parallel import MultiGroupDistributedDataParallel
from olmo_core.optim.moe_optimizer import OLMoDDPOptimizer
from olmo_core.testing import requires_gpu, run_distributed_test


def _activation(x):
    up, gate = x.chunk(2, dim=-1)
    return up * F.silu(gate)


def test_pairwise_requires_compilation():
    pytest.importorskip("triton")
    from olmo_core.ops.swiglu_pairwise import pairwise_swiglu

    with pytest.raises(RuntimeError, match="requires torch.compile"):
        pairwise_swiglu(torch.ones(2, 4))


@requires_gpu
def test_pairwise_flag_preserves_eager_activation():
    torch.manual_seed(179)
    experts = RoutedExperts(
        d_model=128,
        hidden_size=128,
        num_experts=2,
        bias=False,
        dtype=DType.bfloat16,
        init_device="cuda",
    )
    experts._profile_pairwise_swiglu = True
    source = torch.randn(33, 256, device="cuda", dtype=torch.bfloat16)
    outputs = []
    functions: list[Callable[[torch.Tensor], torch.Tensor]] = [
        _activation,
        experts.chunk_and_activate,
    ]
    for fn in functions:
        x = source.clone().requires_grad_(True)
        y = fn(x)
        y.sum().backward()
        outputs.append((y, x.grad))
    torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)


class _RecomputedExperts(torch.nn.Module):
    """Match production: an eager checkpoint boundary around a compiled child."""

    def __init__(self, experts):
        super().__init__()
        self.experts = experts

    def forward(self, x, counts):
        return checkpoint(self.experts, x, counts, use_reentrant=False)


@requires_gpu
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


def _assert_activation_parity(source, grad):
    from olmo_core.ops.swiglu_pairwise import pairwise_swiglu

    outputs = []
    for fn in (_activation, pairwise_swiglu):
        x = source.detach().clone().requires_grad_(True)
        y = torch.compile(fn, fullgraph=True, dynamic=False)(x)
        y.backward(grad)
        outputs.append((y.detach(), x.grad))
    torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0, equal_nan=True)


@requires_gpu
@pytest.mark.parametrize("rows,hidden", [(0, 128), (1, 128), (17, 129), (513, 1536)])
def test_pairwise_activation_shapes(rows, hidden):
    torch.manual_seed(987)
    source = torch.randn(rows, 2 * hidden, device="cuda", dtype=torch.bfloat16)
    grad = torch.randn(rows, hidden, device="cuda", dtype=torch.bfloat16)
    _assert_activation_parity(source, grad)


@requires_gpu
def test_pairwise_activation_all_bf16_gates():
    # Includes the cancellation-sensitive -1.28125 gate from the Torch 2.13
    # regression, all finite BF16 values, signed zeros, infinities and NaNs.
    gates = (
        torch.arange(65536, device="cuda", dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
    )
    torch.manual_seed(132)
    source = torch.stack((torch.randn_like(gates), gates), dim=1)
    grad = torch.randn(65536, 1, device="cuda", dtype=torch.bfloat16)
    _assert_activation_parity(source, grad)


@requires_gpu
@pytest.mark.skipif(
    os.environ.get("OLMO_TEST_LARGE_SWIGLU") != "1", reason="opt-in large allocation"
)
def test_pairwise_activation_large_offsets():
    # Exercise real addresses beyond signed int32, not only the index expression.
    if torch.cuda.mem_get_info()[0] < 24 * 1024**3:
        pytest.skip("requires 24 GiB free device memory")
    from olmo_core.ops.swiglu_pairwise import swiglu_backward_pair

    rows, hidden = 1048577, 1024
    source = torch.ones(rows, 2 * hidden, device="cuda", dtype=torch.bfloat16)
    grad = torch.ones(rows, hidden, device="cuda", dtype=torch.bfloat16)
    actual = swiglu_backward_pair(source, grad)
    x = source[:1].clone().requires_grad_(True)
    torch.compile(_activation, fullgraph=True)(x).sum().backward()
    torch.testing.assert_close(actual, x.grad.expand_as(actual), rtol=0, atol=0)


def _run_routed_adam_parity(
    candidate="activation", reduction="all-reduce", recompute=False, resume=False
):
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    mesh = init_device_mesh("cuda", (world,), mesh_dim_names=("dp",))

    def make_stack(enabled):
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
        if recompute:
            # Recompute the inner module, never the outer DDP wrapper: its
            # forward epoch must advance only once per real microbatch.
            model = _RecomputedExperts(model)
        ddp = MultiGroupDistributedDataParallel(
            model,
            init_sync=False,
            accumulate_grads_in_fp32=True,
            reduce_grads_in_fp32=True,
            use_reduce_scatter=reduction != "all-reduce",
            bucket_cap_mb=1,
        )
        ddp._reduce_scatter_single_param_fast_path = reduction == "reduce-scatter-direct"
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
        if reduction != "all-reduce":
            ddp.configure_reduce_scatter_params(optim.normal_params_with_sharded_optimizer_state())
            assert all(bucket.reduce_scatter for bucket in ddp._grad_buckets)
            assert all(len(bucket.params) == 1 for bucket in ddp._grad_buckets)
        return ddp, optim

    stacks = [make_stack(enabled) for enabled in (False, True)]
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
            if reduction != "all-reduce":
                torch.testing.assert_close(
                    old._olmo_ddp_reduced_grad_shard,
                    new._olmo_ddp_reduced_grad_shard,
                    rtol=0,
                    atol=0,
                )
        for _, optim in stacks:
            optim.step()
            assert optim._step_skipped is not None
            assert not bool(optim._step_skipped.item())
        for old, new in zip(stacks[0][0].parameters(), stacks[1][0].parameters()):
            torch.testing.assert_close(old, new, rtol=0, atol=0)
        old_opt, new_opt = (s[1] for s in stacks)
        assert old_opt.states.keys() == new_opt.states.keys()
        for name in old_opt.states:
            torch.testing.assert_close(
                old_opt.states[name].to_local(), new_opt.states[name].to_local(), rtol=0, atol=0
            )
        if resume and step == 0:
            # Rebuild the candidate's parameters/DDP owners/optimizer from a real
            # serialized checkpoint, then compare its next updates to the
            # uninterrupted reference. No Python callback is checkpoint state.
            saved = io.BytesIO()
            torch.save(
                {"model": stacks[1][0].module.state_dict(), "optim": new_opt.state_dict()}, saved
            )
            saved.seek(0)
            state = torch.load(saved, weights_only=False)
            new_ddp, new_opt = make_stack(True)
            new_ddp.module.load_state_dict(state["model"])
            new_opt.load_state_dict(state["optim"])
            stacks[1] = (new_ddp, new_opt)
            os.environ.pop("OLMO_PROFILE_ROUNDED_WGRAD", None)
        for ddp, _ in stacks:
            ddp.zero_grad(set_to_none=(step == 1))


@requires_gpu
def test_pairwise_routed_experts_sharded_adam():
    if torch.cuda.device_count() < 2:
        pytest.skip("requires two CUDA devices")
    run_distributed_test(_run_routed_adam_parity, backend="nccl", start_method="spawn")


@requires_gpu
@pytest.mark.parametrize(
    "reduction", ("all-reduce", "reduce-scatter-packed", "reduce-scatter-direct")
)
@pytest.mark.parametrize("recompute", [False, True])
def test_rounded_wgrad_routed_experts_sharded_adam(reduction, recompute):
    if torch.cuda.device_count() < 2:
        pytest.skip("requires two CUDA devices")
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("rounded weight-gradient kernel requires Blackwell")
    run_distributed_test(
        partial(_run_routed_adam_parity, "rounded-wgrad", reduction, recompute),
        backend="nccl",
        start_method="spawn",
    )


@requires_gpu
def test_rounded_wgrad_checkpoint_resume():
    if torch.cuda.device_count() < 2 or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("requires two Blackwell GPUs")
    run_distributed_test(
        partial(_run_routed_adam_parity, "rounded-wgrad", "reduce-scatter-direct", True, True),
        backend="nccl",
        start_method="spawn",
    )
