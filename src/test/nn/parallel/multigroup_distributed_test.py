import copy
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh

from olmo_core.nn.parallel.distributed import MultiGroupDistributedDataParallel
from olmo_core.nn.fp8_weight import FP8WeightCacheSpec, FP8WeightStore
from olmo_core.optim.moe_optimizer import OLMoDDPOptimizer
from olmo_core.testing import BACKENDS, MULTI_GPU_MARKS, run_distributed_test
from olmo_core.utils import get_default_device, seed_all


class SelectiveModel(nn.Module):
    """Two independent branches; only one is used per forward."""

    def __init__(self, d: int):
        super().__init__()
        self.fc_a = nn.Linear(d, d)
        self.fc_b = nn.Linear(d, d)

    def forward(self, x: torch.Tensor, use_a: bool = True) -> torch.Tensor:
        return self.fc_a(x) if use_a else self.fc_b(x)


class IgnoredParamModel(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.keep = nn.Linear(d, d)
        self.ignore = nn.Linear(d, d)
        self._ddp_params_and_buffers_to_ignore = {"ignore.weight", "ignore.bias"}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.keep(x) + self.ignore(x)


class MixedReductionModel(nn.Module):
    """Adjacent large parameters share an RS bucket; the small bias uses AR."""

    def __init__(self, d: int):
        super().__init__()
        self.large_a = nn.Parameter(torch.empty(d, d))
        self.large_b = nn.Parameter(torch.empty(d, d))
        self.small_bias = nn.Parameter(torch.empty(d))
        nn.init.normal_(self.large_a)
        nn.init.normal_(self.large_b)
        nn.init.normal_(self.small_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.large_a + x @ self.large_b + self.small_bias


class TwoGroupModel(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.dense_weight = nn.Parameter(torch.empty(d, d))
        self.expert_weight = nn.Parameter(torch.empty(d, d))
        nn.init.normal_(self.dense_weight)
        nn.init.normal_(self.expert_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.dense_weight + x @ self.expert_weight


def _fake_fp8_prequantizer(tensor: torch.Tensor, **_kwargs):
    """Small optimizer-test stand-in for the hardware FP8 cache builder."""
    return SimpleNamespace(
        mat_b_q=tensor.detach().clone(),
        scale_b=torch.ones(1, device=tensor.device, dtype=torch.uint8),
    )


def _reference_grads(model: nn.Module, world_size: int) -> list[torch.Tensor]:
    grads = []
    for p in model.parameters():
        assert p.grad is not None
        g = p.grad.detach().clone()
        dist.all_reduce(g, op=dist.ReduceOp.SUM)
        g /= world_size
        grads.append(g)
    return grads


def _run_no_sync_skipped_param_grad_preserved(d: int):
    device = get_default_device()
    rank, world_size = dist.get_rank(), dist.get_world_size()

    seed_all(0)
    model = SelectiveModel(d).to(device)
    reference = copy.deepcopy(model)
    ddp = MultiGroupDistributedDataParallel(model, init_sync=False)

    torch.manual_seed(100 + rank)
    x_a = torch.randn(4, d, device=device)
    x_b = torch.randn(4, d, device=device)

    # fc_a receives a grad only in the unsynced accumulation micro-batch; fc_b only
    # in the final synced one. fc_a must survive finalize rather than be zeroed.
    with ddp.no_sync():
        ddp(x_a, use_a=True).pow(2).mean().backward()
    ddp(x_b, use_a=False).pow(2).mean().backward()
    ddp.finalize_grad_reduce()

    reference(x_a, use_a=True).pow(2).mean().backward()
    reference(x_b, use_a=False).pow(2).mean().backward()
    expected = _reference_grads(reference, world_size)

    for (name, p), g_ref in zip(ddp.module.named_parameters(), expected):
        assert p.grad is not None, f"missing grad for {name}"
        torch.testing.assert_close(p.grad, g_ref, rtol=1e-5, atol=1e-6)


def _run_fp32_hooks_skip_ignored_params(d: int):
    device = get_default_device()

    seed_all(0)
    model = IgnoredParamModel(d).to(device)
    ddp = MultiGroupDistributedDataParallel(
        model,
        init_sync=False,
        accumulate_grads_in_fp32=True,
        reduce_grads_in_fp32=True,
    )

    torch.manual_seed(100 + dist.get_rank())
    x = torch.randn(4, d, device=device)
    ddp(x).pow(2).mean().backward()
    ddp.finalize_grad_reduce()

    assert getattr(ddp.module.keep.weight, "_main_grad_fp32", None) is not None
    assert getattr(ddp.module.ignore.weight, "_main_grad_fp32", None) is None


def _run_mixed_reduce_scatter_all_reduce_matches_reference(d: int):
    device = get_default_device()
    rank, world_size = dist.get_rank(), dist.get_world_size()

    seed_all(0)
    model = MixedReductionModel(d).to(device)
    reference = copy.deepcopy(model)
    ddp = MultiGroupDistributedDataParallel(
        model,
        init_sync=False,
        accumulate_grads_in_fp32=True,
        reduce_grads_in_fp32=True,
        use_reduce_scatter=True,
    )
    ddp.configure_reduce_scatter_params({model.large_a, model.large_b})

    torch.manual_seed(100 + rank)
    x = torch.randn(4, d, device=device)
    ddp(x).pow(2).mean().backward()
    ddp.finalize_grad_reduce()

    reference(x).pow(2).mean().backward()
    expected = dict(
        zip(reference.state_dict().keys(), _reference_grads(reference, world_size))
    )

    for name, param in ddp.module.named_parameters():
        if name in {"large_a", "large_b"}:
            reduced_shard = getattr(
                param, "_olmo_ddp_reduced_grad_shard", None
            )
            assert reduced_shard is not None, f"missing reduced shard for {name}"
            flat_expected = expected[name].reshape(-1)
            local_numel = flat_expected.numel() // world_size
            expected_shard = flat_expected.narrow(
                0, rank * local_numel, local_numel
            )
            torch.testing.assert_close(
                reduced_shard, expected_shard, rtol=1e-5, atol=1e-6
            )
        else:
            assert not hasattr(param, "_olmo_ddp_reduced_grad_shard")
            main_grad = getattr(param, "_main_grad_fp32", None)
            assert main_grad is not None
            torch.testing.assert_close(
                main_grad, expected[name], rtol=1e-5, atol=1e-6
            )

    assert any(bucket.reduce_scatter for bucket in ddp._grad_buckets)
    assert any(not bucket.reduce_scatter for bucket in ddp._grad_buckets)
    with pytest.raises(RuntimeError, match="before the first forward"):
        ddp.configure_reduce_scatter_params({model.large_a, model.large_b})


def _run_reduce_scatter_no_sync_skipped_param_preserved(d: int):
    device = get_default_device()
    rank, world_size = dist.get_rank(), dist.get_world_size()

    seed_all(0)
    model = SelectiveModel(d).to(device)
    reference = copy.deepcopy(model)
    ddp = MultiGroupDistributedDataParallel(
        model,
        init_sync=False,
        accumulate_grads_in_fp32=True,
        reduce_grads_in_fp32=True,
        use_reduce_scatter=True,
    )
    ddp.configure_reduce_scatter_params(
        {model.fc_a.weight, model.fc_b.weight}
    )

    torch.manual_seed(100 + rank)
    x_a = torch.randn(4, d, device=device)
    x_b = torch.randn(4, d, device=device)

    with ddp.no_sync():
        ddp(x_a, use_a=True).pow(2).mean().backward()
    ddp(x_b, use_a=False).pow(2).mean().backward()
    ddp.finalize_grad_reduce()

    reference(x_a, use_a=True).pow(2).mean().backward()
    reference(x_b, use_a=False).pow(2).mean().backward()
    expected = dict(
        zip(reference.state_dict().keys(), _reference_grads(reference, world_size))
    )

    for name, param in ddp.module.named_parameters():
        if name.endswith("weight"):
            reduced_shard = getattr(
                param, "_olmo_ddp_reduced_grad_shard", None
            )
            assert reduced_shard is not None, f"missing reduced shard for {name}"
            flat_expected = expected[name].reshape(-1)
            local_numel = flat_expected.numel() // world_size
            torch.testing.assert_close(
                reduced_shard,
                flat_expected.narrow(0, rank * local_numel, local_numel),
                rtol=1e-5,
                atol=1e-6,
            )
        else:
            main_grad = getattr(param, "_main_grad_fp32", None)
            assert main_grad is not None
            torch.testing.assert_close(
                main_grad, expected[name], rtol=1e-5, atol=1e-6
            )


def _build_optimizer_test_stack(
    d: int, *, use_reduce_scatter: bool
) -> tuple[
    MultiGroupDistributedDataParallel,
    OLMoDDPOptimizer,
    FP8WeightStore,
]:
    device = get_default_device()
    seed_all(0)
    model = MixedReductionModel(d).to(device=device, dtype=torch.bfloat16)
    ddp = MultiGroupDistributedDataParallel(
        model,
        init_sync=False,
        accumulate_grads_in_fp32=True,
        reduce_grads_in_fp32=True,
        # Force the two large weights into separate buckets. This exercises
        # repeated use of the shared packing scratch while earlier async
        # collectives may still be in flight.
        bucket_cap_mb=0.01,
        use_reduce_scatter=use_reduce_scatter,
    )

    dense_mesh = init_device_mesh(
        device.type,
        (dist.get_world_size(),),
        mesh_dim_names=("dp",),
    )
    fp8_anchor = nn.Parameter(
        torch.empty(d, d, device=device, dtype=torch.bfloat16)
    )
    nn.init.normal_(fp8_anchor)
    fp8_store = FP8WeightStore(
        logical_name="logical_fp8",
        logical_shape=tuple(fp8_anchor.shape),
        cache_specs=(FP8WeightCacheSpec("rhs", lambda weight: weight),),
        anchor_param=fp8_anchor,
        optimizer_enabled=True,
        prequantizer=_fake_fp8_prequantizer,
    )
    named_params = dict(ddp.named_parameters())
    named_params["logical_fp8"] = fp8_store

    optimizer = OLMoDDPOptimizer(
        [
            {
                "named_params": named_params,
                "pg": "dp",
            }
        ],
        world_mesh={"dense": dense_mesh, "moe": None},
        dp_group=dist.group.WORLD,
        model_has_grad_accum_fp32_buffer=True,
        use_distributed=True,
        lr=1e-3,
        max_grad_norm=1.0,
    )
    if use_reduce_scatter:
        ddp.configure_reduce_scatter_params(
            optimizer.normal_params_with_sharded_optimizer_state()
        )
    return ddp, optimizer, fp8_store


def _run_optimizer_step_reduce_scatter_matches_all_reduce(d: int):
    rank = dist.get_rank()
    ddp_ar, optim_ar, fp8_ar = _build_optimizer_test_stack(
        d, use_reduce_scatter=False
    )
    ddp_rs, optim_rs, fp8_rs = _build_optimizer_test_stack(
        d, use_reduce_scatter=True
    )
    # The new option is owned by MultiGroupDDP. The optimizer-wide legacy path
    # stays disabled, so FP8WeightStore reduction continues through its existing
    # helper in _copy_model_grads_to_main_grads().
    assert optim_ar._use_reduce_scatter_grads is False
    assert optim_rs._use_reduce_scatter_grads is False

    sharded_params = optim_rs.normal_params_with_sharded_optimizer_state()
    assert ddp_rs.module.large_a in sharded_params
    assert ddp_rs.module.large_b in sharded_params
    assert ddp_rs.module.small_bias not in sharded_params
    assert fp8_rs not in sharded_params
    assert sum(bucket.reduce_scatter for bucket in ddp_rs._grad_buckets) == 2

    for step in range(2):
        torch.manual_seed(1000 + 10 * step + rank)
        x = torch.randn(4, d, device=get_default_device(), dtype=torch.bfloat16)
        fp8_grad = torch.randn(
            d,
            d,
            device=get_default_device(),
            dtype=torch.float32,
        )
        fp8_ar.main_grad_fp32 = fp8_grad.clone()
        fp8_rs.main_grad_fp32 = fp8_grad.clone()

        for ddp, optimizer in ((ddp_ar, optim_ar), (ddp_rs, optim_rs)):
            ddp(x).float().pow(2).mean().backward()
            ddp.finalize_grad_reduce()
            optimizer.step()
            ddp.zero_grad(set_to_none=False)

        for (name_ar, param_ar), (name_rs, param_rs) in zip(
            ddp_ar.module.named_parameters(), ddp_rs.module.named_parameters()
        ):
            assert name_ar == name_rs
            torch.testing.assert_close(
                param_rs, param_ar, rtol=0, atol=0, msg=lambda msg: f"{name_ar}: {msg}"
            )

        assert optim_ar.latest_grad_norm is not None
        assert optim_rs.latest_grad_norm is not None
        torch.testing.assert_close(
            optim_rs.latest_grad_norm,
            optim_ar.latest_grad_norm,
            rtol=1e-5,
            atol=1e-6,
        )

        assert optim_ar.states.keys() == optim_rs.states.keys()
        for state_name in optim_ar.states:
            torch.testing.assert_close(
                optim_rs.states[state_name].to_local(),
                optim_ar.states[state_name].to_local(),
                rtol=1e-5,
                atol=1e-6,
                msg=lambda msg, state_name=state_name: f"{state_name}: {msg}",
            )

        torch.testing.assert_close(
            fp8_rs.require_cache("rhs").mat_b_q,
            fp8_ar.require_cache("rhs").mat_b_q,
            rtol=0,
            atol=0,
        )


def _run_reduce_scatter_uses_parameter_process_group(d: int):
    device = get_default_device()
    global_rank = dist.get_rank()

    subgroup_01 = dist.new_group([0, 1])
    subgroup_23 = dist.new_group([2, 3])
    expert_group = subgroup_01 if global_rank < 2 else subgroup_23

    seed_all(0)
    model = TwoGroupModel(d).to(device)
    reference = copy.deepcopy(model)

    ddp = MultiGroupDistributedDataParallel(
        model,
        init_sync=False,
        process_group=dist.group.WORLD,
        param_process_group_fn=lambda name, _param: (
            expert_group if name == "expert_weight" else dist.group.WORLD
        ),
        accumulate_grads_in_fp32=True,
        reduce_grads_in_fp32=True,
        use_reduce_scatter=True,
    )
    ddp.configure_reduce_scatter_params(
        {model.dense_weight, model.expert_weight}
    )

    torch.manual_seed(100 + global_rank)
    x = torch.randn(4, d, device=device)
    ddp(x).pow(2).mean().backward()
    ddp.finalize_grad_reduce()

    reference(x).pow(2).mean().backward()
    for name, ref_param in reference.named_parameters():
        assert ref_param.grad is not None
        group = expert_group if name == "expert_weight" else dist.group.WORLD
        expected = ref_param.grad.detach().clone()
        dist.all_reduce(expected, group=group)
        expected /= dist.get_world_size(group)

        param = dict(ddp.module.named_parameters())[name]
        reduced_shard = getattr(
            param, "_olmo_ddp_reduced_grad_shard", None
        )
        assert reduced_shard is not None
        group_rank = dist.get_rank(group=group)
        local_numel = expected.numel() // dist.get_world_size(group)
        torch.testing.assert_close(
            reduced_shard,
            expected.reshape(-1).narrow(
                0, group_rank * local_numel, local_numel
            ),
            rtol=1e-5,
            atol=1e-6,
        )


def _build_two_group_optimizer_stack(
    d: int,
    *,
    use_reduce_scatter: bool,
    dense_mesh,
    moe_mesh,
) -> tuple[MultiGroupDistributedDataParallel, OLMoDDPOptimizer]:
    device = get_default_device()
    seed_all(0)
    model = TwoGroupModel(d).to(device=device, dtype=torch.bfloat16)
    # Dense weights are identical across the full DP group. Expert weights are
    # distinct across EP-MP and identical across their EP-DP replica group.
    seed_all(100 + moe_mesh["ep_mp"].get_local_rank())
    nn.init.normal_(model.expert_weight)
    expert_group = moe_mesh["ep_dp"].get_group()
    ddp = MultiGroupDistributedDataParallel(
        model,
        init_sync=False,
        process_group=dist.group.WORLD,
        param_process_group_fn=lambda name, _param: (
            expert_group if name == "expert_weight" else dist.group.WORLD
        ),
        accumulate_grads_in_fp32=True,
        reduce_grads_in_fp32=True,
        use_reduce_scatter=use_reduce_scatter,
    )
    optimizer = OLMoDDPOptimizer(
        [
            {
                "named_params": {"dense_weight": model.dense_weight},
                "pg": "dp",
            },
            {
                "named_params": {"expert_weight": model.expert_weight},
                "pg": "ep_dp",
            },
        ],
        world_mesh={"dense": dense_mesh, "moe": moe_mesh},
        dp_group=dist.group.WORLD,
        ep_dp_group=expert_group,
        model_has_grad_accum_fp32_buffer=True,
        use_distributed=True,
        lr=1e-3,
        max_grad_norm=1.0,
    )
    if use_reduce_scatter:
        ddp.configure_reduce_scatter_params(
            optimizer.normal_params_with_sharded_optimizer_state()
        )
    return ddp, optimizer


def _run_ep_optimizer_reduce_scatter_matches_all_reduce(d: int):
    rank = dist.get_rank()
    device = get_default_device()
    dense_mesh = init_device_mesh(
        device.type,
        (dist.get_world_size(),),
        mesh_dim_names=("dp",),
    )
    moe_mesh = init_device_mesh(
        device.type,
        (2, 2),
        mesh_dim_names=("ep_dp", "ep_mp"),
    )

    ddp_ar, optim_ar = _build_two_group_optimizer_stack(
        d,
        use_reduce_scatter=False,
        dense_mesh=dense_mesh,
        moe_mesh=moe_mesh,
    )
    ddp_rs, optim_rs = _build_two_group_optimizer_stack(
        d,
        use_reduce_scatter=True,
        dense_mesh=dense_mesh,
        moe_mesh=moe_mesh,
    )

    assert all(bucket.reduce_scatter for bucket in ddp_rs._grad_buckets)
    assert {dist.get_world_size(bucket.process_group) for bucket in ddp_rs._grad_buckets} == {2, 4}

    for step in range(2):
        torch.manual_seed(2000 + 10 * step + rank)
        x = torch.randn(4, d, device=device, dtype=torch.bfloat16)
        for ddp, optimizer in ((ddp_ar, optim_ar), (ddp_rs, optim_rs)):
            ddp(x).float().pow(2).mean().backward()
            ddp.finalize_grad_reduce()
            optimizer.step()
            ddp.zero_grad(set_to_none=False)

        for (name_ar, param_ar), (name_rs, param_rs) in zip(
            ddp_ar.module.named_parameters(), ddp_rs.module.named_parameters()
        ):
            assert name_ar == name_rs
            torch.testing.assert_close(
                param_rs,
                param_ar,
                rtol=0,
                atol=0,
                msg=lambda msg: f"{name_ar}: {msg}",
            )

        assert optim_ar.latest_grad_norm is not None
        assert optim_rs.latest_grad_norm is not None
        torch.testing.assert_close(
            optim_rs.latest_grad_norm,
            optim_ar.latest_grad_norm,
            rtol=1e-5,
            atol=1e-6,
        )
        for state_name in optim_ar.states:
            torch.testing.assert_close(
                optim_rs.states[state_name].to_local(),
                optim_ar.states[state_name].to_local(),
                rtol=1e-5,
                atol=1e-6,
                msg=lambda msg, state_name=state_name: f"{state_name}: {msg}",
            )


@pytest.mark.parametrize("backend", BACKENDS)
def test_no_sync_skipped_param_grad_preserved(backend: str):
    run_distributed_test(
        _run_no_sync_skipped_param_grad_preserved,
        backend=backend,
        func_args=(16,),
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_fp32_hooks_skip_ignored_params(backend: str):
    run_distributed_test(
        _run_fp32_hooks_skip_ignored_params,
        backend=backend,
        func_args=(16,),
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_mixed_reduce_scatter_all_reduce_matches_reference(backend: str):
    run_distributed_test(
        _run_mixed_reduce_scatter_all_reduce_matches_reference,
        backend=backend,
        func_args=(16,),
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_reduce_scatter_no_sync_skipped_param_preserved(backend: str):
    run_distributed_test(
        _run_reduce_scatter_no_sync_skipped_param_preserved,
        backend=backend,
        func_args=(16,),
    )


@pytest.mark.parametrize(
    "backend",
    [pytest.param("cuda:nccl,cpu:gloo", marks=MULTI_GPU_MARKS)],
)
def test_optimizer_step_reduce_scatter_matches_all_reduce(backend: str):
    run_distributed_test(
        _run_optimizer_step_reduce_scatter_matches_all_reduce,
        backend=backend,
        func_args=(64,),
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_reduce_scatter_uses_parameter_process_group(backend: str):
    run_distributed_test(
        _run_reduce_scatter_uses_parameter_process_group,
        world_size=4,
        backend=backend,
        func_args=(16,),
    )


@pytest.mark.parametrize(
    "backend",
    [pytest.param("cuda:nccl,cpu:gloo", marks=MULTI_GPU_MARKS)],
)
@pytest.mark.skipif(
    torch.cuda.device_count() < 4,
    reason="EP-DP=2 x EP-MP=2 parity requires four GPUs",
)
def test_ep_optimizer_reduce_scatter_matches_all_reduce(backend: str):
    run_distributed_test(
        _run_ep_optimizer_reduce_scatter_matches_all_reduce,
        world_size=4,
        backend=backend,
        func_args=(64,),
    )
