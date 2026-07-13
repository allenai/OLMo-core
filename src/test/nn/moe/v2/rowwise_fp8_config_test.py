from types import SimpleNamespace

import pytest
import torch

from olmo_core.config import DType
from olmo_core.kernels.scaled_grouped_mm import ScaledGroupedMMPrequantizedRHS
from olmo_core.nn.fp8_weight import FP8WeightCacheSpec, FP8WeightStore
from olmo_core.nn.moe.v2.fp8 import (
    MoERowwiseFP8Config,
    normalize_rowwise_fp8_config,
    refresh_rowwise_fp8_cache,
)
from olmo_core.nn.moe.v2.routed_experts import (
    RoutedExperts,
    _swiglu_backward_grad_up_gate_impl,
    swiglu_backward_grad_up_gate,
)
from olmo_core.nn.parallel.distributed import MultiGroupDistributedDataParallel


def test_rowwise_fp8_config_validate_block_size():
    cfg = MoERowwiseFP8Config(enabled=True, block_size=16)
    with pytest.raises(ValueError, match="block_size=32"):
        cfg.validate()


def test_rowwise_fp8_config_fail_closed_on_sm90(monkeypatch):
    cfg = MoERowwiseFP8Config(enabled=True, block_size=32)

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _device=0: (9, 0))

    with pytest.raises(RuntimeError, match="pre-SM100"):
        cfg.assert_runtime_supported()


def test_rowwise_fp8_config_requires_scaled_grouped_mm(monkeypatch):
    cfg = MoERowwiseFP8Config(enabled=True, block_size=32)

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _device=0: (10, 0))
    monkeypatch.delattr(torch.nn.functional, "scaled_grouped_mm", raising=False)

    with pytest.raises(RuntimeError, match="scaled_grouped_mm"):
        cfg.assert_runtime_supported()


def test_rowwise_fp8_normalize_from_dict():
    cfg = normalize_rowwise_fp8_config({"enabled": True, "block_size": 32})
    assert isinstance(cfg, MoERowwiseFP8Config)
    assert cfg.enabled is True
    assert cfg.block_size == 32
    assert cfg.fp8_only_params is True


def test_rowwise_fp8_allows_disabling_fp8_only_params():
    cfg = normalize_rowwise_fp8_config(
        {"enabled": True, "block_size": 32, "fp8_only_params": False}
    )
    assert isinstance(cfg, MoERowwiseFP8Config)
    assert cfg.fp8_only_params is False


def test_routed_experts_accepts_rowwise_fp8_dict():
    module = RoutedExperts(
        d_model=512,
        hidden_size=1024,
        num_experts=4,
        bias=False,
        dtype=DType.float32,
        rowwise_fp8={"enabled": True, "block_size": 32},  # type: ignore[arg-type]
        init_device="cpu",
    )
    assert isinstance(module.rowwise_fp8, MoERowwiseFP8Config)
    assert module.rowwise_fp8 is not None and module.rowwise_fp8.enabled is True


def test_routed_experts_zero_grad_clears_mxfp8_store_grads():
    module = RoutedExperts(
        d_model=16,
        hidden_size=8,
        num_experts=2,
        bias=False,
        dtype=DType.bfloat16,
        rowwise_fp8={"enabled": True, "block_size": 32, "fp8_only_params": True},  # type: ignore[arg-type]
        init_device="cpu",
    )
    module._rowwise_fp8_up_gate_weight.grad_bf16 = torch.ones_like(module.w_up_gate)
    module._rowwise_fp8_down_weight.grad_bf16 = torch.ones_like(module.w_down)

    module.invalidate_rowwise_fp8_cache()
    assert module._rowwise_fp8_up_gate_weight.grad_bf16 is not None
    assert module._rowwise_fp8_down_weight.grad_bf16 is not None

    module.zero_grad(set_to_none=False)
    assert module._rowwise_fp8_up_gate_weight.grad_bf16 is not None
    assert module._rowwise_fp8_down_weight.grad_bf16 is not None
    assert not module._rowwise_fp8_up_gate_weight.grad_bf16.any()
    assert not module._rowwise_fp8_down_weight.grad_bf16.any()

    module.zero_grad(set_to_none=True)
    assert module._rowwise_fp8_up_gate_weight.grad_bf16 is None
    assert module._rowwise_fp8_down_weight.grad_bf16 is None


def test_routed_experts_mxfp8_weight_anchors_follow_to_empty():
    if not torch.cuda.is_available():
        return

    module = RoutedExperts(
        d_model=32,
        hidden_size=32,
        num_experts=2,
        bias=False,
        dtype=DType.float32,
        rowwise_fp8={"enabled": True, "block_size": 32, "fp8_only_params": True},  # type: ignore[arg-type]
        init_device="cpu",
    )
    module.to_empty(device=torch.device("cuda"))

    weights = dict(module.named_mxfp8_expert_weights())

    assert weights["w_up_gate"].anchor_param is module.w_up_gate
    assert weights["w_down"].anchor_param is module.w_down
    assert weights["w_up_gate"].device.type == "cuda"
    assert weights["w_down"].device.type == "cuda"


def test_routed_experts_fp8_only_disables_anchor_grads():
    module = RoutedExperts(
        d_model=16,
        hidden_size=8,
        num_experts=2,
        bias=False,
        dtype=DType.bfloat16,
        rowwise_fp8={"enabled": True, "block_size": 32, "fp8_only_params": True},  # type: ignore[arg-type]
        init_device="cpu",
    )

    assert module.w_up_gate.requires_grad
    assert module.w_down.requires_grad

    weights = dict(module.named_mxfp8_expert_weights())

    assert weights["w_up_gate"].optimizer_enabled
    assert weights["w_down"].optimizer_enabled
    assert not module.w_up_gate.requires_grad
    assert not module.w_down.requires_grad


def test_fp8_weight_store_release_anchor_storage_keeps_logical_shape():
    anchor = torch.nn.Parameter(torch.empty(2, 8, 16, dtype=torch.bfloat16))
    old_nbytes = anchor.untyped_storage().nbytes()
    rhs = ScaledGroupedMMPrequantizedRHS(
        mat_b_q=torch.empty(2, 16, 8, dtype=torch.float8_e4m3fn),
        scale_b=torch.empty(1, dtype=torch.float8_e8m0fnu),
        mat_b_shape=(2, 16, 8),
        mat_b_version=-1,
    )
    rhs_t = ScaledGroupedMMPrequantizedRHS(
        mat_b_q=torch.empty(2, 8, 16, dtype=torch.float8_e4m3fn),
        scale_b=torch.empty(1, dtype=torch.float8_e8m0fnu),
        mat_b_shape=(2, 8, 16),
        mat_b_version=-1,
    )
    weight = FP8WeightStore(
        logical_name="w_up_gate",
        logical_shape=tuple(anchor.shape),
        anchor_param=anchor,
        optimizer_enabled=True,
        prequantized_rhs=rhs,
        prequantized_rhs_for_dgrad=rhs_t,
    )

    weight.release_anchor_storage()

    assert weight.anchor_storage_released
    assert tuple(anchor.shape) == (2, 8, 16)
    assert anchor.stride() == (0, 0, 0)
    assert anchor.untyped_storage().nbytes() < old_nbytes
    assert not anchor.requires_grad


def test_routed_experts_refresh_does_not_read_released_anchors(monkeypatch):
    module = RoutedExperts(
        d_model=16,
        hidden_size=8,
        num_experts=2,
        bias=False,
        dtype=DType.bfloat16,
        rowwise_fp8={"enabled": True, "block_size": 32, "fp8_only_params": True},  # type: ignore[arg-type]
        init_device="cpu",
    )
    up_rhs = ScaledGroupedMMPrequantizedRHS(
        mat_b_q=torch.empty(2, 16, 16, dtype=torch.float8_e4m3fn),
        scale_b=torch.empty(1, dtype=torch.float8_e8m0fnu),
        mat_b_shape=(2, 16, 16),
        mat_b_version=-1,
    )
    up_rhs_t = ScaledGroupedMMPrequantizedRHS(
        mat_b_q=torch.empty(2, 16, 16, dtype=torch.float8_e4m3fn),
        scale_b=torch.empty(1, dtype=torch.float8_e8m0fnu),
        mat_b_shape=(2, 16, 16),
        mat_b_version=-1,
    )
    down_rhs = ScaledGroupedMMPrequantizedRHS(
        mat_b_q=torch.empty(2, 8, 16, dtype=torch.float8_e4m3fn),
        scale_b=torch.empty(1, dtype=torch.float8_e8m0fnu),
        mat_b_shape=(2, 8, 16),
        mat_b_version=-1,
    )
    down_rhs_t = ScaledGroupedMMPrequantizedRHS(
        mat_b_q=torch.empty(2, 16, 8, dtype=torch.float8_e4m3fn),
        scale_b=torch.empty(1, dtype=torch.float8_e8m0fnu),
        mat_b_shape=(2, 16, 8),
        mat_b_version=-1,
    )
    module._rowwise_fp8_up_gate_weight.prequantized_rhs = up_rhs
    module._rowwise_fp8_up_gate_weight.prequantized_rhs_for_dgrad = up_rhs_t
    module._rowwise_fp8_down_weight.prequantized_rhs = down_rhs
    module._rowwise_fp8_down_weight.prequantized_rhs_for_dgrad = down_rhs_t
    module.release_mxfp8_expert_anchor_storage()

    def _forbid_prequant(*args, **kwargs):
        del args, kwargs
        raise AssertionError("refresh should not prequantize from released bf16 anchors")

    monkeypatch.setattr(
        "olmo_core.nn.fp8_weight.prequantize_scaled_grouped_mm_rhs",
        _forbid_prequant,
    )

    module.refresh_rowwise_fp8_cache()

    assert module._rowwise_fp8_up_gate_prequant is up_rhs
    assert module._rowwise_fp8_up_gate_prequant_t is up_rhs_t
    assert module._rowwise_fp8_down_prequant is down_rhs
    assert module._rowwise_fp8_down_prequant_t is down_rhs_t


def test_routed_experts_syncs_mxfp8_store_grads_from_anchor_main_grad():
    module = RoutedExperts(
        d_model=16,
        hidden_size=8,
        num_experts=2,
        bias=False,
        dtype=DType.bfloat16,
        rowwise_fp8={"enabled": True, "block_size": 32, "fp8_only_params": True},  # type: ignore[arg-type]
        init_device="cpu",
    )
    up_main_grad = torch.full(module.w_up_gate.shape, 3.0, dtype=torch.float32)
    down_main_grad = torch.full(module.w_down.shape, -2.0, dtype=torch.float32)
    module.w_up_gate._main_grad_fp32 = up_main_grad  # type: ignore[attr-defined]
    module.w_down._main_grad_fp32 = down_main_grad  # type: ignore[attr-defined]

    module.sync_mxfp8_expert_weight_grads_from_anchor()

    assert module._rowwise_fp8_up_gate_weight.grad_bf16 is not None
    assert module._rowwise_fp8_down_weight.grad_bf16 is not None
    assert module._rowwise_fp8_up_gate_weight.grad_bf16.dtype == torch.bfloat16
    assert module._rowwise_fp8_down_weight.grad_bf16.dtype == torch.bfloat16
    torch.testing.assert_close(
        module._rowwise_fp8_up_gate_weight.grad_bf16,
        up_main_grad.to(torch.bfloat16),
    )
    torch.testing.assert_close(
        module._rowwise_fp8_down_weight.grad_bf16,
        down_main_grad.to(torch.bfloat16),
    )


def test_fp8_weight_store_refresh_from_logical_weight_uses_explicit_layout_specs(monkeypatch):
    calls = []

    def _fake_prequant(mat_b: torch.Tensor, *, check_mat_b_version: bool = True):
        del check_mat_b_version
        calls.append(tuple(mat_b.shape))
        return ScaledGroupedMMPrequantizedRHS(
            mat_b_q=torch.empty_like(mat_b, dtype=torch.float8_e4m3fn),
            scale_b=torch.empty((1,), dtype=torch.float8_e8m0fnu),
            mat_b_shape=tuple(mat_b.shape),
            mat_b_version=-1,
        )

    monkeypatch.setattr(
        "olmo_core.nn.fp8_weight.prequantize_scaled_grouped_mm_rhs",
        _fake_prequant,
    )

    up_gate = FP8WeightStore(
        logical_name="w_up_gate",
        logical_shape=(2, 8, 16),
        cache_specs=(
            FP8WeightCacheSpec("rhs", lambda w: w.transpose(1, 2)),
            FP8WeightCacheSpec("rhs_for_dgrad", lambda w: w),
        ),
    )
    up_gate.refresh_from_logical_weight(torch.empty(2, 8, 16))
    down = FP8WeightStore(
        logical_name="w_down",
        logical_shape=(2, 4, 16),
        cache_specs=(
            FP8WeightCacheSpec("rhs", lambda w: w),
            FP8WeightCacheSpec("rhs_for_dgrad", lambda w: w.transpose(1, 2)),
        ),
    )
    down.refresh_from_logical_weight(torch.empty(2, 4, 16))
    shared_up_gate = FP8WeightStore(
        logical_name="shared_experts.w_up_gate",
        logical_shape=(8, 16),
        cache_specs=(
            FP8WeightCacheSpec("rhs", lambda w: w.unsqueeze(0)),
            FP8WeightCacheSpec("rhs_for_dgrad", lambda w: w.transpose(0, 1).unsqueeze(0)),
        ),
    )
    shared_up_gate.refresh_from_logical_weight(torch.empty(8, 16))

    assert calls == [
        (2, 16, 8),
        (2, 8, 16),
        (2, 4, 16),
        (2, 16, 4),
        (1, 8, 16),
        (1, 16, 8),
    ]


def test_multigroup_ddp_zeroes_module_logical_grads():
    calls = []

    class _FakeModule(torch.nn.Module):
        def zero_mxfp8_expert_weight_grads(self, *, set_to_none: bool) -> None:
            calls.append(set_to_none)

    ddp = MultiGroupDistributedDataParallel.__new__(MultiGroupDistributedDataParallel)
    torch.nn.Module.__init__(ddp)
    ddp.module = _FakeModule()

    ddp._zero_module_logical_grads(set_to_none=False)
    ddp._zero_module_logical_grads(set_to_none=True)

    assert calls == [False, True]


def test_multigroup_ddp_syncs_module_logical_grads():
    calls = []

    class _FakeModule(torch.nn.Module):
        def sync_mxfp8_expert_weight_grads_from_anchor(self) -> None:
            calls.append("sync")

    ddp = MultiGroupDistributedDataParallel.__new__(MultiGroupDistributedDataParallel)
    torch.nn.Module.__init__(ddp)
    ddp.module = _FakeModule()

    ddp._sync_module_logical_grads_from_anchor()

    assert calls == ["sync"]


def test_block_refresh_rowwise_fp8_cache_honors_routed_config_without_block_config():
    seen = {"refresh": 0, "invalidate": 0}

    class _FakeRoutedExperts:
        rowwise_fp8 = MoERowwiseFP8Config(enabled=True)

        def refresh_rowwise_fp8_cache(self) -> None:
            seen["refresh"] += 1

        def invalidate_rowwise_fp8_cache(self) -> None:
            seen["invalidate"] += 1

    block = SimpleNamespace(
        rowwise_fp8=None,
        routed_experts=_FakeRoutedExperts(),
        shared_experts=None,
    )

    refresh_rowwise_fp8_cache(block)

    assert seen == {"refresh": 1, "invalidate": 0}
    assert block._shared_rowwise_fp8_up_prequant is None
    assert block._shared_rowwise_fp8_down_prequant is None


def test_block_refresh_rowwise_fp8_cache_disabled_invalidates_without_refresh():
    seen = {"refresh": 0, "invalidate": 0}

    class _FakeRoutedExperts:
        rowwise_fp8 = MoERowwiseFP8Config(enabled=False)

        def refresh_rowwise_fp8_cache(self) -> None:
            seen["refresh"] += 1

        def invalidate_rowwise_fp8_cache(self) -> None:
            seen["invalidate"] += 1

    block = SimpleNamespace(
        rowwise_fp8=MoERowwiseFP8Config(enabled=False),
        routed_experts=_FakeRoutedExperts(),
        shared_experts=None,
        _shared_rowwise_fp8_up_prequant=object(),
        _shared_rowwise_fp8_down_prequant=object(),
        _shared_rowwise_fp8_up_prequant_t=object(),
        _shared_rowwise_fp8_down_prequant_t=object(),
        _shared_rowwise_fp8_weight_versions=(1, 2),
    )

    refresh_rowwise_fp8_cache(block)

    assert seen == {"refresh": 0, "invalidate": 1}
    assert block._shared_rowwise_fp8_up_prequant is None
    assert block._shared_rowwise_fp8_down_prequant is None
    assert block._shared_rowwise_fp8_up_prequant_t is None
    assert block._shared_rowwise_fp8_down_prequant_t is None
    assert block._shared_rowwise_fp8_weight_versions is None


def test_block_refresh_rowwise_fp8_cache_shared_only_does_not_refresh_routed():
    seen = {"refresh": 0, "invalidate": 0}

    class _FakeRoutedExperts:
        rowwise_fp8 = MoERowwiseFP8Config(enabled=False)

        def refresh_rowwise_fp8_cache(self) -> None:
            seen["refresh"] += 1

        def invalidate_rowwise_fp8_cache(self) -> None:
            seen["invalidate"] += 1

    block = SimpleNamespace(
        rowwise_fp8=MoERowwiseFP8Config(enabled=True),
        routed_experts=_FakeRoutedExperts(),
        shared_experts=None,
    )

    refresh_rowwise_fp8_cache(block)

    assert seen == {"refresh": 0, "invalidate": 1}
    assert block._shared_rowwise_fp8_up_prequant is None
    assert block._shared_rowwise_fp8_down_prequant is None


def test_routed_experts_refresh_marks_owned_prequant_caches_versionless(monkeypatch):
    if not torch.cuda.is_available():
        return

    module = RoutedExperts(
        d_model=512,
        hidden_size=256,
        num_experts=2,
        bias=False,
        dtype=DType.float32,
        rowwise_fp8={"enabled": True, "block_size": 32},  # type: ignore[arg-type]
        init_device="cuda",
    )

    def _fake_prequant(mat_b: torch.Tensor, *, check_mat_b_version: bool = True):
        return ScaledGroupedMMPrequantizedRHS(
            mat_b_q=torch.empty_like(mat_b, dtype=torch.float8_e4m3fn),
            scale_b=torch.empty((1,), dtype=torch.float8_e8m0fnu),
            mat_b_shape=tuple(mat_b.shape),
            mat_b_version=0 if check_mat_b_version else -1,
        )

    monkeypatch.setattr(
        "olmo_core.nn.fp8_weight.prequantize_scaled_grouped_mm_rhs",
        _fake_prequant,
    )

    module.refresh_rowwise_fp8_cache()

    assert module._rowwise_fp8_up_gate_prequant is not None
    assert module._rowwise_fp8_up_gate_prequant.mat_b_version == -1
    assert module._rowwise_fp8_up_gate_prequant_t is not None
    assert module._rowwise_fp8_up_gate_prequant_t.mat_b_version == -1
    assert module._rowwise_fp8_down_prequant is not None
    assert module._rowwise_fp8_down_prequant.mat_b_version == -1
    assert module._rowwise_fp8_down_prequant_t is not None
    assert module._rowwise_fp8_down_prequant_t.mat_b_version == -1


def test_routed_experts_forward_rowwise_fp8_uses_cached_prequantized_rhs(monkeypatch):
    module = RoutedExperts(
        d_model=512,
        hidden_size=256,
        num_experts=2,
        bias=False,
        dtype=DType.float32,
        rowwise_fp8={"enabled": True, "block_size": 32},  # type: ignore[arg-type]
        init_device="cpu",
    )

    up_q = module.w_up_gate.transpose(1, 2).to(torch.float8_e4m3fn)
    up_t_q = module.w_up_gate.to(torch.float8_e4m3fn)
    up_s = torch.ones((module.num_experts, 1), dtype=torch.float8_e8m0fnu)
    down_q = module.w_down.to(torch.float8_e4m3fn)
    down_t_q = module.w_down.transpose(1, 2).to(torch.float8_e4m3fn)
    down_s = torch.ones((module.num_experts, 1), dtype=torch.float8_e8m0fnu)

    module._rowwise_fp8_up_gate_prequant = ScaledGroupedMMPrequantizedRHS(
        mat_b_q=up_q,
        scale_b=up_s,
        mat_b_shape=tuple(module.w_up_gate.transpose(1, 2).shape),
        mat_b_version=int(module.w_up_gate._version),
    )
    module._rowwise_fp8_up_gate_prequant_t = ScaledGroupedMMPrequantizedRHS(
        mat_b_q=up_t_q,
        scale_b=up_s,
        mat_b_shape=tuple(module.w_up_gate.shape),
        mat_b_version=int(module.w_up_gate._version),
    )
    module._rowwise_fp8_down_prequant = ScaledGroupedMMPrequantizedRHS(
        mat_b_q=down_q,
        scale_b=down_s,
        mat_b_shape=tuple(module.w_down.shape),
        mat_b_version=int(module.w_down._version),
    )
    module._rowwise_fp8_down_prequant_t = ScaledGroupedMMPrequantizedRHS(
        mat_b_q=down_t_q,
        scale_b=down_s,
        mat_b_shape=tuple(module.w_down.transpose(1, 2).shape),
        mat_b_version=int(module.w_down._version),
    )
    module._rowwise_fp8_weight_versions = (
        int(module.w_up_gate._version),
        int(module.w_down._version),
    )
    module._rowwise_fp8_up_gate_weight.prequantized_rhs = module._rowwise_fp8_up_gate_prequant
    module._rowwise_fp8_up_gate_weight.prequantized_rhs_for_dgrad = (
        module._rowwise_fp8_up_gate_prequant_t
    )
    module._rowwise_fp8_down_weight.prequantized_rhs = module._rowwise_fp8_down_prequant
    module._rowwise_fp8_down_weight.prequantized_rhs_for_dgrad = module._rowwise_fp8_down_prequant_t

    seen = {"calls": 0, "prequantized_lhs_count": 0}

    def _stub_scaled_grouped_mm_q_fp8_weight(
        mat_a: torch.Tensor,
        grad_anchor: torch.Tensor,
        *,
        offs: torch.Tensor,
        input_grad_out: torch.Tensor | None = None,
        use_fast_accum: bool = True,
        prequantized_lhs=None,
        prequantized_rhs=None,
        prequantized_rhs_for_dgrad=None,
        wgrad_sink=None,
        wgrad_sink_transpose_last2: bool = False,
        wgrad_sink_squeeze_first_dim: bool = False,
    ) -> torch.Tensor:
        if prequantized_lhs is not None:
            seen["prequantized_lhs_count"] += 1
        del (
            offs,
            input_grad_out,
            use_fast_accum,
            prequantized_lhs,
            prequantized_rhs_for_dgrad,
            wgrad_sink,
            wgrad_sink_transpose_last2,
            wgrad_sink_squeeze_first_dim,
        )
        seen["calls"] += 1
        assert prequantized_rhs is not None
        return torch.zeros(
            (mat_a.shape[0], grad_anchor.shape[-1]), dtype=mat_a.dtype, device=mat_a.device
        )

    def _forbid_forward_time_prequant(*args, **kwargs):
        del args, kwargs
        raise AssertionError("routed rowwise FP8 forward must not refresh/prequantize weights")

    monkeypatch.setattr(
        "olmo_core.nn.moe.v2.routed_experts.scaled_grouped_mm_q_fp8_weight",
        _stub_scaled_grouped_mm_q_fp8_weight,
    )
    monkeypatch.setattr(
        "olmo_core.nn.fp8_weight.prequantize_scaled_grouped_mm_rhs",
        _forbid_forward_time_prequant,
    )

    x = torch.randn(8, 512, dtype=torch.float32)
    batch = torch.tensor([4, 4], dtype=torch.int32)
    input_q = x.to(torch.float8_e4m3fn)
    input_scales = torch.ones((x.shape[0], x.shape[1] // 32), dtype=torch.float8_e8m0fnu)
    _ = module._forward_rowwise_fp8(
        x,
        batch,
        prequantized_input_q=input_q,
        prequantized_input_scales=input_scales,
    )
    assert seen["calls"] == 2
    assert seen["prequantized_lhs_count"] == 2


def test_swiglu_backward_grad_up_gate_compiled_helper_matches_eager(monkeypatch):
    if not torch.cuda.is_available():
        return

    monkeypatch.setenv("OLMO_MXFP8_COMPILE_SWIGLU_BWD", "1")
    torch.manual_seed(123)
    up_gate = torch.randn(16, 128, device="cuda", dtype=torch.bfloat16)
    grad_h = torch.randn(16, 64, device="cuda", dtype=torch.bfloat16)

    actual = swiglu_backward_grad_up_gate(up_gate, grad_h)
    expected = _swiglu_backward_grad_up_gate_impl(up_gate, grad_h)

    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)
