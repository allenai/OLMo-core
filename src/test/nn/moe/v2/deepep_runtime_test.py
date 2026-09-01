from types import SimpleNamespace
from typing import Any

import pytest
import torch

import olmo_core.nn.moe.v2.ep_deepep_v2 as deepep_v2


def _stub_block(
    *,
    ep_pg: object,
    runtime_cache: dict,
    fp8_enabled: bool = False,
) -> Any:
    deepep_config = SimpleNamespace(
        path="/fake/deepep",
        num_sms=8,
        num_qps=2,
        num_allocated_qps=2,
        expert_alignment=1,
        async_mode=False,
        prefer_overlap_with_compute=True,
        allow_hybrid_mode=True,
        allow_multiple_reduction=True,
    )
    return SimpleNamespace(
        ep_pg=ep_pg,
        ep=SimpleNamespace(capacity_factor=1.25, deepep=deepep_config),
        rowwise_fp8=(SimpleNamespace(enabled=True) if fp8_enabled else None),
        routed_experts_router=SimpleNamespace(num_experts=8),
        num_local_routed_experts=2,
        _deepep_v2_runtime_cache=runtime_cache,
        _deepep_v2_runtime=None,
    )


def test_deepep_runtime_negotiates_capacity_only_on_cache_miss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeElasticBuffer:
        instances = []

        def __init__(self, process_group, **kwargs):
            self.process_group = process_group
            self.kwargs = kwargs
            self.instances.append(self)

        def dispatch_expanded_into(self):
            raise NotImplementedError

        def dispatch_cached_expanded_into(self):
            raise NotImplementedError

    global_capacity_requests = []

    def fake_global_capacity(_block, requested_tokens, _device):
        global_capacity_requests.append(requested_tokens)
        return 96

    monkeypatch.setattr(
        deepep_v2,
        "_import_deepep",
        lambda _path: SimpleNamespace(ElasticBuffer=FakeElasticBuffer),
    )
    monkeypatch.setattr(
        deepep_v2,
        "_global_num_max_tokens_per_rank",
        fake_global_capacity,
    )

    ep_pg = object()
    shared_cache: dict = {}
    first_block = _stub_block(ep_pg=ep_pg, runtime_cache=shared_cache)
    second_block = _stub_block(ep_pg=ep_pg, runtime_cache=shared_cache)

    runtime = deepep_v2._get_deepep_v2_runtime(
        first_block,
        local_tokens=64,
        hidden=256,
        top_k=2,
        device=torch.device("cuda"),
    )
    cached_runtime = deepep_v2._get_deepep_v2_runtime(
        second_block,
        local_tokens=64,
        hidden=256,
        top_k=2,
        device=torch.device("cuda"),
    )

    assert cached_runtime is runtime
    assert second_block._deepep_v2_runtime is runtime
    assert global_capacity_requests == [64]
    assert len(FakeElasticBuffer.instances) == 1
    assert runtime.num_max_tokens_per_rank == 96


def test_deepep_cached_runtime_rejects_local_capacity_growth_without_collective(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ep_pg = object()
    block = _stub_block(ep_pg=ep_pg, runtime_cache={})
    key = deepep_v2._runtime_key(block, hidden=256, top_k=2)
    runtime: Any = SimpleNamespace(num_max_tokens_per_rank=96)
    block._deepep_v2_runtime_cache[key] = runtime

    def unexpected_collective(*_args, **_kwargs):
        raise AssertionError("cached runtime must not renegotiate capacity")

    monkeypatch.setattr(
        deepep_v2,
        "_global_num_max_tokens_per_rank",
        unexpected_collective,
    )

    with pytest.raises(RuntimeError, match="local_requested=100, capacity=96"):
        deepep_v2._get_deepep_v2_runtime(
            block,
            local_tokens=100,
            hidden=256,
            top_k=2,
            device=torch.device("cuda"),
        )


def test_deepep_static_source_capacity_skips_capacity_negotiation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeElasticBuffer:
        def __init__(self, process_group, **kwargs):
            self.process_group = process_group
            self.kwargs = kwargs

        def dispatch_expanded_into(self):
            raise NotImplementedError

        def dispatch_cached_expanded_into(self):
            raise NotImplementedError

    def unexpected_collective(*_args, **_kwargs):
        raise AssertionError("static DeepEP prewarm must not negotiate capacity")

    process_group_warmups = []

    monkeypatch.setattr(
        deepep_v2,
        "_import_deepep",
        lambda _path: SimpleNamespace(ElasticBuffer=FakeElasticBuffer),
    )
    monkeypatch.setattr(
        deepep_v2,
        "_global_num_max_tokens_per_rank",
        unexpected_collective,
    )
    monkeypatch.setattr(
        deepep_v2,
        "_warm_deepep_v2_process_group",
        lambda block, device: process_group_warmups.append((block, device)),
    )

    block = _stub_block(ep_pg=object(), runtime_cache={})
    runtime = deepep_v2.prewarm_deepep_v2_runtime(
        block,
        max_local_tokens=64,
        hidden=256,
        top_k=2,
        device=torch.device("cuda"),
    )

    assert runtime.num_max_tokens_per_rank == 64
    assert runtime.buffer.kwargs["num_max_tokens_per_rank"] == 64
    assert block._deepep_v2_runtime is runtime
    assert process_group_warmups == [(block, torch.device("cuda"))]


def test_deepep_source_capacity_is_not_route_capacity() -> None:
    assert deepep_v2._requested_num_max_tokens_per_rank(64) == 64
    with pytest.raises(ValueError, match="positive source-token capacity"):
        deepep_v2._requested_num_max_tokens_per_rank(0)


@pytest.mark.parametrize("hidden", [128, 256, 384, 1024])
def test_deepep_validates_routed_hidden_size(hidden: int) -> None:
    if hidden % 256 == 0:
        deepep_v2._validate_deepep_v2_hidden_size(hidden)
    else:
        with pytest.raises(RuntimeError, match=f"routed hidden size.*got {hidden}"):
            deepep_v2._validate_deepep_v2_hidden_size(hidden)


def test_deepep_runtime_key_separates_fp8_dispatch() -> None:
    ep_pg = object()
    bf16_block = _stub_block(ep_pg=ep_pg, runtime_cache={})
    fp8_block = _stub_block(ep_pg=ep_pg, runtime_cache={}, fp8_enabled=True)

    bf16_key = deepep_v2._runtime_key(bf16_block, hidden=256, top_k=2)
    fp8_key = deepep_v2._runtime_key(fp8_block, hidden=256, top_k=2)

    assert bf16_key.use_fp8_dispatch is False
    assert fp8_key.use_fp8_dispatch is True
    assert fp8_key != bf16_key


def test_deepep_mxfp8_scale_pack_round_trip() -> None:
    scale_bits = torch.arange(16, dtype=torch.uint8).reshape(2, 8)
    scales = scale_bits.view(torch.float8_e8m0fnu)

    packed = deepep_v2._pack_deepep_mxfp8_scales(scales)
    restored = deepep_v2._unpack_deepep_mxfp8_scales(packed)

    assert packed.dtype == torch.int32
    assert packed.shape == (2, 2)
    assert packed.is_contiguous()
    assert restored.dtype == torch.float8_e8m0fnu
    assert restored.shape == scales.shape
    torch.testing.assert_close(restored.view(torch.uint8), scale_bits)


def test_deepep_mxfp8_scale_pack_rejects_incompatible_layout() -> None:
    with pytest.raises(RuntimeError, match="float8_e8m0fnu"):
        deepep_v2._pack_deepep_mxfp8_scales(torch.ones((2, 4)))

    incompatible = torch.zeros((2, 6), dtype=torch.uint8).view(torch.float8_e8m0fnu)
    with pytest.raises(RuntimeError, match="four-scale packing"):
        deepep_v2._pack_deepep_mxfp8_scales(incompatible)


def test_deepep_fp8_autograd_uses_prequantized_experts_and_post_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hidden = 256
    tokens = 2
    seen = {}

    class FakeHandle:
        psum_num_recv_tokens_per_expert = torch.tensor([tokens], dtype=torch.int32)

    class FakeBuffer:
        def dispatch_expanded_into(
            self,
            payload,
            *,
            topk_idx,
            topk_weights,
            recv_x_out,
            recv_sf_out,
            recv_topk_weights_out,
            **_kwargs,
        ):
            source_q, source_sf = payload
            assert source_q.dtype == torch.float8_e4m3fn
            assert source_sf.dtype == torch.int32
            recv_x_out.zero_()
            recv_sf_out.zero_()
            recv_topk_weights_out.copy_(topk_weights.reshape(-1))
            return (
                (recv_x_out, recv_sf_out),
                topk_idx,
                recv_topk_weights_out,
                FakeHandle(),
                object(),
            )

        def dispatch_cached_expanded_into(self, grad, *, recv_x_out, **_kwargs):
            recv_x_out.copy_(grad)
            return recv_x_out, None, None, FakeHandle(), object()

        def combine(self, value, **_kwargs):
            return value, None, object()

    class FakeRoutedExperts:
        def __call__(self, value, counts, **kwargs):
            seen["counts"] = counts.clone()
            seen["kwargs"] = kwargs
            return value * 3

    fp8_config = SimpleNamespace(
        enabled=True,
        block_size=32,
        scale_mode=SimpleNamespace(value="rceil"),
    )
    block: Any = SimpleNamespace(
        rowwise_fp8=fp8_config,
        routed_experts=FakeRoutedExperts(),
        ep_pg=object(),
        ep_world_size=1,
    )
    runtime: Any = SimpleNamespace(
        buffer=FakeBuffer(),
        use_fp8_dispatch=True,
        hidden=hidden,
        num_experts=1,
        num_max_tokens_per_rank=tokens,
        expert_alignment=1,
        num_sms=4,
        num_qps=1,
        async_with_compute_stream=False,
        num_topk=1,
    )

    class FakeContext:
        needs_input_grad = (True, False, True, False, False, False)

        def save_for_backward(self, *tensors):
            self.saved_tensors = tensors

    def fake_quantize(value, *, block_size, scale_mode):
        assert block_size == 32
        assert scale_mode == "rceil"
        q = torch.zeros_like(value, dtype=torch.float8_e4m3fn)
        scale_bits = torch.zeros(
            (value.shape[0], value.shape[1] // block_size),
            dtype=torch.uint8,
        )
        return q, scale_bits.view(torch.float8_e8m0fnu)

    monkeypatch.setattr(deepep_v2, "quantize_rows_to_mxfp8", fake_quantize)
    monkeypatch.setattr(
        deepep_v2,
        "_logical_rank2_tensor",
        lambda shape, *, dtype, device: torch.ones(shape, dtype=dtype, device=device),
    )
    monkeypatch.setattr(
        deepep_v2,
        "_expanded_weight_grad_to_topk_grad",
        lambda **kwargs: kwargs["expanded_weight_grad"].reshape(tokens, 1),
    )

    source = torch.randn(tokens, hidden)
    topk_idx = torch.zeros((tokens, 1), dtype=torch.int64)
    topk_weights = torch.tensor([[0.5], [0.25]])

    ctx = FakeContext()
    out = deepep_v2._DeepEpV2Autograd.forward(
        ctx,
        source,
        topk_idx,
        topk_weights,
        block,
        runtime,
        tokens,
    )
    grads = deepep_v2._DeepEpV2Autograd.backward(ctx, torch.ones_like(out))

    torch.testing.assert_close(out[0], torch.full((hidden,), 1.5))
    torch.testing.assert_close(out[1], torch.full((hidden,), 0.75))
    torch.testing.assert_close(grads[0][0], torch.full((hidden,), 1.5))
    torch.testing.assert_close(grads[0][1], torch.full((hidden,), 0.75))
    torch.testing.assert_close(
        grads[2],
        torch.full_like(topk_weights, 3.0 * hidden),
    )
    torch.testing.assert_close(seen["counts"], torch.tensor([tokens], dtype=torch.int32))
    assert seen["kwargs"]["use_rowwise_fp8"] is True
    assert "row_weights" not in seen["kwargs"]
    assert seen["kwargs"]["rowwise_fp8_input_q"].dtype == torch.float8_e4m3fn
    assert seen["kwargs"]["rowwise_fp8_input_scales"].dtype == torch.float8_e8m0fnu


def test_deepep_process_group_warmup_initializes_torch_nccl_on_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list = []

    class FakeScalar:
        def zero_(self):
            calls.append(("zero",))
            return self

    def fake_empty(shape, *, device, dtype):
        calls.append(("empty", shape, device, dtype))
        return FakeScalar()

    def fake_all_reduce(tensor, *, group):
        calls.append(("all_reduce", tensor, group))

    def fake_synchronize(device):
        calls.append(("synchronize", device))

    monkeypatch.setenv("EP_REUSE_NCCL_COMM", "1")
    monkeypatch.setattr(deepep_v2.torch, "empty", fake_empty)
    monkeypatch.setattr(deepep_v2.dist, "all_reduce", fake_all_reduce)
    monkeypatch.setattr(deepep_v2.torch.cuda, "synchronize", fake_synchronize)
    ep_pg = object()
    block: Any = SimpleNamespace(ep_pg=ep_pg)

    deepep_v2._warm_deepep_v2_process_group(block, torch.device("cuda:3"))

    assert calls[0] == ("empty", (1,), torch.device("cuda:3"), torch.int32)
    assert calls[1] == ("zero",)
    assert calls[2][0] == "all_reduce"
    assert calls[2][2] is ep_pg
    assert calls[3] == ("synchronize", torch.device("cuda:3"))


def test_global_capacity_scalar_is_filled_on_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list = []

    class FakeScalar:
        def fill_(self, value):
            calls.append(("fill", value))
            return self

        def item(self):
            calls.append(("item",))
            return 112

    def fake_empty(shape, *, device, dtype):
        calls.append(("empty", shape, device, dtype))
        return FakeScalar()

    def fail_tensor(*_args, **_kwargs):
        raise AssertionError("must not stage the capacity scalar through CPU memory")

    def fake_all_reduce(tensor, *, op, group):
        calls.append(("all_reduce", tensor, op, group))

    monkeypatch.setattr(deepep_v2.torch, "empty", fake_empty)
    monkeypatch.setattr(deepep_v2.torch, "tensor", fail_tensor)
    monkeypatch.setattr(deepep_v2.dist, "all_reduce", fake_all_reduce)
    ep_pg = object()
    block: Any = SimpleNamespace(ep_pg=ep_pg)

    capacity = deepep_v2._global_num_max_tokens_per_rank(
        block,
        requested_tokens=80,
        device=torch.device("cuda"),
    )

    assert capacity == 112
    assert calls[0] == ("empty", (1,), torch.device("cuda"), torch.long)
    assert calls[1] == ("fill", 80)
    assert calls[2][0] == "all_reduce"
    assert calls[2][2] is deepep_v2.dist.ReduceOp.MAX
    assert calls[2][3] is ep_pg
    assert calls[3] == ("item",)
