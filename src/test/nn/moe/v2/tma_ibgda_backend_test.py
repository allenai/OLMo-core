from pathlib import Path

import pytest
import torch

from olmo_core.nn.moe.v2.tma_ibgda import (
    TMA_IBGDA_COMPLETION_BYTES,
    TMA_IBGDA_DOORBELL_BYTES,
    TMA_IBGDA_ROUTE_RECORD_BYTES,
    TMA_IBGDA_WORKSPACE_ALIGNMENT,
    TmaIbgdaBackendConfig,
    TmaIbgdaBackendUnavailable,
    build_tma_ibgda_route_metadata,
    is_tma_ibgda_backend_available,
    plan_tma_ibgda_peer_windows,
    plan_tma_ibgda_workspace,
    reference_combine_bf16,
    reference_dispatch_bf16,
    tma_ibgda_empty_symmetric_expert_out,
    tma_ibgda_rowwise_combine_bf16,
    tma_ibgda_rowwise_dispatch_bf16,
)


def test_tma_ibgda_backend_package_imports():
    assert isinstance(is_tma_ibgda_backend_available(), bool)
    assert callable(tma_ibgda_empty_symmetric_expert_out)


def test_tma_ibgda_kernel_loader_uses_separate_extension_symbols():
    from olmo_core.kernels import tma_ibgda_ep

    assert isinstance(tma_ibgda_ep.is_available(), bool)
    assert hasattr(tma_ibgda_ep, "extension_contract")
    assert hasattr(tma_ibgda_ep, "plan_peer_window_layout")
    assert hasattr(tma_ibgda_ep, "dispatch_bf16_peer")
    assert hasattr(tma_ibgda_ep, "preprocess_routes")
    assert hasattr(tma_ibgda_ep, "preprocess_routes_into")
    assert hasattr(tma_ibgda_ep, "route_records_with_probs")
    assert hasattr(tma_ibgda_ep, "dispatch_bf16_ibgda")
    assert hasattr(tma_ibgda_ep, "dispatch_bf16_ibgda_records")
    assert hasattr(tma_ibgda_ep, "dispatch_bf16_ibgda_records_tma")
    assert hasattr(tma_ibgda_ep, "combine_bf16_ibgda")
    assert hasattr(tma_ibgda_ep, "combine_bf16_ibgda_records")
    assert hasattr(tma_ibgda_ep, "route_dot_bf16_ibgda")
    assert hasattr(tma_ibgda_ep, "route_dot_bf16_ibgda_records")
    assert hasattr(tma_ibgda_ep, "barrier_all_on_stream")
    assert hasattr(tma_ibgda_ep, "signal_all_and_wait")


def test_tma_ibgda_built_extension_contract_matches_python_constants():
    from olmo_core.kernels import tma_ibgda_ep

    if not tma_ibgda_ep.is_available():
        pytest.skip("TMA/IBGDA CUDA extension is not built")

    contract = tma_ibgda_ep.extension_contract()
    assert contract["extension_module"] == "_tma_ibgda_ep_ext_gpu"
    assert contract["route_record_bytes"] == TMA_IBGDA_ROUTE_RECORD_BYTES
    assert contract["route_record_words"] == TMA_IBGDA_ROUTE_RECORD_BYTES // 4
    assert contract["route_flag_valid"] == 1
    assert contract["workspace_alignment"] == TMA_IBGDA_WORKSPACE_ALIGNMENT
    assert contract["doorbell_bytes"] == TMA_IBGDA_DOORBELL_BYTES
    assert contract["completion_bytes"] == TMA_IBGDA_COMPLETION_BYTES
    assert contract["peer_window_layout_bytes"] > 0
    assert contract["kernel_launch_config_bytes"] > 0
    assert contract["bf16_only"] is True
    assert contract["has_gpu_route_preprocess"] is True
    assert contract["has_ibgda_dispatch"] is True
    assert contract["has_tma_load_dispatch"] is True
    assert contract["has_ibgda_combine"] is True
    assert contract["has_route_dot_backward"] is True
    assert contract["has_peer_window_layout_planner"] is True


def test_tma_ibgda_backend_contract_validator_accepts_expected_contract():
    from olmo_core.nn.moe.v2.tma_ibgda.backend import _check_kernel_contract

    class FakeTmaIbgdaExt:
        @staticmethod
        def extension_contract():
            return {
                "extension_module": "_tma_ibgda_ep_ext_gpu",
                "route_record_bytes": TMA_IBGDA_ROUTE_RECORD_BYTES,
                "route_record_words": TMA_IBGDA_ROUTE_RECORD_BYTES // 4,
                "route_flag_valid": 1,
                "workspace_alignment": TMA_IBGDA_WORKSPACE_ALIGNMENT,
                "doorbell_bytes": TMA_IBGDA_DOORBELL_BYTES,
                "completion_bytes": TMA_IBGDA_COMPLETION_BYTES,
                "peer_window_layout_bytes": 128,
                "kernel_launch_config_bytes": 24,
                "bf16_only": True,
                "has_gpu_route_preprocess": True,
                "has_ibgda_dispatch": True,
                "has_tma_load_dispatch": True,
                "has_ibgda_combine": True,
                "has_route_dot_backward": True,
                "has_peer_window_layout_planner": True,
            }

    _check_kernel_contract(FakeTmaIbgdaExt())


def test_tma_ibgda_backend_contract_validator_rejects_stale_extension():
    from olmo_core.nn.moe.v2.tma_ibgda.backend import _check_kernel_contract

    class FakeTmaIbgdaExt:
        @staticmethod
        def extension_contract():
            return {
                "extension_module": "_tma_ibgda_ep_ext_gpu",
                "route_record_bytes": 16,
                "route_record_words": 4,
                "route_flag_valid": 1,
                "workspace_alignment": TMA_IBGDA_WORKSPACE_ALIGNMENT,
                "doorbell_bytes": TMA_IBGDA_DOORBELL_BYTES,
                "completion_bytes": TMA_IBGDA_COMPLETION_BYTES,
                "peer_window_layout_bytes": 128,
                "kernel_launch_config_bytes": 24,
                "bf16_only": True,
                "has_gpu_route_preprocess": True,
                "has_ibgda_dispatch": True,
                "has_tma_load_dispatch": True,
                "has_ibgda_combine": True,
                "has_route_dot_backward": True,
                "has_peer_window_layout_planner": True,
            }

    with pytest.raises(TmaIbgdaBackendUnavailable, match="route_record_bytes"):
        _check_kernel_contract(FakeTmaIbgdaExt())


def test_tma_ibgda_backend_contract_validator_rejects_bad_peer_window_layout():
    from olmo_core.nn.moe.v2.tma_ibgda.backend import _check_kernel_contract

    class FakeTmaIbgdaExt:
        @staticmethod
        def extension_contract():
            return {
                "extension_module": "_tma_ibgda_ep_ext_gpu",
                "route_record_bytes": TMA_IBGDA_ROUTE_RECORD_BYTES,
                "route_record_words": TMA_IBGDA_ROUTE_RECORD_BYTES // 4,
                "route_flag_valid": 1,
                "workspace_alignment": TMA_IBGDA_WORKSPACE_ALIGNMENT,
                "doorbell_bytes": TMA_IBGDA_DOORBELL_BYTES,
                "completion_bytes": TMA_IBGDA_COMPLETION_BYTES,
                "peer_window_layout_bytes": 0,
                "kernel_launch_config_bytes": 24,
                "bf16_only": True,
                "has_gpu_route_preprocess": True,
                "has_ibgda_dispatch": True,
                "has_tma_load_dispatch": True,
                "has_ibgda_combine": True,
                "has_route_dot_backward": True,
                "has_peer_window_layout_planner": True,
            }

    with pytest.raises(TmaIbgdaBackendUnavailable, match="peer_window_layout_bytes"):
        _check_kernel_contract(FakeTmaIbgdaExt())


def test_tma_ibgda_backend_contract_validator_requires_contract_symbol():
    from olmo_core.nn.moe.v2.tma_ibgda.backend import _check_kernel_contract

    with pytest.raises(TmaIbgdaBackendUnavailable, match="extension_contract"):
        _check_kernel_contract(object())


def test_tma_ibgda_cuda_side_peer_window_layout_matches_python_plan():
    from olmo_core.kernels import tma_ibgda_ep

    if not tma_ibgda_ep.is_available():
        pytest.skip("TMA/IBGDA CUDA extension is not built")

    metadata = build_tma_ibgda_route_metadata(
        torch.tensor([[0, 1], [1, -1], [0, 1]], dtype=torch.long),
        torch.tensor([[0, 0], [1, -1], [2, 3]], dtype=torch.long),
        ep_world_size=2,
        rank_capacity=4,
    )
    plan = plan_tma_ibgda_peer_windows(
        metadata,
        hidden_size=7,
        dtype=torch.bfloat16,
    )
    dtype_bytes = torch.empty((), dtype=plan.dtype).element_size()
    layout = tma_ibgda_ep.plan_peer_window_layout(
        num_routes=metadata.num_routes,
        ep_world_size=metadata.ep_world_size,
        rank_capacity=metadata.rank_capacity,
        hidden_size=plan.hidden_size,
        dtype_bytes=dtype_bytes,
    )

    for key in (
        "route_records_offset",
        "routes_per_rank_offset",
        "rank_offsets_offset",
        "overflow_by_rank_offset",
        "payload_window_offset",
        "send_doorbells_offset",
        "recv_completions_offset",
        "rank_stride_bytes",
        "total_peer_window_bytes",
        "route_records_bytes",
        "routes_per_rank_bytes",
        "rank_offsets_bytes",
        "overflow_by_rank_bytes",
        "payload_window_bytes_per_rank",
        "send_doorbells_bytes",
        "recv_completions_bytes",
        "ep_world_size",
        "rank_capacity",
        "hidden_size",
    ):
        assert layout[key] == getattr(plan, key)
    assert layout["dtype_bytes"] == dtype_bytes


def test_tma_ibgda_peer_window_payload_view_uses_planned_offset():
    from olmo_core.nn.moe.v2.tma_ibgda.backend import _payload_view_from_peer_window

    metadata = build_tma_ibgda_route_metadata(
        torch.tensor([[0, 1], [1, -1]], dtype=torch.long),
        torch.tensor([[0, 0], [1, -1]], dtype=torch.long),
        ep_world_size=2,
        rank_capacity=3,
    )
    plan = plan_tma_ibgda_peer_windows(
        metadata,
        hidden_size=5,
        dtype=torch.bfloat16,
    )
    window = torch.empty((plan.rank_stride_bytes,), dtype=torch.uint8)

    payload = _payload_view_from_peer_window(window, plan, dtype=torch.bfloat16)

    assert payload.shape == (3, 5)
    assert payload.dtype == torch.bfloat16
    assert payload.data_ptr() == window.data_ptr() + plan.payload_window_offset
    assert payload.is_contiguous()


def test_tma_ibgda_peer_window_section_views_use_planned_offsets():
    from olmo_core.nn.moe.v2.tma_ibgda.backend import _peer_window_views_from_window

    metadata = build_tma_ibgda_route_metadata(
        torch.tensor([[0, 1], [1, -1]], dtype=torch.long),
        torch.tensor([[0, 0], [1, -1]], dtype=torch.long),
        ep_world_size=2,
        rank_capacity=3,
    )
    plan = plan_tma_ibgda_peer_windows(
        metadata,
        hidden_size=5,
        dtype=torch.bfloat16,
    )
    window = torch.empty((plan.rank_stride_bytes,), dtype=torch.uint8)

    views = _peer_window_views_from_window(window, plan)

    expected = {
        "route_records": (
            views.route_records,
            plan.route_records_offset,
            (metadata.num_routes, TMA_IBGDA_ROUTE_RECORD_BYTES // 4),
            torch.int32,
        ),
        "routes_per_rank": (
            views.routes_per_rank,
            plan.routes_per_rank_offset,
            (metadata.ep_world_size,),
            torch.long,
        ),
        "rank_offsets": (
            views.rank_offsets,
            plan.rank_offsets_offset,
            (metadata.ep_world_size + 1,),
            torch.long,
        ),
        "overflow_by_rank": (
            views.overflow_by_rank,
            plan.overflow_by_rank_offset,
            (metadata.ep_world_size,),
            torch.bool,
        ),
        "payload": (
            views.payload,
            plan.payload_window_offset,
            (metadata.rank_capacity, plan.hidden_size),
            torch.bfloat16,
        ),
        "send_doorbells": (
            views.send_doorbells,
            plan.send_doorbells_offset,
            (metadata.ep_world_size,),
            torch.long,
        ),
        "recv_completions": (
            views.recv_completions,
            plan.recv_completions_offset,
            (metadata.ep_world_size,),
            torch.long,
        ),
    }
    assert views.window is window
    for tensor, offset, shape, dtype in expected.values():
        assert tensor.data_ptr() == window.data_ptr() + offset
        assert tuple(tensor.shape) == shape
        assert tensor.dtype == dtype
        assert tensor.is_contiguous()


def test_tma_ibgda_empty_peer_window_views_local_rank_uses_planned_sections():
    from olmo_core.nn.moe.v2.tma_ibgda.backend import _empty_peer_window_views

    metadata = build_tma_ibgda_route_metadata(
        torch.tensor([[0, 0], [0, -1]], dtype=torch.long),
        torch.tensor([[0, 1], [2, -1]], dtype=torch.long),
        ep_world_size=1,
        rank_capacity=3,
    )
    plan = plan_tma_ibgda_peer_windows(
        metadata,
        hidden_size=6,
        dtype=torch.bfloat16,
    )

    views = _empty_peer_window_views(
        plan,
        device=torch.device("cpu"),
        process_group=None,
        world_size=1,
        kind="test",
    )

    assert views.window.shape == (plan.rank_stride_bytes,)
    assert views.window.dtype == torch.uint8
    assert views.route_records.data_ptr() == views.window.data_ptr() + plan.route_records_offset
    assert views.routes_per_rank.data_ptr() == views.window.data_ptr() + plan.routes_per_rank_offset
    assert views.rank_offsets.data_ptr() == views.window.data_ptr() + plan.rank_offsets_offset
    assert views.overflow_by_rank.data_ptr() == views.window.data_ptr() + plan.overflow_by_rank_offset
    assert views.payload.data_ptr() == views.window.data_ptr() + plan.payload_window_offset
    assert views.payload.shape == (metadata.rank_capacity, plan.hidden_size)


def test_tma_ibgda_peer_window_payload_view_rejects_small_window():
    from olmo_core.nn.moe.v2.tma_ibgda.backend import _payload_view_from_peer_window

    metadata = build_tma_ibgda_route_metadata(
        torch.tensor([[0]], dtype=torch.long),
        torch.tensor([[0]], dtype=torch.long),
        ep_world_size=1,
        rank_capacity=1,
    )
    plan = plan_tma_ibgda_peer_windows(
        metadata,
        hidden_size=1,
        dtype=torch.bfloat16,
    )

    with pytest.raises(RuntimeError, match="too small"):
        _payload_view_from_peer_window(
            torch.empty((plan.rank_stride_bytes - 1,), dtype=torch.uint8),
            plan,
            dtype=torch.bfloat16,
        )


def test_tma_ibgda_combine_handle_validator_accepts_matching_handle():
    from olmo_core.nn.moe.v2.tma_ibgda.backend import (
        TmaIbgdaDispatchHandle,
        _check_dispatch_handle_matches_combine,
    )

    cfg = TmaIbgdaBackendConfig(static_route_budget=4)
    group = object()
    ranks = torch.tensor([[0, 1], [1, -1]], dtype=torch.long)
    rows = torch.tensor([[0, 0], [1, -1]], dtype=torch.long)
    metadata = build_tma_ibgda_route_metadata(
        ranks,
        rows,
        ep_world_size=2,
        rank_capacity=4,
        static_route_budget=cfg.static_route_budget,
    )
    handle = TmaIbgdaDispatchHandle(
        metadata=metadata,
        group_name="ep",
        config=cfg,
        process_group=group,
        peer_out_ptrs=torch.empty((0,), dtype=torch.long),
        dst_ranks_version=getattr(ranks, "_version", None),
        dst_rows_version=getattr(rows, "_version", None),
    )

    _check_dispatch_handle_matches_combine(
        handle,
        group_name="ep",
        src_ranks=ranks,
        src_rows=rows,
        ep_world_size=2,
        rank_capacity=4,
        process_group=group,
        config=cfg,
    )


def test_tma_ibgda_combine_handle_validator_rejects_group_mismatch():
    from olmo_core.nn.moe.v2.tma_ibgda.backend import (
        TmaIbgdaDispatchHandle,
        _check_dispatch_handle_matches_combine,
    )

    cfg = TmaIbgdaBackendConfig()
    ranks = torch.tensor([[0]], dtype=torch.long)
    rows = torch.tensor([[0]], dtype=torch.long)
    metadata = build_tma_ibgda_route_metadata(ranks, rows, ep_world_size=1)
    handle = TmaIbgdaDispatchHandle(
        metadata=metadata,
        group_name="dispatch",
        config=cfg,
        process_group=None,
        peer_out_ptrs=torch.empty((0,), dtype=torch.long),
    )

    with pytest.raises(RuntimeError, match="group mismatch"):
        _check_dispatch_handle_matches_combine(
            handle,
            group_name="combine",
            src_ranks=ranks,
            src_rows=rows,
            ep_world_size=1,
            rank_capacity=None,
            process_group=None,
            config=None,
        )


def test_tma_ibgda_combine_handle_validator_rejects_route_tensor_mismatch():
    from olmo_core.nn.moe.v2.tma_ibgda.backend import (
        TmaIbgdaDispatchHandle,
        _check_dispatch_handle_matches_combine,
    )

    cfg = TmaIbgdaBackendConfig()
    ranks = torch.tensor([[0, 0]], dtype=torch.long)
    rows = torch.tensor([[0, 1]], dtype=torch.long)
    metadata = build_tma_ibgda_route_metadata(
        ranks,
        rows,
        ep_world_size=1,
        rank_capacity=2,
    )
    handle = TmaIbgdaDispatchHandle(
        metadata=metadata,
        group_name="ep",
        config=cfg,
        process_group=None,
        peer_out_ptrs=torch.empty((0,), dtype=torch.long),
    )

    with pytest.raises(RuntimeError, match="route tensor mismatch"):
        _check_dispatch_handle_matches_combine(
            handle,
            group_name="ep",
            src_ranks=ranks.clone(),
            src_rows=rows.clone(),
            ep_world_size=1,
            rank_capacity=2,
            process_group=None,
            config=cfg,
        )


def test_tma_ibgda_combine_handle_validator_rejects_config_mismatch():
    from olmo_core.nn.moe.v2.tma_ibgda.backend import (
        TmaIbgdaDispatchHandle,
        _check_dispatch_handle_matches_combine,
    )

    cfg = TmaIbgdaBackendConfig(num_sms_dispatch=32)
    ranks = torch.tensor([[0]], dtype=torch.long)
    rows = torch.tensor([[0]], dtype=torch.long)
    metadata = build_tma_ibgda_route_metadata(ranks, rows, ep_world_size=1)
    handle = TmaIbgdaDispatchHandle(
        metadata=metadata,
        group_name="ep",
        config=cfg,
        process_group=None,
        peer_out_ptrs=torch.empty((0,), dtype=torch.long),
    )

    with pytest.raises(RuntimeError, match="config mismatch"):
        _check_dispatch_handle_matches_combine(
            handle,
            group_name="ep",
            src_ranks=ranks,
            src_rows=rows,
            ep_world_size=1,
            rank_capacity=None,
            process_group=None,
            config=TmaIbgdaBackendConfig(num_sms_dispatch=64),
        )


def test_tma_ibgda_combine_handle_validator_rejects_peer_window_plan_mismatch():
    from olmo_core.nn.moe.v2.tma_ibgda.backend import (
        TmaIbgdaDispatchHandle,
        _check_dispatch_handle_matches_combine,
    )

    cfg = TmaIbgdaBackendConfig()
    ranks = torch.tensor([[0]], dtype=torch.long)
    rows = torch.tensor([[0]], dtype=torch.long)
    metadata = build_tma_ibgda_route_metadata(ranks, rows, ep_world_size=1)
    handle = TmaIbgdaDispatchHandle(
        metadata=metadata,
        group_name="ep",
        config=cfg,
        process_group=None,
        peer_out_ptrs=torch.empty((0,), dtype=torch.long),
        peer_window_plan=plan_tma_ibgda_peer_windows(
            metadata,
            hidden_size=4,
            dtype=torch.bfloat16,
        ),
    )

    with pytest.raises(RuntimeError, match="peer-window hidden_size mismatch"):
        _check_dispatch_handle_matches_combine(
            handle,
            group_name="ep",
            src_ranks=ranks,
            src_rows=rows,
            ep_world_size=1,
            rank_capacity=None,
            process_group=None,
            config=cfg,
            payload_hidden_size=8,
            payload_dtype=torch.bfloat16,
        )


def test_tma_ibgda_combine_handle_validator_rejects_mutated_route_tensor():
    from olmo_core.nn.moe.v2.tma_ibgda.backend import (
        TmaIbgdaDispatchHandle,
        _check_dispatch_handle_matches_combine,
    )

    cfg = TmaIbgdaBackendConfig()
    ranks = torch.tensor([[0]], dtype=torch.long)
    rows = torch.tensor([[0]], dtype=torch.long)
    metadata = build_tma_ibgda_route_metadata(ranks, rows, ep_world_size=1)
    handle = TmaIbgdaDispatchHandle(
        metadata=metadata,
        group_name="ep",
        config=cfg,
        process_group=None,
        peer_out_ptrs=torch.empty((0,), dtype=torch.long),
        dst_ranks_version=getattr(ranks, "_version", None),
        dst_rows_version=getattr(rows, "_version", None),
    )

    rows[0, 0] = 1

    with pytest.raises(RuntimeError, match="version mismatch"):
        _check_dispatch_handle_matches_combine(
            handle,
            group_name="ep",
            src_ranks=ranks,
            src_rows=rows,
            ep_world_size=1,
            rank_capacity=None,
            process_group=None,
            config=cfg,
        )


@pytest.mark.parametrize("expert_out_is_symmetric", [False, True])
def test_tma_ibgda_combine_backward_preserves_symmetric_expert_out_flag(
    monkeypatch,
    expert_out_is_symmetric,
):
    from olmo_core.nn.moe.v2.tma_ibgda import backend as tma_backend

    seen_route_dot_flags = []

    def fake_combine(expert_out, *args, **kwargs):
        return expert_out.clone()

    def fake_dispatch(grad_out, *args, **kwargs):
        return grad_out.clone(), None, None, None

    def fake_route_dot(expert_out, grad_out, src_ranks, src_rows, **kwargs):
        seen_route_dot_flags.append(kwargs["expert_out_is_symmetric"])
        return torch.ones_like(src_ranks, dtype=torch.float32)

    monkeypatch.setattr(tma_backend, "_combine_bf16_peer_raw", fake_combine)
    monkeypatch.setattr(tma_backend, "_dispatch_bf16_peer_raw", fake_dispatch)
    monkeypatch.setattr(tma_backend, "_route_dot_bf16_peer_raw", fake_route_dot)

    expert_out = torch.randn((2, 3), requires_grad=True)
    src_ranks = torch.zeros((2, 1), dtype=torch.long)
    src_rows = torch.zeros((2, 1), dtype=torch.long)
    probs = torch.ones((2, 1), dtype=torch.float32, requires_grad=True)

    out = tma_backend._TmaIbgdaCombineAutograd.apply(
        expert_out,
        src_ranks,
        src_rows,
        probs,
        1,
        2,
        None,
        TmaIbgdaBackendConfig(),
        None,
        expert_out_is_symmetric,
    )
    out.sum().backward()

    assert seen_route_dot_flags == [expert_out_is_symmetric]


def test_tma_ibgda_metadata_counts_offsets_and_ordinals():
    dst_ranks = torch.tensor(
        [
            [0, 1, -1],
            [1, 0, 1],
        ],
        dtype=torch.long,
    )
    dst_rows = torch.tensor(
        [
            [0, 0, -1],
            [1, 2, 3],
        ],
        dtype=torch.long,
    )

    metadata = build_tma_ibgda_route_metadata(
        dst_ranks,
        dst_rows,
        ep_world_size=2,
        rank_capacity=4,
        static_route_budget=2,
    )

    assert metadata.num_tokens == 2
    assert metadata.top_k == 3
    assert metadata.num_routes == 6
    assert metadata.valid_mask.tolist() == [[True, True, False], [True, True, True]]
    assert metadata.source_rows.tolist() == [[0, 0, 0], [1, 1, 1]]
    assert metadata.topk_slots.tolist() == [[0, 1, 2], [0, 1, 2]]
    assert metadata.routes_per_rank.tolist() == [2, 3]
    assert metadata.rank_offsets.tolist() == [0, 2, 5]
    assert metadata.route_ordinals.tolist() == [[0, 0, -1], [1, 1, 2]]
    assert metadata.overflow_by_rank.tolist() == [False, True]


def test_tma_ibgda_metadata_infers_rank_capacity():
    dst_ranks = torch.tensor([[0, -1], [0, 1]], dtype=torch.long)
    dst_rows = torch.tensor([[2, -1], [5, 0]], dtype=torch.long)

    metadata = build_tma_ibgda_route_metadata(
        dst_ranks,
        dst_rows,
        ep_world_size=2,
    )

    assert metadata.rank_capacity == 6
    assert metadata.routes_per_rank.tolist() == [2, 1]


def test_tma_ibgda_metadata_rejects_mismatched_dropped_routes():
    dst_ranks = torch.tensor([[0, -1]], dtype=torch.long)
    dst_rows = torch.tensor([[0, 3]], dtype=torch.long)

    with pytest.raises(RuntimeError, match="both rank and row negative"):
        build_tma_ibgda_route_metadata(dst_ranks, dst_rows, ep_world_size=1)


def test_tma_ibgda_metadata_rejects_out_of_range_rank_and_row():
    dst_ranks = torch.tensor([[2]], dtype=torch.long)
    dst_rows = torch.tensor([[0]], dtype=torch.long)
    with pytest.raises(RuntimeError, match="outside ep_world_size"):
        build_tma_ibgda_route_metadata(dst_ranks, dst_rows, ep_world_size=2)

    dst_ranks = torch.tensor([[1]], dtype=torch.long)
    dst_rows = torch.tensor([[4]], dtype=torch.long)
    with pytest.raises(RuntimeError, match="outside rank_capacity"):
        build_tma_ibgda_route_metadata(
            dst_ranks,
            dst_rows,
            ep_world_size=2,
            rank_capacity=4,
        )


def test_tma_ibgda_reference_dispatch_and_combine_semantics():
    x = torch.tensor(
        [
            [1.0, 2.0],
            [3.0, 5.0],
        ],
        dtype=torch.bfloat16,
    )
    dst_ranks = torch.tensor([[0, 1, -1], [1, 0, 1]], dtype=torch.long)
    dst_rows = torch.tensor([[0, 0, -1], [1, 1, 2]], dtype=torch.long)
    probs = torch.tensor([[0.5, 0.25, 1.0], [0.5, 0.25, 0.125]], dtype=torch.float32)

    dispatched = reference_dispatch_bf16(
        x,
        dst_ranks,
        dst_rows,
        ep_world_size=2,
        rank_capacity=3,
    )
    assert dispatched.shape == (2, 3, 2)
    assert torch.equal(dispatched[0, 0], x[0])
    assert torch.equal(dispatched[0, 1], x[1])
    assert torch.equal(dispatched[1, 0], x[0])
    assert torch.equal(dispatched[1, 1], x[1])
    assert torch.equal(dispatched[1, 2], x[1])

    combined = reference_combine_bf16(dispatched, dst_ranks, dst_rows, probs=probs)
    expected = torch.tensor(
        [
            [0.75, 1.5],
            [2.625, 4.375],
        ],
        dtype=torch.bfloat16,
    )
    assert torch.equal(combined, expected)


def test_tma_ibgda_reference_weighted_dispatch_semantics():
    x = torch.tensor([[2.0, 4.0]], dtype=torch.bfloat16)
    dst_ranks = torch.tensor([[0, 1]], dtype=torch.long)
    dst_rows = torch.tensor([[0, 0]], dtype=torch.long)
    probs = torch.tensor([[0.5, 0.25]], dtype=torch.float32)

    dispatched = reference_dispatch_bf16(
        x,
        dst_ranks,
        dst_rows,
        ep_world_size=2,
        probs=probs,
    )

    assert torch.equal(dispatched[0, 0], torch.tensor([1.0, 2.0], dtype=torch.bfloat16))
    assert torch.equal(dispatched[1, 0], torch.tensor([0.5, 1.0], dtype=torch.bfloat16))


def test_tma_ibgda_workspace_plan_sizes_payload_and_metadata():
    metadata = build_tma_ibgda_route_metadata(
        torch.tensor([[0, 1], [1, -1]], dtype=torch.long),
        torch.tensor([[0, 0], [1, -1]], dtype=torch.long),
        ep_world_size=2,
        rank_capacity=3,
    )

    plan = plan_tma_ibgda_workspace(metadata, hidden_size=4)

    assert plan.ep_world_size == 2
    assert plan.rank_capacity == 3
    assert plan.hidden_size == 4
    assert plan.dtype == torch.bfloat16
    assert plan.route_records_bytes == metadata.num_routes * TMA_IBGDA_ROUTE_RECORD_BYTES
    assert plan.rank_counts_bytes == 2 * 8
    assert plan.rank_offsets_bytes == 3 * 8
    assert plan.payload_window_bytes == 2 * 3 * 4 * 2
    assert (
        plan.total_bytes
        == plan.route_records_bytes
        + plan.rank_counts_bytes
        + plan.rank_offsets_bytes
        + plan.payload_window_bytes
    )


def test_tma_ibgda_peer_window_plan_offsets_are_aligned_and_rank_strided():
    metadata = build_tma_ibgda_route_metadata(
        torch.tensor([[0, 1], [1, -1]], dtype=torch.long),
        torch.tensor([[0, 0], [1, -1]], dtype=torch.long),
        ep_world_size=2,
        rank_capacity=3,
    )

    plan = plan_tma_ibgda_peer_windows(metadata, hidden_size=4)

    assert plan.ep_world_size == 2
    assert plan.rank_capacity == 3
    assert plan.hidden_size == 4
    assert plan.alignment == TMA_IBGDA_WORKSPACE_ALIGNMENT
    assert plan.route_records_bytes == metadata.num_routes * TMA_IBGDA_ROUTE_RECORD_BYTES
    assert plan.routes_per_rank_bytes == 2 * 8
    assert plan.rank_offsets_bytes == 3 * 8
    assert plan.overflow_by_rank_bytes == 2
    assert plan.payload_window_bytes_per_rank == 3 * 4 * 2
    assert plan.send_doorbells_bytes == 2 * TMA_IBGDA_DOORBELL_BYTES
    assert plan.recv_completions_bytes == 2 * TMA_IBGDA_COMPLETION_BYTES
    assert plan.route_records_offset == 0
    offsets = [
        plan.routes_per_rank_offset,
        plan.rank_offsets_offset,
        plan.overflow_by_rank_offset,
        plan.payload_window_offset,
        plan.send_doorbells_offset,
        plan.recv_completions_offset,
        plan.rank_stride_bytes,
    ]
    assert offsets == sorted(offsets)
    assert all(offset % TMA_IBGDA_WORKSPACE_ALIGNMENT == 0 for offset in offsets)
    assert plan.total_peer_window_bytes == plan.rank_stride_bytes * 2
    assert plan.peer_window_offsets == (0, plan.rank_stride_bytes)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_tma_ibgda_cuda_route_preprocess_counts_offsets_and_records():
    from olmo_core.kernels import tma_ibgda_ep

    if not tma_ibgda_ep.is_available():
        pytest.skip("TMA/IBGDA CUDA extension is not built")

    device = torch.device("cuda")
    dst_ranks = torch.tensor([[0, 1], [1, -1], [0, 1]], device=device, dtype=torch.long)
    dst_rows = torch.tensor([[0, 0], [1, -1], [2, 3]], device=device, dtype=torch.long)
    probs = torch.tensor([[1.0, 0.5], [0.25, 1.0], [0.125, 0.75]], device=device)

    (
        route_records,
        routes_per_rank,
        rank_offsets,
        overflow_by_rank,
        route_ordinals,
        errors,
    ) = tma_ibgda_ep.preprocess_routes(
        dst_ranks,
        dst_rows,
        ep_world_size=2,
        rank_capacity=4,
        static_route_budget=2,
        probs=probs,
    )
    torch.cuda.synchronize(device)

    assert route_records.shape == (6, 8)
    assert routes_per_rank.cpu().tolist() == [2, 3]
    assert rank_offsets.cpu().tolist() == [0, 2, 5]
    assert overflow_by_rank.cpu().tolist() == [False, True]
    assert route_ordinals.cpu().tolist() == [[0, 0], [1, -1], [1, 2]]
    assert errors.cpu().tolist() == [0, 0, 0]

    records_cpu = route_records.cpu()
    assert records_cpu[:, 0].tolist() == [0, 0, 1, 1, 2, 2]
    assert records_cpu[:, 1].tolist() == [0, 1, 0, 1, 0, 1]
    assert records_cpu[:, 2].tolist() == [0, 1, 1, -1, 0, 1]
    assert records_cpu[:, 3].tolist() == [0, 0, 1, -1, 2, 3]
    assert records_cpu[:, 5].tolist() == [1, 1, 1, 0, 1, 1]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_tma_ibgda_cuda_route_records_with_probs_reuses_structure():
    from olmo_core.kernels import tma_ibgda_ep

    if not tma_ibgda_ep.is_available():
        pytest.skip("TMA/IBGDA CUDA extension is not built")

    device = torch.device("cuda")
    dst_ranks = torch.tensor([[0, 1], [1, -1]], device=device, dtype=torch.long)
    dst_rows = torch.tensor([[0, 0], [1, -1]], device=device, dtype=torch.long)
    base_probs = torch.ones((2, 2), device=device, dtype=torch.float32)
    next_probs = torch.tensor([[0.5, 0.25], [0.125, 1.0]], device=device)

    (route_records, *_) = tma_ibgda_ep.preprocess_routes(
        dst_ranks,
        dst_rows,
        ep_world_size=2,
        rank_capacity=2,
        static_route_budget=4,
        probs=base_probs,
    )
    prob_records = tma_ibgda_ep.route_records_with_probs(route_records, next_probs)
    torch.cuda.synchronize(device)

    assert torch.equal(prob_records[:, :4].cpu(), route_records[:, :4].cpu())
    assert torch.equal(prob_records[:, 5:].cpu(), route_records[:, 5:].cpu())
    record_probs = prob_records[:, 4].contiguous().cpu().view(torch.float32)
    assert torch.equal(record_probs, next_probs.cpu().view(-1))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_tma_ibgda_cuda_route_preprocess_errors_fail_fast():
    if not is_tma_ibgda_backend_available():
        pytest.skip("TMA/IBGDA CUDA extension is not built")

    x = torch.empty((1, 4), device="cuda", dtype=torch.bfloat16)
    cfg = TmaIbgdaBackendConfig(static_route_budget=1)

    with pytest.raises(RuntimeError, match="both rank and row negative"):
        tma_ibgda_rowwise_dispatch_bf16(
            x,
            torch.tensor([[-1]], device="cuda", dtype=torch.long),
            torch.tensor([[0]], device="cuda", dtype=torch.long),
            "unused",
            ep_world_size=1,
            rank_capacity=1,
            config=cfg,
        )

    with pytest.raises(RuntimeError, match="outside ep_world_size"):
        tma_ibgda_rowwise_dispatch_bf16(
            x,
            torch.tensor([[1]], device="cuda", dtype=torch.long),
            torch.tensor([[0]], device="cuda", dtype=torch.long),
            "unused",
            ep_world_size=1,
            rank_capacity=1,
            config=cfg,
        )

    with pytest.raises(RuntimeError, match="outside rank_capacity"):
        tma_ibgda_rowwise_dispatch_bf16(
            x,
            torch.tensor([[0]], device="cuda", dtype=torch.long),
            torch.tensor([[1]], device="cuda", dtype=torch.long),
            "unused",
            ep_world_size=1,
            rank_capacity=1,
            config=cfg,
        )


def test_tma_ibgda_dispatch_rejects_non_bf16_before_kernel_lookup():
    x = torch.empty((2, 4), dtype=torch.float32)
    dst_ranks = torch.zeros((2, 1), dtype=torch.long)
    dst_rows = torch.zeros((2, 1), dtype=torch.long)

    with pytest.raises(RuntimeError, match="torch.bfloat16"):
        tma_ibgda_rowwise_dispatch_bf16(
            x,
            dst_ranks,
            dst_rows,
            "unused",
            ep_world_size=1,
        )


def test_tma_ibgda_dispatch_requires_cuda_route_maps():
    x = torch.empty((2, 4), dtype=torch.bfloat16)
    dst_ranks = torch.zeros((2, 1), dtype=torch.long)
    dst_rows = torch.zeros((2, 1), dtype=torch.long)

    with pytest.raises(RuntimeError, match="route maps must be CUDA"):
        tma_ibgda_rowwise_dispatch_bf16(
            x,
            dst_ranks,
            dst_rows,
            "unused",
            ep_world_size=1,
            config=TmaIbgdaBackendConfig(static_route_budget=4),
        )


def test_tma_ibgda_combine_requires_cuda_route_maps():
    expert_out = torch.empty((2, 4), dtype=torch.bfloat16)
    src_ranks = torch.zeros((2, 1), dtype=torch.long)
    src_rows = torch.zeros((2, 1), dtype=torch.long)
    probs = torch.ones((2, 1), dtype=torch.float32)

    with pytest.raises(RuntimeError, match="route maps must be CUDA"):
        tma_ibgda_rowwise_combine_bf16(
            expert_out,
            src_ranks,
            src_rows,
            "unused",
            ep_world_size=1,
            probs=probs,
        )


def test_tma_ibgda_scaffold_does_not_import_wave_or_rowwise_transport():
    package_dir = Path(__file__).resolve().parents[4] / "olmo_core" / "nn" / "moe" / "v2" / "tma_ibgda"
    source = "\n".join(path.read_text() for path in sorted(package_dir.glob("*.py")))

    forbidden = [
        "deep_ep",
        "HybridEPBuffer",
        "ep_wave",
        "combined_forward_ep_wave",
        "ep_no_sync_rowwise",
        "rowwise_dispatch_put",
        "rowwise_combine_get",
        "symm_mem_vdev2d",
    ]
    for token in forbidden:
        assert token not in source


def test_tma_ibgda_kernel_shim_is_not_existing_symm_mem_extension():
    source = Path("src/olmo_core/kernels/tma_ibgda_ep.py").read_text()
    assert "_tma_ibgda_ep_ext_gpu" in source
    assert "_symm_mem_vdev2d_ext_gpu" not in source


def test_tma_ibgda_cuda_contract_is_separate_and_sized():
    contract_dir = Path("src/olmo_core/kernels/cuda/olmo_bf16_tma_ibgda_ep")
    metadata_source = (contract_dir / "metadata.cuh").read_text()
    workspace_source = (contract_dir / "workspace.cuh").read_text()
    readme_source = (contract_dir / "README.md").read_text()

    assert "namespace olmo::tma_ibgda_ep" in metadata_source
    assert "struct RouteRecord" in metadata_source
    assert "sizeof(RouteRecord) == 32" in metadata_source
    assert "struct WorkspaceLayout" in workspace_source
    assert "struct PeerWindowLayout" in workspace_source
    assert "struct PeerWindowView" in workspace_source
    assert "kWorkspaceAlignment = 128" in workspace_source
    assert "peer_payload_window" in workspace_source
    assert "wave / MegaMoE" in readme_source
    assert "NVSHMEM extension" in readme_source
