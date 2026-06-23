from pathlib import Path

import pytest
import torch

from olmo_core.nn.moe.v2.tma_ibgda import (
    TMA_IBGDA_COMPLETION_BYTES,
    TMA_IBGDA_DOORBELL_BYTES,
    TMA_IBGDA_ROUTE_RECORD_BYTES,
    TMA_IBGDA_WORKSPACE_ALIGNMENT,
    TmaIbgdaBackendConfig,
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
    assert hasattr(tma_ibgda_ep, "dispatch_bf16_peer")
    assert hasattr(tma_ibgda_ep, "preprocess_routes")
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
