import os
import struct
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh

from olmo_core.config import DType
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.layer_norm import LayerNormConfig, LayerNormType
from olmo_core.nn.moe import MoERouterGatingFunction
from olmo_core.nn.ddp.block import OLMoDDPTransformerBlock
from olmo_core.nn.moe.v2.ep_config import (
    ExpertParallelConfig,
    ExpertParallelPath,
    ExpertParallelSchedule,
)
from olmo_core.nn.moe.v2.routed_experts import RoutedExpertsConfig
from olmo_core.nn.moe.v2.router import MoERouterConfigV2
from olmo_core.nn.moe.v2.shared_experts import SharedExpertsConfig
from olmo_core.testing import requires_gpu, requires_grouped_gemm, requires_multi_gpu, run_distributed_test


def test_v2_extracted_forward_module_names_importable():
    from olmo_core.nn.moe.v2 import (
        activation_debug,
        checkpointing,
        ep_no_sync_1d,
        ep_no_sync_buffers,
        ep_no_sync_rowwise,
        ep_no_sync_rowwise_wave,
        ep_no_sync_tma_ibgda,
        ep_wave,
        ep_sync_1d,
        no_ep,
        tma_ibgda,
    )

    assert hasattr(activation_debug, "maybe_dump_ep_no_sync_saved_activations")
    assert hasattr(ep_sync_1d, "combined_forward_ep_1d")
    assert hasattr(no_ep, "combined_forward_no_ep")
    assert hasattr(checkpointing, "checkpoint_recompute_context_fn")
    assert hasattr(ep_no_sync_buffers, "get_ep_no_sync_buffers")
    assert hasattr(ep_no_sync_buffers, "_NoSyncSymmBuffers")
    assert hasattr(ep_no_sync_1d, "combined_forward_ep_no_sync_1d")
    assert hasattr(ep_no_sync_rowwise, "combined_forward_ep_no_sync_rowwise")
    assert hasattr(ep_no_sync_rowwise_wave, "combined_forward_ep_no_sync_rowwise_wave")
    assert hasattr(ep_no_sync_tma_ibgda, "combined_forward_ep_no_sync_tma_ibgda")
    assert hasattr(ep_wave, "combined_forward_ep_wave")
    assert hasattr(tma_ibgda, "tma_ibgda_rowwise_dispatch_bf16")


def _build_ep_mesh() -> DeviceMesh:
    world_size = dist.get_world_size()
    mesh = torch.arange(world_size, dtype=torch.int).view(1, world_size)
    return DeviceMesh(
        device_type="cuda",
        mesh=mesh,
        mesh_dim_names=("ep_dp", "ep_mp"),
    )


def _build_block(
    *,
    ep_no_sync: bool,
    ep: ExpertParallelConfig | None = None,
    ep_no_sync_capacity_factor: float = 2.0,
    d_model: int = 512,
    hidden_size: int = 1024,
    num_experts: int = 4,
    top_k: int = 1,
    num_shared_experts: int = 0,
    shared_hidden_size: int = 512,
    uniform_expert_assignment: bool = True,
    init_device: str = "cuda",
    checkpoint_combined_ep_tbo: bool = False,
    ep_no_sync_use_wave: bool = False,
    ep_no_sync_wave_use_bf16_persistent_mega_forward: bool = False,
    rowwise_fp8=None,
    ep_no_sync_use_2d_all_to_all: bool = False,
    ep_no_sync_use_rowwise_all_to_all: bool = False,
    ep_no_sync_rowwise_backend: str = "nvshmem",
    ep_no_sync_tma_ibgda_num_sms: int | None = None,
    ep_no_sync_tma_ibgda_symmetric_expert_out: bool = False,
) -> OLMoDDPTransformerBlock:
    if ep is None:
        rowwise_backend = ep_no_sync_rowwise_backend.lower()
        if rowwise_backend not in ("nvshmem", "tma_ibgda"):
            raise OLMoConfigurationError(
                "ep_no_sync_rowwise_backend must be one of 'nvshmem'|'tma_ibgda'"
            )
        if ep_no_sync_use_2d_all_to_all:
            path = ExpertParallelPath.no_sync_2d_removed
        elif not ep_no_sync:
            path = ExpertParallelPath.sync_1d
        elif ep_no_sync_use_wave:
            path = ExpertParallelPath.wave_mega
        elif ep_no_sync_use_rowwise_all_to_all:
            path = (
                ExpertParallelPath.rowwise_tma_ibgda
                if rowwise_backend == "tma_ibgda"
                else ExpertParallelPath.rowwise_nvshmem
            )
        else:
            path = ExpertParallelPath.no_sync_1d
        ep = ExpertParallelConfig(
            path=path,
            capacity_factor=ep_no_sync_capacity_factor,
            rowwise_nblocks=256,
            tma_ibgda_num_sms=ep_no_sync_tma_ibgda_num_sms,
            tma_ibgda_symmetric_expert_out=ep_no_sync_tma_ibgda_symmetric_expert_out,
            wave_use_bf16_persistent_mega_forward=(
                ep_no_sync_wave_use_bf16_persistent_mega_forward
            ),
            checkpoint_tbo=checkpoint_combined_ep_tbo,
        )

    layer_norm = LayerNormConfig(
        name=LayerNormType.rms,
        eps=1e-6,
        bias=False,
        dtype=DType.float32,
    )
    from olmo_core.nn.attention import AttentionConfig, AttentionType

    return OLMoDDPTransformerBlock(
        d_model=d_model,
        block_idx=0,
        n_layers=1,
        sequence_mixer=AttentionConfig(
            name=AttentionType.default,
            n_heads=2,
            n_kv_heads=2,
            bias=False,
            use_flash=False,
            dtype=DType.float32,
        ),
        attention_norm=layer_norm,
        routed_experts_router=MoERouterConfigV2(
            d_model=d_model,
            num_experts=num_experts,
            top_k=top_k,
            gating_function=MoERouterGatingFunction.softmax,
            uniform_expert_assignment=uniform_expert_assignment,
            lb_loss_weight=None,
            z_loss_weight=None,
            dtype=DType.float32,
        ),
        routed_experts=RoutedExpertsConfig(
            d_model=d_model,
            hidden_size=hidden_size,
            num_experts=num_experts,
            bias=False,
            dtype=DType.float32,
        ),
        shared_experts=(
            SharedExpertsConfig(
                d_model=d_model,
                hidden_size=shared_hidden_size,
                num_experts=num_shared_experts,
                bias=False,
                dtype=DType.float32,
            )
            if num_shared_experts > 0
            else None
        ),
        shared_experts_router=None,
        feed_forward_norm=layer_norm,
        ep=ep,
        rowwise_fp8=rowwise_fp8,
        init_device=init_device,
    )


def test_v2_ep_wave_legacy_flag_normalizes_to_wave_path():
    block = _build_block(
        ep_no_sync=True,
        ep_no_sync_use_wave=True,
        ep_no_sync_use_rowwise_all_to_all=False,
        init_device="cpu",
    )
    assert block.ep.path == ExpertParallelPath.wave_mega
    assert block.ep.no_sync is True
    assert block.ep.is_wave is True
    # Current wave/Mega reuses rowwise symmetric-buffer plumbing internally.
    assert block.ep.uses_rowwise_buffers is True


def test_v2_ep_config_selects_rowwise_tma_ibgda_path():
    block = _build_block(
        ep_no_sync=False,
        ep=ExpertParallelConfig(
            path=ExpertParallelPath.rowwise_tma_ibgda,
            tma_ibgda_num_sms=32,
        ),
        init_device="cpu",
    )
    assert block.ep.path == ExpertParallelPath.rowwise_tma_ibgda
    assert block.ep.no_sync is True
    assert block.ep.uses_rowwise_buffers is True
    assert block.ep.rowwise_transport == "tma_ibgda"
    assert block.ep.tma_ibgda_num_sms == 32


def test_v2_ep_config_selects_rowwise_wave_path():
    block = _build_block(
        ep_no_sync=False,
        ep=ExpertParallelConfig(
            path=ExpertParallelPath.rowwise_wave,
            rowwise_wave_num_waves=4,
            rowwise_wave_mode="EXPERT",
        ),
        init_device="cpu",
    )
    assert block.ep.path == ExpertParallelPath.rowwise_wave
    assert block.ep.no_sync is True
    assert block.ep.is_rowwise is True
    assert block.ep.is_wave is False
    assert block.ep.uses_rowwise_buffers is True
    assert block.ep.rowwise_transport == "nvshmem"
    assert block.ep.rowwise_wave_num_waves == 4
    assert block.ep.rowwise_wave_mode == "expert"


def test_v2_ep_config_tbo_only_allows_rowwise_nvshmem():
    ExpertParallelConfig(
        path=ExpertParallelPath.rowwise_nvshmem,
        schedule=ExpertParallelSchedule.tbo,
    ).validate()

    for path in (
        ExpertParallelPath.sync_1d,
        ExpertParallelPath.no_sync_1d,
        ExpertParallelPath.rowwise_tma_ibgda,
        ExpertParallelPath.rowwise_wave,
        ExpertParallelPath.wave_mega,
    ):
        with pytest.raises(OLMoConfigurationError, match="only supported"):
            ExpertParallelConfig(
                path=path,
                schedule=ExpertParallelSchedule.tbo,
            ).validate()


def test_v2_rowwise_nvshmem_tbo_forward_method_is_available():
    block = _build_block(
        ep_no_sync=True,
        ep_no_sync_use_rowwise_all_to_all=True,
        init_device="cpu",
    )
    assert block.ep.path == ExpertParallelPath.rowwise_nvshmem
    assert callable(block.combined_forward_rowwise_nvshmem_tbo)


def test_v2_tma_ibgda_backend_config_is_stored():
    block = _build_block(
        ep_no_sync=True,
        ep_no_sync_use_rowwise_all_to_all=True,
        ep_no_sync_rowwise_backend="tma_ibgda",
        init_device="cpu",
    )
    assert block.ep.rowwise_transport == "tma_ibgda"
    assert callable(block.combined_forward_ep_no_sync_tma_ibgda)


def test_v2_tma_ibgda_num_sms_config_is_stored():
    block = _build_block(
        ep_no_sync=True,
        ep_no_sync_use_rowwise_all_to_all=True,
        ep_no_sync_rowwise_backend="tma_ibgda",
        ep_no_sync_tma_ibgda_num_sms=32,
        init_device="cpu",
    )
    assert block.ep.tma_ibgda_num_sms == 32


def test_v2_tma_ibgda_stage_num_sms_config_is_stored():
    block = _build_block(
        ep_no_sync=False,
        ep=ExpertParallelConfig(
            path=ExpertParallelPath.rowwise_tma_ibgda,
            rowwise_nblocks=96,
            tma_ibgda_num_sms=64,
            tma_ibgda_dispatch_num_sms=128,
            tma_ibgda_combine_num_sms=160,
            tma_ibgda_preprocess_num_sms=192,
        ),
        init_device="cpu",
    )
    assert block.ep.resolved_tma_ibgda_dispatch_num_sms == 128
    assert block.ep.resolved_tma_ibgda_combine_num_sms == 160
    assert block.ep.resolved_tma_ibgda_preprocess_num_sms == 192


def test_v2_tma_ibgda_num_sms_is_stage_fallback():
    block = _build_block(
        ep_no_sync=False,
        ep=ExpertParallelConfig(
            path=ExpertParallelPath.rowwise_tma_ibgda,
            rowwise_nblocks=96,
            tma_ibgda_num_sms=64,
        ),
        init_device="cpu",
    )
    assert block.ep.resolved_tma_ibgda_dispatch_num_sms == 64
    assert block.ep.resolved_tma_ibgda_combine_num_sms == 64
    assert block.ep.resolved_tma_ibgda_preprocess_num_sms == 64


def test_v2_tma_ibgda_stage_num_sms_default_to_rowwise_nblocks_except_preprocess():
    block = _build_block(
        ep_no_sync=False,
        ep=ExpertParallelConfig(
            path=ExpertParallelPath.rowwise_tma_ibgda,
            rowwise_nblocks=96,
        ),
        init_device="cpu",
    )
    assert block.ep.resolved_tma_ibgda_dispatch_num_sms == 96
    assert block.ep.resolved_tma_ibgda_combine_num_sms == 96
    assert block.ep.resolved_tma_ibgda_preprocess_num_sms is None


def test_v2_tma_ibgda_symmetric_expert_out_config_is_stored():
    block = _build_block(
        ep_no_sync=True,
        ep_no_sync_use_rowwise_all_to_all=True,
        ep_no_sync_rowwise_backend="tma_ibgda",
        ep_no_sync_tma_ibgda_symmetric_expert_out=True,
        init_device="cpu",
    )
    assert block.ep.tma_ibgda_symmetric_expert_out is True


def test_v2_tma_ibgda_symmetric_expert_out_flag_detects_direct_buffer():
    from olmo_core.nn.moe.v2.ep_no_sync_tma_ibgda import (
        _resolve_symmetric_expert_out_flag,
    )

    expert_out = torch.empty((4, 8), dtype=torch.bfloat16)
    assert (
        _resolve_symmetric_expert_out_flag(
            requested=True,
            returned_expert_out=expert_out,
            symmetric_expert_out=expert_out,
        )
        is True
    )


def test_v2_tma_ibgda_symmetric_expert_out_flag_ignores_unrequested_buffer():
    from olmo_core.nn.moe.v2.ep_no_sync_tma_ibgda import (
        _resolve_symmetric_expert_out_flag,
    )

    expert_out = torch.empty((4, 8), dtype=torch.bfloat16)
    assert (
        _resolve_symmetric_expert_out_flag(
            requested=False,
            returned_expert_out=expert_out,
            symmetric_expert_out=None,
        )
        is False
    )


def test_v2_tma_ibgda_symmetric_expert_out_flag_rejects_missing_buffer():
    from olmo_core.nn.moe.v2.ep_no_sync_tma_ibgda import (
        _resolve_symmetric_expert_out_flag,
    )

    expert_out = torch.empty((4, 8), dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="no symmetric buffer"):
        _resolve_symmetric_expert_out_flag(
            requested=True,
            returned_expert_out=expert_out,
            symmetric_expert_out=None,
        )


def test_v2_tma_ibgda_symmetric_expert_out_flag_rejects_indirect_output():
    from olmo_core.nn.moe.v2.ep_no_sync_tma_ibgda import (
        _resolve_symmetric_expert_out_flag,
    )

    symmetric_expert_out = torch.empty((4, 8), dtype=torch.bfloat16)
    returned_expert_out = torch.empty_like(symmetric_expert_out)
    with pytest.raises(RuntimeError, match="returned a different tensor"):
        _resolve_symmetric_expert_out_flag(
            requested=True,
            returned_expert_out=returned_expert_out,
            symmetric_expert_out=symmetric_expert_out,
        )


def test_v2_tma_ibgda_num_sms_rejects_currently_unsafe_low_values():
    with pytest.raises(OLMoConfigurationError, match="tma_ibgda_num_sms"):
        _build_block(
            ep_no_sync=True,
            ep_no_sync_use_rowwise_all_to_all=True,
            ep_no_sync_rowwise_backend="tma_ibgda",
            ep_no_sync_tma_ibgda_num_sms=16,
            init_device="cpu",
        )


def test_v2_tma_ibgda_num_sms_requires_tma_ibgda_path():
    with pytest.raises(OLMoConfigurationError, match="tma_ibgda_num_sms"):
        ExpertParallelConfig(
            path=ExpertParallelPath.rowwise_nvshmem,
            tma_ibgda_num_sms=32,
        ).validate()


def test_v2_tma_ibgda_stage_num_sms_requires_tma_ibgda_path():
    for field_name in (
        "tma_ibgda_dispatch_num_sms",
        "tma_ibgda_combine_num_sms",
        "tma_ibgda_preprocess_num_sms",
    ):
        with pytest.raises(OLMoConfigurationError, match=field_name):
            ExpertParallelConfig(
                path=ExpertParallelPath.rowwise_nvshmem,
                **{field_name: 32},
            ).validate()


def test_v2_tma_ibgda_stage_num_sms_rejects_currently_unsafe_low_values():
    for field_name in (
        "tma_ibgda_dispatch_num_sms",
        "tma_ibgda_combine_num_sms",
        "tma_ibgda_preprocess_num_sms",
    ):
        with pytest.raises(OLMoConfigurationError, match=field_name):
            ExpertParallelConfig(
                path=ExpertParallelPath.rowwise_tma_ibgda,
                **{field_name: 16},
            ).validate()


def test_v2_tma_ibgda_symmetric_expert_out_requires_tma_ibgda_path():
    with pytest.raises(OLMoConfigurationError, match="tma_ibgda_symmetric_expert_out"):
        ExpertParallelConfig(
            path=ExpertParallelPath.wave_mega,
            tma_ibgda_symmetric_expert_out=True,
        ).validate()


def test_v2_rowwise_wave_num_waves_requires_rowwise_wave_path():
    with pytest.raises(OLMoConfigurationError, match="rowwise_wave_num_waves"):
        ExpertParallelConfig(
            path=ExpertParallelPath.rowwise_nvshmem,
            rowwise_wave_num_waves=2,
        ).validate()


def test_v2_rowwise_wave_rejects_invalid_mode_and_num_waves():
    with pytest.raises(OLMoConfigurationError, match="rowwise_wave_num_waves"):
        ExpertParallelConfig(
            path=ExpertParallelPath.rowwise_wave,
            rowwise_wave_num_waves=0,
        ).validate()

    with pytest.raises(OLMoConfigurationError, match="rowwise_wave_mode"):
        ExpertParallelConfig(
            path=ExpertParallelPath.rowwise_wave,
            rowwise_wave_mode="token",
        ).validate()


def test_v2_tma_ibgda_backend_rejects_unknown_backend():
    with pytest.raises(OLMoConfigurationError, match="ep_no_sync_rowwise_backend"):
        _build_block(
            ep_no_sync=True,
            ep_no_sync_use_rowwise_all_to_all=True,
            ep_no_sync_rowwise_backend="unknown",
            init_device="cpu",
        )


def test_v2_ep_wave_forward_method_is_available():
    block = _build_block(
        ep_no_sync=True,
        ep_no_sync_use_wave=True,
        ep_no_sync_use_rowwise_all_to_all=True,
        init_device="cpu",
    )
    assert callable(block.combined_forward_ep_wave)


def test_v2_rowwise_wave_forward_method_is_available():
    block = _build_block(
        ep_no_sync=False,
        ep=ExpertParallelConfig(
            path=ExpertParallelPath.rowwise_wave,
            rowwise_wave_num_waves=2,
        ),
        init_device="cpu",
    )
    assert callable(block.combined_forward_ep_no_sync_rowwise_wave)


def test_v2_ep_wave_rejects_tbo_checkpointing():
    with pytest.raises(OLMoConfigurationError, match="only supported"):
        _build_block(
            ep_no_sync=True,
            ep_no_sync_use_wave=True,
            checkpoint_combined_ep_tbo=True,
            ep_no_sync_use_rowwise_all_to_all=True,
            init_device="cpu",
        )


def test_v2_ep_wave_bf16_persistent_mega_forward_config_is_stored():
    block = _build_block(
        ep_no_sync=True,
        ep_no_sync_use_wave=True,
        ep_no_sync_wave_use_bf16_persistent_mega_forward=True,
        ep_no_sync_use_rowwise_all_to_all=True,
        init_device="cpu",
    )
    assert block.ep.wave_use_bf16_persistent_mega_forward is True


def test_v2_ep_wave_bf16_persistent_mega_requires_wave_backend():
    with pytest.raises(
        OLMoConfigurationError,
        match="wave_use_bf16_persistent_mega_forward=True",
    ):
        _build_block(
            ep_no_sync=True,
            ep_no_sync_use_wave=False,
            ep_no_sync_wave_use_bf16_persistent_mega_forward=True,
            ep_no_sync_use_rowwise_all_to_all=True,
            init_device="cpu",
        )


def test_v2_ep_wave_bf16_persistent_mega_cpu_symbol_fails_closed():
    from olmo_core.kernels.symm_mem_vdev2d import (
        rowwise_bf16_mega_moe_forward_persistent,
    )

    source_input = torch.empty((1, 1), dtype=torch.bfloat16)
    gathered_out = torch.empty((1, 1, 1), dtype=torch.bfloat16)
    out = torch.empty((1, 1), dtype=torch.bfloat16)
    route_dst_ranks = torch.zeros((1, 1), dtype=torch.long)
    route_dst_rows = torch.zeros((1, 1), dtype=torch.long)
    route_expert_indices = torch.zeros((1, 1), dtype=torch.long)
    probs = torch.ones((1, 1), dtype=torch.float32)
    up_gate_weight = torch.empty((1, 2, 1), dtype=torch.bfloat16)
    down_weight = torch.empty((1, 1, 1), dtype=torch.bfloat16)
    expert_offsets = torch.ones((1,), dtype=torch.int32)

    try:
        with pytest.raises(RuntimeError, match="unsupported"):
            rowwise_bf16_mega_moe_forward_persistent(
                source_input,
                gathered_out,
                out,
                route_dst_ranks,
                route_dst_rows,
                route_expert_indices,
                probs,
                up_gate_weight,
                down_weight,
                expert_offsets,
                "unused",
            )
    except RuntimeError as e:
        if "extension is unavailable" in str(e):
            pytest.skip("symm_mem_vdev2d CUDA extension is not built")
        raise


def test_v2_ep_wave_bf16_persistent_mega_standard_config():
    from olmo_core.kernels.symm_mem_vdev2d import (
        rowwise_bf16_mega_moe_forward_config,
    )

    try:
        cfg = rowwise_bf16_mega_moe_forward_config(
            num_rows=8192 * 2,
            top_k=4,
            hidden=4096,
            intermediate=4096,
            num_local_experts=8,
            num_sms=120,
        )
    except RuntimeError as e:
        if "extension is unavailable" in str(e):
            pytest.skip("symm_mem_vdev2d CUDA extension is not built")
        raise

    assert cfg["block_m"] == 128
    assert cfg["block_n"] == 128
    assert cfg["block_k"] == 64
    assert cfg["num_experts_per_wave"] == 1
    assert cfg["num_expert_waves"] == 8
    assert cfg["num_max_pool_tokens"] > 0
    assert cfg["workspace_bytes"] > 0
    assert cfg["runtime_buffer_bytes"] > 0
    assert cfg["total_symmetric_bytes"] == cfg["workspace_bytes"] + cfg["runtime_buffer_bytes"]
    assert cfg["num_stages"] > 0
    assert cfg["smem_size"] > 0
    assert cfg["f1_dispatch_sms"] > 0
    assert cfg["f1_finalize_sms"] > 0
    assert cfg["f1_gemm_sms"] > 0
    assert cfg["f1_expected_tokens_per_expert"] == 8192
    assert cfg["f1_gemm_m_tiles_per_expert"] == 64
    assert cfg["f1_gemm_n_tiles"] == 64
    assert cfg["f1_dispatch_route_tasks"] == 256
    assert cfg["f1_finalize_expert_tasks"] == 8
    assert cfg["f1_gemm_tasks"] == 32768
    assert cfg["f1_total_tasks"] == (
        cfg["f1_dispatch_route_tasks"]
        + cfg["f1_finalize_expert_tasks"]
        + cfg["f1_gemm_tasks"]
    )
    assert cfg["f2_combine_sms"] > 0
    assert cfg["f2_reduce_sms"] > 0
    assert cfg["f2_gemm_sms"] > 0
    assert cfg["f2_expected_tokens_per_expert"] == cfg["f1_expected_tokens_per_expert"]
    assert cfg["f2_gemm_m_tiles_per_expert"] == cfg["f1_gemm_m_tiles_per_expert"]
    assert cfg["f2_gemm_n_tiles"] == 32
    assert cfg["f2_combine_scatter_tasks"] == 256
    assert cfg["f2_combine_reduce_tasks"] == 64
    assert cfg["f2_gemm_tasks"] == 16384
    assert cfg["f2_total_tasks"] == (
        cfg["f2_gemm_tasks"]
        + cfg["f2_combine_scatter_tasks"]
        + cfg["f2_combine_reduce_tasks"]
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_v2_ep_wave_bf16_sm100_tma_umma_contract_device_contract():
    if torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("requires Blackwell-class CUDA device")

    from olmo_core.kernels.symm_mem_vdev2d import (
        rowwise_bf16_mega_moe_sm100_tma_umma_contract_debug,
    )

    try:
        debug = rowwise_bf16_mega_moe_sm100_tma_umma_contract_debug().cpu().tolist()
    except RuntimeError as e:
        if "extension is unavailable" in str(e):
            pytest.skip("symm_mem_vdev2d CUDA extension is not built")
        raise

    assert debug[0] >= 1000
    assert debug[2:10] == [128, 128, 64, 4096, 4096, 32, 4, 4]
    assert debug[10:13] == [1, 1, 1]
    if debug[1]:
        assert debug[15] == 1
    else:
        assert debug[15] == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_v2_ep_wave_bf16_sm100_tma_load_contract_device_contract():
    if torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("requires TMA-capable CUDA device")

    from olmo_core.kernels.symm_mem_vdev2d import (
        rowwise_bf16_mega_moe_sm100_tma_load_contract_debug,
    )

    source = torch.arange(16 * 64, device="cuda", dtype=torch.float32).to(torch.bfloat16)
    source = source.view(16, 64).contiguous()
    try:
        debug = rowwise_bf16_mega_moe_sm100_tma_load_contract_debug(source).cpu().tolist()
    except RuntimeError as e:
        if "extension is unavailable" in str(e):
            pytest.skip("symm_mem_vdev2d CUDA extension is not built")
        raise

    assert debug[0] >= 900
    assert debug[1] == 1
    assert debug[2:8] == [0, 1, 63, 64, 16, 64]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_v2_ep_wave_bf16_sm100_tma_umma_tile_contract_device_contract():
    if torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("requires Blackwell-class CUDA device")

    from olmo_core.kernels.symm_mem_vdev2d import (
        rowwise_bf16_mega_moe_sm100_tma_umma_tile_contract_debug,
    )

    a = torch.ones((128, 64), device="cuda", dtype=torch.bfloat16)
    b = torch.ones((128, 64), device="cuda", dtype=torch.bfloat16)
    try:
        debug = rowwise_bf16_mega_moe_sm100_tma_umma_tile_contract_debug(a, b).cpu().tolist()
    except RuntimeError as e:
        if "extension is unavailable" in str(e):
            pytest.skip("symm_mem_vdev2d CUDA extension is not built")
        raise

    expected_raw = struct.unpack("<I", struct.pack("<f", 64.0))[0]
    assert debug[0] >= 1000
    assert debug[1] == 1
    assert debug[2:6] == [128, 128, 64, 128]
    assert debug[7:11] == [1, 1, 1, 1]
    assert debug[12] == 1
    assert debug[13] == 1
    assert debug[16 : 16 + 32 * 8] == [expected_raw] * (32 * 8)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_v2_ep_wave_bf16_sm100_tma_umma_tile_forward_matches_torch():
    if torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("requires Blackwell-class CUDA device")

    from olmo_core.kernels.symm_mem_vdev2d import (
        rowwise_bf16_mega_moe_sm100_tma_umma_tile_forward_debug,
    )

    a_rows = (torch.arange(128, device="cuda", dtype=torch.float32) % 4 + 1).to(torch.bfloat16)
    b_rows = (torch.arange(128, device="cuda", dtype=torch.float32) % 8 + 1).to(torch.bfloat16)
    a = a_rows[:, None].expand(128, 64).contiguous()
    b = b_rows[:, None].expand(128, 64).contiguous()
    try:
        out = rowwise_bf16_mega_moe_sm100_tma_umma_tile_forward_debug(a, b)
    except RuntimeError as e:
        if "extension is unavailable" in str(e):
            pytest.skip("symm_mem_vdev2d CUDA extension is not built")
        raise

    expected = a.float() @ b.float().T
    torch.testing.assert_close(out, expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_v2_ep_wave_bf16_sm100_tma_umma_tile_forward_b_mn_matches_torch():
    if torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("requires Blackwell-class CUDA device")

    from olmo_core.kernels.symm_mem_vdev2d import (
        rowwise_bf16_mega_moe_sm100_tma_umma_tile_forward_b_mn_debug,
    )

    a_rows = (torch.arange(128, device="cuda", dtype=torch.float32) % 4 + 1).to(torch.bfloat16)
    b_cols = (torch.arange(128, device="cuda", dtype=torch.float32) % 8 + 1).to(torch.bfloat16)
    a = a_rows[:, None].expand(128, 64).contiguous()
    b = b_cols[None, :].expand(64, 128).contiguous()
    try:
        out = rowwise_bf16_mega_moe_sm100_tma_umma_tile_forward_b_mn_debug(a, b)
    except RuntimeError as e:
        if "extension is unavailable" in str(e):
            pytest.skip("symm_mem_vdev2d CUDA extension is not built")
        raise

    expected = a.float() @ b.float()
    torch.testing.assert_close(out, expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_v2_ep_wave_bf16_forward_plan_device_contract():
    from olmo_core.kernels.symm_mem_vdev2d import (
        rowwise_bf16_mega_moe_forward_plan_debug,
    )

    debug = rowwise_bf16_mega_moe_forward_plan_debug(
        num_rows=8192 * 2,
        top_k=4,
        hidden=4096,
        intermediate=4096,
        num_local_experts=8,
        num_sms=120,
    ).cpu()

    f1_counts = debug[0].tolist()
    f2_counts = debug[1].tolist()
    assert f1_counts[:7] == [0, 256, 8, 32768, 0, 0, 0]
    assert f1_counts[7] == 0
    assert f2_counts[:7] == [0, 0, 0, 0, 16384, 256, 64]
    assert f2_counts[7] == 0

    # [kind, ordinal, local_expert, m_tile, n_tile, route_task, total_tasks, gemm_tasks]
    assert debug[2].tolist() == [3, 32767, 7, 63, 63, -1, 33032, 32768]
    assert debug[3].tolist() == [4, 16383, 7, 63, 31, -1, 16704, 16384]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_v2_ep_wave_bf16_route_count_device_contract():
    from olmo_core.kernels.symm_mem_vdev2d import (
        rowwise_bf16_mega_moe_route_counts_debug,
    )

    device = torch.device("cuda", torch.cuda.current_device())
    route_expert_indices = torch.tensor(
        [
            [0, 1, 2, 3],
            [1, 2, -1, 99],
            [3, 3, 0, 2],
        ],
        device=device,
        dtype=torch.long,
    )
    counts = rowwise_bf16_mega_moe_route_counts_debug(
        route_expert_indices,
        num_local_experts=4,
    ).cpu()
    assert counts.tolist() == [2, 2, 3, 3]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_v2_ep_wave_bf16_route_pack_device_contract():
    from olmo_core.kernels.symm_mem_vdev2d import (
        rowwise_bf16_mega_moe_route_pack_debug,
    )

    device = torch.device("cuda", torch.cuda.current_device())
    route_expert_indices = torch.tensor(
        [
            [0, 1, 2, 3],
            [1, 2, -1, 99],
            [3, 3, 0, 2],
        ],
        device=device,
        dtype=torch.long,
    )
    expert_offsets, packed = rowwise_bf16_mega_moe_route_pack_debug(
        route_expert_indices,
        num_local_experts=4,
    )
    expert_offsets = expert_offsets.cpu()
    packed = packed.cpu()

    assert expert_offsets.tolist() == [0, 2, 4, 7, 10]
    expected_by_expert = {
        0: [0, 10],
        1: [1, 4],
        2: [2, 5, 11],
        3: [3, 8, 9],
    }
    for expert_idx, expected in expected_by_expert.items():
        start = expert_offsets[expert_idx].item()
        end = expert_offsets[expert_idx + 1].item()
        assert sorted(packed[start:end].tolist()) == expected


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_v2_ep_wave_bf16_route_pack_inputs_device_contract():
    from olmo_core.kernels.symm_mem_vdev2d import (
        rowwise_bf16_mega_moe_route_pack_inputs_debug,
    )

    device = torch.device("cuda", torch.cuda.current_device())
    source_input = torch.arange(12, device=device, dtype=torch.float32).view(3, 4).to(torch.bfloat16)
    route_expert_indices = torch.tensor(
        [
            [0, 1],
            [1, -1],
            [0, 1],
        ],
        device=device,
        dtype=torch.long,
    )
    probs = torch.tensor(
        [
            [0.1, 0.2],
            [0.3, 0.4],
            [0.5, 0.6],
        ],
        device=device,
        dtype=torch.float32,
    )
    expert_offsets, packed_route, packed_input, packed_probs = (
        rowwise_bf16_mega_moe_route_pack_inputs_debug(
            source_input,
            route_expert_indices,
            probs,
            num_local_experts=2,
        )
    )

    expert_offsets = expert_offsets.cpu()
    packed_route = packed_route.cpu()
    packed_input = packed_input.cpu()
    packed_probs = packed_probs.cpu()

    assert expert_offsets.tolist() == [0, 2, 5]
    for expert_idx, expected_routes in {0: [0, 4], 1: [1, 2, 5]}.items():
        start = expert_offsets[expert_idx].item()
        end = expert_offsets[expert_idx + 1].item()
        seen = sorted(packed_route[start:end].tolist())
        assert seen == expected_routes
        for route_idx in seen:
            slot = packed_route.tolist().index(route_idx)
            token_idx = route_idx // route_expert_indices.size(1)
            assert packed_input[slot].tolist() == source_input[token_idx].cpu().tolist()
            assert packed_probs[slot].item() == pytest.approx(probs.flatten()[route_idx].item())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_v2_ep_wave_peer_route_metadata_matches_tma_ibgda_reference():
    from olmo_core.kernels.symm_mem_vdev2d import (
        rowwise_bf16_mega_moe_peer_route_metadata_debug,
    )
    from olmo_core.nn.moe.v2.tma_ibgda import build_tma_ibgda_route_metadata

    device = torch.device("cuda", torch.cuda.current_device())
    dst_ranks = torch.tensor(
        [
            [0, 1, -1],
            [1, 0, 1],
        ],
        device=device,
        dtype=torch.long,
    )
    dst_rows = torch.tensor(
        [
            [0, 0, -1],
            [1, 2, 3],
        ],
        device=device,
        dtype=torch.long,
    )
    probs = torch.tensor(
        [
            [0.5, 0.25, 1.0],
            [0.75, 0.125, 0.625],
        ],
        device=device,
        dtype=torch.float32,
    )
    try:
        (
            route_records_i32,
            route_record_probs,
            routes_per_rank,
            rank_offsets,
            overflow_by_rank,
        ) = rowwise_bf16_mega_moe_peer_route_metadata_debug(
            dst_ranks,
            dst_rows,
            probs,
            ep_world_size=2,
            rank_capacity=4,
            static_route_budget=2,
        )
    except RuntimeError as e:
        if "extension is unavailable" in str(e):
            pytest.skip("symm_mem_vdev2d CUDA extension is not built")
        raise

    reference = build_tma_ibgda_route_metadata(
        dst_ranks,
        dst_rows,
        ep_world_size=2,
        rank_capacity=4,
        static_route_budget=2,
    )
    torch.testing.assert_close(routes_per_rank.cpu(), reference.routes_per_rank.cpu(), rtol=0, atol=0)
    torch.testing.assert_close(rank_offsets.cpu(), reference.rank_offsets.cpu(), rtol=0, atol=0)
    torch.testing.assert_close(
        overflow_by_rank.cpu().to(dtype=torch.bool),
        reference.overflow_by_rank.cpu(),
        rtol=0,
        atol=0,
    )

    records = route_records_i32.cpu()
    record_probs = route_record_probs.cpu()
    flat_ranks = dst_ranks.cpu().reshape(-1)
    flat_rows = dst_rows.cpu().reshape(-1)
    flat_probs = probs.cpu().reshape(-1)
    for rank in range(reference.ep_world_size):
        start = int(rank_offsets.cpu()[rank].item())
        end = int(rank_offsets.cpu()[rank + 1].item())
        got = sorted(
            (
                int(records[idx, 0].item()),
                int(records[idx, 1].item()),
                int(records[idx, 2].item()),
                int(records[idx, 3].item()),
                round(float(record_probs[idx].item()), 6),
            )
            for idx in range(start, end)
        )
        expected = []
        for route_idx, (dst_rank, dst_row) in enumerate(zip(flat_ranks, flat_rows)):
            if int(dst_rank.item()) != rank or int(dst_row.item()) < 0:
                continue
            source_row = route_idx // reference.top_k
            topk_slot = route_idx % reference.top_k
            expected.append(
                (
                    source_row,
                    topk_slot,
                    rank,
                    int(dst_row.item()),
                    round(float(flat_probs[route_idx].item()), 6),
                )
            )
        assert got == sorted(expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_v2_ep_wave_peer_window_dispatch_combine_matches_reference():
    from olmo_core.kernels.symm_mem_vdev2d import (
        rowwise_bf16_mega_moe_peer_window_combine_debug,
        rowwise_bf16_mega_moe_peer_window_dispatch_debug,
    )
    from olmo_core.nn.moe.v2.tma_ibgda import (
        reference_combine_bf16,
        reference_dispatch_bf16,
    )

    device = torch.device("cuda", torch.cuda.current_device())
    source_input = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0],
            [3.0, 5.0, 7.0, 11.0],
        ],
        device=device,
        dtype=torch.bfloat16,
    )
    dst_ranks = torch.tensor([[0, 1, -1], [1, 0, 1]], device=device, dtype=torch.long)
    dst_rows = torch.tensor([[0, 0, -1], [1, 2, 3]], device=device, dtype=torch.long)
    probs = torch.tensor(
        [[0.5, 0.25, 1.0], [0.75, 0.125, 0.625]],
        device=device,
        dtype=torch.float32,
    )

    try:
        dispatched = rowwise_bf16_mega_moe_peer_window_dispatch_debug(
            source_input,
            dst_ranks,
            dst_rows,
            ep_world_size=2,
            rank_capacity=4,
        )
        gathered, combined = rowwise_bf16_mega_moe_peer_window_combine_debug(
            dispatched,
            dst_ranks,
            dst_rows,
            probs,
        )
    except RuntimeError as e:
        if "extension is unavailable" in str(e):
            pytest.skip("symm_mem_vdev2d CUDA extension is not built")
        raise

    expected_dispatch = reference_dispatch_bf16(
        source_input,
        dst_ranks,
        dst_rows,
        ep_world_size=2,
        rank_capacity=4,
    )
    expected_combined = reference_combine_bf16(
        expected_dispatch,
        dst_ranks,
        dst_rows,
        probs=probs,
    )
    expected_gathered = torch.zeros_like(gathered)
    for token_idx in range(dst_ranks.shape[0]):
        for topk_idx in range(dst_ranks.shape[1]):
            rank = int(dst_ranks[token_idx, topk_idx].item())
            row = int(dst_rows[token_idx, topk_idx].item())
            if rank >= 0 and row >= 0:
                expected_gathered[token_idx, topk_idx] = expected_dispatch[rank, row]

    torch.testing.assert_close(dispatched, expected_dispatch, rtol=0, atol=0)
    torch.testing.assert_close(gathered, expected_gathered, rtol=0, atol=0)
    torch.testing.assert_close(combined, expected_combined, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_v2_ep_wave_bf16_grouped_gemm_metadata_device_contract():
    from olmo_core.kernels.symm_mem_vdev2d import (
        rowwise_bf16_mega_moe_grouped_gemm_metadata_debug,
    )

    device = torch.device("cuda", torch.cuda.current_device())
    route_expert_indices = torch.tensor(
        [
            [0, 1, 2, 3],
            [1, 2, -1, 99],
            [3, 3, 0, 2],
        ],
        device=device,
        dtype=torch.long,
    )
    expert_counts, token_offsets, tile_counts, tile_offsets, num_total_m_tiles = (
        rowwise_bf16_mega_moe_grouped_gemm_metadata_debug(
            route_expert_indices,
            num_local_experts=4,
            block_m=2,
        )
    )

    assert expert_counts.cpu().tolist() == [2, 2, 3, 3]
    assert token_offsets.cpu().tolist() == [0, 2, 4, 7, 10]
    assert tile_counts.cpu().tolist() == [1, 1, 2, 2]
    assert tile_offsets.cpu().tolist() == [0, 1, 2, 4, 6]
    assert num_total_m_tiles.cpu().tolist() == [6]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_v2_ep_wave_bf16_grouped_gemm_tile_device_contract():
    from olmo_core.kernels.symm_mem_vdev2d import (
        rowwise_bf16_mega_moe_grouped_gemm_tile_debug,
    )

    device = torch.device("cuda", torch.cuda.current_device())
    route_expert_indices = torch.tensor(
        [
            [0, 1, 2, 3],
            [1, 2, -1, 99],
            [3, 3, 0, 2],
        ],
        device=device,
        dtype=torch.long,
    )
    *_, debug = rowwise_bf16_mega_moe_grouped_gemm_tile_debug(
        route_expert_indices,
        num_local_experts=4,
        block_m=2,
        n_tiles=3,
    )
    debug = debug.cpu()

    assert debug[:4, 0].tolist() == [3, 3, 6, 6]
    assert debug[4].tolist() == [3, 5, 1, 2]
    assert debug[5].tolist() == [18, 6, 3, 36]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_v2_ep_wave_bf16_combine_device_contract():
    from olmo_core.kernels.symm_mem_vdev2d import (
        rowwise_bf16_mega_moe_combine_debug,
    )

    device = torch.device("cuda", torch.cuda.current_device())
    packed_expert_out = torch.tensor(
        [
            [1.0, 2.0],
            [10.0, 20.0],
            [3.0, 4.0],
            [30.0, 40.0],
        ],
        device=device,
        dtype=torch.bfloat16,
    )
    packed_token_topk_indices = torch.tensor([0, 2, 1, 3], device=device, dtype=torch.long)
    probs = torch.tensor(
        [
            [0.25, 0.75],
            [0.5, 0.5],
        ],
        device=device,
        dtype=torch.float32,
    )

    gathered_out, out = rowwise_bf16_mega_moe_combine_debug(
        packed_expert_out,
        packed_token_topk_indices,
        probs,
    )

    assert gathered_out.cpu().float().tolist() == [
        [[1.0, 2.0], [3.0, 4.0]],
        [[10.0, 20.0], [30.0, 40.0]],
    ]
    expected = torch.tensor(
        [
            [2.5, 3.5],
            [20.0, 30.0],
        ],
        dtype=torch.bfloat16,
    )
    assert torch.testing.assert_close(out.cpu(), expected) is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_v2_ep_wave_bf16_w1_wmma_device_contract():
    from olmo_core.kernels.symm_mem_vdev2d import (
        rowwise_bf16_mega_moe_w1_wmma_debug,
    )

    device = torch.device("cuda", torch.cuda.current_device())
    source_input = torch.arange(64, device=device, dtype=torch.float32).view(4, 16).to(torch.bfloat16)
    route_expert_indices = torch.tensor(
        [
            [0],
            [1],
            [0],
            [1],
        ],
        device=device,
        dtype=torch.long,
    )
    up_gate_weight = torch.arange(
        2 * 16 * 16,
        device=device,
        dtype=torch.float32,
    ).view(2, 16, 16).to(torch.bfloat16)

    *_, packed_route, packed_input, w1_out = rowwise_bf16_mega_moe_w1_wmma_debug(
        source_input,
        route_expert_indices,
        up_gate_weight,
    )
    packed_route_cpu = packed_route.cpu()
    packed_input_cpu = packed_input.cpu().float()
    w1_out_cpu = w1_out.cpu().float()

    for slot, route_idx in enumerate(packed_route_cpu.tolist()):
        token_idx = route_idx
        expert_idx = route_expert_indices.flatten()[route_idx].item()
        expected = packed_input_cpu[slot].to(torch.float32) @ up_gate_weight[expert_idx].cpu().float().T
        torch.testing.assert_close(w1_out_cpu[slot], expected, rtol=0.02, atol=2.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_v2_ep_wave_bf16_forward_debug_device_contract():
    from olmo_core.kernels.symm_mem_vdev2d import (
        rowwise_bf16_mega_moe_forward_debug,
    )

    device = torch.device("cuda", torch.cuda.current_device())
    torch.manual_seed(3107)
    tokens = 4
    top_k = 2
    hidden = 16
    intermediate = 16
    experts = 2
    source_input = (0.2 * torch.randn(tokens, hidden, device=device)).to(torch.bfloat16)
    route_expert_indices = torch.tensor(
        [
            [0, 1],
            [1, 0],
            [0, 0],
            [1, 1],
        ],
        device=device,
        dtype=torch.long,
    )
    probs = torch.tensor(
        [
            [0.6, 0.4],
            [0.25, 0.75],
            [0.7, 0.3],
            [0.5, 0.5],
        ],
        device=device,
        dtype=torch.float32,
    )
    up_gate_weight = (
        0.15 * torch.randn(experts, 2 * intermediate, hidden, device=device)
    ).to(torch.bfloat16)
    down_weight = (
        0.15 * torch.randn(experts, intermediate, hidden, device=device)
    ).to(torch.bfloat16)

    (
        *_,
        packed_token_topk_indices,
        packed_input,
        w1_out,
        h,
        packed_expert_out,
        gathered_out,
        out,
    ) = rowwise_bf16_mega_moe_forward_debug(
        source_input,
        route_expert_indices,
        probs,
        up_gate_weight,
        down_weight,
    )

    assert tuple(packed_token_topk_indices.shape) == (tokens * top_k,)
    assert tuple(packed_input.shape) == (tokens * top_k, hidden)
    assert tuple(w1_out.shape) == (tokens * top_k, 2 * intermediate)
    assert tuple(h.shape) == (tokens * top_k, intermediate)
    assert tuple(packed_expert_out.shape) == (tokens * top_k, hidden)
    assert tuple(gathered_out.shape) == (tokens, top_k, hidden)
    assert tuple(out.shape) == (tokens, hidden)

    expected = torch.zeros(tokens, hidden, device=device, dtype=torch.float32)
    expected_gathered = torch.zeros(tokens, top_k, hidden, device=device, dtype=torch.bfloat16)
    for token_idx in range(tokens):
        for topk_idx in range(top_k):
            expert_idx = int(route_expert_indices[token_idx, topk_idx].item())
            up_gate = (
                source_input[token_idx].float() @ up_gate_weight[expert_idx].float().T
            ).to(torch.bfloat16).float()
            up = up_gate[:intermediate]
            gate = up_gate[intermediate:]
            expert_h = (up * torch.nn.functional.silu(gate)).to(torch.bfloat16).float()
            expert_out = (expert_h @ down_weight[expert_idx].float()).to(torch.bfloat16)
            expected_gathered[token_idx, topk_idx] = expert_out
            expected[token_idx] += probs[token_idx, topk_idx] * expert_out.float()
    torch.testing.assert_close(gathered_out, expected_gathered, rtol=0.03, atol=0.03)
    torch.testing.assert_close(out, expected.to(torch.bfloat16), rtol=0.03, atol=0.03)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_v2_ep_wave_bf16_local_persistent_forward_debug_device_contract():
    from olmo_core.kernels.symm_mem_vdev2d import (
        rowwise_bf16_mega_moe_forward_debug,
        rowwise_bf16_mega_moe_local_persistent_forward_debug,
    )

    device = torch.device("cuda", torch.cuda.current_device())
    torch.manual_seed(3108)
    tokens = 4
    top_k = 2
    hidden = 16
    intermediate = 16
    experts = 2
    source_input = (0.2 * torch.randn(tokens, hidden, device=device)).to(torch.bfloat16)
    route_expert_indices = torch.tensor(
        [
            [0, 1],
            [1, 0],
            [0, 0],
            [1, 1],
        ],
        device=device,
        dtype=torch.long,
    )
    probs = torch.tensor(
        [
            [0.6, 0.4],
            [0.25, 0.75],
            [0.7, 0.3],
            [0.5, 0.5],
        ],
        device=device,
        dtype=torch.float32,
    )
    up_gate_weight = (
        0.15 * torch.randn(experts, 2 * intermediate, hidden, device=device)
    ).to(torch.bfloat16)
    down_weight = (
        0.15 * torch.randn(experts, intermediate, hidden, device=device)
    ).to(torch.bfloat16)

    staged = rowwise_bf16_mega_moe_forward_debug(
        source_input,
        route_expert_indices,
        probs,
        up_gate_weight,
        down_weight,
    )
    persistent = rowwise_bf16_mega_moe_local_persistent_forward_debug(
        source_input,
        route_expert_indices,
        probs,
        up_gate_weight,
        down_weight,
    )

    staged_h, staged_packed_out, staged_gathered, staged_out = staged[-4:]
    persistent_h, persistent_packed_out, persistent_gathered, persistent_out = persistent[-5:-1]
    barrier_state = persistent[-1]
    torch.testing.assert_close(persistent_h, staged_h, rtol=0.03, atol=0.03)
    torch.testing.assert_close(persistent_packed_out, staged_packed_out, rtol=0.03, atol=0.03)
    torch.testing.assert_close(persistent_gathered, staged_gathered, rtol=0.03, atol=0.03)
    torch.testing.assert_close(persistent_out, staged_out, rtol=0.03, atol=0.03)
    assert int(barrier_state.cpu()[0].item()) == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_v2_ep_wave_bf16_local_full_forward_megakernel_debug_device_contract():
    from olmo_core.kernels.symm_mem_vdev2d import (
        rowwise_bf16_mega_moe_forward_debug,
        rowwise_bf16_mega_moe_local_full_forward_megakernel_debug,
    )

    device = torch.device("cuda", torch.cuda.current_device())
    torch.manual_seed(3109)
    tokens = 4
    top_k = 2
    hidden = 16
    intermediate = 16
    experts = 2
    source_input = (0.2 * torch.randn(tokens, hidden, device=device)).to(torch.bfloat16)
    route_expert_indices = torch.tensor(
        [
            [0, 1],
            [1, 0],
            [0, 0],
            [1, 1],
        ],
        device=device,
        dtype=torch.long,
    )
    probs = torch.tensor(
        [
            [0.6, 0.4],
            [0.25, 0.75],
            [0.7, 0.3],
            [0.5, 0.5],
        ],
        device=device,
        dtype=torch.float32,
    )
    up_gate_weight = (
        0.15 * torch.randn(experts, 2 * intermediate, hidden, device=device)
    ).to(torch.bfloat16)
    down_weight = (
        0.15 * torch.randn(experts, intermediate, hidden, device=device)
    ).to(torch.bfloat16)

    staged = rowwise_bf16_mega_moe_forward_debug(
        source_input,
        route_expert_indices,
        probs,
        up_gate_weight,
        down_weight,
    )
    full = rowwise_bf16_mega_moe_local_full_forward_megakernel_debug(
        source_input,
        route_expert_indices,
        probs,
        up_gate_weight,
        down_weight,
    )

    staged_gathered, staged_out = staged[-2:]
    (
        expert_counts,
        token_offsets,
        packed_route,
        route_to_slot,
        *_,
        gathered_out,
        out,
        barrier_state,
    ) = full
    assert expert_counts.cpu().tolist() == [4, 4]
    assert token_offsets.cpu().tolist() == [0, 4, 8]
    packed_route_cpu = packed_route.cpu()
    for expert_idx, expected_routes in {0: [0, 3, 4, 5], 1: [1, 2, 6, 7]}.items():
        start = token_offsets[expert_idx].item()
        end = token_offsets[expert_idx + 1].item()
        assert sorted(packed_route_cpu[start:end].tolist()) == expected_routes
    assert sorted(route_to_slot.cpu().tolist()) == list(range(tokens * top_k))
    torch.testing.assert_close(gathered_out, staged_gathered, rtol=0.03, atol=0.03)
    torch.testing.assert_close(out, staged_out, rtol=0.03, atol=0.03)
    assert int(barrier_state.cpu()[0].item()) == 0
    assert int(barrier_state.cpu()[1].item()) > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_v2_ep_wave_bf16_standard_scheduler_device_contract():
    from olmo_core.kernels.symm_mem_vdev2d import (
        rowwise_bf16_mega_moe_standard_scheduler_debug,
    )

    device = torch.device("cuda", torch.cuda.current_device())
    expert_counts = torch.full((8,), 8192, device=device, dtype=torch.long)
    _, debug = rowwise_bf16_mega_moe_standard_scheduler_debug(expert_counts)
    debug_cpu = debug.cpu()

    m_blocks = (8192 + 128 - 1) // 128
    expected_l1_tiles = m_blocks * (2 * 4096 // 128)
    expected_l2_tiles = m_blocks * (4096 // 128)
    assert debug_cpu[:-1, 0].tolist() == [expected_l1_tiles] * 8
    assert debug_cpu[:-1, 1].tolist() == [expected_l2_tiles] * 8
    assert debug_cpu[-1].tolist() == [8, 4, 64, 32]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_v2_ep_wave_bf16_standard_ep_dispatch_metadata_device_contract():
    from olmo_core.kernels.symm_mem_vdev2d import (
        rowwise_bf16_mega_moe_standard_ep_dispatch_metadata_debug,
    )

    device = torch.device("cuda", torch.cuda.current_device())
    route_expert_indices = torch.tensor(
        [
            [0, 8, 16, 24],
            [1, 9, 17, 25],
            [2, 10, 18, 26],
            [7, 15, 23, 31],
        ],
        device=device,
        dtype=torch.long,
    )
    _, recv_counts, recv_ready_counts, src_token_topk_indices, barrier_state = (
        rowwise_bf16_mega_moe_standard_ep_dispatch_metadata_debug(route_expert_indices)
    )

    expected_counts = torch.zeros((4, 8), dtype=torch.long)
    expected_counts[:, [0, 1, 2, 7]] = 1
    assert recv_counts.cpu().tolist() == expected_counts.tolist()
    assert recv_ready_counts.cpu().tolist() == [[8] * 8 for _ in range(4)]

    src_cpu = src_token_topk_indices.cpu()
    expected_sources = {
        (0, 0): [0],
        (1, 0): [1],
        (2, 0): [2],
        (3, 0): [3],
        (0, 1): [4],
        (1, 1): [5],
        (2, 1): [6],
        (3, 1): [7],
        (0, 2): [8],
        (1, 2): [9],
        (2, 2): [10],
        (3, 2): [11],
        (0, 7): [12],
        (1, 7): [13],
        (2, 7): [14],
        (3, 7): [15],
    }
    for rank_idx in range(4):
        for local_expert_idx in range(8):
            count = int(recv_counts.cpu()[rank_idx, local_expert_idx].item())
            seen = sorted(src_cpu[rank_idx, local_expert_idx, :count].tolist())
            assert seen == expected_sources.get((rank_idx, local_expert_idx), [])
            assert src_cpu[rank_idx, local_expert_idx, count:].tolist() == [-1] * (16 - count)
    assert int(barrier_state.cpu()[0].item()) == 0
    assert int(barrier_state.cpu()[1].item()) == 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_v2_ep_wave_bf16_standard_ep_dispatch_metadata_peer_map_device_contract():
    from olmo_core.kernels.symm_mem_vdev2d import (
        rowwise_bf16_mega_moe_standard_ep_dispatch_metadata_debug,
        rowwise_bf16_mega_moe_standard_ep_dispatch_metadata_peer_map_debug,
    )

    device = torch.device("cuda", torch.cuda.current_device())
    route_expert_indices = torch.tensor(
        [
            [0, 8, 16, 24],
            [1, 9, 17, 25],
            [2, 10, 18, 26],
            [7, 15, 23, 31],
        ],
        device=device,
        dtype=torch.long,
    )
    contiguous = rowwise_bf16_mega_moe_standard_ep_dispatch_metadata_debug(
        route_expert_indices
    )
    peer_map = rowwise_bf16_mega_moe_standard_ep_dispatch_metadata_peer_map_debug(
        route_expert_indices
    )

    _, contig_recv_counts, contig_ready_counts, contig_src, contig_barrier = contiguous
    (
        _,
        rank_workspace_bases,
        peer_recv_counts,
        peer_ready_counts,
        peer_src,
        peer_barrier,
    ) = peer_map
    torch.testing.assert_close(peer_recv_counts, contig_recv_counts, rtol=0, atol=0)
    torch.testing.assert_close(peer_ready_counts, contig_ready_counts, rtol=0, atol=0)
    torch.testing.assert_close(peer_src, contig_src, rtol=0, atol=0)
    torch.testing.assert_close(peer_barrier, contig_barrier, rtol=0, atol=0)

    bases = rank_workspace_bases.cpu().tolist()
    assert all(base > 0 for base in bases)
    strides = [bases[idx + 1] - bases[idx] for idx in range(3)]
    assert strides[0] > 0
    assert strides == [strides[0]] * 3


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_v2_ep_wave_bf16_standard_ep_dispatch_pack_inputs_device_contract():
    from olmo_core.kernels.symm_mem_vdev2d import (
        rowwise_bf16_mega_moe_standard_ep_dispatch_pack_inputs_debug,
    )

    device = torch.device("cuda", torch.cuda.current_device())
    source_input = torch.arange(4 * 16, device=device, dtype=torch.float32).view(4, 16).to(torch.bfloat16)
    route_expert_indices = torch.tensor(
        [
            [0, 8, 16, 24],
            [1, 9, 17, 25],
            [2, 10, 18, 26],
            [7, 15, 23, 31],
        ],
        device=device,
        dtype=torch.long,
    )
    (
        _,
        global_counts,
        global_offsets,
        packed_route,
        packed_input,
        *_,
    ) = rowwise_bf16_mega_moe_standard_ep_dispatch_pack_inputs_debug(
        source_input,
        route_expert_indices,
    )

    expected_counts = [0] * 32
    for expert_idx in [0, 1, 2, 7, 8, 9, 10, 15, 16, 17, 18, 23, 24, 25, 26, 31]:
        expected_counts[expert_idx] = 1
    expected_offsets = [0]
    for count in expected_counts:
        expected_offsets.append(expected_offsets[-1] + count)
    expected_routes = [0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15]

    assert global_counts.cpu().tolist() == expected_counts
    assert global_offsets.cpu().tolist() == expected_offsets
    assert packed_route.cpu().tolist() == expected_routes
    for packed_idx, route_idx in enumerate(expected_routes):
        token_idx = route_idx // 4
        assert packed_input[packed_idx].cpu().tolist() == source_input[token_idx].cpu().tolist()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_v2_ep_wave_bf16_standard_ep_forward_from_dispatch_device_contract():
    from olmo_core.kernels.symm_mem_vdev2d import (
        rowwise_bf16_mega_moe_standard_ep_forward_from_dispatch_debug,
    )

    device = torch.device("cuda", torch.cuda.current_device())
    torch.manual_seed(7211)
    tokens = 4
    top_k = 4
    hidden = 16
    intermediate = 16
    experts = 32
    source_input = (0.2 * torch.randn(tokens, hidden, device=device)).to(torch.bfloat16)
    route_expert_indices = torch.tensor(
        [
            [0, 8, 16, 24],
            [1, 9, 17, 25],
            [2, 10, 18, 26],
            [7, 15, 23, 31],
        ],
        device=device,
        dtype=torch.long,
    )
    probs = torch.tensor(
        [
            [0.1, 0.2, 0.3, 0.4],
            [0.4, 0.3, 0.2, 0.1],
            [0.25, 0.25, 0.25, 0.25],
            [0.7, 0.1, 0.1, 0.1],
        ],
        device=device,
        dtype=torch.float32,
    )
    up_gate_weight = (
        0.15 * torch.randn(experts, 2 * intermediate, hidden, device=device)
    ).to(torch.bfloat16)
    down_weight = (
        0.15 * torch.randn(experts, intermediate, hidden, device=device)
    ).to(torch.bfloat16)

    (
        _,
        global_counts,
        global_offsets,
        packed_route,
        packed_input,
        _,
        packed_expert_out,
        gathered_out,
        out,
        compute_barrier_state,
        recv_counts,
        recv_ready_counts,
        src_token_topk_indices,
        metadata_barrier_state,
    ) = rowwise_bf16_mega_moe_standard_ep_forward_from_dispatch_debug(
        source_input,
        route_expert_indices,
        probs,
        up_gate_weight,
        down_weight,
    )

    expected_counts = [0] * experts
    for expert_idx in [0, 1, 2, 7, 8, 9, 10, 15, 16, 17, 18, 23, 24, 25, 26, 31]:
        expected_counts[expert_idx] = 1
    expected_offsets = [0]
    for count in expected_counts:
        expected_offsets.append(expected_offsets[-1] + count)
    expected_routes = [0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15]

    assert global_counts.cpu().tolist() == expected_counts
    assert global_offsets.cpu().tolist() == expected_offsets
    assert packed_route.cpu().tolist() == expected_routes
    for packed_idx, route_idx in enumerate(expected_routes):
        token_idx = route_idx // top_k
        assert packed_input[packed_idx].cpu().tolist() == source_input[token_idx].cpu().tolist()

    expected_gathered = torch.empty(tokens, top_k, hidden, device=device, dtype=torch.bfloat16)
    expected_out = torch.zeros(tokens, hidden, device=device, dtype=torch.float32)
    expected_packed_out = torch.empty(tokens * top_k, hidden, device=device, dtype=torch.bfloat16)
    for token_idx in range(tokens):
        for topk_idx in range(top_k):
            expert_idx = int(route_expert_indices[token_idx, topk_idx].item())
            up_gate = source_input[token_idx].float() @ up_gate_weight[expert_idx].float().T
            up = up_gate[:intermediate].to(torch.bfloat16).float()
            gate = up_gate[intermediate:].to(torch.bfloat16).float()
            hidden_state = (up * torch.nn.functional.silu(gate)).to(torch.bfloat16)
            expert_out = (hidden_state.float() @ down_weight[expert_idx].float()).to(
                torch.bfloat16
            )
            route_idx = token_idx * top_k + topk_idx
            packed_idx = expected_routes.index(route_idx)
            expected_packed_out[packed_idx] = expert_out
            expected_gathered[token_idx, topk_idx] = expert_out
            expected_out[token_idx] += probs[token_idx, topk_idx] * expert_out.float()

    torch.testing.assert_close(
        packed_expert_out,
        expected_packed_out,
        rtol=0.03,
        atol=0.03,
    )
    torch.testing.assert_close(gathered_out, expected_gathered, rtol=0.03, atol=0.03)
    torch.testing.assert_close(out, expected_out.to(torch.bfloat16), rtol=0.03, atol=0.03)

    expected_recv_counts = torch.zeros((4, 8), dtype=torch.long)
    expected_recv_counts[:, [0, 1, 2, 7]] = 1
    assert recv_counts.cpu().tolist() == expected_recv_counts.tolist()
    assert recv_ready_counts.cpu().tolist() == [[8] * 8 for _ in range(4)]
    assert int(src_token_topk_indices.cpu()[0, 0, 0].item()) == 0
    assert int(src_token_topk_indices.cpu()[3, 7, 0].item()) == 15
    assert int(compute_barrier_state.cpu()[0].item()) == 0
    assert int(compute_barrier_state.cpu()[1].item()) == 16
    assert int(metadata_barrier_state.cpu()[0].item()) == 0
    assert int(metadata_barrier_state.cpu()[1].item()) == 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_v2_ep_wave_bf16_standard_ep_forward_from_dispatch_umma_matches_wmma():
    if torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("requires Blackwell-class CUDA device")

    from olmo_core.kernels.symm_mem_vdev2d import (
        rowwise_bf16_mega_moe_standard_ep_forward_from_dispatch_debug,
        rowwise_bf16_mega_moe_standard_ep_forward_from_dispatch_umma_debug,
    )

    device = torch.device("cuda", torch.cuda.current_device())
    torch.manual_seed(9823)
    tokens = 32
    top_k = 4
    hidden = 128
    intermediate = 128
    experts = 32
    source_input = (0.2 * torch.randn(tokens, hidden, device=device)).to(torch.bfloat16)
    route_expert_indices = (
        torch.arange(tokens * top_k, device=device, dtype=torch.long).view(tokens, top_k)
        % experts
    ).contiguous()
    assert route_expert_indices.numel() == 128
    probs = (
        torch.tensor([0.4, 0.3, 0.2, 0.1], device=device, dtype=torch.float32)
        .view(1, top_k)
        .expand(tokens, top_k)
        .contiguous()
    )
    up_gate_weight = (
        0.15 * torch.randn(experts, 2 * intermediate, hidden, device=device)
    ).to(torch.bfloat16)
    down_weight = (
        0.15 * torch.randn(experts, intermediate, hidden, device=device)
    ).to(torch.bfloat16)

    try:
        wmma = rowwise_bf16_mega_moe_standard_ep_forward_from_dispatch_debug(
            source_input,
            route_expert_indices,
            probs,
            up_gate_weight,
            down_weight,
        )
        umma = rowwise_bf16_mega_moe_standard_ep_forward_from_dispatch_umma_debug(
            source_input,
            route_expert_indices,
            probs,
            up_gate_weight,
            down_weight,
        )
    except RuntimeError as e:
        if "extension is unavailable" in str(e):
            pytest.skip("symm_mem_vdev2d CUDA extension is not built")
        raise

    assert len(umma) > len(wmma)
    (
        _,
        wmma_counts,
        wmma_offsets,
        wmma_routes,
        wmma_packed_input,
        wmma_h,
        wmma_packed_expert_out,
        wmma_gathered_out,
        wmma_out,
        *_,
    ) = wmma
    (
        _,
        umma_counts,
        umma_offsets,
        umma_routes,
        umma_packed_input,
        umma_h,
        umma_packed_expert_out,
        umma_gathered_out,
        umma_out,
        *_,
    ) = umma

    torch.testing.assert_close(umma_counts, wmma_counts, rtol=0, atol=0)
    torch.testing.assert_close(umma_offsets, wmma_offsets, rtol=0, atol=0)
    assert sorted(umma_routes.cpu().tolist()) == sorted(wmma_routes.cpu().tolist())

    def by_route(packed_routes: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        normalized = torch.empty_like(values)
        normalized[packed_routes.long()] = values
        return normalized

    torch.testing.assert_close(
        by_route(umma_routes, umma_packed_input),
        by_route(wmma_routes, wmma_packed_input),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        by_route(umma_routes, umma_h),
        by_route(wmma_routes, wmma_h),
        rtol=0.05,
        atol=0.05,
    )
    torch.testing.assert_close(
        by_route(umma_routes, umma_packed_expert_out),
        by_route(wmma_routes, wmma_packed_expert_out),
        rtol=0.05,
        atol=0.05,
    )
    torch.testing.assert_close(
        umma_gathered_out,
        wmma_gathered_out,
        rtol=0.05,
        atol=0.05,
    )
    torch.testing.assert_close(umma_out, wmma_out, rtol=0.05, atol=0.05)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_v2_ep_wave_bf16_local_umma_compute_matches_torch_reference():
    if torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("requires Blackwell-class CUDA device")

    from olmo_core.kernels.symm_mem_vdev2d import (
        rowwise_bf16_mega_moe_local_umma_compute,
        rowwise_bf16_mega_moe_local_umma_compute_debug,
    )

    device = torch.device("cuda", torch.cuda.current_device())
    torch.manual_seed(9137)
    local_experts = 8
    hidden = 128
    intermediate = 128
    expert_counts = torch.tensor(
        [17, 0, 33, 1, 24, 8, 0, 45],
        device=device,
        dtype=torch.long,
    )
    rows = int(expert_counts.sum().item())
    packed_input = (0.2 * torch.randn(rows, hidden, device=device)).to(torch.bfloat16)
    up_gate_weight = (
        0.15 * torch.randn(local_experts, 2 * intermediate, hidden, device=device)
    ).to(torch.bfloat16)
    down_weight = (
        0.15 * torch.randn(local_experts, intermediate, hidden, device=device)
    ).to(torch.bfloat16)

    try:
        (
            counts,
            token_offsets,
            tile_counts,
            tile_offsets,
            num_total_m_tiles,
            h,
            packed_expert_out,
            *_,
        ) = rowwise_bf16_mega_moe_local_umma_compute_debug(
            packed_input,
            expert_counts,
            up_gate_weight,
            down_weight,
        )
    except RuntimeError as e:
        if "extension is unavailable" in str(e):
            pytest.skip("symm_mem_vdev2d CUDA extension is not built")
        raise

    packed_expert_out_prod = rowwise_bf16_mega_moe_local_umma_compute(
        packed_input,
        expert_counts,
        up_gate_weight,
        down_weight,
    )

    expected_offsets = torch.zeros(local_experts + 1, dtype=torch.long)
    expected_offsets[1:] = torch.cumsum(expert_counts.cpu(), dim=0)
    expected_tile_counts = torch.div(
        expert_counts.cpu() + 127,
        128,
        rounding_mode="floor",
    )
    expected_tile_counts[expert_counts.cpu() == 0] = 0
    expected_tile_offsets = torch.zeros(local_experts + 1, dtype=torch.long)
    expected_tile_offsets[1:] = torch.cumsum(expected_tile_counts, dim=0)

    torch.testing.assert_close(counts.cpu(), expert_counts.cpu(), rtol=0, atol=0)
    torch.testing.assert_close(token_offsets.cpu(), expected_offsets, rtol=0, atol=0)
    torch.testing.assert_close(tile_counts.cpu(), expected_tile_counts, rtol=0, atol=0)
    torch.testing.assert_close(tile_offsets.cpu(), expected_tile_offsets, rtol=0, atol=0)
    assert int(num_total_m_tiles.cpu()[0].item()) == int(expected_tile_counts.sum().item())

    expected_h = torch.zeros_like(h)
    expected_out = torch.zeros_like(packed_expert_out)
    start = 0
    for expert_idx, count in enumerate(expert_counts.cpu().tolist()):
        end = start + int(count)
        if end > start:
            up_gate = (
                packed_input[start:end].float()
                @ up_gate_weight[expert_idx].float().T
            ).to(torch.bfloat16)
            up = up_gate[:, :intermediate]
            gate = up_gate[:, intermediate:]
            hidden_state = (up.float() * torch.nn.functional.silu(gate.float())).to(
                torch.bfloat16
            )
            expert_out = (
                hidden_state.float() @ down_weight[expert_idx].float()
            ).to(torch.bfloat16)
            expected_h[start:end] = hidden_state
            expected_out[start:end] = expert_out
        start = end

    torch.testing.assert_close(h, expected_h, rtol=0.05, atol=0.05)
    torch.testing.assert_close(packed_expert_out, expected_out, rtol=0.05, atol=0.05)
    torch.testing.assert_close(packed_expert_out_prod, expected_out, rtol=0.05, atol=0.05)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_v2_ep_wave_bf16_standard_ep_full_forward_megakernel_device_contract():
    from olmo_core.kernels.symm_mem_vdev2d import (
        rowwise_bf16_mega_moe_standard_ep_forward_from_dispatch_debug,
        rowwise_bf16_mega_moe_standard_ep_forward_persistent,
        rowwise_bf16_mega_moe_standard_ep_forward_persistent_workspace,
        rowwise_bf16_mega_moe_standard_ep_full_forward_megakernel,
        rowwise_bf16_mega_moe_standard_ep_full_forward_megakernel_debug,
        rowwise_bf16_mega_moe_standard_ep_workspace_config,
    )

    device = torch.device("cuda", torch.cuda.current_device())
    torch.manual_seed(1840)
    tokens = 4
    top_k = 4
    hidden = 16
    intermediate = 16
    experts = 32
    source_input = (0.2 * torch.randn(tokens, hidden, device=device)).to(torch.bfloat16)
    route_expert_indices = torch.tensor(
        [
            [0, 8, 16, 24],
            [1, 9, 17, 25],
            [2, 10, 18, 26],
            [7, 15, 23, 31],
        ],
        device=device,
        dtype=torch.long,
    )
    probs = torch.tensor(
        [
            [0.1, 0.2, 0.3, 0.4],
            [0.4, 0.3, 0.2, 0.1],
            [0.25, 0.25, 0.25, 0.25],
            [0.7, 0.1, 0.1, 0.1],
        ],
        device=device,
        dtype=torch.float32,
    )
    up_gate_weight = (
        0.15 * torch.randn(experts, 2 * intermediate, hidden, device=device)
    ).to(torch.bfloat16)
    down_weight = (
        0.15 * torch.randn(experts, intermediate, hidden, device=device)
    ).to(torch.bfloat16)

    bridge = rowwise_bf16_mega_moe_standard_ep_forward_from_dispatch_debug(
        source_input,
        route_expert_indices,
        probs,
        up_gate_weight,
        down_weight,
    )
    full = rowwise_bf16_mega_moe_standard_ep_full_forward_megakernel_debug(
        source_input,
        route_expert_indices,
        probs,
        up_gate_weight,
        down_weight,
    )
    prod_gathered_out, prod_out = rowwise_bf16_mega_moe_standard_ep_full_forward_megakernel(
        source_input,
        route_expert_indices,
        probs,
        up_gate_weight,
        down_weight,
    )
    persistent_gathered_out = torch.empty_like(prod_gathered_out)
    persistent_out = torch.empty_like(prod_out)
    rowwise_bf16_mega_moe_standard_ep_forward_persistent(
        source_input,
        persistent_gathered_out,
        persistent_out,
        route_expert_indices,
        probs,
        up_gate_weight,
        down_weight,
    )

    workspace_config = rowwise_bf16_mega_moe_standard_ep_workspace_config(
        num_tokens=tokens,
        hidden=hidden,
        intermediate=intermediate,
    )
    assert workspace_config["num_route_slots"] == tokens * top_k
    assert workspace_config["num_ranks"] == 4
    assert workspace_config["num_total_experts"] == experts
    assert workspace_config["num_local_experts"] == experts // workspace_config["num_ranks"]
    assert workspace_config["top_k"] == top_k
    assert workspace_config["barrier_state_len"] >= 2
    assert workspace_config["packed_values"] == tokens * top_k * hidden
    assert workspace_config["h_values"] == tokens * top_k * intermediate

    workspace = torch.empty(
        (workspace_config["workspace_bytes"],),
        device=device,
        dtype=torch.uint8,
    )
    workspace_rank_workspace_bases = torch.empty(
        (workspace_config["num_ranks"],),
        device=device,
        dtype=torch.long,
    )
    workspace_counts = torch.empty(
        (workspace_config["num_total_experts"],),
        device=device,
        dtype=torch.long,
    )
    workspace_offsets = torch.empty(
        (workspace_config["num_total_experts"] + 1,),
        device=device,
        dtype=torch.long,
    )
    workspace_expert_cursors = torch.empty_like(workspace_counts)
    workspace_packed_route = torch.empty(
        (workspace_config["num_route_slots"],),
        device=device,
        dtype=torch.long,
    )
    workspace_route_to_slot = torch.empty_like(workspace_packed_route)
    workspace_packed_input = torch.empty(
        (
            workspace_config["num_route_slots"],
            workspace_config["packed_values"] // workspace_config["num_route_slots"],
        ),
        device=device,
        dtype=torch.bfloat16,
    )
    workspace_h = torch.empty(
        (
            workspace_config["num_route_slots"],
            workspace_config["h_values"] // workspace_config["num_route_slots"],
        ),
        device=device,
        dtype=torch.bfloat16,
    )
    workspace_packed_expert_out = torch.empty_like(workspace_packed_input)
    workspace_barrier_state = torch.empty(
        (workspace_config["barrier_state_len"],),
        device=device,
        dtype=torch.int32,
    )
    workspace_gathered_out = torch.empty_like(prod_gathered_out)
    workspace_out = torch.empty_like(prod_out)

    def run_workspace_forward(
        *,
        caller_rank_idx: int = 0,
        use_peer_workspace_bases: bool = False,
        enable_cross_rank_barriers: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        workspace_gathered_out.fill_(float("nan"))
        workspace_out.fill_(float("nan"))
        rowwise_bf16_mega_moe_standard_ep_forward_persistent_workspace(
            source_input,
            workspace_gathered_out,
            workspace_out,
            route_expert_indices,
            probs,
            up_gate_weight,
            down_weight,
            workspace,
            workspace_rank_workspace_bases,
            workspace_counts,
            workspace_offsets,
            workspace_expert_cursors,
            workspace_packed_route,
            workspace_route_to_slot,
            workspace_packed_input,
            workspace_h,
            workspace_packed_expert_out,
            workspace_barrier_state,
            caller_rank_idx=caller_rank_idx,
            use_peer_workspace_bases=use_peer_workspace_bases,
            enable_cross_rank_barriers=enable_cross_rank_barriers,
        )
        return workspace_gathered_out.clone(), workspace_out.clone()

    workspace_gathered_out_first, workspace_out_first = run_workspace_forward()
    workspace.fill_(123)
    workspace_rank_workspace_bases.fill_(-1)
    workspace_counts.fill_(-1)
    workspace_offsets.fill_(-1)
    workspace_expert_cursors.fill_(-1)
    workspace_packed_route.fill_(-1)
    workspace_route_to_slot.fill_(-1)
    workspace_packed_input.fill_(float("nan"))
    workspace_h.fill_(float("nan"))
    workspace_packed_expert_out.fill_(float("nan"))
    workspace_barrier_state.fill_(-1)
    workspace_gathered_out_second, workspace_out_second = run_workspace_forward()
    workspace.fill_(123)
    workspace_barrier_state.fill_(-1)
    workspace_gathered_out_rank1, workspace_out_rank1 = run_workspace_forward(caller_rank_idx=1)
    workspace_packed_route_rank1 = workspace_packed_route.clone()
    workspace.zero_()
    workspace_barrier_state.fill_(-1)
    workspace_gathered_out_peer, workspace_out_peer = run_workspace_forward(
        use_peer_workspace_bases=True,
    )
    with pytest.raises(RuntimeError, match="collective-launch"):
        run_workspace_forward(
            use_peer_workspace_bases=True,
            enable_cross_rank_barriers=True,
        )

    (
        _,
        bridge_counts,
        bridge_offsets,
        bridge_packed_route,
        bridge_packed_input,
        bridge_h,
        bridge_packed_expert_out,
        bridge_gathered_out,
        bridge_out,
        _,
        bridge_recv_counts,
        bridge_recv_ready_counts,
        bridge_src_token_topk_indices,
        _,
    ) = bridge
    (
        _,
        full_rank_workspace_bases,
        full_counts,
        full_offsets,
        full_packed_route,
        full_route_to_slot,
        full_packed_input,
        full_h,
        full_packed_expert_out,
        full_gathered_out,
        full_out,
        full_recv_counts,
        full_recv_ready_counts,
        full_src_token_topk_indices,
        full_barrier_state,
    ) = full

    full_bases = full_rank_workspace_bases.cpu().tolist()
    assert all(base > 0 for base in full_bases)
    full_strides = [full_bases[idx + 1] - full_bases[idx] for idx in range(3)]
    assert full_strides[0] > 0
    assert full_strides == [full_strides[0]] * 3
    torch.testing.assert_close(full_counts, bridge_counts, rtol=0, atol=0)
    torch.testing.assert_close(full_offsets, bridge_offsets, rtol=0, atol=0)
    torch.testing.assert_close(full_packed_route, bridge_packed_route, rtol=0, atol=0)
    torch.testing.assert_close(full_packed_input, bridge_packed_input, rtol=0, atol=0)
    torch.testing.assert_close(full_h, bridge_h, rtol=0.03, atol=0.03)
    torch.testing.assert_close(
        full_packed_expert_out,
        bridge_packed_expert_out,
        rtol=0.03,
        atol=0.03,
    )
    torch.testing.assert_close(full_gathered_out, bridge_gathered_out, rtol=0.03, atol=0.03)
    torch.testing.assert_close(full_out, bridge_out, rtol=0.03, atol=0.03)
    torch.testing.assert_close(prod_gathered_out, full_gathered_out, rtol=0.03, atol=0.03)
    torch.testing.assert_close(prod_out, full_out, rtol=0.03, atol=0.03)
    torch.testing.assert_close(persistent_gathered_out, prod_gathered_out, rtol=0.03, atol=0.03)
    torch.testing.assert_close(persistent_out, prod_out, rtol=0.03, atol=0.03)
    torch.testing.assert_close(workspace_gathered_out_first, prod_gathered_out, rtol=0.03, atol=0.03)
    torch.testing.assert_close(workspace_out_first, prod_out, rtol=0.03, atol=0.03)
    torch.testing.assert_close(workspace_gathered_out_second, full_gathered_out, rtol=0.03, atol=0.03)
    torch.testing.assert_close(workspace_out_second, full_out, rtol=0.03, atol=0.03)
    torch.testing.assert_close(workspace_gathered_out_rank1, full_gathered_out, rtol=0.03, atol=0.03)
    torch.testing.assert_close(workspace_out_rank1, full_out, rtol=0.03, atol=0.03)
    torch.testing.assert_close(workspace_gathered_out_peer, full_gathered_out, rtol=0.03, atol=0.03)
    torch.testing.assert_close(workspace_out_peer, full_out, rtol=0.03, atol=0.03)
    valid_rank1_routes = workspace_packed_route_rank1[workspace_packed_route_rank1 >= 0]
    assert valid_rank1_routes.numel() == tokens * top_k
    assert sorted((valid_rank1_routes % (tokens * top_k)).cpu().tolist()) == list(
        range(tokens * top_k)
    )
    assert sorted((valid_rank1_routes // (tokens * top_k)).cpu().tolist()) == [1] * (
        tokens * top_k
    )
    torch.testing.assert_close(
        workspace_gathered_out_second,
        workspace_gathered_out_first,
        rtol=0.03,
        atol=0.03,
    )
    torch.testing.assert_close(workspace_out_second, workspace_out_first, rtol=0.03, atol=0.03)
    torch.testing.assert_close(full_recv_counts, bridge_recv_counts, rtol=0, atol=0)
    torch.testing.assert_close(full_recv_ready_counts, bridge_recv_ready_counts, rtol=0, atol=0)
    torch.testing.assert_close(
        full_src_token_topk_indices,
        bridge_src_token_topk_indices,
        rtol=0,
        atol=0,
    )

    assert sorted(full_route_to_slot.cpu().tolist()) == list(range(tokens * top_k))
    for packed_idx, route_idx in enumerate(full_packed_route.cpu().tolist()):
        assert int(full_route_to_slot.cpu()[route_idx].item()) == packed_idx
    assert int(full_barrier_state.cpu()[0].item()) == 0
    assert int(full_barrier_state.cpu()[1].item()) == 26


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_v2_ep_wave_bf16_standard_ep_local_owner_matches_full_for_one_rank():
    from olmo_core.kernels.symm_mem_vdev2d import (
        rowwise_bf16_mega_moe_standard_ep_forward_persistent_workspace,
        rowwise_bf16_mega_moe_standard_ep_full_forward_megakernel,
        rowwise_bf16_mega_moe_standard_ep_workspace_config,
    )

    device = torch.device("cuda", torch.cuda.current_device())
    torch.manual_seed(1841)
    tokens = 4
    top_k = 4
    hidden = 16
    intermediate = 16
    experts = 32
    caller_rank_idx = 1
    source_input = (0.2 * torch.randn(tokens, hidden, device=device)).to(torch.bfloat16)
    route_expert_indices = torch.tensor(
        [
            [8, 9, 10, 15],
            [9, 10, 11, 14],
            [10, 11, 12, 13],
            [8, 12, 13, 15],
        ],
        device=device,
        dtype=torch.long,
    )
    probs = torch.softmax(
        torch.randn(tokens, top_k, device=device, dtype=torch.float32),
        dim=-1,
    ).contiguous()
    up_gate_weight = (
        0.15 * torch.randn(experts, 2 * intermediate, hidden, device=device)
    ).to(torch.bfloat16)
    down_weight = (
        0.15 * torch.randn(experts, intermediate, hidden, device=device)
    ).to(torch.bfloat16)

    expected_gathered, expected_out = rowwise_bf16_mega_moe_standard_ep_full_forward_megakernel(
        source_input,
        route_expert_indices,
        probs,
        up_gate_weight,
        down_weight,
    )
    workspace_config = rowwise_bf16_mega_moe_standard_ep_workspace_config(
        num_tokens=tokens,
        hidden=hidden,
        intermediate=intermediate,
    )
    assert workspace_config["local_packed_capacity"] >= workspace_config["num_route_slots"]

    workspace = torch.empty((workspace_config["workspace_bytes"],), device=device, dtype=torch.uint8)
    rank_workspace_bases = torch.empty(
        (workspace_config["num_ranks"],),
        device=device,
        dtype=torch.long,
    )
    global_counts = torch.empty(
        (workspace_config["num_total_experts"],),
        device=device,
        dtype=torch.long,
    )
    global_offsets = torch.empty(
        (workspace_config["num_total_experts"] + 1,),
        device=device,
        dtype=torch.long,
    )
    expert_cursors = torch.empty_like(global_counts)
    packed_route = torch.empty(
        (workspace_config["local_packed_capacity"],),
        device=device,
        dtype=torch.long,
    )
    route_to_slot = torch.empty(
        (workspace_config["num_route_slots"],),
        device=device,
        dtype=torch.long,
    )
    packed_input = torch.empty(
        (workspace_config["local_packed_capacity"], hidden),
        device=device,
        dtype=torch.bfloat16,
    )
    h = torch.empty(
        (workspace_config["local_packed_capacity"], intermediate),
        device=device,
        dtype=torch.bfloat16,
    )
    packed_expert_out = torch.empty_like(packed_input)
    barrier_state = torch.empty(
        (workspace_config["barrier_state_len"],),
        device=device,
        dtype=torch.int32,
    )
    gathered_out = torch.empty_like(expected_gathered)
    out = torch.empty_like(expected_out)

    rowwise_bf16_mega_moe_standard_ep_forward_persistent_workspace(
        source_input,
        gathered_out,
        out,
        route_expert_indices,
        probs,
        up_gate_weight,
        down_weight,
        workspace,
        rank_workspace_bases,
        global_counts,
        global_offsets,
        expert_cursors,
        packed_route,
        route_to_slot,
        packed_input,
        h,
        packed_expert_out,
        barrier_state,
        caller_rank_idx=caller_rank_idx,
        rank_local_expert_owner=True,
    )

    torch.testing.assert_close(gathered_out, expected_gathered, rtol=0.03, atol=0.03)
    torch.testing.assert_close(out, expected_out, rtol=0.03, atol=0.03)
    assert int(global_offsets.cpu()[workspace_config["num_local_experts"]].item()) == tokens * top_k
    assert int(global_offsets.cpu()[workspace_config["num_total_experts"]].item()) == tokens * top_k
    valid = packed_route[: tokens * top_k]
    assert sorted((valid % (tokens * top_k)).cpu().tolist()) == list(range(tokens * top_k))
    assert sorted((valid // (tokens * top_k)).cpu().tolist()) == [caller_rank_idx] * (
        tokens * top_k
    )


def _run_standard_ep_peer_group_forward_matches_torch():
    from olmo_core.kernels import olmo_symm_mem
    from olmo_core.kernels.symm_mem_vdev2d import (
        rowwise_bf16_mega_moe_standard_ep_forward_persistent_workspace_peer_group,
        rowwise_bf16_mega_moe_standard_ep_workspace_config,
    )

    os.environ["OLMO_USE_OWN_SYMM_MEM"] = "1"
    group = dist.group.WORLD
    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)
    if world_size != 4:
        raise RuntimeError(f"standard EP peer-group test requires world_size=4, got {world_size}")

    device = torch.device("cuda", torch.cuda.current_device())
    torch.manual_seed(1842)
    tokens = 3
    top_k = 4
    hidden = 16
    intermediate = 16
    experts = 32
    up_gate_weight = (
        0.12 * torch.randn(experts, 2 * intermediate, hidden, device=device)
    ).to(torch.bfloat16)
    down_weight = (
        0.12 * torch.randn(experts, intermediate, hidden, device=device)
    ).to(torch.bfloat16)

    torch.manual_seed(4000 + rank)
    source_input = (0.2 * torch.randn(tokens, hidden, device=device)).to(torch.bfloat16)
    route_expert_indices = torch.empty((tokens, top_k), device=device, dtype=torch.long)
    for dst_rank in range(top_k):
        route_expert_indices[:, dst_rank] = (
            dst_rank * 8 + (torch.arange(tokens, device=device) + rank + dst_rank) % 8
        )
    probs = torch.softmax(
        torch.randn(tokens, top_k, device=device, dtype=torch.float32),
        dim=-1,
    ).contiguous()

    workspace_config = rowwise_bf16_mega_moe_standard_ep_workspace_config(
        num_tokens=tokens,
        hidden=hidden,
        intermediate=intermediate,
    )
    workspace = olmo_symm_mem.empty(
        (workspace_config["workspace_stride_bytes"],),
        dtype=torch.uint8,
        device=device,
        group=group,
    )
    rank_workspace_bases = olmo_symm_mem.peer_base_ptrs(workspace, group=group)
    gathered_out = torch.empty((tokens, top_k, hidden), device=device, dtype=torch.bfloat16)
    out = torch.empty((tokens, hidden), device=device, dtype=torch.bfloat16)
    global_counts = torch.empty(
        (workspace_config["num_total_experts"],),
        device=device,
        dtype=torch.long,
    )
    global_offsets = torch.empty(
        (workspace_config["num_total_experts"] + 1,),
        device=device,
        dtype=torch.long,
    )
    expert_cursors = torch.empty_like(global_counts)
    packed_route = torch.empty(
        (workspace_config["local_packed_capacity"],),
        device=device,
        dtype=torch.long,
    )
    route_to_slot = torch.empty(
        (workspace_config["num_route_slots"],),
        device=device,
        dtype=torch.long,
    )
    packed_input = torch.empty(
        (workspace_config["local_packed_capacity"], hidden),
        device=device,
        dtype=torch.bfloat16,
    )
    h = torch.empty(
        (workspace_config["local_packed_capacity"], intermediate),
        device=device,
        dtype=torch.bfloat16,
    )
    packed_expert_out = torch.empty_like(packed_input)
    barrier_state = torch.empty(
        (workspace_config["barrier_state_len"],),
        device=device,
        dtype=torch.int32,
    )

    rowwise_bf16_mega_moe_standard_ep_forward_persistent_workspace_peer_group(
        source_input,
        gathered_out,
        out,
        route_expert_indices,
        probs,
        up_gate_weight,
        down_weight,
        workspace,
        rank_workspace_bases,
        global_counts,
        global_offsets,
        expert_cursors,
        packed_route,
        route_to_slot,
        packed_input,
        h,
        packed_expert_out,
        barrier_state,
        caller_rank_idx=rank,
    )

    expected_gathered = torch.empty_like(gathered_out)
    expected_out = torch.zeros(tokens, hidden, device=device, dtype=torch.float32)
    for token_idx in range(tokens):
        for topk_idx in range(top_k):
            expert_idx = int(route_expert_indices[token_idx, topk_idx].item())
            up_gate = (
                source_input[token_idx].float() @ up_gate_weight[expert_idx].float().T
            ).to(torch.bfloat16)
            up = up_gate[:intermediate]
            gate = up_gate[intermediate:]
            hidden_state = (up.float() * torch.nn.functional.silu(gate.float())).to(
                torch.bfloat16
            )
            expert_out = (
                hidden_state.float() @ down_weight[expert_idx].float()
            ).to(torch.bfloat16)
            expected_gathered[token_idx, topk_idx] = expert_out
            expected_out[token_idx] += probs[token_idx, topk_idx] * expert_out.float()

    torch.testing.assert_close(gathered_out, expected_gathered, rtol=0.05, atol=0.05)
    torch.testing.assert_close(out, expected_out.to(torch.bfloat16), rtol=0.05, atol=0.05)
    assert int(global_offsets.cpu()[workspace_config["num_local_experts"]].item()) == (
        tokens * world_size
    )
    dist.barrier(group=group)


@requires_multi_gpu
def test_v2_ep_wave_bf16_standard_ep_peer_group_forward_matches_torch():
    if torch.cuda.device_count() < 4:
        pytest.skip("requires at least 4 GPUs")
    run_distributed_test(
        _run_standard_ep_peer_group_forward_matches_torch,
        world_size=4,
        backend="nccl",
        start_method="spawn",
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_v2_ep_wave_bf16_persistent_mega_cuda_local_forward_matches_torch():
    if torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("requires Blackwell-class CUDA device")

    from olmo_core.kernels.symm_mem_vdev2d import (
        rowwise_bf16_mega_moe_forward_persistent,
    )

    device = torch.device("cuda", torch.cuda.current_device())
    torch.manual_seed(260623)
    tokens = 16
    hidden = 128
    intermediate = 128
    top_k = 2
    experts = 2

    source_input = (0.2 * torch.randn(tokens, hidden, device=device)).to(torch.bfloat16)
    gathered_out = torch.empty((tokens, top_k, hidden), device=device, dtype=torch.bfloat16)
    out = torch.empty((tokens, hidden), device=device, dtype=torch.bfloat16)
    route_dst_ranks = torch.zeros((tokens, top_k), device=device, dtype=torch.long)
    route_dst_rows = torch.arange(tokens * top_k, device=device, dtype=torch.long).view(tokens, top_k)
    route_expert_indices = (
        torch.arange(tokens * top_k, device=device, dtype=torch.long).view(tokens, top_k) % experts
    )
    probs = torch.tensor([0.625, 0.375], device=device, dtype=torch.float32).view(1, top_k)
    probs = probs.expand(tokens, top_k).contiguous()
    up_gate_weight = (
        0.15 * torch.randn(experts, 2 * intermediate, hidden, device=device)
    ).to(torch.bfloat16)
    down_weight = (
        0.15 * torch.randn(experts, intermediate, hidden, device=device)
    ).to(torch.bfloat16)
    expert_offsets = torch.tensor([0, tokens, tokens * top_k], device=device, dtype=torch.int32)

    try:
        rowwise_bf16_mega_moe_forward_persistent(
            source_input,
            gathered_out,
            out,
            route_dst_ranks,
            route_dst_rows,
            route_expert_indices,
            probs,
            up_gate_weight,
            down_weight,
            expert_offsets,
            "unused",
        )
    except RuntimeError as e:
        if "extension is unavailable" in str(e):
            pytest.skip("symm_mem_vdev2d CUDA extension is not built")
        raise

    expected_gathered = torch.zeros_like(gathered_out)
    expected_out = torch.zeros(tokens, hidden, device=device, dtype=torch.float32)
    for token_idx in range(tokens):
        for topk_idx in range(top_k):
            expert_idx = int(route_expert_indices[token_idx, topk_idx].item())
            up_gate = (
                source_input[token_idx].float() @ up_gate_weight[expert_idx].float().T
            ).to(torch.bfloat16)
            up = up_gate[:intermediate]
            gate = up_gate[intermediate:]
            hidden_state = (up.float() * torch.nn.functional.silu(gate.float())).to(
                torch.bfloat16
            )
            expert_out = (
                hidden_state.float() @ down_weight[expert_idx].float()
            ).to(torch.bfloat16)
            expected_gathered[token_idx, topk_idx] = expert_out
            expected_out[token_idx] += probs[token_idx, topk_idx] * expert_out.float()

    torch.testing.assert_close(gathered_out, expected_gathered, rtol=0.05, atol=0.05)
    torch.testing.assert_close(out, expected_out.to(torch.bfloat16), rtol=0.05, atol=0.05)


def test_v2_ep_wave_has_no_deepep_runtime_dependency():
    repo_root = Path(__file__).resolve().parents[4]
    assert not (repo_root / "src" / "olmo_core" / "kernels" / "deepgemm_mega").exists()
    files = [
        repo_root / "pyproject.toml",
        repo_root / "src" / "olmo_core" / "nn" / "ddp" / "block.py",
        repo_root / "src" / "olmo_core" / "nn" / "moe" / "v2" / "ep_wave.py",
        repo_root / "src" / "olmo_core" / "kernels" / "wave_mega_ep.py",
        repo_root / "src" / "olmo_core" / "kernels" / "cuda" / "symm_mem_vdev2d_kernel.cu",
        repo_root / "src" / "olmo_core" / "kernels" / "cuda" / "wave_mega_ep.cpp",
        repo_root / "src" / "olmo_core" / "kernels" / "cuda" / "symm_mem_vdev2d_wave_kernel.cu",
        repo_root / "src" / "olmo_core" / "kernels" / "cuda" / "olmo_bf16_mega_moe" / "barrier.cuh",
        repo_root / "src" / "olmo_core" / "kernels" / "cuda" / "olmo_bf16_mega_moe" / "buffers.cuh",
        repo_root / "src" / "olmo_core" / "kernels" / "cuda" / "olmo_bf16_mega_moe" / "dispatch.cuh",
        repo_root / "src" / "olmo_core" / "kernels" / "cuda" / "olmo_bf16_mega_moe" / "layout.cuh",
        repo_root / "src" / "olmo_core" / "kernels" / "cuda" / "olmo_bf16_mega_moe" / "megakernel_plan.cuh",
        repo_root / "src" / "olmo_core" / "kernels" / "cuda" / "olmo_bf16_mega_moe" / "mma.cuh",
        repo_root / "src" / "olmo_core" / "kernels" / "cuda" / "olmo_bf16_mega_moe" / "ptx.cuh",
        repo_root / "src" / "olmo_core" / "kernels" / "cuda" / "olmo_bf16_mega_moe" / "runtime.cuh",
        repo_root / "src" / "olmo_core" / "kernels" / "cuda" / "olmo_bf16_mega_moe" / "scheduler.cuh",
        repo_root / "src" / "olmo_core" / "kernels" / "cuda" / "olmo_bf16_mega_moe" / "shared_storage.cuh",
        repo_root / "src" / "olmo_core" / "kernels" / "cuda" / "olmo_bf16_mega_moe" / "tensor_map.cuh",
        repo_root / "src" / "olmo_core" / "kernels" / "cuda" / "olmo_bf16_mega_moe" / "forward.cuh",
        repo_root / "src" / "olmo_core" / "kernels" / "cuda" / "olmo_bf16_mega_moe" / "forward_kernels.cuh",
    ]
    for path in files:
        text = path.read_text()
        lowered = text.lower()
        assert "import deep_ep" not in lowered
        assert "from deep_ep" not in lowered
        assert "deepep" not in lowered


def test_v2_ep_wave_has_no_cuda_synchronization_calls():
    repo_root = Path(__file__).resolve().parents[4]
    files = [
        repo_root / "src" / "olmo_core" / "nn" / "ddp" / "block.py",
        repo_root / "src" / "olmo_core" / "nn" / "moe" / "v2" / "ep_wave.py",
        repo_root / "src" / "olmo_core" / "kernels" / "symm_mem_vdev2d.py",
        repo_root / "src" / "olmo_core" / "kernels" / "wave_mega_ep.py",
        repo_root / "src" / "olmo_core" / "kernels" / "cuda" / "symm_mem_vdev2d.cpp",
        repo_root / "src" / "olmo_core" / "kernels" / "cuda" / "wave_mega_ep.cpp",
        repo_root / "src" / "olmo_core" / "kernels" / "cuda" / "symm_mem_vdev2d_kernel.cu",
        repo_root / "src" / "olmo_core" / "kernels" / "cuda" / "symm_mem_vdev2d_wave_kernel.cu",
        repo_root / "src" / "olmo_core" / "kernels" / "cuda" / "olmo_bf16_mega_moe" / "barrier.cuh",
        repo_root / "src" / "olmo_core" / "kernels" / "cuda" / "olmo_bf16_mega_moe" / "buffers.cuh",
        repo_root / "src" / "olmo_core" / "kernels" / "cuda" / "olmo_bf16_mega_moe" / "dispatch.cuh",
        repo_root / "src" / "olmo_core" / "kernels" / "cuda" / "olmo_bf16_mega_moe" / "layout.cuh",
        repo_root / "src" / "olmo_core" / "kernels" / "cuda" / "olmo_bf16_mega_moe" / "megakernel_plan.cuh",
        repo_root / "src" / "olmo_core" / "kernels" / "cuda" / "olmo_bf16_mega_moe" / "mma.cuh",
        repo_root / "src" / "olmo_core" / "kernels" / "cuda" / "olmo_bf16_mega_moe" / "ptx.cuh",
        repo_root / "src" / "olmo_core" / "kernels" / "cuda" / "olmo_bf16_mega_moe" / "runtime.cuh",
        repo_root / "src" / "olmo_core" / "kernels" / "cuda" / "olmo_bf16_mega_moe" / "scheduler.cuh",
        repo_root / "src" / "olmo_core" / "kernels" / "cuda" / "olmo_bf16_mega_moe" / "shared_storage.cuh",
        repo_root / "src" / "olmo_core" / "kernels" / "cuda" / "olmo_bf16_mega_moe" / "tensor_map.cuh",
        repo_root / "src" / "olmo_core" / "kernels" / "cuda" / "olmo_bf16_mega_moe" / "forward.cuh",
        repo_root / "src" / "olmo_core" / "kernels" / "cuda" / "olmo_bf16_mega_moe" / "forward_kernels.cuh",
    ]
    forbidden = (
        "torch.cuda.synchronize",
        "cuda.synchronize",
        "cudaDeviceSynchronize",
        "cudaStreamSynchronize",
    )
    for path in files:
        text = path.read_text()
        for pattern in forbidden:
            assert pattern not in text, f"{pattern} found in {path}"


def test_v2_ep_wave_uses_dedicated_extension_boundary():
    from olmo_core.kernels import symm_mem_vdev2d, wave_mega_ep

    assert wave_mega_ep._EXTENSION_MODULE_NAME.endswith("._wave_mega_ep_ext_gpu")
    assert symm_mem_vdev2d._EXTENSION_MODULE_NAME.endswith("._symm_mem_vdev2d_ext_gpu")
    assert (
        symm_mem_vdev2d.rowwise_bf16_mega_moe_standard_ep_workspace_config.__module__
        == "olmo_core.kernels.wave_mega_ep"
    )

    repo_root = Path(__file__).resolve().parents[4]
    cmake_source = (
        repo_root / "src" / "olmo_core" / "kernels" / "cuda" / "CMakeLists.txt"
    ).read_text()
    symm_target_source = cmake_source.split("set(WAVE_MEGA_TARGET_NAME", maxsplit=1)[0]
    wave_target_source = cmake_source.split("set(WAVE_MEGA_TARGET_NAME", maxsplit=1)[1]
    assert "symm_mem_vdev2d_wave_kernel.cu" not in symm_target_source
    assert "wave_mega_ep.cpp" in wave_target_source
    assert "symm_mem_vdev2d_wave_kernel.cu" in wave_target_source


def test_v2_ep_wave_training_script_exposes_wave_toggles():
    repo_root = Path(__file__).resolve().parents[4]
    script = repo_root / "src" / "scripts" / "train" / "OLMoE3-dev-260614-wave-ep.py"
    text = script.read_text()

    expected_snippets = [
        "OLMO_WAVE_EP_TOP_K",
        "OLMO_WAVE_EP_CAPACITY_FACTOR",
        "OLMO_WAVE_EP_BF16_PERSISTENT_MEGA",
        "OLMO_WAVE_EP_ROWWISE_NBLOCKS",
        "USE_TBO=False",
        "ep_no_sync_use_wave=USE_WAVE_EP",
        "ep_no_sync_wave_use_bf16_persistent_mega_forward=USE_WAVE_BF16_PERSISTENT_MEGA_FORWARD",
        "ep_no_sync_capacity_factor=EP_NO_SYNC_CAPACITY_FACTOR",
    ]
    for snippet in expected_snippets:
        assert snippet in text


def test_v2_s003_script_exposes_persistent_mega_wave_toggle():
    repo_root = Path(__file__).resolve().parents[4]
    script = repo_root / "src" / "scripts" / "train" / "OLMoE3-dev-260614-s003.py"
    text = script.read_text()

    expected_snippets = [
        "OLMO_S003_USE_BF16_PERSISTENT_MEGA_EP",
        "OLMO_EP_WAVE_USE_BF16_PERSISTENT_MEGA",
        "OLMO_S003_EXPERIMENTAL_WAVE_ALLOW_COMPILE",
        "USE_BF16_PERSISTENT_MEGA_EP",
        "USE_EXPERIMENTAL_WAVE_EP",
        "DType.bfloat16 if USE_EXPERIMENTAL_WAVE_EP else DType.float32",
        "ep_no_sync_use_wave=USE_EXPERIMENTAL_WAVE_EP",
        "ep_no_sync_wave_use_bf16_persistent_mega_forward=USE_BF16_PERSISTENT_MEGA_EP",
    ]
    for snippet in expected_snippets:
        assert snippet in text


def _init_block_params(block: OLMoDDPTransformerBlock):
    torch.manual_seed(1234)
    with torch.no_grad():
        for p in block.parameters():
            if p.is_floating_point():
                p.normal_(mean=0.0, std=0.02)


def _install_forced_router(block: OLMoDDPTransformerBlock):
    def _make_forced_forward(router):
        def _forced_forward(local_x, scores_only, loss_div_factor=None):
            del loss_div_factor
            B, S, _ = local_x.shape
            if scores_only:
                return torch.ones(
                    B,
                    S,
                    router.num_experts,
                    device=local_x.device,
                    dtype=local_x.dtype,
                ), None, None, None

            expert_weights = torch.ones(
                B,
                S,
                router.top_k,
                device=local_x.device,
                dtype=local_x.dtype,
            )
            expert_indices = torch.zeros(
                B,
                S,
                router.top_k,
                device=local_x.device,
                dtype=torch.long,
            )
            batch_size_per_expert = torch.zeros(
                router.num_experts,
                device=local_x.device,
                dtype=torch.long,
            )
            batch_size_per_expert[0] = B * S * router.top_k
            return expert_weights, expert_indices, batch_size_per_expert, None

        return _forced_forward

    assert block.routed_experts_router is not None
    block.routed_experts_router.forward = _make_forced_forward(block.routed_experts_router)
    if block.shared_experts_router is not None:
        block.shared_experts_router.forward = _make_forced_forward(block.shared_experts_router)


def _install_deterministic_topk_router(block: OLMoDDPTransformerBlock):
    def _make_deterministic_forward(router):
        def _deterministic_forward(local_x, scores_only, loss_div_factor=None):
            del loss_div_factor
            B, S, _ = local_x.shape
            if scores_only:
                return torch.ones(
                    B,
                    S,
                    router.num_experts,
                    device=local_x.device,
                    dtype=local_x.dtype,
                ), None, None, None

            top_k = router.top_k
            num_experts = router.num_experts
            token_ids = torch.arange(B * S, device=local_x.device, dtype=torch.long).unsqueeze(1)
            route_offsets = torch.arange(top_k, device=local_x.device, dtype=torch.long).unsqueeze(0)
            expert_indices = (token_ids + route_offsets + dist.get_rank() * 3) % num_experts
            expert_indices = expert_indices.view(B, S, top_k)

            weights = torch.arange(1, top_k + 1, device=local_x.device, dtype=local_x.dtype)
            weights = weights / weights.sum().clamp_min(1e-6)
            expert_weights = weights.view(1, 1, top_k).expand(B, S, top_k).contiguous()

            batch_size_per_expert = torch.bincount(
                expert_indices.reshape(-1), minlength=num_experts
            ).to(dtype=torch.long)
            return expert_weights, expert_indices, batch_size_per_expert, None

        return _deterministic_forward

    assert block.routed_experts_router is not None
    block.routed_experts_router.forward = _make_deterministic_forward(block.routed_experts_router)
    if block.shared_experts_router is not None:
        block.shared_experts_router.forward = _make_deterministic_forward(block.shared_experts_router)


def _install_local_deterministic_topk_router(block: OLMoDDPTransformerBlock):
    """Deterministic router for single-process tests that do not initialize dist."""

    def _make_deterministic_forward(router):
        def _deterministic_forward(local_x, scores_only, loss_div_factor=None):
            del scores_only, loss_div_factor
            B, S, _ = local_x.shape
            tokens = B * S
            token_ids = torch.arange(tokens, device=local_x.device).unsqueeze(1)
            route_offsets = torch.arange(router.top_k, device=local_x.device).unsqueeze(0)
            expert_indices = (token_ids * 3 + route_offsets * 5) % router.num_experts
            logits = torch.linspace(
                0.25,
                0.75,
                steps=router.top_k,
                device=local_x.device,
                dtype=torch.float32,
            ).view(1, router.top_k)
            expert_weights = torch.softmax(logits.expand(tokens, router.top_k), dim=-1).to(
                dtype=local_x.dtype
            )
            batch_size_per_expert = torch.bincount(
                expert_indices.reshape(-1),
                minlength=router.num_experts,
            ).to(dtype=torch.int32)
            return expert_weights, expert_indices.to(dtype=torch.long), batch_size_per_expert, None

        return _deterministic_forward

    assert block.routed_experts_router is not None
    block.routed_experts_router.forward = _make_deterministic_forward(block.routed_experts_router)
    if block.shared_experts_router is not None:
        block.shared_experts_router.forward = _make_deterministic_forward(block.shared_experts_router)


@requires_gpu
@requires_grouped_gemm
def test_v2_no_ep_forward_backward_smoke():
    block = _build_block(ep_no_sync=False, init_device="cuda")
    _init_block_params(block)
    _install_forced_router(block)
    block.train()

    x = torch.randn(1, 8, block.d_model, device="cuda", dtype=torch.float32, requires_grad=True)
    y = block(x)

    assert y.shape == x.shape
    assert torch.isfinite(y).all()

    y.square().mean().backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    for p in block.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all()


@requires_gpu
@requires_grouped_gemm
def test_v2_no_ep_repeated_forward_backward_is_stable():
    block = _build_block(
        ep_no_sync=False,
        d_model=256,
        hidden_size=512,
        num_experts=8,
        top_k=4,
        num_shared_experts=1,
        shared_hidden_size=256,
        uniform_expert_assignment=False,
        init_device="cuda",
    )
    _init_block_params(block)
    _install_local_deterministic_topk_router(block)
    block.train()

    torch.manual_seed(1234)
    x0 = torch.randn(2, 256, block.d_model, device="cuda", dtype=torch.float32)

    def run_once():
        block.zero_grad(set_to_none=True)
        x = x0.detach().clone().requires_grad_(True)
        y = block(x)
        loss = y.float().square().mean() + 0.03125 * y.float().sum()
        loss.backward()
        torch.cuda.synchronize()
        grads = {
            name: p.grad.detach().clone()
            for name, p in block.named_parameters()
            if p.grad is not None
        }
        assert x.grad is not None
        return y.detach().clone(), x.grad.detach().clone(), grads

    ref_y, ref_x_grad, ref_grads = run_once()
    for _ in range(3):
        y, x_grad, grads = run_once()
        torch.testing.assert_close(y, ref_y, atol=0.0, rtol=0.0)
        torch.testing.assert_close(x_grad, ref_x_grad, atol=2e-8, rtol=0.0)
        for name, ref_grad in ref_grads.items():
            torch.testing.assert_close(grads[name], ref_grad, atol=1e-6, rtol=0.0)


@requires_gpu
@requires_grouped_gemm
def test_v2_no_ep_apply_compile_forward_smoke():
    block = _build_block(
        ep_no_sync=False,
        d_model=128,
        hidden_size=256,
        num_experts=4,
        top_k=1,
        init_device="cuda",
    )
    _init_block_params(block)
    block.to(dtype=torch.bfloat16)
    _install_forced_router(block)
    block.train()
    block.apply_compile()

    x = torch.randn(1, 4, block.d_model, device="cuda", dtype=torch.bfloat16)
    y = block(x)

    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def _run_ep_no_sync_matches_synced():
    ep_mesh = _build_ep_mesh()

    block_ep = _build_block(ep_no_sync=False, uniform_expert_assignment=True)
    block_no_sync = _build_block(ep_no_sync=True, uniform_expert_assignment=True)
    block_ep.apply_ep(ep_mesh)
    block_no_sync.apply_ep(ep_mesh)

    _init_block_params(block_ep)
    block_no_sync.load_state_dict(block_ep.state_dict())
    block_ep.train()
    block_no_sync.train()

    x = torch.randn(1, 8, block_ep.d_model, device="cuda", dtype=torch.float32, requires_grad=True)
    x_ref = x.detach().clone().requires_grad_(True)

    y_ep = block_ep(x)
    y_no_sync = block_no_sync(x_ref)
    torch.testing.assert_close(y_no_sync, y_ep, atol=2e-4, rtol=2e-4)

    y_ep.sum().backward()
    y_no_sync.sum().backward()
    torch.testing.assert_close(x_ref.grad, x.grad, atol=3e-4, rtol=3e-4)

    ep_params = dict(block_ep.named_parameters())
    no_sync_params = dict(block_no_sync.named_parameters())
    for name, p_ep in ep_params.items():
        p_no_sync = no_sync_params[name]
        if p_ep.grad is None or p_no_sync.grad is None:
            continue
        torch.testing.assert_close(p_no_sync.grad, p_ep.grad, atol=8e-4, rtol=8e-4)


def _run_ep_no_sync_drop_behavior():
    ep_mesh = _build_ep_mesh()

    block_no_sync = _build_block(
        ep_no_sync=True,
        ep_no_sync_capacity_factor=0.25,
        uniform_expert_assignment=False,
    )
    block_no_sync.apply_ep(ep_mesh)
    _init_block_params(block_no_sync)
    _install_forced_router(block_no_sync)
    block_no_sync.train()

    x = torch.randn(1, 8, block_no_sync.d_model, device="cuda", dtype=torch.float32, requires_grad=True)
    y = block_no_sync(x)
    assert torch.isfinite(y).all()
    y.sum().backward()
    assert x.grad is not None

    dbg = block_no_sync._ep_no_sync_last_debug
    assert dbg["num_dropped"].item() > 0
    assert dbg["received_tokens_after_drop"].item() <= dbg["rank_capacity"].item()
    assert dbg["allowed_splits"].sum().item() == dbg["local_kept_tokens"].item()
    assert dbg["combined_tokens"].item() == dbg["local_kept_tokens"].item()
    assert dbg["zero_rows_after_local_unpermute"].item() >= dbg["num_dropped"].item()


def _run_ep_no_sync_quota_invariants():
    ep_mesh = _build_ep_mesh()

    block_no_sync = _build_block(
        ep_no_sync=True,
        ep_no_sync_capacity_factor=0.5,
        uniform_expert_assignment=False,
    )
    block_no_sync.apply_ep(ep_mesh)
    _init_block_params(block_no_sync)
    _install_forced_router(block_no_sync)
    block_no_sync.train()

    x = torch.randn(1, 8, block_no_sync.d_model, device="cuda", dtype=torch.float32, requires_grad=True)
    y = block_no_sync(x)
    y.sum().backward()

    dbg = block_no_sync._ep_no_sync_last_debug
    assert dbg["allowed_splits"].sum().item() == dbg["local_kept_tokens"].item()
    assert dbg["received_tokens_after_drop"].item() <= dbg["rank_capacity"].item()
    assert dbg["combined_tokens"].item() == dbg["local_kept_tokens"].item()


def _run_ep_no_sync_hard_fail_setup():
    import olmo_core.nn.ddp.block as block_module

    ep_mesh = _build_ep_mesh()
    old_symm = block_module._symm_mem
    block_module._symm_mem = None
    try:
        block = _build_block(ep_no_sync=True)
        try:
            block.apply_ep(ep_mesh)
        except RuntimeError:
            pass
        else:
            raise AssertionError("Expected RuntimeError when symmetric memory is unavailable")
    finally:
        block_module._symm_mem = old_symm


def _run_ep_no_sync_rowwise_matches_synced():
    ep_mesh = _build_ep_mesh()

    block_ep = _build_block(
        ep_no_sync=False,
        d_model=512,
        hidden_size=1024,
        num_experts=8,
        top_k=2,
        uniform_expert_assignment=False,
    )
    block_rowwise = _build_block(
        ep_no_sync=True,
        ep_no_sync_use_rowwise_all_to_all=True,
        d_model=512,
        hidden_size=1024,
        num_experts=8,
        top_k=2,
        uniform_expert_assignment=False,
    )
    block_ep.apply_ep(ep_mesh)
    block_rowwise.apply_ep(ep_mesh)

    _init_block_params(block_ep)
    block_rowwise.load_state_dict(block_ep.state_dict())
    _install_deterministic_topk_router(block_ep)
    _install_deterministic_topk_router(block_rowwise)

    block_rowwise.ep.rowwise_nblocks = 128
    block_rowwise.ep.validate()

    block_ep.train()
    block_rowwise.train()

    x = torch.randn(1, 8, block_ep.d_model, device="cuda", dtype=torch.float32, requires_grad=True)
    x_rowwise = x.detach().clone().requires_grad_(True)

    y_ep = block_ep(x)
    y_rowwise = block_rowwise(x_rowwise)
    torch.testing.assert_close(y_rowwise, y_ep, atol=5e-4, rtol=5e-4)

    loss_ep = y_ep.square().mean() + (0.1 * y_ep.sum())
    loss_rowwise = y_rowwise.square().mean() + (0.1 * y_rowwise.sum())
    loss_ep.backward()
    loss_rowwise.backward()

    torch.testing.assert_close(x_rowwise.grad, x.grad, atol=5e-4, rtol=5e-4)

    ep_params = dict(block_ep.named_parameters())
    rowwise_params = dict(block_rowwise.named_parameters())
    for name, p_ep in ep_params.items():
        p_rowwise = rowwise_params[name]
        if p_ep.grad is None or p_rowwise.grad is None:
            continue
        torch.testing.assert_close(p_rowwise.grad, p_ep.grad, atol=1e-3, rtol=1e-3)


def _run_ep_no_sync_rowwise_wave_matches_rowwise():
    ep_mesh = _build_ep_mesh()

    block_rowwise = _build_block(
        ep_no_sync=True,
        ep_no_sync_use_rowwise_all_to_all=True,
        d_model=128,
        hidden_size=256,
        num_experts=8,
        top_k=4,
        uniform_expert_assignment=False,
    )
    block_wave = _build_block(
        ep_no_sync=False,
        ep=ExpertParallelConfig(
            path=ExpertParallelPath.rowwise_wave,
            capacity_factor=2.0,
            rowwise_nblocks=128,
            rowwise_wave_num_waves=2,
        ),
        d_model=128,
        hidden_size=256,
        num_experts=8,
        top_k=4,
        uniform_expert_assignment=False,
    )
    block_rowwise.apply_ep(ep_mesh)
    block_wave.apply_ep(ep_mesh)

    _init_block_params(block_rowwise)
    block_wave.load_state_dict(block_rowwise.state_dict())
    _install_deterministic_topk_router(block_rowwise)
    _install_deterministic_topk_router(block_wave)

    block_rowwise.ep.rowwise_nblocks = 128
    block_wave.ep.rowwise_nblocks = 128
    block_rowwise.ep.validate()
    block_wave.ep.validate()
    block_rowwise.train()
    block_wave.train()

    x = torch.randn(1, 16, block_rowwise.d_model, device="cuda", dtype=torch.float32, requires_grad=True)
    x_wave = x.detach().clone().requires_grad_(True)

    with pytest.warns(RuntimeWarning, match="rowwise_wave"):
        y_wave = block_wave(x_wave)
    y_rowwise = block_rowwise(x)
    assert y_wave.shape == y_rowwise.shape
    assert torch.isfinite(y_wave).all()
    torch.testing.assert_close(y_wave, y_rowwise, atol=1e-3, rtol=1e-3)

    loss_rowwise = y_rowwise.square().mean() + (0.1 * y_rowwise.sum())
    loss_wave = y_wave.square().mean() + (0.1 * y_wave.sum())
    loss_rowwise.backward()
    loss_wave.backward()

    assert x_wave.grad is not None
    assert x.grad is not None
    torch.testing.assert_close(x_wave.grad, x.grad, atol=2e-3, rtol=2e-3)

    rowwise_params = dict(block_rowwise.named_parameters())
    wave_params = dict(block_wave.named_parameters())
    for name, p_rowwise in rowwise_params.items():
        p_wave = wave_params[name]
        if p_rowwise.grad is None or p_wave.grad is None:
            continue
        torch.testing.assert_close(
            p_wave.grad,
            p_rowwise.grad,
            atol=3e-3,
            rtol=3e-3,
            msg=f"rowwise_wave gradient mismatch for {name}",
        )

    block_rowwise_eval = _build_block(
        ep_no_sync=True,
        ep_no_sync_use_rowwise_all_to_all=True,
        d_model=128,
        hidden_size=256,
        num_experts=8,
        top_k=4,
        uniform_expert_assignment=False,
    )
    block_wave_eval = _build_block(
        ep_no_sync=False,
        ep=ExpertParallelConfig(
            path=ExpertParallelPath.rowwise_wave,
            capacity_factor=2.0,
            rowwise_nblocks=32,
            rowwise_wave_num_waves=4,
        ),
        d_model=128,
        hidden_size=256,
        num_experts=8,
        top_k=4,
        uniform_expert_assignment=False,
    )
    block_rowwise_eval.apply_ep(ep_mesh)
    block_wave_eval.apply_ep(ep_mesh)

    _init_block_params(block_rowwise_eval)
    block_wave_eval.load_state_dict(block_rowwise_eval.state_dict())
    _install_deterministic_topk_router(block_rowwise_eval)
    _install_deterministic_topk_router(block_wave_eval)

    block_rowwise_eval.to(dtype=torch.bfloat16)
    block_wave_eval.to(dtype=torch.bfloat16)
    block_rowwise_eval.ep.rowwise_nblocks = 32
    block_wave_eval.ep.rowwise_nblocks = 32
    block_rowwise_eval.ep.validate()
    block_wave_eval.ep.validate()
    block_rowwise_eval.eval()
    block_wave_eval.eval()

    x_eval = (0.2 * torch.randn(1, 16, block_rowwise_eval.d_model, device="cuda")).to(
        dtype=torch.bfloat16
    )
    with torch.no_grad():
        y_rowwise_eval = block_rowwise_eval(x_eval)
        y_wave_eval = block_wave_eval(x_eval.detach().clone())
    assert y_wave_eval.shape == y_rowwise_eval.shape
    assert torch.isfinite(y_wave_eval).all()
    torch.testing.assert_close(y_wave_eval, y_rowwise_eval, atol=2e-2, rtol=2e-2)


def _run_ep_no_sync_tma_ibgda_matches_synced_bf16(
    *,
    symmetric_expert_out: bool = False,
    run_backward: bool = True,
):
    ep_mesh = _build_ep_mesh()

    block_ep = _build_block(
        ep_no_sync=False,
        d_model=128,
        hidden_size=256,
        num_experts=8,
        top_k=2,
        uniform_expert_assignment=False,
    )
    block_tma = _build_block(
        ep_no_sync=True,
        ep_no_sync_use_rowwise_all_to_all=True,
        ep_no_sync_rowwise_backend="tma_ibgda",
        d_model=128,
        hidden_size=256,
        num_experts=8,
        top_k=2,
        uniform_expert_assignment=False,
        ep_no_sync_tma_ibgda_symmetric_expert_out=symmetric_expert_out,
    )
    block_ep.apply_ep(ep_mesh)
    block_tma.apply_ep(ep_mesh)

    _init_block_params(block_ep)
    block_tma.load_state_dict(block_ep.state_dict())
    _install_deterministic_topk_router(block_ep)
    _install_deterministic_topk_router(block_tma)

    block_ep.to(dtype=torch.bfloat16)
    block_tma.to(dtype=torch.bfloat16)
    if run_backward:
        block_ep.train()
        block_tma.train()
    else:
        block_ep.eval()
        block_tma.eval()
    block_tma.ep.rowwise_nblocks = 128
    block_tma.ep.validate()

    x = (0.2 * torch.randn(1, 16, block_ep.d_model, device="cuda")).to(
        dtype=torch.bfloat16
    )
    x_ep = x.detach().clone().requires_grad_(run_backward)
    x_tma = x.detach().clone().requires_grad_(run_backward)

    with torch.set_grad_enabled(run_backward):
        y_ep = block_ep(x_ep)
        y_tma = block_tma(x_tma)
    assert y_tma.shape == y_ep.shape
    assert torch.isfinite(y_tma).all()
    torch.testing.assert_close(y_tma, y_ep, atol=2e-2, rtol=2e-2)
    if not run_backward:
        return

    loss_ep = y_ep.float().square().mean() + 0.1 * y_ep.float().sum()
    loss_tma = y_tma.float().square().mean() + 0.1 * y_tma.float().sum()
    loss_ep.backward()
    loss_tma.backward()

    assert x_tma.grad is not None
    assert x_ep.grad is not None
    torch.testing.assert_close(x_tma.grad, x_ep.grad, atol=3e-2, rtol=3e-2)

    ep_params = dict(block_ep.named_parameters())
    tma_params = dict(block_tma.named_parameters())
    for name, p_ep in ep_params.items():
        p_tma = tma_params[name]
        if p_ep.grad is None or p_tma.grad is None:
            continue
        torch.testing.assert_close(
            p_tma.grad,
            p_ep.grad,
            atol=5e-2,
            rtol=5e-2,
            msg=f"TMA/IBGDA gradient mismatch for {name}",
        )


def _run_ep_wave_forward_matches_rowwise_forward():
    old_allow_fallback = os.environ.get("OLMO_EP_WAVE_ALLOW_ROWWISE_FALLBACK")
    os.environ["OLMO_EP_WAVE_ALLOW_ROWWISE_FALLBACK"] = "1"
    ep_mesh = _build_ep_mesh()

    try:
        block_rowwise = _build_block(
            ep_no_sync=True,
            ep_no_sync_use_rowwise_all_to_all=True,
            d_model=128,
            hidden_size=128,
            num_experts=8,
            top_k=2,
            uniform_expert_assignment=False,
        )
        block_wave = _build_block(
            ep_no_sync=True,
            ep_no_sync_use_rowwise_all_to_all=True,
            ep_no_sync_use_wave=True,
            ep_no_sync_wave_use_bf16_persistent_mega_forward=True,
            d_model=128,
            hidden_size=128,
            num_experts=8,
            top_k=2,
            uniform_expert_assignment=False,
        )
        block_rowwise.apply_ep(ep_mesh)
        block_wave.apply_ep(ep_mesh)

        _init_block_params(block_rowwise)
        block_wave.load_state_dict(block_rowwise.state_dict())
        _install_deterministic_topk_router(block_rowwise)
        _install_deterministic_topk_router(block_wave)

        block_rowwise.to(dtype=torch.bfloat16)
        block_wave.to(dtype=torch.bfloat16)
        block_rowwise.eval()
        block_wave.eval()
        block_rowwise.ep.rowwise_nblocks = 128
        block_wave.ep.rowwise_nblocks = 128
        block_rowwise.ep.validate()
        block_wave.ep.validate()

        x = (0.2 * torch.randn(1, 64, block_rowwise.d_model, device="cuda")).to(torch.bfloat16)
        with torch.no_grad():
            y_rowwise = block_rowwise(x)
            y_wave = block_wave(x.detach().clone())

        assert y_wave.shape == y_rowwise.shape
        assert torch.isfinite(y_wave).all()
        torch.testing.assert_close(y_wave, y_rowwise, atol=8e-2, rtol=8e-2)
    finally:
        if old_allow_fallback is None:
            os.environ.pop("OLMO_EP_WAVE_ALLOW_ROWWISE_FALLBACK", None)
        else:
            os.environ["OLMO_EP_WAVE_ALLOW_ROWWISE_FALLBACK"] = old_allow_fallback


def _run_ep_wave_standard_peer_group_forward_matches_rowwise_forward():
    ep_mesh = _build_ep_mesh()

    block_rowwise = _build_block(
        ep_no_sync=True,
        ep_no_sync_use_rowwise_all_to_all=True,
        d_model=16,
        hidden_size=16,
        num_experts=32,
        top_k=4,
        num_shared_experts=0,
        uniform_expert_assignment=False,
    )
    block_wave = _build_block(
        ep_no_sync=True,
        ep_no_sync_use_rowwise_all_to_all=True,
        ep_no_sync_use_wave=True,
        ep_no_sync_wave_use_bf16_persistent_mega_forward=True,
        d_model=16,
        hidden_size=16,
        num_experts=32,
        top_k=4,
        num_shared_experts=0,
        uniform_expert_assignment=False,
    )
    block_rowwise.apply_ep(ep_mesh)
    block_wave.apply_ep(ep_mesh)

    _init_block_params(block_rowwise)
    block_wave.load_state_dict(block_rowwise.state_dict())
    _install_deterministic_topk_router(block_rowwise)
    _install_deterministic_topk_router(block_wave)

    block_rowwise.to(dtype=torch.bfloat16)
    block_wave.to(dtype=torch.bfloat16)
    block_rowwise.eval()
    block_wave.eval()
    block_rowwise.ep.rowwise_nblocks = 128
    block_wave.ep.rowwise_nblocks = 128
    block_rowwise.ep.validate()
    block_wave.ep.validate()

    x = (0.2 * torch.randn(1, 8, block_rowwise.d_model, device="cuda")).to(torch.bfloat16)
    with torch.no_grad():
        y_rowwise = block_rowwise(x)
        y_wave = block_wave(x.detach().clone())

    assert y_wave.shape == y_rowwise.shape
    assert torch.isfinite(y_wave).all()
    torch.testing.assert_close(y_wave, y_rowwise, atol=8e-2, rtol=8e-2)


def _run_ep_no_sync_rowwise_drop_matches_independent_rowwise_block():
    ep_mesh = _build_ep_mesh()

    block_a = _build_block(
        ep_no_sync=True,
        ep_no_sync_use_rowwise_all_to_all=True,
        ep_no_sync_capacity_factor=0.5,
        d_model=512,
        hidden_size=1024,
        num_experts=8,
        top_k=4,
        uniform_expert_assignment=False,
    )
    block_b = _build_block(
        ep_no_sync=True,
        ep_no_sync_use_rowwise_all_to_all=True,
        ep_no_sync_capacity_factor=0.5,
        d_model=512,
        hidden_size=1024,
        num_experts=8,
        top_k=4,
        uniform_expert_assignment=False,
    )
    block_a.apply_ep(ep_mesh)
    block_b.apply_ep(ep_mesh)

    _init_block_params(block_a)
    block_b.load_state_dict(block_a.state_dict())
    _install_deterministic_topk_router(block_a)
    _install_deterministic_topk_router(block_b)

    block_a.ep.rowwise_nblocks = 128
    block_a.ep.validate()

    block_b.ep.rowwise_nblocks = 128
    block_b.ep.validate()

    block_a.train()
    block_b.train()

    x = torch.randn(2, 64, block_a.d_model, device="cuda", dtype=torch.float32, requires_grad=True)
    x_b = x.detach().clone().requires_grad_(True)

    y_a = block_a(x)
    y_b = block_b(x_b)
    assert torch.isfinite(y_a).all()
    assert torch.isfinite(y_b).all()
    torch.testing.assert_close(y_b, y_a, atol=8e-4, rtol=8e-4)

    loss_a = y_a.square().mean() + (0.1 * y_a.sum())
    loss_b = y_b.square().mean() + (0.1 * y_b.sum())
    loss_a.backward()
    loss_b.backward()

    assert torch.isfinite(x.grad).all()
    assert torch.isfinite(x_b.grad).all()
    torch.testing.assert_close(x_b.grad, x.grad, atol=2e-3, rtol=2e-3)

    params_a = dict(block_a.named_parameters())
    params_b = dict(block_b.named_parameters())
    for name, p_a in params_a.items():
        p_b = params_b[name]
        if p_a.grad is None or p_b.grad is None:
            continue
        assert torch.isfinite(p_a.grad).all()
        assert torch.isfinite(p_b.grad).all()
        torch.testing.assert_close(p_b.grad, p_a.grad, atol=3e-3, rtol=3e-3)


def _poison_rowwise_capacity_tails(block: OLMoDDPTransformerBlock, *, value: float) -> int:
    recv_splits = getattr(block, "_debug_rowwise_recv_splits_by_src_local", None)
    if recv_splits is None:
        raise RuntimeError("rowwise debug tensors were not captured")
    valid_rows = int(recv_splits.sum().item())
    poisoned_rows = 0

    def poison_tensor(tensor: torch.Tensor | None) -> None:
        nonlocal poisoned_rows
        if tensor is None or tensor.ndim != 2 or not tensor.is_floating_point():
            return
        tail_rows = int(tensor.shape[0]) - valid_rows
        if tail_rows <= 0:
            return
        tensor.narrow(0, valid_rows, tail_rows).fill_(value)
        poisoned_rows += tail_rows

    with torch.no_grad():
        pools = getattr(block, "_ep_no_sync_symm_lease_pools", {})
        dispatch_pool = pools.get("dispatch_out")
        if dispatch_pool is not None:
            for slot in dispatch_pool._slots:
                poison_tensor(slot.get("dispatch_out"))

        for buffers in getattr(block, "_ep_no_sync_static_buffer_cache", {}).values():
            poison_tensor(getattr(buffers, "combine_in", None))

    return poisoned_rows


def _run_ep_no_sync_rowwise_capacity_tail_poison_does_not_change_backward():
    old_debug = os.environ.get("OLMO_MOE_ROWWISE_DEBUG_TENSORS")
    os.environ["OLMO_MOE_ROWWISE_DEBUG_TENSORS"] = "1"
    try:
        ep_mesh = _build_ep_mesh()

        block_a = _build_block(
            ep_no_sync=True,
            ep_no_sync_use_rowwise_all_to_all=True,
            ep_no_sync_capacity_factor=8.0,
            d_model=512,
            hidden_size=1024,
            num_experts=8,
            top_k=4,
            uniform_expert_assignment=False,
        )
        block_b = _build_block(
            ep_no_sync=True,
            ep_no_sync_use_rowwise_all_to_all=True,
            ep_no_sync_capacity_factor=8.0,
            d_model=512,
            hidden_size=1024,
            num_experts=8,
            top_k=4,
            uniform_expert_assignment=False,
        )
        block_a.apply_ep(ep_mesh)
        block_b.apply_ep(ep_mesh)

        _init_block_params(block_a)
        block_b.load_state_dict(block_a.state_dict())
        _install_deterministic_topk_router(block_a)
        _install_deterministic_topk_router(block_b)

        block_a.ep.rowwise_nblocks = 128
        block_b.ep.rowwise_nblocks = 128
        block_a.ep.validate()
        block_b.ep.validate()
        block_a.train()
        block_b.train()

        x = torch.randn(2, 64, block_a.d_model, device="cuda", dtype=torch.float32, requires_grad=True)
        x_b = x.detach().clone().requires_grad_(True)

        y_a = block_a(x)
        y_b = block_b(x_b)
        torch.testing.assert_close(y_b, y_a, atol=8e-4, rtol=8e-4)

        poisoned_rows = _poison_rowwise_capacity_tails(block_b, value=2048.0)
        assert poisoned_rows > 0

        loss_a = y_a.square().mean() + (0.1 * y_a.sum())
        loss_b = y_b.square().mean() + (0.1 * y_b.sum())
        loss_a.backward()
        loss_b.backward()

        assert x.grad is not None
        assert x_b.grad is not None
        torch.testing.assert_close(x_b.grad, x.grad, atol=2e-3, rtol=2e-3)

        params_a = dict(block_a.named_parameters())
        params_b = dict(block_b.named_parameters())
        for name, p_a in params_a.items():
            p_b = params_b[name]
            if p_a.grad is None or p_b.grad is None:
                continue
            torch.testing.assert_close(
                p_b.grad,
                p_a.grad,
                atol=3e-3,
                rtol=3e-3,
                msg=f"capacity tail poison changed gradient for {name}",
            )
    finally:
        if old_debug is None:
            os.environ.pop("OLMO_MOE_ROWWISE_DEBUG_TENSORS", None)
        else:
            os.environ["OLMO_MOE_ROWWISE_DEBUG_TENSORS"] = old_debug


@requires_multi_gpu
def test_v2_ep_no_sync_matches_synced():
    run_distributed_test(_run_ep_no_sync_matches_synced, backend="nccl", start_method="spawn")


@requires_multi_gpu
def test_v2_ep_no_sync_drop_behavior():
    run_distributed_test(_run_ep_no_sync_drop_behavior, backend="nccl", start_method="spawn")


@requires_multi_gpu
def test_v2_ep_no_sync_quota_invariants():
    run_distributed_test(_run_ep_no_sync_quota_invariants, backend="nccl", start_method="spawn")


@requires_multi_gpu
def test_v2_ep_no_sync_hard_fail_setup():
    run_distributed_test(_run_ep_no_sync_hard_fail_setup, backend="nccl", start_method="spawn")


@requires_multi_gpu
def test_v2_ep_no_sync_rowwise_matches_synced():
    run_distributed_test(_run_ep_no_sync_rowwise_matches_synced, backend="nccl", start_method="spawn")


@requires_multi_gpu
@requires_grouped_gemm
def test_v2_ep_no_sync_rowwise_wave_matches_rowwise():
    run_distributed_test(
        _run_ep_no_sync_rowwise_wave_matches_rowwise,
        backend="nccl",
        start_method="spawn",
    )


@requires_multi_gpu
@requires_grouped_gemm
def test_v2_ep_no_sync_tma_ibgda_matches_synced_bf16():
    run_distributed_test(
        _run_ep_no_sync_tma_ibgda_matches_synced_bf16,
        backend="nccl",
        start_method="spawn",
    )


@requires_multi_gpu
@requires_grouped_gemm
def test_v2_ep_no_sync_tma_ibgda_symmetric_expert_out_matches_synced_bf16():
    run_distributed_test(
        _run_ep_no_sync_tma_ibgda_matches_synced_bf16,
        backend="nccl",
        start_method="spawn",
        func_kwargs={"symmetric_expert_out": True, "run_backward": False},
    )


@requires_multi_gpu
def test_v2_ep_wave_forward_matches_rowwise_forward():
    if torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("requires Blackwell-class CUDA device")
    run_distributed_test(
        _run_ep_wave_forward_matches_rowwise_forward,
        backend="nccl",
        start_method="spawn",
    )


@requires_multi_gpu
def test_v2_ep_wave_standard_peer_group_forward_matches_rowwise_forward():
    if torch.cuda.device_count() < 4:
        pytest.skip("requires at least 4 GPUs")
    if torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("requires Blackwell-class CUDA device")
    run_distributed_test(
        _run_ep_wave_standard_peer_group_forward_matches_rowwise_forward,
        world_size=4,
        backend="nccl",
        start_method="spawn",
    )


def test_v2_ep_no_sync_2d_all_to_all_rejected():
    with pytest.raises(OLMoConfigurationError, match="2D all_to_all path was removed"):
        _build_block(
            ep_no_sync=True,
            init_device="cpu",
            ep_no_sync_use_2d_all_to_all=True,
        )


@requires_multi_gpu
def test_v2_ep_no_sync_rowwise_drop_matches_independent_rowwise_block():
    run_distributed_test(
        _run_ep_no_sync_rowwise_drop_matches_independent_rowwise_block,
        backend="nccl",
        start_method="spawn",
    )


@requires_multi_gpu
def test_v2_ep_no_sync_rowwise_capacity_tail_poison_does_not_change_backward():
    run_distributed_test(
        _run_ep_no_sync_rowwise_capacity_tail_poison_does_not_change_backward,
        backend="nccl",
        start_method="spawn",
    )
