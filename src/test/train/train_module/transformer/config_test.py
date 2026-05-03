from types import SimpleNamespace

import pytest
import torch

from olmo_core.distributed.parallel import PipelineScheduleType, PipelineSplitStyle
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.train.train_module.transformer import TransformerPipelineParallelConfig


def test_generate_pipeline_split_points():
    pp_config = TransformerPipelineParallelConfig(
        degree=2, schedule=PipelineScheduleType.single_1F1B, style=PipelineSplitStyle.loop
    )
    assert pp_config.get_split_points(4) == [2]

    pp_config = TransformerPipelineParallelConfig(
        degree=4, schedule=PipelineScheduleType.single_1F1B, style=PipelineSplitStyle.loop
    )
    assert pp_config.get_split_points(4) == [1, 2, 3]

    pp_config = TransformerPipelineParallelConfig(
        degree=2, schedule=PipelineScheduleType.interleaved_1F1B, style=PipelineSplitStyle.loop
    )
    assert pp_config.get_split_points(4) == [1, 2, 3]


def test_custom_pipeline_schedule_split_styles():
    pp_config = TransformerPipelineParallelConfig(
        degree=4, schedule=PipelineScheduleType.custom_interleaved_1F1B
    )
    assert pp_config.infer_style() == PipelineSplitStyle.loop
    assert [pp_config.stage_ids_this_rank(rank, 8) for rank in range(4)] == [
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]
    assert pp_config.final_stage_rank() == 3
    assert list(pp_config.rank_completion_order()) == [3, 2, 1, 0]

    pp_config = TransformerPipelineParallelConfig(
        degree=4, schedule=PipelineScheduleType.custom_1F1B_V
    )
    assert pp_config.infer_style() == PipelineSplitStyle.v
    assert [pp_config.stage_ids_this_rank(rank, 8) for rank in range(4)] == [
        (0, 7),
        (1, 6),
        (2, 5),
        (3, 4),
    ]
    assert pp_config.final_stage_rank() == 0
    assert list(pp_config.rank_completion_order()) == [0, 1, 2, 3]


def test_custom_pipeline_schedule_rejects_mismatched_split_styles():
    with pytest.raises(OLMoConfigurationError, match="custom_interleaved_1F1B"):
        TransformerPipelineParallelConfig(
            degree=4,
            schedule=PipelineScheduleType.custom_interleaved_1F1B,
            style=PipelineSplitStyle.v,
        ).infer_style()


def test_pipeline_p2p_rank_groups_follow_pp_dimension():
    mesh = SimpleNamespace(
        mesh_dim_names=("pp", "dp"),
        mesh=torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7]]),
    )

    assert TransformerPipelineParallelConfig._pipeline_rank_groups(mesh) == [
        [0, 2, 4, 6],
        [1, 3, 5, 7],
    ]

    mesh = SimpleNamespace(
        mesh_dim_names=("dp", "pp"),
        mesh=torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]]),
    )

    assert TransformerPipelineParallelConfig._pipeline_rank_groups(mesh) == [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
    ]

    with pytest.raises(OLMoConfigurationError, match="custom_1F1B_V"):
        TransformerPipelineParallelConfig(
            degree=4,
            schedule=PipelineScheduleType.custom_1F1B_V,
            style=PipelineSplitStyle.loop,
        ).infer_style()


def test_pipeline_p2p_nccl_cta_caps_require_separate_group():
    mesh = SimpleNamespace(mesh_dim_names=("pp",), mesh=torch.tensor([0, 1, 2, 3]))
    pp_config = TransformerPipelineParallelConfig(
        degree=4,
        p2p_nccl_min_ctas=1,
        p2p_nccl_max_ctas=2,
    )

    with pytest.raises(OLMoConfigurationError, match="p2p_use_separate_group"):
        pp_config.build_p2p_process_group(mesh)


def test_pipeline_p2p_nccl_cta_caps_validate_range():
    pp_config = TransformerPipelineParallelConfig(
        degree=4,
        p2p_use_separate_group=True,
        p2p_nccl_min_ctas=4,
        p2p_nccl_max_ctas=2,
    )

    with pytest.raises(OLMoConfigurationError, match="cannot exceed"):
        pp_config._validate_p2p_nccl_ctas()
