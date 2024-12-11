from olmo_core.distributed.parallel import PipelineScheduleType
from olmo_core.train.train_module.transformer import TransformerPipelineParallelConfig


def test_generate_pipeline_split_points():
    pp_config = TransformerPipelineParallelConfig(
        degree=2, schedule=PipelineScheduleType.single_1F1B
    )
    assert pp_config.get_split_points(4) == [2]

    pp_config = TransformerPipelineParallelConfig(
        degree=4, schedule=PipelineScheduleType.single_1F1B
    )
    assert pp_config.get_split_points(4) == [1, 2, 3]

    pp_config = TransformerPipelineParallelConfig(
        degree=2, schedule=PipelineScheduleType.interleaved_1F1B
    )
    assert pp_config.get_split_points(4) == [1, 2, 3]
