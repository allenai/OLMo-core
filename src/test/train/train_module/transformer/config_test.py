import torch

from olmo_core.distributed.parallel import (
    DataParallelType,
    PipelineScheduleType,
    PipelineSplitStyle,
)
from olmo_core.nn.feed_forward import FeedForwardConfig
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.optim import AdamWConfig
from olmo_core.testing import run_distributed_test
from olmo_core.train.train_module.transformer import (
    TransformerDataParallelConfig,
    TransformerPipelineParallelConfig,
    TransformerTrainModuleConfig,
)


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


def _run_pp_num_flops_per_token():
    """
    Verifies that TransformerPipelineTrainModule.num_flops_per_token returns total-model
    FLOPs (not just the local pipeline stage's FLOPs) by comparing against the full
    unsplit model.
    """
    device = torch.device("cpu")
    seq_len = 512

    transformer_config = TransformerConfig.llama_like(
        d_model=64,
        vocab_size=128,
        n_layers=4,
        n_heads=2,
        feed_forward=FeedForwardConfig(hidden_size=128, bias=False),
    )

    # Expected FLOPs from the full model (all 4 layers + lm_head).
    expected_flops = transformer_config.build(init_device="meta").num_flops_per_token(seq_len)

    pp_config = TransformerPipelineParallelConfig(
        degree=2, schedule=PipelineScheduleType.single_1F1B, style=PipelineSplitStyle.loop
    )
    dp_config = TransformerDataParallelConfig(name=DataParallelType.ddp)
    train_module_config = TransformerTrainModuleConfig(
        rank_microbatch_size=seq_len,
        max_sequence_length=seq_len,
        optim=AdamWConfig(),
        pp_config=pp_config,
        dp_config=dp_config,
    )

    model = transformer_config.build(init_device="meta")
    train_module = train_module_config.build(model, device=device)

    actual_flops = train_module.num_flops_per_token(seq_len)
    assert (
        actual_flops == expected_flops
    ), f"PP train module reported {actual_flops} FLOPs/token but full model has {expected_flops}"


def test_pp_num_flops_per_token():
    run_distributed_test(
        _run_pp_num_flops_per_token, world_size=2, backend="gloo", start_method="spawn"
    )
