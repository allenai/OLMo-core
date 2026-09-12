import pytest
import torch

from olmo_core.distributed.parallel import (
    DataParallelType,
    PipelineScheduleType,
    PipelineSplitStyle,
)
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.feed_forward import FeedForwardConfig
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.optim import AdamWConfig, OLMoDDPOptimizerConfig
from olmo_core.testing import run_distributed_test
from olmo_core.train.train_module.transformer import (
    OLMoDDPTrainModuleConfig,
    TransformerDataParallelConfig,
    TransformerPipelineParallelConfig,
    TransformerTrainModuleConfig,
)


def test_olmo_ddp_reduce_scatter_config_is_nested_under_data_parallel():
    default_dp = TransformerDataParallelConfig(name=DataParallelType.ddp)
    assert default_dp.use_reduce_scatter is False

    dp_config = TransformerDataParallelConfig(
        name=DataParallelType.ddp,
        use_reduce_scatter=True,
    )
    train_config = OLMoDDPTrainModuleConfig(
        rank_microbatch_size=16,
        max_sequence_length=16,
        optim=OLMoDDPOptimizerConfig(),
        dp_config=dp_config,
    )
    build_kwargs = train_config.as_dict(exclude_none=True, recurse=False)
    assert build_kwargs["dp_config"].use_reduce_scatter is True
    assert "reduce_scatter_grads" not in build_kwargs


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


@pytest.mark.parametrize(
    "schedule",
    [
        PipelineScheduleType.custom_interleaved_1F1B,
        PipelineScheduleType.custom_1F1B_V,
    ],
)
def test_custom_schedule_requires_custom_stage(schedule: PipelineScheduleType):
    # A custom schedule paired with torch's PipelineStage would fail during pre_train/step; the
    # config should reject it up front (the check runs before the model/mesh are touched).
    pp_config = TransformerPipelineParallelConfig(
        degree=2, schedule=schedule, use_custom_stage_implementation=False
    )
    with pytest.raises(OLMoConfigurationError):
        pp_config.split_model(None, pp_mesh=None, device=torch.device("cpu"))  # type: ignore[arg-type]


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


def _tiny_dense_train_module_config() -> "TransformerTrainModuleConfig":
    return TransformerTrainModuleConfig(
        rank_microbatch_size=128,
        max_sequence_length=128,
        optim=AdamWConfig(),
    )


def _tiny_dense_model():
    return TransformerConfig.llama_like(
        d_model=64,
        vocab_size=128,
        n_layers=2,
        n_heads=2,
        feed_forward=FeedForwardConfig(hidden_size=128, bias=False),
    ).build(init_device="cpu")


def test_dense_train_module_eval_only_skips_optimizer():
    tm = _tiny_dense_train_module_config().build(
        _tiny_dense_model(), device=torch.device("cpu"), eval_only=True
    )
    assert tm.eval_only is True
    assert tm.optim is None
    # No optimizer -> state dict must omit optim state, and train-only entry points are unavailable.
    assert "optim" not in tm.state_dict()
    assert "optim" not in tm.state_dict_to_save()
    with pytest.raises(AssertionError):
        tm.optim_step()
    with pytest.raises(AssertionError):
        tm.zero_grads()


def test_dense_train_module_builds_optimizer_by_default():
    tm = _tiny_dense_train_module_config().build(_tiny_dense_model(), device=torch.device("cpu"))
    assert tm.eval_only is False
    assert tm.optim is not None
    assert "optim" in tm.state_dict()
