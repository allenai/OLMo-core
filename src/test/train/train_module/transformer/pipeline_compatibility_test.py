import copy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.pipelining import PipelineStage
from torch.distributed.pipelining.schedules import (
    PipelineScheduleMulti,
    PipelineScheduleSingle,
    get_schedule_class,
)

from olmo_core.distributed.parallel import pipeline_parallel as pipeline
from olmo_core.distributed.parallel.pipeline_parallel import PipelineScheduleType
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.feed_forward import FeedForwardConfig
from olmo_core.nn.lm_head import LMOutputWithLoss
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.optim import OLMoDDPOptimizerConfig
from olmo_core.testing import run_distributed_test
from olmo_core.train.callbacks.hf_converter import HFConverterCallback
from olmo_core.train.train_module.transformer import config as module_config
from olmo_core.train.train_module.transformer import ddp_train_module
from olmo_core.train.train_module.transformer import (
    pipeline_train_module as train_module,
)
from olmo_core.train.train_module.transformer.pipeline import (
    pipeline_schedule as custom_schedules,
)

SMOKE_SCHEDULES = (
    PipelineScheduleType.single_1F1B,
    PipelineScheduleType.interleaved_1F1B,
    PipelineScheduleType.gpipe,
)
STANDARD_SCHEDULES = (
    *SMOKE_SCHEDULES,
    PipelineScheduleType.looped_bfs,
    PipelineScheduleType.interleaved_zero_bubble,
    PipelineScheduleType.zbv_zero_bubble,
)


@pytest.mark.parametrize("pp_group_size", [1, 2, 4])
@pytest.mark.parametrize("local_parts", [0, 1, 2])
def test_pipeline_model_accessor_requires_one_complete_model(pp_group_size, local_parts):
    module = object.__new__(train_module.TransformerPipelineTrainModule)
    module.pp_group_size = pp_group_size
    module.model_parts = [nn.Identity() for _ in range(local_parts)]

    if pp_group_size == 1 and local_parts == 1:
        assert module.model is module.model_parts[0]
    else:
        with pytest.raises(RuntimeError, match="requires a single pipeline rank and model part"):
            _ = module.model


@pytest.mark.parametrize("schedule_name", SMOKE_SCHEDULES)
def test_pipeline_hf_export_rejects_partial_model_before_state_dict_collectives(
    monkeypatch, schedule_name
):
    module = object.__new__(train_module.TransformerPipelineTrainModule)
    module.pp_group_size = 2
    module.model_parts = [nn.Identity() for _ in range(1 if schedule_name.is_single_stage else 2)]
    callback = HFConverterCallback()
    callback._trainer = SimpleNamespace(train_module=module)
    get_state_dict = Mock()
    monkeypatch.setattr(
        "olmo_core.train.callbacks.hf_converter.dist_cp_sd.get_model_state_dict", get_state_dict
    )

    with pytest.raises(RuntimeError, match="requires a single pipeline rank and model part"):
        callback._get_full_model_state_dict()
    get_state_dict.assert_not_called()


def test_pipeline_defaults_preserve_torch_interleaved_schedule():
    config = module_config.TransformerPipelineParallelConfig(degree=2)
    assert config.schedule == PipelineScheduleType.interleaved_1F1B
    assert config.use_custom_stage_implementation is False
    assert config.get_split_points(4) == [1, 2, 3]


@pytest.mark.parametrize("schedule_name", STANDARD_SCHEDULES)
@pytest.mark.parametrize("microbatches", [None, 8])
def test_standard_schedule_constructor_preserves_stage_and_loss_contract(
    monkeypatch, schedule_name, microbatches
):
    calls = []
    base = PipelineScheduleSingle if schedule_name.is_single_stage else PipelineScheduleMulti

    class RecordingSchedule(base):
        def __init__(self, stages, *, n_microbatches, loss_fn):
            calls.append((stages, n_microbatches, loss_fn))

        def _step_microbatches(self, *args, **kwargs):
            raise AssertionError("Constructor test must not execute a schedule")

    monkeypatch.setattr(pipeline, "get_schedule_class", lambda name: RecordingSchedule)
    monkeypatch.setattr(dist, "get_rank", lambda: 0)
    stages = [object()] if schedule_name.is_single_stage else [object(), object()]
    loss_fn = Mock()
    schedule = pipeline.PipelineSchedule(
        model_parts=[nn.Identity() for _ in stages],
        stages=stages,
        pp_mesh=SimpleNamespace(size=lambda: 2),
        schedule_name=schedule_name,
        loss_fn=loss_fn,
        num_microbatches=microbatches,
    )
    assert calls == [
        (stages[0] if schedule_name.is_single_stage else stages, microbatches or 2, loss_fn)
    ]
    assert isinstance(schedule.base_schedule, RecordingSchedule)
    assert schedule._is_custom_schedule is False


@pytest.mark.parametrize(
    "schedule_name", [PipelineScheduleType.single_1F1B, PipelineScheduleType.gpipe]
)
def test_single_stage_schedule_rejects_multiple_local_parts(monkeypatch, schedule_name):
    monkeypatch.setattr(dist, "get_rank", lambda: 0)
    with pytest.raises(OLMoConfigurationError):
        pipeline.PipelineSchedule(
            model_parts=[nn.Identity(), nn.Identity()],
            stages=[object(), object()],
            pp_mesh=SimpleNamespace(size=lambda: 2),
            schedule_name=schedule_name,
        )


@pytest.mark.parametrize(
    "last,with_loss", [(False, False), (False, True), (True, False), (True, True)]
)
def test_standard_step_preserves_output_and_stacked_loss_api(last, with_loss):
    schedule = pipeline.PipelineSchedule.__new__(pipeline.PipelineSchedule)
    schedule._is_custom_schedule = False
    schedule.stages = [SimpleNamespace(is_first=True, is_last=last)]
    schedule.loss_fn = Mock() if with_loss else None
    expected_output = object()
    seen = []

    def step(*args, target=None, losses=None, **kwargs):
        seen.append((args, target, losses, kwargs))
        if losses is not None:
            losses.extend([torch.tensor(2.0), torch.tensor(3.0)])
        return expected_output

    schedule.base_schedule = SimpleNamespace(step=step)
    schedule.schedule_impl = schedule.base_schedule
    inputs = torch.zeros((4, 8), dtype=torch.long)
    targets = torch.ones_like(inputs)
    output, losses = schedule.step(inputs, target=targets, labels=targets, marker=7)
    assert output is expected_output
    args, actual_target, _, kwargs = seen[0]
    assert args == (inputs,)
    assert kwargs == {"labels": targets, "marker": 7}
    if last and with_loss:
        assert actual_target is targets
        torch.testing.assert_close(losses, torch.tensor([2.0, 3.0]))
    else:
        assert actual_target is None
        assert losses is None


@pytest.mark.parametrize("argument_count", [0, 1, 2])
def test_standard_step_preserves_variadic_inputs(argument_count):
    schedule = pipeline.PipelineSchedule.__new__(pipeline.PipelineSchedule)
    schedule._is_custom_schedule = False
    schedule.stages = [SimpleNamespace(is_first=True, is_last=False)]
    schedule.loss_fn = None
    schedule.base_schedule = SimpleNamespace(step=Mock(return_value="output"))
    arguments = tuple(torch.ones((2, 3)) for _ in range(argument_count))
    assert schedule.step(*arguments, marker=7) == ("output", None)
    args, kwargs = schedule.base_schedule.step.call_args
    assert len(args) == argument_count
    assert all(actual is expected for actual, expected in zip(args, arguments))
    assert kwargs == {"target": None, "losses": None, "marker": 7}


@pytest.mark.parametrize("override", [{"forward_only": True}, {"num_microbatches": 2}])
def test_standard_schedule_rejects_custom_step_controls(override):
    schedule = pipeline.PipelineSchedule.__new__(pipeline.PipelineSchedule)
    schedule._is_custom_schedule = False
    schedule.num_microbatches = 4
    schedule.stages = [SimpleNamespace(is_first=True, is_last=True)]
    schedule.loss_fn = Mock()
    schedule.base_schedule = Mock()
    schedule.schedule_impl = schedule.base_schedule
    with pytest.raises((OLMoConfigurationError, NotImplementedError)):
        schedule.step(torch.zeros((4, 8), dtype=torch.long), **override)
    schedule.base_schedule.step.assert_not_called()


@pytest.mark.parametrize("schedule_name", STANDARD_SCHEDULES)
def test_standard_schedule_selects_torch_stage(monkeypatch, schedule_name):
    config = module_config.TransformerPipelineParallelConfig(degree=2, schedule=schedule_name)
    model = TransformerConfig.llama_like(
        d_model=16,
        vocab_size=32,
        n_layers=4,
        n_heads=2,
        feed_forward=FeedForwardConfig(hidden_size=32),
    ).build(init_device="meta")
    constructor = Mock(
        side_effect=lambda *args, **kwargs: SimpleNamespace(args=args, kwargs=kwargs)
    )
    monkeypatch.setattr(module_config, "PipelineStage", constructor)
    custom = Mock(side_effect=AssertionError("Selected custom stage for a Torch schedule"))
    monkeypatch.setattr(module_config, "CustomPipelineStage", custom)
    mesh = SimpleNamespace(get_local_rank=lambda: 0, get_group=lambda name: "pp-group")
    stages, parts = config.split_model(model, pp_mesh=mesh, device=torch.device("cpu"))
    assert len(stages) == len(parts) == (1 if schedule_name.is_single_stage else 2)
    assert constructor.call_count == len(stages)
    assert all(stage.kwargs["group"] == "pp-group" for stage in stages)
    custom.assert_not_called()


@pytest.mark.parametrize("schedule_name", STANDARD_SCHEDULES)
def test_standard_schedule_rejects_custom_stage(schedule_name):
    config = module_config.TransformerPipelineParallelConfig(
        degree=2, schedule=schedule_name, use_custom_stage_implementation=True
    )
    with pytest.raises(OLMoConfigurationError):
        config.split_model(None, pp_mesh=None, device=torch.device("cpu"))


def test_pipeline_pre_train_passes_loss_callback(monkeypatch):
    module = object.__new__(train_module.TransformerPipelineTrainModule)
    module._trainer = SimpleNamespace(global_batch_size=32, dp_process_group=None)
    module.rank_microbatch_size = 8
    module.model_parts = [nn.Identity()]
    module._pp_stages = [object()]
    module._train_pp_schedule = None
    module._pp_config = module_config.TransformerPipelineParallelConfig(degree=2)
    module.world_mesh = object()
    mesh = object()
    monkeypatch.setattr(train_module, "get_world_size", lambda group: 1)
    monkeypatch.setattr(train_module, "get_pp_mesh", lambda world_mesh: mesh)
    constructor = Mock()
    monkeypatch.setattr(train_module, "PipelineSchedule", constructor)
    module.pre_train()
    kwargs = constructor.call_args.kwargs
    assert kwargs["loss_fn"] == module.loss_fn
    assert kwargs["num_microbatches"] == 4
    assert kwargs["pp_mesh"] is mesh
    assert kwargs["schedule_name"] == PipelineScheduleType.interleaved_1F1B


@pytest.mark.parametrize(
    "global_batch_size,rank_microbatch_size,dp_world_size,expected_microbatches",
    [
        (64, 16, 1, 4),
        (128, 16, 2, 4),
        (16, 16, 1, 1),
        (56, 16, 1, None),
        (56, 16, 2, None),
        (65, 16, 2, None),
        (8, 16, 1, None),
        (16, 16, 2, None),
        (0, 16, 1, None),
        (-16, 16, 1, None),
        (64, 0, 1, None),
    ],
)
def test_native_pipeline_validates_batch_size_before_schedule_setup(
    monkeypatch, global_batch_size, rank_microbatch_size, dp_world_size, expected_microbatches
):
    module = object.__new__(ddp_train_module.OLMoDDPTrainModule)
    module._trainer = SimpleNamespace(global_batch_size=global_batch_size, dp_process_group=None)
    module.rank_microbatch_size = rank_microbatch_size
    module._pp_config = module_config.TransformerPipelineParallelConfig(
        degree=2,
        schedule=PipelineScheduleType.custom_interleaved_1F1B,
        use_custom_stage_implementation=True,
    )
    module.model_parts = []
    module._pp_stages = []
    module._train_pp_schedule = None
    module.world_mesh = {"dense": {"pp": object()}}
    prewarm = Mock()
    monkeypatch.setattr(module, "_rowwise_lifetime_lease_slots_env_is_set", lambda: False)
    monkeypatch.setattr(module, "_prewarm_ep_no_sync_symm_buffers", prewarm)
    monkeypatch.setattr(
        module, "_estimate_pp_rowwise_lifetime_lease_slots_for_model_parts", lambda: 1
    )
    monkeypatch.setattr(ddp_train_module, "get_world_size", lambda group: dp_world_size)
    constructor = Mock()
    monkeypatch.setattr(ddp_train_module, "PipelineSchedule", constructor)

    if expected_microbatches is None:
        with pytest.raises(OLMoConfigurationError, match="batch size"):
            module.on_attach()
        constructor.assert_not_called()
        prewarm.assert_not_called()
        assert module._train_pp_schedule is None
    else:
        module.on_attach()
        assert constructor.call_args.kwargs["num_microbatches"] == expected_microbatches
        prewarm.assert_called_once()


@pytest.mark.parametrize(
    "schedule_name,class_name",
    [
        (PipelineScheduleType.custom_interleaved_1F1B, "CustomScheduleInterleaved1F1B"),
        (PipelineScheduleType.custom_1F1B_V, "CustomSchedule1F1BV"),
    ],
)
def test_custom_schedule_constructor_retains_execution_options(
    monkeypatch, schedule_name, class_name
):
    implementation = object()
    constructor = Mock(return_value=implementation)
    monkeypatch.setattr(custom_schedules, class_name, constructor)
    monkeypatch.setattr(dist, "get_rank", lambda: 0)
    stages = [object(), object()]
    schedule = pipeline.PipelineSchedule(
        model_parts=[nn.Identity(), nn.Identity()],
        stages=stages,
        pp_mesh=SimpleNamespace(size=lambda: 2),
        schedule_name=schedule_name,
        num_microbatches=4,
        forward_pull_ahead_extra_activations=[1, 2],
    )
    assert schedule._is_custom_schedule is True
    assert schedule.schedule_impl is implementation
    constructor.assert_called_once_with(
        stages, n_microbatches=4, forward_pull_ahead_extra_activations=[1, 2]
    )


@pytest.mark.parametrize("custom", [False, True])
def test_pipeline_loss_hook_preserves_selected_stage_output_contract(custom):
    class FinalStage(nn.Module):
        def __init__(self):
            super().__init__()
            self.lm_head = nn.Identity()
            self.weight = nn.Parameter(torch.tensor(2.0))

        def forward(self, inputs, **kwargs):
            loss = self.weight * inputs.float().sum()
            return LMOutputWithLoss(logits=None, loss=loss, ce_loss=loss * 0.75, z_loss=loss * 0.25)

    model = FinalStage()
    module = object.__new__(train_module.TransformerPipelineTrainModule)
    module._trainer = object()
    module._pp_config = SimpleNamespace(use_custom_stage_implementation=custom)
    module.model_parts = [model]
    module.z_loss_multiplier = 0.25
    module.autocast_precision = None

    def step(inputs, **kwargs):
        output = model(inputs, **kwargs)
        if custom:
            assert isinstance(output, LMOutputWithLoss)
            output.loss.backward()
        else:
            assert isinstance(output, torch.Tensor)
            assert output.shape == (1,)
            output.backward()

    module._train_pp_schedule = SimpleNamespace(step=step)
    inputs = torch.ones((2, 4), dtype=torch.long)
    ce_loss, z_loss = module.run_pipeline(inputs, inputs, 8)
    torch.testing.assert_close(ce_loss, torch.tensor(12.0))
    torch.testing.assert_close(z_loss, torch.tensor(4.0))
    torch.testing.assert_close(model.weight.grad, torch.tensor(8.0))
    assert not model._forward_hooks


@pytest.mark.parametrize("schedule_name", STANDARD_SCHEDULES)
def test_native_ddp_rejects_standard_pipeline_before_allocating_runtime(schedule_name):
    model = nn.Identity()
    model._olmo_ddp_compatible = True
    with pytest.raises(OLMoConfigurationError, match="pipeline execution requires"):
        ddp_train_module.OLMoDDPTrainModule(
            model=model,
            optim=OLMoDDPOptimizerConfig(),
            rank_microbatch_size=8,
            max_sequence_length=8,
            pp_config=module_config.TransformerPipelineParallelConfig(
                degree=2, schedule=schedule_name
            ),
            device=torch.device("cpu"),
        )


def _run_standard_pipeline_schedules():
    torch.set_num_threads(1)
    rank = dist.get_rank()
    mesh = init_device_mesh("cpu", (2,), mesh_dim_names=("pp",))
    inputs = torch.arange(32, dtype=torch.float32).reshape(8, 4) / 32
    target = torch.zeros_like(inputs)
    for schedule_name in SMOKE_SCHEDULES:
        torch.manual_seed(123)
        count = 2 if schedule_name.is_single_stage else 4
        modules = [nn.Linear(4, 4, bias=False) for _ in range(count)]
        reference = nn.Sequential(*copy.deepcopy(modules))
        local_indices = [rank] if count == 2 else [rank, rank + 2]
        local_parts = [modules[index] for index in local_indices]
        stages = [
            PipelineStage(
                module,
                index,
                count,
                torch.device("cpu"),
                input_args=torch.zeros((2, 4)),
                output_args=torch.zeros((2, 4)),
                group=mesh.get_group(),
            )
            for index, module in zip(local_indices, local_parts)
        ]
        schedule = pipeline.PipelineSchedule(
            model_parts=local_parts,
            stages=stages,
            pp_mesh=mesh,
            schedule_name=schedule_name,
            loss_fn=nn.functional.mse_loss,
            num_microbatches=4,
        )
        assert isinstance(schedule.base_schedule, get_schedule_class(schedule_name))
        output, losses = schedule.step(inputs, target=target)
        expected_output = reference(inputs)
        expected_loss = nn.functional.mse_loss(expected_output, target)
        expected_loss.backward()
        if rank == 1:
            torch.testing.assert_close(output, expected_output)
            torch.testing.assert_close(losses.mean(), expected_loss)
        else:
            assert losses is None
        for index, local in zip(local_indices, local_parts):
            assert local.weight.grad is not None
            assert torch.isfinite(local.weight.grad).all()
            torch.testing.assert_close(local.weight.grad, reference[index].weight.grad)
        dist.barrier()


def test_standard_pipeline_schedules_train_on_two_cpu_ranks():
    run_distributed_test(
        _run_standard_pipeline_schedules,
        world_size=2,
        backend="gloo",
        start_method="spawn",
    )
