"""Tests for evaluator scheduling and downstream batch sizing."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from olmo_core.data import TokenizerConfig
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.train.callbacks import evaluator_callback
from olmo_core.train.train_module import (
    EvalBatchSizeUnit,
    EvalBatchSpec,
    OLMoDDPTrainModule,
    TransformerPipelineTrainModule,
    TransformerTrainModule,
)


@pytest.mark.parametrize("module_type", [TransformerTrainModule, OLMoDDPTrainModule])
def test_post_attach_accepts_non_pipeline_transformer_modules(module_type):
    module = object.__new__(module_type)
    module._pp_config = None
    module._cp_config = None
    callback = evaluator_callback.EvaluatorCallback()
    callback.trainer = SimpleNamespace(train_module=module)

    callback.post_attach()


@pytest.mark.parametrize("pp_group_rank", [0, 1])
def test_post_attach_rejects_olmo_ddp_pipeline_parallelism_on_every_stage(pp_group_rank):
    module = object.__new__(OLMoDDPTrainModule)
    module._pp_config = SimpleNamespace(degree=2)
    module.pp_group_rank = pp_group_rank
    module.pp_final_stage_rank = 1
    callback = evaluator_callback.EvaluatorCallback()
    callback.trainer = SimpleNamespace(train_module=module)

    with pytest.raises(
        OLMoConfigurationError, match="does not support OLMoDDP pipeline parallelism"
    ):
        callback.post_attach()


def test_post_attach_rejects_olmo_ddp_context_parallelism():
    module = object.__new__(OLMoDDPTrainModule)
    module._pp_config = None
    module._cp_config = SimpleNamespace(degree=2)
    callback = evaluator_callback.EvaluatorCallback()
    callback.trainer = SimpleNamespace(train_module=module)

    with pytest.raises(
        OLMoConfigurationError, match="does not support OLMoDDP context parallelism"
    ):
        callback.post_attach()


@pytest.mark.parametrize("module_type", [object, TransformerPipelineTrainModule])
def test_post_attach_rejects_unsupported_modules(module_type):
    callback = evaluator_callback.EvaluatorCallback()
    callback.trainer = SimpleNamespace(train_module=object.__new__(module_type))

    with pytest.raises(OLMoConfigurationError, match="only supports transformer train modules"):
        callback.post_attach()


@pytest.mark.parametrize("interval,fixed_steps", [(500, None), (None, [500])])
def test_finish_reuses_successful_step_evaluation(interval, fixed_steps):
    callback = evaluator_callback.EvaluatorCallback(
        eval_interval=interval, fixed_steps=fixed_steps, eval_on_finish=True
    )
    callback.trainer = SimpleNamespace(global_step=500, dp_process_group=None, record_metric=Mock())
    callback.post_step()
    callback.trainer.record_metric.reset_mock()
    callback.post_train()
    callback.trainer.record_metric.assert_not_called()

    callback.trainer.global_step = 501
    callback.post_train()
    assert callback.trainer.record_metric.call_count == 2


def test_restart_or_checkpoint_load_repeats_endpoint_evaluation():
    callback = evaluator_callback.EvaluatorCallback(eval_on_finish=True)
    callback.trainer = SimpleNamespace(global_step=500, dp_process_group=None, record_metric=Mock())
    for reset in (callback.pre_train, lambda: callback.post_checkpoint_loaded("/checkpoint")):
        callback.post_train()
        reset()
        callback.trainer.record_metric.reset_mock()
        callback.post_train()
        assert callback.trainer.record_metric.call_count == 2


def test_failed_or_alternate_evaluation_does_not_suppress_finish():
    callback = evaluator_callback.EvaluatorCallback(eval_interval=500, eval_on_finish=True)
    recorder = Mock(side_effect=RuntimeError("metric failure"))
    callback.trainer = SimpleNamespace(
        global_step=500, dp_process_group=None, record_metric=recorder
    )
    with pytest.raises(RuntimeError, match="metric failure"):
        callback.post_step()
    recorder.side_effect = None
    callback.perform_eval(prefix="eval/alternate")
    recorder.reset_mock()
    callback.post_train()
    assert recorder.call_count == 2


def _build_downstream_callback(monkeypatch, rank_batch_size_instances):
    captured = []

    class FakeHFTokenizer:
        def __init__(self, *args, **kwargs):
            pass

    class FakeDownstreamEvaluator:
        def __init__(self, **kwargs):
            captured.append(kwargs)

    monkeypatch.setattr(evaluator_callback, "DownstreamEvaluator", FakeDownstreamEvaluator)
    monkeypatch.setattr(evaluator_callback, "_all_tasks", lambda: {"task-a", "task-b"})
    monkeypatch.setattr("olmo_eval.HFTokenizer", FakeHFTokenizer)

    native_spec = EvalBatchSpec(rank_batch_size=20_480, max_sequence_length=2_560)
    trainer = SimpleNamespace(
        train_module=SimpleNamespace(eval_batch_spec=native_spec),
        device=None,
        dp_process_group=None,
    )
    config = evaluator_callback.DownstreamEvaluatorCallbackConfig(
        tasks=["task-b", "task-a"],
        tokenizer=TokenizerConfig.dolma2(),
        rank_batch_size_instances=rank_batch_size_instances,
    )

    config.build(trainer)
    return native_spec, captured


def test_downstream_evaluator_instance_batch_override_reaches_every_task(monkeypatch):
    native_spec, captured = _build_downstream_callback(monkeypatch, 1)

    assert len(captured) == 2
    assert all(kwargs["batch_spec"].rank_batch_size == 1 for kwargs in captured)
    assert all(
        kwargs["batch_spec"].batch_size_unit == EvalBatchSizeUnit.instances for kwargs in captured
    )
    assert native_spec.rank_batch_size == 20_480
    assert native_spec.batch_size_unit == EvalBatchSizeUnit.tokens


def test_downstream_evaluator_default_preserves_train_module_batch_spec(monkeypatch):
    native_spec, captured = _build_downstream_callback(monkeypatch, None)

    assert len(captured) == 2
    assert all(kwargs["batch_spec"] is native_spec for kwargs in captured)
