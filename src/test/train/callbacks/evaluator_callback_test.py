"""Tests for downstream evaluator callback batch sizing."""

from types import SimpleNamespace

from olmo_core.data import TokenizerConfig
from olmo_core.train.callbacks import evaluator_callback
from olmo_core.train.train_module import EvalBatchSizeUnit, EvalBatchSpec


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
