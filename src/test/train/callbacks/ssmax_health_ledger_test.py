from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest

from olmo_core.train.callbacks import (
    SSMaxHealthLedgerCallback,
    SSMaxHealthLedgerError,
    extract_ssmax_health_ledgers,
    validate_ssmax_health_ledger_state,
)
from olmo_core.train.trainer import Trainer


class _Loader:
    def __init__(self, data_errors: int = 0) -> None:
        self.data_errors = data_errors

    def state_dict(self):
        return {"total_data_errors": self.data_errors}


def _callback() -> tuple[SSMaxHealthLedgerCallback, SimpleNamespace]:
    callback = SSMaxHealthLedgerCallback(
        model_variant="ssmax_head_qknorm",
        phase="perception",
        run_name="pair-treatment",
    )
    trainer = SimpleNamespace(global_step=0, data_loader=_Loader())
    callback.trainer = trainer
    return callback, trainer


def _metrics(*, skipped: float = 0.0):
    return {
        "train/CE loss": 2.0,
        "optim/total grad norm": 1.25,
        "optim/step skipped": skipped,
    }


def _trainer_state(
    *,
    rank: int,
    step: int = 2,
    skipped: bool = False,
    data_errors: int = 0,
):
    del rank  # Rank is represented by sequence position in the trainer-state inventory.
    callback, trainer = _callback()
    trainer.data_loader.data_errors = data_errors
    for current_step in range(1, step + 1):
        trainer.global_step = current_step
        callback.log_metrics(
            current_step,
            _metrics(skipped=float(skipped and current_step == step)),
        )
    return {
        "global_step": step,
        "world_size": 2,
        "data_loader": {
            "batches_processed": step,
            "total_data_errors": data_errors,
        },
        "callbacks": {"ssmax_health_ledger": callback.state_dict()},
    }


def test_ledger_is_contiguous_checkpoint_bound_and_resume_safe() -> None:
    callback, trainer = _callback()
    trainer.global_step = 1
    callback.log_metrics(1, _metrics())
    trainer.global_step = 2
    callback.log_metrics(2, _metrics(skipped=1.0))
    state = callback.state_dict()

    validated = validate_ssmax_health_ledger_state(
        state,
        expected_model_variant="ssmax_head_qknorm",
        expected_phase="perception",
        expected_run_name="pair-treatment",
        expected_step=2,
        expected_data_errors=0,
    )
    assert validated["optimizer_guard_skips"] == 1
    assert validated["nonfinite_losses"] == 0
    assert validated["nonfinite_gradients"] == 0
    assert [event["loss"] for event in validated["events"]] == [2.0, 2.0]
    assert [event["grad_norm"] for event in validated["events"]] == [1.25, 1.25]

    resumed, resumed_trainer = _callback()
    resumed_trainer.global_step = 2
    resumed.load_state_dict(state)
    resumed.post_checkpoint_loaded("step2")
    resumed_trainer.global_step = 3
    resumed.log_metrics(3, _metrics())
    assert resumed.state_dict()["last_step"] == 3


def test_ledger_rejects_missing_metrics_gaps_and_tampering() -> None:
    callback, trainer = _callback()
    trainer.global_step = 1
    with pytest.raises(RuntimeError, match="missing"):
        callback.log_metrics(1, {"train/CE loss": 2.0})
    with pytest.raises(RuntimeError, match="expected step 1"):
        callback.log_metrics(2, _metrics())

    callback.log_metrics(1, _metrics())
    state = callback.state_dict()
    changed = copy.deepcopy(state)
    changed["events"][0]["optimizer_guard_skipped"] = True
    with pytest.raises(SSMaxHealthLedgerError, match="event SHA-256"):
        validate_ssmax_health_ledger_state(
            changed,
            expected_model_variant="ssmax_head_qknorm",
            expected_phase="perception",
            expected_run_name="pair-treatment",
            expected_step=1,
        )

    changed = copy.deepcopy(state)
    changed["events"][0]["loss"] = None
    changed["events"][0]["event_sha256"] = callback._events[0]["event_sha256"]
    with pytest.raises(SSMaxHealthLedgerError, match="finite loss must be numeric"):
        validate_ssmax_health_ledger_state(
            changed,
            expected_model_variant="ssmax_head_qknorm",
            expected_phase="perception",
            expected_run_name="pair-treatment",
            expected_step=1,
        )


def test_extract_ledgers_recomputes_rank_health_and_rejects_divergence() -> None:
    trainer_states = [
        _trainer_state(rank=0, skipped=True, data_errors=2),
        _trainer_state(rank=1, skipped=True, data_errors=2),
    ]
    summary = extract_ssmax_health_ledgers(
        trainer_states,
        expected_model_variant="ssmax_head_qknorm",
        expected_phase="perception",
        expected_run_name="pair-treatment",
        expected_step=2,
        expected_world_size=2,
    )
    assert summary["counters"] == {
        "data_errors": 4,
        "optimizer_guard_skips": 1,
        "nonfinite_losses": 0,
        "nonfinite_gradients": 0,
    }

    divergent = copy.deepcopy(trainer_states)
    divergent[1] = _trainer_state(rank=1, skipped=False, data_errors=2)
    with pytest.raises(SSMaxHealthLedgerError, match="event chains differ"):
        extract_ssmax_health_ledgers(
            divergent,
            expected_model_variant="ssmax_head_qknorm",
            expected_phase="perception",
            expected_run_name="pair-treatment",
            expected_step=2,
            expected_world_size=2,
        )

    mismatched_loader = copy.deepcopy(trainer_states)
    mismatched_loader[0]["data_loader"]["total_data_errors"] = 3
    with pytest.raises(SSMaxHealthLedgerError, match="data-error count differs"):
        extract_ssmax_health_ledgers(
            mismatched_loader,
            expected_model_variant="ssmax_head_qknorm",
            expected_phase="perception",
            expected_run_name="pair-treatment",
            expected_step=2,
            expected_world_size=2,
        )


def test_trainer_checkpoint_flushes_ledger_before_serializing(tmp_path) -> None:
    callback, _ = _callback()
    order = []

    class Checkpointer:
        @staticmethod
        def checkpoint_dirname(step):
            return f"step{step}"

        @staticmethod
        def save(path, train_module, state, *, ephemeral):
            del path, train_module, ephemeral
            order.append("save")
            assert state["callbacks"]["ssmax_health_ledger"]["last_step"] == 1

    class FakeTrainer:
        global_step = 1
        global_train_tokens_seen = 8
        global_train_petaflops = 0.0
        max_steps = 1
        epoch = 0
        save_folder = str(tmp_path)
        train_module = object()
        data_loader = _Loader()
        checkpointer = Checkpointer()
        callbacks = {"ssmax_health_ledger": callback}

        def _log_metrics(self):
            order.append("log_metrics")
            callback.log_metrics(1, _metrics())

        def _join_bookkeeping_ops(self):
            order.append("join")

        def state_dict(self):
            order.append("state_dict")
            return {"callbacks": {name: item.state_dict() for name, item in self.callbacks.items()}}

        def record_metric(self, *args, **kwargs):
            del args, kwargs

        def _iter_callbacks(self):
            return iter(self.callbacks.values())

    trainer = FakeTrainer()
    callback.trainer = trainer
    Trainer.save_checkpoint(trainer)
    assert order == ["log_metrics", "join", "state_dict", "save"]
