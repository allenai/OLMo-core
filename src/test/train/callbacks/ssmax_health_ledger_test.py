from __future__ import annotations

import copy
import hashlib
import json
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


def _callback(
    *, rolling_interval_length: int = 128
) -> tuple[SSMaxHealthLedgerCallback, SimpleNamespace]:
    callback = SSMaxHealthLedgerCallback(
        model_variant="ssmax_head_qknorm",
        phase="perception",
        run_name="pair-treatment",
    )
    trainer = SimpleNamespace(
        global_step=0,
        data_loader=_Loader(),
        train_module=SimpleNamespace(
            optim=SimpleNamespace(rolling_interval_length=rolling_interval_length)
        ),
    )
    callback.trainer = trainer
    return callback, trainer


def _metrics(*, skipped: float = 0.0, guard_active: bool = False):
    return {
        "train/CE loss": 2.0,
        "optim/total grad norm": 1.25,
        "optim/step skipped": skipped,
        "optim/guard active": float(guard_active),
        "optim/guard loss within": 1.0,
        "optim/guard gradient within": float(not skipped),
    }


def _rehash_ledger(state):
    payload = {name: value for name, value in state.items() if name != "content_sha256"}
    state["content_sha256"] = hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    return state


def _rehash_event_chain(state):
    previous_sha = "0" * 64
    for event in state["events"]:
        event["previous_event_sha256"] = previous_sha
        event["event_sha256"] = hashlib.sha256(
            json.dumps(
                {name: value for name, value in event.items() if name != "event_sha256"},
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
        previous_sha = event["event_sha256"]
    state["event_chain_sha256"] = previous_sha
    return _rehash_ledger(state)


def _trainer_state(
    *,
    rank: int,
    step: int = 2,
    skipped: bool = False,
    data_errors: int = 0,
):
    del rank  # Rank is represented by sequence position in the trainer-state inventory.
    callback, trainer = _callback(rolling_interval_length=2)
    trainer.data_loader.data_errors = data_errors
    for current_step in range(1, step + 1):
        trainer.global_step = current_step
        callback.log_metrics(
            current_step,
            _metrics(
                skipped=float(skipped and current_step == step),
                guard_active=current_step >= 2,
            ),
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
    callback, trainer = _callback(rolling_interval_length=2)
    trainer.global_step = 1
    callback.log_metrics(1, _metrics())
    trainer.global_step = 2
    callback.log_metrics(2, _metrics(skipped=1.0, guard_active=True))
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
    assert validated["version"] == 3
    assert validated["optimizer_guard_history_reset_steps"] == []
    assert validated["optimizer_guard_rolling_interval_length"] == 2

    resumed, resumed_trainer = _callback(rolling_interval_length=2)
    resumed_trainer.global_step = 2
    resumed.load_state_dict(state)
    resumed.post_checkpoint_loaded("step2")
    resumed.pre_train()
    resumed.log_metrics(2, {"checkpoint/load_duration_s": 1.0})
    resumed.log_metrics(2, _metrics())
    resumed_trainer.global_step = 3
    resumed.log_metrics(3, _metrics())
    resumed_state = resumed.state_dict()
    assert resumed_state["last_step"] == 3
    assert resumed_state["optimizer_guard_history_reset_steps"] == [2]
    with pytest.raises(RuntimeError, match="expected step 4"):
        resumed.log_metrics(3, _metrics())


def test_ledger_v3_boundaries_fail_closed_and_legacy_v2_remains_readable() -> None:
    callback, trainer = _callback()
    trainer.global_step = 1
    callback.log_metrics(1, _metrics())
    state = callback.state_dict()

    legacy = copy.deepcopy(state)
    legacy["version"] = 2
    legacy.pop("optimizer_guard_history_reset_steps")
    legacy.pop("optimizer_guard_rolling_interval_length")
    legacy["metrics"] = {
        "loss": "train/CE loss",
        "grad_norm": "optim/total grad norm",
        "optimizer_guard_skip": "optim/step skipped",
    }
    previous_sha = "0" * 64
    for event in legacy["events"]:
        event.pop("optimizer_guard_active")
        event.pop("optimizer_guard_loss_within")
        event.pop("optimizer_guard_gradient_within")
        event["previous_event_sha256"] = previous_sha
        event["event_sha256"] = hashlib.sha256(
            json.dumps(
                {name: value for name, value in event.items() if name != "event_sha256"},
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
        previous_sha = event["event_sha256"]
    legacy["event_chain_sha256"] = previous_sha
    _rehash_ledger(legacy)
    validated = validate_ssmax_health_ledger_state(
        legacy,
        expected_model_variant="ssmax_head_qknorm",
        expected_phase="perception",
        expected_run_name="pair-treatment",
        expected_step=1,
    )
    assert validated["version"] == 2

    resumed_legacy, resumed_legacy_trainer = _callback()
    resumed_legacy_trainer.global_step = 1
    resumed_legacy.load_state_dict(legacy)
    resumed_legacy.post_checkpoint_loaded("legacy-step1")
    resumed_legacy_trainer.global_step = 2
    resumed_legacy.log_metrics(2, _metrics())
    continued_legacy = resumed_legacy.state_dict()
    assert continued_legacy["version"] == 2
    assert "optimizer_guard_history_reset_steps" not in continued_legacy
    assert "optimizer_guard_active" not in continued_legacy["events"][1]

    wrong_version_type = copy.deepcopy(state)
    wrong_version_type["version"] = 3.0
    with pytest.raises(SSMaxHealthLedgerError, match="format/version differs"):
        validate_ssmax_health_ledger_state(
            wrong_version_type,
            expected_model_variant="ssmax_head_qknorm",
            expected_phase="perception",
            expected_run_name="pair-treatment",
            expected_step=1,
        )

    for boundaries in ([0, 0], [1, 0], [0, 2]):
        malformed = copy.deepcopy(state)
        malformed["optimizer_guard_history_reset_steps"] = boundaries
        _rehash_ledger(malformed)
        with pytest.raises(SSMaxHealthLedgerError, match="not strictly increasing"):
            validate_ssmax_health_ledger_state(
                malformed,
                expected_model_variant="ssmax_head_qknorm",
                expected_phase="perception",
                expected_run_name="pair-treatment",
                expected_step=1,
            )

    malformed = copy.deepcopy(state)
    malformed["optimizer_guard_history_reset_steps"] = [False]
    _rehash_ledger(malformed)
    with pytest.raises(SSMaxHealthLedgerError, match="non-negative integer"):
        validate_ssmax_health_ledger_state(
            malformed,
            expected_model_variant="ssmax_head_qknorm",
            expected_phase="perception",
            expected_run_name="pair-treatment",
            expected_step=1,
        )

    fresh, _fresh_trainer = _callback()
    fresh.post_checkpoint_loaded("parent")
    assert fresh.state_dict()["optimizer_guard_history_reset_steps"] == [0]
    fresh.post_checkpoint_loaded("same-parent-again")
    assert fresh.state_dict()["optimizer_guard_history_reset_steps"] == [0]


def test_ledger_ignores_only_run_segment_baseline_metrics() -> None:
    callback, trainer = _callback()
    callback.pre_train()
    trainer.global_step = 1

    callback.log_metrics(0, {"checkpoint/save_duration_s": 1.0})
    callback.log_metrics(0, _metrics())
    callback.log_metrics(1, _metrics())

    state = callback.state_dict()
    assert state["last_step"] == 1
    assert [event["global_step"] for event in state["events"]] == [1]


def test_ledger_ignores_ancillary_batches_without_weakening_health_contract() -> None:
    callback, trainer = _callback()
    callback.pre_train()
    trainer.global_step = 1

    callback.log_metrics(1, {"gpu_memory/active": 1.0})
    callback.log_metrics(1, _metrics())
    callback.log_metrics(1, {"checkpoint/save_duration_s": 1.0})

    state = callback.state_dict()
    assert state["last_step"] == 1
    assert [event["global_step"] for event in state["events"]] == [1]
    with pytest.raises(RuntimeError, match="expected step 2"):
        callback.log_metrics(1, _metrics())
    with pytest.raises(RuntimeError, match="missing"):
        callback.log_metrics(2, {"train/CE loss": 1.0})

    trainer.global_step = 2
    callback.log_metrics(2, {"checkpoint/save_duration_s": 1.0})
    with pytest.raises(RuntimeError, match="has 1 events at trainer step 2"):
        callback.state_dict()


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


def test_ledger_v3_binds_live_guard_activation_and_skip_reason() -> None:
    callback, trainer = _callback(rolling_interval_length=2)
    trainer.global_step = 1
    callback.log_metrics(1, _metrics())
    trainer.global_step = 2
    callback.log_metrics(2, _metrics(guard_active=True))
    state = callback.state_dict()

    changed = copy.deepcopy(state)
    changed["events"][0]["optimizer_guard_active"] = True
    _rehash_event_chain(changed)
    with pytest.raises(SSMaxHealthLedgerError, match="activation differs"):
        validate_ssmax_health_ledger_state(
            changed,
            expected_model_variant="ssmax_head_qknorm",
            expected_phase="perception",
            expected_run_name="pair-treatment",
            expected_step=2,
        )

    changed = copy.deepcopy(state)
    changed["events"][1]["optimizer_guard_gradient_within"] = False
    _rehash_event_chain(changed)
    with pytest.raises(SSMaxHealthLedgerError, match="skip differs"):
        validate_ssmax_health_ledger_state(
            changed,
            expected_model_variant="ssmax_head_qknorm",
            expected_phase="perception",
            expected_run_name="pair-treatment",
            expected_step=2,
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

    divergent_boundaries = copy.deepcopy(trainer_states)
    divergent_boundaries[1]["callbacks"]["ssmax_health_ledger"][
        "optimizer_guard_history_reset_steps"
    ] = [0]
    _rehash_ledger(divergent_boundaries[1]["callbacks"]["ssmax_health_ledger"])
    with pytest.raises(SSMaxHealthLedgerError, match="reset steps differ across ranks"):
        extract_ssmax_health_ledgers(
            divergent_boundaries,
            expected_model_variant="ssmax_head_qknorm",
            expected_phase="perception",
            expected_run_name="pair-treatment",
            expected_step=2,
            expected_world_size=2,
        )

    divergent_intervals = copy.deepcopy(trainer_states)
    divergent_intervals[1]["callbacks"]["ssmax_health_ledger"][
        "optimizer_guard_rolling_interval_length"
    ] = 3
    _rehash_ledger(divergent_intervals[1]["callbacks"]["ssmax_health_ledger"])
    with pytest.raises(SSMaxHealthLedgerError, match="rolling intervals differ across ranks"):
        extract_ssmax_health_ledgers(
            divergent_intervals,
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
        train_module = SimpleNamespace(optim=SimpleNamespace(rolling_interval_length=128))
        data_loader = _Loader()
        checkpointer = Checkpointer()
        callbacks = {"ssmax_health_ledger": callback}

        def _log_metrics(self):
            order.append("log_metrics")
            callback.log_metrics(0, {"checkpoint/save_duration_s": 1.0})
            callback.log_metrics(0, _metrics())
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
    trainer.global_step = 0
    callback.pre_train()
    trainer.global_step = 1
    Trainer.save_checkpoint(trainer)
    assert order == ["log_metrics", "join", "state_dict", "save"]
