"""Checkpoint-native, resume-safe health ledger for SSMax training phases."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, Mapping, Sequence

from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.train.common import OPTIM_GRAD_NORM_METRIC, TRAIN_CE_LOSS_METRIC

from .callback import Callback

SSMAX_HEALTH_LEDGER_FORMAT = "ssmax_training_health_ledger"
SSMAX_HEALTH_LEDGER_VERSION = 2
SSMAX_MODEL_VARIANTS = frozenset({"ssmax_head_qknorm", "ssmax_no_qknorm"})
SSMAX_PHASES = frozenset({"bridge", "perception", "joint"})
OPTIM_STEP_SKIPPED_METRIC = "optim/step skipped"
_TRAIN_HEALTH_METRICS = frozenset(
    {
        TRAIN_CE_LOSS_METRIC,
        OPTIM_GRAD_NORM_METRIC,
        OPTIM_STEP_SKIPPED_METRIC,
    }
)

_EVENT_FIELDS = frozenset(
    {
        "global_step",
        "loss",
        "grad_norm",
        "loss_finite",
        "gradients_finite",
        "optimizer_guard_skipped",
        "previous_event_sha256",
        "event_sha256",
    }
)
_STATE_FIELDS = frozenset(
    {
        "format",
        "version",
        "model_variant",
        "phase",
        "run_name",
        "metrics",
        "last_step",
        "events",
        "optimizer_guard_skips",
        "nonfinite_losses",
        "nonfinite_gradients",
        "data_errors",
        "event_chain_sha256",
        "content_sha256",
    }
)
_ZERO_SHA256 = "0" * 64


class SSMaxHealthLedgerError(ValueError):
    """Raised when a checkpoint-native SSMax health ledger is malformed."""


def _canonical_sha256(value: Any) -> str:
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise SSMaxHealthLedgerError("SSMax health ledger is not finite JSON") from error
    return hashlib.sha256(encoded).hexdigest()


def _nonnegative_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SSMaxHealthLedgerError(f"{name} must be a non-negative integer")
    return value


def _validate_identity(model_variant: str, phase: str, run_name: str) -> None:
    if model_variant not in SSMAX_MODEL_VARIANTS:
        raise SSMaxHealthLedgerError(f"Unsupported SSMax model variant {model_variant!r}")
    if phase not in SSMAX_PHASES:
        raise SSMaxHealthLedgerError(f"Unsupported SSMax phase {phase!r}")
    if not isinstance(run_name, str) or not run_name:
        raise SSMaxHealthLedgerError("SSMax health ledger run name must be non-empty")


def validate_ssmax_health_ledger_state(
    value: Any,
    *,
    expected_model_variant: str,
    expected_phase: str,
    expected_run_name: str,
    expected_step: int,
    expected_data_errors: int | None = None,
) -> Mapping[str, Any]:
    """Validate and recompute a checkpoint-native SSMax health ledger.

    The ledger is stored inside every trainer-rank state, so its raw bytes are covered by the
    checkpoint trainer-state inventory. Every event and cumulative counter is recomputed here.
    """

    if not isinstance(value, Mapping) or set(value) != _STATE_FIELDS:
        raise SSMaxHealthLedgerError("SSMax health ledger state fields differ")
    if (
        value["format"] != SSMAX_HEALTH_LEDGER_FORMAT
        or value["version"] != SSMAX_HEALTH_LEDGER_VERSION
    ):
        raise SSMaxHealthLedgerError("SSMax health ledger format/version differs")
    _validate_identity(expected_model_variant, expected_phase, expected_run_name)
    for name, expected in (
        ("model_variant", expected_model_variant),
        ("phase", expected_phase),
        ("run_name", expected_run_name),
        ("last_step", expected_step),
    ):
        if value[name] != expected:
            raise SSMaxHealthLedgerError(f"SSMax health ledger {name} differs")
    expected_metrics = {
        "loss": TRAIN_CE_LOSS_METRIC,
        "grad_norm": OPTIM_GRAD_NORM_METRIC,
        "optimizer_guard_skip": OPTIM_STEP_SKIPPED_METRIC,
    }
    if value["metrics"] != expected_metrics:
        raise SSMaxHealthLedgerError("SSMax health ledger metric contract differs")
    events = value["events"]
    if not isinstance(events, list) or len(events) != expected_step:
        raise SSMaxHealthLedgerError("SSMax health ledger event count differs from global step")
    previous_sha = _ZERO_SHA256
    optimizer_skips = 0
    nonfinite_losses = 0
    nonfinite_gradients = 0
    for step, raw_event in enumerate(events, start=1):
        if not isinstance(raw_event, Mapping) or set(raw_event) != _EVENT_FIELDS:
            raise SSMaxHealthLedgerError(f"SSMax health ledger step{step} event fields differ")
        event = dict(raw_event)
        if event["global_step"] != step or event["previous_event_sha256"] != previous_sha:
            raise SSMaxHealthLedgerError("SSMax health ledger event chain is not contiguous")
        for name in ("loss_finite", "gradients_finite", "optimizer_guard_skipped"):
            if type(event[name]) is not bool:
                raise SSMaxHealthLedgerError(f"SSMax health ledger {name} must be boolean")
        for value_name, finite_name in (
            ("loss", "loss_finite"),
            ("grad_norm", "gradients_finite"),
        ):
            metric = event[value_name]
            if event[finite_name]:
                if (
                    isinstance(metric, bool)
                    or not isinstance(metric, (int, float))
                    or not math.isfinite(float(metric))
                ):
                    raise SSMaxHealthLedgerError(
                        f"SSMax health ledger finite {value_name} must be numeric"
                    )
                if value_name == "grad_norm" and float(metric) < 0:
                    raise SSMaxHealthLedgerError(
                        "SSMax health ledger finite grad_norm must be non-negative"
                    )
            elif metric is not None:
                raise SSMaxHealthLedgerError(
                    f"SSMax health ledger non-finite {value_name} must be encoded as null"
                )
        event_sha = event.pop("event_sha256")
        if event_sha != _canonical_sha256(event):
            raise SSMaxHealthLedgerError("SSMax health ledger event SHA-256 differs")
        previous_sha = event_sha
        optimizer_skips += int(raw_event["optimizer_guard_skipped"])
        nonfinite_losses += int(not raw_event["loss_finite"])
        nonfinite_gradients += int(not raw_event["gradients_finite"])
    data_errors = _nonnegative_int(value["data_errors"], name="SSMax health data errors")
    if expected_data_errors is not None and data_errors != expected_data_errors:
        raise SSMaxHealthLedgerError("SSMax health ledger data-error count differs")
    for name, expected in (
        ("optimizer_guard_skips", optimizer_skips),
        ("nonfinite_losses", nonfinite_losses),
        ("nonfinite_gradients", nonfinite_gradients),
        ("event_chain_sha256", previous_sha),
    ):
        if value[name] != expected:
            raise SSMaxHealthLedgerError(f"SSMax health ledger {name} differs")
    content_sha = value["content_sha256"]
    if content_sha != _canonical_sha256(
        {name: item for name, item in value.items() if name != "content_sha256"}
    ):
        raise SSMaxHealthLedgerError("SSMax health ledger content SHA-256 differs")
    return value


def extract_ssmax_health_ledgers(
    trainer_states: Sequence[Mapping[str, Any]],
    *,
    expected_model_variant: str,
    expected_phase: str,
    expected_run_name: str,
    expected_step: int,
    expected_world_size: int,
) -> dict[str, Any]:
    """Extract and validate one checkpoint-bound ledger from every trainer rank state."""

    if len(trainer_states) != expected_world_size or expected_world_size <= 0:
        raise SSMaxHealthLedgerError("SSMax health ledger trainer-rank count differs")
    ledgers: list[Mapping[str, Any]] = []
    for rank, trainer_state in enumerate(trainer_states):
        if (
            not isinstance(trainer_state, Mapping)
            or trainer_state.get("global_step") != expected_step
            or trainer_state.get("world_size") != expected_world_size
        ):
            raise SSMaxHealthLedgerError(f"SSMax health trainer rank{rank} identity differs")
        loader = trainer_state.get("data_loader")
        callbacks = trainer_state.get("callbacks")
        if not isinstance(loader, Mapping) or not isinstance(callbacks, Mapping):
            raise SSMaxHealthLedgerError(f"SSMax health trainer rank{rank} state is incomplete")
        if loader.get("batches_processed") != expected_step:
            raise SSMaxHealthLedgerError(f"SSMax health trainer rank{rank} cursor differs")
        data_errors = loader.get("total_data_errors", 0)
        ledger = validate_ssmax_health_ledger_state(
            callbacks.get("ssmax_health_ledger"),
            expected_model_variant=expected_model_variant,
            expected_phase=expected_phase,
            expected_run_name=expected_run_name,
            expected_step=expected_step,
            expected_data_errors=data_errors,
        )
        ledgers.append(ledger)
    event_chain = ledgers[0]["event_chain_sha256"]
    if any(ledger["event_chain_sha256"] != event_chain for ledger in ledgers):
        raise SSMaxHealthLedgerError("SSMax health ledger event chains differ across ranks")
    return {
        "rank_ledgers": [dict(ledger) for ledger in ledgers],
        "event_chain_sha256": event_chain,
        "counters": {
            "data_errors": sum(int(ledger["data_errors"]) for ledger in ledgers),
            "optimizer_guard_skips": int(ledgers[0]["optimizer_guard_skips"]),
            "nonfinite_losses": int(ledgers[0]["nonfinite_losses"]),
            "nonfinite_gradients": int(ledgers[0]["nonfinite_gradients"]),
        },
    }


@dataclass
class SSMaxHealthLedgerCallback(Callback):
    """Record every reduced SSMax train-step health outcome in trainer checkpoint state.

    Trainer metric collection may be delayed, but OLMo Core flushes and joins every pending
    metric callback before serializing trainer state. Thus permanent checkpoint step *N* always
    contains exactly events 1..N. Loading trainer state restores this chain before a resume.
    """

    priority: ClassVar[int] = 0

    model_variant: str = ""
    phase: str = ""
    run_name: str = ""
    enabled: bool = True
    _events: list[dict[str, Any]] = field(default_factory=list, init=False, repr=False)
    _loaded_data_errors: int = field(default=0, init=False, repr=False)
    _metrics_baseline_step: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.enabled:
            try:
                _validate_identity(self.model_variant, self.phase, self.run_name)
            except SSMaxHealthLedgerError as error:
                raise OLMoConfigurationError(str(error)) from error

    @property
    def last_step(self) -> int:
        """Return the final recorded global step."""

        return len(self._events)

    def _data_errors(self) -> int:
        loader_state = self.trainer.data_loader.state_dict()
        value = loader_state.get("total_data_errors", 0)
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < self._loaded_data_errors
        ):
            raise RuntimeError("SSMax health ledger observed an invalid data-error counter")
        return value

    def _set_metrics_baseline(self) -> None:
        if self.last_step != self.step:
            raise RuntimeError("SSMax health ledger step differs from trainer state")
        self._metrics_baseline_step = self.step

    def pre_train(self) -> None:
        """Bind non-training startup and dry-run metrics to this run segment's baseline."""

        if self.enabled:
            self._set_metrics_baseline()

    def log_metrics(self, step: int, metrics: Dict[str, float]) -> None:
        if not self.enabled:
            return
        # Trainer startup and its compile/OOM dry-run emit ordinary metrics at the current
        # checkpoint step. They are not optimizer-step outcomes and may be delivered more than
        # once (for example after a pre-train save or checkpoint load), so exclude every metric
        # batch at or before this run segment's immutable baseline. Newly trained steps remain
        # strictly contiguous below.
        if step <= self._metrics_baseline_step:
            return
        # OLMo can dispatch multiple disjoint metric batches for one global step. In particular,
        # a synchronous checkpoint first flushes the train metrics needed by this ledger, then
        # records checkpoint timing and other ancillary metrics under that same step for a later
        # flush. Ignore only batches that contain none of the ledger's health contract. A duplicate
        # or partial health batch still reaches the strict step/field checks below and fails closed.
        if not (_TRAIN_HEALTH_METRICS & metrics.keys()):
            return
        if step != self.last_step + 1:
            raise RuntimeError(
                f"SSMax health ledger expected step {self.last_step + 1}, received step {step}"
            )
        missing = _TRAIN_HEALTH_METRICS - set(metrics)
        if missing:
            raise RuntimeError(f"SSMax health ledger metrics are missing {sorted(missing)}")
        loss = float(metrics[TRAIN_CE_LOSS_METRIC])
        grad_norm = float(metrics[OPTIM_GRAD_NORM_METRIC])
        skip = float(metrics[OPTIM_STEP_SKIPPED_METRIC])
        if skip not in (0.0, 1.0):
            raise RuntimeError(f"SSMax health ledger optimizer skip is not boolean: {skip!r}")
        event: dict[str, Any] = {
            "global_step": step,
            "loss": loss if math.isfinite(loss) else None,
            "grad_norm": grad_norm if math.isfinite(grad_norm) else None,
            "loss_finite": math.isfinite(loss),
            "gradients_finite": math.isfinite(grad_norm),
            "optimizer_guard_skipped": bool(skip),
            "previous_event_sha256": (
                self._events[-1]["event_sha256"] if self._events else _ZERO_SHA256
            ),
        }
        event["event_sha256"] = _canonical_sha256(event)
        self._events.append(event)

    def state_dict(self) -> Dict[str, Any]:
        if not self.enabled:
            return {}
        if self.last_step != self.step:
            raise RuntimeError(
                f"SSMax health ledger has {self.last_step} events at trainer step {self.step}"
            )
        data_errors = self._data_errors()
        state: dict[str, Any] = {
            "format": SSMAX_HEALTH_LEDGER_FORMAT,
            "version": SSMAX_HEALTH_LEDGER_VERSION,
            "model_variant": self.model_variant,
            "phase": self.phase,
            "run_name": self.run_name,
            "metrics": {
                "loss": TRAIN_CE_LOSS_METRIC,
                "grad_norm": OPTIM_GRAD_NORM_METRIC,
                "optimizer_guard_skip": OPTIM_STEP_SKIPPED_METRIC,
            },
            "last_step": self.last_step,
            "events": [dict(event) for event in self._events],
            "optimizer_guard_skips": sum(
                int(event["optimizer_guard_skipped"]) for event in self._events
            ),
            "nonfinite_losses": sum(int(not event["loss_finite"]) for event in self._events),
            "nonfinite_gradients": sum(
                int(not event["gradients_finite"]) for event in self._events
            ),
            "data_errors": data_errors,
            "event_chain_sha256": (
                self._events[-1]["event_sha256"] if self._events else _ZERO_SHA256
            ),
        }
        state["content_sha256"] = _canonical_sha256(state)
        validate_ssmax_health_ledger_state(
            state,
            expected_model_variant=self.model_variant,
            expected_phase=self.phase,
            expected_run_name=self.run_name,
            expected_step=self.step,
            expected_data_errors=data_errors,
        )
        return state

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        if not self.enabled:
            if state_dict:
                raise RuntimeError("Disabled SSMax health ledger received checkpoint state")
            return
        try:
            state = validate_ssmax_health_ledger_state(
                state_dict,
                expected_model_variant=self.model_variant,
                expected_phase=self.phase,
                expected_run_name=self.run_name,
                expected_step=int(state_dict.get("last_step", -1)),
            )
        except (SSMaxHealthLedgerError, AttributeError, TypeError, ValueError) as error:
            raise RuntimeError(f"Could not restore SSMax health ledger: {error}") from error
        self._events = [dict(event) for event in state["events"]]
        self._loaded_data_errors = int(state["data_errors"])

    def post_checkpoint_loaded(self, path: Any) -> None:
        del path
        if self.enabled:
            self._set_metrics_baseline()
