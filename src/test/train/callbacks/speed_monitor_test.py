import time
from types import SimpleNamespace

import pytest

from olmo_core.train.callbacks.speed_monitor import (
    SpeedMonitorCallback,
    get_device_peak_flops_per_second,
)


@pytest.mark.parametrize(
    ("device_name", "expected"),
    [
        ("NVIDIA B200", int(4.5e15 * 0.5)),
        ("NVIDIA B300", int(4.5e15 * 0.5)),
        ("NVIDIA GB300", int(4.5e15 * 0.5)),
        ("NVIDIA RTX PRO 6000", int(1008e12 * 0.5)),
        ("NVIDIA H100 NVL", int(1671e12 * 0.5)),
    ],
)
def test_gpu_peak_flops_uses_dense_bf16_spec(device_name: str, expected: int):
    assert get_device_peak_flops_per_second(device_name, using_half_precision=True) == expected


def test_device_peak_flops_returns_none_without_half_precision():
    assert get_device_peak_flops_per_second("NVIDIA B300", using_half_precision=False) is None


def _run_pre_step(monkeypatch, env_value, load_seconds):
    cb = SpeedMonitorCallback()
    # Only ``self.step`` is read from the trainer in the warn path.
    cb._trainer = SimpleNamespace(global_step=7)  # type: ignore[assignment]
    cb._first_step = True  # pre_step returns right after the warn block.
    if env_value is None:
        monkeypatch.delenv("OLMO_BATCH_LOAD_WARN_SECONDS", raising=False)
    else:
        monkeypatch.setenv("OLMO_BATCH_LOAD_WARN_SECONDS", env_value)
    cb._batch_load_start = time.perf_counter() - load_seconds
    cb.pre_step({})
    return cb


def test_slow_batch_load_warns(monkeypatch, caplog):
    with caplog.at_level("WARNING"):
        cb = _run_pre_step(monkeypatch, "0.01", load_seconds=0.05)
    assert cb._batch_load_warn_threshold == 0.01
    assert any("Slow batch load" in r.getMessage() for r in caplog.records)


def test_fast_batch_load_does_not_warn(monkeypatch, caplog):
    with caplog.at_level("WARNING"):
        cb = _run_pre_step(monkeypatch, "10", load_seconds=0.0)
    assert cb._batch_load_warn_threshold == 10.0
    assert not any("Slow batch load" in r.getMessage() for r in caplog.records)


def test_invalid_threshold_is_ignored_with_warning(monkeypatch, caplog):
    with caplog.at_level("WARNING"):
        cb = _run_pre_step(monkeypatch, "not-a-number", load_seconds=0.05)
    # Invalid value disables the warning (threshold 0) rather than raising.
    assert cb._batch_load_warn_threshold == 0.0
    assert any(
        "Ignoring invalid OLMO_BATCH_LOAD_WARN_SECONDS" in r.getMessage() for r in caplog.records
    )
    assert not any("Slow batch load" in r.getMessage() for r in caplog.records)


def test_no_env_var_disables_warning(monkeypatch, caplog):
    with caplog.at_level("WARNING"):
        cb = _run_pre_step(monkeypatch, None, load_seconds=0.05)
    assert cb._batch_load_warn_threshold == 0.0
    assert not any("Slow batch load" in r.getMessage() for r in caplog.records)
