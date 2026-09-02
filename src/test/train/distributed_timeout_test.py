from datetime import timedelta

import pytest

from olmo_core.train import _distributed_timeout_from_env


def test_distributed_timeout_uses_default(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("OLMO_DISTRIBUTED_TIMEOUT_SECONDS", raising=False)
    default = timedelta(minutes=15)
    assert _distributed_timeout_from_env(default) == default


def test_distributed_timeout_override(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OLMO_DISTRIBUTED_TIMEOUT_SECONDS", "300")
    assert _distributed_timeout_from_env(timedelta(minutes=15)) == timedelta(minutes=5)


@pytest.mark.parametrize("value", ["0", "-1", "not-a-number"])
def test_distributed_timeout_rejects_invalid_values(monkeypatch: pytest.MonkeyPatch, value: str):
    monkeypatch.setenv("OLMO_DISTRIBUTED_TIMEOUT_SECONDS", value)
    with pytest.raises(ValueError, match="must be"):
        _distributed_timeout_from_env(timedelta(minutes=15))
