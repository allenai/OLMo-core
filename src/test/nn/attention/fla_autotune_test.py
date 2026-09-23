import sys
import types

import pytest

from olmo_core.nn.attention.fla_autotune import freeze_fla_length_autotune

autotuner = pytest.importorskip("triton.runtime.autotuner")


def _fake_autotuner(keys):
    tuner = autotuner.Autotuner.__new__(autotuner.Autotuner)
    tuner.keys = list(keys)
    return tuner


def test_freeze_drops_only_length_keys(monkeypatch):
    mod = types.ModuleType("fla._fake_kernels")
    mod.conv = _fake_autotuner(["D", "W", "NB"])
    mod.norm = _fake_autotuner(["D", "IS_RMS_NORM"])
    other = types.ModuleType("not_fla._fake_kernels")
    other.conv = _fake_autotuner(["D", "NB"])
    monkeypatch.setitem(sys.modules, "fla._fake_kernels", mod)
    monkeypatch.setitem(sys.modules, "not_fla._fake_kernels", other)

    assert freeze_fla_length_autotune() >= 1
    assert mod.conv.keys == ["D", "W"]
    assert mod.norm.keys == ["D", "IS_RMS_NORM"]
    assert other.conv.keys == ["D", "NB"]  # only fla kernels are touched
    # idempotent
    freeze_fla_length_autotune()
    assert mod.conv.keys == ["D", "W"]
