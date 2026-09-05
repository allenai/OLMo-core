"""A new comparison must not accidentally use the old slow reference bundle."""

import importlib.util
from pathlib import Path

import pytest

_PATH = Path(__file__).parents[2] / "examples/olmo_ddp/olmoe3_integration_policy.py"
_SPEC = importlib.util.spec_from_file_location("integration_policy", _PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
POLICY = _MODULE.QUALIFIED_POLICY
DEFER = "OLMO_PROFILE_DDP_DEFER_REPLICATED_REDUCTIONS"
OVERLAP = "OLMO_PROFILE_LB_COUNT_OVERLAP"


def test_original_campaign_unchanged():
    reference = _MODULE.integration_policy("reference", POLICY)
    optimized = _MODULE.integration_policy("optimized", POLICY)
    assert not any(int(v) for v in reference["flags"].values())
    assert reference["kda_min_ctas"] == 256
    assert not reference["reduce_scatter"]
    assert not reference["inverse_scatter"]
    assert optimized["kda_min_ctas"] == 128
    assert optimized["reduce_scatter"] and optimized["inverse_scatter"]
    assert optimized["flags"][DEFER] == optimized["flags"][OVERLAP] == "0"


@pytest.mark.parametrize("communication", ["none", "deferred", "lb-overlap", "deferred-lb"])
def test_wave2_baseline_and_only_selected_switches(communication, monkeypatch):
    # Ambient experiments must not enable switches in the reference arm.
    for key in _MODULE.FLAGS:
        monkeypatch.setenv(key, "1")
    old_optimized = _MODULE.integration_policy("optimized", POLICY)
    reference = _MODULE.integration_policy("reference", POLICY, "optimized100b", communication)
    candidate = _MODULE.integration_policy("optimized", POLICY, "optimized100b", communication)
    assert reference["flags"] == old_optimized["flags"]
    assert reference["communication"] == "none"
    for key in ("kda_min_ctas", "reduce_scatter", "inverse_scatter"):
        assert reference[key] == candidate[key] == old_optimized[key]
    expected = dict(reference["flags"])
    expected[DEFER] = "1" if communication in ("deferred", "deferred-lb") else "0"
    expected[OVERLAP] = "1" if communication in ("lb-overlap", "deferred-lb") else "0"
    assert candidate["flags"] == expected


@pytest.mark.parametrize(
    "arm,policy,baseline,communication",
    [
        ("bad", POLICY, "optimized100b", "none"),
        ("optimized", "bad", "optimized100b", "none"),
        ("optimized", POLICY, "bad", "none"),
        ("optimized", POLICY, "optimized100b", "bad"),
        ("optimized", POLICY, "original", "deferred-lb"),
        ("optimized", "core-docpool", "optimized100b", "deferred-lb"),
    ],
)
def test_ambiguous_or_incomplete_policy_rejected(arm, policy, baseline, communication):
    with pytest.raises(ValueError):
        _MODULE.integration_policy(arm, policy, baseline, communication)
