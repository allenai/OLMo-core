"""CPU checks for the bounded medium continuation campaign."""

import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples/olmo_ddp"))
from olmoe3_medium_cbs64_control import EXCLUDED, config_spec, training_spec
from olmoe3_medium_cbs64_plan import (
    FORK_TOKENS,
    PHASES,
    TARGET_TOKENS,
    WAVES,
    old_rank_for_half,
    validate,
)
from olmoe3_medium_followup_plan import microbatch_sequence_sizes, parse_test


def test_horizons_and_scaled_lrs():
    validate()
    assert FORK_TOKENS == 67_108_864_000 and TARGET_TOKENS == 100_663_296_000
    a, b = PHASES["64g-32mi-cbs"], PHASES["64g-64mi-cbs"]
    assert a.end - a.start == 1000 and b.end - b.start == 500
    assert a.lr == pytest.approx(0.00092 * 2**0.5)
    assert b.lr == pytest.approx(0.00184)
    assert a.save_interval * a.batch == b.save_interval * b.batch
    assert a.eval_interval * a.batch == b.eval_interval * b.batch


@pytest.mark.parametrize("group", ["dp", "ep_dp"])
def test_reshard_rank_mapping(group):
    seen = []
    for rank in range(64):
        halves = [old_rank_for_half(rank, half, group) for half in (0, 1)]
        seen.extend(halves)
        if group == "ep_dp":
            assert all(old % 8 == rank % 8 for old in halves)
        else:
            assert halves == [2 * rank, 2 * rank + 1]
    assert sorted(seen) == list(range(128))


def template():
    return {
        "version": "v2",
        "tasks": [
            {
                "name": "train",
                "resources": {"gpuCount": 8},
                "constraints": {
                    "hostname": [f"holmes-cs-aus-{n}.reviz.ai2.in" for n in range(485, 557)]
                },
                "envVars": [
                    {"name": "BEAKER_TOKEN", "secret": "jacobm_BEAKER_TOKEN"},
                    {"name": "OLMOE3_MEDIUM_CBS_RUN", "value": "old-parent"},
                ],
            }
        ],
    }


@pytest.mark.parametrize("wave", WAVES)
def test_specs_preserve_secrets_and_resources(wave):
    original = template()
    saved = copy.deepcopy(original)
    spec = training_spec(original, wave, "a" * 40)
    assert original == saved
    task = spec["tasks"][0]
    first = PHASES[WAVES[wave][0]]
    assert task["replicas"] * task["resources"]["gpuCount"] == first.gpus
    env = {item["name"]: item for item in task["envVars"]}
    assert env["BEAKER_TOKEN"] == {"name": "BEAKER_TOKEN", "secret": "jacobm_BEAKER_TOKEN"}
    assert "OLMOE3_MEDIUM_CBS_RUN" not in env
    assert env["NUM_NODES"]["value"] == str(first.gpus // 8)
    assert task["context"] == {"priority": "urgent", "minRuntime": "1h", "autoResume": False}
    assert task["result"] == {"path": "/noop-results"}
    for host in task["constraints"]["hostname"]:
        assert host.split(".")[0] not in EXCLUDED


def test_config_gate_compiles():
    spec = config_spec(template(), "a" * 40)
    compile(spec["tasks"][0]["arguments"][-1], "config-gate", "exec")


@pytest.mark.parametrize("gpus", [64, 128])
@pytest.mark.parametrize("batch", [32, 64])
def test_microbatch_geometry(gpus, batch):
    chunks = microbatch_sequence_sizes(batch * 1024**2, gpus, 2)
    assert set(chunks) == {2}
    assert sum(chunks) * gpus * 8192 == batch * 1024**2
    assert parse_test(f"optimized-metrics5-mb2-b{batch}mi")[2] == batch * 1024**2
