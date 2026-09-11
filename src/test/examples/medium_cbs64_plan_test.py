"""CPU checks for the bounded medium continuation campaign."""

import copy
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples/olmo_ddp"))
from olmoe3_medium_cbs64_control import (
    EXCLUDED,
    config_spec,
    registration_matches,
    training_spec,
    validation_environments,
)
from olmoe3_medium_cbs64_plan import (
    FORK_TOKENS,
    PHASES,
    TARGET_TOKENS,
    WAVES,
    old_rank_for_half,
    phase_environment,
    validate,
)
from olmoe3_medium_followup_plan import microbatch_sequence_sizes, parse_test
from olmoe3_nsys_tools import NsysSettings


def test_horizons_and_scaled_lrs():
    validate()
    assert FORK_TOKENS == 67_108_864_000 and TARGET_TOKENS == 100_663_296_000
    a, b = PHASES["64g-32mi-cbs"], PHASES["64g-64mi-cbs"]
    assert a.end - a.start == 1000 and b.end - b.start == 500
    assert a.lr == pytest.approx(0.00092 * 2**0.5)
    assert b.lr == pytest.approx(0.00184)
    assert a.save_interval * a.batch == b.save_interval * b.batch
    assert a.eval_interval * a.batch == b.eval_interval * b.batch


def test_registration_restart_ignores_only_creation_time():
    @dataclass(frozen=True)
    class Registration:
        run_id: str = "run"
        bucket_id: str = "private-bucket"
        deletion_mode: str = "apply"
        min_local_checkpoints: int = 2
        enabled: bool = True
        created_at: str = "original"

    original = Registration()
    requested = replace(original, created_at="restart")
    assert registration_matches(original, requested)
    assert original.created_at == "original" and requested.created_at == "restart"
    for field, value in (
        ("run_id", "other"),
        ("bucket_id", "other"),
        ("deletion_mode", "report_only"),
        ("min_local_checkpoints", 1),
        ("enabled", False),
    ):
        assert not registration_matches(original, replace(requested, **{field: value}))


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
                    {"name": "OLMOE3_NSYS_RANKS", "value": ",".join(map(str, range(0, 128, 8)))},
                    {"name": "OLMOE3_NSYS_VERSION", "value": "2026.4.1"},
                    {"name": "OLMOE3_MEDIUM_CAPTURE", "value": "1"},
                    {"name": "OLMOE3_DEEP_PROFILE_PASS", "value": "nsys"},
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
    assert not any(k.startswith("OLMOE3_NSYS_") for k in env)
    assert env["OLMOE3_MEDIUM_CAPTURE"]["value"] == "0"
    assert env["OLMOE3_DEEP_PROFILE_PASS"]["value"] == "timing"
    assert env["NUM_NODES"]["value"] == str(first.gpus // 8)
    assert task["context"] == {"priority": "urgent", "minRuntime": "1h", "autoResume": False}
    assert task["result"] == {"path": "/noop-results"}
    for host in task["constraints"]["hostname"]:
        assert host.split(".")[0] not in EXCLUDED


def test_config_gate_compiles():
    cpu = template()
    cpu["tasks"][0]["resources"] = {"cpuCount": 4, "memory": "16 GiB", "sharedMemory": "2 GiB"}
    spec = config_spec(cpu, "a" * 40, template())
    assert spec["tasks"][0]["constraints"] == {"cluster": ["ai2/phobos"]}
    assert "resources" not in spec["tasks"][0]
    assert spec["tasks"][0]["context"]["minRuntime"] == "0s"
    compile(spec["tasks"][0]["arguments"][-1], "config-gate", "exec")


def test_config_gate_rejects_gpu_template():
    with pytest.raises(AssertionError, match="must not request GPUs"):
        config_spec(template(), "a" * 40, template())


def test_config_gate_accepts_no_resource_template():
    cpu = template()
    del cpu["tasks"][0]["resources"]
    assert "resources" not in config_spec(cpu, "a" * 40, template())["tasks"][0]


@pytest.mark.parametrize("key", PHASES)
def test_all_phase_environments_clear_stale_capture_settings(key):
    stale = {v["name"]: v["value"] for v in template()["tasks"][0]["envVars"] if "value" in v}
    stale["PATH"] = "/unchanged"
    saved = dict(stale)
    phase = PHASES[key]
    env = phase_environment(stale, phase)
    assert stale == saved and env["PATH"] == "/unchanged"
    assert env["OLMOE3_MEDIUM_BATCH"] == str(phase.batch)
    assert env["OLMOE3_MEDIUM_GPUS"] == str(phase.gpus)
    assert env["OLMOE3_MEDIUM_CAPTURE"] == "0"
    assert env["OLMOE3_DEEP_PROFILE_PASS"] == "timing"
    assert not any(k.startswith("OLMOE3_NSYS_") for k in env)
    assert max(NsysSettings.from_env(env, phase.gpus).ranks) < phase.gpus


def test_cpu_gate_uses_actual_gpu_spec_environments(monkeypatch):
    import subprocess

    cpu = template()
    cpu["tasks"][0].pop("resources")
    training = template()
    training["tasks"][0]["envVars"].append(
        {"name": "OLMO_PROFILE_TEST_MARKER", "value": "gpu-spec"}
    )
    expected = validation_environments(training, "a" * 40)
    calls = []
    monkeypatch.setattr(subprocess, "run", lambda command, **kw: calls.append((command, kw)))
    monkeypatch.setenv("OLMOE3_NSYS_RANKS", "0,9999")
    monkeypatch.setenv("OLMO_CPU_TEMPLATE_ONLY", "stale")
    code = config_spec(cpu, "a" * 40, training)["tasks"][0]["arguments"][-1]
    assert "BEAKER_TOKEN" not in code and "jacobm_BEAKER_TOKEN" not in code
    exec(compile(code, "actual-gpu-config-gate", "exec"), {})
    assert len(calls) == len(PHASES)
    for command, kwargs in calls:
        assert command[-1] == "--validate-only" and kwargs["check"]
        env = kwargs["env"]
        key = env["OLMOE3_MEDIUM_CBS64_PHASE"]
        assert env["OLMO_PROFILE_TEST_MARKER"] == "gpu-spec"
        assert "OLMO_CPU_TEMPLATE_ONLY" not in env
        actual = {k: v for k, v in env.items() if k.startswith(("OLMO", "NCCL"))}
        assert actual == expected[key]


@pytest.mark.parametrize("gpus", [64, 128])
@pytest.mark.parametrize("batch", [32, 64])
def test_microbatch_geometry(gpus, batch):
    chunks = microbatch_sequence_sizes(batch * 1024**2, gpus, 2)
    assert set(chunks) == {2}
    assert sum(chunks) * gpus * 8192 == batch * 1024**2
    assert parse_test(f"optimized-metrics5-mb2-b{batch}mi")[2] == batch * 1024**2
