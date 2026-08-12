"""Focused tests for the distributed production perception runtime preflight."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import pytest


def _load_module():
    path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "eval"
        / "vision_alignment_perception_runtime_preflight.py"
    )
    spec = importlib.util.spec_from_file_location(
        "vision_alignment_perception_runtime_preflight_test_module", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _valid_packets(module, git_ref: str = "a" * 40):
    packets = []
    for rank in range(module.WORLD_SIZE):
        replica_rank = rank // module.LOCAL_WORLD_SIZE
        local_rank = rank % module.LOCAL_WORLD_SIZE
        node_id = f"node-{replica_rank}"
        hostname = f"holmes-test-{replica_rank}.example.ai2.in"
        packets.append(
            {
                "rank": rank,
                "env_rank": rank,
                "local_rank": local_rank,
                "world_size": module.WORLD_SIZE,
                "local_world_size": module.LOCAL_WORLD_SIZE,
                "num_nodes": module.NUM_NODES,
                "replica_count": module.NUM_NODES,
                "replica_rank": replica_rank,
                "assigned_gpu_count": module.LOCAL_WORLD_SIZE,
                "cuda_device_count": module.LOCAL_WORLD_SIZE,
                "cuda_device": local_rank,
                "workspace_id": module.CANONICAL_WORKSPACE_ID,
                "experiment_id": "experiment-0",
                "workload_id": "experiment-0",
                "task_id": f"task-{replica_rank}",
                "job_id": f"job-{replica_rank}",
                "job_kind": "batch",
                "node_id": node_id,
                "hostname": hostname,
                "leader_node_id": "node-0",
                "leader_hostname": "holmes-test-0.example.ai2.in",
                "git_branch": module.CANONICAL_GIT_BRANCH,
                "git_ref": git_ref,
                "checkout_ref": git_ref,
                "tracked_checkout_dirty": False,
                "credential_env_names": [],
            }
        )
    return packets


def test_validate_rank_metadata_accepts_exact_two_by_eight_holmes_topology() -> None:
    module = _load_module()
    summary = module._validate_rank_metadata(_valid_packets(module), expected_git_ref="a" * 40)
    assert summary == {
        "world_size": 16,
        "nodes": 2,
        "gpus_per_node": 8,
        "workspace": "ai2/molmofication",
        "workspace_id": "01KSTRJHG4A32N7GDM82KY8J3E",
        "cluster": "ai2/holmes",
        "experiment_id": "experiment-0",
        "node_hostnames": [
            "holmes-test-0.example.ai2.in",
            "holmes-test-1.example.ai2.in",
        ],
    }


def test_validate_rank_metadata_propagates_exact_rank_error() -> None:
    module = _load_module()
    packets = _valid_packets(module)
    packets[7] = {"rank": 7, "error": "ValueError: missing BEAKER_NODE_ID"}
    with pytest.raises(
        module.PerceptionRuntimePreflightError,
        match=r"rank 7: ValueError: missing BEAKER_NODE_ID",
    ):
        module._validate_rank_metadata(packets, expected_git_ref="a" * 40)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda packet: packet.update(workspace_id="wrong"), "workspace_id differs"),
        (lambda packet: packet.update(assigned_gpu_count=7), "assigned_gpu_count differs"),
        (lambda packet: packet.update(hostname="jupiter-test.example.ai2.in"), "Holmes metadata"),
        (lambda packet: packet.update(checkout_ref="b" * 40), "checkout_ref differs"),
        (lambda packet: packet.update(tracked_checkout_dirty=True), "tracked_checkout_dirty"),
        (
            lambda packet: packet.update(credential_env_names=["AWS_SESSION_TOKEN"]),
            "credential_env_names differs",
        ),
    ],
)
def test_validate_rank_metadata_rejects_authority_or_checkout_drift(mutation, message) -> None:
    module = _load_module()
    packets = _valid_packets(module)
    mutation(packets[9])
    with pytest.raises(module.PerceptionRuntimePreflightError, match=message):
        module._validate_rank_metadata(packets, expected_git_ref="a" * 40)


def test_validate_rank_metadata_requires_two_distinct_eight_rank_nodes() -> None:
    module = _load_module()
    packets = _valid_packets(module)
    packets[8]["node_id"] = "node-0"
    packets[8]["hostname"] = "holmes-test-0.example.ai2.in"
    with pytest.raises(module.PerceptionRuntimePreflightError, match="contains 9 ranks"):
        module._validate_rank_metadata(packets, expected_git_ref="a" * 40)


@pytest.mark.parametrize(
    ("flag", "value", "message"),
    [
        ("--expected-recipe-sha256", "A" * 64, "64 lowercase"),
        ("--expected-profile-sha256", "a" * 63, "64 lowercase"),
        ("--expected-git-ref", "a" * 64, "40 lowercase"),
    ],
)
def test_cli_rejects_malformed_identity_pins(flag, value, message, capsys) -> None:
    module = _load_module()
    argv = [
        "--recipe=recipe.py",
        f"--expected-recipe-sha256={'a' * 64}",
        "--profile=profile.yaml",
        f"--expected-profile-sha256={'b' * 64}",
        f"--expected-git-ref={'c' * 40}",
    ]
    prefix = f"{flag}="
    argv = [f"{flag}={value}" if item.startswith(prefix) else item for item in argv]
    with pytest.raises(SystemExit, match="2"):
        module._parse_args(argv)
    assert message in capsys.readouterr().err


def test_pinned_file_rejects_changed_bytes(tmp_path: Path) -> None:
    module = _load_module()
    path = tmp_path / "profile.yaml"
    path.write_bytes(b"reviewed bytes\n")
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    assert module._pinned_file(path, expected, name="profile") == path.resolve()
    path.write_bytes(b"changed bytes\n")
    with pytest.raises(module.PerceptionRuntimePreflightError, match="SHA-256 differs"):
        module._pinned_file(path, expected, name="profile")


def test_credential_env_names_reports_names_only() -> None:
    module = _load_module()
    environment = {
        "AWS_ACCESS_KEY_ID": "secret-value-must-not-be-returned",
        "RUSTINS_AWS_SESSION_TOKEN": "another-secret-value",
        "S3_PROFILE": "S3",
        "WEKA_PROFILE": "WEKA",
    }
    assert module._credential_env_names(environment) == [
        "AWS_ACCESS_KEY_ID",
        "RUSTINS_AWS_SESSION_TOKEN",
    ]
