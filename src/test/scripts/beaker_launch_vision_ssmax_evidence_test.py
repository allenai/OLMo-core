from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from scripts import beaker_launch_vision_ssmax_evidence as launcher


def _arguments(root: Path, *, stage: str = "bridge") -> list[str]:
    manifest = root / "manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text("{}\n")
    values = [
        "--manifest",
        str(manifest),
        "--expected-manifest-sha256",
        hashlib.sha256(manifest.read_bytes()).hexdigest(),
    ]
    if stage == "perception":
        values.extend(["--arm", "treatment"])
    values.extend(
        [
            "--step",
            "0",
            "--work-dir",
            str(root / "work"),
            "--output",
            str(root / "receipt.json"),
        ]
    )
    return values


@pytest.mark.parametrize("stage", ["bridge", "perception", "joint"])
def test_build_launch_config_fixes_operational_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stage: str
) -> None:
    monkeypatch.setattr(launcher, "EVIDENCE_ROOT", tmp_path)
    config = launcher.build_launch_config(
        name=f"ssmax-{stage}-step0",
        stage=stage,
        evaluator_arguments=_arguments(tmp_path, stage=stage),
    )

    assert config.workspace == "ai2/scaling-ladders"
    assert config.budget == "ai2/oe-other"
    assert config.clusters == ["ai2/holmes"]
    assert config.num_nodes == 2
    assert config.num_gpus == 8
    assert config.torchrun is True
    assert config.priority == "urgent"
    assert config.min_runtime == "8h"
    assert config.allow_dirty is False
    assert config.follow is False
    assert config.shared_filesystem is True
    assert config.post_setup is None
    assert [secret.name for secret in config.env_secrets] == ["BEAKER_TOKEN"]
    assert config.cmd[0] == launcher._EVALUATORS[stage]


def test_launch_rejects_manifest_byte_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(launcher, "EVIDENCE_ROOT", tmp_path)
    arguments = _arguments(tmp_path)
    (tmp_path / "manifest.json").write_text('{"drift": true}\n')

    with pytest.raises(launcher.EvidenceLaunchError, match="manifest bytes differ"):
        launcher.build_launch_config(
            name="ssmax-bridge-step0",
            stage="bridge",
            evaluator_arguments=arguments,
        )


@pytest.mark.parametrize(
    ("stage", "mutator", "message"),
    [
        ("bridge", lambda values: values + ["--arm", "treatment"], "unsupported"),
        ("bridge", lambda values: values[:-2], "missing required"),
        (
            "perception",
            lambda values: ["frozen_vision" if value == "treatment" else value for value in values],
            "perception arm",
        ),
        (
            "bridge",
            lambda values: ["99" if value == "0" else value for value in values],
            "unsupported bridge evidence step",
        ),
        (
            "joint",
            lambda values: ["3000" if value == "0" else value for value in values],
            "unsupported joint evidence step",
        ),
    ],
)
def test_launch_rejects_contract_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
    mutator,
    message: str,
) -> None:
    monkeypatch.setattr(launcher, "EVIDENCE_ROOT", tmp_path)
    arguments = mutator(_arguments(tmp_path, stage=stage))

    with pytest.raises(launcher.EvidenceLaunchError, match=message):
        launcher.build_launch_config(
            name=f"ssmax-{stage}-step0",
            stage=stage,
            evaluator_arguments=arguments,
            verify_manifest_bytes=False,
        )
