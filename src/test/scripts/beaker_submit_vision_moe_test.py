"""Tests for the project-scoped Vision-MoE Beaker submission wrapper."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _load_module():
    path = Path(__file__).resolve().parents[2] / "scripts" / "beaker_submit_vision_moe.py"
    spec = importlib.util.spec_from_file_location("beaker_submit_vision_moe_test_module", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_submit_uses_fixed_workspace_and_argv_without_a_shell(monkeypatch, tmp_path: Path) -> None:
    module = _load_module()
    spec = tmp_path / "eval spec.yaml"
    spec.write_text("description: test\n")
    calls: list[tuple[list[str], dict[str, object]]] = []

    def fake_run(command: list[str], **kwargs: object) -> None:
        calls.append((command, kwargs))

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    assert (
        module.main(
            [
                str(spec),
                "--name=vision-moe-eval.v1",
                "--format=json",
                "--quiet",
            ]
        )
        == 0
    )
    assert calls == [
        (
            [
                "beaker",
                "experiment",
                "create",
                str(spec.resolve()),
                "--workspace=ai2/molmofication",
                "--name=vision-moe-eval.v1",
                "--format=json",
                "--quiet",
            ],
            {"check": True, "shell": False},
        )
    ]


@pytest.mark.parametrize("workspace_arg", ["--workspace=ai2/OLMo-core", "-w=ai2/OLMo-core"])
def test_submit_rejects_every_workspace_override(
    monkeypatch, tmp_path: Path, workspace_arg: str
) -> None:
    module = _load_module()
    spec = tmp_path / "eval.yaml"
    spec.write_text("description: test\n")

    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail(
            "Beaker must not run when a workspace override is supplied"
        ),
    )

    with pytest.raises(SystemExit):
        module.main([str(spec), workspace_arg])


def test_submit_rejects_arbitrary_beaker_passthrough(monkeypatch, tmp_path: Path) -> None:
    module = _load_module()
    spec = tmp_path / "eval.yaml"
    spec.write_text("description: test\n")
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("Beaker must not run with an unexposed argument"),
    )

    with pytest.raises(SystemExit):
        module.main([str(spec), "--priority=low"])


@pytest.mark.parametrize(
    "unsafe_name",
    ["--workspace=ai2/OLMo-core", "-w", "name;beaker", "name\n--workspace=wrong"],
)
def test_submit_rejects_injection_looking_names(
    monkeypatch, tmp_path: Path, unsafe_name: str
) -> None:
    module = _load_module()
    spec = tmp_path / "eval.yaml"
    spec.write_text("description: test\n")
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("Beaker must not run for an unsafe name"),
    )

    with pytest.raises(SystemExit):
        module.main([str(spec), f"--name={unsafe_name}"])


@pytest.mark.parametrize(
    "unsafe_relative_spec",
    [
        "../escape.yaml",
        "--workspace=wrong.yaml",
        "-w.yaml",
        "spec;beaker.yaml",
        "spec\n-w.yaml",
    ],
)
def test_submit_rejects_injection_looking_or_traversing_specs(
    monkeypatch, tmp_path: Path, unsafe_relative_spec: str
) -> None:
    module = _load_module()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("Beaker must not run for an unsafe spec"),
    )

    with pytest.raises(SystemExit):
        module.main(["--", unsafe_relative_spec])


def test_submit_rejects_symlinked_spec(monkeypatch, tmp_path: Path) -> None:
    module = _load_module()
    real_spec = tmp_path / "real.yaml"
    real_spec.write_text("description: test\n")
    linked_spec = tmp_path / "linked.yaml"
    linked_spec.symlink_to(real_spec)
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("Beaker must not run for a symlinked spec"),
    )

    with pytest.raises(SystemExit):
        module.main([str(linked_spec)])


def test_submit_rejects_missing_spec_before_invoking_beaker(monkeypatch, tmp_path: Path) -> None:
    module = _load_module()
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("Beaker must not run for a missing spec"),
    )

    with pytest.raises(SystemExit):
        module.main([str(tmp_path / "missing.yaml")])
