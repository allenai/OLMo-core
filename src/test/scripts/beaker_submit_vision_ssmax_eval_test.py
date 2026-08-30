"""Tests for the project-scoped dense-SSMax evaluation submission wrapper."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import yaml


def _module():
    path = Path(__file__).resolve().parents[2] / "scripts/beaker_submit_vision_ssmax_eval.py"
    spec = importlib.util.spec_from_file_location("beaker_submit_vision_ssmax_eval_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _template_path() -> Path:
    return (
        Path(__file__).resolve().parents[3]
        / "configs/vision_moe/vision_alignment/eval/downstream/ssmax_joint_fast_pair.yaml.template"
    )


def _template(module):
    with _template_path().open() as file_handle:
        payload = yaml.safe_load(file_handle)
    module.validate_spec(payload, allow_placeholders=True)
    return payload


def _materialize(payload: dict) -> dict:
    replacements = {
        "<OLMO_CORE_GIT_REF>": "1" * 40,
        "<JOINT_GLOBAL_STEP>": "100",
        "<SSMAX_HEAD_QKNORM_JOINT_CHECKPOINT>": (
            "/weka/oe-training-default/rustin/experiments/vision-ssmax-molmofication/"
            "vision-alignment/checkpoints/qk/step100"
        ),
        "<SSMAX_NO_QKNORM_JOINT_CHECKPOINT>": (
            "/weka/oe-training-default/rustin/experiments/vision-ssmax-molmofication/"
            "vision-alignment/checkpoints/noqk/step100"
        ),
        "<SSMAX_HEAD_QKNORM_RESULT_PATH>": (
            "/weka/oe-training-default/rustin/experiments/vision-ssmax-molmofication/"
            "vision-alignment/evals/qk.json"
        ),
        "<SSMAX_NO_QKNORM_RESULT_PATH>": (
            "/weka/oe-training-default/rustin/experiments/vision-ssmax-molmofication/"
            "vision-alignment/evals/noqk.json"
        ),
    }
    for task in payload["tasks"]:
        task["arguments"] = [
            replacements.get(value, "2" * 64 if "SHA256>" in str(value) else value)
            for value in task["arguments"]
        ]
        for row in task["envVars"]:
            if row.get("value") in replacements:
                row["value"] = replacements[row["value"]]
    return payload


def test_template_and_materialized_spec_satisfy_exact_contract() -> None:
    module = _module()
    payload = _template(module)
    module.validate_spec(_materialize(payload), allow_placeholders=False)


def test_single_arm_staging_is_explicit_and_exact() -> None:
    module = _module()
    payload = _materialize(_template(module))
    payload["tasks"] = [payload["tasks"][0]]

    with pytest.raises(module.SpecValidationError, match="two model-arm tasks"):
        module.validate_spec(payload, allow_placeholders=False)

    module.validate_spec(
        payload,
        allow_placeholders=False,
        expected_single_arm="ssmax_head_qknorm",
    )
    with pytest.raises(module.SpecValidationError, match="unexpected SSMax variant"):
        module.validate_spec(
            payload,
            allow_placeholders=False,
            expected_single_arm="ssmax_no_qknorm",
        )


def test_single_arm_cli_submits_validated_spec(tmp_path, monkeypatch) -> None:
    module = _module()
    payload = _materialize(_template(module))
    payload["tasks"] = [payload["tasks"][0]]
    spec_path = tmp_path / "single-arm.yaml"
    spec_path.write_text(yaml.safe_dump(payload))
    calls = []
    monkeypatch.setattr(
        module.subprocess, "run", lambda *args, **kwargs: calls.append((args, kwargs))
    )

    assert (
        module.main(
            [
                str(spec_path),
                "--single-arm",
                "ssmax_head_qknorm",
                "--name",
                "ssmax-head-eval",
            ]
        )
        == 0
    )
    assert calls == [
        (
            (
                [
                    "beaker",
                    "experiment",
                    "create",
                    str(spec_path),
                    "--workspace=ai2/scaling-ladders",
                    "--name=ssmax-head-eval",
                ],
            ),
            {"check": True, "shell": False},
        )
    ]


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("context", "priority"), "normal", "priority"),
        (("context", "minRuntime"), "7h59m59s", "eight hours"),
        (("constraints", "cluster"), ["ai2/jupiter"], "cluster"),
        (("resources", "gpuCount"), 8, "one GPU"),
    ],
)
def test_spec_rejects_weakened_launch_contract(path, value, message) -> None:
    module = _module()
    payload = _materialize(_template(module))
    payload["tasks"][0][path[0]][path[1]] = value
    with pytest.raises(module.SpecValidationError, match=message):
        module.validate_spec(payload, allow_placeholders=False)


def test_spec_requires_one_matching_phase_and_step_for_the_pair() -> None:
    module = _module()
    payload = _materialize(_template(module))
    arguments = payload["tasks"][1]["arguments"]
    arguments[arguments.index("--expected-phase") + 1] = "perception"
    with pytest.raises(module.SpecValidationError, match="same declared phase"):
        module.validate_spec(payload, allow_placeholders=False)

    payload = _materialize(_template(module))
    arguments = payload["tasks"][1]["arguments"]
    arguments[arguments.index("--expected-global-step") + 1] = "200"
    checkpoint_index = arguments.index("--checkpoint") + 1
    arguments[checkpoint_index] = arguments[checkpoint_index].replace("step100", "step200")
    with pytest.raises(module.SpecValidationError, match="same declared global step"):
        module.validate_spec(payload, allow_placeholders=False)


@pytest.mark.parametrize(
    ("phase", "step"),
    [("bridge", "0"), ("bridge", "500"), ("perception", "3000"), ("joint", "4000")],
)
def test_spec_accepts_each_reviewed_matched_trajectory_point(phase: str, step: str) -> None:
    module = _module()
    payload = _materialize(_template(module))
    for task in payload["tasks"]:
        arguments = task["arguments"]
        arguments[arguments.index("--expected-phase") + 1] = phase
        arguments[arguments.index("--expected-global-step") + 1] = step
        checkpoint_index = arguments.index("--checkpoint") + 1
        arguments[checkpoint_index] = arguments[checkpoint_index].replace("step100", f"step{step}")
    module.validate_spec(payload, allow_placeholders=False)


def test_spec_rejects_unreviewed_arguments_and_secret_surface() -> None:
    module = _module()
    payload = _materialize(_template(module))
    payload["tasks"][0]["arguments"].extend(["--limit", "1"])
    with pytest.raises(module.SpecValidationError, match="exact reviewed evaluator argument"):
        module.validate_spec(payload, allow_placeholders=False)

    payload = _materialize(_template(module))
    payload["tasks"][0]["envVars"].append({"name": "HF_TOKEN", "secret": "HF_TOKEN"})
    with pytest.raises(module.SpecValidationError, match="secret surface"):
        module.validate_spec(payload, allow_placeholders=False)

    payload = _materialize(_template(module))
    payload["tasks"][0]["envVars"].append({"name": "OPENAI_API_KEY", "value": "forbidden"})
    with pytest.raises(module.SpecValidationError, match="auth environment"):
        module.validate_spec(payload, allow_placeholders=False)


def test_submission_command_has_fixed_scaling_ladders_workspace() -> None:
    module = _module()
    command = module._command(
        Path("spec.yaml"), name="ssmax-eval", output_format="json", quiet=True
    )
    assert command == [
        "beaker",
        "experiment",
        "create",
        "spec.yaml",
        "--workspace=ai2/scaling-ladders",
        "--name=ssmax-eval",
        "--format=json",
        "--quiet",
    ]


def test_placeholder_validation_never_invokes_beaker(monkeypatch) -> None:
    module = _module()
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("template validation must not submit"),
    )
    assert module.main([str(_template_path()), "--validate-only", "--allow-placeholders"]) == 0
