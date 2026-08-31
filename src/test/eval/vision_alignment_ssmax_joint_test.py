from __future__ import annotations

import ast
import copy
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from olmo_core.eval import vision_alignment_ssmax_joint as joint

ZERO = "0" * 64


def _spec(variant: str = "ssmax_head_qknorm") -> dict[str, Any]:
    return {
        "format": joint.MANIFEST_SPEC_FORMAT,
        "version": joint.MANIFEST_SPEC_VERSION,
        "run_id": f"{variant}-joint-v1",
        "model_variant": variant,
        "run_name": f"{variant}-joint",
        "checkpoint_root": "/checkpoints/joint",
        "checkpoint_config_sha256s": {
            step: row["config_sha256"]
            for step, row in joint.TRAINING_RESUME_SCHEDULES[variant].items()
        },
        "evidence_git": {
            "repo": "allenai/OLMo-core",
            "repo_url": "https://github.com/allenai/OLMo-core",
            "ref": "f" * 40,
        },
        "training_profile": "/profiles/joint.yaml",
        "recipe": "/repo/Vision-Alignment.py",
        "perception_parent_gate": "/evidence/perception-v5.json",
        "joint_visual_projection": "/artifacts/projection.json",
        "source_audit": "/artifacts/audit.json",
        "attention_probe": "/artifacts/joint-attention.json",
        "pairing_paths": {source: f"/pairings/{source}.json" for source in joint.VISUAL_SOURCES},
        "evaluation": {
            "visual_sources": list(joint.VISUAL_SOURCES),
            "steps": list(joint.REQUIRED_STEPS),
            "windows": list(joint.WINDOWS),
            "examples_per_source": 496,
            "eligible_rows_per_source": dict(joint.ELIGIBLE_VISUAL_ROWS),
            "native_holdout_examples": 992,
            "pairing_seed": 6198,
            "single_response_projection_seed": 95818,
            "rank_batch_instances": 1,
        },
        "topology": {
            "world_size": 16,
            "num_nodes": 2,
            "gpus_per_node": 8,
            "data_parallel": "hsdp",
        },
        "policy": {
            "decision_scope": "descriptive_non_promotion",
            "maximum_data_errors": 0,
            "maximum_optimizer_guard_skips": 0,
            "maximum_nonfinite_losses": 0,
            "maximum_nonfinite_gradients": 0,
            "native_text_ce_max_relative_increase": 0.02,
            "native_text_bootstrap_samples": 10000,
            "native_text_bootstrap_seed": 6198,
            "require_exact_frozen_surfaces": True,
        },
        "companion_protocols": {"downstream_fast_pair": "/protocols/downstream.yaml"},
    }


@pytest.mark.parametrize(
    "filename",
    [
        "ssmax_head_qknorm_joint_manifest_v1.json.template",
        "ssmax_no_qknorm_joint_manifest_v1.json.template",
    ],
)
def test_checked_in_manifest_templates_lock_joint_trajectory(filename: str) -> None:
    path = (
        Path(__file__).resolve().parents[3]
        / "configs/vision_moe/vision_alignment/eval/joint"
        / filename
    )
    spec = joint._validate_spec(joint.load_json(path))

    assert spec["evaluation"]["steps"] == [0, 4000, 8000, 12000, 16000]
    assert spec["evaluation"]["examples_per_source"] == 496
    assert spec["evaluation"]["eligible_rows_per_source"] == joint.ELIGIBLE_VISUAL_ROWS
    assert spec["evaluation"]["native_holdout_examples"] == 992
    assert spec["topology"] == {
        "world_size": 16,
        "num_nodes": 2,
        "gpus_per_node": 8,
        "data_parallel": "hsdp",
    }
    assert spec["policy"]["decision_scope"] == "descriptive_non_promotion"


_CHECKPOINT_CONFIG_SHA256S = {
    "ssmax_head_qknorm_joint_manifest_v1.json.template": {
        "0": "18b0ce331150767c71ead409c116f75cb3d1fa4c163365b80b06a43b51eb8d4e",
        "4000": "18b0ce331150767c71ead409c116f75cb3d1fa4c163365b80b06a43b51eb8d4e",
        "8000": "3a7b54a6f6e313f9d9288a1e177c21dfca434bc42f60cc48d7e7a0c30031ac14",
        "12000": "3cdc518325fd9c02cddb52c4803253b7525dd12dab11ebd5ff151f98d9f8b4b3",
        "16000": "3cdc518325fd9c02cddb52c4803253b7525dd12dab11ebd5ff151f98d9f8b4b3",
    },
    "ssmax_no_qknorm_joint_manifest_v1.json.template": {
        "0": "991eb1f631c99d82b791b52f37ebb9d1fe31a35b342aece9b29eaa0edc0edbe2",
        "4000": "a11bc8da1699c012972b00cb6943e71668ac82e4762090695531bd99dfe5eaf7",
        "8000": "a11bc8da1699c012972b00cb6943e71668ac82e4762090695531bd99dfe5eaf7",
        "12000": "17493e32d50252fc314d6eb7c1cb14576bb45ab37b8280428eafe6af6cc0afba",
        "16000": "17493e32d50252fc314d6eb7c1cb14576bb45ab37b8280428eafe6af6cc0afba",
    },
}


@pytest.mark.parametrize("filename", sorted(_CHECKPOINT_CONFIG_SHA256S))
def test_checked_in_manifest_templates_pin_completed_resume_configs(filename: str) -> None:
    path = (
        Path(__file__).resolve().parents[3]
        / "configs/vision_moe/vision_alignment/eval/joint"
        / filename
    )
    spec = joint._validate_spec(joint.load_json(path))

    assert spec["version"] == joint.MANIFEST_SPEC_VERSION == 2
    assert joint.SCHEMA_VERSION == 2
    assert spec["checkpoint_config_sha256s"] == _CHECKPOINT_CONFIG_SHA256S[filename]
    assert spec["evidence_git"] == {
        "repo": "allenai/OLMo-core",
        "repo_url": "https://github.com/allenai/OLMo-core",
        "ref": "<FILL_WITH_CLEAN_EVIDENCE_COMMIT_SHA>",
    }


@pytest.mark.parametrize(
    ("variant", "expected_refs"),
    [
        (
            "ssmax_head_qknorm",
            [
                "7cc97a77cbfe9a625531653ac6ec64382a56e56c",
                "7cc97a77cbfe9a625531653ac6ec64382a56e56c",
                "e53e8ee6db022366790e5a4ef3a94c62ab50928f",
                "26eebf08c91caf407bdae31fb989c02682946a3c",
                "26eebf08c91caf407bdae31fb989c02682946a3c",
            ],
        ),
        (
            "ssmax_no_qknorm",
            [
                "7cc97a77cbfe9a625531653ac6ec64382a56e56c",
                "7cc97a77cbfe9a625531653ac6ec64382a56e56c",
                "7cc97a77cbfe9a625531653ac6ec64382a56e56c",
                "26eebf08c91caf407bdae31fb989c02682946a3c",
                "26eebf08c91caf407bdae31fb989c02682946a3c",
            ],
        ),
    ],
)
def test_reviewed_resume_schedules_preserve_asymmetric_training_refs(
    variant: str, expected_refs: list[str]
) -> None:
    schedule = joint.TRAINING_RESUME_SCHEDULES[variant]
    assert [schedule[str(step)]["git_ref"] for step in joint.REQUIRED_STEPS] == expected_refs
    assert joint.CROSS_ARM_SCHEDULE_CLASSIFICATION == {
        "schedule": "asymmetric_code_transition",
        "causal_interpretation": "confounded",
        "decision_scope": "descriptive_only",
    }


def _perception_parent_case(
    tmp_path: Path, *, gate_version: Any
) -> tuple[dict[str, Any], dict[str, str]]:
    parent = tmp_path / "step4000"
    parent.mkdir()
    parent_config = {
        "model_variant": "ssmax_head_qknorm",
        "phase": "perception",
        "perception_trainability_arm": "treatment",
        "vision_alignment": {
            "recipe_version": 1,
            "formatter_version": "vision-alignment-document-v1",
            "model_variant": "ssmax_head_qknorm",
            "phase": "perception",
            "data_contract_sha256": "a" * 64,
            "trainable_contract_sha256": "b" * 64,
        },
    }
    config_path = parent / "config.json"
    config_path.write_text(json.dumps(parent_config) + "\n")
    gate_path = tmp_path / "perception-parent-gate.json"
    gate: dict[str, Any] = {"version": gate_version}
    if gate_version == joint.EXPLORATORY_WAIVER_PARENT_GATE_VERSION:
        gate = {field: None for field in joint.exploratory_waiver_perception._GATE_FIELDS}
        gate.update(
            {
                "format": "vision_alignment_parent_gate",
                "version": joint.EXPLORATORY_WAIVER_PARENT_GATE_VERSION,
                "status": "approved",
                "scope": joint.exploratory_waiver_perception.GATE_SCOPE,
                "recipe_version": 1,
                "formatter_version": "vision-alignment-document-v1",
                "phase": "perception",
                "model_variant": "ssmax_head_qknorm",
                "lineage_kind": joint.direct_perception.LINEAGE_KIND,
                "run_id": "issued-v9-test",
                "checkpoint": str(parent),
                "checkpoint_config_sha256": joint.sha256_file(config_path),
                "checkpoint_identity_sha256": "d" * 64,
                "data_contract_sha256": "a" * 64,
                "trainable_contract_sha256": "b" * 64,
                "global_step": 4000,
                "evidence_report_status": joint.exploratory_waiver_perception.REPORT_STATUS,
                "promotion_decision": False,
                "winner_selection": False,
            }
        )
    gate_path.write_text(json.dumps(gate) + "\n")
    return (
        {
            "initialization": {
                "checkpoint": str(parent),
                "parent_config_sha256": joint.sha256_file(config_path),
                "parent_gate_path": str(gate_path),
                "parent_gate_sha256": joint.sha256_file(gate_path),
            }
        },
        joint.artifact_reference(gate_path),
    )


@pytest.mark.parametrize(
    ("gate_version", "expected_validator"),
    [
        (5, "paired"),
        (6, "paired"),
        (7, "direct"),
        (8, "exploratory"),
    ],
)
def test_perception_parent_dispatches_to_exact_versioned_validator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    gate_version: int,
    expected_validator: str,
) -> None:
    config_summary, gate_reference = _perception_parent_case(tmp_path, gate_version=gate_version)
    calls: list[tuple[str, Mapping[str, Any], dict[str, Any]]] = []

    def validate_paired(gate: Mapping[str, Any], **kwargs: Any) -> dict[str, Any]:
        calls.append(("paired", gate, kwargs))
        return {"candidate": {"identity_sha256": "d" * 64}}

    def validate_direct(gate: Mapping[str, Any], **kwargs: Any) -> dict[str, Any]:
        calls.append(("direct", gate, kwargs))
        return {"candidate": {"identity_sha256": "d" * 64}}

    def validate_exploratory(gate: Mapping[str, Any], **kwargs: Any) -> dict[str, Any]:
        calls.append(("exploratory", gate, kwargs))
        return {"candidate": {"identity_sha256": "d" * 64}}

    monkeypatch.setattr(
        joint.perception,
        "validate_ssmax_perception_parent_gate",
        validate_paired,
    )
    monkeypatch.setattr(
        joint.direct_perception,
        "validate_ssmax_perception_direct_parent_gate",
        validate_direct,
    )
    monkeypatch.setattr(
        joint.exploratory_perception,
        "validate_ssmax_perception_exploratory_parent_gate",
        validate_exploratory,
    )
    result = joint._validate_perception_parent(
        config_summary,
        gate_reference=gate_reference,
        model_variant="ssmax_head_qknorm",
        verify_live_checkpoint=False,
    )

    assert len(calls) == 1
    validator, gate, kwargs = calls[0]
    assert validator == expected_validator
    assert gate == {"version": gate_version}
    assert kwargs == {
        "expected_checkpoint": tmp_path / "step4000",
        "expected_checkpoint_config_sha256": config_summary["initialization"][
            "parent_config_sha256"
        ],
        "expected_model_variant": "ssmax_head_qknorm",
        "expected_data_contract_sha256": "a" * 64,
        "expected_trainable_contract_sha256": "b" * 64,
        "verify_live_checkpoint": False,
    }
    assert result == {
        "checkpoint": str(tmp_path / "step4000"),
        "checkpoint_config_sha256": config_summary["initialization"]["parent_config_sha256"],
        "checkpoint_identity_sha256": "d" * 64,
        "data_contract_sha256": "a" * 64,
        "trainable_contract_sha256": "b" * 64,
        "gate": gate_reference,
        "gate_semantic_sha256": joint.canonical_sha256({"version": gate_version}),
    }


def _rewrite_perception_gate(
    config_summary: dict[str, Any], gate_reference: Mapping[str, str], gate: Mapping[str, Any]
) -> dict[str, str]:
    gate_path = Path(gate_reference["path"])
    gate_path.write_text(json.dumps(gate) + "\n")
    reference = joint.artifact_reference(gate_path)
    config_summary["initialization"]["parent_gate_sha256"] = reference["sha256"]
    return reference


def test_perception_parent_consumes_issued_v9_and_rehashes_live_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_summary, gate_reference = _perception_parent_case(tmp_path, gate_version=9)
    parent = tmp_path / "step4000"
    config_sha256 = config_summary["initialization"]["parent_config_sha256"]
    calls: list[tuple[Path, int]] = []

    def unexpected_global_validator(*args: Any, **kwargs: Any) -> Mapping[str, Any]:
        raise AssertionError("the issued v9 consumer must not rebuild historical evidence")

    def checkpoint_identity(checkpoint: Path, *, workers: int) -> dict[str, Any]:
        calls.append((checkpoint, workers))
        return {
            "path": str(parent),
            "global_step": 4000,
            "config_sha256": config_sha256,
            "identity_sha256": "d" * 64,
        }

    monkeypatch.setattr(
        joint.exploratory_waiver_perception,
        "validate_ssmax_perception_exploratory_waiver_parent_gate",
        unexpected_global_validator,
    )
    monkeypatch.setattr(joint.bridge, "checkpoint_identity", checkpoint_identity)

    result = joint._validate_perception_parent(
        config_summary,
        gate_reference=gate_reference,
        model_variant="ssmax_head_qknorm",
        verify_live_checkpoint=True,
        hash_workers=3,
    )

    assert calls == [(parent, 3)]
    assert result["checkpoint_identity_sha256"] == "d" * 64


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("status", "rejected", "status differs"),
        ("promotion_decision", True, "promotion_decision differs"),
        ("unexpected", None, "fields differ"),
    ],
)
def test_issued_v9_parent_rejects_tampered_authorization_claims(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: Any,
    message: str,
) -> None:
    config_summary, gate_reference = _perception_parent_case(tmp_path, gate_version=9)
    gate = dict(joint.load_json(Path(gate_reference["path"])))
    gate[field] = value
    gate_reference = _rewrite_perception_gate(config_summary, gate_reference, gate)
    monkeypatch.setattr(
        joint.bridge,
        "checkpoint_identity",
        lambda checkpoint, *, workers: pytest.fail("tampered gate reached checkpoint hashing"),
    )

    with pytest.raises(joint.SSMaxJointEvidenceError, match=message):
        joint._validate_perception_parent(
            config_summary,
            gate_reference=gate_reference,
            model_variant="ssmax_head_qknorm",
            verify_live_checkpoint=True,
        )


def test_issued_v9_parent_rejects_live_checkpoint_identity_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_summary, gate_reference = _perception_parent_case(tmp_path, gate_version=9)
    parent = tmp_path / "step4000"
    monkeypatch.setattr(
        joint.bridge,
        "checkpoint_identity",
        lambda checkpoint, *, workers: {
            "path": str(parent),
            "global_step": 4000,
            "config_sha256": config_summary["initialization"]["parent_config_sha256"],
            "identity_sha256": "e" * 64,
        },
    )

    with pytest.raises(
        joint.SSMaxJointEvidenceError,
        match="live perception checkpoint identity differs",
    ):
        joint._validate_perception_parent(
            config_summary,
            gate_reference=gate_reference,
            model_variant="ssmax_head_qknorm",
            verify_live_checkpoint=True,
        )


@pytest.mark.parametrize("gate_version", [4, True, 7.0])
def test_perception_parent_rejects_unsupported_or_aliased_gate_version(
    tmp_path: Path, gate_version: Any
) -> None:
    config_summary, gate_reference = _perception_parent_case(tmp_path, gate_version=gate_version)

    with pytest.raises(
        joint.SSMaxJointEvidenceError,
        match="version must be exactly integer 5, 6, 7, 8, or 9",
    ):
        joint._validate_perception_parent(
            config_summary,
            gate_reference=gate_reference,
            model_variant="ssmax_head_qknorm",
            verify_live_checkpoint=False,
        )


def test_spec_rejects_native_population_not_divisible_by_world() -> None:
    spec = _spec()
    spec["evaluation"]["native_holdout_examples"] = 1000

    with pytest.raises(joint.SSMaxJointEvidenceError, match="native holdout examples"):
        joint._validate_spec(spec)


def test_spec_rejects_padded_visual_population_or_eligibility_drift() -> None:
    spec = _spec()
    spec["evaluation"]["examples_per_source"] = 512
    with pytest.raises(joint.SSMaxJointEvidenceError, match="largest common"):
        joint._validate_spec(spec)


@pytest.mark.parametrize(
    ("pairing_seed", "projection_seed"),
    [(95818, 6198), (6198, 6198), (95818, 95818)],
)
def test_spec_rejects_swapped_or_conflated_independent_seeds(
    pairing_seed: int, projection_seed: int
) -> None:
    spec = _spec()
    spec["evaluation"]["pairing_seed"] = pairing_seed
    spec["evaluation"]["single_response_projection_seed"] = projection_seed

    with pytest.raises(joint.SSMaxJointEvidenceError, match="independent fixed contracts"):
        joint._validate_spec(spec)

    spec = _spec()
    spec["evaluation"]["eligible_rows_per_source"]["pixmo_caption"] = 512
    with pytest.raises(joint.SSMaxJointEvidenceError, match="live eligibility"):
        joint._validate_spec(spec)


def _resume_config_case() -> (
    tuple[dict[str, dict[str, Any]], dict[str, dict[str, str]], dict[str, Any]]
):
    spec = _spec()
    configs = {}
    checkpoints = {}
    schedule = joint.TRAINING_RESUME_SCHEDULES[spec["model_variant"]]
    for step in joint.REQUIRED_STEPS:
        key = str(step)
        configs[key] = {
            "payload": {"sequence_length": 8192, "seed": 6198, "typed_value": 1},
            "launch": {
                "name": schedule[key]["launch_name"],
                "git": {
                    "repo": "allenai/OLMo-core",
                    "repo_url": "https://github.com/allenai/OLMo-core",
                    "branch": "rustin/vision-ssmax-molmofication",
                    "ref": schedule[key]["git_ref"],
                },
            },
        }
        checkpoints[key] = {"config_sha256": spec["checkpoint_config_sha256s"][key]}
    return configs, checkpoints, spec


def test_resume_configs_require_exact_raw_pins_and_only_ignore_launch_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configs, checkpoints, spec = _resume_config_case()
    seen = []

    def validate(config: Mapping[str, Any], **_: Any) -> dict[str, Any]:
        seen.append(config["launch"]["name"])
        return {"selected": config["launch"]["name"]}

    monkeypatch.setattr(joint, "_validate_saved_config", validate)

    summary, lineage = joint._validate_resume_config_set(
        configs,
        checkpoints,
        spec=spec,
        profile={"path": "/profile", "sha256": ZERO},
    )

    schedule = joint.TRAINING_RESUME_SCHEDULES[spec["model_variant"]]
    assert seen == [schedule[str(step)]["launch_name"] for step in joint.REQUIRED_STEPS]
    assert summary == {"selected": schedule["16000"]["launch_name"]}
    assert lineage["cross_arm_schedule"] == {
        "schedule": "asymmetric_code_transition",
        "causal_interpretation": "confounded",
        "decision_scope": "descriptive_only",
    }
    assert {
        step: {
            "config_sha256": row["config_sha256"],
            "launch_name": row["launch_name"],
            "git_ref": lineage["steps"][step]["training_git"]["ref"],
        }
        for step, row in schedule.items()
    } == schedule

    checkpoints["8000"]["config_sha256"] = "f" * 64
    with pytest.raises(joint.SSMaxJointEvidenceError, match="reviewed raw SHA-256 pin"):
        joint._validate_resume_config_set(
            configs,
            checkpoints,
            spec=spec,
            profile={"path": "/profile", "sha256": ZERO},
        )


def test_resume_configs_reject_pinned_structural_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configs, checkpoints, spec = _resume_config_case()
    configs["8000"]["payload"]["sequence_length"] = 4096
    monkeypatch.setattr(
        joint,
        "_validate_saved_config",
        lambda config, **kwargs: {"selected": config["launch"]["name"]},
    )

    with pytest.raises(joint.SSMaxJointEvidenceError, match="outside launch.name"):
        joint._validate_resume_config_set(
            configs,
            checkpoints,
            spec=spec,
            profile={"path": "/profile", "sha256": ZERO},
        )


@pytest.mark.parametrize("aliased_value", [True, 1.0])
def test_resume_structural_hash_does_not_alias_json_scalar_types(
    monkeypatch: pytest.MonkeyPatch, aliased_value: Any
) -> None:
    configs, checkpoints, spec = _resume_config_case()
    configs["8000"]["payload"]["typed_value"] = aliased_value
    assert joint._resume_structural_config(configs["0"]) == joint._resume_structural_config(
        configs["8000"]
    )
    monkeypatch.setattr(
        joint,
        "_validate_saved_config",
        lambda config, **kwargs: {"selected": config["launch"]["name"]},
    )

    with pytest.raises(joint.SSMaxJointEvidenceError, match="differ structurally"):
        joint._validate_resume_config_set(
            configs,
            checkpoints,
            spec=spec,
            profile={"path": "/profile", "sha256": ZERO},
        )


@pytest.mark.parametrize("field", ["checkpoint_config_sha256s", "evidence_git"])
def test_manifest_spec_requires_resume_and_evidence_identity(field: str) -> None:
    spec = _spec()
    del spec[field]

    with pytest.raises(joint.SSMaxJointEvidenceError, match="fields differ"):
        joint._validate_spec(spec)


def test_evidence_git_checkout_is_separate_from_training_resume_git(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evidence_git = {
        "repo": "allenai/OLMo-core",
        "repo_url": "https://github.com/allenai/OLMo-core",
        "ref": "e" * 40,
    }
    repository_root = joint._repository_root()
    assert joint._builder_source_path() == repository_root / joint.BUILDER_REPO_RELATIVE_PATH
    recipe = repository_root / "src/scripts/train/Vision-Alignment.py"
    profile = (
        repository_root
        / "configs/vision_moe/vision_alignment/joint/ssmax_head_qknorm_1p4b_cx8_direct_v1.yaml"
    )
    monkeypatch.setattr(joint.bridge, "_validate_repository_checkout", lambda *args, **kwargs: None)

    def blob(
        git: Mapping[str, str], *, repository_root: Path, repo_relative_path: str, name: str
    ) -> bytes:
        return (repository_root / repo_relative_path).read_bytes()

    monkeypatch.setattr(joint.bridge, "_git_blob_bytes", blob)
    live_git, builder = joint._validate_evidence_git_checkout(
        evidence_git, recipe_path=recipe, profile_path=profile
    )

    assert live_git == evidence_git
    assert builder == {
        "repo_relative_path": joint.BUILDER_REPO_RELATIVE_PATH,
        "sha256": joint.sha256_file(joint._builder_source_path()),
        "git_ref": evidence_git["ref"],
    }

    redirected_recipe = tmp_path / "Vision-Alignment.py"
    redirected_profile = tmp_path / "joint.yaml"
    redirected_recipe.write_text("redirected\n")
    redirected_profile.write_text("redirected\n")
    with pytest.raises(joint.SSMaxJointEvidenceError, match="builder repository"):
        joint._validate_evidence_git_checkout(
            evidence_git,
            recipe_path=redirected_recipe,
            profile_path=redirected_profile,
        )

    evidence_git["ref"] = "<FILL_WITH_CLEAN_EVIDENCE_COMMIT_SHA>"
    with pytest.raises(joint.SSMaxJointEvidenceError, match="40-character commit SHA"):
        joint._validate_evidence_git_checkout(
            evidence_git, recipe_path=recipe, profile_path=profile
        )


def test_evaluator_source_reference_is_portable_and_git_blob_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository_root = tmp_path / "repo"
    source = repository_root / joint.EVALUATOR_REPO_RELATIVE_PATH
    source.parent.mkdir(parents=True)
    source.write_text("# immutable evaluator\n")
    evidence_git = {
        "repo": "allenai/OLMo-core",
        "repo_url": "https://github.com/allenai/OLMo-core",
        "ref": "e" * 40,
    }
    checkout_calls: list[tuple[Mapping[str, str], Path]] = []
    monkeypatch.setattr(joint, "_repository_root", lambda: repository_root)
    monkeypatch.setattr(
        joint.bridge,
        "_validate_repository_checkout",
        lambda git, *, repository_root: checkout_calls.append((git, repository_root)),
    )
    monkeypatch.setattr(
        joint.bridge,
        "_git_blob_bytes",
        lambda git, *, repository_root, repo_relative_path, name: source.read_bytes(),
    )

    reference = joint.evaluator_source_reference(source, git_ref=evidence_git["ref"])

    assert reference == {
        "repo_relative_path": joint.EVALUATOR_REPO_RELATIVE_PATH,
        "sha256": joint.sha256_file(source),
        "git_ref": evidence_git["ref"],
    }
    assert "path" not in reference
    assert (
        joint._validate_evaluator_source_reference(reference, evidence_git=evidence_git)
        == reference
    )
    assert checkout_calls == [(evidence_git, repository_root)]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("repo_relative_path", "tmp/gantry/vision_alignment_ssmax_joint.py"),
        ("git_ref", "f" * 40),
    ],
)
def test_evaluator_source_reference_rejects_path_or_ref_drift(field: str, value: str) -> None:
    reference = {
        "repo_relative_path": joint.EVALUATOR_REPO_RELATIVE_PATH,
        "sha256": ZERO,
        "git_ref": "e" * 40,
    }
    reference[field] = value
    with pytest.raises(joint.SSMaxJointEvidenceError, match="path or git ref differs"):
        joint._validate_evaluator_source_reference(
            reference,
            evidence_git={
                "repo": "allenai/OLMo-core",
                "repo_url": "https://github.com/allenai/OLMo-core",
                "ref": "e" * 40,
            },
        )


def test_evaluator_source_reference_rejects_git_blob_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository_root = tmp_path / "repo"
    source = repository_root / joint.EVALUATOR_REPO_RELATIVE_PATH
    source.parent.mkdir(parents=True)
    source.write_text("# live evaluator\n")
    evidence_git = {
        "repo": "allenai/OLMo-core",
        "repo_url": "https://github.com/allenai/OLMo-core",
        "ref": "e" * 40,
    }
    reference = {
        "repo_relative_path": joint.EVALUATOR_REPO_RELATIVE_PATH,
        "sha256": joint.sha256_file(source),
        "git_ref": evidence_git["ref"],
    }
    monkeypatch.setattr(joint, "_repository_root", lambda: repository_root)
    monkeypatch.setattr(joint.bridge, "_validate_repository_checkout", lambda *args, **kwargs: None)
    monkeypatch.setattr(joint.bridge, "_git_blob_bytes", lambda *args, **kwargs: b"drift")

    with pytest.raises(joint.SSMaxJointEvidenceError, match="evidence Git blob"):
        joint._validate_evaluator_source_reference(reference, evidence_git=evidence_git)


def test_joint_attention_probe_uses_projection_pixmo_content_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts.eval import vision_alignment_ssmax_joint as evaluator

    expected_content_ids = {
        source: (f"{index + 1:064x}", f"{index + 101:064x}")
        for index, source in enumerate(joint.VISUAL_SOURCES)
    }

    class Projection:
        def selection(self, source: str, logical_split: str) -> Any:
            assert logical_split == "validation"
            return type(
                "Selection",
                (),
                {"row_image_content_sha256": expected_content_ids[source]},
            )()

    monkeypatch.setattr(
        evaluator,
        "load_joint_visual_projection_manifest",
        lambda *args, **kwargs: Projection(),
    )
    monkeypatch.setattr(
        evaluator,
        "build_selected_joint_dataset",
        lambda projection, tokenizer, token_ids, source, **kwargs: f"dataset:{source}",
    )
    monkeypatch.setattr(
        evaluator,
        "SSMaxSingleResponseDataset",
        lambda dataset, **kwargs: dataset,
    )
    monkeypatch.setattr(
        evaluator,
        "_UnpackedModelInputDataset",
        lambda dataset, *, source: dataset,
    )
    raw_config = {
        "data": {
            "joint_visual_projection_path": "/projection.json",
            "joint_visual_projection_sha256": ZERO,
            "ssmax_single_response_projection": {"seed": 95818},
            "loss_token_weighting": "root_subsegments_root_tokens",
        }
    }
    manifest = {
        "joint_visual_projection": {"path": "/projection.json", "sha256": ZERO},
        "evaluation": {"single_response_projection_seed": 95818},
        "attention_probe": {"path": "/probe.json", "sha256": "a" * 64},
    }

    visual, content_ids = evaluator._load_visual_datasets(
        raw_config, manifest, tokenizer=object(), token_ids=object()
    )

    assert content_ids == expected_content_ids
    captured: dict[str, Any] = {}
    monkeypatch.setattr(
        evaluator,
        "validate_artifact_reference",
        lambda value, *, name: tmp_path / "probe.json",
    )

    def run_attention_probe(train_module: Any, dataset: Any, **kwargs: Any) -> dict[str, Any]:
        captured.update({"train_module": train_module, "dataset": dataset, **kwargs})
        return {"status": "passed"}

    monkeypatch.setattr(evaluator.bridge_runner, "_run_attention_probe", run_attention_probe)
    result = evaluator._run_attention_diagnostics(
        "module",
        visual,
        content_ids,
        manifest=manifest,
        collator="collator",  # type: ignore[arg-type]
        checkpoint_identity={"identity_sha256": "b" * 64},
    )

    assert result == {"status": "passed"}
    assert captured["dataset"] == "dataset:pixmo_caption"
    assert captured["content_ids"] == expected_content_ids["pixmo_caption"]
    assert captured["probe_sha256"] == "a" * 64


def test_joint_evaluator_bridge_runner_dependencies_exist() -> None:
    from scripts.eval import vision_alignment_ssmax_joint as evaluator

    source = ast.parse(Path(evaluator.__file__).read_text())
    dependencies = {
        node.attr
        for node in ast.walk(source)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "bridge_runner"
    }

    assert sorted(name for name in dependencies if not hasattr(evaluator.bridge_runner, name)) == []


def test_manifest_spec_reference_pins_raw_and_canonical_semantic_identity(tmp_path: Path) -> None:
    spec = _spec()
    path = tmp_path / "manifest-spec.json"
    path.write_text(json.dumps(spec, indent=2) + "\n")

    assert joint._manifest_spec_reference(path, spec) == {
        "path": str(path.resolve()),
        "sha256": joint.sha256_file(path),
        "semantic_sha256": joint.canonical_sha256(spec),
    }


def test_finalized_manifest_rebinds_all_spec_derived_fields() -> None:
    spec = _spec()
    manifest = {
        "run_id": spec["run_id"],
        "model_variant": spec["model_variant"],
        "run_name": spec["run_name"],
        "git": dict(spec["evidence_git"]),
        "recipe": {"path": spec["recipe"]},
        "training_profile": {"path": spec["training_profile"]},
        "joint_visual_projection": {"path": spec["joint_visual_projection"]},
        "source_audit": {"path": spec["source_audit"]},
        "attention_probe": {"path": spec["attention_probe"]},
        "perception_parent": {"gate": {"path": spec["perception_parent_gate"]}},
        "pairings": {source: {"path": path} for source, path in spec["pairing_paths"].items()},
        "companion_protocols": {
            "downstream_fast_pair": {"path": spec["companion_protocols"]["downstream_fast_pair"]}
        },
        "evaluation": copy.deepcopy(spec["evaluation"]),
        "topology": copy.deepcopy(spec["topology"]),
        "policy": copy.deepcopy(spec["policy"]),
        "checkpoints": {
            str(step): {
                "path": str(Path(spec["checkpoint_root"]) / f"step{step}"),
                "config_sha256": spec["checkpoint_config_sha256s"][str(step)],
            }
            for step in joint.REQUIRED_STEPS
        },
    }
    joint._validate_finalized_manifest_against_spec(manifest, spec)

    manifest["evaluation"]["native_holdout_examples"] = 976
    with pytest.raises(joint.SSMaxJointEvidenceError, match="evaluation differs"):
        joint._validate_finalized_manifest_against_spec(manifest, spec)


def test_repository_artifact_resolves_portably_from_build_path() -> None:
    recipe = joint._repository_root() / "src/scripts/train/Vision-Alignment.py"
    reference = {
        "path": "/different/build/worktree/src/scripts/train/Vision-Alignment.py",
        "sha256": joint.sha256_file(recipe),
        "repo_relative_path": "src/scripts/train/Vision-Alignment.py",
    }

    assert joint.resolve_repository_artifact(reference, name="recipe") == recipe


def test_native_ce_ppl_receipt_is_recomputed() -> None:
    manifest = {"evaluation": {"native_holdout_examples": 992}}
    result = {
        "examples": 992,
        "tokens": 990,
        "loss_weight": 50.0,
        "summed_ce": 100.0,
        "ce": 2.0,
        "ppl": math.exp(2.0),
        "filtered_examples": 2,
        "dataset_order_sha256": ZERO,
        "row_provenance_sha256": ZERO,
        "native_identity_sha256": ZERO,
        "per_example": [
            {
                "position": position,
                "tokens": 0 if position < 2 else 1,
                "mask_weight": 0.0 if position < 2 else 1.0,
                "loss_weight": 0.0 if position < 2 else 50.0 / 990,
                "summed_ce": 0.0 if position < 2 else 100.0 / 990,
                "filtered": position < 2,
            }
            for position in range(992)
        ],
    }

    assert joint._validate_native_result(result, manifest=manifest) == result
    result["ppl"] += 0.1
    with pytest.raises(joint.SSMaxJointEvidenceError, match="PPL"):
        joint._validate_native_result(result, manifest=manifest)


def test_visual_rows_recompute_each_window_gap() -> None:
    manifest = {"evaluation": {"examples_per_source": 8}}
    rows = []
    for position in range(8):
        correct = {window: 1.0 + position / 100 for window in joint.WINDOWS}
        wrong = {window: value + 0.5 for window, value in correct.items()}
        rows.append(
            {
                "pairing_position": position,
                "recipient_index": position,
                "donor_index": (position + 1) % 8,
                "response_tokens": 40,
                "correct_ce": correct,
                "wrong_ce": wrong,
                "ce_gap_wrong_minus_correct": {window: 0.5 for window in joint.WINDOWS},
            }
        )

    assert (
        len(joint._validate_rows(rows, source="pixmo_caption", manifest=manifest, pairing=None))
        == 8
    )
    rows[0]["ce_gap_wrong_minus_correct"]["first_8"] = 0.4
    with pytest.raises(joint.SSMaxJointEvidenceError, match="inconsistent"):
        joint._validate_rows(rows, source="pixmo_caption", manifest=manifest, pairing=None)


def test_joint_health_reopens_and_validates_all_checkpoint_rank_ledgers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    step = 4000
    world = 16
    loader_state = {"batches_processed": step}
    rank_states = [
        {
            "rank": rank,
            "global_step": step,
            "batches_processed": step,
            "data_loader_state_sha256": joint.canonical_sha256(loader_state),
            "trainer_state_sha256": ZERO,
            "health_ledger": {"rank": rank},
        }
        for rank in range(world)
    ]
    targets = {source: 1 / len(joint.TRAIN_SOURCES) for source in joint.TRAIN_SOURCES}
    counters = {
        "data_errors": 0,
        "optimizer_guard_skips": 2,
        "nonfinite_losses": 0,
        "nonfinite_gradients": 0,
    }
    receipt = {
        "checkpoint": {"path": "/checkpoint"},
        "rank_states": rank_states,
        "sources": {
            source: {
                "examples": 1,
                "tokens": 2,
                "positive_tokens": 1,
                "loss_weight": 1.0,
                "active_loss_weight": 1.0,
                "target_loss_mass": target,
            }
            for source, target in targets.items()
        },
        "run_counters": counters,
        "evidence": {
            "recipe": {"path": "/recipe", "sha256": ZERO},
            "producer": {"path": "/producer", "sha256": ZERO},
        },
    }
    manifest = {
        "model_variant": "ssmax_head_qknorm",
        "run_name": "joint-run",
        "topology": {"world_size": world},
        "loss_mass_targets": targets,
    }
    monkeypatch.setattr(
        Path,
        "glob",
        lambda self, pattern: [self / f"rank{rank}.pt" for rank in range(world)],
    )
    monkeypatch.setattr(joint, "sha256_file", lambda path: ZERO)
    monkeypatch.setattr(
        joint.torch,
        "load",
        lambda path, **kwargs: {"data_loader": loader_state},
    )
    seen = {}

    def extract(states: Any, **kwargs: Any) -> dict[str, Any]:
        seen.update(kwargs)
        assert len(states) == world
        return {
            "rank_ledgers": [{"rank": rank} for rank in range(world)],
            "counters": counters,
        }

    monkeypatch.setattr(joint, "extract_ssmax_health_ledgers", extract)
    monkeypatch.setattr(joint, "validate_artifact_reference", lambda *args, **kwargs: Path("/x"))

    validated = joint._validate_health_receipt(receipt, manifest=manifest, step=step)

    assert len(validated["rank_states"]) == 16
    assert seen["expected_world_size"] == 16
    assert seen["expected_phase"] == "joint"


def _surface(mismatches: int = 0) -> dict[str, Any]:
    return {
        "protocol": "logical-tensor-comparison-sha256-v1",
        "tensor_count": 1,
        "reference_inventory_sha256": ZERO,
        "candidate_inventory_sha256": ZERO if mismatches == 0 else "1" * 64,
        "mismatch_count": mismatches,
    }


def _evaluation(step: int, *, frozen_mismatches: int = 0) -> dict[str, Any]:
    rows = []
    for position in range(8):
        correct = {window: 1.0 + step / 100_000 for window in joint.WINDOWS}
        wrong = {window: value + 0.5 + step / 100_000 for window, value in correct.items()}
        rows.append(
            {
                "pairing_position": position,
                "recipient_index": position,
                "donor_index": (position + 1) % 8,
                "response_tokens": 8,
                "correct_ce": correct,
                "wrong_ce": wrong,
                "ce_gap_wrong_minus_correct": {
                    window: wrong[window] - correct[window] for window in joint.WINDOWS
                },
            }
        )
    native_ce = 2.0 + step / 1_000_000
    native_rows = [
        {
            "position": position,
            "tokens": 1,
            "mask_weight": 1.0,
            "loss_weight": 1.0,
            "summed_ce": native_ce,
            "filtered": False,
        }
        for position in range(8)
    ]
    return {
        "state": {
            "frozen_lexical_input_rows": _surface(frozen_mismatches),
            "frozen_output_projection": _surface(),
        },
        "native": {
            "ce": native_ce,
            "ppl": math.exp(native_ce),
            "native_identity_sha256": ZERO,
            "dataset_order_sha256": ZERO,
            "per_example": native_rows,
        },
        "rows": {source: copy.deepcopy(rows) for source in joint.VISUAL_SOURCES},
        "attention": {"step": step},
    }


def _health(step: int, *, nonfinite: int = 0, optimizer_guard_skips: int = 0) -> dict[str, Any]:
    return {
        "rank_states": [{"rank": 0}],
        "sources": {
            source: {"active_loss_weight": 0.0 if step == 0 else 100.0 / len(joint.TRAIN_SOURCES)}
            for source in joint.TRAIN_SOURCES
        },
        "run_counters": {
            "data_errors": 0,
            "optimizer_guard_skips": optimizer_guard_skips,
            "nonfinite_losses": nonfinite,
            "nonfinite_gradients": 0,
        },
    }


def _report_manifest() -> dict[str, Any]:
    return {
        "run_id": "qk-joint",
        "model_variant": "ssmax_head_qknorm",
        "training_resume_lineage": {
            "cross_arm_schedule": dict(joint.CROSS_ARM_SCHEDULE_CLASSIFICATION)
        },
        "topology": {"world_size": 1},
        "policy": {
            "maximum_data_errors": 0,
            "maximum_optimizer_guard_skips": 0,
            "maximum_nonfinite_losses": 0,
            "maximum_nonfinite_gradients": 0,
            "native_text_ce_max_relative_increase": 0.02,
            "native_text_bootstrap_samples": 10000,
            "native_text_bootstrap_seed": 6198,
        },
        "loss_mass_targets": {
            source: 1 / len(joint.TRAIN_SOURCES) for source in joint.TRAIN_SOURCES
        },
        "companion_protocols": {"downstream_fast_pair": {"path": "/x", "sha256": ZERO}},
        "content_sha256": ZERO,
    }


def test_trajectory_is_descriptive_and_only_hard_invariants_fail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _report_manifest()
    evaluations = {step: _evaluation(step) for step in joint.REQUIRED_STEPS}
    health = {step: _health(step) for step in joint.REQUIRED_STEPS}
    monkeypatch.setattr(joint, "load_manifest", lambda *args, **kwargs: manifest)
    monkeypatch.setattr(
        joint,
        "manifest_reference",
        lambda *args, **kwargs: {"path": "/manifest", "sha256": ZERO, "content_sha256": ZERO},
    )
    monkeypatch.setattr(
        joint,
        "_load_receipt",
        lambda *args, step, expected_format, **kwargs: {
            "status": "passed",
            "step": step,
            "format": expected_format,
        },
    )
    monkeypatch.setattr(
        joint,
        "_validate_evaluation_receipt",
        lambda receipt, *, manifest, step: evaluations[step],
    )
    monkeypatch.setattr(
        joint,
        "_validate_health_receipt",
        lambda receipt, *, manifest, step: health[step],
    )
    monkeypatch.setattr(
        joint,
        "compare_ssmax_attention_reports",
        lambda baseline, candidate: {"baseline": baseline["step"], "candidate": candidate["step"]},
    )
    references = {step: {"path": f"/{step}.json", "sha256": ZERO} for step in joint.REQUIRED_STEPS}

    report = joint.build_trajectory_report(
        manifest_path=tmp_path / "manifest.json",
        evaluation_receipts=references,
        health_receipts=references,
        created_at="2026-08-20T00:00:00+00:00",
    )

    assert report["status"] == "passed_hard_invariants"
    assert report["decision_scope"] == "descriptive_non_promotion"
    assert report["cross_arm_schedule"] == joint.CROSS_ARM_SCHEDULE_CLASSIFICATION
    assert (
        report["trajectory"]["16000"]["visual"]["pixmo_caption"]["first_8"]["retention_vs_step0"]
        > 1
    )
    assert report["trajectory"]["16000"]["native_text"]["ce_change_vs_step0"] > 0
    assert report["attention_trajectory"]["16000"] == {
        "baseline": 0,
        "candidate": 16000,
    }

    health[8000] = _health(8000, nonfinite=1)
    failed = joint.build_trajectory_report(
        manifest_path=tmp_path / "manifest.json",
        evaluation_receipts=references,
        health_receipts=references,
        created_at="2026-08-20T00:00:00+00:00",
    )
    assert failed["status"] == "failed_hard_invariants"

    health[8000] = _health(8000, optimizer_guard_skips=1)
    skipped = joint.build_trajectory_report(
        manifest_path=tmp_path / "manifest.json",
        evaluation_receipts=references,
        health_receipts=references,
        created_at="2026-08-20T00:00:00+00:00",
    )
    assert skipped["status"] == "failed_hard_invariants"
    assert (
        skipped["hard_invariants"]["by_step"]["8000"]["optimizer_guard_skips_within_limit"] is False
    )

    health[8000] = _health(8000)
    regressed_native = evaluations[8000]["native"]
    for row in regressed_native["per_example"]:
        row["summed_ce"] = 2.4
    regressed_native["ce"] = 2.4
    regressed_native["ppl"] = math.exp(2.4)
    native_regression = joint.build_trajectory_report(
        manifest_path=tmp_path / "manifest.json",
        evaluation_receipts=references,
        health_receipts=references,
        created_at="2026-08-20T00:00:00+00:00",
    )
    assert native_regression["status"] == "failed_hard_invariants"
    assert (
        native_regression["hard_invariants"]["by_step"]["8000"]["native_text_ce_noninferior"]
        is False
    )


def test_pair_comparison_directly_compares_attention_at_every_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    left = {field: None for field in joint._REPORT_FIELDS}
    left.update(
        {
            "decision_scope": "descriptive_non_promotion",
            "cross_arm_schedule": dict(joint.CROSS_ARM_SCHEDULE_CLASSIFICATION),
            "run_id": "left",
            "model_variant": "ssmax_head_qknorm",
            "content_sha256": ZERO,
            "hard_invariants": {"passed": True},
            "trajectory": {},
            "paired_visual_rows": {},
            "attention_reports": {},
            "attention_trajectory": {},
        }
    )
    for step in joint.REQUIRED_STEPS:
        evaluation = _evaluation(step)
        left["trajectory"][str(step)] = {
            "visual": {
                source: {
                    window: {
                        "gap_wrong_minus_correct": 0.5,
                        "correct_ce": 1.0,
                        "retention_vs_step0": 1.0,
                    }
                    for window in joint.WINDOWS
                }
                for source in joint.VISUAL_SOURCES
            },
            "native_text": {"ce": 2.0, "ppl": math.exp(2.0)},
        }
        left["paired_visual_rows"][str(step)] = evaluation["rows"]
        left["attention_reports"][str(step)] = {
            "arm": "left",
            "step": step,
        }
        if step:
            left["attention_trajectory"][str(step)] = {"collapse_flags": []}
    right = copy.deepcopy(left)
    right["run_id"] = "right"
    right["model_variant"] = "ssmax_no_qknorm"
    for step in joint.REQUIRED_STEPS:
        right["attention_reports"][str(step)]["arm"] = "right"
        for source in joint.VISUAL_SOURCES:
            for row in right["paired_visual_rows"][str(step)][source]:
                for window in joint.WINDOWS:
                    row["correct_ce"][window] = 1.0
                    row["wrong_ce"][window] = 1.5
                    row["ce_gap_wrong_minus_correct"][window] = 0.5
    calls = []

    def compare_attention(left_report: Any, right_report: Any) -> dict[str, Any]:
        calls.append((left_report["step"], right_report["step"]))
        return {
            "left_arm": left_report["arm"],
            "right_arm": right_report["arm"],
            "step": left_report["step"],
        }

    monkeypatch.setattr(joint, "compare_ssmax_attention_reports", compare_attention)

    comparison = joint.compare_trajectory_reports(left, right)

    assert comparison["winner"] is None
    assert comparison["decision_scope"] == "descriptive_non_promotion"
    assert comparison["cross_arm_schedule"] == joint.CROSS_ARM_SCHEDULE_CLASSIFICATION
    assert comparison["trajectory_deltas"]["16000"]["native_text"]["ce_delta_left_minus_right"] == 0
    adaptation = comparison["trajectory_deltas"]["16000"]["visual"]["pixmo_caption"]["first_8"][
        "paired_intervals"
    ]
    assert adaptation["gap_same_step_left_minus_right"]["mean"] == pytest.approx(0.16)
    assert adaptation["gap_adaptation_did_left_minus_right"]["mean"] == pytest.approx(0.16)
    assert adaptation["gap_adaptation_did_left_minus_right"]["direction"] == (
        "positive_left_minus_right"
    )
    signal = comparison["adaptation_interval_rule"]["signals"][
        "pixmo_caption/first_8/gap_adaptation_did_left_minus_right"
    ]
    assert signal["consistent_direction"] == "positive_left_minus_right"
    assert comparison["trajectory_deltas"]["0"]["attention"] == {
        "baseline": "left",
        "candidate": "right",
        "comparison": {
            "left_arm": "left",
            "right_arm": "right",
            "step": 0,
        },
    }
    assert calls == [(step, step) for step in joint.REQUIRED_STEPS]

    right["cross_arm_schedule"]["causal_interpretation"] = "unconfounded"
    with pytest.raises(joint.SSMaxJointEvidenceError, match="confounded resume schedule"):
        joint.compare_trajectory_reports(left, right)
