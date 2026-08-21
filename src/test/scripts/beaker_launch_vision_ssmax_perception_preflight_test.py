"""Tests for the fixed SSMax perception runtime-preflight launcher."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from gantry.api import GitRepoState

from scripts import beaker_launch_vision_ssmax_perception_preflight as launcher


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _case(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    model_variant: str = "ssmax_head_qknorm",
) -> dict[str, object]:
    repository_root = tmp_path / "repo"
    artifact_root = tmp_path / "artifacts"
    checkpoint_root = tmp_path / "checkpoints"
    monkeypatch.setattr(launcher, "REPOSITORY_ROOT", repository_root)
    monkeypatch.setattr(launcher, "ARTIFACT_ROOT", artifact_root)
    monkeypatch.setattr(launcher, "CHECKPOINT_ROOT", checkpoint_root)

    recipe = repository_root / launcher.RECIPE_PATH
    profile = repository_root / launcher.PROFILE_PATHS[model_variant]
    recipe.parent.mkdir(parents=True)
    profile.parent.mkdir(parents=True)
    artifact_root.mkdir(parents=True)
    checkpoint_root.mkdir(parents=True)
    recipe.write_text("# recipe\n")
    profile.write_text("# profile\n")
    recipe_sha256 = _sha256(recipe)
    profile_sha256 = _sha256(profile)
    git_ref = "1" * 40
    receipt = artifact_root / launcher.PAIR_RECEIPT_NAMES[model_variant]
    payload = {
        "format": launcher._PAIR_FORMAT,
        "version": launcher._PAIR_VERSION,
        "status": "passed",
        "model_variant": model_variant,
        "launch_contract": launcher._LAUNCH_CONTRACT,
        "recipe": {"sha256": recipe_sha256},
        "git": {"branch": launcher.GIT_BRANCH, "ref": git_ref},
        "profiles": {
            arm: {
                "name": name,
                "repository_path": (
                    launcher.PROFILE_PATHS[model_variant] if arm == "treatment" else "control.yaml"
                ),
                "sha256": profile_sha256 if arm == "treatment" else "2" * 64,
            }
            for arm, name in launcher.PROFILE_NAMES[model_variant].items()
        },
        "save_folders": {
            "status": "verified_absent_and_distinct",
            **{
                arm: str(checkpoint_root / name)
                for arm, name in launcher.PROFILE_NAMES[model_variant].items()
            },
        },
    }
    receipt.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
    return {
        "model_variant": model_variant,
        "expected_recipe_sha256": recipe_sha256,
        "expected_profile_sha256": profile_sha256,
        "profile_pair_receipt": receipt,
        "expected_profile_pair_receipt_sha256": _sha256(receipt),
        "expected_git_ref": git_ref,
        "verify_remote_ref": False,
        "git_state": GitRepoState(
            repo="allenai/OLMo-core",
            repo_url="https://github.com/allenai/OLMo-core",
            ref=git_ref,
            branch=None,
            _is_remote=True,
        ),
    }


@pytest.mark.parametrize("model_variant", launcher.MODEL_VARIANTS)
def test_build_launch_config_fixes_exact_operational_and_argument_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    model_variant: str,
) -> None:
    case = _case(tmp_path, monkeypatch, model_variant=model_variant)
    config = launcher.build_launch_config(**case)

    assert config.workspace == "ai2/scaling-ladders"
    assert config.budget == "ai2/oe-other"
    assert config.clusters == ["ai2/holmes"]
    assert config.hostnames is None
    assert config.num_nodes == 2
    assert config.num_gpus == 8
    assert config.torchrun is True
    assert config.priority == "urgent"
    assert config.min_runtime == "8h"
    assert config.allow_dirty is False
    assert config.follow is False
    assert config.shared_filesystem is True
    assert config.pre_setup is None
    assert config.post_setup is None
    assert config.google_credentials_secret is None
    assert config.aws_config_secret is None
    assert config.aws_credentials_secret is None
    assert [(item.name, item.secret) for item in config.env_secrets] == [
        ("BEAKER_TOKEN", "RUSTINS_BEAKER_TOKEN")
    ]
    assert [(item.bucket, item.mount) for item in config.weka_buckets] == [
        ("oe-training-default", "/weka/oe-training-default")
    ]
    assert config.git.ref == case["expected_git_ref"]
    assert config.git.branch == launcher.GIT_BRANCH
    assert config.cmd == [
        launcher.PREFLIGHT_PATH,
        f"--model-variant={model_variant}",
        f"--recipe={launcher.RECIPE_PATH}",
        f"--expected-recipe-sha256={case['expected_recipe_sha256']}",
        f"--profile={launcher.PROFILE_PATHS[model_variant]}",
        f"--expected-profile-sha256={case['expected_profile_sha256']}",
        f"--profile-pair-receipt={case['profile_pair_receipt']}",
        (
            "--expected-profile-pair-receipt-sha256="
            f"{case['expected_profile_pair_receipt_sha256']}"
        ),
        f"--expected-git-ref={case['expected_git_ref']}",
    ]


def test_build_rejects_receipt_drift_and_wrong_canonical_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = _case(tmp_path, monkeypatch)
    receipt = case["profile_pair_receipt"]
    assert isinstance(receipt, Path)
    receipt.write_text('{"drift":true}\n')
    with pytest.raises(launcher.PerceptionPreflightLaunchError, match="SHA-256 differs"):
        launcher.build_launch_config(**case)

    case = _case(tmp_path / "second", monkeypatch)
    case["profile_pair_receipt"] = tmp_path / "wrong-name.json"
    with pytest.raises(launcher.PerceptionPreflightLaunchError, match="must be exactly"):
        launcher.build_launch_config(**case)


def test_build_rejects_receipt_lineage_or_existing_production_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = _case(tmp_path, monkeypatch)
    receipt = case["profile_pair_receipt"]
    assert isinstance(receipt, Path)
    payload = json.loads(receipt.read_text())
    payload["model_variant"] = "ssmax_no_qknorm"
    receipt.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
    case["expected_profile_pair_receipt_sha256"] = _sha256(receipt)
    with pytest.raises(launcher.PerceptionPreflightLaunchError, match="identity"):
        launcher.build_launch_config(**case)

    case = _case(tmp_path / "second", monkeypatch)
    checkpoint_root = launcher.CHECKPOINT_ROOT
    output = checkpoint_root / launcher.PROFILE_NAMES["ssmax_head_qknorm"]["treatment"]
    output.mkdir(parents=True)
    with pytest.raises(launcher.PerceptionPreflightLaunchError, match="absent production path"):
        launcher.build_launch_config(**case)


def test_build_rejects_launch_contract_type_alias(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = _case(tmp_path, monkeypatch)
    receipt = case["profile_pair_receipt"]
    assert isinstance(receipt, Path)
    payload = json.loads(receipt.read_text())
    payload["launch_contract"]["num_nodes"] = 2.0
    receipt.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
    case["expected_profile_pair_receipt_sha256"] = _sha256(receipt)
    with pytest.raises(launcher.PerceptionPreflightLaunchError, match="launch contract"):
        launcher.build_launch_config(**case)


def test_build_rejects_unpushed_reviewed_git_ref(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = _case(tmp_path, monkeypatch)
    case["verify_remote_ref"] = True
    monkeypatch.setattr(
        launcher.subprocess,
        "check_output",
        lambda *args, **kwargs: f"{'2' * 40}\trefs/heads/{launcher.GIT_BRANCH}\n",
    )
    with pytest.raises(launcher.PerceptionPreflightLaunchError, match="exact tip"):
        launcher.build_launch_config(**case)


def test_parser_has_no_arbitrary_worker_argument_tail() -> None:
    with pytest.raises(SystemExit):
        launcher._parser().parse_args(
            [
                "dry_run",
                "--model-variant=ssmax_head_qknorm",
                f"--expected-recipe-sha256={'1' * 64}",
                f"--expected-profile-sha256={'2' * 64}",
                "--profile-pair-receipt=/tmp/pair.json",
                f"--expected-profile-pair-receipt-sha256={'3' * 64}",
                f"--expected-git-ref={'4' * 40}",
                "--",
                "--unreviewed-argument=1",
            ]
        )
