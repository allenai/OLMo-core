"""Focused tests for the CPU-only perception promotion evidence producers."""

from __future__ import annotations

import importlib.util
import io
import json
import pickle
from copy import deepcopy
from pathlib import Path

import pytest
import torch

from olmo_core.eval import vision_alignment_perception_promotion as promotion


def _load_script(name: str):
    path = Path(__file__).resolve().parents[2] / "scripts" / "eval" / name
    spec = importlib.util.spec_from_file_location(f"test_{path.stem}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _complete_log(*, skipped: tuple[int, ...] = ()) -> str:
    lines = []
    for step in range(1, promotion.PRIMARY_STEP + 1):
        lines.extend(
            (
                f"[step={step}/4000,epoch=1]",
                "    train/CE loss=1.25",
                f"    optim/step skipped={1 if step in skipped else 0}",
            )
        )
    lines.append("Finalizing successful W&B run...")
    return "\n".join(lines) + "\n"


def test_run_health_log_audit_reconstructs_all_steps_and_exact_skips(tmp_path: Path) -> None:
    module = _load_script("vision_alignment_perception_run_health.py")
    path = tmp_path / "output.log"
    path.write_text(_complete_log(skipped=promotion.EXPECTED_TREATMENT_SKIP_STEPS))
    audit = module._audit_log(path)
    assert audit["metric_step_count"] == 4000
    assert audit["numeric_metric_count"] == 8000
    assert audit["guarded_skip_steps"] == list(promotion.EXPECTED_TREATMENT_SKIP_STEPS)
    assert audit["nonfinite_metric_count"] == 0
    assert audit["every_next_step_finite"] is True


def test_run_health_recovery_requires_a_real_finite_non_guard_metric(tmp_path: Path) -> None:
    module = _load_script("vision_alignment_perception_run_health.py")
    path = tmp_path / "output.log"
    text = _complete_log(skipped=(294,))
    recovery = "[step=295/4000,epoch=1]\n" "    train/CE loss=1.25\n" "    optim/step skipped=0"
    assert recovery in text
    path.write_text(
        text.replace(
            recovery,
            "[step=295/4000,epoch=1]\n    optim/step skipped=0",
            1,
        )
    )
    with pytest.raises(promotion.PromotionValidationError, match="recovery steps"):
        module._audit_log(path)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda text: text.replace("train/CE loss=1.25", "train/CE loss=nan", 1), "clean"),
        (lambda text: text.replace("[step=1/4000,epoch=1]\n", "", 1), "omits"),
        (lambda text: text + "Traceback (most recent call last):\n", "clean"),
    ],
)
def test_run_health_log_audit_fails_closed(tmp_path: Path, mutation, message: str) -> None:
    module = _load_script("vision_alignment_perception_run_health.py")
    path = tmp_path / "output.log"
    path.write_text(mutation(_complete_log()))
    with pytest.raises(promotion.PromotionValidationError, match=message):
        module._audit_log(path)


def test_run_health_json_reference_uses_one_hashed_semantic_buffer(
    tmp_path: Path, monkeypatch
) -> None:
    module = _load_script("vision_alignment_perception_run_health.py")
    path = tmp_path / "evidence.json"
    path.write_text('{"marker":"bytes-currently-on-path"}\n')
    audited_raw = b'{"marker":"audited-buffer"}\n'
    read_count = 0
    original_read_bytes = Path.read_bytes

    def substituted_read_bytes(candidate: Path) -> bytes:
        nonlocal read_count
        if candidate.resolve() == path.resolve():
            read_count += 1
            return audited_raw
        return original_read_bytes(candidate)

    monkeypatch.setattr(Path, "read_bytes", substituted_read_bytes)
    observed_path, payload, reference = module._read_json_reference(path, name="test evidence")
    assert observed_path == path.resolve()
    assert payload == {"marker": "audited-buffer"}
    assert reference == {
        "path": str(path.resolve()),
        "sha256": module.hashlib.sha256(audited_raw).hexdigest(),
    }
    assert read_count == 1


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        (b'{"step":1,"step":2}', "repeats key"),
        (b'{"loss":NaN}', "non-finite"),
    ],
)
def test_run_health_strict_json_rejects_ambiguous_values(raw: bytes, message: str) -> None:
    module = _load_script("vision_alignment_perception_run_health.py")
    with pytest.raises(promotion.PromotionValidationError, match=message):
        module._strict_json_bytes(raw, name="test evidence")


def test_run_health_checkpoint_identity_helper_pins_the_executed_buffer(monkeypatch) -> None:
    module = _load_script("vision_alignment_perception_run_health.py")
    path = Path(module.__file__).resolve().with_name("vision_alignment_perception_matched_wrong.py")
    original_read_bytes = Path.read_bytes
    injected_raw = original_read_bytes(path) + b"\n_RUN_HEALTH_EXACT_BUFFER_SENTINEL = True\n"

    def substituted_read_bytes(candidate: Path) -> bytes:
        if candidate.resolve() == path:
            return injected_raw
        return original_read_bytes(candidate)

    monkeypatch.setattr(Path, "read_bytes", substituted_read_bytes)
    helper, reference = module._load_checkpoint_identity_helper()
    assert helper.__file__ == str(path)
    assert helper._RUN_HEALTH_EXACT_BUFFER_SENTINEL is True
    assert callable(helper._checkpoint_identity)
    assert reference == {
        "path": str(path),
        "sha256": module.hashlib.sha256(injected_raw).hexdigest(),
    }


def test_beaker_snapshot_requires_workspace_and_attested_exit_code() -> None:
    module = _load_script("vision_alignment_perception_run_health.py")
    payload = [
        {
            "id": "experiment-1",
            "workspace": {"fullName": "ai2/molmofication"},
            "tasks": [
                {
                    "jobs": [
                        {"id": "job-0", "status": {"exitCode": 0}},
                        {"id": "job-failed", "status": {"exitCode": 10}},
                    ]
                }
            ],
        }
    ]
    assert "ai2/molmofication" in module._workspace_names(payload)
    module._verify_job_claim(payload, "job-0", exit_code=0)
    module._verify_job_claim(payload, "job-failed", exit_code=10)
    with pytest.raises(promotion.PromotionValidationError, match="exit code"):
        module._verify_job_claim(payload, "job-0", exit_code=1)


def test_beaker_job_claim_cannot_borrow_a_sibling_exit_code() -> None:
    module = _load_script("vision_alignment_perception_run_health.py")
    payload = {
        "id": "experiment",
        "jobs": [
            {"id": "job-0", "status": {"exitCode": 0}},
            {"id": "job-1", "status": {"exitCode": 10}},
        ],
    }
    module._verify_job_claim(payload, "job-0", exit_code=0)
    with pytest.raises(promotion.PromotionValidationError, match="uniquely attest"):
        module._verify_job_claim(payload, "job-0", exit_code=10)


def test_beaker_job_claim_rejects_boolean_expected_code() -> None:
    module = _load_script("vision_alignment_perception_run_health.py")
    payload = {"id": "job-0", "status": {"exitCode": 0}}
    with pytest.raises(promotion.PromotionValidationError, match="malformed"):
        module._verify_job_claim(payload, "job-0", exit_code=False)


def _locked_beaker_job(module, *, arm: str, job_id: str, rank: int, canceled: bool) -> dict:
    experiment_id = promotion.EXPECTED_EXPERIMENT_IDS[arm]
    env = {
        "GIT_REF": module.EXPECTED_GIT_REF,
        "BEAKER_JOB_ID": job_id,
        "BEAKER_WORKLOAD_ID": experiment_id,
        "BEAKER_EXPERIMENT_ID": experiment_id,
        "BEAKER_WORKSPACE_ID": module.WORKSPACE_ID,
        "BEAKER_REPLICA_RANK": str(rank),
        "BEAKER_REPLICA_COUNT": "2",
    }
    if canceled:
        status = {
            "canceledCode": 10,
            "canceledFor": module.CONTROL_PRESTART_FAILURE["reason"],
            "canceled": "2026-08-13T01:46:40Z",
            "finalized": "2026-08-13T01:46:40Z",
        }
    else:
        status = {
            "started": "2026-08-13T01:51:37Z",
            "exited": "2026-08-13T04:47:52Z",
            "finalized": "2026-08-13T04:47:54Z",
            "exitCode": 0,
        }
    return {
        "id": job_id,
        "name": f"train-replica-{rank}",
        "workspace": module.WORKSPACE_ID,
        "status": status,
        "execution": {
            "experiment": experiment_id,
            "workspace": module.WORKSPACE_ID,
            "replicaRank": rank,
            "spec": {
                "command": ["bash", "/gantry/entrypoint.sh"],
                "arguments": list(module.EXPECTED_LAUNCH_ARGUMENTS[arm]),
                "envVars": [{"name": name, "value": value} for name, value in env.items()],
                "replicas": 2,
                "leaderSelection": True,
                "hostNetworking": True,
            },
        },
    }


def _locked_beaker_snapshot(module, arm: str) -> list[dict]:
    jobs = [
        _locked_beaker_job(module, arm=arm, job_id=job_id, rank=rank, canceled=False)
        for rank, job_id in module.EXPECTED_SUCCESSFUL_JOBS[arm].items()
    ]
    if arm == promotion.CONTROL_ARM:
        jobs.append(
            _locked_beaker_job(
                module,
                arm=arm,
                job_id=module.CONTROL_PRESTART_FAILURE["job_id"],
                rank=module.CONTROL_PRESTART_FAILURE["replica_rank"],
                canceled=True,
            )
        )
    return [
        {
            "id": promotion.EXPECTED_EXPERIMENT_IDS[arm],
            "name": module.EXPECTED_EXPERIMENT_NAMES[arm],
            "workspaceRef": {"id": module.WORKSPACE_ID, "fullName": module.WORKSPACE},
            "jobs": jobs,
        }
    ]


@pytest.mark.parametrize("arm", promotion.ARMS)
def test_beaker_snapshot_locks_exact_jobs_workspace_git_ref_and_profile(arm: str) -> None:
    module = _load_script("vision_alignment_perception_run_health.py")
    payload = _locked_beaker_snapshot(module, arm)
    module._verify_locked_experiment_snapshot(payload, arm=arm)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda payload: payload[0]["workspaceRef"].__setitem__("id", "wrong-workspace"),
            "workspace",
        ),
        (
            lambda payload: payload[0]["jobs"][0]["execution"]["spec"]["envVars"][0].__setitem__(
                "value", "wrong-git-ref"
            ),
            "GIT_REF",
        ),
        (
            lambda payload: payload[0]["jobs"][0]["execution"]["spec"]["arguments"].append(
                "--unreviewed-override=true"
            ),
            "profile command",
        ),
        (
            lambda payload: payload[0]["jobs"][0]["execution"].__setitem__("replicaRank", 1),
            "rank",
        ),
        (
            lambda payload: payload[0]["jobs"][0].__setitem__("id", "wrong-job-id"),
            "job inventory",
        ),
    ],
)
def test_beaker_snapshot_rejects_launch_provenance_mutations(mutation, message: str) -> None:
    module = _load_script("vision_alignment_perception_run_health.py")
    payload = deepcopy(_locked_beaker_snapshot(module, promotion.TREATMENT_ARM))
    mutation(payload)
    with pytest.raises(promotion.PromotionValidationError, match=message):
        module._verify_locked_experiment_snapshot(payload, arm=promotion.TREATMENT_ARM)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("replicaRank", True),
        ("replicaRank", 1.0),
        ("replicas", 2.0),
        ("exitCode", False),
        ("exitCode", 0.0),
        ("leaderSelection", 1),
        ("hostNetworking", 1),
    ],
)
def test_beaker_snapshot_rejects_numeric_and_boolean_type_confusion(field: str, value) -> None:
    module = _load_script("vision_alignment_perception_run_health.py")
    payload = deepcopy(_locked_beaker_snapshot(module, promotion.TREATMENT_ARM))
    job = payload[0]["jobs"][1]
    if field == "replicaRank":
        job["execution"][field] = value
    elif field in ("replicas", "leaderSelection", "hostNetworking"):
        job["execution"]["spec"][field] = value
    else:
        job["status"][field] = value
    with pytest.raises(promotion.PromotionValidationError):
        module._verify_locked_experiment_snapshot(payload, arm=promotion.TREATMENT_ARM)


@pytest.mark.parametrize("value", [10.0, True])
def test_control_canceled_code_requires_a_plain_integer(value) -> None:
    module = _load_script("vision_alignment_perception_run_health.py")
    payload = deepcopy(_locked_beaker_snapshot(module, promotion.CONTROL_ARM))
    payload[0]["jobs"][-1]["status"]["canceledCode"] = value
    with pytest.raises(promotion.PromotionValidationError, match="canceledCode=10"):
        module._verify_locked_experiment_snapshot(payload, arm=promotion.CONTROL_ARM)


def test_control_healthcheck_cancellation_is_code10_and_never_started() -> None:
    module = _load_script("vision_alignment_perception_run_health.py")
    payload = _locked_beaker_snapshot(module, promotion.CONTROL_ARM)
    failure = payload[0]["jobs"][-1]
    module._verify_job_claim(
        payload,
        failure["id"],
        exit_code=10,
        canceled=True,
    )
    failure["status"]["started"] = "2026-08-13T01:46:00Z"
    with pytest.raises(promotion.PromotionValidationError, match="no-start"):
        module._verify_locked_experiment_snapshot(payload, arm=promotion.CONTROL_ARM)


@pytest.mark.parametrize("step", [4000.0, True, "4000"])
def test_wandb_summary_step_requires_a_plain_integer(step) -> None:
    module = _load_script("vision_alignment_perception_run_health.py")
    with pytest.raises(promotion.PromotionValidationError, match="step4000"):
        module._audit_summary_value({"_step": step}, expected_run_id="run-id")


def test_safe_trainer_state_loader_rejects_pickle_code(tmp_path: Path) -> None:
    marker = tmp_path / "executed"

    class Payload:
        def __reduce__(self):
            return (marker.write_text, ("unsafe",))

    path = tmp_path / "rank0.pt"
    path.write_bytes(pickle.dumps(Payload()))
    with pytest.raises(promotion.PromotionValidationError, match="safely load"):
        promotion.load_trainer_state(path)
    assert not marker.exists()


def test_run_health_immutable_writer_never_replaces_output(tmp_path: Path) -> None:
    module = _load_script("vision_alignment_perception_run_health.py")
    path = tmp_path / "receipt.json"
    module._write_json_once(path, {"first": True})
    with pytest.raises(FileExistsError, match="overwrite"):
        module._write_json_once(path, {"first": False})
    assert json.loads(path.read_text()) == {"first": True}


def _fake_checkpoint_identity(root: Path) -> dict:
    inventory = [
        {
            "path": "model_and_optim/.metadata",
            "size": 8,
            "sha256": "1" * 64,
        }
    ]
    identity = {
        "root": str(root.resolve()),
        "state_dir": str((root / "model_and_optim").resolve()),
        "config_sha256": "2" * 64,
        "checkpoint_marker_sha256": promotion.sha256_file(root / ".metadata.json"),
        "dcp_metadata_sha256": "1" * 64,
        "state_file_hash_algorithm": "sha256",
        "state_file_inventory_sha256": promotion.canonical_sha256(inventory),
        "state_file_inventory": inventory,
    }
    identity["identity_sha256"] = promotion.canonical_sha256(identity)
    return identity


def test_run_health_permanent_checkpoints_emit_full_stable_identities(tmp_path: Path) -> None:
    module = _load_script("vision_alignment_perception_run_health.py")
    lineage = tmp_path / "lineage"
    expected_identities = {}
    for step in module.PERMANENT_STEPS:
        root = lineage / f"step{step}"
        root.mkdir(parents=True)
        (root / ".metadata.json").write_text('{"ephemeral":false}\n')
        expected_identities[step] = _fake_checkpoint_identity(root)

    calls = []

    class IdentityHelper:
        def _checkpoint_identity(self, root: Path, config_path: Path, *, hash_workers: int):
            step = int(root.name.removeprefix("step"))
            calls.append((root, config_path, hash_workers))
            return expected_identities[step]

    checkpoints = module._permanent_checkpoints(
        lineage / "step4000", identity_helper=IdentityHelper()
    )
    assert [item["step"] for item in checkpoints] == [0, 1000, 2000, 3000, 4000]
    assert all(set(item) == {"step", "identity"} for item in checkpoints)
    assert [item["identity"] for item in checkpoints] == [
        expected_identities[step] for step in module.PERMANENT_STEPS
    ]
    assert calls == [
        (lineage / f"step{step}", lineage / f"step{step}" / "config.json", 8)
        for step in module.PERMANENT_STEPS
    ]


def test_run_health_permanent_marker_semantics_are_bound_to_identity_bytes(
    tmp_path: Path, monkeypatch
) -> None:
    module = _load_script("vision_alignment_perception_run_health.py")
    lineage = tmp_path / "lineage"
    root = lineage / "step0"
    root.mkdir(parents=True)
    marker = root / ".metadata.json"
    marker.write_text('{"ephemeral":false}\n')
    identity = _fake_checkpoint_identity(root)

    class IdentityHelper:
        def _checkpoint_identity(self, root: Path, config_path: Path, *, hash_workers: int):
            return identity

    original_read_bytes = Path.read_bytes

    def substituted_read_bytes(candidate: Path) -> bytes:
        if candidate.resolve() == marker.resolve():
            return b'{"ephemeral":false,"unhashed_semantics":true}\n'
        return original_read_bytes(candidate)

    monkeypatch.setattr(Path, "read_bytes", substituted_read_bytes)
    with pytest.raises(promotion.PromotionValidationError, match="marker bytes differ"):
        module._permanent_checkpoints(lineage / "step4000", identity_helper=IdentityHelper())


def _loader_state(rank: int, *, cursor: int = 4) -> dict:
    return {
        "global_step": 4000,
        "world_size": 16,
        "data_loader": {
            "batches_processed": 4000,
            "epoch": 1,
            "total_data_errors": 0,
            "packing_state": {
                "version": 5,
                "dp_rank": rank,
                "dp_world_size": 16,
                "refs_consumed": cursor,
            },
        },
    }


def test_loss_mass_proves_byte_semantic_cursor_equality() -> None:
    module = _load_script("vision_alignment_perception_loss_mass.py")
    control = [_loader_state(rank) for rank in range(16)]
    treatment = [_loader_state(rank) for rank in range(16)]
    inventory, digest = module._prove_arm_cursor_equality(control, treatment)
    assert len(inventory) == 16
    assert digest == promotion.canonical_sha256(inventory)

    treatment[7] = _loader_state(7, cursor=5)
    with pytest.raises(promotion.PromotionValidationError, match=r"ranks \[7\]"):
        module._prove_arm_cursor_equality(control, treatment)


def test_loss_mass_accumulates_the_runtime_source_fields() -> None:
    module = _load_script("vision_alignment_perception_loss_mass.py")
    stats = module._empty_stats()
    batch = {
        "pack_source_names": [["audited_alignment", "scalar_count"]],
        "example_ids": torch.tensor([[0, 0, 1, 1]]),
        "router_token_mask": torch.tensor([[True, True, True, False]]),
        "loss_masks": torch.tensor([[0.5, 1.0, 2.0, 4.0]]),
        "labels": torch.tensor([[10, -100, 11, 12]]),
    }
    module._accumulate_batch(stats, batch)
    assert stats["audited_alignment"] == {
        "examples": 1.0,
        "tokens": 2.0,
        "positive_tokens": 1.0,
        "loss_weight": 1.5,
        "active_loss_weight": 0.5,
    }
    assert stats["scalar_count"] == {
        "examples": 1.0,
        "tokens": 1.0,
        "positive_tokens": 1.0,
        "loss_weight": 2.0,
        "active_loss_weight": 2.0,
    }


def test_loss_mass_loads_recipe_with_direct_main_identity(tmp_path: Path) -> None:
    module = _load_script("vision_alignment_perception_loss_mass.py")
    recipe = tmp_path / "Vision-Alignment.py"
    recipe.write_text(
        "class ExperimentConfig:\n"
        "    pass\n"
        "def _load_tokenizer(config):\n"
        "    return None\n"
        "def _build_mixture_sources(tokenizer, token_ids, config):\n"
        "    return None\n"
        "if __name__ == '__main__':\n"
        "    raise RuntimeError('CLI guard must not execute')\n"
    )
    loaded = module._load_recipe(recipe, promotion.sha256_file(recipe))
    assert loaded.ExperimentConfig.__module__ == "__main__"


def test_loss_mass_recipe_pin_hashes_the_exact_parsed_buffer(tmp_path: Path, monkeypatch) -> None:
    module = _load_script("vision_alignment_perception_loss_mass.py")
    recipe = tmp_path / "Vision-Alignment.py"
    recipe.write_text(
        "class ExperimentConfig:\n"
        "    pass\n"
        "def _load_tokenizer(config):\n"
        "    return None\n"
        "def _build_mixture_sources(tokenizer, token_ids, config):\n"
        "    return None\n"
        "if __name__ == '__main__':\n"
        "    pass\n"
    )
    expected_sha256 = promotion.sha256_file(recipe)
    unpinned_raw = b"# different bytes returned by the audited read\n" + recipe.read_bytes()
    original_read_bytes = Path.read_bytes

    def substituted_read_bytes(path: Path) -> bytes:
        if path.resolve() == recipe.resolve():
            return unpinned_raw
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", substituted_read_bytes)
    with pytest.raises(promotion.PromotionValidationError, match="explicit pin"):
        module._load_recipe(recipe, expected_sha256)


def _torch_save_bytes(value) -> bytes:
    buffer = io.BytesIO()
    torch.save(value, buffer)
    return buffer.getvalue()


def test_run_health_trainer_state_digest_and_load_share_one_buffer(
    tmp_path: Path, monkeypatch
) -> None:
    module = _load_script("vision_alignment_perception_run_health.py")
    path = tmp_path / "rank0.pt"
    path.write_bytes(_torch_save_bytes({"marker": "bytes-currently-on-path"}))
    audited_raw = _torch_save_bytes({"marker": "audited-buffer"})
    read_count = 0
    original_read_bytes = Path.read_bytes

    def substituted_read_bytes(candidate: Path) -> bytes:
        nonlocal read_count
        if candidate.resolve() == path.resolve():
            read_count += 1
            return audited_raw
        return original_read_bytes(candidate)

    monkeypatch.setattr(Path, "read_bytes", substituted_read_bytes)
    state, digest = module._load_trainer_state_and_sha256(path)
    assert state["marker"] == "audited-buffer"
    assert digest == module.hashlib.sha256(audited_raw).hexdigest()
    assert read_count == 1


def test_run_health_rank_inventory_uses_same_buffer_digest(tmp_path: Path, monkeypatch) -> None:
    module = _load_script("vision_alignment_perception_run_health.py")
    checkpoint = tmp_path / "step4000"
    paths = [checkpoint / "train" / f"rank{rank}.pt" for rank in range(16)]
    monkeypatch.setattr(module, "_exact_rank_state_paths", lambda value: paths)

    def load(path: Path):
        rank = int(path.stem.removeprefix("rank"))
        payload = {
            "global_step": 4000,
            "data_loader": {"batches_processed": 4000, "total_data_errors": 0},
            "callbacks": {"wandb": {"run_id": "run-id" if rank == 0 else None}},
        }
        return payload, f"{rank:064x}"

    monkeypatch.setattr(module, "_load_trainer_state_and_sha256", load)
    inventory, run_id, total_errors = module._rank_states(checkpoint)
    assert run_id == "run-id"
    assert total_errors == 0
    assert [item["sha256"] for item in inventory] == [f"{rank:064x}" for rank in range(16)]


def test_run_health_same_buffer_loader_rejects_pickle_code(tmp_path: Path) -> None:
    module = _load_script("vision_alignment_perception_run_health.py")
    marker = tmp_path / "executed"

    class Payload:
        def __reduce__(self):
            return (marker.write_text, ("unsafe",))

    path = tmp_path / "rank0.pt"
    path.write_bytes(pickle.dumps(Payload()))
    with pytest.raises(promotion.PromotionValidationError, match="safely load"):
        module._load_trainer_state_and_sha256(path)
    assert not marker.exists()


def test_loss_mass_trainer_state_digest_and_load_share_one_buffer(
    tmp_path: Path, monkeypatch
) -> None:
    module = _load_script("vision_alignment_perception_loss_mass.py")
    path = tmp_path / "rank0.pt"
    path.write_bytes(_torch_save_bytes({"marker": "bytes-currently-on-path"}))
    audited_raw = _torch_save_bytes({"marker": "audited-buffer"})
    read_count = 0
    original_read_bytes = Path.read_bytes

    def substituted_read_bytes(candidate: Path) -> bytes:
        nonlocal read_count
        if candidate.resolve() == path.resolve():
            read_count += 1
            return audited_raw
        return original_read_bytes(candidate)

    monkeypatch.setattr(Path, "read_bytes", substituted_read_bytes)
    state, digest = module._load_trainer_state_and_sha256(path)
    assert state["marker"] == "audited-buffer"
    assert digest == module.hashlib.sha256(audited_raw).hexdigest()
    assert read_count == 1


def test_loss_mass_rank_inventory_uses_same_buffer_loader(tmp_path: Path, monkeypatch) -> None:
    module = _load_script("vision_alignment_perception_loss_mass.py")
    checkpoint = tmp_path / "step4000"
    paths = [checkpoint / "train" / f"rank{rank}.pt" for rank in range(16)]
    monkeypatch.setattr(module, "_exact_rank_state_paths", lambda value: paths)
    observed: list[Path] = []

    def load(path: Path):
        observed.append(path)
        rank = int(path.stem.removeprefix("rank"))
        return _loader_state(rank), f"{rank:064x}"

    monkeypatch.setattr(module, "_load_trainer_state_and_sha256", load)
    states, inventory = module._rank_states(checkpoint)
    assert observed == paths
    assert states == [_loader_state(rank) for rank in range(16)]
    assert inventory == [
        {"rank": rank, "path": str(path.resolve()), "sha256": f"{rank:064x}"}
        for rank, path in enumerate(paths)
    ]


def test_loss_mass_same_buffer_loader_rejects_pickle_code(tmp_path: Path) -> None:
    module = _load_script("vision_alignment_perception_loss_mass.py")
    marker = tmp_path / "executed"

    class Payload:
        def __reduce__(self):
            return (marker.write_text, ("unsafe",))

    path = tmp_path / "rank0.pt"
    path.write_bytes(pickle.dumps(Payload()))
    with pytest.raises(promotion.PromotionValidationError, match="safely load"):
        module._load_trainer_state_and_sha256(path)
    assert not marker.exists()


def test_loss_mass_immutable_writer_never_replaces_output(tmp_path: Path) -> None:
    module = _load_script("vision_alignment_perception_loss_mass.py")
    path = tmp_path / "receipt.json"
    module._write_json_once(path, {"first": True})
    with pytest.raises(FileExistsError, match="overwrite"):
        module._write_json_once(path, {"first": False})
    assert json.loads(path.read_text()) == {"first": True}


@pytest.mark.parametrize(
    "script_name",
    [
        "vision_alignment_perception_run_health.py",
        "vision_alignment_perception_loss_mass.py",
    ],
)
def test_rank_state_paths_are_exact_checkpoint_train_files(
    tmp_path: Path, script_name: str
) -> None:
    module = _load_script(script_name)
    checkpoint = tmp_path / "step4000"
    train = checkpoint / "train"
    train.mkdir(parents=True)
    expected = []
    for rank in range(16):
        path = train / f"rank{rank}.pt"
        path.write_bytes(b"trainer-state")
        expected.append(path)
    assert module._exact_rank_state_paths(checkpoint) == expected

    extra = train / "rank16.pt"
    extra.write_bytes(b"unexpected")
    with pytest.raises(promotion.PromotionValidationError, match="exact checkpoint/train"):
        module._exact_rank_state_paths(checkpoint)
    extra.unlink()

    outside = tmp_path / "outside-rank15.pt"
    outside.write_bytes(b"trainer-state")
    expected[-1].unlink()
    expected[-1].symlink_to(outside)
    with pytest.raises(promotion.PromotionValidationError, match="exact checkpoint/train"):
        module._exact_rank_state_paths(checkpoint)
