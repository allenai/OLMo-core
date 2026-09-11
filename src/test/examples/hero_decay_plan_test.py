"""CPU safety tests for the bounded hero decay controller (no service mutations)."""

import hashlib
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples/olmo_ddp"))
import olmoe3_hero_decay_control as control
import olmoe3_hero_decay_eval as evals
import olmoe3_hero_decay_plan as plan
from olmoe3_lr_sweep_watch import atomic_json


def checkpoint(root, step=plan.START):
    for name, content in [
        (".metadata.json", "{}"),
        ("model_and_optim/.metadata", "metadata"),
        ("model_and_optim/weights.distcp", "modelbytes"),
    ]:
        p = root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
    for rank in range(64):
        p = root / "train" / f"rank{rank}.pt"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"trainer")
        atomic_json(
            root / "resume_audit" / f"rank{rank}.json",
            dict(step=step, tokens=step * plan.BATCH, rank=rank, gpus=64),
        )


def test_schedule_and_namespace():
    assert plan.START * plan.BATCH == 1_811_939_328_000
    assert plan.END * plan.BATCH == 2_013_265_920_000
    assert plan.END - plan.START == plan.END // 10
    for r in plan.runs():
        assert r.root != r.parent.root and r.run_id != r.parent.run_id
        assert not r.prefix.startswith(r.arm + "/")
        assert r.bucket == r.parent.bucket
        assert plan.find_run(r.run_id) == r
    with pytest.raises(ValueError):
        plan.DecayRun("other")
    with pytest.raises(StopIteration):
        plan.find_run(plan.runs()[0].parent.run_id)


def test_ready_requires_all_rank_audits(tmp_path, monkeypatch):
    r = SimpleNamespace(root=tmp_path / "run", run_id="test")
    monkeypatch.setattr(plan, "CONTROL", tmp_path / "control")
    assert not plan.ready(r, plan.START)
    cp = r.root / f"step{plan.START}"
    checkpoint(cp)
    assert not plan.ready(r, plan.START)
    event = dict(
        event="checkpoint_ready",
        step=plan.START,
        run_id="test",
        lineage_id="test",
        checkpoint_path=str(cp),
        checkpoint_metadata_sha256=hashlib.sha256(b"{}").hexdigest(),
    )
    path = plan.CONTROL / "inbox/test" / f"step-{plan.START:012d}.ready.json"
    atomic_json(path, event)
    assert plan.ready(r, plan.START)
    atomic_json(cp / "resume_audit/rank63.json", dict(step=0, tokens=0, rank=63, gpus=64))
    with pytest.raises(ValueError):
        plan.ready(r, plan.START)


def test_ready_rejects_wrong_event(tmp_path, monkeypatch):
    r = SimpleNamespace(root=tmp_path / "run", run_id="test")
    monkeypatch.setattr(plan, "CONTROL", tmp_path / "control")
    cp = r.root / f"step{plan.START}"
    checkpoint(cp)
    atomic_json(
        plan.CONTROL / "inbox/test" / f"step-{plan.START:012d}.ready.json",
        dict(event="checkpoint_ready", step=plan.START, run_id="another"),
    )
    with pytest.raises(ValueError):
        plan.ready(r, plan.START)


def test_copy_verifies_bytes_and_idempotence(tmp_path, monkeypatch):
    src, dst = tmp_path / "source", tmp_path / "owned/step108000"
    checkpoint(src)
    monkeypatch.setattr(Path, "is_mount", lambda self: True)
    monkeypatch.setattr(shutil, "disk_usage", lambda p: SimpleNamespace(free=30_000_000_000_000))
    row = plan.verified_copy(src, dst, plan.START)
    assert row["all_file_hashes_verified"]
    assert plan.verified_copy(src, dst, plan.START) == row
    for name, expected in row["sha256"].items():
        assert hashlib.sha256((dst / name).read_bytes()).hexdigest() == expected
        assert (dst / name).stat().st_ino != (src / name).stat().st_ino


def test_copy_rejects_links_and_low_space(tmp_path, monkeypatch):
    src, dst = tmp_path / "source", tmp_path / "owned/step108000"
    checkpoint(src)
    monkeypatch.setattr(Path, "is_mount", lambda self: True)
    monkeypatch.setattr(shutil, "disk_usage", lambda p: SimpleNamespace(free=1))
    with pytest.raises(RuntimeError):
        plan.verified_copy(src, dst, plan.START)
    assert not dst.exists()
    (src / "link").symlink_to(src / ".metadata.json")
    with pytest.raises(ValueError):
        plan.inventory(src)


def test_protection_restores_only_parent_policy(tmp_path, monkeypatch):
    r = plan.runs()[0]
    monkeypatch.setattr(control, "AUTOMATION", tmp_path)
    path = tmp_path / "registration.json"
    original = dict(
        run_id=r.parent.run_id,
        lineage_id=r.parent.run_id,
        checkpoint_root=str(r.parent.root),
        bucket_id=r.bucket,
        remote_prefix=r.arm,
        deletion_mode="apply",
        min_local_checkpoints=2,
        delete_grace_seconds=3600,
    )
    atomic_json(path, original)
    changes = []

    def update(lineage, **policy):
        assert lineage == r.parent.run_id
        row = json.loads(path.read_text())
        row.update(policy)
        changes.append(policy)
        atomic_json(path, row)

    store = SimpleNamespace(registration_path=lambda _: path, set_lineage_deletion_policy=update)
    control.protect(store, r)
    assert changes[-1]["deletion_mode"] == "report_only"
    control.protect(store, r)
    assert len(changes) == 1
    original_is_file = Path.is_file
    monkeypatch.setattr(
        Path, "is_file", lambda p: True if p.name.endswith("-copy.json") else original_is_file(p)
    )
    control.protect(store, r)
    assert json.loads(path.read_text()) == original


def test_training_spec_isolation():
    r = plan.runs()[0]
    task = dict(
        envVars=[
            dict(name="GIT_REF", value=plan.BASE_COMMIT),
            dict(name="WANDB_RUN_ID", value="parent"),
            dict(name="WANDB_RESUME", value="must"),
            dict(name="SECRET", secret="reference-only"),
        ],
        resources=dict(gpuCount=8),
        context={},
        result=dict(path="/noop-results"),
    )
    original = dict(version="v2", tasks=[task] * 8)
    spec = control.training_spec(original, r, "a" * 40)
    t = spec["tasks"][0]
    env = {v["name"]: v for v in t["envVars"]}
    assert "WANDB_RUN_ID" not in env and "WANDB_RESUME" not in env
    assert env["SECRET"] == dict(name="SECRET", secret="reference-only")
    assert t["replicas"] * t["resources"]["gpuCount"] == 64
    assert t["context"]["minRuntime"] == "1h"
    assert original["tasks"][0] == task and len(original["tasks"]) == 8


@pytest.mark.parametrize("bundle,gpus", [("gen_mc", 4), ("math", 8), ("code", 8)])
def test_eval_profile_and_exact_target(bundle, gpus):
    command = f"{evals.OLD_HELPER} {evals.OLD_HELPER}\n"
    command += f"python ladders/olmoe3/workloads/hero_full_eval.py {bundle} {evals.OLD_ROOT}/emo/step6000/hf --instances {gpus}"
    if bundle == "code":
        command += "\n# hero_code_sandbox_preflight.py google-cloud-cli-583.0.0"
    template = dict(
        tasks=[
            dict(
                arguments=[command],
                resources=dict(gpuCount=gpus),
                context={},
                envVars=[],
                result=dict(path=""),
            )
        ]
    )
    r = plan.runs()[0]
    spec = evals.build_spec(template, bundle, r, "b" * 40)
    t = spec["tasks"][0]
    assert t["context"]["minRuntime"] == "1h" and t["context"]["priority"] == "urgent"
    assert "--arm emo --source /tmp/hero-ladder" in t["arguments"][0]
    assert plan.HELPER_REF in t["arguments"][0]
    assert "hero_hf_cleanup.py" not in t["arguments"][0]
