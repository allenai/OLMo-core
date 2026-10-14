"""Durable, single-chain LC -> SFT scheduling, checkpoint protection, and evaluations."""

import fcntl
import hashlib
import json
import math
import os
import time
from pathlib import Path
from types import SimpleNamespace

import olmoe3_14t_lc_sft_plan as p

p.install()
from olmoe3_lr_sweep_watch import atomic_json, log, status
from olmoe3_qkgain_control import (
    control,
    cpu_spec,
    ensure_saved,
    training_spec,
    verify_native,
)

TRAIN_TEMPLATE = "01M33PEVEBAN6RNAG80FFDK8CK"
IB_DOWN = {496, 511, 523, 549, 550}


def pin(source, destination):
    """Protect immutable completed native files through same-filesystem hardlinks."""
    source, destination = Path(source), Path(destination)
    receipt = destination.parent / "protected.json"
    if receipt.exists():
        saved = json.loads(receipt.read_text())
        assert saved["source"] == str(source) and saved["passed"]
        assert (
            hashlib.sha256((destination / ".metadata.json").read_bytes()).hexdigest()
            == saved["metadata_sha256"]
        )
        return True
    if not (source / ".metadata.json").is_file():
        return False
    if not all(
        (source / "resume_audit" / f"rank{x}.json").is_file()
        and (source / "train" / f"rank{x}.pt").is_file()
        for x in range(64)
    ):
        return False
    assert (source / "model_and_optim/.metadata").is_file()
    files, size = 0, 0
    for src in sorted(source.rglob("*")):
        assert not src.is_symlink()
        dest = destination / src.relative_to(source)
        if src.is_dir():
            dest.mkdir(parents=True, exist_ok=True)
        elif src.is_file():
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.exists():
                os.link(src, dest)
            assert os.path.samefile(src, dest)
            files += 1
            size += src.stat().st_size
    atomic_json(
        receipt,
        dict(
            passed=True,
            source=str(source),
            destination=str(destination),
            files=files,
            bytes=size,
            policy="immutable hardlinks; no deletion",
            metadata_sha256=hashlib.sha256(
                (destination / ".metadata.json").read_bytes()
            ).hexdigest(),
        ),
    )
    log("CHECKPOINT_PROTECTED", source=str(source), destination=str(destination))
    return True


def hosts_for(template):
    """Keep the established Holmes allowlist, excluding known IB-down nodes."""
    return [
        h
        for h in template["tasks"][0]["constraints"]["hostname"]
        if int(h.split(".")[0].rsplit("-", 1)[-1]) not in IB_DOWN
    ]


def verify_startup(r):
    """Record real finite updates, all-rank source restores, and the first complete save."""
    proof = p.AUTO / "startup" / (r.stage + ".json")
    if proof.exists():
        return json.loads(proof.read_text())
    metrics = r.root / "audit/metrics.jsonl"
    if not metrics.exists():
        return None
    steps = {}
    with metrics.open() as stream:
        for line in stream:
            if not line.endswith("\n"):
                break
            row = json.loads(line)
            if all(key in row for key in ("train/CE loss", "optim/total grad norm")):
                assert all(
                    math.isfinite(float(row[key]))
                    for key in ("train/CE loss", "optim/total grad norm")
                )
                if row["step"] > 0:
                    steps[row["step"]] = row
            if len(steps) >= 5:
                break
    checkpoint = r.root / "step10"
    if len(steps) < 5 or not (checkpoint / ".metadata.json").exists():
        return None
    if not all((checkpoint / "resume_audit" / f"rank{i}.json").exists() for i in range(64)):
        return None
    p.base.validate_checkpoint(checkpoint, 10, r.batch, r.gpus)
    for rank in range(64):
        restore = json.loads((r.root / "audit" / f"restore-0-rank{rank}.json").read_text())
        assert restore["passed"] and restore["fresh_stage"]
        assert restore["source"] == str(r.source)
    row = dict(
        passed=True,
        run=r.run_id,
        gpus=r.gpus,
        finite_steps=list(steps),
        first_checkpoint=str(checkpoint),
        source=str(r.source),
        metrics=list(steps.values()),
    )
    atomic_json(proof, row)
    log("TRAINING_STARTUP_VERIFIED", run=r.run_id, finite_steps=list(steps))
    return row


def train_spec(template, r, commit):
    """Eight synchronized 8-GPU replicas, preserving batch and resume semantics."""
    s = training_spec(template, r, commit, hosts_for(template))
    t = s["tasks"][0]
    t["arguments"] = ["python", p.SCRIPT, "node", r.run_id]
    t["context"].update(priority="urgent", minRuntime="8h", autoResume=True)
    s["retry"] = dict(allowedTaskRetries=10)
    assert t["replicas"] == 8 and t["resources"]["gpuCount"] == 8
    return s


def config_spec(template, commit, stage=None):
    """Validate on one Holmes GPU before allocating the 64-GPU training group."""
    s = cpu_spec(template, commit, "config", template)
    t = s["tasks"][0]
    t["resources"] = dict(gpuCount=1, cpuCount=8, memory="32 GiB", sharedMemory="2 GiB")
    t["constraints"] = dict(hostname=hosts_for(template))
    t["context"].update(minRuntime="1h", autoResume=True)
    if stage is None:
        t["arguments"] = [
            "bash",
            "-euc",
            (
                f"python {p.SCRIPT} validate {p.run('lc').run_id}\n"
                f"python {p.SCRIPT} validate {p.run('sft').run_id} --future"
            ),
        ]
    else:
        t["arguments"] = ["python", p.SCRIPT, "validate", p.run(stage).run_id]
    s["description"] = p.CAMPAIGN + ": native/config/tokenizer validation on one Holmes GPU"
    return s


def ensure_validation(c, name, make):
    """Retry node-health cancellations, preserving application failures and manual stops."""
    from google.protobuf.json_format import MessageToDict

    w, state = ensure_saved(c, name, make)
    if w is None or state != "STATUS_CANCELED":
        return w, state
    jobs = list(c.beaker.job.list(task=w.experiment.tasks[0], limit=1))
    if not jobs:
        return w, state
    job = jobs[0]
    reason = MessageToDict(job).get("status", {}).get("canceledCode")
    if reason != "CANCELATION_CODE_NODE_UNHEALTHY":
        return w, state
    root = c.automation / "validation-restarts" / name
    receipt = root / (job.id + ".json")
    # Save intent first: a lost API response must not cause duplicate restarts.
    if receipt.exists() or len(list(root.glob("*.json"))) >= 10:
        return w, state
    row = dict(job=job.id, reason=reason, phase="restarting", time=time.time())
    atomic_json(receipt, row)
    restarted = c.beaker.workload.restart_tasks(w)
    atomic_json(receipt, dict(row, phase="restarted"))
    log("VALIDATION_NODE_FAILURE_RECOVERED", name=name, job=job.id)
    return restarted, status(restarted)


def watcher_spec(template, commit):
    """A persistent CPU service survives the interactive session and trainer restarts."""
    s = cpu_spec(template, commit, "watch")
    t = s["tasks"][0]
    t["arguments"][2] = (
        t["arguments"][2]
        .replace("src/examples/olmo_ddp/olmoe3_qkgain_control.py watch", p.SCRIPT + " watch")
        .replace("beaker-py==2.7.2", "beaker-py==2.8.0")
    )
    s["retry"] = dict(allowedTaskRetries=10)
    s["description"] = p.CAMPAIGN + ": automatic native LC -> SFT, endpoint protection and evals"
    assert not t.get("resources")
    return s


def evaluation_specs(b, r, commit):
    """Retarget the existing conversion and scoring workers to this chain only."""
    import olmoe3_qkgain_eval as e

    if r.stage == "sft":
        import olmoe3_corrected_sft_control as c

        result = {"convert": c.export_spec(b, r, commit)}
        for bundle in ("math500", "ifbench", "humaneval", "alpaca"):
            for temperature in (0.6, 0.8, 1.0):
                result[f"{bundle}-t{round(temperature*10):02d}"] = c.export_spec(
                    b, r, commit, bundle, temperature
                )
        proxy = SimpleNamespace(stage="lc", run_id=r.run_id, end=r.end, hf=r.hf, as_dict=r.as_dict)
        result["ruler"] = e.eval_specs(b, proxy, commit)["ruler"]
    else:
        result = e.eval_specs(b, r, commit)
    for s in result.values():
        t = s["tasks"][0]
        t["arguments"][0] = (
            t["arguments"][0]
            .replace("olmoe3_qkgain_eval.py ", f"olmoe3_14t_lc_sft.py base-eval {r.run_id} ")
            .replace(f" --run {r.run_id}", "")
        )
        t["context"].update(priority="urgent", minRuntime="8h", autoResume=True)
    return result


def watch():
    """Admit SFT only after successful LC and a verified, protected native endpoint."""
    from beaker import Beaker
    from olmo_checkpoint_uploader.backend import HuggingFaceBucketBackend
    from olmo_checkpoint_uploader.models import Registration
    from olmo_checkpoint_uploader.state import StateStore

    assert p.base.MOUNT.is_mount()
    p.AUTO.mkdir(parents=True, exist_ok=True)
    lock = (p.AUTO / "LOCK").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    # Controller-only updates must not change an already submitted training specification.
    previous_plan = p.AUTO / "plan.json"
    commit = (
        json.loads(previous_plan.read_text())["commit"]
        if previous_plan.exists()
        else os.environ["GIT_REF"]
    )
    with Beaker.from_env(check_for_upgrades=False) as b:
        c = control(b, commit, p.AUTO)
        ec = control(b, commit, p.AUTO / "evals", p.base.EVAL_WORKSPACE)
        template = b.experiment.get_spec(b.workload.get(TRAIN_TEMPLATE)).to_json()
        HuggingFaceBucketBackend().assert_private(p.base.BUCKET)
        store = StateStore(p.base.CONTROL, p.base.STATE)
        for r in p.runs():
            r.root.mkdir(parents=True, exist_ok=True)
            store.register(
                Registration(
                    run_id=r.run_id,
                    lineage_id=r.run_id,
                    checkpoint_root=str(r.root),
                    bucket_id=r.bucket,
                    remote_prefix=r.prefix,
                    deletion_mode="apply",
                    min_local_checkpoints=2,
                    delete_grace_seconds=3600,
                )
            )
        assert pin(p.ORIGINAL_PARENT, p.PARENT)
        atomic_json(
            p.AUTO / "plan.json",
            dict(
                commit=commit,
                controller_commit=os.environ["GIT_REF"],
                runs=[r.as_dict() for r in p.runs()],
                workspace=p.base.WORKSPACE,
                retries=10,
                min_runtime="8h",
                slack="disabled by user",
            ),
        )
        eval_cache = {}
        previous = None
        while True:
            rows = {}
            try:
                gate, gs = ensure_validation(
                    c, p.CAMPAIGN + "-config-" + commit[:8], lambda: config_spec(template, commit)
                )
                rows["config"] = dict(id=gate.experiment.id if gate else None, status=gs)
                free = os.statvfs(p.base.MOUNT)
                admit = free.f_bavail * free.f_frsize >= 8_000_000_000_000
                admit = admit and status(b.workload.get(p.base.UPLOADER)) == "STATUS_RUNNING"
                lc_done = False
                for r in p.runs():
                    # Protect SFT epoch 1 promptly, before uploader retention advances.
                    if r.stage == "sft":
                        pin(r.root / "step2701", p.PROTECTED / "sft/step2701/olmo-core")
                    if gs != "STATUS_SUCCEEDED":
                        rows[r.stage] = dict(waiting="configuration validation")
                        continue
                    if r.stage == "sft":
                        if not lc_done:
                            rows[r.stage] = dict(waiting="successful LC endpoint")
                            continue
                        sw, ss = ensure_validation(
                            c,
                            r.run_id + "-source-check-holmes-" + commit[:8],
                            lambda: config_spec(template, commit, "sft"),
                        )
                        if ss != "STATUS_SUCCEEDED":
                            rows[r.stage] = dict(
                                waiting="LC -> SFT source validation",
                                validation=ss,
                                id=sw.experiment.id if sw else None,
                                needs_attention=ss in ("STATUS_FAILED", "STATUS_CANCELED"),
                            )
                            continue
                    if not admit:
                        rows[r.stage] = dict(waiting="uploader/storage admission")
                        continue
                    w, state = ensure_saved(
                        c, r.run_id + "-train", lambda r=r: train_spec(template, r, commit)
                    )
                    row = rows[r.stage] = dict(id=w.experiment.id if w else None, status=state)
                    if (r.root / "STOP.json").exists():
                        row["needs_attention"] = json.loads((r.root / "STOP.json").read_text())
                        if w and state not in (
                            "STATUS_SUCCEEDED",
                            "STATUS_FAILED",
                            "STATUS_CANCELED",
                        ):
                            list(b.workload.cancel(w))
                        continue
                    startup = verify_startup(r)
                    if startup:
                        row["startup_verified"] = True
                    if state != "STATUS_SUCCEEDED":
                        continue
                    verify_native(r)
                    assert pin(
                        r.root / f"step{r.end}", p.PROTECTED / f"{r.stage}/step{r.end}/olmo-core"
                    )
                    row["native_complete"] = True
                    if r.stage == "lc":
                        lc_done = True
                # Evaluation is independent of the native LC -> SFT handoff.
                for r in p.runs():
                    row = rows.get(r.stage, {})
                    if not row.get("native_complete"):
                        continue
                    try:
                        if r.stage not in eval_cache:
                            eval_cache[r.stage] = evaluation_specs(b, r, commit)
                        row["evals"] = {}
                        for kind, spec in eval_cache[r.stage].items():
                            ew, es = ensure_saved(ec, r.run_id + "-" + kind, lambda spec=spec: spec)
                            row["evals"][kind] = dict(
                                id=ew.experiment.id if ew else None, status=es
                            )
                            if kind == "convert" and es != "STATUS_SUCCEEDED":
                                break
                    except Exception as exc:
                        row["eval_error"] = f"{type(exc).__name__}: {exc}"
            except Exception as exc:
                rows["controller_error"] = f"{type(exc).__name__}: {exc}"
            atomic_json(
                p.AUTO / "status.json",
                dict(
                    updated_at=time.time(),
                    commit=commit,
                    controller_commit=os.environ["GIT_REF"],
                    runs=rows,
                ),
            )
            if rows != previous:
                log("LC_SFT_STATUS", runs=rows)
                previous = rows
            time.sleep(45)
