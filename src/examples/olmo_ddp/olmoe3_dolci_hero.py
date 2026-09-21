"""Durable, bounded campaign controller and worker entrypoints for the approved run set."""

import copy
import fcntl
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import olmoe3_dolci_hero_plan as p

p.install()
from olmoe3_lr_sweep_watch import atomic_json, log, replace_env, status
from olmoe3_qkgain_control import (
    TRAIN_TEMPLATE,
    CPU_TEMPLATE,
    training_spec,
    cpu_spec,
    control,
    ensure_saved,
    verify_native,
)


def worker_spec(template, r, commit, hosts, mb=4):
    spec = training_spec(template, r, commit, hosts)
    t = spec["tasks"][0]
    t["arguments"] = ["python", p.SCRIPT, "node", r.run_id]
    replace_env(t, dict(GIT_BRANCH=p.BRANCH, QKGAIN_TRAIN_SCRIPT=p.SCRIPT, CAMPAIGN_PT_MB=mb))
    return spec


def service_spec(template, commit, mode):
    spec = cpu_spec(
        template,
        commit,
        "watch" if mode == "watch" else "config",
        template if mode != "watch" else None,
    )
    t = spec["tasks"][0]
    if mode == "watch":
        t["arguments"][-1] = t["arguments"][-1].replace(
            "olmoe3_qkgain_control.py watch", "olmoe3_dolci_hero.py watch"
        )
    else:
        t["arguments"] = ["python", p.SCRIPT, mode]
        # Data consolidation/packing benefits from a Weka-native Rhea allocation.
        t["resources"] = dict(gpuCount=1, cpuCount=16, memory="96 GiB", sharedMemory="8 GiB")
        t["context"].update(minRuntime="2h", priority="urgent", autoResume=True)
        t["timeout"] = "6h"
    replace_env(t, dict(GIT_BRANCH=p.BRANCH))
    return spec


def exports(b, r, commit):
    from olmoe3_qkgain_eval import eval_specs

    specs = eval_specs(b, r, commit)
    for spec in specs.values():
        t = spec["tasks"][0]
        t["arguments"][0] = t["arguments"][0].replace(
            "olmoe3_qkgain_eval.py ", "olmoe3_dolci_hero.py eval "
        )
    return specs


def evaluate():
    import olmoe3_qkgain_eval as e

    r = p.find_run(sys.argv[sys.argv.index("--run") + 1])
    if r.uses_dolci:
        e.SFT_DATA = p.DATA
    sys.argv.pop(1)
    e.main()


def validate():
    """Validate real-image model/data dispatch, with no training or checkpoint mutation."""
    from olmoe3_dolci_prepare import prepare

    prepare()
    p.self_test()
    import olmoe3_dolci_hero_train as train
    from olmo_core.internal.experiment import CliContext, SubCmd
    from olmo_core.optim.scheduler import WSD

    a = train.adapter
    a.hero.qualified.apply_policy()
    # Avoid cross-trial monkeypatch state by validate each config in its own process.
    if os.environ.get("VALIDATE_ONE"):
        r = p.find_run(os.environ["VALIDATE_ONE"])
        train.install_adapters(r)
        os.environ.update(
            QKGAIN_RUN=r.run_id, QKGAIN_START=str(r.start), QKGAIN_STOP=str(r.start + 2)
        )
        c = a.builder(r)(CliContext(p.SCRIPT, SubCmd.dry_run, r.run_id, "ai2/holmes", [])).merge([])
        assert (
            c.data_loader.global_batch_size == r.batch
            and c.train_module.rank_microbatch_size == r.microbatch
        )
        assert c.model.num_active_params == (794233472 if r.split else 787359872)
        assert not c.trainer.callbacks["checkpointer"].save_async
        assert c.trainer.callbacks["checkpointer"].fixed_steps == r.saves
        assert c.train_module.optim.lr == r.lr
        if r.uses_dolci:
            assert c.train_module.optim.weight_decay == 0 and not c.train_module.compile_model
            assert c.trainer.load_path == str(r.source) and not c.trainer.load_optim_state
            assert a.sft_data_plan(r)["total_steps"] == r.end
            if r.kind in ("dolci-emo", "dolci-non-emo") or r.source.exists():
                p.base.validate_checkpoint(r.source, 5961, 16777216, 64)
        for block in [c.model.block, *c.model.block_overrides.values()]:
            router = getattr(block, "routed_experts_router", None)
            assert router is None or router.emo is None
            mixer = block.sequence_mixer
            if hasattr(mixer, "qk_norm_per_head_gains"):
                assert mixer.qk_norm_per_head_gains == r.split
        if r.kind == "hero":
            schedule = c.train_module.scheduler
            assert isinstance(schedule, WSD)
            assert schedule.get_lr(r.lr, p.DECAY_START, r.end) == r.lr
            assert schedule.get_lr(r.lr, r.end, r.end) == 0
            assert "branch_pin" in c.trainer.callbacks
        if r.kind == "hero2t-mt":
            assert c.dataset.source_mixture_config.requested_tokens == 100_000_000_000
            assert c.train_module.scheduler.warmup == 2000
            assert c.train_module.scheduler.alpha_f == 0
        if r.kind == "hero2t-lc":
            assert r.batch == 16777216 and r.lr == 1.1e-4
            assert c.train_module.scheduler.warmup == 2000
        print("CAMPAIGN_CONFIG_VERIFIED", r.as_dict(), flush=True)
        return
    for r in p.runs():
        for mb in ([4, 2] if r.kind in ("hero", "decay") else [4]):
            subprocess.run(
                [sys.executable, p.SCRIPT, "validate"],
                env=dict(os.environ, VALIDATE_ONE=r.run_id, CAMPAIGN_PT_MB=str(mb)),
                check=True,
            )
    # Training and inference templates must agree on completed turns and stop IDs.
    from olmoe3_hero_sft_metadata import inference_template
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(p.DATA / "train/tokenizer", local_files_only=True)
    original = tok.chat_template
    tok.chat_template = inference_template(original)
    assert tok.apply_chat_template(
        [{"role": "user", "content": "2+2?"}], tokenize=False, add_generation_prompt=True
    ).endswith("assistant\n<think>")
    atomic_json(
        p.AUTO / "config-proof.json",
        dict(passed=True, commit=os.environ["GIT_REF"], runs=[r.as_dict() for r in p.runs()]),
    )


def node():
    """In-allocation save/resume qualification; continuation uses the very same state."""
    from beaker import Beaker
    from olmoe3_profile_node import resolve_ready_leader
    from olmoe3_profile_topology import validate_topology
    from olmoe3_hero_decay_runtime import verify_runtime

    verify_runtime()
    r = p.find_run(sys.argv[2])
    rank = int(os.environ.get("BEAKER_REPLICA_RANK", "0"))
    assert int(os.environ.get("BEAKER_REPLICA_COUNT", "1")) == r.nodes
    assert int(os.environ["BEAKER_ASSIGNED_GPU_COUNT"]) == 8
    exp, job = os.environ["BEAKER_EXPERIMENT_ID"], os.environ["BEAKER_JOB_ID"]
    topology = subprocess.check_output(["nvidia-smi", "topo", "-m"], text=True, timeout=30)
    atomic_json(p.ROOT / "topology" / exp / f"{job}.json", validate_topology(topology, 8))
    ready = p.ROOT / "rendezvous" / exp
    atomic_json(ready / f"{job}.json", dict(job=job, rank=rank))
    host = "127.0.0.1"
    if r.nodes > 1:
        with Beaker.from_env(check_for_upgrades=False) as b:
            deadline = time.monotonic() + 900
            while time.monotonic() < deadline:
                leader = resolve_ready_leader(b, b.workload.get(exp), ready, r.nodes)
                if leader:
                    break
                print("WAIT_CURRENT_REPLICAS", rank, flush=True)
                time.sleep(10)
            else:
                raise TimeoutError("Current replica rendezvous")
        _, host = leader
    checkpoints = sorted(
        int(x.name[4:])
        for x in r.root.glob("step*")
        if re.fullmatch(r"step\d+", x.name) and (x / ".metadata.json").exists()
    )
    start = checkpoints[-1] if checkpoints else r.start
    source = r.root / f"step{start}" if checkpoints else r.source
    stops = [r.start + 2, r.start + 4] + ([25] if r.kind == "hero" else []) + [r.end]
    port = 28000 + int(hashlib.sha256(exp.encode()).hexdigest()[:8], 16) % 1000
    for i, stop in enumerate(stops):
        if start >= stop:
            continue
        env = dict(os.environ, QKGAIN_RUN=r.run_id, QKGAIN_START=str(start), QKGAIN_STOP=str(stop))
        if source:
            env["QKGAIN_LOAD"] = str(source)
        else:
            env.pop("QKGAIN_LOAD", None)
        print("CAMPAIGN_TRAIN_SEGMENT", r.run_id, start, stop, flush=True)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "torch.distributed.run",
                f"--nnodes={r.nodes}",
                "--nproc-per-node=8",
                f"--node-rank={rank}",
                "--rdzv-backend=static",
                f"--rdzv-endpoint={host}:{port+i}",
                f"--rdzv-id={exp}-{i}",
                "--rdzv-conf=read_timeout=900",
                "--max-restarts=0",
                p.SCRIPT,
                "train",
                r.run_id,
                "ai2/holmes",
            ],
            env=env,
            check=True,
        )
        source = r.root / f"step{stop}"
        start = stop
        p.base.validate_checkpoint(source, stop, r.batch, r.gpus)
        if stop == 25 and r.kind == "hero" and rank == 0:
            metrics = [
                json.loads(line)
                for line in (r.root / "audit/metrics.jsonl").read_text().splitlines()
            ]
            losses = [x["train/CE loss"] for x in metrics if "train/CE loss" in x]
            assert losses and all(__import__("math").isfinite(x) for x in losses)
            atomic_json(
                r.root / "audit/qualification.json",
                dict(
                    passed=True,
                    gpus=128,
                    microbatch=r.microbatch,
                    accumulation=r.batch // (r.gpus * r.microbatch),
                    step=25,
                    losses=losses,
                    metrics=metrics[-20:],
                ),
            )
    if rank == 0:
        for k in range(r.gpus):
            proof = json.loads((r.root / "audit" / f"restore-{r.start+2}-rank{k}.json").read_text())
            assert proof["passed"] and not proof["fresh_stage"]
        atomic_json(
            r.root / "audit/success.json",
            dict(
                passed=True,
                step=r.end,
                gpus=r.gpus,
                smoke=False,
                checkpoint_metadata_sha256=hashlib.sha256(
                    (r.root / f"step{r.end}/.metadata.json").read_bytes()
                ).hexdigest(),
            ),
        )


def classify_failure(text):
    """Only the explicitly approved bounded recovery cases; never suppress correctness failures."""
    lower = text.lower()
    if any(
        x in lower
        for x in ("nonfinite", "non-finite", "full-state resume", "assertionerror", "nan loss")
    ):
        return "manual"
    if "out of memory" in lower or "outofmemoryerror" in lower:
        return "oom"
    if any(
        x in lower
        for x in (
            "connection reset",
            "connection refused",
            "rendezvous",
            "ncclremoteerror",
            "ncclsystemerror",
            "preempt",
            "unhealthy node",
            "uncorrectable ecc",
            "gpu has fallen off",
        )
    ):
        return "infra"
    return "manual"


def restore_2t():
    """Restore only the two verified original 2T LC endpoints, never their MT initializers."""
    import logging
    from huggingface_hub import HfApi
    import olmoe3_hero_bucket_download as downloader

    logging.basicConfig(level=logging.INFO)
    downloader.SCRATCH = p.AUTO / "sources/old-2t"
    downloader.prepare_scratch()
    api = HfApi()
    assert api.bucket_info(p.base.BUCKET).private
    proofs = {}
    for kind, (prefix, lineage) in p.OLD_2T_SOURCES.items():
        r = p.run(kind)
        restored = downloader.download(api, prefix, 5961, lineage_id=lineage)
        assert restored == r.source
        p.base.validate_checkpoint(restored, 5961, 16777216, 64)
        assert (restored / "config.json").is_file()
        proofs[kind] = dict(
            source=str(restored),
            prefix=prefix,
            lineage=lineage,
            step=5961,
            config_sha256=hashlib.sha256((restored / "config.json").read_bytes()).hexdigest(),
        )
    atomic_json(p.AUTO / "old-2t-source-proof.json", dict(passed=True, sources=proofs))


def restore_2t_spec(template, commit):
    """One urgent I/O worker; no checkpoint payloads enter Beaker results."""
    spec = cpu_spec(template, commit, "download")
    task = spec["tasks"][0]
    task["arguments"][-1] = task["arguments"][-1].replace(
        "olmoe3_qkgain_control.py download", "olmoe3_dolci_hero.py restore-2t"
    )
    task["resources"] = dict(gpuCount=1, cpuCount=8, memory="64 GiB", sharedMemory="8 GiB")
    task["context"].update(minRuntime="1h", priority="urgent", autoResume=True)
    task["timeout"] = "6h"
    replace_env(task, dict(GIT_BRANCH=p.BRANCH))
    return spec


def watch():
    """SFT-first admission, once-only dependencies, final evals, and bounded PT recovery."""
    from beaker import Beaker
    from olmo_checkpoint_uploader.models import Registration
    from olmo_checkpoint_uploader.state import StateStore
    from olmo_checkpoint_uploader.backend import HuggingFaceBucketBackend
    from olmoe3_qkgain_eval import validate_result

    commit = os.environ["GIT_REF"]
    p.AUTO.mkdir(parents=True, exist_ok=True)
    with (p.AUTO / "LOCK").open("a") as lock, Beaker.from_env(check_for_upgrades=False) as b:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        gate = os.environ["DOLCI_HERO_GATE"]
        while status(b.workload.get(gate)) != "STATUS_SUCCEEDED":
            state = status(b.workload.get(gate))
            assert state not in ("STATUS_FAILED", "STATUS_CANCELED"), (
                "Configuration/data gate failed",
                gate,
            )
            log("WAIT_CONFIG_DATA_GATE", id=gate, status=state)
            time.sleep(60)
        c = control(b, commit, p.AUTO)
        ec = control(b, commit, p.AUTO / "evals", p.base.EVAL_WORKSPACE)
        template = b.experiment.get_spec(b.workload.get(TRAIN_TEMPLATE)).to_json()
        # Same audited hardware allowlist; remove absent nodes, never widen silently.
        names = {n.hostname for n in b.node.list()}
        hosts = [h for h in template["tasks"][0]["constraints"]["hostname"] if h in names]
        assert len(hosts) >= 16
        atomic_json(p.AUTO / "hosts.json", hosts)
        store = StateStore(p.base.CONTROL, p.base.STATE)
        HuggingFaceBucketBackend().assert_private(p.base.BUCKET)
        selected = p.runs()
        for r in selected:
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
        atomic_json(
            p.AUTO / "plan.json",
            dict(commit=commit, runs=[r.as_dict() for r in selected], order=list(p.KINDS)),
        )
        previous = None
        cached = {}
        while True:
            rows = {}
            proof = json.loads((p.AUTO / "config-proof.json").read_text())
            assert proof["passed"] and proof["commit"] == commit
            fs = os.statvfs(p.base.MOUNT)
            admit = (
                fs.f_bavail * fs.f_frsize >= 10_000_000_000_000
                and status(b.workload.get(p.base.UPLOADER)) == "STATUS_RUNNING"
            )
            for r in selected:
                try:
                    if not admit:
                        rows[r.kind] = dict(waiting="storage/uploader admission")
                        continue
                    if r.kind == "hero" and not all(rows.get(k, {}).get("id") for k in p.KINDS[:2]):
                        rows[r.kind] = dict(waiting="submit both Dolci SFT jobs first")
                        continue
                    if r.kind == "decay":
                        if not (p.AUTO / "branch-source.json").exists():
                            rows[r.kind] = dict(waiting="protected step6000")
                            continue
                        bp = json.loads((p.AUTO / "branch-source.json").read_text())
                        assert bp["passed"] and bp["source"] == str(p.PIN)
                    if r.kind in p.OLD_2T_SOURCES:
                        hero = p.run("hero")
                        audit = hero.root / "audit/restore-22000-rank0.json"
                        metrics = hero.root / "audit/metrics.jsonl"
                        if not (p.AUTO / "old-2t-sft-admitted.json").exists() and not (
                            rows.get("hero", {}).get("status") == "STATUS_RUNNING"
                            and audit.exists()
                            and metrics.exists()
                            and metrics.stat().st_mtime_ns > audit.stat().st_mtime_ns
                        ):
                            rows[r.kind] = dict(
                                waiting="128-GPU hero resumed before old-2T SFT admission"
                            )
                            continue
                        if not (p.AUTO / "old-2t-sft-admitted.json").exists():
                            atomic_json(
                                p.AUTO / "old-2t-sft-admitted.json",
                                dict(passed=True, hero=rows["hero"]["id"]),
                            )
                        restored, restored_status = ensure_saved(
                            c,
                            p.CAMPAIGN + "-restore-old-2t-lc",
                            lambda: restore_2t_spec(template, commit),
                        )
                        if restored_status != "STATUS_SUCCEEDED":
                            rows[r.kind] = dict(
                                waiting="verified 2T LC restore",
                                restore_status=restored_status,
                                restore_id=restored.experiment.id if restored else None,
                            )
                            continue
                        source_proof = json.loads((p.AUTO / "old-2t-source-proof.json").read_text())
                        assert source_proof["passed"] and source_proof["sources"][r.kind][
                            "source"
                        ] == str(r.source)
                        p.base.validate_checkpoint(r.source, 5961, 16777216, 64)
                    if r.kind in p.PARENT_KINDS:
                        parent = p.run(p.PARENT_KINDS[r.kind])
                        if rows.get(parent.kind, {}).get("status") != "STATUS_SUCCEEDED":
                            rows[r.kind] = dict(waiting="native " + parent.kind)
                            continue
                        verify_native(parent)
                    recovery_path = p.AUTO / "recovery" / f"{r.kind}.json"
                    recovery = (
                        json.loads(recovery_path.read_text())
                        if recovery_path.exists()
                        else dict(attempt=0, mb=4, excluded=[])
                    )
                    if r.kind == "decay":
                        hero_recovery = p.AUTO / "recovery/hero.json"
                        if hero_recovery.exists():
                            recovery["mb"] = json.loads(hero_recovery.read_text())["mb"]
                    name = (
                        r.run_id
                        + "-train"
                        + (f"-retry{recovery['attempt']}" if recovery["attempt"] else "")
                    )
                    w, s = ensure_saved(
                        c,
                        name,
                        lambda: worker_spec(
                            template,
                            r,
                            commit,
                            [h for h in hosts if h not in recovery["excluded"]],
                            recovery["mb"],
                        ),
                    )
                    rows[r.kind] = dict(
                        status=s,
                        id=w.experiment.id if w else None,
                        attempt=recovery["attempt"],
                        microbatch_sequences=(
                            recovery["mb"] if r.stage == "pt" else r.microbatch // r.sequence
                        ),
                    )
                    if (
                        s in ("STATUS_FAILED", "STATUS_CANCELED")
                        and r.kind in ("hero", "decay")
                        and recovery["attempt"] < 3
                        and not (r.root / "STORAGE_PAUSED.json").exists()
                    ):
                        parts = []
                        faulty_hosts = []
                        for task in w.experiment.tasks:
                            job = b.workload.get_latest_job(w, task=task)
                            lines = []
                            for line in b.job.logs(job, tail_lines=160):
                                lines.append(
                                    line.message.decode(errors="replace")
                                    if isinstance(line.message, bytes)
                                    else line.message
                                )
                            text = "\n".join(lines)
                            parts.append(text)
                            if any(
                                marker in text.lower()
                                for marker in (
                                    "uncorrectable ecc",
                                    "gpu has fallen off",
                                    "unhealthy node",
                                )
                            ):
                                from google.protobuf.json_format import MessageToDict

                                assigned = MessageToDict(job).get("assignmentDetails", {})
                                for entry in assigned.get("assignedEnvironmentVariables", []):
                                    if entry["name"] == "BEAKER_NODE_HOSTNAME":
                                        faulty_hosts.append(entry["literal"])
                        why = classify_failure("\n".join(parts))
                        allow = (why == "oom" and recovery["mb"] == 4) or (
                            why == "infra" and recovery["attempt"] < 2
                        )
                        if allow:
                            recovery["excluded"] = sorted(set(recovery["excluded"] + faulty_hosts))
                            assert len(set(hosts) - set(recovery["excluded"])) >= r.nodes
                            recovery.update(
                                attempt=recovery["attempt"] + 1,
                                mb=2 if why == "oom" else recovery["mb"],
                                reason=why,
                                previous=w.experiment.id,
                            )
                            atomic_json(recovery_path, recovery)
                            rows[r.kind]["recovery_queued_next_poll"] = recovery
                        else:
                            rows[r.kind]["needs_attention"] = why
                    if s == "STATUS_SUCCEEDED":
                        verify_native(r)
                except Exception as exc:
                    rows[r.kind] = dict(error=f"{type(exc).__name__}: {exc}")
            # Do not wait for base-model evaluation before releasing native training descendants.
            for r in selected:
                if rows.get(r.kind, {}).get("status") != "STATUS_SUCCEEDED":
                    continue
                try:
                    if r.kind not in cached:
                        cached[r.kind] = exports(b, r, commit)
                    results = {}
                    for kind, spec in cached[r.kind].items():
                        w, s = ensure_saved(ec, r.run_id + "-" + kind, lambda: spec)
                        results[kind] = dict(status=s, id=w.experiment.id if w else None)
                        if s == "STATUS_SUCCEEDED":
                            validate_result(r, kind)
                        if kind == "convert" and s != "STATUS_SUCCEEDED":
                            break
                    rows[r.kind]["evals"] = results
                except Exception as exc:
                    rows[r.kind]["eval_error"] = f"{type(exc).__name__}: {exc}"
            atomic_json(p.AUTO / "status.json", dict(time=time.time(), runs=rows))
            if rows != previous:
                log("DOLCI_HERO_STATUS", runs=rows)
                previous = rows
            time.sleep(60)


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "train":
        from olmoe3_dolci_hero_train import train

        train()
    elif mode == "validate":
        validate()
    elif mode == "node":
        node()
    elif mode == "watch":
        watch()
    elif mode == "restore-2t":
        restore_2t()
    elif mode == "eval":
        evaluate()
    elif mode == "self-test":
        p.self_test()
    else:
        raise ValueError(mode)
