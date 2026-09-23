"""Bounded, durable two-slot SFT queue with per-dataset gates and final eval automation."""

import copy
import fcntl
import json
import os
import subprocess
import time
from types import SimpleNamespace

import olmoe3_corrected_sft_plan as p

p.install()
from olmoe3_lr_sweep_watch import atomic_json, log, replace_env, status
from olmoe3_qkgain_control import (
    CPU_TEMPLATE,
    TRAIN_TEMPLATE,
    control,
    cpu_spec,
    ensure_saved,
    training_spec,
    verify_native,
)


def service_spec(template, commit, mode, kind=None):
    """CPU watcher has no resources; data preparation uses an unallocated Rhea worker."""
    s = cpu_spec(template, commit, "watch" if mode == "watch" else "config", template)
    t = s["tasks"][0]
    replace_env(t, dict(GIT_BRANCH=p.BRANCH))
    if mode == "watch":
        old = "src/examples/olmo_ddp/olmoe3_qkgain_control.py watch"
        assert t["arguments"][2].count(old) == 1
        t["arguments"][2] = t["arguments"][2].replace(old, p.SCRIPT + " watch")
        assert not t.get("resources")
    else:
        assert kind in {r.dataset for r in p.runs()}
        t["arguments"] = ["python", p.SCRIPT, "prepare", kind]
        t["resources"] = dict(gpuCount=1, cpuCount=8, memory="64 GiB", sharedMemory="8 GiB")
        t["context"] = dict(priority="urgent", minRuntime="0s", autoResume=True)
        t["timeout"] = "12h"
    s["description"] = f"{p.CAMPAIGN} {mode} {kind or ''}; no model data in Beaker results"
    return s


def train_spec(template, r, commit, hosts):
    """Allocate the run's node count with the qualified model and runtime."""
    s = training_spec(template, r, commit, hosts)
    t = s["tasks"][0]
    t["arguments"] = ["python", p.SCRIPT, "node", r.run_id]
    replace_env(t, dict(GIT_BRANCH=p.BRANCH, QKGAIN_TRAIN_SCRIPT=None))
    assert t["replicas"] * t["resources"]["gpuCount"] == r.gpus
    return s


def validation_spec(template, r, commit):
    """Check the actual future LC checkpoint/config before allocating training GPUs."""
    s = service_spec(template, commit, "prepare", r.dataset)
    s["tasks"][0]["arguments"] = ["python", p.SCRIPT, "validate", r.run_id]
    s["description"] = f"{r.run_id}: native-parent/config/tokenizer gate; no training"
    return s


def export_spec(b, r, commit, bundle=None, temperature=None):
    """Frozen inference packages and scoring, with explicit corrected-tokenizer checks."""
    import olmoe3_hero_decay_eval as d
    import olmoe3_hero_sft_eval_control as sft

    if bundle is None:
        source = b.experiment.get_spec(b.workload.get(d.TEMPLATES["convert"])).to_json()
        s = d.build_spec(source, "convert", SimpleNamespace(run_id=r.run_id, arm="emo"), commit)
        old = (
            "python /tmp/hero-decay-wrapper/src/examples/olmo_ddp/olmoe3_hero_decay_eval.py "
            "convert --arm emo --source /tmp/hero-conversion-source"
        )
        new = f"python /tmp/hero-decay-wrapper/{p.SCRIPT} convert {r.run_id}"
    else:
        assert bundle in p.BUNDLES and temperature in p.TEMPERATURES
        source = b.experiment.get_spec(b.workload.get(d.TEMPLATES["gen_mc"])).to_json()
        sft.model_path = lambda _: r.hf
        s = sft.spec_for(source, bundle, SimpleNamespace(run_id=r.run_id, arm="emo"), commit)
        old = (
            "python /tmp/hero-sft-evals/src/examples/olmo_ddp/olmoe3_hero_sft_eval.py "
            f"--run {r.run_id} --bundle {bundle}"
        )
        new = f"python /tmp/hero-sft-evals/{p.SCRIPT} eval {r.run_id} {bundle} {temperature}"
    t = s["tasks"][0]
    assert t["arguments"][0].count(old) == 1
    t["arguments"][0] = t["arguments"][0].replace(old, new)
    if not any(x["mountPath"] == "/weka/oe-adapt-default" for x in t["datasets"]):
        t["datasets"].append(
            dict(mountPath="/weka/oe-adapt-default", source=dict(weka="oe-adapt-default"))
        )
    t["context"].update(priority="urgent", minRuntime="8h", autoResume=True)
    t["timeout"] = "24h"
    assert not t["result"].get("path")
    s["description"] = json.dumps(
        dict(
            run=r.as_dict(),
            kind=bundle or "convert",
            temperature=temperature,
            model=str(r.hf),
            new_tokenizer=True,
            numerical_parity="standing waiver; serialized-weight and metadata checks retained",
        )
    )
    subprocess.run(["bash", "-n"], input=t["arguments"][0], text=True, check=True)
    return s


def recorded(c, name):
    """Read only this controller's exact receipt; never infer another run by substring."""
    path = c.automation / "submissions" / f"{name}.json"
    if not path.exists():
        return None
    receipt = json.loads(path.read_text())
    eid = receipt.get("experiment_id")
    return c.beaker.workload.get(eid) if eid else None


def launch():
    """Submit only the restartable controller, which admits the authorized data/training jobs."""
    from beaker import Beaker, BeakerExperimentSpec

    repo = p.SHARED / "corrected-sft-20260921"
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    assert not subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
    with Beaker.from_env(check_for_upgrades=False) as b:
        template = b.experiment.get_spec(b.workload.get(CPU_TEMPLATE)).to_json()
        spec = service_spec(template, commit, "watch")
        BeakerExperimentSpec.from_json(copy.deepcopy(spec))
        c = control(b, commit, repo / "controller-submission")
        w = c.ensure(p.CAMPAIGN + "-watch", spec)
        assert w is not None
        atomic_json(
            repo / "controller.json",
            dict(
                commit=commit,
                experiment=w.experiment.id,
                url="https://beaker.org/ex/" + w.experiment.id,
            ),
        )
        log("CORRECTED_SFT_CONTROLLER_LAUNCHED", id=w.experiment.id, commit=commit)


def watch():
    """Schedule at most two SFT experiments; do not mutate PT or existing eval campaigns."""
    from beaker import Beaker
    from olmo_checkpoint_uploader.backend import HuggingFaceBucketBackend
    from olmo_checkpoint_uploader.models import Registration
    from olmo_checkpoint_uploader.state import StateStore
    from olmoe3_hero_4t_eval_policy import validate_export

    assert p.base.MOUNT.is_mount()
    p.AUTO.mkdir(parents=True, exist_ok=True)
    lock = (p.AUTO / "LOCK").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    commit = os.environ["GIT_REF"]
    with Beaker.from_env(check_for_upgrades=False) as b:
        c = control(b, commit, p.AUTO)
        ec = control(b, commit, p.AUTO / "evals", p.base.EVAL_WORKSPACE)
        template = b.experiment.get_spec(b.workload.get(TRAIN_TEMPLATE)).to_json()
        hosts = template["tasks"][0]["constraints"]["hostname"]
        assert len(hosts) >= max(r.nodes for r in p.runs())
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
        atomic_json(
            p.AUTO / "queue-plan.json",
            dict(
                commit=commit,
                max_sft_in_flight=2,
                temperatures=p.TEMPERATURES,
                bundles=p.BUNDLES,
                runs=[
                    dict(
                        run_id=r.run_id,
                        dataset=r.dataset,
                        milestone=r.milestone,
                        lineage=r.lineage,
                        source=str(r.source),
                        lr=r.lr,
                        epochs=2,
                        gpus=r.gpus,
                        batch=r.batch,
                        emo=False,
                    )
                    for r in p.runs()
                ],
            ),
        )
        prior = None
        while True:
            rows, ready, preparation = {}, {}, {}
            for kind in dict.fromkeys(r.dataset for r in p.runs()):
                manifest = p.INPUT / kind / "manifest.json"
                if (
                    not manifest.exists()
                    or json.loads(manifest.read_text())["status"] != "complete"
                ):
                    preparation[kind] = dict(waiting="corrected tokenization")
                    ready[kind] = False
                    continue
                w, state = ensure_saved(
                    c,
                    p.CAMPAIGN + "-prepare-" + kind,
                    lambda kind=kind: service_spec(template, commit, "prepare", kind),
                )
                preparation[kind] = dict(status=state, id=w.experiment.id if w else None)
                ready[kind] = state == "STATUS_SUCCEEDED"
            live = {}
            for r in p.runs():
                w = recorded(c, r.run_id + "-train")
                if w:
                    live[r.run_id] = w
            active = sum(
                status(w) not in ("STATUS_SUCCEEDED", "STATUS_FAILED", "STATUS_CANCELED")
                for w in live.values()
            )
            fs = os.statvfs(p.base.MOUNT)
            free = fs.f_bavail * fs.f_frsize
            admit = (
                free >= 8_000_000_000_000
                and status(b.workload.get(p.base.UPLOADER)) == "STATUS_RUNNING"
            )
            admission_blocked = False
            for r in p.runs():
                try:
                    w = live.get(r.run_id)
                    if w is None:
                        if not r.parent_ready():
                            rows[r.run_id] = dict(
                                waiting="canonical 2T 3:1 LC checkpoint", source=str(r.source)
                            )
                            continue
                        if admission_blocked or not ready[r.dataset] or active >= 2 or not admit:
                            rows[r.run_id] = dict(
                                waiting="earlier wave/data gate/two-slot capacity/storage"
                            )
                            admission_blocked = True
                            continue
                        proof_path = p.AUTO / "config-proofs" / f"{r.run_id}.json"
                        if r.future_parent:
                            vw, vs = ensure_saved(
                                c,
                                r.run_id + "-validate",
                                lambda r=r: validation_spec(template, r, commit),
                            )
                            if vs != "STATUS_SUCCEEDED":
                                rows[r.run_id] = dict(
                                    waiting="native-parent/config/tokenizer validation",
                                    validation_status=vs,
                                    validation_id=vw.experiment.id if vw else None,
                                )
                                continue
                        proof = json.loads(proof_path.read_text())
                        assert proof["passed"] and proof["run"] == json.loads(
                            json.dumps(r.as_dict())
                        )
                        w, state = ensure_saved(
                            c,
                            r.run_id + "-train",
                            lambda r=r: train_spec(template, r, commit, hosts),
                        )
                        active += 1  # Reserve the slot even after an ambiguous API response.
                        if w is None:
                            rows[r.run_id] = dict(state="ambiguous_submission")
                            admission_blocked = True
                            continue
                    state = status(w)
                    row = rows[r.run_id] = dict(status=state, id=w.experiment.id)
                    if state in ("STATUS_FAILED", "STATUS_CANCELED"):
                        # Never fan out copies of failures; leave explicit diagnosis for the operator.
                        admission_blocked = True
                    if state != "STATUS_SUCCEEDED":
                        continue
                    verify_native(r)
                    cw, cs = ensure_saved(
                        ec, r.run_id + "-convert", lambda r=r: export_spec(b, r, commit)
                    )
                    row["conversion"] = dict(status=cs, id=cw.experiment.id if cw else None)
                    if cs != "STATUS_SUCCEEDED":
                        continue
                    validate_export(r.hf)
                    row["evals"] = {}
                    for temperature in p.TEMPERATURES:
                        for bundle in p.BUNDLES:
                            label = f"{bundle}-t{round(temperature*10):02d}"
                            ew, es = ensure_saved(
                                ec,
                                r.run_id + "-" + label,
                                lambda r=r, bundle=bundle, temperature=temperature: export_spec(
                                    b, r, commit, bundle, temperature
                                ),
                            )
                            row["evals"][label] = dict(
                                status=es, id=ew.experiment.id if ew else None
                            )
                            if es == "STATUS_SUCCEEDED":
                                out = (
                                    r.hf.parent / f"posttrain-t{round(temperature*10):02d}" / bundle
                                )
                                assert json.loads((out / "success.json").read_text())["passed"]
                                assert (out / "overruns.json").is_file()
                    row["uploaded_epochs"] = []
                    for step in r.saves:
                        cp = store.load_checkpoint(r.run_id, step)
                        if cp and cp.remote_verified:
                            row["uploaded_epochs"].append(step)
                except Exception as exc:
                    rows[r.run_id] = dict(
                        state="needs_attention", error=f"{type(exc).__name__}: {exc}"
                    )
                    admission_blocked = True
            snapshot = dict(
                preparation=preparation,
                runs=rows,
                free_bytes=free,
                sft_in_flight=active,
                commit=commit,
            )
            atomic_json(p.AUTO / "status.json", dict(updated_at=time.time(), **snapshot))
            comparison = dict(preparation=preparation, runs=rows)
            if comparison != prior:
                log("CORRECTED_SFT_STATUS", **snapshot)
                prior = comparison
            if all(
                len(row.get("uploaded_epochs", [])) == 2
                and len(row.get("evals", {})) == 12
                and all(x["status"] == "STATUS_SUCCEEDED" for x in row["evals"].values())
                for row in rows.values()
            ):
                log("CORRECTED_SFT_ALL_COMPLETE")
                return
            time.sleep(60)
