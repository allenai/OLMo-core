"""Entrypoints for the twelve-run corrected-tokenizer SFT campaign."""

import hashlib
import json
import os
import re
import subprocess
import sys
import time

import olmoe3_corrected_sft_plan as p
from olmoe3_lr_sweep_watch import atomic_json


def node(r):
    """Reuse qualified rendezvous/runtime; resume only completed local native checkpoints."""
    from beaker import Beaker
    from olmoe3_hero_decay_runtime import verify_runtime
    from olmoe3_profile_node import resolve_ready_leader
    from olmoe3_profile_topology import validate_topology

    verify_runtime()
    rank = int(os.environ.get("BEAKER_REPLICA_RANK", "0"))
    assert int(os.environ["BEAKER_REPLICA_COUNT"]) == 8
    assert int(os.environ["BEAKER_ASSIGNED_GPU_COUNT"]) == 8
    exp, job = os.environ["BEAKER_EXPERIMENT_ID"], os.environ["BEAKER_JOB_ID"]
    topology = subprocess.check_output(["nvidia-smi", "topo", "-m"], text=True, timeout=30)
    atomic_json(p.ROOT / "topology" / exp / f"{job}.json", validate_topology(topology, 8))
    ready = p.ROOT / "rendezvous" / exp
    atomic_json(ready / f"{job}.json", dict(job=job, rank=rank))
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
    start = checkpoints[-1] if checkpoints else 0
    source = r.root / f"step{start}" if checkpoints else r.source
    if checkpoints:
        p.base.validate_checkpoint(source, start, r.batch, r.gpus)
    assert 0 <= start <= r.end
    if start < r.end:
        env = dict(
            os.environ,
            QKGAIN_RUN=r.run_id,
            QKGAIN_START=str(start),
            QKGAIN_STOP=str(r.end),
            QKGAIN_LOAD=str(source),
        )
        port = 28000 + int(hashlib.sha256(exp.encode()).hexdigest()[:8], 16) % 1000
        print("CORRECTED_SFT_TRAIN", r.run_id, start, r.end, flush=True)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--nnodes=8",
                "--nproc-per-node=8",
                f"--node-rank={rank}",
                "--rdzv-backend=static",
                f"--rdzv-endpoint={host}:{port}",
                f"--rdzv-id={exp}",
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
    p.base.validate_checkpoint(r.root / f"step{r.end}", r.end, r.batch, r.gpus)
    if rank == 0:
        atomic_json(
            r.root / "audit/success.json",
            dict(
                passed=True,
                step=r.end,
                gpus=64,
                smoke=False,
                checkpoint_metadata_sha256=hashlib.sha256(
                    (r.root / f"step{r.end}/.metadata.json").read_bytes()
                ).hexdigest(),
            ),
        )


def main():
    """Dispatch in separately pinned workers so data, training, and eval dependencies differ."""
    mode = sys.argv[1]
    if mode == "launch":
        from olmoe3_corrected_sft_control import launch

        return launch()
    if mode == "watch":
        from olmoe3_corrected_sft_control import watch

        return watch()
    if mode == "prepare":
        from olmoe3_corrected_sft_data import prepare

        kind = sys.argv[2]
        prepare(kind)
        for r in p.runs():
            if r.dataset == kind:
                subprocess.run([sys.executable, p.SCRIPT, "validate", r.run_id], check=True)
        return
    r = p.find_run(sys.argv[2])
    if mode == "node":
        return node(r)
    if mode in ("validate", "train"):
        from olmoe3_corrected_sft_train import train, validate

        return (train if mode == "train" else validate)(r)
    if mode in ("convert", "eval"):
        from olmoe3_corrected_sft_eval import convert, evaluate

        if mode == "convert":
            return convert(r)
        return evaluate(r, sys.argv[3], float(sys.argv[4]))
    raise ValueError(mode)


if __name__ == "__main__":
    main()
