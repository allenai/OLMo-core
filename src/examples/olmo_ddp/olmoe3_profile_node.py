"""Resolve Beaker's current replica generation before starting per-GPU workers.

Injected leader hostnames can be stale after a pre-training health-check replacement.
Ready markers are keyed by actual job ID, so an old job cannot satisfy the barrier.
"""

import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path


def profile_plan(variants, modes, explicit_plan=""):
    """Allow timing several variants but capturing only the selected candidate."""
    pairs = (
        [tuple(item.split(":")) for item in explicit_plan.split(",")]
        if explicit_plan
        else [(variant, mode) for variant in variants for mode in modes]
    )
    if not pairs or len(pairs) != len(set(pairs)):
        raise ValueError("Empty or duplicate profile passes would overwrite artifacts")
    for pair in pairs:
        if (
            len(pair) != 2
            or not re.fullmatch(r"[a-z0-9-]+", pair[0])
            or pair[1] not in ("timing", "torch", "nsys")
        ):
            raise ValueError(f"Invalid profile pair: {pair}")
    return pairs


def named_profile_plan(run_name, pairs, repeats=1):
    """Give independent A/A restores unique paths without changing implementation flags."""
    if not 1 <= repeats <= 4:
        raise ValueError("Profile repeats must be in [1,4]")
    multiple_variants = len({variant for variant, _ in pairs}) > 1
    output = []
    for repeat in range(1, repeats + 1):
        prefix = f"{run_name}-repeat{repeat}" if repeats > 1 else run_name
        for variant, mode in pairs:
            name = f"{prefix}-{variant}" if multiple_variants else prefix
            output.append((name, variant, mode))
    return output


def resolve_ready_leader(beaker, workload, ready_dir: Path, expected_nodes: int):
    """Return the current leader only once every current job has published readiness."""
    tasks = workload.experiment.tasks
    if len(tasks) != expected_nodes:
        raise RuntimeError(f"Expected {expected_nodes} tasks, found {len(tasks)}")
    leaders = [t for t in tasks if t.system_details.replica_group_details.is_leader_replica]
    if len(leaders) != 1:
        raise RuntimeError(f"Expected exactly one leader task, found {len(leaders)}")
    leader = None
    for task in tasks:
        job = beaker.workload.get_latest_job(workload, task=task, finalized=False)
        if job is None or job.status.HasField("exited") or job.status.HasField("canceled"):
            return None
        marker = ready_dir / f"{job.id}.json"
        if not marker.is_file():
            return None
        if task.id == leaders[0].id:
            leader = job
    if leader is None or not leader.assignment_details.node_id:
        return None
    hostname = beaker.node.get(leader.assignment_details.node_id).hostname
    return leader.id, hostname


def main():
    """Run isolated distributed agents per variant/pass on the same allocation."""
    from beaker import Beaker

    run_name, cluster = sys.argv[1:]
    workload_id = os.environ["BEAKER_EXPERIMENT_ID"]
    job_id = os.environ["BEAKER_JOB_ID"]
    rank = int(os.environ["BEAKER_REPLICA_RANK"])
    nodes = int(os.environ["BEAKER_REPLICA_COUNT"])
    gpus = int(os.environ["BEAKER_ASSIGNED_GPU_COUNT"])
    if nodes != 8 or gpus != 8:
        raise RuntimeError(f"This profile requires 8x8 GPUs, got {nodes}x{gpus}")
    topology_dir = (
        Path("/weka/olmo-3p5-checkpoints/production-profiling/topology") / workload_id / job_id
    )
    subprocess.run(
        [sys.executable, "src/examples/olmo_ddp/olmoe3_profile_topology.py", str(topology_dir)],
        check=True,
    )
    pairs = profile_plan(
        os.environ.get(
            "OLMOE3_DEEP_PROFILE_VARIANTS",
            os.environ.get("OLMOE3_DEEP_PROFILE_VARIANT", "baseline"),
        ).split(","),
        os.environ.get("OLMOE3_DEEP_PROFILE_PASSES", "nsys,torch").split(","),
        os.environ.get("OLMOE3_DEEP_PROFILE_PLAN", ""),
    )
    if any(mode == "nsys" for _, mode in pairs):
        from olmoe3_nsys_tools import NsysSettings, install_nsys

        settings = NsysSettings.from_env()
        if settings.version != "installed" and any(r // gpus == rank for r in settings.ranks):
            # Install once per selected node, before publishing readiness; never once per GPU.
            os.environ["OLMOE3_NSYS_BINARY"] = str(install_nsys())
    ready_dir = Path("/weka/olmo-3p5-checkpoints/production-profiling/rendezvous") / workload_id
    ready_dir.mkdir(parents=True, exist_ok=True)
    temporary = ready_dir / f"{job_id}.tmp"
    temporary.write_text(json.dumps({"job_id": job_id, "node_rank": rank}))
    temporary.replace(ready_dir / f"{job_id}.json")
    with Beaker.from_env() as beaker:
        deadline = time.monotonic() + 900
        while time.monotonic() < deadline:
            workload = beaker.workload.get(workload_id)
            leader = resolve_ready_leader(beaker, workload, ready_dir, nodes)
            if leader is not None:
                break
            print(f"Node {rank}: waiting for all current Beaker jobs to finish setup", flush=True)
            time.sleep(10)
        else:
            raise TimeoutError("Current Beaker replica generation did not become ready in 15m")
    leader_job, hostname = leader
    port = 29000 + int(hashlib.sha256(workload_id.encode()).hexdigest()[:8], 16) % 1000
    print(
        f"Node {rank}: resolved current leader {leader_job} at {hostname}:{port}; "
        f"injected hostname was {os.environ.get('BEAKER_LEADER_REPLICA_HOSTNAME')}",
        flush=True,
    )
    named_pairs = named_profile_plan(
        run_name, pairs, int(os.environ.get("OLMOE3_DEEP_PROFILE_REPEATS", "1"))
    )
    for index, (name, variant, mode) in enumerate(named_pairs):
        # A separate agent and port avoids retaining rendezvous keys from the previous
        # training process. The same eight nodes are retained for fair timing comparisons.
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            f"--nnodes={nodes}",
            f"--nproc-per-node={gpus}",
            f"--node-rank={rank}",
            "--rdzv-backend=static",
            f"--rdzv-endpoint={hostname}:{port + index}",
            f"--rdzv-id={workload_id}-{index}",
            "--rdzv-conf=read_timeout=900",
            "--max-restarts=0",
            "src/examples/olmo_ddp/olmoe3_profile_worker.py",
            name,
            cluster,
        ]
        env = dict(
            os.environ,
            OLMOE3_DEEP_PROFILE_VARIANT=variant,
            OLMOE3_DEEP_PROFILE_PASSES=mode,
            OLMO_PROFILE_SAFE_NOOP_NVTX="1" if variant == "compile-noop-nvtx" else "0",
            OLMO_PROFILE_RS_SINGLE_PARAM_FAST_PATH="1"
            if variant == "reduce-scatter-single-param"
            else "0",
            OLMO_PROFILE_FP32_GRAD_ADD_VECTORIZE="1" if "-grad-add" in variant else "0",
            OLMO_PROFILE_SWIGLU_PAIRWISE="1" if variant.endswith("-act-pair") else "0",
        )
        print(f"Node {rank}: starting isolated {variant}/{mode} agent", flush=True)
        subprocess.run(command, env=env, check=True)


if __name__ == "__main__":
    main()
