"""
Model-scale ladder for the FLOP-scaling study (Prasann 2026-09-02 evening: "scale models up within
family"): the same recipe as the Qwen3.5-4B grid at 0.8B / 2B / 9B on oolong + contradiction,
with the arm that worked per task (oolong: KV keep 1/6 gold-blind; contradiction: KV gold + 1/3)
plus dense and the two-sided routed-FFN arm, at two budgets per task.

One orchestrator process per scale (state file per scale); reuses orchestrate35's Beaker
plumbing (status/eval launch/harvest/collect) and launch_grid35's per-arm trainer arguments.

    FS_SCALE=0.8b setsid nohup python debug/flop_scaling/orchestrate_scale.py >> debug/flop_scaling/orchestrate_s08b.log 2>&1 &
"""

from __future__ import annotations

import json
import os
import sys
import time

REPO = "/accounts/projects/berkeleynlp/prasann/projects/OLMo-core"
D = f"{REPO}/debug/flop_scaling"
sys.path.insert(0, D)
import launch_grid35 as lg  # noqa: E402
import orchestrate35 as o35  # noqa: E402

SCALE = os.environ["FS_SCALE"]            # 0.8b | 2b | 9b | 27b
FAMILY = os.environ.get("FS_FAMILY", "qwen3_5")  # qwen3_5 (GDN hybrid) | qwen3 (dense; arms re-tokenized, see build_task_rungs TASKSCALE_*)
TAG = ("q3" if FAMILY == "qwen3" else "") + "s" + SCALE.replace(".", "") + os.environ.get("FS_TAG_SUFFIX", "")  # s08b, s2b, s9b; suffix = new state file
W = "/weka/oe-training-default/ai2-llm/checkpoints/prasanns"
BASES = {"0.8b": f"{W}/ctc_suite/bases/q35-08b-base-markerfix/model_and_optim",
         "2b": f"{W}/ctc_suite/bases/q35-2b-base-markerfix/model_and_optim",
         "4b": f"{W}/ctc_suite/bases/q35-4b-base-markerfix/model_and_optim",
         "9b": f"{W}/ctc_suite/bases/q35-9b-base-markerfix/model_and_optim",
         "27b": f"{W}/ctc_suite/bases/q35-27b-base-markerfix/model_and_optim"}
N_LAYERS = {"0.8b": 24, "2b": 24, "4b": 32, "9b": 32, "27b": 64}
FFN_H = {"0.8b": 3584, "2b": 6144, "4b": 9216, "9b": 12288, "27b": 17408}   # intermediate width per scale
if FAMILY == "qwen3":  # dense Qwen3: every layer is attention+FFN; marker-repaired base from the q3-vs-q35 study
    BASES = {"4b": f"{W}/ctc_suite/bases/qwen3-4b-base-trainedmark/model_and_optim"}
    N_LAYERS = {"4b": 36}
    FFN_H = {"4b": 9728}
TRAINABLE_W = {s: h // 16 for s, h in FFN_H.items()}                          # "train what you route to": H/16 prefix
GPUS = {"0.8b": 4, "2b": 4, "4b": 4, "9b": 8, "27b": 8}
if FAMILY == "qwen3":
    # 36 attention layers on the flex path OOM a 4x80GB node at 65k (flexa-c40 / flex-c45, 2026-09-05);
    # 8 ranks halve the FSDP param/optimizer shard per GPU.
    GPUS = {"4b": 8}
# 27B full fine-tune on 80GB H100s: fp32 master + grads + Adam = 16 B/param = 432 GB sharded ->
# 54 GB/GPU on one node before activations, so default to TWO nodes (27 GB/GPU). FS_NUM_NODES overrides.
NUM_NODES = {"0.8b": 1, "2b": 1, "4b": 1, "9b": 1, "27b": int(os.environ.get("FS_NUM_NODES", "2"))}
KV_MICRO = {"0.8b": 2, "2b": 2, "4b": 2, "9b": 1, "27b": 1}
TASKS = os.environ.get("FS_TASKS", "oolong,contradiction").split(",")
BUDGETS = {"oolong": ["20M", "80M"], "contradiction": ["14M", "56M"]}
if os.environ.get("FS_BUDGETS"):  # FS_BUDGETS=oolong:80M,contradiction:56M restricts per task
    for kv in os.environ["FS_BUDGETS"].split(","):
        t, b = kv.split(":"); BUDGETS[t] = b.split("+")
ARMS = {"oolong": ["dense", "kv17", "ffnmoe-t10"], "contradiction": ["dense", "kv33", "ffnmoe-t10"]}
if os.environ.get("FS_ARMS"):  # FS_ARMS=ffnmoe-t10p limits every task to these arms
    ARMS = {t: os.environ["FS_ARMS"].split(",") for t in ARMS}
if os.environ.get("FS_EXTRA_ARMS"):  # FS_EXTRA_ARMS=ffnmoe-t10p appends to each task's list
    ARMS = {t: a + [x for x in os.environ["FS_EXTRA_ARMS"].split(",") if x not in a] for t, a in ARMS.items()}
CLUSTER = os.environ.get("FS35_CLUSTER", "ai2/jupiter-cirrascale-2,ai2/ceres-cirrascale,ai2/saturn-cirrascale")
STATE = f"{D}/orchestrate_{TAG}_state.json"
o35.STATE = STATE  # orchestrate35.save/load write this file
LAUNCHER = f"{REPO}/src/scripts/train/memexpress/ctc_suite/beaker_ctc_suite.py"
log = o35.log


def run_name(task, arm, budget):
    return f"fs35{TAG}-{task}-{arm}-s{budget}"


def launch_train(st, task, budget, arm):
    name = run_name(task, arm, budget)
    lg.BASE = BASES[SCALE]  # arm_args reads the module-level base
    if arm == "dense":
        variant, data, largs, extra = "full", f"{W}/{lg.ARMS[task][budget]}", ["--pack", "--seq-len", "65536", "--global-batch", "8", "--micro-batch-instances", "1", "--base-checkpoint", BASES[SCALE]], ""
    else:
        variant, data, largs, extra = lg.arm_args(task, arm, budget)
        if arm.startswith(("ffnmoe", "flex-")):  # (flexa / flexs keep start-layer 0)
            # "layers 12+" of a 32-layer model = the top 62.5%; keep that fraction at other depths
            start = round(N_LAYERS[SCALE] * 12 / 32)
            extra = extra.replace("--ffn-moe-start-layer 12", f"--ffn-moe-start-layer {start}")
            extra = extra.replace("--ffn-moe-trainable-width 576", f"--ffn-moe-trainable-width {TRAINABLE_W[SCALE]}")
        if arm.startswith("kv"):  # padded single-example rows; micro-batch by scale
            largs = ["--seq-len", "65536", "--global-batch", "160", "--micro-batch-instances", str(KV_MICRO[SCALE]),
                     "--base-checkpoint", BASES[SCALE]]
    if FAMILY == "qwen3":
        data = data.replace("arms_tokenized/", "arms_tokenized_qwen3/")
    nodes = NUM_NODES.get(SCALE, 1)
    # (ffnmoe-t10p on ONE 80GB node at 27B was tried 2026-09-04 and OOMed at 75 GB: the frozen tail
    # still holds fp32 shards, the forward concatenates full weights per layer, and the 65k row's
    # checkpointed activations are ~43 GB. It runs on the same 2-node/CP-2 footprint as the others.)
    if nodes > 1 and not arm.startswith("kv"):
        # packed arms keep global batch 8 (one 65k row per DP rank): with 2 nodes = 16 ranks that
        # needs Ulysses CP 2 (dp 8 x cp 2), same batch/schedule as every other scale. KV arms
        # (160 padded rows/step) split across 16 ranks without CP.
        largs = largs + ["--cp-degree", str(nodes)]
    if SCALE == "0.8b":
        # the trainer's scale default for 0.8b is NO activation checkpointing (fits on 141GB H200s);
        # a 65k packed row OOMs an 80GB H100 without it (fs35s08b-oolong-dense-s20M, 23:20)
        extra = (extra + " --activation-checkpointing full").strip()
    cmd = [o35.PY, "-u", LAUNCHER, "--task", task, "--variant", variant, "--model-family", FAMILY, "--model-scale", SCALE,
           "--data-root", data, "--run-name", name, "--exact-run-name", "--num-nodes", str(nodes), "--num-gpus", str(GPUS[SCALE]),
           "--epochs", "1", "--lr", "5e-6", "--cluster", CLUSTER, "--wandb-group", "flop-scaling-q35-scale",
           "--no-follow", "--no-compile"] + largs + (["--extra-args", extra] if extra else []) + ["launch"]
    rc, out = o35.sh(cmd, timeout=1200)
    os.makedirs(f"{D}/launch_logs", exist_ok=True)
    open(f"{D}/launch_logs/{name}.launch.log", "w").write(" ".join(cmd) + "\n\n" + out)
    ex = o35.parse_id(out)
    prev = st["runs"].get(name, {})
    st["runs"][name] = {"ex": ex, "task": task, "budget": budget, "arm": arm, "scale": SCALE, "state": "S" if ex else "LAUNCH-FAILED",
                        "rc": None, "retries": prev.get("retries", 0), "eval": prev.get("eval"), "s2": None}
    log(f"launch {name} -> {ex or 'FAILED: ' + out[-300:]}")
    o35.save(st)


def cycle(st):
    if not st.get("launched"):
        for task in TASKS:
            for b in BUDGETS[task]:
                for arm in ARMS[task]:
                    if run_name(task, arm, b) not in st["runs"]:
                        launch_train(st, task, b, arm)
        st["launched"] = True
        o35.save(st)
    # training runs
    for name, r in list(st["runs"].items()):
        if r["state"] == "DONE" and not r.get("eval"):
            o35.launch_eval(st, name)
        if r["state"] in ("DONE", "FAILED") or not r.get("ex"):
            if r["state"] == "LAUNCH-FAILED" and r["retries"] < 1:
                r["retries"] += 1
                launch_train(st, r["task"], r["budget"], r["arm"])
            continue
        s, rc = o35.status(r["ex"])
        if s == "F":
            r["rc"] = rc
            if rc == 0:
                r["state"] = "DONE"; log(f"{name} DONE"); o35.launch_eval(st, name)
            elif r["retries"] < 1:
                r["retries"] += 1; log(f"{name} failed rc={rc}; relaunching"); launch_train(st, r["task"], r["budget"], r["arm"])
                st["runs"][name]["retries"] = r["retries"]
            else:
                r["state"] = "FAILED"; log(f"{name} FAILED rc={rc}")
            o35.save(st)
        elif s == "R" and r["state"] != "R":
            r["state"] = "R"; log(f"{name} running"); o35.save(st)
    # evals
    for name, e in list(st["evals"].items()):
        if e["state"] == "DONE" or not e.get("ex"):
            if e["state"] == "LAUNCH-FAILED" and e["retries"] < 3:
                e["retries"] += 1; o35.launch_eval(st, name)
            continue
        s, rc = o35.status(e["ex"])
        if s == "F":
            if rc == 0:
                e["state"] = "DONE"; log(f"eval {name} DONE")
            elif e["retries"] < 3:
                e["retries"] += 1; log(f"eval {name} failed rc={rc}; relaunching"); o35.launch_eval(st, name); st["evals"][name]["retries"] = e["retries"]
            else:
                e["state"] = "FAILED"; log(f"eval {name} FAILED rc={rc}")
            o35.save(st)
    # harvest
    if time.time() - st.get("last_harvest", 0) > o35.HARVEST_EVERY:
        o35.harvest(st)
    all_runs = all(r["state"] in ("DONE", "FAILED") for r in st["runs"].values()) and st.get("launched")
    all_evals = all(e["state"] in ("DONE", "FAILED") for e in st["evals"].values()) and all(
        r.get("eval") for r in st["runs"].values() if r["state"] == "DONE")
    if all_runs and all_evals and not st.get("done"):
        o35.harvest(st); st["done"] = True; o35.save(st); log("ALL_DONE")


def main():
    st = o35.load()
    st.setdefault("runs", {}); st.setdefault("evals", {}); st.setdefault("last_harvest", 0)
    log(f"orchestrate_scale {SCALE} up; runs tracked: {len(st['runs'])}")
    while True:
        try:
            cycle(st)
        except Exception as e:  # keep going
            log(f"cycle error: {type(e).__name__}: {e}")
        if st.get("done"):
            break
        time.sleep(o35.CYCLE)


if __name__ == "__main__":
    main()
