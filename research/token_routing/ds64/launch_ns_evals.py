"""Wait for the slot-less ds64 training runs (chk32ns / flk64ns) to finalize, then launch their
5-rung ladder evals exactly as the orchestrator does (same command, cluster, rung files), so
collect_ds64.py / results.csv rows are comparable with chk32 / flk64. Detached:

    setsid nohup python research/token_routing/ds64/launch_ns_evals.py >> research/token_routing/ds64/launch_ns_evals.log 2>&1 &
"""
import json, os, subprocess, sys, time
D = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, D)
import orchestrate_ds64 as O  # noqa: E402

RUNS = {  # name -> (task, training experiment id)
    "ds64-contradiction-chk32ns-b128f3-u16M": ("contradiction", "01M35M81YV5HQHH8ENXS2N2GHC"),
    "ds64-contradiction-chk32ns-b128f3-u32M": ("contradiction", "01M35M8TX6N4Z7HSSZ7RV8A9Q3"),
    "ds64-outlier-flk64ns-b128f3-u16M": ("outlier", "01M35M9JSC7PA4FDXRABYTG2G0"),
    "ds64-outlier-flk64ns-b128f3-u32M": ("outlier", "01M35MABFX7VV8454SD3YPK2RJ"),
}
# Each eval is recorded in the ops run registry by the eval launcher (--experiment ds64); a training
# failure is recorded as an update of the training run. ns_evals.tsv is kept as history only.
sys.path.insert(0, f"{O.REPO}/ops")
try:
    from opslib import safe as ops_safe  # stdlib-only; its calls never raise
except Exception as e:  # noqa: BLE001
    print(f"[ops] WARNING: cannot import ops ({e}); training failures will not be recorded", flush=True)
    ops_safe = None

def status(ex):
    try:
        out = subprocess.run(["beaker", "experiment", "get", ex, "--format", "json"], capture_output=True, text=True, timeout=120).stdout
        d = json.loads(out); d = d[0] if isinstance(d, list) else d
        j = (d.get("jobs") or [{}])[-1]; s = j.get("status", {})
        return ("done" if s.get("finalized") else "running"), s.get("exitCode")
    except Exception as e:  # noqa: BLE001
        return "unknown", repr(e)

def launch_eval(name, task):
    files = O.RUNG_FILES[task]
    cmd = [O.PY, "-u", f"{O.REPO}/recipes/eval/legacy_native/run_q4b_beaker_multirung_eval.py", name, O.EVAL_CLUSTER,
           "--task", O.TASK_KEY[task], "--variant", "docchunk", "--ckpt", f"{O.W}/ctc_suite/ckpts/{name}",
           "--query-position", "after", "--cot-mode", "none", "--tokenizer", O.TOKENIZER,
           "--ngpu", O.EVAL_NGPU, "--max-test", "500", "--max-length", "70000", "--priority", "urgent",
           "--experiment", "ds64",
           "--dc-rung-files", json.dumps(files), "--dc-rungs", ",".join(next(iter(files.values())).keys())]
    rc, out = O.sh(cmd, timeout=1200)
    return O.parse_id(out), out[-300:]

pending = dict(RUNS)
while pending:
    for name, (task, ex) in list(pending.items()):
        st, rc = status(ex)
        print(time.strftime("%H:%M:%S"), name, st, rc, flush=True)
        if st == "done":
            if rc == 0:
                eid, tail = launch_eval(name, task)
                print(f"  eval {name} -> {eid or 'FAILED ' + tail}", flush=True)
            else:
                print(f"  TRAINING FAILED rc={rc}: {name}", flush=True)
                if ops_safe is not None:
                    ops_safe.update_run(name, final_state="failed", state_source=f"launch_ns_evals: beaker exit {rc}")
            del pending[name]
    if pending:
        time.sleep(300)
print("ALL_DONE", flush=True)
