"""
FAST **8k** screening loop for soft-token (pooled-doc KV) training recipes -- the length sibling of
``debug/ds64_fast2k/launch_fast2k.py``, whose arm vocabulary and helpers it imports rather than
copies.

WHY AN 8k SCREEN EXISTS AT ALL. fast2k screens a recipe in under an hour, and it is decisive about
whether a recipe can learn the task. It is blind to the one thing outlier actually fails at:
**length generalisation**. Measured on the ds64 ladder (records/ds64-overnight-2026-09-14.md,
09-15): ``cc00`` -- keep 0, header real, ``cent_cmean`` slot -- scores 0.80 at the 2k rung and
**0.17 at 8k**, and ``kv33`` scores 0.890 at 2k and 0.018 at 32k. A 2k screen calls both of those a
success. fast8k moves the whole loop up one rung so the failure is INSIDE the screen: training rows
come from the ds64 **8k** pool (n ~ 56 documents), and the decisive eval is the 8k rung.

WHAT IS HELD FIXED vs fast2k: same task, same generator, same repaired Qwen3.5-4B base, same
``--st-*`` flag strings, same ds64 pools, same unpacked matched-step construction, same FLOP meter,
the same eval files the ds64 ladder uses. What differs: the 8k rung instead of the 2k rung, seq-len
sized to the 8k rows, 16 rows/step instead of 32 (so an 8k step carries a comparable token count),
and an eval on the **8k rung first**, with the 2k rung kept for continuity with the fast2k table.

THE PRIMARY METRIC IS EVAL-TIME CE PARITY, NOT f1 (2026-09-15). For each trained checkpoint the
``parity`` mode runs ONE checkpoint on TWO inputs -- full real text (what the ladder eval feeds it)
vs its own soft construction -- and reports ``CE_full / CE_soft / dCE``, the same on answer DIGITS
only, and generative F1 both ways. That separates "the recipe cannot do the task" from "the recipe
can do the task but its compaction throws away the answer", which an f1 number alone cannot.

    python debug/ds64_fast8k/launch_fast8k.py --arms dense --budgets 8M,16M,20M dry_run
    python debug/ds64_fast8k/launch_fast8k.py --arms cc00,cpi17,cpi33,cpi50,kvgb50 --budgets 16M launch
    python debug/ds64_fast8k/launch_fast8k.py status
    python debug/ds64_fast8k/launch_fast8k.py eval        # 8k + 2k rungs, 500 rows each
    python debug/ds64_fast8k/launch_fast8k.py parity      # eval-time CE parity, the primary signal
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import subprocess
import sys

REPO = "/accounts/projects/berkeleynlp/prasann/projects/OLMo-core"
D = f"{REPO}/debug/ds64_fast8k"
sys.path.insert(0, f"{REPO}/debug/ds64_fast2k")

# Reused verbatim from the fast2k loop (path-independent helpers + the arm flag vocabulary, so a
# fast2k screen and a fast8k screen of the same arm NAME are the same recipe at two lengths).
from launch_fast2k import (  # noqa: E402
    ARM_EXTRA as F2K_ARM_EXTRA,
    BASE,
    CKPTS,
    SCALE,
    TOKENIZER,
    W,
    _XHDR,
    _XSLOT,
    beaker_status,
    parse_id,
    sh,
)

LAUNCHER = f"{REPO}/src/scripts/train/memexpress/ctc_suite/beaker_ctc_suite.py"
EVAL_LAUNCHER = f"{REPO}/src/scripts/train/memexpress/singletask_ladder/run_q4b_beaker_multirung_eval.py"
PROBE_LAUNCHER = f"{REPO}/debug/flop_scaling/beaker_bench_launch.py"
PROBE = "debug/pooled_kv/outlier_probe/outlier_slot_probe.py"
SHARDS = f"{W}/ds64/fast8k/shards"
LEDGER = f"{D}/LAUNCH_LEDGER.tsv"
STATE = f"{D}/state.json"

TASK = os.environ.get("F8K_TASK", "outlier")
BUDGETS = ["8M", "16M", "20M"]
#: rows per budget/slice -- ``compose_fast8k.py`` writes these; the build job prints them and the
#: shard's ``fast8k_stats.json`` carries the realised ``shard_num_instances``. P1/P2 are the
#: two-phase arm's DISJOINT windows: 85% then the remaining 15% of the 16M budget.
EXAMPLES = {"8M": 977, "16M": 1953, "20M": 2441, "P1": 1660, "P2": 293}
#: 16, not fast2k's 32: an 8k row carries ~4x the tokens of a 2k row, so 16 rows/step is ~141k real
#: tokens/step -- the same order as fast2k's 32 x 2.2k -- and it keeps the smallest budget at 61
#: steps rather than 30. Keep GLOBAL_BATCH % (micro * ngpu) == 0.
GLOBAL_BATCH = int(os.environ.get("F8K_GB", "16"))
#: 2 GPUs is enough: 4b resolves to shard_degree=world_size + FULL activation checkpointing
#: (train_ctc_suite.resolve_activation_checkpointing), so a 2-rank FSDP job holds ~1/2 of an 8 GB
#: bf16 param set, its grads and its ~32 GB fp32 AdamW state, and 2 x 12288 tokens of checkpointed
#: activations is far below the 4-GPU x 65536 the ds64 ladder already runs. Memory is not the
#: binding constraint here; queue turnaround is.
GPUS = int(os.environ.get("F8K_NGPU", "2"))
NODES = 1
#: Train-time padded row length; MUST be >= the shard's max_example_len or PadToLength silently
#: SKIPS the long rows. Set from the build job's metadata (an 8k-rung outlier row tokenizes to
#: ~1.1-1.4x its nominal length once the markers and the query are added).
SEQ_LEN = int(os.environ.get("F8K_SEQ_LEN", "12288"))
SOFT_BACKEND = os.environ.get("F8K_SOFT_BACKEND", "flash_2")
CLUSTER = os.environ.get("F8K_CLUSTER", "ai2/ceres-cirrascale,ai2/saturn-cirrascale,ai2/jupiter-cirrascale-2")
EVAL_CLUSTER = os.environ.get("F8K_EVAL_CLUSTER", CLUSTER)
WANDB_GROUP = os.environ.get("F8K_WANDB_GROUP", f"f8k-q35-{SCALE}")

# ---------------------------------------------------------------------------------------------
# Arm vocabulary. fast2k's arms carry over unchanged (same names, same flags); fast8k adds the
# gold_pooled_random family.
#
# ``cpi<p>`` = **gold always POOLED, p of the NON-gold bodies real** (--st-keep-mode
# gold_pooled_random, this repo's new keep mode). It is the middle ground between the two regimes
# the 2k screen separated:
#   * cc00 (keep 0) reads the slots -- 0.482 vs a 0.239 same-FLOP control -- because with no real
#     body anywhere, guessing an id is exactly chance and the slots are the only gradient left. But
#     a model that never saw a real body scores 0.17 at 8k;
#   * cc03 / cc08 / cc17 (1/36, 1/12, 1/6 of bodies real, GOLD-BLIND) all revert to id-guessing,
#     because a gold body that happens to be real still pays off.
# Pooling GOLD removes the payoff without removing real text: a visible id is never an answer, so
# copying one earns exactly chance, while the real non-gold bodies keep the model in the
# distribution it meets at eval (where every document is real).
# NB the fraction knob on the gold-sidecar path is --st-keep-frac; --st-keep-prob is accepted as an
# alias for this mode (train_ctc_suite.resolve_keep_frac) since that is how the gold-blind arms
# spell the same quantity.
_CPI = f"--st-keep-mode gold_pooled_random {_XHDR} --st-slot-mode cent_cmean {_XSLOT}"
#
# ``sc<C>`` / ``gw<K>`` = the WHOLE-CATEGORY policies (olmo_core.nn.attention.doc_categories).
# They come from a sharper reading of why gold-forcing collapsed: under ``gold_plus_random`` the
# gold category was the only category kept WHOLE -- every gold outlier real -- while other
# categories showed up as one or two scattered real documents, so "which category is complete?" was
# a content-free detector of the answer. Both policies keep whole categories ONLY, and both keep
# more than one, so completeness names nothing.
#   sc<C>  smallcat_keep, GOLD-BLIND: the C smallest categories (by the documents' own cent_cmean
#          vectors) kept whole + 1 decoy large category. The rule never reads the label.
#   gw<K>  gold_plus_wholecats: the gold document's category kept whole (as gold_plus_random did)
#          PLUS the K smallest non-gold categories, also whole.
# ⚠ NOT LAUNCHED. Held until the frozen-model probe reports whether the rule reaches eval-loss
# parity; the real-token fraction (and therefore the FLOP cost) is data-dependent and is printed
# every 50 steps by the trainer's ``[cat-keep]`` line.
_CAT = f"{_XHDR} --st-slot-mode cent_cmean {_XSLOT}"
ARM_EXTRA = dict(F2K_ARM_EXTRA)
ARM_EXTRA.update({
    "cpi17": f"--st-keep-frac 0.1667 {_CPI}",
    "cpi33": f"--st-keep-frac 0.3333 {_CPI}",
    "cpi50": f"--st-keep-frac 0.5 {_CPI}",
    "sc3": f"--st-keep-mode smallcat_keep --st-keep-smallcat 3 --st-keep-decoy-cats 1 {_CAT}",
    "sc5": f"--st-keep-mode smallcat_keep --st-keep-smallcat 5 --st-keep-decoy-cats 1 {_CAT}",
    "gw3": f"--st-keep-mode gold_plus_wholecats --st-keep-cats 3 {_CAT}",
    "gw5": f"--st-keep-mode gold_plus_wholecats --st-keep-cats 5 {_CAT}",
})
ARM_MICRO = {}          # every arm at micro 2: the FLOP audit (09-15 10:30) requires identical
DEFAULT_MICRO = 2       # gpus/micro across any pair that will be compared at matched FLOPs.

#: Two-phase arms. ``<arm>-warm`` trains ``<arm>``'s recipe from ANOTHER fast8k run's exported
#: weights: the trainer loads --base-checkpoint with load_optim_state=False /
#: load_trainer_state=False, so phase 2 gets a fresh LR schedule and a fresh budget and phase 1's
#: flags do not carry over. No trainer support beyond --base-checkpoint is needed.
#: ``kvgb50-warm`` on the P2 slice is the "cc00 for 85% of the budget, then real bodies for the
#: last 15%" arm, and P1/P2 are disjoint windows of the 16M rows so phase 2 is not a second epoch.
WARM_FROM = {"kvgb50-warm": ("kvgb50", f"{TASK}-cc00-P1"),
             "cpi33-warm": ("cpi33", f"{TASK}-cc00-P1")}

# --- eval-time CE parity -----------------------------------------------------------------------
# The soft construction the probe runs beside FULL real text, written as the TRAINER's own flags
# with commas for spaces (outlier_slot_probe.py --construction; commas so the spec survives a
# launcher that splits argv on whitespace).
#
# ⚠ These are the GOLD-BLIND analogues of the cpi arms, not their literal training construction:
# parse_construction REFUSES --st-keep-frac, because the probe implements the gold-blind keep_prob
# path and the gold-only keep, not the gold-sidecar policies. That is the right thing to measure
# anyway -- at eval there is no gold sidecar, so "keep p of the bodies real, blind" is the
# construction a cpi arm would actually be DEPLOYED under, and the pooled-set shortcut a cpi arm
# could learn in training (see collect_fast8k.pooled_floor) is not available here either.
_SLOT = "--st-header-stop-id,5491,--st-header-stop-count,1,--st-slot-mode,cent_cmean"
CONSTRUCTIONS = {
    # everything pooled -- the COMMON maximal-compaction reference, run for every arm so dCE is
    # comparable across arms and across budgets
    "cc00": f"--st-gold-blind,--st-keep-prob,0.0,{_SLOT}",
    "gb17c": f"--st-gold-blind,--st-keep-prob,0.1667,{_SLOT}",
    "gb33c": f"--st-gold-blind,--st-keep-prob,0.3333,{_SLOT}",
    "gb50c": f"--st-gold-blind,--st-keep-prob,0.5,{_SLOT}",
    "kvgb50": "--st-gold-blind,--st-keep-prob,0.5",     # plain mean slot, no header
}
#: arm -> the constructions to score it under. First entry is the one promoted into results.csv.
PARITY_ARMS = {"dense": ["cc00"], "cc00": ["cc00"], "kvgb50": ["kvgb50", "cc00"],
               "cpi17": ["gb17c", "cc00"], "cpi33": ["gb33c", "cc00"], "cpi50": ["gb50c", "cc00"],
               # a whole-CATEGORY construction is not reproducible by outlier_slot_probe's
               # keep_prob/gold-only conditions, so these are scored against the common
               # maximal-compaction reference only.
               "sc3": ["cc00"], "sc5": ["cc00"], "gw3": ["cc00"], "gw5": ["cc00"]}
PARITY_RUNGS = os.environ.get("F8K_PARITY_RUNGS", "8k,32k")
PARITY_ROWS = os.environ.get("F8K_PARITY_ROWS", "200")
PARITY_GEN_ROWS = os.environ.get("F8K_PARITY_GEN_ROWS", "32")


def run_name(task, arm, budget):
    return f"f8k-{task}-{arm}-{budget}"


def resolve(arm):
    """(base recipe arm, base checkpoint) for ``arm`` -- resolves the ``-warm`` suffix."""
    if arm in WARM_FROM:
        recipe, src = WARM_FROM[arm]
        return recipe, f"{CKPTS}/f8k-{src}/model_and_optim"
    return arm, BASE


def arm_args(task, arm, budget):
    recipe, base = resolve(arm)
    data = f"{SHARDS}/{task}_g{budget}"
    micro = ARM_MICRO.get(recipe, DEFAULT_MICRO)
    if GLOBAL_BATCH % (micro * GPUS * NODES):
        raise SystemExit(f"global batch {GLOBAL_BATCH} not divisible by micro {micro} x {GPUS * NODES} GPUs")
    common = ["--seq-len", str(SEQ_LEN), "--global-batch", str(GLOBAL_BATCH),
              "--micro-batch-instances", str(micro), "--base-checkpoint", base]
    if recipe == "dense":
        # NOT --pack: every arm trains unpacked so rows/step and step counts match exactly.
        return "full", data, common, ""
    if recipe in ARM_EXTRA:
        return "softtoken", data, common, f"{ARM_EXTRA[recipe]} --attn-backend {SOFT_BACKEND}"
    raise SystemExit(f"unknown arm {arm} (known: dense, {', '.join(ARM_EXTRA)}, "
                     f"plus the -warm variants {', '.join(WARM_FROM)})")


def load_state():
    return json.load(open(STATE)) if os.path.exists(STATE) else {"runs": {}, "evals": {}, "parity": {}}


def save_state(st):
    st.setdefault("parity", {})
    json.dump(st, open(STATE + ".tmp", "w"), indent=1)
    os.replace(STATE + ".tmp", STATE)


def _log(name, mode, out):
    os.makedirs(f"{D}/launch_logs", exist_ok=True)
    open(f"{D}/launch_logs/{name}.{mode}.log", "w").write(out)


def do_launch(args, mode):
    st = load_state()
    rows = []
    for arm in args.arms.split(","):
        for budget in args.budgets.split(","):
            variant, data, largs, extra = arm_args(args.task, arm, budget)
            name = run_name(args.task, arm, budget)
            if name in st["runs"] and mode == "launch" and not args.force:
                print(f"[skip] {name} already launched ({st['runs'][name].get('ex')})")
                continue
            cmd = [sys.executable, "-u", LAUNCHER, "--task", args.task, "--variant", variant,
                   "--model-family", "qwen3_5", "--model-scale", SCALE, "--data-root", data,
                   "--run-name", name, "--exact-run-name", "--num-nodes", str(NODES),
                   "--num-gpus", str(GPUS), "--epochs", "1", "--lr", str(args.lr),
                   "--cluster", CLUSTER, "--wandb-group", args.wandb_group, "--priority", "urgent",
                   "--no-follow", "--no-compile"] + largs + (["--extra-args", extra] if extra else []) + [mode]
            print(" ".join(cmd), flush=True)
            rc, out = sh(cmd)
            _log(name, mode, out)
            ex = parse_id(out)
            tail = "\n".join(out.strip().splitlines()[-2:]).replace("\t", " ")[:200]
            print(f"  -> rc={rc} ex={ex}: {tail[:200]}", flush=True)
            if mode == "launch":
                st["runs"][name] = {"ex": ex, "task": args.task, "arm": arm, "budget": budget,
                                    "steps_expected": -(-EXAMPLES.get(budget, 0) // GLOBAL_BATCH),
                                    "gpus": GPUS * NODES, "state": "S" if ex else "LAUNCH-FAILED"}
                save_state(st)
                rows.append((args.task, arm, budget, name, rc, (ex or "") + " " + tail))
    if rows:
        with open(LEDGER, "a") as f:
            w = csv.writer(f, delimiter="\t")
            for task, arm, budget, name, rc, tail in rows:
                w.writerow(["fast8k", SCALE, task, arm, budget, name, CLUSTER,
                            dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
                            "LAUNCHED" if rc == 0 else "LAUNCH-FAILED", tail])
    print(f"\nwandb: https://wandb.ai/prasanns-allen-institute-for-ai/memory-networks/groups/{args.wandb_group}")


#: 8k FIRST (the decisive rung -- it is where the outlier soft arms die) and 2k for continuity with
#: the fast2k table. Same files the ds64 orchestrator's ladder uses.
RUNGS = {"outlier": {"8k": f"{W}/outlier_lengthmix/eval_rungs/outlier/rung_8192.jsonl",
                     "2k": f"{W}/outlier_lengthmix/eval_rungs/outlier/rung_2048.jsonl"},
         "nq": {"8k": f"{W}/outlier_lengthmix/eval_rungs/nq/rung_8192.jsonl",
                "2k": f"{W}/outlier_lengthmix/eval_rungs/nq/rung_2048.jsonl"}}
TASK_KEY = {"nq": "nq", "outlier": "outlier", "oolong": "oolong", "contradiction": "contra"}


def do_eval(args):
    st = load_state()
    names = args.runs.split(",") if args.runs else list(st["runs"])
    for name in names:
        r = st["runs"].get(name)
        if r is None:
            print(f"[skip] {name} not in state.json")
            continue
        if st["evals"].get(name, {}).get("ex") and not args.force:
            print(f"[skip] eval {name} already launched ({st['evals'][name]['ex']})")
            continue
        task = r["task"]
        files = {task: RUNGS[task]}
        cmd = [sys.executable, "-u", EVAL_LAUNCHER, name, EVAL_CLUSTER,
               "--task", TASK_KEY[task], "--variant", "docchunk", "--ckpt", f"{CKPTS}/{name}",
               "--query-position", "after", "--cot-mode", "none", "--tokenizer", TOKENIZER,
               "--ngpu", "2", "--max-test", "500", "--priority", "urgent",
               # >= the 8k rung's prompt + generation, with headroom: a --max-length BELOW the
               # prompt truncates it away and the run scores exactly 0.000 (records/ memory
               # "Goldgrad eval MAXLEN truncation bug"). 20000 covers an 8k-rung outlier row.
               "--max-length", "20000", "--batch-size", "8",
               "--dc-rung-files", json.dumps(files), "--dc-rungs", ",".join(RUNGS[task])]
        rc, out = sh(cmd)
        ex = parse_id(out)
        _log(name, "eval", out)
        st["evals"][name] = {"ex": ex, "state": "S" if ex else "LAUNCH-FAILED"}
        save_state(st)
        print(f"eval {name} -> {ex or 'FAILED: ' + out[-300:]}", flush=True)


def do_parity(args):
    """Eval-time CE parity: ONE trained checkpoint, TWO inputs (full real text vs its own soft
    construction), on the 8k rung and -- where affordable -- 32k.

    This is the primary signal. A ladder f1 conflates "the checkpoint cannot do outlier" with "the
    checkpoint can do outlier but the compaction destroys the answer"; running the same weights on
    both inputs separates them, and ``CE on the answer DIGITS`` isolates the id choice from the
    output format. Submitted through the shared 1-GPU bench launcher so it uses the same image and
    the same PUSHED commit the training jobs do.
    """
    st = load_state()
    names = args.runs.split(",") if args.runs else list(st["runs"])
    for name in names:
        r = st["runs"].get(name)
        if r is None:
            print(f"[skip] {name} not in state.json")
            continue
        arm = r["arm"]
        conds = args.parity_arms.split(",") if args.parity_arms else PARITY_ARMS.get(
            arm.replace("-warm", ""), ["cc00"])
        for cond in conds:
            key = f"{name}::{cond}"
            if st.setdefault("parity", {}).get(key, {}).get("ex") and not args.force:
                print(f"[skip] parity {key} already launched ({st['parity'][key]['ex']})")
                continue
            spec = CONSTRUCTIONS.get(cond, cond)
            extra = [
                # --trained-parity implies --no-reset-projector and scores every rung in ONE
                # process, so the 4B checkpoint is loaded once.
                # ``--construction=<spec>``, never ``--construction <spec>``: the spec itself
                # starts with "--", and argparse refuses an option value that looks like another
                # option ("expected one argument"). The "=" form always parses.
                "--trained-parity", "--arm", "none", f"--construction={spec}",
                "--rungs", args.parity_rungs, "--rows", args.parity_rows,
                "--gen-rows", args.parity_gen_rows,
                "--ckpt", f"{CKPTS}/{name}", "--ckpt-name", name,
                "--tokenizer", TOKENIZER, "--tag", f"f8k-{cond}",
                "--work", "/tmp/f8k_probe_work",
                # the stop set MUST come from the TRAINING shard, or the eval-side cent_cmean slot
                # is not the slot the arm trained with
                "--slot-stop-shard", f"{SHARDS}/{r['task']}_g{r['budget']}",
            ]
            cmd = [sys.executable, "-u", PROBE_LAUNCHER, "--cluster", EVAL_CLUSTER,
                   "--script", PROBE, "--gpus", "1", "--priority", "urgent",
                   "--extra", " ".join(extra)]
            rc, out = sh(cmd)
            ex = parse_id(out)
            _log(f"{name}.{cond}", "parity", out)
            st["parity"][key] = {"ex": ex, "run": name, "cond": cond,
                                 "state": "S" if ex else "LAUNCH-FAILED"}
            save_state(st)
            print(f"parity {key} -> {ex or 'FAILED: ' + out[-400:]}", flush=True)


def do_status(args):
    st = load_state()
    changed = False
    for kind in ("runs", "evals", "parity"):
        for name, r in st.get(kind, {}).items():
            if not r.get("ex") or r.get("state") in ("DONE", "FAILED"):
                continue
            s, rc = beaker_status(r["ex"])
            if s == "F":
                r["state"] = "DONE" if rc == 0 else "FAILED"
                r["rc"] = rc
            elif s != "?":
                r["state"] = s
            changed = True
    if changed:
        save_state(st)
    for name, r in sorted(st["runs"].items()):
        e = st["evals"].get(name, {})
        p = [f"{k.split('::')[1]}={v.get('state')}" for k, v in sorted(st.get("parity", {}).items())
             if v.get("run") == name]
        print(f"{name:32} train={str(r.get('state')):12} {r.get('ex','')}  "
              f"eval={str(e.get('state','-')):12} {e.get('ex','')}  parity[{' '.join(p)}]")
    return st


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", default=TASK)
    ap.add_argument("--arms", default="dense")
    ap.add_argument("--budgets", default=",".join(BUDGETS))
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--wandb-group", default=WANDB_GROUP)
    ap.add_argument("--runs", default="", help="eval/parity: comma run names (default = all)")
    ap.add_argument("--parity-arms", default="", help="override the PARITY_ARMS conditions")
    ap.add_argument("--parity-rungs", default=PARITY_RUNGS)
    ap.add_argument("--parity-rows", default=PARITY_ROWS)
    ap.add_argument("--parity-gen-rows", default=PARITY_GEN_ROWS)
    ap.add_argument("--force", action="store_true", help="relaunch even if state.json has it")
    ap.add_argument("mode", choices=["launch", "dry_run", "eval", "parity", "status"])
    args = ap.parse_args()
    if args.mode in ("launch", "dry_run"):
        do_launch(args, args.mode)
    elif args.mode == "eval":
        do_eval(args)
    elif args.mode == "parity":
        do_parity(args)
    else:
        do_status(args)


if __name__ == "__main__":
    main()
