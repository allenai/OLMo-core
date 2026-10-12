#!/usr/bin/env python
"""Assemble the full CTC-suite dense-vs-chunked results table from S3 grades.

Pulls every grade JSON under ctc_dense_results/ and ctc_chunked_results/ on S3,
builds a task x rung table (dense | chunked per metric), and emits markdown.
Flags: parse_rate<0.5 (corruption), chunked-only tasks (no dense counterpart),
and known-unreliable tasks (helmet_qa/helmet_summ/mathmatch: near-floor both arms).
"""
import json
import subprocess
import sys
from collections import defaultdict

S3 = "s3://ai2-llm/checkpoints/prasanns/_transfer"
RUNGS = [2048, 4096, 8192, 16384, 32768]
# obliq_retrieval: mode-collapse from a data-poor (~1797-example) training run (verified via
# gens: constant templated predictions) -> undertrained, exclude.
# helmet_qa (token_f1~0.03) and mathmatch (set_f1~0.005) are UNDER INVESTIGATION -- suspiciously
# low, NOT yet confirmed genuine (helmet_qa may be a rambling/verbose-answer artifact; mathmatch
# uses the contradiction parse_pairs scorer just fixed for the primed-bracket bug). Do not treat
# as settled exclusions until the gen-level verdict lands.
# NOTE: helmet_summ (rouge1_f~0.33) is NOT floor -- a normal summarization score, kept in-table.
KNOWN_UNRELIABLE = {"obliq_retrieval"}
UNDER_INVESTIGATION = {"helmet_qa", "mathmatch"}
# the -full (dense) checkpoint const-collapsed during training (constant prediction under FULL
# attention) -> exclude the DENSE cell from the CTC comparison; the -cmix (chunked) arm is fine.
BROKEN_FULL_CKPT = {"scifact", "groups4"}
# no usable -full checkpoint at all -> chunked-only
NO_FULL_CKPT = {"hotpotqa", "oolong"}
# reorder-16k is a genuine model capability floor (only ~1% valid 116-permutations), not an
# eval artifact; reorder 2k/4k are reliable, 8k/16k reflect real difficulty scaling.
CAPABILITY_FLOOR_NOTE = {"reorder": "16k=genuine floor (model can't emit exact 116-perm)"}


def s3_ls(prefix):
    out = subprocess.run(
        ["aws", "s3", "ls", "--recursive", f"{S3}/{prefix}/"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    files = []
    for line in out.stdout.splitlines():
        parts = line.split()
        if parts and parts[-1].endswith(".json"):
            files.append(parts[-1])  # key path
    return files


def s3_get(key):
    out = subprocess.run(
        ["aws", "s3", "cp", f"s3://ai2-llm/{key}", "-"], capture_output=True, text=True, timeout=60
    )
    try:
        return json.loads(out.stdout)
    except Exception:
        return None


def metric_of(d):
    """Return (metric_name, value, parse_rate) from a grade dict, schema-tolerant."""
    if not isinstance(d, dict):
        return (None, None, None)
    mn = d.get("metric_name")
    mv = d.get("metric_value")
    if mv is None:
        # some graders store the metric under its own key
        for k in (
            "pairwise_f1",
            "cycle_f1",
            "set_f1",
            "gold_id_f1",
            "token_f1",
            "kendall_tau",
            "partial_credit",
            "mrr@10",
            "rouge1_f",
            "textgroups_f1",
            "pair_f1",
            "exact_match",
            "em",
        ):
            if k in d:
                mn, mv = mn or k, d[k]
                break
    return (mn, mv, d.get("parse_rate"))


def collect(arm_prefix):
    """task -> rung -> (metric_name, value, parse_rate)"""
    table = defaultdict(dict)
    for key in s3_ls(arm_prefix):
        # .../<arm_prefix>/<task>/rung_<N>.json
        segs = key.split("/")
        try:
            task = segs[-2]
            rung = int(segs[-1].replace("rung_", "").replace(".json", ""))
        except Exception:
            continue
        d = s3_get(key)
        table[task][rung] = metric_of(d)
    return table


def fmt(v):
    if v is None:
        return "—"
    try:
        return f"{float(v):.3f}"
    except Exception:
        return str(v)


def main():
    dense = collect("ctc_dense_results")
    chunked = collect("ctc_chunked_results")
    all_tasks = sorted(set(dense) | set(chunked))

    lines = []
    lines.append("# CTC-suite: Qwen3.5-4B dense (full-attn) vs document-chunked\n")
    lines.append(
        "All eval_size=500 unless noted. metric per task. `flag`: ⚠=parse_rate<0.5 (corruption); "
        "CHUNKED-ONLY=no dense checkpoint; UNRELIABLE=near-floor in both arms (task-level issue).\n"
    )
    lines.append(
        "| task | metric | arm | " + " | ".join(f"{r//1024}k" for r in RUNGS) + " | flag |"
    )
    lines.append("|---|---|---|" + "|".join("---" for _ in RUNGS) + "|---|")

    for task in all_tasks:
        d_r = dense.get(task, {})
        c_r = chunked.get(task, {})
        # metric name from whichever arm has data
        mname = "?"
        for r in RUNGS:
            for src in (d_r, c_r):
                if r in src and src[r][0]:
                    mname = src[r][0]
                    break
            if mname != "?":
                break
        flags = []
        if task in NO_FULL_CKPT or (task not in dense):
            flags.append("CHUNKED-ONLY(no -full ckpt)")
        elif task not in chunked:
            flags.append("DENSE-ONLY")
        if task in BROKEN_FULL_CKPT:
            flags.append("DENSE EXCLUDED (-full ckpt const-collapsed under full attn)")
        if task in KNOWN_UNRELIABLE:
            flags.append("UNRELIABLE (floor in BOTH arms)")
        if task in CAPABILITY_FLOOR_NOTE:
            flags.append(CAPABILITY_FLOOR_NOTE[task])
        flag = "; ".join(flags)

        d_cells = " | ".join(fmt(d_r.get(r, (None, None, None))[1]) for r in RUNGS)
        c_cells = " | ".join(fmt(c_r.get(r, (None, None, None))[1]) for r in RUNGS)
        lines.append(f"| {task} | {mname} | dense | {d_cells} | {flag} |")
        lines.append(f"| | | chunked | {c_cells} | |")

    # coverage summary
    n_dense = sum(len(v) for v in dense.values())
    n_chunk = sum(len(v) for v in chunked.values())
    lines.append(
        f"\n**Coverage:** dense grades={n_dense}, chunked grades={n_chunk}, "
        f"tasks={len(all_tasks)}."
    )
    print("\n".join(lines))


if __name__ == "__main__":
    main()
