"""
Accuracy conditioned on GOLD OUTPUT LENGTH, for two checkpoints, averaged over tasks.

Motivation
----------
The 256k dense and 256k compressive-landmark runs are scored by the same multirung ladder,
so their headline numbers are directly comparable -- but a single scalar per task hides
*where* a model loses. One hypothesis worth testing is that a compressed-KV model degrades
as the answer it must emit gets longer (more pairs / more ids / longer list answers), because
every emitted item needs another independent read out of the compressed context.

This script slices every per-example score by the length of the GOLD answer and reports the
per-bucket mean for both models side by side.

Input
-----
The `<task>_multirung.generations.jsonl` sidecars written by
``src/scripts/ctc_eval/eval/eval_lc_native.py`` (``_record_gens``). Each line is::

    {"task": ..., "rung": ..., "idx": ..., "generation": ..., "prompt_tail": ...,
     "detail": {<per-example metrics AND the gold>}}

The gold lives in ``detail`` under a task-dependent key -- see ``GOLD_SPEC``. Nothing else is
needed: no checkpoint, no GPU, no re-decoding. This is a pure CPU re-read of results that
already exist on weka.

Definition of "gold output length"
----------------------------------
Two measures are reported, because the tasks are structured-output tasks and the two can
disagree:

``gold_items``
    How many things the gold answer contains -- contradiction pairs, retrieved doc ids,
    outlier ids, oolong list entries. This is the natural notion of "how much does the model
    have to produce", and it is the primary axis.
``gold_tokens``
    Tokens in the canonical rendering of the gold answer (the string the model is graded
    against). Uses the Qwen3.5 tokenizer when available and falls back to a whitespace-word
    count, which is monotone with tokens over answers this short. The fallback is recorded in
    the output so a reader always knows which one produced the numbers.

Pairing
-------
Both models ran the same bundle with the same ordering, so ``(task, rung, idx)`` is a valid
join key. The headline table is computed on the PAIRED intersection only, so a bucket can
never compare one model's examples against a different subset of the other's. The unpaired
per-model coverage is reported alongside so any asymmetry is visible.

Usage
-----
::

    python analysis/gold_length/gold_length_conditioned_accuracy.py \
        --model dense=/weka/.../q35-4b-dense-xlong5-dolci25-256k \
        --model compressive=/weka/.../q35-4b-fastcomplm-xlong5-dolci25-256k \
        --out-dir /weka/.../_eval_results/gold_length_analysis
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import statistics
import sys
from collections import Counter, defaultdict

# --------------------------------------------------------------------------------------
# Task table.
#
# gold_key      : where the gold lives inside the `detail` dict of a generations record.
# metric        : the per-example score used as "accuracy" for this task. All are in [0, 1].
# metric_alts   : fallbacks, in order, when `metric` is absent (rerank only emits ndcg@10 when
#                 the rung carries cross-encoder scores).
# render        : how to turn the gold into the answer string the model is graded against.
# gold_kind     : documentation only -- surfaced in the report so a reader knows what a
#                 "length" means for that task.
#
# Ladder task names come from LSPEC in eval_lc_native.py; the eval FUNCTION is what decides
# the detail schema, so tasks sharing a function share a row shape (nq/fiqa/scifact are all
# _eval_retrieval; outlier/outlier_review are _eval_outlier; contradiction/contra_fever are
# _eval_contradiction).
# --------------------------------------------------------------------------------------


def _render_pairs(gold):
    return ", ".join(
        f"[{p[0]}, {p[1]}]" for p in gold if isinstance(p, (list, tuple)) and len(p) >= 2
    )


def _render_ids(gold):
    return ", ".join(f"[{g}]" for g in gold)


def _render_text_list(gold):
    return ", ".join(str(g) for g in gold)


GOLD_SPEC = {
    "contradiction": dict(
        gold_key="gold_pairs",
        metric="f1",
        metric_alts=(),
        render=_render_pairs,
        gold_kind="contradicting document pairs",
    ),
    "contra_fever": dict(
        gold_key="gold_pairs",
        metric="f1",
        metric_alts=(),
        render=_render_pairs,
        gold_kind="contradicting document pairs",
    ),
    "nq": dict(
        gold_key="gold_ids",
        metric="f1",
        metric_alts=(),
        render=_render_ids,
        gold_kind="relevant document ids",
    ),
    "fiqa": dict(
        gold_key="gold_ids",
        metric="f1",
        metric_alts=(),
        render=_render_ids,
        gold_kind="relevant document ids",
    ),
    "scifact": dict(
        gold_key="gold_ids",
        metric="f1",
        metric_alts=(),
        render=_render_ids,
        gold_kind="relevant document ids",
    ),
    "outlier": dict(
        gold_key="gold",
        metric="f1",
        metric_alts=(),
        render=_render_ids,
        gold_kind="outlier document ids",
    ),
    "outlier_review": dict(
        gold_key="gold",
        metric="f1",
        metric_alts=(),
        render=_render_ids,
        gold_kind="outlier document ids",
    ),
    "oolong": dict(
        gold_key="gold",
        metric="score",
        metric_alts=("exact_match",),
        render=_render_text_list,
        gold_kind="answer list entries (free text)",
    ),
    # rerank's OUTPUT is a fixed-length ranking (top-10) no matter how large the relevant set
    # is, so its gold length does not measure "how much must be emitted" the way the others do.
    # It is parsed and reported, but excluded from the headline macro-average by default --
    # see --include-fixed-output-tasks.
    "rerank": dict(
        gold_key="gold",
        metric="ndcg@10",
        metric_alts=("mrr@10", "recall@10"),
        render=_render_ids,
        gold_kind="relevant ids (output is a fixed top-10 ranking)",
        fixed_output=True,
    ),
}

# Tasks whose emitted answer length is fixed by the prompt rather than by the gold.
FIXED_OUTPUT_TASKS = {t for t, s in GOLD_SPEC.items() if s.get("fixed_output")}

DEFAULT_ITEM_EDGES = [1, 2, 3, 4, 6, 11]
DEFAULT_TOKEN_EDGES = [1, 2, 4, 8, 16, 32, 64]


# --------------------------------------------------------------------------------------
# Tokenizer (optional).
# --------------------------------------------------------------------------------------


class GoldTokenizer:
    """Token counter for gold strings, with an explicit, reported fallback.

    The analysis must not die because a CPU node cannot reach the HF hub, but it also must
    not silently swap in a different length measure -- so which path was taken is recorded
    in ``self.kind`` and echoed into the report.
    """

    def __init__(self, name: str | None):
        self.kind = "whitespace-words"
        self._tok = None
        if not name:
            return
        try:
            from transformers import AutoTokenizer  # noqa: PLC0415  (optional dependency)

            self._tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
            self.kind = f"tokenizer:{name}"
        except Exception as exc:  # noqa: BLE001 -- any failure downgrades, never aborts
            print(
                f"[tokenizer] could not load {name!r} ({type(exc).__name__}: {exc}); "
                f"falling back to whitespace word count",
                flush=True,
            )

    def count(self, text: str) -> int:
        if not text:
            return 0
        if self._tok is None:
            return len(text.split())
        return len(self._tok.encode(text, add_special_tokens=False))


# --------------------------------------------------------------------------------------
# Loading.
# --------------------------------------------------------------------------------------


def discover_gen_files(model_root: str, dir_globs: list[str], ladder_version: str) -> list[str]:
    """All generations sidecars for one model, across its eval tag dirs.

    ``model_root`` is the checkpoint dir (``.../<run-name>``); the eval tag dirs sit directly
    under it (``eval``, ``eval_xlong``, ``eval_xlong256k``, ``eval_yarn2-256k``, ...). The
    filename encodes the ladder: v2 is the bare ``<task>_multirung``, anything else carries a
    ``_<version>`` suffix (run_beaker_multirung_eval.sh).
    """
    suffix = "" if ladder_version == "v2" else f"_{ladder_version}"
    found: list[str] = []
    for dg in dir_globs:
        pattern = os.path.join(model_root, dg, f"*_multirung{suffix}.generations.jsonl")
        found.extend(sorted(glob.glob(pattern)))
    # A file can be reachable through two overlapping globs; keep first-seen order.
    return list(dict.fromkeys(found))


def eval_tag_of(path: str, model_root: str) -> str:
    """The eval tag dir a result file came from, e.g. ``eval_xlong256k``."""
    rel = os.path.relpath(path, model_root)
    return rel.split(os.sep)[0] if os.sep in rel else "."


def load_examples(path: str, model_root: str, tokenizer: GoldTokenizer, audit: dict):
    """Parse one generations sidecar into per-example records.

    Returns a list of dicts keyed for joining: ``(task, rung, idx)`` plus the score and both
    gold-length measures. Records whose task is unknown or whose gold is missing are counted
    into ``audit`` and dropped -- a dropped record is always a reported number, never silent.
    """
    tag = eval_tag_of(path, model_root)
    out = []
    with open(path) as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                audit["bad_json"] += 1
                continue

            task = rec.get("task")
            spec = GOLD_SPEC.get(task)
            if spec is None:
                audit["unknown_task"][str(task)] += 1
                continue

            detail = rec.get("detail")
            if not isinstance(detail, dict):
                audit["no_detail"][task] += 1
                continue

            audit["detail_keys"][task].update(detail.keys())

            gold = detail.get(spec["gold_key"])
            if gold is None:
                # Multi-query retrieval aggregates away the per-query gold; nothing to bucket on.
                audit["no_gold"][task] += 1
                continue
            if not isinstance(gold, (list, tuple)):
                gold = [gold]

            score = None
            used_metric = None
            for key in (spec["metric"], *spec["metric_alts"]):
                val = detail.get(key)
                if isinstance(val, (int, float)):
                    score = float(val)
                    used_metric = key
                    break
            if score is None:
                audit["no_metric"][task] += 1
                continue

            gold_text = spec["render"](gold)
            out.append(
                {
                    "task": task,
                    "rung": rec.get("rung"),
                    "idx": rec.get("idx"),
                    "eval_tag": tag,
                    "score": score,
                    "metric": used_metric,
                    "gold_items": len(gold),
                    "gold_tokens": tokenizer.count(gold_text),
                    "gold_chars": len(gold_text),
                }
            )
            audit["kept"][task] += 1
    return out


def new_audit():
    return {
        "bad_json": 0,
        "unknown_task": Counter(),
        "no_detail": Counter(),
        "no_gold": Counter(),
        "no_metric": Counter(),
        "kept": Counter(),
        "detail_keys": defaultdict(set),
    }


# --------------------------------------------------------------------------------------
# Bucketing + aggregation.
# --------------------------------------------------------------------------------------


def bucket_label(value: int, edges: list[int]) -> str:
    """Half-open buckets from ascending ``edges``; the last edge opens to infinity.

    edges [1,2,3,4,6,11] -> "1", "2", "3", "4-5", "6-10", "11+"  (and "<1" below the first edge)
    """
    if value < edges[0]:
        return f"<{edges[0]}"
    for i, lo in enumerate(edges):
        hi = edges[i + 1] if i + 1 < len(edges) else None
        if hi is None:
            return f"{lo}+"
        if value < hi:
            return str(lo) if hi == lo + 1 else f"{lo}-{hi - 1}"
    return f"{edges[-1]}+"


def bucket_order(edges: list[int]) -> list[str]:
    labels = [f"<{edges[0]}"]
    for i, lo in enumerate(edges):
        hi = edges[i + 1] if i + 1 < len(edges) else None
        labels.append(f"{lo}+" if hi is None else (str(lo) if hi == lo + 1 else f"{lo}-{hi - 1}"))
    return labels


def mean_sem(values: list[float]) -> tuple[float, float]:
    """Mean and standard error of the mean.

    Deliberately the SAMPLE sem, not a binomial one: f1/ndcg/oolong-score are continuous, so
    sqrt(p(1-p)/n) would understate the spread on exactly the tasks with partial credit.
    """
    n = len(values)
    if n == 0:
        return float("nan"), float("nan")
    m = statistics.fmean(values)
    if n == 1:
        return m, float("nan")
    return m, statistics.stdev(values) / math.sqrt(n)


def aggregate(paired, length_field, edges, include_fixed, models):
    """Per-(bucket, model) task means and the macro-average over tasks.

    Averaging is macro over tasks -- the task mean first, then the unweighted mean of those --
    so a task that happens to contribute 600 examples to a bucket cannot outvote one that
    contributes 40. The example-weighted pooled mean is reported next to it; when the two
    disagree the per-task table underneath says why.
    """
    labels = bucket_order(edges)
    # per_task[bucket][task][model] -> list of scores
    per_task = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for rec in paired:
        task = rec["task"]
        if not include_fixed and task in FIXED_OUTPUT_TASKS:
            continue
        b = bucket_label(rec[length_field], edges)
        for model, score in rec["scores"].items():
            per_task[b][task][model].append(score)

    table = []
    for b in labels:
        if b not in per_task:
            continue
        tasks = sorted(per_task[b])
        row = {"bucket": b, "tasks": tasks, "n_tasks": len(tasks), "per_task": {}, "models": {}}
        for task in tasks:
            row["per_task"][task] = {
                m: dict(
                    zip(("mean", "sem"), mean_sem(per_task[b][task][m])),
                    eval_size=len(per_task[b][task][m]),
                )
                for m in models
                if per_task[b][task][m]
            }
        for m in models:
            task_means = [row["per_task"][t][m]["mean"] for t in tasks if m in row["per_task"][t]]
            pooled = [s for t in tasks for s in per_task[b][t][m]]
            macro, macro_sem = mean_sem(task_means)
            pooled_mean, pooled_sem = mean_sem(pooled)
            row["models"][m] = {
                "macro_mean": macro,
                "macro_sem_across_tasks": macro_sem,
                "pooled_mean": pooled_mean,
                "pooled_sem": pooled_sem,
                "eval_size": len(pooled),
                "n_tasks": len(task_means),
            }
        table.append(row)
    return table


def pair_records(by_model: dict[str, list[dict]], models: list[str]):
    """Intersect the models on (task, rung, idx, eval_tag).

    ``eval_tag`` is part of the key on purpose: the same (task, rung, idx) appears in several
    tag dirs (eval_xlong256k re-runs 2k-32k alongside 256k; the yarn dirs re-run them again
    under a different RoPE), and those are different serving configurations whose scores must
    not be silently merged.
    """
    indexed = {
        m: {(r["task"], r["rung"], r["idx"], r["eval_tag"]): r for r in recs}
        for m, recs in by_model.items()
    }
    common = set(indexed[models[0]])
    for m in models[1:]:
        common &= set(indexed[m])

    paired = []
    for key in sorted(
        common, key=lambda k: (k[0], str(k[1]), k[3], k[2] if k[2] is not None else -1)
    ):
        base = indexed[models[0]][key]
        # Gold is a property of the example, not the model; if the two disagree the join is
        # wrong (different bundle or different ordering) and the whole comparison is void.
        mismatch = [m for m in models[1:] if indexed[m][key]["gold_items"] != base["gold_items"]]
        paired.append(
            {
                "task": base["task"],
                "rung": base["rung"],
                "idx": base["idx"],
                "eval_tag": base["eval_tag"],
                "gold_items": base["gold_items"],
                "gold_tokens": base["gold_tokens"],
                "gold_chars": base["gold_chars"],
                "gold_mismatch": mismatch,
                "scores": {m: indexed[m][key]["score"] for m in models},
            }
        )
    coverage = {m: {"loaded": len(indexed[m]), "paired": len(common)} for m in models}
    return paired, coverage


# --------------------------------------------------------------------------------------
# Reporting.
# --------------------------------------------------------------------------------------


def fmt(v, digits=3):
    return "  n/a" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.{digits}f}"


def render_table(table, models, length_field, tokenizer_kind, min_eval_size):
    """Markdown table. Any cell under ``min_eval_size`` carries its size and error bar inline,
    per the repo's reporting rule -- a small slice must never be readable as a bare number."""
    lines = []
    axis = {
        "gold_items": "gold items (pairs / ids / list entries)",
        "gold_tokens": f"gold tokens ({tokenizer_kind})",
        "gold_chars": "gold characters",
    }[length_field]
    lines.append(f"### Accuracy by gold output length -- {axis}")
    lines.append("")
    lines.append("Macro-average over tasks (task mean first, then unweighted mean over tasks).")
    lines.append("")
    head = ["bucket", "n_tasks"] + [f"{m}" for m in models]
    if len(models) == 2:
        head.append(f"delta ({models[1]}-{models[0]})")
    head.append("eval_size/model")
    lines.append("| " + " | ".join(head) + " |")
    lines.append("|" + "|".join(["---"] * len(head)) + "|")
    for row in table:
        cells = [row["bucket"], str(row["n_tasks"])]
        for m in models:
            s = row["models"][m]
            cell = f"{fmt(s['macro_mean'])} ±{fmt(s['macro_sem_across_tasks'])}"
            if s["eval_size"] < min_eval_size:
                cell += f" ⚠ eval_size={s['eval_size']}"
            cells.append(cell)
        if len(models) == 2:
            a, b = row["models"][models[0]], row["models"][models[1]]
            cells.append(fmt(b["macro_mean"] - a["macro_mean"]))
        cells.append(", ".join(f"{m}={row['models'][m]['eval_size']}" for m in models))
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    lines.append(
        "`±` on a model column is the spread ACROSS TASKS, not eval noise; the per-task "
        "table below carries the per-task sem over examples."
    )
    return "\n".join(lines)


def render_per_task(table, models, length_field, min_eval_size):
    lines = ["", f"### Per-task detail ({length_field})", ""]
    head = ["bucket", "task"] + [f"{m} (mean ±sem, eval_size)" for m in models]
    lines.append("| " + " | ".join(head) + " |")
    lines.append("|" + "|".join(["---"] * len(head)) + "|")
    for row in table:
        for task in row["tasks"]:
            cells = [row["bucket"], task]
            for m in models:
                d = row["per_task"][task].get(m)
                if d is None:
                    cells.append("n/a")
                    continue
                cell = f"{fmt(d['mean'])} ±{fmt(d['sem'])}, {d['eval_size']}"
                if d["eval_size"] < min_eval_size:
                    cell += " ⚠"
                cells.append(cell)
            lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    lines.append(
        f"⚠ = fewer than {min_eval_size} eval examples in that cell; read the error bar, "
        f"not the third decimal."
    )
    return "\n".join(lines)


def render_length_distribution(paired, models):
    """How much the gold length actually VARIES per task.

    Without this the headline table is unreadable: a task whose gold is always k=3 contributes
    one bucket and tells you nothing about length sensitivity, but it still shifts the macro
    average in whatever bucket it lands in.
    """
    dist = defaultdict(Counter)
    for r in paired:
        dist[r["task"]][r["gold_items"]] += 1
    lines = [
        "",
        "### Gold-length variation per task",
        "",
        "| task | gold_kind | distinct gold_items | distribution (items x count) | varies? |",
        "|---|---|---|---|---|",
    ]
    for task in sorted(dist):
        c = dist[task]
        shown = ", ".join(f"{k}x{v}" for k, v in sorted(c.items())[:8])
        if len(c) > 8:
            shown += ", ..."
        varies = "yes" if len(c) > 1 else "**NO (single bucket)**"
        kind = GOLD_SPEC[task]["gold_kind"]
        lines.append(f"| {task} | {kind} | {len(c)} | {shown} | {varies} |")
    return "\n".join(lines)


def render_audit(audits, coverage, model_roots, files_per_model):
    lines = ["", "### Input audit", ""]
    for m in sorted(audits):
        a = audits[m]
        lines.append(f"**{m}** -- `{model_roots[m]}`")
        lines.append("")
        lines.append(f"- files read: {len(files_per_model[m])}")
        for f in files_per_model[m]:
            lines.append(f"    - `{f}`")
        lines.append(f"- kept per task: {dict(a['kept'])}")
        lines.append(
            f"- dropped: no_gold={dict(a['no_gold'])} no_metric={dict(a['no_metric'])} "
            f"no_detail={dict(a['no_detail'])} unknown_task={dict(a['unknown_task'])} "
            f"bad_json={a['bad_json']}"
        )
        lines.append(
            f"- loaded={coverage[m]['loaded']} paired={coverage[m]['paired']} "
            f"(unpaired={coverage[m]['loaded'] - coverage[m]['paired']})"
        )
        lines.append("")
        for task in sorted(a["detail_keys"]):
            lines.append(f"    - detail keys [{task}]: {sorted(a['detail_keys'][task])}")
        lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--model",
        action="append",
        required=True,
        metavar="NAME=CKPT_DIR",
        help="repeatable; NAME=<checkpoint dir containing the eval* tag dirs>",
    )
    ap.add_argument(
        "--eval-dirs",
        default="eval,eval_xlong,eval_xlong256k",
        help="comma-separated tag dirs (globs allowed) under each model root. "
        "Default = the native-RoPE main evals; the yarn dirs serve a DIFFERENT "
        "RoPE config and are excluded unless you ask for them.",
    )
    ap.add_argument(
        "--ladder-version",
        default="v2",
        help="v2 reads <task>_multirung.generations.jsonl; anything else reads the "
        "_<version>-suffixed files.",
    )
    ap.add_argument(
        "--rungs",
        default="",
        help="comma-separated rung labels to keep (e.g. 2k,8k,16k,32k). Empty = all.",
    )
    ap.add_argument("--tasks", default="", help="comma-separated task filter. Empty = all known.")
    ap.add_argument(
        "--tokenizer",
        default="Qwen/Qwen3.5-0.8B",
        help="HF tokenizer for gold token counts; '' to force the word-count fallback.",
    )
    ap.add_argument("--item-edges", default=",".join(map(str, DEFAULT_ITEM_EDGES)))
    ap.add_argument("--token-edges", default=",".join(map(str, DEFAULT_TOKEN_EDGES)))
    ap.add_argument(
        "--include-fixed-output-tasks",
        action="store_true",
        help=f"include tasks whose emitted answer length is fixed by the prompt "
        f"({sorted(FIXED_OUTPUT_TASKS)}) in the macro-average.",
    )
    ap.add_argument(
        "--min-eval-size",
        type=int,
        default=500,
        help="cells below this are flagged inline with their size (repo rule).",
    )
    ap.add_argument(
        "--out-dir", default="", help="write report.md + per_example.jsonl + summary.json here"
    )
    args = ap.parse_args()

    model_roots = {}
    # CLI order is the report's column order: the baseline the user named first stays first, so
    # the delta column reads as (variant - baseline) rather than in alphabetical accident order.
    models: list[str] = []
    for spec in args.model:
        if "=" not in spec:
            ap.error(f"--model must be NAME=DIR, got {spec!r}")
        name, root = spec.split("=", 1)
        if name in model_roots:
            ap.error(f"duplicate --model name {name!r}")
        model_roots[name] = root.rstrip("/")
        models.append(name)

    dir_globs = [d.strip() for d in args.eval_dirs.split(",") if d.strip()]
    rung_filter = {r.strip() for r in args.rungs.split(",") if r.strip()}
    task_filter = {t.strip() for t in args.tasks.split(",") if t.strip()}

    tokenizer = GoldTokenizer(args.tokenizer or None)
    print(f"[setup] gold token counts via {tokenizer.kind}", flush=True)

    by_model, audits, files_per_model = {}, {}, {}
    for name, root in model_roots.items():
        files = discover_gen_files(root, dir_globs, args.ladder_version)
        files_per_model[name] = files
        if not files:
            print(
                f"[{name}] NO generations files under {root} matching {dir_globs} "
                f"(ladder {args.ladder_version}). Nothing to analyse.",
                file=sys.stderr,
            )
        audit = new_audit()
        recs = []
        for f in files:
            recs.extend(load_examples(f, root, tokenizer, audit))
        if rung_filter:
            recs = [r for r in recs if r["rung"] in rung_filter]
        if task_filter:
            recs = [r for r in recs if r["task"] in task_filter]
        by_model[name] = recs
        audits[name] = audit
        print(f"[{name}] {len(files)} file(s) -> {len(recs)} usable examples", flush=True)

    if not any(by_model.values()):
        print(
            "FATAL: no usable examples for any model -- check --eval-dirs / --ladder-version "
            "against the audit above.",
            file=sys.stderr,
        )
        return 2

    paired, coverage = pair_records(by_model, models)
    print(f"[pair] {len(paired)} examples present in all {len(by_model)} models", flush=True)
    mismatched = [r for r in paired if r["gold_mismatch"]]
    if mismatched:
        print(
            f"[pair] WARNING: {len(mismatched)} paired examples disagree on gold between models "
            f"-- the (task, rung, idx) join is NOT aligned; results are void until this is 0.",
            flush=True,
        )

    item_edges = [int(x) for x in args.item_edges.split(",")]
    token_edges = [int(x) for x in args.token_edges.split(",")]

    sections, summary = [], {}
    for field, edges in (("gold_items", item_edges), ("gold_tokens", token_edges)):
        table = aggregate(paired, field, edges, args.include_fixed_output_tasks, models)
        sections.append(render_table(table, models, field, tokenizer.kind, args.min_eval_size))
        sections.append(render_per_task(table, models, field, args.min_eval_size))
        summary[field] = {"edges": edges, "table": table, "models": models}

    report = "\n".join(
        [
            "# Accuracy conditioned on gold output length",
            "",
            f"Models: {', '.join(f'`{m}` = {model_roots[m]}' for m in models)}",
            f"Eval dirs: {dir_globs} | ladder {args.ladder_version} | "
            f"rungs {sorted(rung_filter) or 'all'} | paired examples: {len(paired)}",
            f"Macro-average excludes fixed-output tasks {sorted(FIXED_OUTPUT_TASKS)}: "
            f"{not args.include_fixed_output_tasks}",
            "",
            "Per-example metric: contradiction/contra_fever/nq/fiqa/scifact/outlier/outlier_review = f1, "
            "oolong = score, rerank = ndcg@10 (fallback mrr@10).",
            "",
            render_length_distribution(paired, models),
            "",
            *sections,
            render_audit(audits, coverage, model_roots, files_per_model),
        ]
    )
    print()
    print(report, flush=True)

    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        with open(os.path.join(args.out_dir, "report.md"), "w") as fh:
            fh.write(report + "\n")
        with open(os.path.join(args.out_dir, "summary.json"), "w") as fh:
            json.dump(
                {
                    "models": model_roots,
                    "eval_dirs": dir_globs,
                    "ladder_version": args.ladder_version,
                    "tokenizer": tokenizer.kind,
                    "n_paired": len(paired),
                    "n_gold_mismatch": len(mismatched),
                    "coverage": coverage,
                    "include_fixed_output_tasks": args.include_fixed_output_tasks,
                    "buckets": summary,
                },
                fh,
                indent=2,
            )
        with open(os.path.join(args.out_dir, "per_example.jsonl"), "w") as fh:
            for r in paired:
                fh.write(json.dumps(r) + "\n")
        print(
            f"\n[out] wrote report.md, summary.json, per_example.jsonl to {args.out_dir}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
