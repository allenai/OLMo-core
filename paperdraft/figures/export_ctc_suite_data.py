"""Export the CTC-suite artifact ("CTC Suite Grid", frozen 2026-08-13) as tidy CSVs.

The artifact page is rendered by `debug/ctc_final_suite/render_artifact.py` from harvested grade
JSONs. This reads the SAME JSONs the page reads --- never the rendered HTML --- so the CSVs cannot
drift from the page, and re-harvesting is the whole update path for both.

Outputs (into figures/data/), in the column order of `export_figure_data.py`:

  ctc_suite_grid_data.csv          one row per (task, rung): the 22-task 2k-32k grid, 4B
  ctc_scale_data.csv               one row per (task, model scale, rung): 0.8B / 2B / 4B
  ctc_length_generalization_data.csv   64k / 128k past the 32k ceiling, plus the fix-k n sweep

Sources:
  debug/ctc_final_suite/suite_table.json          the grid (built by build_suite_table.py)
  debug/ctc_final_suite/scale_table.json          the scale axis (built by build_scale_table.py)
  debug/ctc_modelscale/lengthgen_results/**       one grade JSON per long rung
  debug/ctc_oolong_eval/results/**                oolong's 64k rung, which lives in its own tree
  debug/ctc_modelscale/lengthgen_rung_manifest/   measured p50 tokens + n_docs for the long rungs
  debug/ctc_modelscale/rung_token_audit.json      measured p50 tokens for the 2k-32k ladder

Usage:  python3 paperdraft/figures/export_ctc_suite_data.py
"""
import csv
import glob
import json
import math
import os
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "data"
SUITE = REPO / "debug/ctc_final_suite"
SCALE = REPO / "debug/ctc_modelscale"

RUNGS = [2048, 4096, 8192, 16384, 32768]

# ── CTC classification ────────────────────────────────────────────────────────────────
# suite_table.json stores a coarse class of N / NM / N2. The paper's two axes are `ctc_class`
# (low vs high, the orange/green split in Fig 1) and the finer `ctc_order`; the mapping below is
# the one data/README.md already documents for the existing figure CSVs.
CLASS = {"N": ("low", "O_T(N)"), "NM": ("high", "O_T(NM)"), "N2": ("high", "O_T(N^2)+")}

# Now a pure passthrough -- deliberately kept as an empty dict rather than deleted, because this is
# where a class correction goes if one is ever needed again in a hurry.
#
# It used to carry `absence` and `outlier_amazon`, both of which the figure scripts classed O_T(N)
# while suite_table.json said otherwise. Both are fixed at the source now (prasann confirmed
# outlier_amazon 2026-08-13 and absence 2026-08-13; build_suite_table.py was corrected for each),
# so the JSON, the paper tables and these CSVs agree without an override. `xabsence` is the
# genuinely O_T(N^2) member of that pair and was never overridden.
#
# This drives the low/high split the CTC figures are built on, so an entry here silently moves a
# task between the two lines. See data/README.md.
CLASS_OVERRIDE = {}

# textgroups also disagrees (figure scripts say O_T(N^2)+, suite_table says NM) but both map to
# ctc_class=high, so it changes no grouping and the suite_table order is kept.

# suite_table.json records the grader's metric key; the paper tables spell them differently.
# Table 1 wins for the `metric` column, exactly as in export_figure_data.py; `metric_key` keeps
# the grader's name so a row can be traced back to its grade JSON.
METRIC = {
    "gold_id_f1": "gold-ID F1", "set_f1": "set-F1", "pair_f1": "pair-F1",
    "pairwise_f1": "pairwise-F1", "mrr@10": "MRR@10", "kendall_tau": "Kendall tau",
    "partial_credit": "partial credit", "textgroups_f1": "group-F1", "cycle_f1": "cycle-F1",
}

# Human labels + per-row coverage status, mirrored from render_artifact.py so the CSV and the
# artifact page agree on what each row is called and which rows are not yet trustworthy.
LABEL = {
    "fiqa": "fiqa", "nq": "nq", "hpqa": "hpqa", "qdmatch_fiqa": "qdmatch fiqa",
    "qdmatch_nq": "qdmatch nq", "qdmatch_hpqa": "qdmatch hpqa",
    "outlier_amazon": "outlier amazon", "outlier_scalek": "outlier scale-k",
    "outlier_fixk": "outlier fix-k", "oolong": "OOLONG", "grouping": "grouping",
    "absence": "absence", "xabsence": "xabsence", "rerank": "rerank", "msmarco": "msmarco",
    "reorder": "re-order", "obliq": "obliq", "niah_contra": "niah contra",
    "contra_real": "contradiction (realistic)", "strmatch": "strmatch",
    "textgroups": "textgroups", "scifact": "scifact",
}

STATUS = {
    "fiqa": "ok", "nq": "ok", "hpqa": "ok", "qdmatch_fiqa": "ok", "qdmatch_nq": "ok",
    "qdmatch_hpqa": "ok", "outlier_amazon": "ok", "outlier_scalek": "ok",
    "outlier_fixk": "partial", "oolong": "ok", "grouping": "ok", "absence": "partial",
    "xabsence": "ok", "rerank": "ok", "msmarco": "ok", "reorder": "ok",
    "obliq": "partial", "niah_contra": "partial", "contra_real": "ok", "strmatch": "partial",
    "textgroups": "ok", "scifact": "ok",
}

NOTE = {
    "fiqa": "Ladder built and both arms evaluated 2026-08-13.",
    "hpqa": "Dense backfilled 2026-08-12 after the corrupt S3 distcp.",
    "qdmatch_fiqa": "Trained on Beaker and evaluated 2026-08-13, closing the last blank row. "
                    "Scored on the q61 rung family, matching the trained shard -- the generator "
                    "emitted two query-count ladders, so which one a number came from matters.",
    "qdmatch_nq": "Backfilled from the model-scale 4B runs 2026-08-13.",
    "qdmatch_hpqa": "Rung labels run 0.74-0.82x of the stated tokens.",
    "outlier_amazon": "Rung labels run 1.12-1.24x of the stated tokens.",
    "outlier_fixk": "32k (n=220) deliberately not shown on the page: the row is a smooth decline "
                    "in n, not a cliff. See the n sweep in the length-generalization CSV.",
    "oolong": "Both arms are the ctcms-oolong-*-4b-vsl retrain on the rebuilt, decontaminated, "
              "query-after shard, graded query-after; pinned by run name, not newest-file-wins.",
    "grouping": "Regraded after the parser fix; the old 0.439/0.358 row is dead.",
    "absence": "32k rung never built; 16k graded on eval_size=148 only. Labels run 3.0-3.6x. "
               "Classified low / O_T(N) per the 2026-08-03 correction, overriding suite_table.",
    # The old row (dense 0.61->0.52, chunked 0.135->0.008) was NOT a trained model -- training CE
    # sat on the 0.78 flatline. Root cause was capacity, not the task: 0.8B and 2B collapse to the
    # chance floor on the identical shard, 4B learns it. Retrained 2026-08-13 on the two-sided
    # `xabsence_both_ladder5` shard (20k examples over 5 rungs, missing item can come from either
    # corpus) at 4B; both arms evaluated in their native mode. The row is live and quotable.
    "xabsence": "Retrained at 4B on the two-sided ladder5 shard 2026-08-13; the pre-2026-08-13 "
                "row (dense 0.61->0.52) was an untrained CE-0.78 flatline and is dead. Capacity "
                "threshold, not task difficulty: 0.8B/2B sit at the chance floor on this shard.",
    "rerank": "Ladder rebuilt token-accurate and fully regraded 2026-08-12.",
    "msmarco": "Ladder rebuilt token-accurate and fully regraded 2026-08-12.",
    "reorder": "32k capped by rung policy -- deliberate, not a gap.",
    "obliq": "Twitter rebuild. Table-only numbers, eval_size=126.",
    "niah_contra": "Dense 2k conflicts: 0.164 on disk vs 0.988 in the July table, and the two "
                   "64k/128k builds disagree with each other -- the ladder needs a rebuild.",
    "contra_real": "PubMed multi-claim, scored on the IID realistic-mode ladder that matches the "
                   "training generator. 2k column is the n=56 / rung_2560 file.",
    # ⚠ render_artifact.py still says "Dense 32k ungraded" here and "dense 16k ungraded" for
    # absence. Both are stale: suite_table.json carries a graded number in each of those cells
    # (strmatch dense 32k = 0.9944, absence dense 16k = 0.9315 on eval_size=148), and the page
    # itself renders them. The notes are corrected here to match the data.
    "strmatch": "Rung labels run 0.60x of the stated tokens.",
    "textgroups": "Labels drift 0.89x to 1.55x across the ladder.",
    "scifact": "eval_size=300. The old dense-collapse verdict is stale.",
}

# The only rows whose per-rung corpus size is stated unambiguously by the artifact itself. Every
# other 2k-32k row is left blank rather than joined against hardneg_audit.json, which carries
# three ladder variants per task (orig / rebuilt / ext32k) with no field saying which one was
# graded --- guessing there would put a wrong N next to a right score.
N_DOCS_LADDER = {"contra_real": {2048: 56, 4096: 92, 8192: 187, 16384: 379, 32768: 762}}

# The IID contradiction rebuild starts one step up --- n=44 sat below the training minimum of 52
# --- so its "2k" column is the rung_2560 file, and that is the label the token audit measured.
AUDIT_RUNG_ALIAS = {("contradiction_iid", 2048): 2560}

# The stop-token bug: `stop_token_ids` ships only <|endoftext|> (248044), but our SFT targets end
# the assistant turn with <|im_end|> (248046), so vLLM never stops and the model repeats itself to
# the token cap. Tasks graded with stop="newline" get an accidental text-level cut; the stop="eos"
# ones do not. It is not uniformly harmless --- adding the stop token moved outlier fix-k from
# 0.3191 to 0.3913 at n=220 --- so the affected cells are lower bounds until the re-eval lands.
# Two signals, because neither alone is complete. A parse_rate below 1.0 is the bug caught in the
# act --- the answer ran past the token cap and the grader could not parse it back out --- but a
# task can ramble and still parse, and many graders never record a parse_rate at all. So the
# measured rate is used where it exists, AND the rows the artifact's to-do list names are flagged
# from the depth it names regardless.
STOP_TOKEN_ROWS = {"reorder", "grouping", "outlier_scalek", "outlier_fixk"}
STOP_TOKEN_FROM = 8192
STOP_TOKEN_CAVEAT = ("stop='eos' + the <|im_end|> stop-token bug: generation runs to the token "
                     "cap, so this is a LOWER BOUND, not a measurement, pending re-eval")

# Individual cells the artifact says are not quotable even though a number exists for them.
#
# niah_contra@2048 CAME OFF this list on 2026-08-14: build_suite_table.py now substitutes the July
# table's 0.988 for the implausible on-disk dense 0.164 (CELL_OVERRIDES, on Prasann's instruction),
# so the pair the export sees is 0.988/0.984 and is comparable again. Keeping the drop here would
# exclude a cell the suite table itself stands behind.
CELL_DROP = {}

# ── length generalization ─────────────────────────────────────────────────────────────
# Injection safety, from the gold_semantics table in debug/ctc_modelscale/expand_ctc_rung.py.
# The long rungs are built by injecting filler documents. That is sound when gold is defined by a
# relation to something specific, and unsound when gold is defined by absence or by a structure
# over the whole corpus, because an injected document then satisfies the gold condition without
# being labelled. The unsound rows were built and graded before the rule existed; they are
# exported with plotted_in_figure=no rather than dropped, so the retraction stays visible.
GOLD_SEMANTICS = {
    "nq": "query_match", "hotpotqa": "query_match", "msmarco": "query_match",
    "rerank": "query_match", "niah": "query_match", "qdmatch_nq": "query_match",
    "oolong": "query_match", "contradiction_iid": "pairwise",
    "absence_gutenberg": "absence", "xabsence": "absence",
    "outlier": "structural", "outlier_amzn": "structural", "cycle": "structural",
    "textgroups": "structural", "outlier_fixedM": "structural",
}
UNSAFE = {"absence", "structural"}

# length-gen result directory -> the suite row it extends (for the 32k in-ladder anchor).
LG_TO_ROW = {
    "absence_gutenberg": "absence", "contradiction_iid": "contra_real", "hotpotqa": "hpqa",
    "msmarco": "msmarco", "niah": "niah_contra", "nq": "nq", "outlier": "outlier_scalek",
    "outlier_amzn": "outlier_amazon", "outlier_fixedM": "outlier_fixk",
    "qdmatch_nq": "qdmatch_nq", "rerank": "rerank", "textgroups": "textgroups",
    "oolong": "oolong",
    "cycle": None,  # cycle is not one of the 22 suite rows, so it has no in-ladder anchor
}

# oolong's long rungs were run by the oolong driver and land in its own results tree, not under
# lengthgen_results/. 128k added 2026-08-14 (eval_size 669, partial_credit 0.5750).
OOLONG_LONG = [
    REPO / "debug/ctc_oolong_eval/results/ctcms-oolong-full-4b-vsl/grade_65536.json",
    REPO / "debug/ctc_oolong_eval/results/ctcms-oolong-full-4b-vsl/grade_131072.json",
]

# The two niah long-rung builds disagree with each other by 0.73 f1 at the same length, so no
# niah long-rung number is quotable until the ladder is rebuilt.
NIAH_DROP = ("two independent 64k/128k niah builds disagree (0.736 vs 0.002 at 64k); "
             "ladder needs a rebuild before any of it is quoted")


def se(p, n):
    """Binomial standard error sqrt(p(1-p)/n) -- an UPPER bound for a [0,1] per-example metric."""
    if p is None or not n:
        return ""
    return round(math.sqrt(max(p * (1 - p), 0.0) / n), 4)


def rel_gap(full, chunked):
    """1 - chunked/full, blank where undefined (missing arm or a zero full arm)."""
    if full in (None, 0, "") or chunked in (None, ""):
        return ""
    return round(1.0 - chunked / full, 6)


def classify(row, cls):
    ctc_class, ctc_order = CLASS[CLASS_OVERRIDE.get(row, cls)]
    return ctc_class, ctc_order


def _eval_size(dense, chunked):
    """One number when both arms agree, otherwise spell out which arm is which."""
    d = dense.get("eval_size") if dense else None
    c = chunked.get("eval_size") if chunked else None
    if d is not None and c is not None and d != c:
        return f"dense={d} chunked={c}"
    return d if d is not None else (c if c is not None else "")


# ── measured token lengths ────────────────────────────────────────────────────────────
def measured_tokens():
    """(task_dir, rung label) -> measured p50 tokens and p50/label, where an audit exists.

    Several ladders are labelled by a token budget they do not hit -- absence runs 3.0-3.6x its
    label, qdmatch hpqa 0.74-0.82x -- which mislabels the x-axis rather than the metric. Where an
    audit measured the rung, both the measurement and the ratio ride along.
    """
    out = {}
    for name in ("rung_token_audit.json", "lengthgen_measured_tokens.json"):
        path = SCALE / name
        if not path.exists():
            continue
        for r in json.load(path.open()):
            out.setdefault((r["task_dir"], r["label"]), (r["p50"], r["p50_over_label"]))
    return out


# ── the 2k-32k grid ───────────────────────────────────────────────────────────────────
def parse_rates():
    """(task_dir, arm, rung) -> parse_rate, from the harvested grades.

    suite_table.json drops parse_rate on the way through, but it is what makes the stop-token bug
    visible per cell rather than per row, so it is joined back on here from the same harvest the
    table was built from. Carried cells (obliq, xabsence) have no harvested grade and stay blank.
    """
    path = SUITE / "harvested_grades.json"
    if not path.exists():
        return {}
    out = {}
    for r in json.load(path.open()):
        out[(r["task"], r["arm"], r["rung"])] = r.get("parse_rate")
    return out


def export_grid():
    table = json.load((SUITE / "suite_table.json").open())
    tok = measured_tokens()
    pr = parse_rates()
    rows = []
    for e in table:
        row, task_dir = e["row"], e["task_dir"]
        ctc_class, ctc_order = classify(row, e["class"])
        for rung in RUNGS:
            d = e["cells"]["dense"].get(str(rung))
            c = e["cells"]["chunked"].get(str(rung))
            full = d["value"] if d else None
            chunk = c["value"] if c else None

            # plotted_in_figure answers "is this a number the page currently stands behind".
            src = (d or c or {}).get("source")
            if full is None and chunk is None:
                plotted, reason = "no", "no grade JSON: this cell has never been evaluated"
            elif src == "superseded":
                plotted, reason = "no", ("superseded by the in-flight rebuild -- training CE never "
                                         "left the 0.78 flatline, so this is not a trained model")
            elif (row, rung) in CELL_DROP:
                plotted, reason = "no", CELL_DROP[(row, rung)]
            else:
                plotted, reason = "yes", ""

            # Caveats qualify a number without invalidating it; drop_reason is only ever set on a
            # row that is actually dropped, so the two columns never have to be read together.
            caveats = []
            if plotted == "yes" and (full is None or chunk is None):
                caveats.append("one arm only; relative_gap is undefined")
            if src == "table-2026-07-27":
                caveats.append("carried from the 2026-07-27 table; no grade JSON on disk")
            n = d["eval_size"] if d else (c["eval_size"] if c else None)
            worst = max([x for x in (d and d["se"], c and c["se"]) if x], default=0)
            if n and n < 500:
                caveats.append(f"eval_size={n} (<500): SE up to +/-{worst:.3f}, "
                               "do not read a smaller difference as real")
            audit_rung = AUDIT_RUNG_ALIAS.get((task_dir, rung), rung)
            pr_full = pr.get((task_dir, "dense", audit_rung))
            pr_chunk = pr.get((task_dir, "chunked", audit_rung))
            if plotted == "yes":
                low = [f"{a} parse_rate={p}"
                       for a, p in (("dense", pr_full), ("chunked", pr_chunk))
                       if p is not None and p < 1.0]
                if low:
                    caveats.append(f"{', '.join(low)}: {STOP_TOKEN_CAVEAT}")
                elif row in STOP_TOKEN_ROWS and rung >= STOP_TOKEN_FROM:
                    caveats.append(STOP_TOKEN_CAVEAT)

            p50, ratio = tok.get((task_dir, audit_rung), ("", ""))
            rows.append({
                "figure": "ctc_suite_grid",
                "series": STATUS[row],
                "task": row,
                "task_label": LABEL[row],
                "ctc_class": ctc_class,
                "ctc_order": ctc_order,
                "metric": METRIC[e["metric"]],
                "metric_key": e["metric"],
                "context_tokens": rung,
                "context_tokens_measured_p50": p50,
                "measured_over_label": ratio,
                "n_docs": N_DOCS_LADDER.get(row, {}).get(rung, ""),
                "source": "CTC suite (token ladder), Qwen3.5-4B",
                "full_attention": "" if full is None else full,
                "chunked_attention": "" if chunk is None else chunk,
                "relative_gap": rel_gap(full, chunk),
                "absolute_gap": "" if full is None or chunk is None else round(full - chunk, 4),
                "eval_size": _eval_size(d, c),
                "full_se": d["se"] if d else "",
                "chunked_se": c["se"] if c else "",
                "full_parse_rate": "" if pr_full is None else pr_full,
                "chunked_parse_rate": "" if pr_chunk is None else pr_chunk,
                "plotted_in_figure": plotted,
                "drop_reason": reason,
                # " | " separates caveats; several of them contain a semicolon of their own.
                "caveats": " | ".join(caveats),
                "ladder": (d or c or {}).get("ladder") or "",
                "task_dir": task_dir,
                "row_note": NOTE.get(row, ""),
            })
    _write("ctc_suite_grid_data.csv", rows)
    return rows


# ── model scale ───────────────────────────────────────────────────────────────────────
SCALE_LABEL = {"0.8b": "Qwen3.5-0.8B", "2b": "Qwen3.5-2B", "4b": "Qwen3.5-4B"}

# Why a scale cell can be missing. All are documented on the artifact page; none is a result.
#
# ⚠ ("contradiction", "0.8b") WAS HERE AND IS NOT ANY MORE. That entry recorded the 0.8B x
# chunked-mix x seq_len 40960 OOM as an open mystery. The arm was re-run and landed complete on
# 2026-08-15 (0.7980/0.7547/0.6871/0.6089/0.5199 on contradiction_iid, all five rungs, eval_size
# 500, parse_rate 1.0). Leaving the entry in would have stamped "do NOT read the absence as a gap"
# onto cells that are now populated. reorder's 0.8B chunked arm is still genuinely absent.
SCALE_MISSING = {
    ("fiqa", "2b"): "fiqa was only ever run at 0.8B and 4B; the 2B pair was not launched",
    ("reorder", "0.8b"): "unexplained 0.8B x chunked-mix x seq_len 40960 OOM on both saturn and "
                         "jupiter while 2B/4B train fine; four hypotheses tested and refuted, "
                         "cause still unknown -- do NOT read the absence as a gap",
}


def export_scale():
    table = json.load((SUITE / "scale_table.json").open())
    grid = {e["row"]: e for e in json.load((SUITE / "suite_table.json").open())}
    rows = []
    for e in table:
        row = e["row"]
        ctc_class, ctc_order = classify(row, grid[row]["class"])
        # Rungs this task was ever run at, at ANY scale. A rung missing from one scale but present
        # in another is a hole worth seeing (fiqa has no 2B column at all); a rung missing from
        # every scale is just where the ladder ends (reorder stops at 16k by rung policy), and
        # emitting empty tail rows for it would invent 15 rows of nothing.
        task_rungs = [r for r in RUNGS
                      if any(str(r) in arm for a in e["scales"].values() for arm in a.values())]
        for scale, arms in e["scales"].items():
            dense, chunked = arms.get("dense", {}), arms.get("chunked", {})
            for rung in task_rungs:
                d, c = dense.get(str(rung)), chunked.get(str(rung))
                full = d["value"] if d else None
                chunk = c["value"] if c else None
                reason = ""
                if full is None and chunk is None:
                    reason = SCALE_MISSING.get((e["task"], scale), "neither arm run at this scale")
                elif chunk is None:
                    reason = SCALE_MISSING.get((e["task"], scale), "no chunked run at this scale")
                elif full is None:
                    reason = SCALE_MISSING.get((e["task"], scale), "no dense run at this scale")
                rows.append({
                    "figure": "ctc_scale",
                    "series": SCALE_LABEL[scale],
                    "model_scale": scale,
                    "task": row,
                    "task_label": e["label"],
                    "ctc_class": ctc_class,
                    "ctc_order": ctc_order,
                    "metric": METRIC[e["metric"]],
                    "metric_key": e["metric"],
                    "context_tokens": rung,
                    "n_docs": N_DOCS_LADDER.get(row, {}).get(rung, ""),
                    "source": "CTC model-scale runs (identical shards, identical ladders)",
                    "full_attention": "" if full is None else full,
                    "chunked_attention": "" if chunk is None else chunk,
                    "relative_gap": rel_gap(full, chunk),
                    "absolute_gap": "" if full is None or chunk is None else round(full - chunk, 4),
                    "eval_size": _eval_size(d, c),
                    "full_se": (d or {}).get("se", ""),
                    "chunked_se": (c or {}).get("se", ""),
                    "full_parse_rate": (d or {}).get("parse_rate") if d else "",
                    "chunked_parse_rate": (c or {}).get("parse_rate") if c else "",
                    "plotted_in_figure": "yes" if (full is not None and chunk is not None) else "no",
                    "drop_reason": reason,
                    # A parse_rate below 1.0 is the stop-token bug showing up directly: the answer
                    # ran past the token cap and the grader could not parse it back out.
                    "caveats": " | ".join(
                        f"{arm} parse_rate={pr}: {STOP_TOKEN_CAVEAT}"
                        for arm, pr in (("dense", (d or {}).get("parse_rate")),
                                        ("chunked", (c or {}).get("parse_rate")))
                        if pr is not None and pr < 1.0),
                })
    # None -> blank, so an unrecorded parse_rate does not read as the string "None"
    for r in rows:
        for k in ("full_parse_rate", "chunked_parse_rate"):
            if r[k] is None:
                r[k] = ""
    _write("ctc_scale_data.csv", rows)
    return rows


# ── length generalization ─────────────────────────────────────────────────────────────
def _manifest():
    path = SCALE / "lengthgen_rung_manifest/manifest.json"
    if not path.exists():
        return {}
    out = {}
    for eval_path, m in json.load(path.open()).items():
        out[os.path.basename(eval_path)] = m
    return out


def _lengthgen_grades():
    """(task_dir, rung file) -> {arm: grade}, over every long-rung grade JSON on disk."""
    found = {}
    files = sorted(glob.glob(str(SCALE / "lengthgen_results/*/*.json")))
    oolong_files = [str(p) for p in OOLONG_LONG if p.exists()]
    files.extend(oolong_files)
    for f in files:
        g = json.load(open(f))
        task_dir = ("oolong" if f in oolong_files
                    else os.path.basename(os.path.dirname(f)))
        rung_file = os.path.basename(g.get("eval_data", ""))
        arm = "dense" if g.get("mode") == "full" else "chunked"
        found.setdefault((task_dir, rung_file), {})[arm] = g
    return found


def export_lengthgen():
    grid = {e["row"]: e for e in json.load((SUITE / "suite_table.json").open())}
    manifest = _manifest()
    grades = _lengthgen_grades()
    rows = []

    def base_fields(task_dir, metric_key):
        row = LG_TO_ROW.get(task_dir)
        cls = grid[row]["class"] if row in grid else {"cycle": "N2"}.get(task_dir, "")
        ctc_class, ctc_order = classify(row, cls) if cls else ("", "")
        return {
            "task": row or task_dir,
            "task_label": LABEL.get(row, task_dir),
            "ctc_class": ctc_class,
            "ctc_order": ctc_order,
            "metric": METRIC.get(metric_key, metric_key),
            "metric_key": metric_key,
        }

    # 1. the 32k in-ladder anchor, read from the grid so the two files cannot disagree
    anchored = sorted({t for (t, _) in grades} - {"outlier_fixedM"})
    for task_dir in anchored:
        row = LG_TO_ROW.get(task_dir)
        if row not in grid:
            continue
        e = grid[row]
        d, c = e["cells"]["dense"].get("32768"), e["cells"]["chunked"].get("32768")
        if not d and not c:
            continue
        full = d["value"] if d else None
        chunk = c["value"] if c else None
        rows.append({
            "figure": "ctc_length_generalization",
            "series": LABEL.get(row, task_dir),
            **base_fields(task_dir, e["metric"]),
            "context_tokens": 32768,
            "context_tokens_measured_p50": "",
            "n_docs": N_DOCS_LADDER.get(row, {}).get(32768, ""),
            "source": "CTC suite 32k rung (in-ladder anchor)",
            "gold_semantics": GOLD_SEMANTICS.get(task_dir, ""),
            "full_attention": "" if full is None else full,
            "chunked_attention": "" if chunk is None else chunk,
            "relative_gap": rel_gap(full, chunk),
            "eval_size": _eval_size(d, c),
            "full_se": d["se"] if d else "",
            "chunked_se": c["se"] if c else "",
            "parse_rate": "",
            "plotted_in_figure": "yes",
            "drop_reason": "",
            "caveats": "",
            "rung_file": "",
        })

    # 2. the 64k / 128k rungs
    for (task_dir, rung_file), arms in sorted(grades.items()):
        if task_dir == "outlier_fixedM":
            continue
        g = arms.get("dense") or arms.get("chunked")
        m = manifest.get(rung_file, {})
        # The rung file is named by its measured p50, so it is the length when no manifest row
        # exists; the manifest is authoritative where it does.
        p50 = m.get("p50") or int(re.search(r"(\d+)", rung_file).group(1))
        full = arms["dense"]["metric_value"] if "dense" in arms else None
        chunk = arms["chunked"]["metric_value"] if "chunked" in arms else None
        n_full = arms["dense"]["eval_size"] if "dense" in arms else None
        n_chunk = arms["chunked"]["eval_size"] if "chunked" in arms else None

        semantics = GOLD_SEMANTICS.get(task_dir, "")
        if task_dir == "niah":
            plotted, reason = "no", NIAH_DROP
        elif semantics in UNSAFE:
            plotted, reason = "no", (
                f"RETRACTED: gold_semantics={semantics}, so injected filler documents silently "
                "satisfy the gold condition and become unlabelled true positives. Needs "
                "natively-generated rungs, not padding")
        else:
            plotted, reason = "yes", ""

        rows.append({
            "figure": "ctc_length_generalization",
            "series": LABEL.get(LG_TO_ROW.get(task_dir), task_dir),
            **base_fields(task_dir, g["metric_name"]),
            "context_tokens": p50,
            "context_tokens_measured_p50": p50,
            "n_docs": m.get("n_docs", ""),
            "source": "length-generalization rungs (4B, past the 32k training ceiling)",
            "gold_semantics": semantics,
            "full_attention": "" if full is None else round(full, 4),
            "chunked_attention": "" if chunk is None else round(chunk, 4),
            "relative_gap": rel_gap(full, chunk),
            "eval_size": (f"dense={n_full} chunked={n_chunk}"
                          if n_full and n_chunk and n_full != n_chunk
                          else (n_full or n_chunk or "")),
            "full_se": se(full, n_full),
            "chunked_se": se(chunk, n_chunk),
            "parse_rate": g.get("parse_rate") if g.get("parse_rate") is not None else "",
            "plotted_in_figure": plotted,
            "drop_reason": reason,
            "caveats": ("dense arm only; no chunked long-rung run, so relative_gap is undefined"
                        if chunk is None else ""),
            "rung_file": rung_file,
        })

    # 3. the outlier fix-k n sweep: a DOCUMENT-count axis, not a length axis. It is here because
    #    it is what replaced fix-k's 32k rung on the page --- the apparent cliff at 32k was rung
    #    spacing (111 -> 220 in one jump), and filling the gap shows a smooth monotone decline
    #    that continues past the n ~ U[14,220] training support.
    for (task_dir, rung_file), arms in sorted(grades.items()):
        if task_dir != "outlier_fixedM":
            continue
        g = arms["dense"]
        n = int(re.search(r"n(\d+)", rung_file).group(1))
        rows.append({
            "figure": "ctc_length_generalization",
            "series": "outlier fix-k n sweep",
            **base_fields(task_dir, g["metric_name"]),
            "context_tokens": "",
            "context_tokens_measured_p50": "",
            "n_docs": n,
            "source": "outlier fix-k n sweep (document axis, not tokens; dense only)",
            "gold_semantics": GOLD_SEMANTICS[task_dir],
            "full_attention": round(g["metric_value"], 4),
            "chunked_attention": "",
            "relative_gap": "",
            "eval_size": g["eval_size"],
            "full_se": se(g["metric_value"], g["eval_size"]),
            "chunked_se": "",
            "parse_rate": g.get("parse_rate", ""),
            "plotted_in_figure": "yes",
            "drop_reason": "",
            "caveats": "dense arm only; no chunked run, so relative_gap is undefined",
            "rung_file": rung_file,
        })

    rows.sort(key=lambda r: (r["task"], int(r["context_tokens"] or 0),
                             int(r["n_docs"]) if str(r["n_docs"]).isdigit() else 0))
    _write("ctc_length_generalization_data.csv", rows)
    return rows


def _write(name, rows):
    OUT.mkdir(exist_ok=True)
    path = OUT / name
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    plotted = sum(1 for r in rows if r["plotted_in_figure"] == "yes")
    print(f"wrote {path}  ({len(rows)} rows, {plotted} plotted)")


if __name__ == "__main__":
    export_grid()
    export_scale()
    export_lengthgen()
