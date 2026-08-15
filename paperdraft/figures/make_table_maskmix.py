"""TABLE-MASKMIX -- what curriculum mask-mixing buys, per task.

Emits `paperdraft/iclr2026/sections/table_maskmix.tex` (+ a tidy CSV next to the other figure data).

The quantity is the MASK-MIXING GAIN:

    gain(rung) = chunked-mix(rung) - pure-chunked(rung)

Both arms use the identical document-chunked attention mask at eval time; they differ ONLY in
training, where `chunked-mix` anneals a per-step coin p_standard 0.8 -> 0.0 and `chunked` never
sees an unrestricted step. Dense is carried as a reference column so the reader can see how much
of the dense->chunked drop mask-mixing recovers, but it is NOT part of the gain.

⚠ THE SUITE'S "chunked" COLUMN IS THE MIXED ARM. Every one of the 31 chunked runs in
debug/ctc_modelscale/LAUNCH_LEDGER.tsv is `chunked-mix`, so until the pure-chunked control landed
(debug/ctc_purechunk/, 2026-08-14/15) this gain had never been measured -- the grid's chunked
column silently included the curriculum. Reading suite_table.json's `chunked` as "no mask-mixing"
is the mistake this table exists to prevent.

⚠ ALL FIVE TASKS ARE MATCHED ON LADDER AND HYPERPARAMETERS. The pure-chunked runs reused the
chunked-mix launch knobs exactly (--epochs 1, --lr 5e-5, --global-batch 8, per-task seq_len read
from the shard metadata); see the header of debug/ctc_purechunk/launch_purechunk.sh. Contradiction
is scored on contradiction_iid for BOTH arms -- the ladder that matches the training shard.

⚠ CONTRADICTION AND REORDER PURE-CHUNKED NEVER FIT THE TRAINING DATA, AND THAT IS THE POINT, NOT A
BUG -- but it is also not the same claim as "the mask cannot express the task". Contradiction's
pure-chunked CE is FLAT at 1.175 -> 1.071 across the whole run (deciles in
debug/ctc_purechunk/trainlogs/), while fiqa (0.48 -> 0.065) and qdmatch_fiqa (0.57 -> 0.27) on the
identical recipe descend normally. So the recipe and infrastructure are fine and the failure is
task-specific -- but a flat CE is consistent with BOTH "chunked attention cannot represent
pair-finding" and "this optimization run failed". Quote the number as measured; do not upgrade it
to an expressivity claim without the matched chunked-mix CE, which was not recoverable (the 0.8B
cmix contradiction job on record crashed, and the 4B July run's logs are gone).

    python paperdraft/figures/make_table_maskmix.py
"""
import csv
import glob
import json
import math
import os
import pathlib

FIGURE_LABEL = "TABLE-MASKMIX"
HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent.parent
SUITE = REPO / "debug/ctc_final_suite"
PUREDIR = REPO / "debug/ctc_purechunk/harvest"
OUT_TEX = REPO / "paperdraft/iclr2026/sections/table_maskmix.tex"

RUNGS = [2048, 4096, 8192, 16384, 32768]

# suite row -> (display name, pure-chunked S3 group, CTC class label)
#
# The pure-chunked group names are NOT guessable from the row name and each one was checked against
# the `eval_data` path inside the JSON:
#   * contradiction uses ..._MATCHED -- the plain `contradiction_iid__purechunk` directory holds an
#     earlier pass run in mode=full (patch never applied), which would compare a DENSE evaluation
#     against a chunked one and read as a huge spurious gain.
#   * `outlier` is the scale-k row (suite row outlier_scalek); `outlier_fixedM` is a different task.
TASKS = [
    ("fiqa",           "BEIR FiQA",      "fiqa__purechunk",                     r"$O_T(N)$"),
    ("outlier_scalek", "Outlier (scale-$k$)", "outlier__purechunk",             r"$O_T(NM)$"),
    ("qdmatch_fiqa",   "QDmatch (FiQA)", "qdmatch_fiqa__purechunk",             r"$O_T(N^2)$"),
    ("reorder",        "Reordering",     "reorder__purechunk",                  r"$O_T(N^2)$"),
    ("contra_real",    "Contradiction",  "contradiction_iid__purechunk_MATCHED", r"$O_T(N^2)$"),
]

METRIC_TEX = {"gold_id_f1": "gold-ID F1", "set_f1": "set-F1",
              "pair_f1": "pair-F1", "kendall_tau": r"Kendall $\tau$"}

# The realistic contradiction ladder starts at 2560, not 2048: n=44 sits below the training minimum
# of 52, so the IID rebuild starts at n=56. Same alias the grid and the scale table apply.
RUNG_ALIAS = {"contra_real": {2560: 2048}}

# Flagged inline in the table via a footnote marker rather than silently averaged in.
FLAT_CE = {"contra_real", "reorder"}

# ⚠ A NEAR-ZERO GAIN AT THE DEEPEST RUNG MEANS TWO COMPLETELY DIFFERENT THINGS AND THE TABLE MUST
# SAY WHICH. On FiQA the gain is ~0 because the chunked arm already matches dense -- there is
# nothing to recover. On reorder it is ~0 because BOTH arms sit on the floor: dense itself scores
# 0.047 at 16k, so the task is unsolvable at that depth even with full attention, and mask-mixing
# is being credited with recovering headroom that does not exist. Rows whose DENSE arm is below
# FLOOR at the deepest rung are marked so "+0.000" is never read as "mixing does not help" --
# reorder's gain at 2k is +0.473. FLOOR matches the 0.15 used by fig1_gap_vs_context.py.
FLOOR = 0.15


def load_pure(group, row):
    out = {}
    for f in glob.glob(str(PUREDIR / group / "rung_*.json")):
        d = json.load(open(f))
        rung = int(os.path.basename(f).split("_")[1].split(".")[0])
        rung = RUNG_ALIAS.get(row, {}).get(rung, rung)
        # A chunked-mode result whose mask patch never fired is a DENSE number wearing a chunked
        # label -- the exact failure that published a bogus row once already. Refuse it.
        if d.get("mode") == "chunked" and not (d.get("patch_debug") or {}).get("applied"):
            raise SystemExit(f"FATAL: {f} is mode=chunked but patch_debug.applied is falsy")
        out[rung] = {"value": d["metric_value"], "eval_size": d.get("eval_size"),
                     "mode": d.get("mode")}
    return out


def se(p, n):
    if p is None or not n:
        return None
    return math.sqrt(max(p * (1 - p), 0.0) / n)


def main():
    suite = {e["row"]: e for e in json.load((SUITE / "suite_table.json").open())}
    rows, csv_rows = [], []

    for row, name, group, cls in TASKS:
        e = suite[row]
        dense, cmix = e["cells"]["dense"], e["cells"]["chunked"]
        pure = load_pure(group, row)

        gains, deepest = [], None
        for rung in RUNGS:
            D = dense.get(str(rung))
            C = cmix.get(str(rung))
            P = pure.get(rung)
            if C is None or P is None:
                continue
            g = C["value"] - P["value"]
            gains.append(g)
            deepest = (rung, g, C["value"], P["value"], D["value"] if D else None)
            csv_rows.append({
                "table": "maskmix", "task": row, "task_label": name, "ctc_order": cls,
                "metric_key": e["metric"], "context_tokens": rung,
                "dense": "" if D is None else D["value"],
                "chunked_mix": C["value"], "pure_chunked": P["value"],
                "maskmix_gain": round(g, 4),
                "eval_size": P.get("eval_size"),
                "gain_se": round(math.hypot(se(C["value"], 500) or 0, se(P["value"], 500) or 0), 4),
                "flat_ce_purechunk": "yes" if row in FLAT_CE else "no",
            })
        if not gains:
            continue
        shallow = min(g for g in [gains[0]])
        # Ladder means per arm. Both average over the SAME rung set, so mean(mix) - mean(nomix) is
        # exactly the mean per-rung gain -- the table's two columns and the gain reported elsewhere
        # cannot disagree. Only rungs where both arms exist contribute, which is what `gains` walks.
        paired = [(C, P) for rung in RUNGS
                  for C in [cmix.get(str(rung))] for P in [pure.get(rung)]
                  if C is not None and P is not None]
        mix_avg = sum(c["value"] for c, _ in paired) / len(paired)
        nomix_avg = sum(p["value"] for _, p in paired) / len(paired)
        rows.append({
            "name": name, "cls": cls, "metric": METRIC_TEX[e["metric"]],
            "mix_avg": mix_avg, "nomix_avg": nomix_avg,
            "n_rungs": len(gains), "avg": sum(gains) / len(gains),
            "deep_rung": deepest[0], "deep_gain": deepest[1],
            "deep_cmix": deepest[2], "deep_pure": deepest[3], "deep_dense": deepest[4],
            "shallow_gain": shallow,
            "flag": row in FLAT_CE,
            # Floor-limited: the dense arm itself has collapsed at this rung, so neither chunked
            # arm has room to differ and the deepest-rung gain is not informative.
            "floored": deepest[4] is not None and deepest[4] < FLOOR,
        })

    with (HERE / "data/ctc_maskmix_data.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(csv_rows[0]))
        w.writeheader()
        w.writerows(csv_rows)

    def fmt(v, sign=False):
        if v is None:
            return "--"
        return f"{v:+.3f}" if sign else f"{v:.3f}"

    by = {r["name"]: r for r in rows}
    fiqa_avg = fmt(by["BEIR FiQA"]["avg"], True)
    contra_avg = fmt(by["Contradiction"]["avg"], True)
    out_avg = fmt(by["Outlier (scale-$k$)"]["avg"], True)
    reo_sh = fmt(by["Reordering"]["shallow_gain"], True)
    qd_sh = fmt(by["QDmatch (FiQA)"]["shallow_gain"], True)

    # The caveats that used to live in the caption. With no caption they have nowhere to go inside
    # the table, so they are emitted as LaTeX comments directly above it AND printed to stdout --
    # they must reach the body text, because two of these five numbers are misleading without them.
    NOTES = [
        "CAVEATS -- the caption is one sentence, so these belong in the body text:",
        "  * Both columns are ladder-mean scores. They average over the SAME rungs, so",
        "    Mix - No Mix is exactly the mean per-rung mask-mixing gain:",
        "    " + ", ".join(f"{r['name']} {r['avg']:+.3f}" for r in rows) + ".",
        "  * Each task uses its own metric (gold-ID F1 / set-F1 / pair-F1 / Kendall tau), so the",
        "    columns are NOT comparable across rows -- read a row, not a column.",
        "  * Both arms use the identical document-chunked mask at eval; they differ only in",
        "    training, where Mix anneals an unrestricted-attention coin p_standard 0.8 -> 0.",
        "    Qwen3.5-4B, identical shards / ladders / hyperparameters.",
        f"  * eval_size = 500 per rung, so a gain under ~0.04 is within noise ({fiqa_avg} on FiQA is not",
        "    distinguishable from zero).",
        "  * Each number is the mean per-rung gain over the 2k-32k ladder: 5 rungs, except",
        "    reordering, whose ladder stops at 16k (4 rungs).",
        f"  * Reordering and QDmatch earn their whole gain at the shallow rungs ({reo_sh} and {qd_sh}",
        "    at 2k, +0.000 and +0.006 at the deepest). By the deepest rung the FULL-attention arm",
        f"    has itself fallen below {FLOOR}, so no arm has headroom left to differ.",
        "  * Pure-chunked training CE never descended on contradiction (1.175 -> 1.071, flat across",
        "    the whole run) or reordering, while FiQA and QDmatch on the identical recipe converged",
        "    normally. So the collapse is task-specific rather than broken infrastructure -- but it",
        "    is a failure to FIT the training set, and is not by itself a statement about what the",
        "    chunked mask can represent.",
    ]

    L = []
    L.append(r"% " + FIGURE_LABEL + r" -- generated by paperdraft/figures/make_table_maskmix.py")
    L.append(r"% Do not hand-edit: re-run the script instead.")
    L.append(r"% Requires \usepackage{wrapfig} (and booktabs). A wraptable must NOT start in a")
    L.append(r"% paragraph's first line or inside a list -- put it just before the paragraph it")
    L.append(r"% should sit beside, or wrapfig silently drops the wrap and it overlaps the text.")
    L.extend("% " + n for n in NOTES)
    L.append(r"\begin{wraptable}{r}{0.46\textwidth}")
    L.append(r"\centering")
    L.append(r"\small")
    L.append(r"\setlength{\tabcolsep}{5pt}")
    # One row per task, one number each. Everything trimmed from this table -- per-rung arm scores,
    # the deepest-rung gain, the rung count -- is still in data/ctc_maskmix_data.csv.
    #
    # ⚠ THE \dagger AND \ddagger MARKERS ARE GONE BECAUSE THE CAPTION THAT DEFINED THEM IS GONE.
    # A superscript with no matching footnote is worse than no marker at all -- it tells the reader
    # a caveat exists and then refuses to say what it is. The caveats themselves did NOT go away:
    # they are emitted as LaTeX comments above the table and printed to stdout (see NOTES), and
    # they belong in the body text. Both `flag` and `floored` are still computed and still printed.
    L.append(r"\begin{tabular}{lrr}")
    L.append(r"\toprule")
    L.append(r"Task & No Mix & Mix \\")
    L.append(r"\midrule")
    for r in rows:
        L.append(f"{r['name']} & {fmt(r['nomix_avg'])} & "
                 f"\\textbf{{{fmt(r['mix_avg'])}}} \\\\")
    L.append(r"\bottomrule")
    L.append(r"\end{tabular}")
    # ⚠ ONE SENTENCE, BUT THE "read across a row" CLAUSE IS NOT PADDING. With the metric column
    # gone, this table stacks gold-ID F1, set-F1, pair-F1 and Kendall tau in one pair of columns;
    # nothing stops a reader comparing FiQA's 0.283 to reordering's 0.011 as if they were the same
    # scale. The rows are ordered by traversal complexity so the trend is still readable downward.
    L.append(r"\caption{Curriculum mask-mixing is worth almost nothing on $O_T(N)$ retrieval and "
             r"becomes the difference between learning the task and not learning it at all as "
             r"corpus-traversal complexity rises; both columns are ladder-mean scores for "
             r"document-chunked attention trained without and with mixing, each task on its own "
             r"metric, so read across a row rather than down a column.}")
    L.append(r"\label{tab:maskmix}")
    L.append(r"\end{wraptable}")

    OUT_TEX.parent.mkdir(parents=True, exist_ok=True)
    OUT_TEX.write_text("\n".join(L) + "\n")

    print(f"[{FIGURE_LABEL}] wrote {OUT_TEX}")
    print(f"[{FIGURE_LABEL}] wrote {HERE/'data/ctc_maskmix_data.csv'}  ({len(csv_rows)} rows)")
    print()
    for n in NOTES:
        print("  " + n)
    print()
    hdr = f"{'task':22s} {'CTC':10s} {'rungs':>5s} {'mean gain':>10s} {'deepest':>16s}"
    print(hdr); print("-" * len(hdr))
    for r in rows:
        print(f"{r['name']:22s} {r['cls']:10s} {r['n_rungs']:5d} {r['avg']:+10.3f} "
              f"{r['deep_gain']:+9.3f} @{r['deep_rung']//1024}k{'  (flat CE)' if r['flag'] else ''}")


if __name__ == "__main__":
    main()
