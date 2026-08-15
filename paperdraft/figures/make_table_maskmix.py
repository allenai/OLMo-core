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
        rows.append({
            "name": name, "cls": cls, "metric": METRIC_TEX[e["metric"]],
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

    L = []
    L.append(r"% " + FIGURE_LABEL + r" -- generated by paperdraft/figures/make_table_maskmix.py")
    L.append(r"% Do not hand-edit: re-run the script instead.")
    L.append(r"\begin{table}[t]")
    L.append(r"\centering")
    L.append(r"\small")
    L.append(r"\setlength{\tabcolsep}{5pt}")
    # One row per task, one number each. Everything trimmed from this table -- per-rung arm scores,
    # the deepest-rung gain, the rung count -- is still in data/ctc_maskmix_data.csv.
    #
    # ⚠ THE \ddagger FLOOR MARKER WAS DROPPED WITH THE "At deepest" COLUMN, NOT FORGOTTEN. It
    # existed to warn that reorder's +0.000 and qdmatch's +0.006 at the deepest rung were
    # floor artefacts (the full-attention arm is itself under FLOOR there, so no arm has headroom).
    # With that column gone there is no number left for it to qualify. The underlying fact still
    # shapes how the MEAN should be read -- both tasks earn their whole gain at the shallow rungs --
    # so it moves into the caption rather than disappearing. `floored` is still computed and still
    # printed to stdout.
    L.append(r"\begin{tabular}{llr}")
    L.append(r"\toprule")
    L.append(r"Task & CTC & Mask-mixing gain \\")
    L.append(r"\midrule")
    for r in rows:
        star = r"$^{\dagger}$" if r["flag"] else ""
        L.append(f"{r['name']}{star} & {r['cls']} & \\textbf{{{fmt(r['avg'], True)}}} \\\\")
    L.append(r"\bottomrule")
    L.append(r"\end{tabular}")
    by = {r["name"]: r for r in rows}
    fiqa_avg = fmt(by["BEIR FiQA"]["avg"], True)
    contra_avg = fmt(by["Contradiction"]["avg"], True)
    out_avg = fmt(by["Outlier (scale-$k$)"]["avg"], True)
    reo_sh = fmt(by["Reordering"]["shallow_gain"], True)
    qd_sh = fmt(by["QDmatch (FiQA)"]["shallow_gain"], True)

    L.append(r"\caption{\textbf{What curriculum mask-mixing buys, by corpus-traversal complexity.}")
    L.append(r"Both chunked arms use the identical document-chunked mask at evaluation time and")
    L.append(r"differ only in training: \emph{+Mix} anneals a per-step unrestricted-attention coin")
    L.append(r"$p_{\text{standard}}\!:\!0.8\!\to\!0$, \emph{Chunked} never sees an unrestricted")
    L.append(r"step. \emph{Gain} is $(\text{+Mix}) - (\text{Chunked})$. Qwen3.5-4B, identical")
    L.append(r"shards, identical ladders, matched hyperparameters; \emph{eval\_size}~$=500$ per")
    L.append(r"rung, so a gain under $\sim\!0.04$ is within noise.")
    # The rung count is no longer a column, so state the ladder coverage here -- reordering's
    # ladder stops at 16k by rung policy, so its mean is over 4 rungs and everything else over 5.
    L.append(r"Each figure is the mean per-rung gain over the 2k--32k ladder (5 rungs; reordering")
    L.append(r"stops at 16k, 4 rungs).")
    L.append(rf"Mixing is worth nothing on $O_T(N)$ retrieval (FiQA, {fiqa_avg}) and is the")
    L.append(rf"difference between learning the task and not learning it at all on contradiction")
    L.append(rf"({contra_avg}) and outlier detection ({out_avg}).")
    # Without the deepest-rung column a reader cannot see that two of these means are front-loaded,
    # which changes what the mean means for those rows. State it.
    L.append(r"\emph{Reordering and QDmatch earn their entire gain at the shallow rungs}: by the")
    L.append(rf"deepest rung the full-attention arm has itself fallen below {FLOOR}, leaving no")
    L.append(rf"headroom for either chunked arm, so their gains run {reo_sh} and {qd_sh} at 2k but")
    L.append(r"$+0.000$ and $+0.006$ at the bottom of the ladder.")
    L.append(r"$^{\dagger}$Pure-chunked training cross-entropy never descended on these two tasks")
    L.append(r"(contradiction: $1.175\!\to\!1.071$, flat across the whole run), while FiQA and")
    L.append(r"QDmatch on the identical recipe converged normally -- so the collapse is")
    L.append(r"task-specific rather than an infrastructure failure, but it reflects a failure to")
    L.append(r"fit the training set and is not by itself a statement about what the mask can")
    L.append(r"represent.}")
    L.append(r"\label{tab:maskmix}")
    L.append(r"\end{table}")

    OUT_TEX.parent.mkdir(parents=True, exist_ok=True)
    OUT_TEX.write_text("\n".join(L) + "\n")

    print(f"[{FIGURE_LABEL}] wrote {OUT_TEX}")
    print(f"[{FIGURE_LABEL}] wrote {HERE/'data/ctc_maskmix_data.csv'}  ({len(csv_rows)} rows)")
    print()
    hdr = f"{'task':22s} {'CTC':10s} {'rungs':>5s} {'mean gain':>10s} {'deepest':>16s}"
    print(hdr); print("-" * len(hdr))
    for r in rows:
        print(f"{r['name']:22s} {r['cls']:10s} {r['n_rungs']:5d} {r['avg']:+10.3f} "
              f"{r['deep_gain']:+9.3f} @{r['deep_rung']//1024}k{'  (flat CE)' if r['flag'] else ''}")


if __name__ == "__main__":
    main()
