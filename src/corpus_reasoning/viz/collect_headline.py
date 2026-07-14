"""
Stage 2d — Headline 32k-scale SFT matrix.

The flagship comparison: {dense, landmark, compressive} x {+CPT-mix, no-CPT} fine-tuned
at the 32k scale, evaluated on the 5-task long-context ladder (contradiction, NQ, oolong,
msmarco-rerank, outlier) + RULER. Reads the per-run ladder eval JSONs in
``outputs/eval_results`` and emits ``outputs/headline.json`` for the renderer.

Each ladder JSON stores per-rung scores as flat ``<task>_<rung>`` keys
(``contradiction_32k``, ``nq_3k``, ...). Base (pre-SFT) JSONs instead store a single
short-context number per task (``oolong_primary`` / ``contradiction.f1`` / ...).

Run-name -> descriptive-label mapping lives in ``ROWS`` below; the raw run name is kept
as ``raw`` so the renderer can surface it as a tooltip.
"""

import json
import os
import re

try:
    from . import config
except ImportError:
    import config

# The 5 long-context tasks (in display order) + RULER handled separately.
TASKS = ["contradiction", "nq", "oolong", "rerank", "outlier"]
TASK_LABELS = {
    "contradiction": "contradiction",
    "nq": "NQ",
    "oolong": "oolong",
    "rerank": "rerank",
    "outlier": "outlier",
}
# Full rung set across tasks (for the expandable per-rung ladder).
RUNGS = ["1k", "2k", "3k", "4k", "8k", "16k", "32k", "64k"]
HEADLINE_RUNG = "32k"

# Row spec: (raw run-name, descriptive label, variant, cpt-status, scale, status, note).
#   status: "real" (full ladder), "partial" (some rungs missing), "running", "base".
ROWS = [
    # ---- + CPT-mix ----
    dict(raw="q4b-dense-cptmix-32k-native_ladder",
         label="Dense · +CPT-mix · 32k SFT",
         variant="dense", cpt=True, scale="32k", group="+ CPT-mix", status="real",
         note="Native re-eval (olmo_core distcp, native rope/max-position) of the dense +CPT-mix "
              "32k checkpoint, replacing the earlier broken HF-export row whose long-context cells "
              "all read 0.000 (HF rope / max-position export bug). The native checkpoint is the "
              "model-only resave at /scratch/users/prasann/dense_modelonly/"
              "q4b-cptmix-5task-32k-am_modelonly. "
              "CAVEAT: contradiction@32k (0.035) is a suspected no-CoT answer-budget / parse "
              "artifact, NOT a confirmed model collapse — nq@32k (0.128) and oolong@32k (0.405) "
              "generate fine on this same checkpoint, so 32k generation works; the contradiction "
              "near-zero is contradiction-specific (the 96-token no-CoT budget likely truncating "
              "the answer at long context). The 2k->16k contradiction rungs degrade gracefully "
              "(0.53/0.36/0.21) and are trusted."),
    dict(raw="q4b-lm-cptmix-5task-40k-v3_ladder_native500",
         label="Landmark · +CPT-mix · 40k SFT",
         variant="landmark", cpt=True, scale="40k", group="+ CPT-mix", status="real",
         note="Full 5-task ladder (native landmark eval, top-k 0.1) merged in alongside the "
              "earlier RULER sweep. RULER still reads optimistically (partial subtask set)."),
    dict(raw="q4b-comp-cptmix-32k_ladder",
         label="Compressive · +CPT-mix · 32k SFT",
         variant="compressive", cpt=True, scale="32k", group="+ CPT-mix", status="real",
         note="Full 5-task ladder + RULER (native compressive-landmark eval, top-k 0.1, 500/rung), "
              "merged from 3 Beaker GPU splits run directly off the weka checkpoint. Like the dense "
              "and landmark +CPT-mix rows, mixing CPT data DEGRADES contradiction/NQ vs the no-CPT "
              "row (contra@2k 0.49 vs 0.89; cf. dense 0.53 vs 0.97, landmark 0.85 vs 0.92) — a real, "
              "variant-consistent CPT-mix effect, not an eval artifact. outlier@32k=0.0 is a "
              "task-wide difficulty cliff (dense +CPT-mix and dense no-CPT also read 0.0 there)."),
    # ---- no CPT-mix ----
    dict(raw="q4b-dense-32k_ladder",
         label="Dense · no-CPT · 32k SFT",
         variant="dense", cpt=False, scale="32k", group="no CPT-mix", status="real"),
    dict(raw="q4b-lm-32k_ladder",
         label="Landmark · no-CPT · 32k SFT",
         variant="landmark", cpt=False, scale="32k", group="no CPT-mix", status="real",
         note="Full 5-task ladder: the earlier run's contradiction + NQ + oolong(<=8k) merged "
              "with a follow-up filling oolong 16k/32k and the rerank / outlier ladders "
              "(native landmark eval, top-k 0.1)."),
    dict(raw="q4b-comp-nocpt-32k_ladder",
         label="Compressive · no-CPT · 32k SFT",
         variant="compressive", cpt=False, scale="32k", group="no CPT-mix", status="real",
         note="Full 5-task ladder + RULER (native compressive-landmark eval, top-k 0.1, 500/rung), "
              "merged from 3 Beaker GPU splits run directly off the weka checkpoint. Sits with the "
              "no-CPT cohort: contra@2k 0.89 (dense 0.97, landmark 0.92), oolong@32k 0.43, "
              "RULER avg recall 0.78. outlier@32k≈0.01 is the usual task-wide cliff."),
    # ---- document-chunked attention variants (box-marked docs; chunked mask) ----
    dict(raw="q4b-docchunk-dense-5task-24k-nocpt_ladder",
         label="Doc-chunked dense · no-CPT · 24k (local 4xH200)",
         variant="dense", cpt=False, scale="24k", group="no CPT-mix", status="real",
         note="DocumentChunkedAttention (each context document is a chunk, cross-doc attention "
              "restricted by the chunked mask). 5-task mix, dense CPT base, doc-chunked data "
              "(PadToLength, one example/window). Window 24576 (NOT 40960): the recipe is built for "
              "8x SXM H200 (141 GiB) but this local run is capped at 4x NVL H200 (139.8 GiB) by the "
              "preemptive_high QOS, where seq>=32768 OOMs (FlexAttention's ~8 GiB workspace is "
              "squeezed and the O(seq^2) dense-mask fallback explodes). 24576 fits with headroom and "
              "unifies the scale across all four local doc-chunked rows."),
    dict(raw="q4b-docchunk-hier-5task-24k-nocpt_ladder",
         label="Doc-chunked hierarchical · no-CPT · 24k (local 4xH200)",
         variant="dense", cpt=False, scale="24k", group="no CPT-mix", status="running",
         note="DocumentChunkedAttention with cross_doc_mode=hierarchical_dilated (dilation_n=4, "
              "m=2): layer l attends 4 docs at stride 2^l. SAME doc-chunked dense data + dense base "
              "as the doc-chunked dense row; same 24576 local-4xH200 window (see dense note)."),
    dict(raw="q4b-docchunk-landmark-5task-24k-nocpt_ladder",
         label="Doc-chunked landmark · no-CPT · 24k (local 4xH200)",
         variant="landmark", cpt=False, scale="24k", group="no CPT-mix", status="real",
         note="DocumentLandmarkAttention (grouped-softmax memory; landmark-format doc-chunked data, "
              "one landmark per packed window); fast-landmark CPT base, native landmark eval. Same "
              "24576 local-4xH200 window as the dense/hier/compressive doc-chunked rows, so the four "
              "are directly comparable at 24k. (24576 is the unified local cap: the eager grouped "
              "softmax is O(seq^2) and the dense recipe OOMs >=32768 on 4x NVL H200 anyway.)"),
    dict(raw="q4b-docchunk-compressive-5task-24k-nocpt_ladder",
         label="Doc-chunked compressive · no-CPT · 24k (local 4xH200)",
         variant="compressive", cpt=False, scale="24k", group="no CPT-mix", status="running",
         note="DocumentCompressiveLandmarkAttention (each past block's landmark token also contributes "
              "its value as a compressed block summary, alpha=0.1). Same 24576 local-4xH200 window as "
              "the other three doc-chunked rows. BLOCKED locally: its compressive CPT base is weka-only "
              "and not yet pulled to the Berkeley host (dense + fast-landmark bases are local)."),
    # ---- base CPT references (pre-SFT) ----
    dict(raw="q4b-dense-base_lc",
         label="Dense — CPT base (pre-SFT)",
         variant="dense", cpt=None, scale="base", group="Base CPT (pre-SFT reference)",
         status="base"),
    dict(raw="q4b-lm-base_lmnative",
         label="Landmark — CPT base (pre-SFT)",
         variant="landmark", cpt=None, scale="base", group="Base CPT (pre-SFT reference)",
         status="base"),
    # ---- larger-scale context point ----
    dict(raw="q4b-cptmix-5task-64k-am_ladder_local",
         label="Dense · +CPT-mix · 64k SFT",
         variant="dense", cpt=True, scale="64k", group="Larger-scale reference",
         status="real", suspect=True,
         note="Numbers withheld — same broken HF-export eval as the 32k row (perfect NIAH at "
              "L1024/L2048, 0 at L4096+; tasks parse to ~0). A larger-scale (64k) point for "
              "context, not part of the 32k matrix. Pending a native re-eval."),
]


def _eval_dir() -> str:
    return os.environ.get(
        "CR_EVAL_RESULTS_DIR",
        os.path.join(config.CR_REPO_ROOT, "outputs", "eval_results"),
    )


def _load(raw: str):
    path = os.path.join(_eval_dir(), raw + ".json")
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _ladder(d: dict) -> dict:
    """Pull flat ``<task>_<rung>`` keys into ``{task: {rung: score}}``."""
    out: dict = {}
    for k, v in d.items():
        m = re.match(r"^(contradiction|nq|oolong|rerank|outlier)_(\d+k)$", k)
        if m and isinstance(v, (int, float)):
            out.setdefault(m.group(1), {})[m.group(2)] = round(float(v), 3)
    return out


def _base_cells(d: dict) -> dict:
    """Single short-context number per task from a base (pre-SFT) eval JSON."""
    contra = (d.get("contradiction") or {})
    nq = (d.get("nq") or {})

    def _r(x):
        return round(float(x), 3) if isinstance(x, (int, float)) else None

    return {
        "contradiction": _r(contra.get("f1")),
        "nq": _r(nq.get("f1")),
        "oolong": _r(d.get("oolong_primary")),
        "rerank": _r(d.get("rerank_primary")),
        "outlier": _r(d.get("outlier_primary")),
    }


def collect() -> dict:
    rows = []
    for spec in ROWS:
        row = dict(spec)
        # Running rows have no data yet; suspect rows have a broken eval whose 0.000s we
        # must NOT present as real numbers — both render as placeholders (with a note).
        if spec["status"] == "running" or spec.get("suspect"):
            row["ladder"] = {}
            row["cell"] = {t: None for t in TASKS}
            row["ruler"] = None
            rows.append(row)
            continue

        d = _load(spec["raw"])
        if d is None:
            # Treat a missing file like a pending/running row rather than dropping it.
            row["status"] = "running"
            row["note"] = (row.get("note") or "") + " (eval JSON not found at build time)"
            row["ladder"] = {}
            row["cell"] = {t: None for t in TASKS}
            row["ruler"] = None
            rows.append(row)
            continue

        row["ruler"] = (
            round(float(d["ruler_avg_recall"]), 3)
            if isinstance(d.get("ruler_avg_recall"), (int, float))
            else None
        )

        if spec["status"] == "base":
            cells = _base_cells(d)
            row["ladder"] = {}
            row["cell"] = cells
        else:
            ladder = _ladder(d)
            row["ladder"] = ladder
            row["cell"] = {t: ladder.get(t, {}).get(HEADLINE_RUNG) for t in TASKS}
        rows.append(row)

    # Group rows preserving first-seen group order.
    group_order = []
    for r in rows:
        if r["group"] not in group_order:
            group_order.append(r["group"])
    groups = [{"group": g, "rows": [r for r in rows if r["group"] == g]} for g in group_order]

    caption = (
        "The headline comparison: three attention variants — <b>dense</b>, "
        "<b>landmark</b> (grouped-softmax memory), and <b>compressive</b> — each "
        "fine-tuned at the <b>32k</b> scale, with and without mixing continued-pretraining "
        "(CPT) text back into SFT. Columns show the <b>32k rung</b> of each task plus RULER "
        "avg recall; expand a row for the full per-rung ladder. All contradiction scores are "
        "<b>no-CoT</b>."
    )
    return {
        "title": "32k-scale SFT matrix",
        "caption": caption,
        "tasks": TASKS,
        "task_labels": TASK_LABELS,
        "rungs": RUNGS,
        "headline_rung": HEADLINE_RUNG,
        "groups": groups,
        "source": "outputs/eval_results/*_ladder*.json + *_lc / *_lmnative base evals",
        "footnotes": [
            "Headline column = the 32k rung of each task ladder; base rows show the "
            "pre-SFT short-context eval (not a 32k number) for reference.",
            "RULER avg recall is averaged over each run's own context sweep, so the ranges "
            "differ: base evals cover only L1024/L2048, the SFT ladders sweep up to L16384, "
            "and the landmark +CPT-mix run reports a partial (optimistic) subtask set.",
            "Contradiction is reported NO-CoT throughout.",
            "The Dense +CPT-mix 32k row is now a native re-eval (replacing the broken HF export). "
            "The remaining \"eval n/a\" row (Dense +CPT-mix 64k reference) was scored from a "
            "broken HF export whose 0.000s are eval artifacts, withheld pending a native re-eval.",
        ],
    }


def main():
    config.ensure_out_dir()
    payload = collect()
    out_path = os.path.join(config.OUT_DIR, "headline.json")
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    n = sum(len(g["rows"]) for g in payload["groups"])
    print(f"[collect_headline] wrote {out_path} ({n} rows, {len(payload['groups'])} groups)")


if __name__ == "__main__":
    main()
