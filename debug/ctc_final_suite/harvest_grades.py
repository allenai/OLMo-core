"""Harvest every per-rung CTC-suite grade JSON reachable over /net and emit one merged table.

Sources are the node-local `ctc_suite_vllm_results{,_chunked}` / `ctc_newrung_results{,_chunked}`
trees; when the same (task, arm, rung) appears on more than one node the NEWEST file wins, so a
re-grade on a fixed ladder supersedes the original.
"""
import json, os, glob, re, sys

NODES = ["cubbins", "mooney", "sneetches", "lorax", "horton"]
TREES = [("ctc_suite_vllm_results", "dense"), ("ctc_suite_vllm_results_chunked", "chunked"),
         ("ctc_newrung_results", "dense"), ("ctc_newrung_results_chunked", "chunked")]

best = {}   # (task, arm, rung) -> record
for node in NODES:
    for tree, arm in TREES:
        root = f"/net/{node}/data/prasann/{tree}"
        if not os.path.isdir(root):
            continue
        for p in glob.glob(f"{root}/*/grade_*.json"):
            try:
                d = json.load(open(p))
            except Exception as e:
                print(f"  !! unreadable {p}: {e}", file=sys.stderr)
                continue
            task = os.path.basename(os.path.dirname(p))   # dir name, not the canonical scorer name
            m = re.search(r"(\d+)\.json$", os.path.basename(p))
            if not m:
                print(f"  !! no rung in {p}", file=sys.stderr)
                continue
            rung = int(m.group(1))
            mode = d.get("mode", arm)
            arm_final = "chunked" if "chunk" in str(mode) else arm
            key = (task, arm_final, rung)
            mt = os.path.getmtime(p)
            if key in best and best[key]["mtime"] >= mt:
                continue
            best[key] = dict(
                mtime=mt, path=p, node=node,
                metric_name=d.get("metric_name"), metric_value=d.get("metric_value"),
                eval_size=d.get("eval_size"), parse_rate=d.get("parse_rate"),
                eval_data=d.get("eval_data"), hf_model=d.get("hf_model"), mode=mode,
                canonical_task=d.get("task"),
            )

# ---- second source: the in-repo 4B result trees -------------------------------------------------
# Two suite rows are NOT produced by the node-local sweep drivers and so are invisible to the loop
# above: outlier_fixk and qdmatch_nq were backfilled by the model-scale drivers, which write into
# the repo under debug/ctc_modelscale/. Without this block those rows render blank even though the
# numbers exist, which is what "the artifact is missing settings" has meant more than once.
#
# ⚠ 4B ONLY. The same trees hold 0.8B and 2B cells from the model-scale sweep. This grid is the 4B
# dense-vs-chunked comparison; folding a 2B number into it would be a silent contamination that no
# downstream reader could detect. The dir-name suffix is the filter, and anything that does not
# parse as 4B is skipped loudly rather than guessed at.
REPO_TREES = ["debug/ctc_modelscale/results_4b"]
RUN_RE = re.compile(r"^ctcms-(?P<task>.+)-(?P<arm>full|cmix)-4b(?:-\w+)?$")
for tree in REPO_TREES:
    if not os.path.isdir(tree):
        continue
    for p in glob.glob(f"{tree}/*/grade_*.json"):
        run = os.path.basename(os.path.dirname(p))
        m_run = RUN_RE.match(run)
        if not m_run:
            print(f"  .. skipping non-4B/unparsed run dir {run}", file=sys.stderr)
            continue
        m = re.search(r"(\d+)\.json$", os.path.basename(p))
        if not m:
            print(f"  !! no rung in {p}", file=sys.stderr)
            continue
        try:
            d = json.load(open(p))
        except Exception as e:
            print(f"  !! unreadable {p}: {e}", file=sys.stderr)
            continue
        # The grade JSON's own `task` field is the SCORER (outlier_fixedM grades as "outlier"), so
        # the suite-row key has to come from the run name, not from the file.
        task = m_run.group("task")
        arm_final = "chunked" if m_run.group("arm") == "cmix" else "dense"
        key = (task, arm_final, int(m.group(1)))
        mt = os.path.getmtime(p)
        if key in best and best[key]["mtime"] >= mt:
            continue
        best[key] = dict(
            mtime=mt, path=p, node="repo",
            metric_name=d.get("metric_name"), metric_value=d.get("metric_value"),
            eval_size=d.get("eval_size"), parse_rate=d.get("parse_rate"),
            eval_data=d.get("eval_data"), hf_model=d.get("hf_model"), mode=d.get("mode", arm_final),
            canonical_task=d.get("task"),
        )

# ---- third source: explicit overrides where a task was RETRAINED -------------------------------
# The newest-file-wins rule resolves duplicate grades of the SAME model. It cannot resolve two
# different models sharing a task dir, and oolong has exactly that: `ctc-4b-oolong-*` is the
# original run and `ctcms-oolong-*-4b-vsl` is the retrain that fixed it.
#
# Left alone, the grid rendered oolong as dense 0.232 vs chunked 0.628 at 2k -- i.e. the dense arm
# LOSING to the chunked arm by a wide margin, which is the headline result inverted. That pairing
# was an artifact of the dense arm being the broken original while the chunked arm was not. The
# retrain gives 0.710 vs 0.691, a small dense win, which is the comparable measurement.
#
# These are keyed by run name, not by "newest", so a future re-grade of the ORIGINAL run cannot
# quietly take the row back.
OVERRIDES = [
    # (tree, run dir, task, arm)
    ("debug/ctc_oolong_eval/results", "ctcms-oolong-full-4b-vsl", "oolong", "dense"),
    ("debug/ctc_oolong_eval/results", "ctcms-oolong-cmix-4b-vsl", "oolong", "chunked"),
]
for tree, run, task, arm_final in OVERRIDES:
    root = os.path.join(tree, run)
    if not os.path.isdir(root):
        print(f"  !! override source missing: {root}", file=sys.stderr)
        continue
    n = 0
    for p in glob.glob(f"{root}/grade_*.json"):
        m = re.search(r"(\d+)\.json$", os.path.basename(p))
        if not m:
            continue
        try:
            d = json.load(open(p))
        except Exception as e:
            print(f"  !! unreadable {p}: {e}", file=sys.stderr)
            continue
        best[(task, arm_final, int(m.group(1)))] = dict(
            mtime=os.path.getmtime(p), path=p, node="repo-override",
            metric_name=d.get("metric_name"), metric_value=d.get("metric_value"),
            eval_size=d.get("eval_size"), parse_rate=d.get("parse_rate"),
            eval_data=d.get("eval_data"), hf_model=d.get("hf_model"), mode=d.get("mode", arm_final),
            canonical_task=d.get("task"),
        )
        n += 1
    print(f"  override: {task}/{arm_final} <- {run} ({n} rungs)")

rows = [dict(task=k[0], arm=k[1], rung=k[2], **v) for k, v in best.items()]
rows.sort(key=lambda r: (r["task"], r["arm"], r["rung"]))
out = "debug/ctc_final_suite/harvested_grades.json"
json.dump(rows, open(out, "w"), indent=1)
print(f"{len(rows)} (task, arm, rung) cells -> {out}")
tasks = sorted({r["task"] for r in rows})
print(f"{len(tasks)} tasks: {' '.join(tasks)}")
