"""
Golden parity check for the prompt/format/eval de-duplication.

Before the de-dup there were three copies of the prompt renderer (the library
``olmo_core.data.corpus_reasoning_prompts``, ``research/common/corpus_reasoning/lib`` and
``recipes/eval/legacy_native/ctc_eval/lib``) and two copies of the eval harness (``corpus_reasoning.eval``
and ``ctc_eval.eval``). This script renders prompts and runs the graders through every caller-facing
entry point of a source tree and dumps the results, so a tree from BEFORE the change and one from
AFTER it can be compared byte-for-byte:

    # 1. sample ~20 real rows per task (HF ctc-suite-eval r2k parquet + legacy eval jsonl + synthetic)
    python check_dedupe_parity.py sample --out rows.jsonl
    # 2. capture each tree (run with that tree's src/ first on PYTHONPATH)
    python check_dedupe_parity.py capture --root <old tree> --rows rows.jsonl --out old.json
    python check_dedupe_parity.py capture --root <new tree> --rows rows.jsonl --out new.json
    # 3. compare; exits non-zero on any unexplained difference
    python check_dedupe_parity.py compare --old old.json --new new.json

Entry points captured, per row:

* ``lib``      -- ``olmo_core.data.corpus_reasoning_prompts.build_prompt`` / ``build_prompt_parts``
  under several caller configurations (the training converters and the library call this).
* ``ctc``      -- ``ctc_eval.eval.evaluate.load_unified_examples`` (the canonical eval harness).
* ``research`` -- ``corpus_reasoning.eval.evaluate.load_unified_examples`` (research drivers).
* graders      -- each task's ``_eval_*`` grader, via both module paths, on the reference answer
  plus perturbed predictions (primed bracket dropped, truncated, shifted ids, ...), and the
  standalone answer parsers.
* ``TASK_CFG`` / ``TASK_ALIASES`` of ``eval_lc_native_docchunk`` via both module paths.

Rules enforced by ``compare`` (the canonical code must reproduce the copy each caller used, except
for the deliberate unifications, which must be EXACTLY those):

* ``lib`` and ``ctc`` prompts: identical.
* ``research`` prompts: identical, except rows whose task/data the research copy predated
  (``xformmatch``, ``recmatch``, one-sided ``xabsence`` rows carrying ``orphan_side == "A"``) --
  there the new output must equal the ctc_eval copy's.
* graders/parsers: identical to the research copy (which carried the primed-bracket / snippet
  parser fixes). Differences from the old ctc_eval copy are allowed only where the research copy
  already differed from it, i.e. exactly those fixes.
"""

import argparse
import ast
import copy
import glob
import importlib
import json
import os
import random
import re
import sys
import tempfile
from collections import Counter, defaultdict

HF_SNAPSHOT_GLOB = "/data/prasann/hf/hub/datasets--PrasannSinghal--ctc-suite-eval/snapshots/*/data/{cfg}/r2k.parquet"
CR_DATA = [
    "/data/prasann/corpus-reasoning/data",
    "/data/prasann/from_scratch/corpus-reasoning/data",
]
N_PER_TASK = 20

#: HF ctc-suite-eval config -> evaluate.py task name.
HF_TASKS = {
    "absence_gutenberg": "absence",
    "contradiction_iid": "contradiction",
    "fiqa": "retrieval",
    "grouping": "grouping",
    "hotpotqa": "retrieval",
    "msmarco": "retrieval",
    "niah": "retrieval",
    "nq": "retrieval",
    "obliq_twitter": "retrieval",
    "oolong": "oolong",
    "outlier": "outlier",
    "outlier_amzn": "outlier",
    "outlier_fixedM": "outlier",
    "qdmatch_fiqa": "qdmatch",
    "qdmatch_hpqa": "qdmatch",
    "qdmatch_nq": "qdmatch",
    "reorder": "reorder",
    "rerank": "rerank",
    "scifact": "retrieval",
    "strmatch": "strmatch",
    "textgroups": "textgroups",
    "xabsence": "xabsence",
}
#: legacy unified eval jsonl (first N rows) -> task; covers tasks/formats the HF suite does not.
JSONL_TASKS = [
    ("contradiction_eval_pubmed_both_n20_k3.jsonl", "contradiction"),
    ("redundancy_eval_pubmed_both_n100_k3_hn6.jsonl", "redundancy"),
    ("xabsence_eval_pubmed_p18_k3.jsonl", "xabsence"),
    ("absence_eval_pubmed_n20_p01.jsonl", "absence"),
    ("cycle_eval_n500_len3_k1.jsonl", "cycle"),
    ("groups4_eval_n100_g4_k1_x5_numsonly.jsonl", "groups4"),
    ("mathmatch_n20_k3_x16_r-500to500_eval.jsonl", "mathmatch"),
    ("matching_ngram_eval_wiki100w_n2_N100_m3.jsonl", "matching_ngram"),
    ("strmatch_eval_n500_k3_sub3_l10_h3.jsonl", "strmatch"),
    ("outlier_wiki100w_n20_k3_eval_100.jsonl", "outlier"),
    ("reorder_gutenberg100w_n20_eval_500.jsonl", "reorder"),
    ("qdmatch_eval_hpqa_q20_n20_k3_separate.jsonl", "qdmatch"),
    ("niah_contradiction_eval_n49_p1.jsonl", "retrieval"),
    ("oolong_test_synth_ctx1024_spliteval.jsonl", "oolong"),
    ("msmarco_dev_rerank_k20_1000.jsonl", "rerank"),  # binary (no ce_scores) rerank
    ("msmarco_trainhn_eval_k20_500.jsonl", "rerank"),
    ("msmarco_helmet_rerank_dev_k20_500.jsonl", "rerank_helmet"),
    ("helmet_summ_govreport_validation_L16000.jsonl", "summarization"),
    ("hotpotqa_eval_k20_bridge_unified_hn18_500_cot.jsonl", "cot_retrieval"),
    ("hotpotqa_eval_k20_bridge_unified_hn18_500.jsonl", "qa"),
    ("helmet_narrativeqa_validation_L8000.jsonl", "qa"),
]


# --------------------------------------------------------------------------------------------- sample
def _decode_parquet_value(k, v):
    if isinstance(v, str) and k in ("meta", "_meta") and v[:1] in "{[":
        try:
            return json.loads(v)
        except ValueError:
            return ast.literal_eval(v)
    return v


def _synthetic_rows(rng):
    words = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu".split()
    rows = []
    for i in range(N_PER_TASK):
        # ruler: documents + "Answer:" target; one subtask per row so every instruction is hit.
        sub = [
            "niah_single",
            "niah_multikey",
            "niah_multivalue",
            "niah_multiquery",
            "vt",
            "cwe",
            "fwe",
        ][i % 7]
        docs = [{"title": f"T{j}", "text": " ".join(rng.choices(words, k=12))} for j in range(6)]
        rows.append(
            dict(
                task="ruler",
                origin="synthetic",
                ex=dict(
                    documents=docs,
                    queries=[f"What is the special magic number for {words[i % 12]}?"],
                    answers=[str(rng.randint(1000, 9999)), str(rng.randint(1000, 9999))],
                    gold_doc_indices=[rng.randrange(6)],
                    _meta={"ruler_subtask": sub},
                ),
            )
        )
        # xformmatch: originals and rewrites in one shared index.
        items = [{"text": " ".join(rng.choices(words, k=10)), "type": "original"} for _ in range(4)]
        items += [{"text": " ".join(rng.choices(words, k=10)), "type": "rewrite"} for _ in range(8)]
        rng.shuffle(items)
        rows.append(
            dict(
                task="xformmatch",
                origin="synthetic",
                ex=dict(
                    documents=items,
                    queries=["Target procedure: reverse the word order."],
                    answers=[],
                    gold_pairs=[[1, 5], [3, 9]],
                    gold_doc_indices=[],
                ),
            )
        )
        # recmatch: labeled-field records, unordered pairs.
        recs = [{"text": f"name={words[j % 12]}; color={words[(j * 5) % 12]}"} for j in range(10)]
        rows.append(
            dict(
                task="recmatch",
                origin="synthetic",
                ex=dict(
                    documents=recs,
                    queries=["Query: pair records that share the same color."],
                    answers=[],
                    gold_doc_indices=[[1, 4], [2, 7]],
                ),
            )
        )
    return rows


def cmd_sample(args):
    import pyarrow.parquet as pq

    rng = random.Random(0)
    rows = []
    for cfg, task in HF_TASKS.items():
        files = sorted(glob.glob(HF_SNAPSHOT_GLOB.format(cfg=cfg)))
        if not files:
            print(f"[sample] MISSING hf config {cfg}", flush=True)
            continue
        recs = pq.read_table(files[0]).to_pylist()
        for r in rng.sample(recs, min(N_PER_TASK, len(recs))):
            ex = {k: _decode_parquet_value(k, v) for k, v in r.items() if v is not None}
            rows.append(dict(task=task, origin=f"hf:{cfg}", ex=ex))
        # grouping_labeled shares the grouping rows (same data, labeled instruction).
        if cfg == "grouping":
            for r in rows[-N_PER_TASK:]:
                rows.append(
                    dict(task="grouping_labeled", origin="hf:grouping", ex=copy.deepcopy(r["ex"]))
                )
        print(f"[sample] hf {cfg:18s} -> {task}: {min(N_PER_TASK, len(recs))} rows", flush=True)
    for fname, task in JSONL_TASKS:
        path = next(
            (os.path.join(d, fname) for d in CR_DATA if os.path.exists(os.path.join(d, fname))),
            None,
        )
        if path is None:
            print(f"[sample] MISSING jsonl {fname}", flush=True)
            continue
        got = 0
        with open(path) as f:
            for line in f:
                rows.append(dict(task=task, origin=f"jsonl:{fname}", ex=json.loads(line)))
                got += 1
                if got >= N_PER_TASK:
                    break
        print(f"[sample] jsonl {fname} -> {task}: {got} rows", flush=True)
    rows += _synthetic_rows(rng)
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")
    c = Counter(r["task"] for r in rows)
    print(
        f"[sample] wrote {len(rows)} rows over {len(c)} tasks -> {args.out}: {dict(sorted(c.items()))}"
    )


# -------------------------------------------------------------------------------------------- capture
LIB_CFGS = [
    dict(use_alpaca=False, cot_mode="none", query_position="both", use_titles=False),
    dict(use_alpaca=False, cot_mode="label", query_position="after"),
    dict(),
    dict(use_alpaca=True, query_position="before", use_titles=False),
    dict(use_alpaca=False, unified_prompt=True, query_position="both"),
]
EVAL_CFGS = [
    dict(),
    dict(query_position="both", use_titles=False, use_alpaca=False),
    dict(query_position="before"),
    dict(query_position="both", wrap_docs=True),
]
GRADERS = {
    "contradiction": "_eval_contradiction",
    "redundancy": "_eval_contradiction",
    "matching_ngram": "_eval_contradiction",
    "mathmatch": "_eval_contradiction",
    "strmatch": "_eval_contradiction",
    "qdmatch": "_eval_qdmatch",
    "absence": "_eval_absence",
    "xabsence": "_eval_absence",
    "oolong": "_eval_oolong",
    "rerank": "_eval_rerank",
    "rerank_helmet": "_eval_rerank_helmet",
    "summarization": "_eval_summarization",
    "cycle": "_eval_cycle",
    "groups4": "_eval_cycle",
    "textgroups": "_eval_cycle",
    "retrieval": "_eval_retrieval",
    "cot_retrieval": "_eval_retrieval",
    "outlier": "_eval_outlier",
    "grouping": "_eval_grouping",
    "grouping_labeled": "_eval_grouping",
    "reorder": "_eval_reorder",
    "ruler": "_eval_ruler",
    "qa": "_eval_qa",
    "xformmatch": "_eval_qa",  # evaluate.py's dispatch falls through to _eval_qa for these
    "recmatch": "_eval_qa",
}


def _shift_ids(s):
    return re.sub(r"\d+", lambda m: str(int(m.group()) + 1), s)


PERTURB = {
    "ref": lambda e: e,
    "drop1": lambda e: e[1:],  # primed '[' dropped
    "drop2": lambda e: e[2:],  # primed '[[' / '["' dropped
    "lead_nl": lambda e: "\n" + e,
    "empty": lambda e: "",
    "half": lambda e: e[: len(e) // 2],
    "answer_prefix": lambda e: "Answer: " + e,
    "shift": _shift_ids,
    "strip_open": lambda e: e.lstrip('[("'),
    "think": lambda e: "<think>\nhmm\n</think>\n" + e,
}
PARSERS = [
    ("metrics", "parse_doc_ids", lambda f, t: f(t)),
    ("evaluate", "parse_pairs", lambda f, t: f(t)),
    ("evaluate", "parse_qd_pairs", lambda f, t: f(t)),
    ("evaluate", "parse_cycles", lambda f, t: f(t)),
    ("evaluate", "_parse_snippet_list", lambda f, t: f(t)),
    ("evaluate", "_parse_id_set", lambda f, t: f(t, 60)),
    ("evaluate", "_parse_ranking", lambda f, t: f(t, 60)),
    ("evaluate", "_oolong_extract", lambda f, t: f(t)),
    ("evaluate", "parse_retrieval_output", lambda f, t: f(t)),
    ("eval_tasks", "parse_outlier_ids", lambda f, t: f(t, 60)),
    ("eval_tasks", "parse_partition", lambda f, t: f(t, 60)),
    ("eval_tasks", "parse_permutation", lambda f, t: f(t, 60)),
]


def _norm(x):
    """JSON-stable form: sets sorted, floats by repr, tuples as lists."""
    if isinstance(x, dict):
        return {str(k): _norm(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_norm(v) for v in x]
    if isinstance(x, (set, frozenset)):
        return sorted((_norm(v) for v in x), key=repr)
    if isinstance(x, float):
        return repr(x)
    if x is None or isinstance(x, (bool, int, str)):
        return x
    return repr(x)


def _call(fn, *a, **kw):
    try:
        return {"ok": _norm(fn(*a, **kw))}
    except Exception as e:  # recorded, compared like any other output
        return {"err": f"{type(e).__name__}: {str(e)[:200]}"}


def cmd_capture(args):
    root = os.path.abspath(args.root)
    for p in (
        os.path.join(root, "research", "common"),
        os.path.join(root, "recipes", "eval", "legacy_native"),
        os.path.join(root, "src"),
    ):
        sys.path.insert(0, p)
    import olmo_core

    assert os.path.abspath(olmo_core.__file__).startswith(
        root
    ), f"olmo_core resolved to {olmo_core.__file__}, not under {root} -- put {root}/src first"
    lib = importlib.import_module("olmo_core.data.corpus_reasoning_prompts")
    mods = {
        "ctc": {
            "evaluate": importlib.import_module("ctc_eval.eval.evaluate"),
            "metrics": importlib.import_module("ctc_eval.lib.metrics"),
            "eval_tasks": importlib.import_module("ctc_eval.lib.eval_tasks"),
            "docchunk": importlib.import_module("ctc_eval.eval.eval_lc_native_docchunk"),
        },
        "research": {
            "evaluate": importlib.import_module("corpus_reasoning.eval.evaluate"),
            "metrics": importlib.import_module("corpus_reasoning.lib.metrics"),
            "eval_tasks": importlib.import_module("corpus_reasoning.lib.eval_tasks"),
            "docchunk": importlib.import_module("corpus_reasoning.eval.eval_lc_native_docchunk"),
        },
    }
    for k, m in mods.items():
        print(f"[capture] {k}.evaluate -> {m['evaluate'].__file__}", flush=True)
    rows = [json.loads(line) for line in open(args.rows)]
    out = {"meta": {}, "rows": [], "prompts": {}, "graders": {}, "parsers": {}, "taskcfg": {}}
    out["meta"]["rows"] = [
        {"task": r["task"], "origin": r["origin"], "orphan_side": r["ex"].get("orphan_side")}
        for r in rows
    ]

    # A. prompts
    tmp = tempfile.mkdtemp(prefix="parity_", dir=os.environ.get("TMPDIR"))
    for i, r in enumerate(rows):
        task = r["task"]
        for ci, cfg in enumerate(LIB_CFGS):
            out["prompts"][f"lib|{i}|{ci}"] = _call(
                lib.build_prompt, copy.deepcopy(r["ex"]), task=task, **cfg
            )
            pcfg = {k: v for k, v in cfg.items() if k not in ("use_alpaca", "unified_prompt")}
            out["prompts"][f"libparts|{i}|{ci}"] = _call(
                lib.build_prompt_parts, copy.deepcopy(r["ex"]), task=task, **pcfg
            )
        path = os.path.join(tmp, f"row{i}.jsonl")
        with open(path, "w") as f:
            f.write(json.dumps(r["ex"]) + "\n")
        for who in ("ctc", "research"):
            for ci, cfg in enumerate(EVAL_CFGS):
                res = _call(
                    lambda: [
                        [e["prompt"], e["expected_output"]]
                        for e in mods[who]["evaluate"].load_unified_examples(path, 0, task, **cfg)
                    ]
                )
                out["prompts"][f"{who}|{i}|{ci}"] = res
    print(f"[capture] prompts: {len(out['prompts'])} renders", flush=True)

    # B. graders. Examples come from the ctc eval-side renderer for BOTH module paths, so the two
    # graders see identical inputs (the research renderer is compared separately in A).
    by_task = defaultdict(list)
    for i, r in enumerate(rows):
        by_task[r["task"]].append(i)
    preds_pool = set()
    for task, idxs in sorted(by_task.items()):
        examples = []
        for i in idxs:
            path = os.path.join(tmp, f"row{i}.jsonl")
            try:
                examples += mods["ctc"]["evaluate"].load_unified_examples(path, 0, task)
            except Exception as e:
                print(
                    f"[capture] {task} row {i}: cannot render ({type(e).__name__}); skipped for graders"
                )
        if not examples:
            continue
        for pname, pf in PERTURB.items():
            responses = [pf(e["expected_output"]) for e in examples]
            preds_pool.update(responses)
            for who in ("ctc", "research"):
                fn = getattr(mods[who]["evaluate"], GRADERS[task])
                out["graders"][f"{who}|{task}|{pname}"] = _call(
                    fn, copy.deepcopy(examples), responses
                )
    print(f"[capture] graders: {len(out['graders'])} runs", flush=True)

    # C. standalone parsers over every prediction string produced above.
    preds = sorted(preds_pool)
    out["meta"]["n_preds"] = len(preds)
    for who in ("ctc", "research"):
        for modkey, fname, how in PARSERS:
            fn = getattr(mods[who][modkey], fname)
            out["parsers"][f"{who}|{fname}"] = [_call(how, fn, t) for t in preds]
    out["meta"]["preds"] = preds

    # D. eval_lc_native_docchunk task tables.
    for who in ("ctc", "research"):
        m = mods[who]["docchunk"]
        out["taskcfg"][who] = _norm(
            {"TASK_CFG": m.TASK_CFG, "TASK_ALIASES": getattr(m, "TASK_ALIASES", {})}
        )

    with open(args.out, "w") as f:
        json.dump(out, f)
    print(f"[capture] wrote {args.out}")


# -------------------------------------------------------------------------------------------- compare
def cmd_compare(args):
    old, new = json.load(open(args.old)), json.load(open(args.new))
    meta = old["meta"]["rows"]
    assert meta == new["meta"]["rows"], "captures were made over different rows"
    fails, notes = [], []

    def predated(i):  # tasks/data the research copy of data_format never knew about
        m = meta[i]
        return m["task"] in ("xformmatch", "recmatch") or (
            m["task"] == "xabsence" and m["orphan_side"] == "A"
        )

    # A. prompts
    research_explained = Counter()
    for key, nv in new["prompts"].items():
        who, i, ci = key.split("|")
        i = int(i)
        ov = old["prompts"][key]
        if who in ("lib", "libparts", "ctc"):
            if ov != nv:
                fails.append(f"prompt {key} ({meta[i]['task']}, {meta[i]['origin']}) changed")
        else:  # research
            ctc_old = old["prompts"][f"ctc|{i}|{ci}"]
            if nv != ctc_old:
                fails.append(
                    f"prompt {key}: research path does not match the canonical (ctc) render"
                )
            elif ov != nv:
                if predated(i):
                    research_explained[
                        meta[i]["task"] + ("(orphan_side=A)" if meta[i]["orphan_side"] else "")
                    ] += 1
                else:
                    fails.append(
                        f"prompt {key} ({meta[i]['task']}, {meta[i]['origin']}): research render changed"
                    )
    n_lib = sum(k.startswith("lib") for k in new["prompts"])
    n_err = Counter(
        (k.split("|")[0], meta[int(k.split("|")[1])]["task"], v["err"].split(":")[0])
        for k, v in new["prompts"].items()
        if "err" in v
    )
    notes.append(
        f"prompts: {len(new['prompts'])} renders ({n_lib} lib), errors recorded (same in old): {dict(n_err)}"
    )
    notes.append(
        f"research render: intended changes (now = ctc_eval render): {dict(research_explained)}"
    )

    # B. graders
    fixed = Counter()
    for key, nv in new["graders"].items():
        who, task, pname = key.split("|")
        ov_r = old["graders"][f"research|{task}|{pname}"]
        ov_c = old["graders"][f"ctc|{task}|{pname}"]
        if nv != ov_r:
            fails.append(
                f"grader {key}: differs from the research copy (which has every grader fix)"
            )
        if nv != ov_c:
            if ov_r == ov_c:
                fails.append(
                    f"grader {key}: differs from ctc copy where research == ctc (unexplained)"
                )
            elif who == "ctc":
                fixed[f"{task}/{pname}"] += 1
    notes.append(
        f"graders: {len(new['graders'])} runs; ctc-path score changes = research-copy fixes: {dict(fixed)}"
    )

    # C. parsers
    preds = new["meta"]["preds"]
    assert preds == old["meta"]["preds"]
    pfixed = Counter()
    for key, nv in new["parsers"].items():
        who, fname = key.split("|")
        ov_r, ov_c = old["parsers"][f"research|{fname}"], old["parsers"][f"ctc|{fname}"]
        for t, a, b, c in zip(preds, nv, ov_r, ov_c):
            if a != b:
                fails.append(f"parser {key} on {t[:60]!r}: {a} != research {b}")
            if a != c:
                if b == c:
                    fails.append(f"parser {key} on {t[:60]!r}: changed vs ctc but research == ctc")
                elif who == "ctc":
                    pfixed[fname] += 1
    notes.append(
        f"parsers: {len(preds)} prediction strings; ctc-path changes = research fixes: {dict(pfixed)}"
    )

    # D. task tables
    if new["taskcfg"]["ctc"] != new["taskcfg"]["research"]:
        fails.append("TASK_CFG differs between module paths")
    if new["taskcfg"]["research"] != old["taskcfg"]["research"]:
        fails.append("TASK_CFG/TASK_ALIASES differ from the old research copy")
    oc = old["taskcfg"]["ctc"]["TASK_CFG"]
    for k, v in oc.items():
        if new["taskcfg"]["ctc"]["TASK_CFG"].get(k) != v:
            fails.append(f"TASK_CFG[{k}] differs from the old ctc_eval copy")
    notes.append(
        f"TASK_CFG: {len(new['taskcfg']['ctc']['TASK_CFG'])} tasks (old ctc_eval had {len(oc)}, all unchanged)"
    )

    c = Counter(m["task"] for m in meta)
    print(f"rows: {len(meta)} over {len(c)} tasks: {dict(sorted(c.items()))}")
    for n in notes:
        print("  " + n)
    if fails:
        print(f"\nFAIL: {len(fails)} unexplained differences")
        for f_ in fails[:50]:
            print("  " + f_)
        sys.exit(1)
    print(
        "\nPASS: every caller's prompts and scores match; all differences are the intended ones above."
    )


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--out", required=True)
    c = sub.add_parser("capture")
    c.add_argument("--root", required=True)
    c.add_argument("--rows", required=True)
    c.add_argument("--out", required=True)
    m = sub.add_parser("compare")
    m.add_argument("--old", required=True)
    m.add_argument("--new", required=True)
    args = ap.parse_args()
    {"sample": cmd_sample, "capture": cmd_capture, "compare": cmd_compare}[args.cmd](args)


if __name__ == "__main__":
    main()
