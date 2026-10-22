"""
Which eval rows are IN-DOMAIN for the mt11 training mix (setA rows), for contradiction / nq / outlier /
oolong: the old ds64 files run_eval_lambda.sbatch uses by default, or the CTC-suite rows?

Reads the REAL training rows from a merged mt11 shard (``merge_tasks.py`` output: raw uint32/bool
parts, rows end at EOS, one part per source task in ``metadata.json`` ``merged_from`` order) and puts
every eval row through the converter exactly as the shard was built (``query_position`` /
``tokenizer`` from the shard's metadata.json), so both sides are measured by one code path:

* **format**: query position in the training rows (does the prompt text before the first document
  repeat after the last one?), and the out-of-document template text of a training row vs a
  converted eval row with the per-row query masked (contradiction has no query: must be identical);
* **distribution**, on training rows whose length falls in each eval file's p5-p95 token range:
  documents per row, tokens per document, gold-answer cardinality, prompt length; contradiction gold
  pairs' word-Jaccard (realistic vs both mode); oolong answer type / items per row;
* **overlap** of the old and suite files (shared first documents).

CPU only (tokenizers), run on a compute node:

    PYTHONPATH=<src> python audit_mt11_eval_domain.py --shard <tdlr>/shards/mt11s720 \\
        --old-root <tdlr>/eval --suite-root <out>/suite --out audit.json
"""

import argparse
import hashlib
import json
import os
import re
import statistics
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir, os.pardir))

#: task -> (converter loadtask, chunk_by, old 2k file, old 8k file, suite subset, suite 2k tokens)
TASKS = {
    "contradiction": ("contradiction", "document",
                      "_eval_bundle_eval500_v3/contra/contradiction_eval_pubmed_realistic_n100_k3.jsonl",
                      "_eval_bundle_eval500_v3/contra/contradiction_eval_pubmed_realistic_n190_k3.jsonl",
                      "contradiction_iid", 2560),
    "nq": ("retrieval", "document", "outlier_lengthmix/eval_rungs/nq/rung_2048.jsonl",
           "outlier_lengthmix/eval_rungs/nq/rung_8192.jsonl", "nq", 2048),
    "outlier": ("outlier", "document", "outlier_lengthmix/eval_rungs/outlier/rung_2048.jsonl",
                "outlier_lengthmix/eval_rungs/outlier/rung_8192.jsonl", "outlier", 2048),
    "oolong": ("oolong", "line",
               "_eval_bundle_eval500_v2_clean/oolong/oolong_test_synth_ctx2048_spliteval.jsonl",
               "_eval_bundle_eval500_v2_clean/oolong/oolong_test_synth_ctx8192_spliteval.jsonl",
               "oolong", 2048),
}
#: merged_from "src" basename for each task
SRC_NAME = {"contradiction": "contradiction", "nq": "nq", "outlier": "outlier", "oolong": "oolong"}


def _words(s):
    return set(re.findall(r"[a-z0-9]+", s.lower()))


def _jacc(a, b):
    A, B = _words(a), _words(b)
    return len(A & B) / max(1, len(A | B))


def _q(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))] if xs else None


def _summ(xs):
    if not xs:
        return None
    return {"mean": round(statistics.mean(xs), 3), "p10": round(_q(xs, 0.1), 3),
            "p50": round(_q(xs, 0.5), 3), "p90": round(_q(xs, 0.9), 3), "n": len(xs)}


def row_stats(tok, ids, mask, ds, de, task):
    """Stats of one converted row (token ids + loss mask, EOS-terminated)."""
    first = next((j for j, m in enumerate(mask) if m), len(ids))
    prompt, answer = ids[:first], [t for t, m in zip(ids, mask) if m]
    docs, outside, cur, inside = [], [], [], False
    for t in prompt:
        if t == ds:
            inside, cur = True, []
            outside.append(ds)
        elif t == de:
            inside = False
            docs.append(cur)
        elif inside:
            cur.append(t)
        else:
            outside.append(t)
    ans = tok.decode(answer, skip_special_tokens=True)
    out_text = tok.decode(outside, skip_special_tokens=False)
    st = {"prompt_len": len(prompt), "n_docs": len(docs),
          "doc_tok": statistics.mean(len(d) for d in docs) if docs else 0,
          "answer": ans, "outside": out_text}
    if task == "contradiction":
        pairs = [(int(a), int(b)) for a, b in re.findall(r"\[\s*(\d+)\s*,\s*(\d+)\s*\]", ans)]
        st["gold_n"] = len(pairs)
        texts = [tok.decode(d, skip_special_tokens=True) for d in docs]
        js = [_jacc(texts[a - 1], texts[b - 1]) for a, b in pairs
              if 1 <= a <= len(texts) and 1 <= b <= len(texts)]
        st["pair_jaccard"] = js
    elif task in ("nq", "outlier"):
        st["gold_n"] = len(re.findall(r"\d+", ans))
    else:  # oolong: items are the ||-lines (each wrapped by the line chunker)
        st["gold_n"] = None
        st["numeric"] = bool(re.search(r"answer:\s*-?\d", ans.lower()))
    return st


def load_train_rows(shard, task):
    import numpy as np

    meta = json.load(open(os.path.join(shard, "metadata.json")))
    idx = [os.path.basename(m["src"]) for m in meta["merged_from"]].index(SRC_NAME[task])
    tok = np.fromfile(os.path.join(shard, f"token_ids_part_t{idx}_000.npy"), dtype=meta["dtype"])
    msk = np.fromfile(os.path.join(shard, f"labels_mask_t{idx}_000.npy"), dtype=meta["mask_dtype"])
    eos = meta["eos_token_id"]
    ends = np.nonzero(tok == eos)[0]
    rows, s = [], 0
    for e in ends:
        rows.append((tok[s:e + 1].tolist(), msk[s:e + 1].tolist()))
        s = e + 1
    assert len(rows) == meta["merged_from"][idx]["num_instances"], (task, len(rows))
    return rows, meta


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shard", required=True)
    ap.add_argument("--old-root", required=True)
    ap.add_argument("--suite-root", required=True)
    ap.add_argument("--rows", type=int, default=300, help="eval rows per file to convert")
    ap.add_argument("--tasks", nargs="+", default=list(TASKS))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    sys.path.insert(0, os.path.join(REPO, "src"))
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "conv", os.path.join(REPO, "recipes/data/convert_unified_to_document_landmark.py"))
    C = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(C)
    from transformers import AutoTokenizer

    from olmo_core.data.document_chunk_landmark import RESERVED_IDS

    report = {}
    for task in args.tasks:
        loadtask, chunk_by, old2, old8, sub, s2 = TASKS[task]
        rows, meta = load_train_rows(args.shard, task)
        tok = AutoTokenizer.from_pretrained(meta["tokenizer"])
        ids = RESERVED_IDS[meta["marker_set"]]
        qp = meta["query_position"]
        train = [row_stats(tok, r, m, ids.doc_start, ids.doc_end, task) for r, m in rows]
        # query position: does the text before the first document recur after the last one?
        both_like = 0
        marker = tok.decode([ids.doc_start])
        for st in train:
            pre = st["outside"].split(marker, 1)[0]
            post = st["outside"].rsplit(marker, 1)[-1]
            lines = [ln.strip() for ln in pre.split("\n") if len(ln.strip()) > 15]
            both_like += any(ln in post for ln in lines)
        rep = {"shard_query_position": qp, "shard_tokenizer": meta["tokenizer"],
               "train_rows": len(train),
               "train_rows_query_line_repeated_after_docs": f"{both_like}/{len(train)}",
               "train_outside_docs_example": train[0]["outside"][:1500],
               "train_answer_example": train[0]["answer"][:200], "files": {}}
        files = {"old_2k": os.path.join(args.old_root, old2), "old_8k": os.path.join(args.old_root, old8),
                 "suite_2k": os.path.join(args.suite_root, sub, f"rung_{s2}.jsonl"),
                 "suite_8k": os.path.join(args.suite_root, sub, "rung_8192.jsonl")}
        first_docs = {}
        for name, path in files.items():
            raw = [json.loads(l) for l in open(path)]
            first_docs[name] = {hashlib.md5(json.dumps(r["documents"][:3]).encode()).hexdigest()
                                for r in raw}
            evs = []
            for r in raw[: args.rows]:
                out = C.tokenize_example(tok, r, loadtask, emit="dense", query_position=qp,
                                         cot_mode="none", mem_freq=63, seq_len=10 ** 9,
                                         chunk_by=chunk_by, item_regex=r"\|\|", use_titles=False,
                                         ids_set=ids)
                if out is None:
                    continue
                st = row_stats(tok, [int(x) for x in out[0]], [bool(x) for x in out[1]],
                               ids.doc_start, ids.doc_end, task)
                st["raw"] = r
                evs.append(st)
            lo, hi = _q([e["prompt_len"] for e in evs], 0.05), _q([e["prompt_len"] for e in evs], 0.95)
            tr = [t for t in train if lo <= t["prompt_len"] <= hi]
            d = {"path": path, "rows_in_file": len(raw), "converted": len(evs),
                 "prompt_len_p5_p95": [lo, hi], "train_rows_in_range": len(tr)}
            for k in ("prompt_len", "n_docs", "doc_tok"):
                d[k] = {"eval": _summ([e[k] for e in evs]), "train": _summ([t[k] for t in tr])}
            if task != "oolong":
                d["gold_n"] = {"eval": dict(Counter(e["gold_n"] for e in evs).most_common(6)),
                               "train": dict(Counter(t["gold_n"] for t in tr).most_common(6))}
            if task == "contradiction":
                d["pair_jaccard"] = {
                    "eval": _summ([j for e in evs for j in e["pair_jaccard"]]),
                    "train": _summ([j for t in tr for j in t["pair_jaccard"]]),
                    "eval_frac_gt_0.5": round(sum(j > 0.5 for e in evs for j in e["pair_jaccard"])
                                              / max(1, sum(len(e["pair_jaccard"]) for e in evs)), 3),
                    "train_frac_gt_0.5": round(sum(j > 0.5 for t in tr for j in t["pair_jaccard"])
                                               / max(1, sum(len(t["pair_jaccard"]) for t in tr)), 3)}
            if task == "oolong":
                kind = lambda st: (re.findall(r"The following lines contain \d+ ([a-z ]+)",
                                              st["outside"]) or ["?"])[0]
                d["context_kind"] = {"eval": dict(Counter(kind(e) for e in evs).most_common(12)),
                                     "train": dict(Counter(kind(t) for t in tr).most_common(12))}
                d["numeric_answer_frac"] = {
                    "eval": round(statistics.mean(e["numeric"] for e in evs), 3),
                    "train": round(statistics.mean(t["numeric"] for t in tr), 3) if tr else None}
                d["eval_task_group"] = dict(Counter((e["raw"].get("_meta") or {}).get("task_group")
                                                    for e in evs).most_common(8))
                d["eval_dataset"] = dict(Counter((e["raw"].get("_meta") or {}).get("dataset")
                                                 for e in evs).most_common(12))
            if task == "outlier":
                d["eval_meta_mode"] = dict(Counter((e["raw"].get("meta") or {}).get("mode")
                                                   for e in evs))
                d["eval_meta_num_categories"] = _summ([(e["raw"].get("meta") or {}).get(
                    "num_categories") or 0 for e in evs])
            # template: the out-of-document text with the row's own query removed and digits masked
            def _tmpl(text, q, m=marker):
                if q:
                    text = text.replace(q, "<Q>")
                text = re.sub(f"(?:{re.escape(m)})+", "<DOCS>", text)  # n_docs-independent
                return re.sub(r"\d+", "#", text)
            e0 = evs[0]
            q0 = (e0["raw"].get("queries") or [""])[0] if task != "contradiction" else ""
            d["eval_outside_docs_template"] = _tmpl(e0["outside"], q0)[:1500]
            if task == "contradiction":
                d["template_identical_to_train_row"] = (
                    _tmpl(e0["outside"], "") == _tmpl(train[0]["outside"], ""))
            rep["files"][name] = d
        rep["overlap_first3docs"] = {
            "old_2k&suite_2k": len(first_docs["old_2k"] & first_docs["suite_2k"]),
            "old_8k&suite_8k": len(first_docs["old_8k"] & first_docs["suite_8k"])}
        report[task] = rep
        print(json.dumps({task: {k: v for k, v in rep.items() if "example" not in k}}, indent=1),
              flush=True)
    if args.out:
        json.dump(report, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
