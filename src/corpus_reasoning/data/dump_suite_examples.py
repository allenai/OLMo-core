"""Render a few human-readable examples (prompt + target) for every suite task,
so the data can be eyeballed. Long document contexts are abbreviated; the query
and target are shown in full, plus each task's CoT target(s).

Run from the repo root:  PYTHONPATH=. python scripts/data/dump_suite_examples.py [N]
"""
import json
import sys

from corpus_reasoning.lib.data_format import build_prompt

# (display name, eval task, JSONL, [cot_modes])
TASKS = [
    ("NIAH-contradiction", "retrieval", "data/niah_contradiction_eval_n99_p1.jsonl", []),
    ("strmatch", "strmatch", "data/strmatch_eval_n100_k3_sub3_l10_h3.jsonl", ["template", "enumerate"]),
    ("cycle", "cycle", "data/cycle_eval_n100_len3_k1.jsonl", ["template", "trace"]),
    ("groups-of-4", "groups4", "data/groups4_eval_n100_g4_k1_x5_numsonly.jsonl", ["template", "sort"]),
    ("redundancy", "redundancy", "data/redundancy_eval_pubmed_both_n100_k3_hn6.jsonl", ["template", "enumerate"]),
    ("absence", "absence", "data/absence_eval_pubmed_n100_p01.jsonl", ["template", "walk"]),
    ("N2-ified NQ", "retrieval", "data/n2ified_eval_nq_q20.jsonl", []),
    ("BEIR SciFact", "retrieval", "data/beir_scifact_test_k20_300.jsonl", []),
    ("MS MARCO rerank", "rerank", "data/msmarco_dev_rerank_k20_1000.jsonl", ["template", "score"]),
    ("OOLONG", "oolong", "data/oolong_validation_synth_ctx1024.jsonl", ["plan"]),
    ("HELMET NarrativeQA", "qa", "data/helmet_narrativeqa_validation_L8000.jsonl", []),
    ("HELMET GovReport summ", "summarization", "data/helmet_summ_govreport_validation_L16000.jsonl", []),
]

ABBREV = 700  # chars of context to show

# Tasks whose gold_doc_indices are 1-indexed (claim/expression/string IDs);
# the rest (retrieval/absence/rerank) are 0-indexed positions.
ONE_INDEXED = {"contradiction", "redundancy", "strmatch", "cycle", "groups4",
               "mathmatch", "matching_ngram"}


def gold_positions(task, gold):
    """Flatten gold_doc_indices (pairs/groups/per-query lists) to sorted 0-indexed
    document positions."""
    flat = []
    for g in (gold or []):
        flat.extend(g) if isinstance(g, list) else flat.append(g)
    base = 1 if task in ONE_INDEXED else 0
    return sorted({i - base for i in flat})


def gold_block(ex, task):
    """A section showing the actual gold/target documents in full (so they're
    never hidden by context abbreviation)."""
    docs = ex.get("documents", [])
    pos = gold_positions(task, ex.get("gold_doc_indices"))
    pos = [p for p in pos if 0 <= p < len(docs)]
    if not pos or len(docs) <= 1:
        return ""  # single-doc tasks (qa/summ/oolong) — the answer is the target
    lines = []
    for p in pos[:16]:
        t = docs[p].get("text", "")
        t = t if len(t) <= 600 else t[:600] + " …"
        lines.append(f"- [{p + 1}] {t}")
    extra = f"\n  …(+{len(pos) - 16} more)" if len(pos) > 16 else ""
    return ("**Gold/target documents** (the actual docs the answer points to):\n"
            + "\n".join(lines) + extra)


def abbreviate(prompt):
    if "### Input:" not in prompt:
        return prompt
    head, inp = prompt.split("### Input:", 1)
    body, resp = inp.split("### Response:", 1) if "### Response:" in inp else (inp, "")
    body = body.strip()
    if len(body) > ABBREV * 2:
        body = (body[:ABBREV]
                + f"\n\n...[{len(body)} chars of context abbreviated]...\n\n"
                + body[-ABBREV:])
    return head + "### Input:\n" + body + "\n\n### Response:" + resp


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    out = ["# Task-suite examples (prompt + target)\n",
           "Document contexts abbreviated; query + target shown in full. "
           "CoT targets shown where the task has a `cot_mode`.\n"]
    for name, task, path, cots in TASKS:
        try:
            rows = [json.loads(l) for l in open(path)]
        except FileNotFoundError:
            out.append(f"\n## {name} — (data file missing: {path})\n")
            continue
        out.append(f"\n\n{'='*78}\n## {name}   (`--task {task}`)\n`{path}`\n{'='*78}")
        for i in range(min(n, len(rows))):
            p, o = build_prompt(rows[i], task=task)
            out.append(f"\n### example {i+1}\n```\n{abbreviate(p)}\n```")
            gb = gold_block(rows[i], task)
            if gb:
                out.append(gb)
            out.append(f"**Target:** `{o[:400]}`")
            for cot in cots:
                try:
                    _, oc = build_prompt(rows[i], task=task, cot_mode=cot)
                    if oc != o:
                        out.append(f"**Target (CoT `{cot}`):**\n```\n{oc[:1500]}\n```")
                except Exception as e:
                    out.append(f"_(CoT {cot} failed: {e})_")
    open("examples/suite_examples.md", "w").write("\n".join(out))
    print(f"wrote examples/suite_examples.md ({len(TASKS)} tasks, {n} examples each)")


if __name__ == "__main__":
    main()
