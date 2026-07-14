"""Render a few human-readable examples (prompt + target) for every suite task,
so the data can be eyeballed. Long document contexts are abbreviated; the query
and target are shown in full. Optionally also shows the CoT target per task.
"""
import json
import sys

from corpus_reasoning.lib.data_format import build_prompt

# task -> (eval JSONL, cot_mode-or-None)
TASKS = [
    ("NIAH-contradiction", "retrieval", "data/niah_contradiction_eval_n99_p1.jsonl", None),
    ("strmatch", "strmatch", "data/strmatch_eval_n100_k3_sub3_l10_h3.jsonl", "template"),
    ("cycle", "cycle", "data/cycle_eval_n100_len3_k1.jsonl", "template"),
    ("groups-of-4", "groups4", "data/groups4_eval_n100_g4_k1_x5_numsonly.jsonl", "template"),
    ("redundancy", "redundancy", "data/redundancy_eval_pubmed_both_n100_k3_hn6.jsonl", "template"),
    ("absence", "absence", "data/absence_eval_pubmed_n100_p01.jsonl", "template"),
    ("N2-ified NQ", "retrieval", "data/n2ified_eval_nq_q20.jsonl", None),
    ("BEIR SciFact", "retrieval", "data/beir_scifact_test_k20_300.jsonl", None),
    ("MS MARCO rerank", "rerank", "data/msmarco_dev_rerank_k20_1000.jsonl", "template"),
    ("OOLONG", "oolong", "data/oolong_validation_synth_ctx1024.jsonl", "plan"),
    ("HELMET NarrativeQA", "qa", "data/helmet_narrativeqa_validation_L8000.jsonl", None),
    ("HELMET GovReport summ", "summarization", "data/helmet_summ_govreport_validation_L16000.jsonl", None),
]

ABBREV = 700  # chars of context to show


def abbreviate(prompt):
    """Shorten the '### Input:' block so long docs are readable."""
    if "### Input:" not in prompt:
        return prompt
    head, inp = prompt.split("### Input:", 1)
    body, resp = inp.split("### Response:", 1) if "### Response:" in inp else (inp, "")
    body = body.strip()
    if len(body) > ABBREV * 2:
        body = body[:ABBREV] + f"\n\n...[{len(body)} chars of context abbreviated]...\n\n" + body[-ABBREV:]
    return head + "### Input:\n" + body + "\n\n### Response:" + resp


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    out = ["# Task-suite examples (prompt + target)\n",
           "Document contexts are abbreviated for readability; query + target shown in full.\n"]
    for name, task, path, cot in TASKS:
        try:
            rows = [json.loads(l) for l in open(path)]
        except FileNotFoundError:
            out.append(f"\n## {name}  — (data file missing: {path})\n")
            continue
        out.append(f"\n\n{'='*78}\n## {name}   (`--task {task}`)\n`{path}`\n{'='*78}")
        for i in range(min(n, len(rows))):
            p, o = build_prompt(rows[i], task=task)
            out.append(f"\n### example {i+1}\n```\n{abbreviate(p)}\n```")
            out.append(f"**Target:** `{o[:400]}`")
            if cot:
                try:
                    _, oc = build_prompt(rows[i], task=task, cot_mode=cot)
                    if oc != o:
                        out.append(f"**Target (CoT `{cot}`):**\n```\n{oc[:1200]}\n```")
                except Exception as e:
                    out.append(f"_(CoT {cot} not available: {e})_")
    open("examples/suite_examples.md", "w").write("\n".join(out))
    print(f"wrote examples/suite_examples.md ({len(TASKS)} tasks, {n} examples each)")


if __name__ == "__main__":
    main()
