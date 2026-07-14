"""Render FULL examples exactly as they go into the model, for every suite task,
in both no-CoT and CoT settings — one .txt file PER TASK, each covering several
corpus lengths so length-dependent bugs are easy to spot.

Unlike dump_suite_examples.py (which abbreviates the document context), this shows
the COMPLETE prompt string fed to the model, then the target(s). During TRAINING
the model sees `prompt + target` concatenated; during EVAL it sees `prompt` and
must generate `target`. We print them separately so both reads are clear.

Run from repo root:  PYTHONPATH=. python scripts/data/dump_suite_examples_full.py
Optional arg: output DIR (default examples/suite_examples_full/).
"""
import json
import os
import sys

from corpus_reasoning.lib.data_format import build_prompt

# (display name, slug, eval task, [cot_modes], n_examples_per_length,
#  [(length_label, jsonl_path), ...])
# cot_modes = the WITH-CoT target variants ([] = task has no CoT scaffold —
#   retrieval/qa/summ/grouping just emit the answer).
# n_examples kept to 1 for long-context tasks so files stay inspectable.
TASKS = [
    # ── O(N) ──
    ("NIAH-contradiction", "niah_contradiction", "retrieval", [], 2, [
        ("n19", "data/niah_contradiction_eval_n19_p1.jsonl"),
        ("n99", "data/niah_contradiction_eval_n99_p1.jsonl"),
        ("n499", "data/niah_contradiction_eval_n499_p1.jsonl"),
    ]),
    ("OOLONG", "oolong", "oolong", ["plan"], 1, [
        ("ctx1024", "data/oolong_validation_synth_ctx1024.jsonl"),
        ("ctx16384", "data/oolong_validation_synth_ctx16384.jsonl"),
        ("ctx131072", "data/oolong_validation_synth_ctx131072.jsonl"),
    ]),
    ("HELMET NarrativeQA", "helmet_narrativeqa", "qa", [], 1, [
        ("L8000", "data/helmet_narrativeqa_validation_L8000.jsonl"),
        ("L32000", "data/helmet_narrativeqa_validation_L32000.jsonl"),
        ("L131072", "data/helmet_narrativeqa_validation_L131072.jsonl"),
    ]),
    ("HELMET GovReport summ", "helmet_summ_govreport", "summarization", [], 1, [
        ("L16000", "data/helmet_summ_govreport_validation_L16000.jsonl"),
        ("L64000", "data/helmet_summ_govreport_validation_L64000.jsonl"),
    ]),
    ("BEIR SciFact", "beir_scifact", "retrieval", [], 2, [
        ("k20", "data/beir_scifact_test_k20_300.jsonl"),
        ("k50", "data/beir_scifact_test_k50_300.jsonl"),
        ("k100", "data/beir_scifact_test_k100_299.jsonl"),
    ]),
    # CE-margin-cleaned negatives (FiQA is sparsely judged → BM25 distractors hide
    # unlabeled positives). Hard = CE-survivors (10% of slots), rest random real
    # corpus docs. See generate_beir_ce_data.py.
    ("BEIR FiQA", "beir_fiqa", "retrieval", [], 2, [
        ("k20", "data/beir_fiqa_ce_test_k20_648.jsonl"),
        ("k50", "data/beir_fiqa_ce_test_k50_648.jsonl"),
        ("k100", "data/beir_fiqa_ce_test_k100_648.jsonl"),
    ]),
    # Canonical MS MARCO retrieval task: 2k train / 500 eval per pool size,
    # CE-margin-cleaned SBERT hard negs (10% of non-gold slots) + format-identical
    # random fill (real index pids). No false negatives. Supersedes the TREC-DL
    # judged-pool and dev-qrels variants. See generate_msmarco_trainhn_data.py.
    # (Eval files shown here for inspection.)
    ("BEIR MS MARCO", "beir_msmarco", "retrieval", [], 2, [
        ("k20", "data/msmarco_trainhn_eval_k20_500.jsonl"),
        ("k50", "data/msmarco_trainhn_eval_k50_500.jsonl"),
        ("k100", "data/msmarco_trainhn_eval_k100_500.jsonl"),
        ("k500", "data/msmarco_trainhn_eval_k500_500.jsonl"),
    ]),
    # rerank now ranks the WHOLE pool by cross-encoder score (CE = ground-truth
    # relevance order, model cross-encoder/ms-marco-MiniLM-L-6-v2). Same CE-scored
    # trainhn files as the retrieval row above; the rerank target is the CE-desc
    # ID order. The dump also shows the HELMET top-10 target (--output-top-k 10).
    # cot_modes=[] — rerank has no CoT scaffold (target is the CE order regardless).
    ("MS MARCO rerank", "msmarco_rerank", "rerank", [], 2, [
        ("k20", "data/msmarco_trainhn_eval_k20_500.jsonl"),
        ("k50", "data/msmarco_trainhn_eval_k50_500.jsonl"),
        ("k100", "data/msmarco_trainhn_eval_k100_500.jsonl"),
    ]),
    # HELMET-native rerank format: explicit per-doc [ID: x] lines, graded 0-3
    # labels (CE-bucketed), "Ranking: ID3 > ID1 > ..." target. Shown 0-shot; the
    # eval/HELMET 2-shot setting prepends demos from the k10 demo file.
    ("MS MARCO rerank (HELMET format)", "msmarco_rerank_helmet", "rerank_helmet",
     [], 2, [
        ("k20", "data/msmarco_helmet_rerank_dev_k20_500.jsonl"),
        ("k50", "data/msmarco_helmet_rerank_dev_k50_500.jsonl"),
        ("k100", "data/msmarco_helmet_rerank_dev_k100_500.jsonl"),
    ]),
    # Multi-hop / open-domain retrieval (task=retrieval → emit gold doc IDs; no CoT
    # scaffold here — CoT is the separate cot_retrieval task). Eval/validation files
    # across pool sizes, matching the std_qboth retrieval configs + eval jobs.
    ("HotpotQA retrieval", "hpqa_retrieval", "retrieval", [], 2, [
        ("k20", "data/hotpotqa_eval_k20_bridge_unified_hn18_500.jsonl"),
        ("k50", "data/hotpotqa_eval_k50_bridge_unified_hn48_500.jsonl"),
        ("k100", "data/hotpotqa_eval_k100_bridge_unified_hn98_500.jsonl"),
        ("k200", "data/hotpotqa_eval_k200_bridge_unified_hn198_500.jsonl"),
    ]),
    ("NQ retrieval", "nq_retrieval", "retrieval", [], 2, [
        ("k20", "data/nq_validation_k20_hn19_500.jsonl"),
        ("k50", "data/nq_validation_k50_hn49_500.jsonl"),
        ("k100", "data/nq_validation_k100_hn99_500.jsonl"),
        ("k200", "data/nq_validation_k200_hn199_500.jsonl"),
    ]),
    # ObliQ docs are ~10x NQ length, so n_ex=1 keeps the file inspectable. Two
    # corpora (writing, congress) at the std k30 and chunked k100 pool sizes.
    ("ObliQ retrieval", "obliq_retrieval", "retrieval", [], 1, [
        ("writing_k30", "data/obliq_writing_test_k30_461.jsonl"),
        ("writing_k100", "data/obliq_writing_test_k100_461.jsonl"),
        ("congress_k30", "data/obliq_congress_test_k30_229.jsonl"),
        ("congress_k100", "data/obliq_congress_test_k100_229.jsonl"),
    ]),
    # ── O(N^2) ──
    ("redundancy", "redundancy", "redundancy", ["template", "enumerate"], 2, [
        ("n20", "data/redundancy_eval_pubmed_both_n20_k2_hn4.jsonl"),
        ("n100", "data/redundancy_eval_pubmed_both_n100_k3_hn6.jsonl"),
    ]),
    ("absence (Gutenberg text-diff)", "absence", "absence", ["list"], 2, [
        ("n10_k3", "data/absence_eval_gutenberg_n10_k3.jsonl"),
        ("n50_k3", "data/absence_eval_gutenberg_n50_k3.jsonl"),
        ("n200_k3", "data/absence_eval_gutenberg_n200_k3.jsonl"),
    ]),
    ("strmatch", "strmatch", "strmatch", ["template", "enumerate"], 2, [
        ("n20", "data/strmatch_eval_n20_k3_sub3_l10_h3.jsonl"),
        ("n100", "data/strmatch_eval_n100_k3_sub3_l10_h3.jsonl"),
        ("n500", "data/strmatch_eval_n500_k3_sub3_l10_h3.jsonl"),
    ]),
    ("qdmatch (HPQA)", "qdmatch_hpqa", "qdmatch", ["enumerate"], 2, [
        ("q20", "data/qdmatch_eval_hpqa_q20_n20_k3_separate.jsonl"),
        ("q50", "data/qdmatch_eval_hpqa_q50_n50_k3_separate.jsonl"),
        ("q100", "data/qdmatch_eval_hpqa_q100_n100_k3_separate.jsonl"),
    ]),
    ("xabsence", "xabsence", "xabsence", [], 2, [
        ("p8", "data/xabsence_eval_pubmed_p8_k3.jsonl"),
        ("p18", "data/xabsence_eval_pubmed_p18_k3.jsonl"),
        ("p48", "data/xabsence_eval_pubmed_p48_k3.jsonl"),
    ]),
    ("qdmatch (NQ)", "qdmatch_nq", "qdmatch", ["enumerate"], 1, [
        ("q20", "data/qdmatch_eval_nq_q20_n20_k3_separate.jsonl"),
        ("q50", "data/qdmatch_eval_nq_q50_n50_k3_separate.jsonl"),
        ("q100", "data/qdmatch_eval_nq_q100_n100_k3_separate.jsonl"),
        ("q250", "data/qdmatch_eval_nq_q250_n250_k3_separate.jsonl"),
    ]),
    ("qdmatch (ObliQ)", "qdmatch_obliq", "qdmatch", ["enumerate"], 1, [
        ("q20", "data/qdmatch_eval_obliq_q20_n20_k3_separate.jsonl"),
        ("q50", "data/qdmatch_eval_obliq_q50_n50_k3_separate.jsonl"),
        ("q100", "data/qdmatch_eval_obliq_q100_n100_k3_separate.jsonl"),
    ]),
    ("mathmatch", "mathmatch", "mathmatch", ["template"], 2, [
        ("n20_x16", "data/mathmatch_n20_k3_x16_r-500to500_numsonly_eval.jsonl"),
        ("n100_x4", "data/mathmatch_n100_k3_x4_r-500to500_numsonly_eval.jsonl"),
    ]),
    ("contradiction", "contradiction", "contradiction",
     ["template", "enumerate", "conflicts"], 2, [
        ("n20", "data/contradiction_eval_pubmed_both_n20_k3.jsonl"),
        ("n100", "data/contradiction_eval_pubmed_both_n100_k3.jsonl"),
        ("n500", "data/contradiction_eval_pubmed_both_n500_k3.jsonl"),
    ]),
    # ── O(N^3+) ──
    ("cycle", "cycle", "cycle", ["template", "trace"], 2, [
        ("n20", "data/cycle_eval_n20_len3_k1.jsonl"),
        ("n100", "data/cycle_eval_n100_len3_k1.jsonl"),
        ("n500", "data/cycle_eval_n500_len3_k1.jsonl"),
    ]),
    ("textgroups (sum-to-target over a textual feature)", "textgroups",
     "textgroups", ["template", "scan"], 2, [
        ("n20", "data/textgroups_eval_n20_g3_k2_t70_w0_mixed.jsonl"),
        ("n100", "data/textgroups_eval_n100_g3_k2_t70_w0_mixed.jsonl"),
        ("n500", "data/textgroups_eval_n500_g3_k2_t70_w0_mixed.jsonl"),
    ]),
    # ── pre-existing ──
    ("outlier", "outlier", "outlier", ["label"], 2, [
        ("n30", "data/review_outlier_n30_eval_500.jsonl"),
        ("v2_n100", "data/review_outlier_v2_n100_eval_500.jsonl"),
    ]),
    ("grouping", "grouping", "grouping", [], 1, [
        ("n20", "data/openalex_grouping_n20_levels_eval_500.jsonl"),
        ("n100", "data/openalex_grouping_n100_levels_eval_200.jsonl"),
    ]),
    ("reorder", "reorder", "reorder", ["successor", "anchor"], 2, [
        ("n5", "data/reorder_gutenberg100w_n5_eval_500.jsonl"),
        ("n20", "data/reorder_gutenberg100w_n20_eval_500.jsonl"),
        ("n50", "data/reorder_gutenberg100w_n50_eval_500.jsonl"),
    ]),
]

SEP = "=" * 80


def pick_indices(slug, rows, n_ex, length_idx):
    """Which row indices to render for this (task, length).

    Default: the first n_ex rows. OOLONG is heterogeneous (task groups
    counting/user/timeline × datasets spam/trec_coarse × several answer
    types), but the rows are grouped, so the first row is always the same
    counting/spam "most common label" question. For OOLONG we instead show
    one example per distinct (task_group, dataset, answer_type) combo — but
    only at the SHORTEST length (length_idx==0), where prompts are tiny;
    longer lengths keep n_ex so the file stays inspectable while still
    surfacing length scaling.
    """
    if slug != "oolong":
        return list(range(min(n_ex, len(rows))))
    if length_idx != 0:
        return list(range(min(n_ex, len(rows))))
    seen = {}
    for i, r in enumerate(rows):
        m = r.get("_meta", {})
        key = (m.get("task_group"), m.get("dataset"), m.get("answer_type"))
        if key not in seen:
            seen[key] = i
    return sorted(seen.values())


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "examples/suite_examples_full"
    os.makedirs(out_dir, exist_ok=True)
    written = []
    for name, slug, task, cots, n_ex, lengths in TASKS:
        out = [
            f"{SEP}\nTASK: {name}   (--task {task})\n{SEP}",
            "\nFULL prompt (no abbreviation) exactly as the unified evaluator builds it,",
            "then the target(s). TRAINING feeds `prompt + target` concatenated; EVAL",
            "feeds `prompt` and the model generates `target`.",
            "no-CoT target shown for every length; CoT-mode targets shown where the task",
            "has a CoT scaffold. Multiple corpus lengths below to surface length bugs.\n",
        ]
        for length_idx, (length_label, path) in enumerate(lengths):
            try:
                rows = [json.loads(l) for l in open(path)]
            except FileNotFoundError:
                out.append(f"\n{SEP}\n## length {length_label} — DATA FILE MISSING: {path}\n{SEP}")
                continue
            out.append(f"\n\n{SEP}\n## length {length_label}   {path}  ({len(rows)} rows)\n{SEP}")
            for i in pick_indices(slug, rows, n_ex, length_idx):
                ex = rows[i]
                prompt, target_plain = build_prompt(ex, task=task, cot_mode="none")
                tag = ""
                if slug == "oolong":
                    m = ex.get("_meta", {})
                    tag = (f"  [task_group={m.get('task_group')} "
                           f"dataset={m.get('dataset')} "
                           f"answer_type={m.get('answer_type')}]")
                out.append(f"\n----- {length_label} · example {i+1}{tag} -----")
                out.append(f"\n>>> FULL PROMPT (chars={len(prompt)}):\n{prompt}")
                out.append(f"\n>>> TARGET — no CoT:\n{target_plain}")
                # rerank: also show the HELMET top-10 setting (truncated instruction
                # + CE-desc top-10 IDs) so the .txt covers both output modes. Only
                # the instruction line changes vs the full-pool prompt above.
                if task == "rerank":
                    from corpus_reasoning.lib.prompts import rerank_instruction
                    _, t10 = build_prompt(ex, task=task, output_top_k=10)
                    out.append(f"\n>>> INSTRUCTION — compact top-10 (--output-top-k 10):"
                               f"\n{rerank_instruction(10)}")
                    out.append(f"\n>>> TARGET — compact top-10 (--output-top-k 10):\n{t10}")
                for cot in cots:
                    try:
                        _, target_cot = build_prompt(ex, task=task, cot_mode=cot)
                    except Exception as e:
                        out.append(f"\n>>> TARGET — CoT {cot}: (FAILED: {e})")
                        continue
                    if target_cot == target_plain:
                        out.append(f"\n>>> TARGET — CoT {cot}: (identical to no-CoT — "
                                   f"likely missing `chain_of_thought` in the data)")
                    else:
                        out.append(f"\n>>> TARGET — CoT {cot}:\n{target_cot}")
        out_path = os.path.join(out_dir, f"{slug}.txt")
        with open(out_path, "w") as f:
            f.write("\n".join(out))
        written.append(out_path)
    print(f"wrote {len(written)} files to {out_dir}/:")
    for p in written:
        print(f"  {p}")


if __name__ == "__main__":
    main()
