# evals/ — standalone eval launchers (local)
Cross-family eval sbatches: dense base/vllm, generic native gen, 32k ladder, landmark native +
top-k, run_eval_q4b_lc. Family-specific evals live inside their family folder (attn_explore/,
goldgrad/, singletask_ladder/). Eval harness code: src/corpus_reasoning/eval/eval_lc_native*.py.
Reporting rules (eval_size, ≥500, error bars): CLAUDE.md.
