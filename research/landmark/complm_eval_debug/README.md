# complm_eval_debug

* `check_eval_query_position.py` -- proves `--query-position` reaches the v2 ladder and OOD probes in `eval_lc_native.py`
  (`PYTHONPATH=research/common:src python research/landmark/complm_eval_debug/check_eval_query_position.py`).
* `topk_ablation.sbatch` -- top-k landmark-block ablation with `corpus_reasoning/eval/eval_lc_native_landmark.py`
  (now `research/common/...`). Inputs/outputs under `/data/prasann/complm_debug/` (outside git).
