# research/landmark

Landmark / document-chunked / summary-token attention work: SFT launchers, vLLM chunked-eval
validation and speed-up, sparse-landmark training and inference benchmarks. Library code lives in
`src/olmo_core/nn/attention/` (not here); eval helpers it imports live in `research/common/`.

| dir | what it is | entry points |
|---|---|---|
| `singletask_ladder/` | per-task length-ladder SFT (Qwen3-4B, dense/landmark/compressive/docchunk) + launchers and the eval-bundle uploader | `launch_singletask_10k_overnight.sh`, `launch_singletask_10k_3variant.sh`, `run_q4b_singletask_ladder.sbatch`, `EVAL.md` |
| `sft_docchunk/` | the single local (Berkeley, no weka) 5-task docchunk SFT script | `Qwen3-4B-docchunk-5task-local.py` |
| `summtoken/` | summary-token (SummTokenSFT) 5-task SFT arms, shard builder, mask verifier | `Qwen3.5-4B-summ-*-5task-SFT.py`, `build_summary_token_shards{.py,_gantry.sh}`, `verify_summary_mask_*` |
| `chunked_eval_speedup/` | rounds of measurement that took chunked-mask vLLM eval from 17.8x to ~1.7x dense wall-clock | `ROUND2_FINDINGS.md`, `RUNG_GATE.tsv`, `*_sweep.sbatch`, `test_*.py` |
| `ctc_vllm_validation/` | the vLLM serving-copy recipe and the run/grade drivers used to validate chunked vLLM against native olmo-core eval | `make_vllm_serving_copy.py`, `make_vl_weights.py`, `add_dummy_visual.py`, `run_vllm_eval*.py`, `grade_responses*.py`, `setup_vllm_venv.sh`, `beaker/` |
| `sparse_landmark_training/` | sparse-landmark training design, benchmark + results | `DESIGN.md`, `RESULTS.md`, `bench_train_step.py` |
| `sparse_landmark_inference/` | sparse landmark decode kernel tests and speed-up notes | `README.md`, `bench_decode.py`, `test_sparse_decode.py` |
| `complm_eval_debug/` | query-position check + top-k ablation for the compressive-landmark eval | `check_eval_query_position.py`, `topk_ablation.sbatch` |

## Paths and layout

* Repo root is derived (`REPO_ROOT=${REPO_ROOT:-$(git -C "${SLURM_SUBMIT_DIR:-$(dirname "$0")}" rev-parse --show-toplevel)}`
  in shell, `__file__`-relative in Python). Set `REPO_ROOT` to run from a different checkout.
* `corpus_reasoning.*` imports resolve through `research/common` -- put it on `PYTHONPATH`
  (`PYTHONPATH=research/common:src`). Scripts here already do.
* The three eval launchers that used to sit in `singletask_ladder/` now live in
  `recipes/eval/legacy_native/` (`run_q4b_beaker_multirung_eval.py`, `run_beaker_multirung_eval.sh`,
  `run_q4b_stl_multirung_eval.sbatch`); references were updated.
* Many `chunked_eval_speedup/*.sbatch` build a node-local mirror (`/data/prasann/repo_r2|r3/OLMo-core`)
  by rsync from `$REPO_ROOT`, then run from it; `LOCAL_REPO` (default `/data/prasann/repo/OLMo-core`) is
  the same idea. These also `source /data/prasann/node_local_env.sh`, which lives OUTSIDE the repo
  (node-local env preamble) -- it must exist on the target node.
* Checkpoints, rungs, vLLM serving copies, results and job logs live outside git (`/data/prasann/...`,
  weka, `/data/prasann/joblogs`).

## Not ported (look for `TODO(port)` comments)

Scripts still reference these; they were deliberately left out of this branch:
`debug/ctc_vllm_validation/general/` (`build_prefills_any.py`, `grade_any.py`),
`debug/ctc_vllm_validation/node_local/` (`gpu_eval_task*.sh`, `prep_*`),
`src/corpus_reasoning/train/export_olmo_to_hf.py`, `debug/ctc_olmo_hybrid/`, `debug/ctc_crossfamily/`,
`src/scripts/eval/ctc_suite/` (`write_vllm_result.py`, `validate_chunk_leak.py`),
`sft_docchunk/_docchunk_5task_32k_nocpt_common.py` (comment references only),
and the standalone `corpus-reasoning` repo clone (`CR=` in `engine_*`/`hf_vs_vllm_chunked.sbatch`, HF-backend arm).

## Records

* `records/landmark-packing-cp-task.md`
* `records/landmark-sparse-decode-task.md`
* `records/multihop-gold-routing-experiment.md`
* `records/free-pad-probe-is-confounded.md`
* `records/n100-clean-results-2026-07-14.md`
* `records/soft-detach-plus-compressive-landmark.md`
* `records/document-chunked-marker-embeddings.md` (run `recipes/data/fix_marker_embeddings.py` before any docchunk/landmark training)
* `records/n100-chunked-marker-position-bug.md`
