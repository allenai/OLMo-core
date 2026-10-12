# research/common

Code shared by several `research/*` components. Put this directory on `PYTHONPATH`
(`PYTHONPATH=research/common:src`, or `sys.path.insert(0, "research/common")`) and
`import corpus_reasoning.lib.x` / `corpus_reasoning.eval.x` works.

## Contents

* `corpus_reasoning/{eval,lib}/` -- vendored from the old `src/corpus_reasoning/`, **only** the import closure of what
  `research/` code uses (8 eval + 11 lib files + `__init__.py`s; computed from `from corpus_reasoning...` imports,
  transitively, starting at the files named below). Files are unmodified copies except `corpus_reasoning/__init__.py`
  (docstring) and two path comments in `lib/vllm_chunked_patch.py`.
  * eval: `evaluate`, `eval_lc_native_docchunk`, `eval_lc_native_docchunk_contra`, `eval_lc_native_landmark`,
    `batched_native_decode`, `batched_chunked_serial_prefill`, `chunked_batch_helpers`, `shared_corpus_cache`
  * lib: `vllm_chunked_patch`, `eval_tasks`, `metrics`, `io`, `data_format`, `prompts`, `chunked_attention`,
    `vllm_utils`, `adapter_save`, `longlora_attn`, `olmo3_mask_patch`
* `beaker_native_lengthmix_eval.py` -- launches the native (olmo-core) length-mix ladder eval on Beaker; from
  `debug/outlier_lengthmix_scaling/`. It runs `recipes/eval/legacy_native/ctc_eval/eval/eval_lc_native.py` in the container.
* `taskscale_points.json` -- per-task/arm/length accuracy points (from `debug/taskscale_lengthmix/points.json`) read by the
  task-scale/length-mix scaling plots (those plot scripts are not ported).

## Who imports it

| consumer | uses |
|---|---|
| `research/landmark/ctc_vllm_validation/` (`run_vllm_eval*.py`, `build_prefills*.py`, `grade_responses*.py`, `validate_chunk_leak.py`, `sweep_task_vllm.sh`, pipelines) | `lib.vllm_chunked_patch`, `eval.evaluate`, `eval.eval_lc_native_docchunk{,_contra}` |
| `research/landmark/chunked_eval_speedup/` (tests, benches, sbatch) | `lib.vllm_chunked_patch`, `eval.eval_lc_native_docchunk` |
| `research/landmark/complm_eval_debug/` | `eval.evaluate`, `eval.eval_lc_native_landmark` |
| `research/landmark/singletask_ladder/` launchers | `PYTHONPATH` only (the evals they drive live in `recipes/eval/legacy_native/`) |
| `research/token_routing/devloss_grid/ctc_devloss_grid.py` | `eval.eval_lc_native_docchunk.TASK_CFG` |
| `research/ffn_compute/ffnmoe_train/eval_q4b_ffnmoe_contra.sbatch`, `research/token_routing/pooledkv_train/eval_q4b_pooledkv_contra.sbatch` | `eval.eval_lc_native_docchunk_contra` (run as a script) |
| `research/common/beaker_native_lengthmix_eval.py` | launcher only |

Not included on purpose: everything else in the old `src/corpus_reasoning/` (data generators, training, analysis, other eval
drivers). `recipes/data/generators/cr_io.py` carries its own copy of `lib/io.py`.
