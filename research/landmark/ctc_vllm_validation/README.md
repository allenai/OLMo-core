# ctc_vllm_validation

Validate chunked-mask vLLM against native olmo-core eval, and the recipe for serving olmo-exported
Qwen3.5 checkpoints in vLLM.

* **Serving copy** (all three needed; see CLAUDE.md "Loading an olmo-exported Qwen3.5 in vLLM"):
  `make_vllm_serving_copy.py` -> `make_vl_weights.py` -> `add_dummy_visual.py`. `beaker/` has Beaker-side copies.
* **Env:** `setup_vllm_venv.sh` (+ `.sbatch`) builds the node-local venv (`/data/prasann/ctc_vllm_venv`).
* **Drivers:** `build_prefills*.py` (native-identical prompts) -> `run_vllm_eval.py` / `run_vllm_eval_generic.py`
  (modes `full` | `chunked`) -> `grade_responses*.py`; `pipeline*.sh`, `sweep_*_vllm.sh`, `eval_*_vllm_*.sbatch` chain them.
  Diagnostics: `validate_chunk_leak.py`, `regrade_*.py`, `diag_niah.py`, `smoke_*`.
* **Beaker:** `beaker/launch_eval.sh`, `beaker/eval_pipeline_cu129_apt.sh`, `beaker/run_pipeline*.sh`, `beaker/job_status.py`.
  `beaker/launches/` (ledger + per-launch logs) was dropped.

Dropped from the port: `general/`, `node_local/` (one-off sbatch fills; the tracked `gpu_eval_task.sh` too), `beaker/launches/`,
all grade/response/prefill JSONs, logs, and the findings notes (`*.md`). Scripts that call dropped pieces carry
`TODO(port)` comments. `export_olmo_to_hf.py` (src/corpus_reasoning/train) is also not ported.

Imports `corpus_reasoning.*` from `research/common`. Results/serving copies live on node-local `/data/prasann/ctc_suite*`.
Related: `../chunked_eval_speedup/`, `records/n100-chunked-marker-position-bug.md`.
