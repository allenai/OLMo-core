# chunked_eval_speedup

Why chunked-mask vLLM eval was 17.8x slower than dense, and the config that brought it to ~1.7x
(`CHUNK_CACHE_IDS=1`, `max_num_seqs`, `CHUNK_VARLEN_PREFILL=1`, ...). Read `ROUND2_FINDINGS.md` first;
`RUNG_GATE.tsv` / `ROUND4_FREEPATH.tsv` are the small result ledgers the sbatch files append to.

* Microbenchmarks / proofs: `bench_chunk_ids*.py`, `test_chunk_ids_cache_parity.py`,
  `test_free_query_equivalence.py`, `test_varlen_decomposition.py` (import `corpus_reasoning.lib.vllm_chunked_patch`
  from `research/common`).
* Sweeps / gates (sbatch, H200 nodes): `tile_sweep`, `conc_sweep`, `combo_sweep`, `kvcache_sweep`, `blockn_sweep`,
  `fullrung_validate`, `rung_gate_16k32k`, `rung_scaling_sweep`, `validate_varlen*`, `varlen_16k32k_iid`, `engine_*`, ...

Sibling: `../ctc_vllm_validation/` (drivers these call: `run_vllm_eval.py`, serving copies). The two reference
each other heavily (and `ctc_olmo_hybrid`, not ported).

**External dependency:** every sbatch `source /data/prasann/node_local_env.sh` (node-local env preamble: PATH to the
node-local conda/venv, HOME, caches). It lives outside the repo and must be present on the node. Several sbatch also
build a node-local repo mirror under `/data/prasann/repo_r2|r3/OLMo-core` from `$REPO_ROOT`.

Not ported (see `TODO(port)`): `ctc_vllm_validation/general/{build_prefills_any,grade_any}.py`,
`ctc_vllm_validation/node_local/gpu_eval_task_chunked.sh`, standalone `corpus-reasoning` repo (HF-backend arm).
