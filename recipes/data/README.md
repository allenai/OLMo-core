# recipes/data — task rows → tokenized SFT shards, and base-checkpoint marker repair

Overview and the canonical setA command: `recipes/README.md` §1. Outputs (sets, shards, repaired
bases) live on weka / node-local `/data`, never in git.

| File | What |
|---|---|
| `ctc_sft/build_ctc_sft.py` | One-command, manifest-checked builder of the CTC SFT sets (`--set set-a`); `--convert` tokenizes, `--check` re-hashes a build. Needs `pip install -e ctc/`. |
| `ctc_sft/ctc_calibrated.py` | Runs `ctc-data` with rung sizes from a measured calibration (`CTC_DOCS_OVERRIDE`). |
| `ctc_sft/measure_eval_ladder.py` → `ctc_sft/eval_calibration.json` | Per-(task, rung) document counts measured from the eval's own rows (input to `--calibration`). Reads `recipes/eval/setA_audit/olmo_eval_roster.json`. |
| `generators/generate_xabsence_data.py`, `generators/generate_arxiv_grouping_data.py` | The two in-repo task generators `build_ctc_sft.py` calls (ex-`src/corpus_reasoning/data/`); `cr_io.py` is the vendored `corpus_reasoning.lib.io`. xabsence's LLM paraphrase-pool mode (`build_pool_llm`) still imports `corpus_reasoning.*` and needs `research/common` on `PYTHONPATH`; the `--pool` path used by setA does not. |
| `convert_unified_to_document_landmark.py` | Unified JSONL → shards: `--emit dense` (doc-chunked/dense), `landmark`, or `summary`; writes `metadata.json` (incl. `query_position`, per-task gold index base for `--emit-gold-sidecar`). |
| `convert_longctx_tasks_to_sft.py` | Older JSONL → SFT-shard converter (no document markers); imported by `research/token_routing/pooled_kv/landmark_probe/landmark_slot_probe.py`. |
| `fix_marker_embeddings.py` (+ `_gantry.sh`, `_qwen35.py`) | Audit (`--audit-only`; API `audit`/`print_audit`/`load_keys`/`EMB_KEY` used by `train/setA/prep_setA_sft.py`) and repair of the untrained marker rows (cosine AND norm). See `records/document-chunked-marker-embeddings.md`, `records/n100-chunked-marker-position-bug.md`. |
