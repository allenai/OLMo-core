# Deprecated scripts

Scripts that must NOT be used anymore, kept for reference instead of deleted. Nothing in the live
tree may reference anything in here.

**Convention:** when a script is retired, `git mv` it here preserving its repo-relative path
(`src/corpus_reasoning/train/foo.py` → `deprecated/corpus_reasoning/train/foo.py`), and add a row
below stating why it's dead and what replaces it. If the reason is a directive (a data source or
recipe that must never be used again), say so explicitly — those entries exist precisely so a
future reader/agent doesn't resurrect the pattern.

| Script | Deprecated | Why / replacement |
|---|---|---|
| `corpus_reasoning/train/convert_qwen3_to_olmo.py` | 2026-07-13 | Body just raises `NotImplementedError`. Use `src/corpus_reasoning/train/convert_hf_to_olmo.py`. |
| `corpus_reasoning/train/convert_qwen3_4b_to_olmo.py` | 2026-07-13 | Hardcoded to Qwen3-4B and an eos=151645 sidecar footgun; generalized by `src/corpus_reasoning/train/convert_hf_to_olmo.py` (resolves arch from HF config, derives eos/pad from the real tokenizer). |
| `corpus_reasoning/data/build_nq_ladder64k.sh` | 2026-07-13 | **DIRECTIVE: never use the old ~98%-hard-negative NQ data** (hn49/hn99/hn199/ladder64k). Only the p10 pipeline (hard-neg ratio ≈ 10% of k + cross-encoder filter) is valid; v2 eval ladders are p10-sourced. Audit hard-ratio ≈ 0.10 before using any NQ file. |

| `launch_long_context_evals.sh` | 2026-07-13 | Pre-Berkeley gantry/weka HELMET+RULER eval launcher; depends on sibling checkouts `../ai2-helmet` / `../olmo-cookbook` that no longer exist. Local evals: see `local_cluster.md`; Beaker evals: the eval-ladder sbatch/gantry scripts in `src/scripts/train/` (evals/, singletask_ladder/). |
| `launch_cr_suite_evals.sh`, `launch_longctx_task_evals.sh` | 2026-07-13 | Pre-Berkeley gantry/weka cr-suite + oolong/contradiction eval launchers (oe-eval branch `prasann/longctx-eval`); superseded by the native local eval flow (`src/corpus_reasoning/eval/eval_lc_native*.py`). |
| `scratch_transfer_cptmix32k.sh` | 2026-07-13 | Completed one-off (weka→s3 model-only resave of the cptmix32k step122 ckpt). Keep as a template for future weka→s3 transfers — adapt a copy, don't rerun. |

| `corpus_reasoning/debug/` (53 files), `corpus_reasoning/diag/` (9 files) | 2026-07-13 | One-off debugging/diagnostic probes (qwen35 parity suite, hierarchical-mask diags, smoke scripts). Zero live imports; kept as historical evidence — several are cited in experiment records. |
| `corpus_reasoning/train/{train_retrieval_baseline,train_pair_classifier,train_chunked_smoothed,olmo_qwen3_cp_smoke,olmo_nq_sft,benchmark_chunked_attn}.py` | 2026-07-13 | HF-era / smoke trainers with zero references; training now runs via the olmo-core scripts in `src/scripts/train/memexpress/`. (`convert_hf_to_olmo.py`, `export_olmo_to_hf.py`, `train.py`, `train_chunked*.py` remain live in `src/corpus_reasoning/train/`.) |

The standalone data checkout at `/scratch/users/prasann/corpus-reasoning` has its own
`deprecated/` with the pre-port env-bootstrap sbatch scripts (`setup_olmo_env.sh` etc.) that
referenced the deleted `/scratch/users/prasann/OLMo-core` clone.
