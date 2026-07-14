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

The standalone data checkout at `/scratch/users/prasann/corpus-reasoning` has its own
`deprecated/` with the pre-port env-bootstrap sbatch scripts (`setup_olmo_env.sh` etc.) that
referenced the deleted `/scratch/users/prasann/OLMo-core` clone.
