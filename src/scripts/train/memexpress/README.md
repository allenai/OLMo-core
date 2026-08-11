# memexpress — Prasann's training hub

All of our training code lives here, one folder per experiment family, separated from the
upstream AI2 scripts one level up. Every family folder has its own README with what was run and
where results live. `local_cluster.md` (repo root) covers HOW to run
locally; CLAUDE.md covers Beaker.

| Folder | Family | Where it runs |
|---|---|---|
| `cpt/` | Qwen3/Qwen3.5 base CPT on dolma3+longmino (dense/landmark/sparse/compressive, lr sweeps, 8-node) + `interleaved/`, `debug/` | Beaker |
| `sft_5task/` | The canonical 5task 32k SFT family: dense/compressive/fast-landmark × nocpt/fixnq/cptmix, 64k variants | Beaker |
| `sft_docchunk/` | Document-chunked 5task SFT: dense/compressive/landmark/hier/hierK25/randomdoc (+ `_docchunk_5task_32k_nocpt_common.py` shared config) | Beaker (+ one local) |
| `sft_longctx/` | Earlier Beaker SFT generations: longctx/unified/noruler/10task1k/packed, sparse-landmark — kept for parallel Beaker use | Beaker |
| `sft_xlong256k/` | 256k-window SFT on the xlong5 2k→256k ladder + Dolci 25% (Qwen3.5 dense; query-after data build) | Beaker (2 nodes) |
| `attn_explore/` | 0.6B (+Qwen3.5-0.8B, +4B eval) contradiction-n20 mask-design experiments: dense/dilated/compressive/docchunk-mask-mix/fast-landmark, train+eval | Local (mooney/cubbins) |
| `goldgrad/` | Gold-gradient O(1)-backward probe: train/eval/bench/reap | Local |
| `local_4b/` | 4B local runs: contra n250, cptmix sweeps, docchunk oolong, fastlm cpt40m | Local |
| `singletask_ladder/` | Per-task (not mixed) ladder SFT + multi-rung evals (see its README + EVAL.md) | Local + Beaker |
| `evals/` | Standalone eval launchers (dense/landmark native, vllm, 32k ladder). Family-specific evals live with their family. | Local |
| `probes/` | One-off diagnostics: `sanity_check_packing.py`, `scan_doc_lengths.py` | — |

Related code that stays where it is: `src/corpus_reasoning/train/` (HF-era trainers + the
`convert_hf_to_olmo.py` base-checkpoint converter — package code), `src/scripts/data/` (shard
converters, `fix_marker_embeddings.py`), `src/scripts/local_env.sh` (shared local-job env).
Retired scripts: `deprecated/` at the repo root.
