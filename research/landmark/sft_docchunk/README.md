# sft_docchunk

`Qwen3-4B-docchunk-5task-local.py` -- 5-task document-chunked SFT on Berkeley slurm (local data, no weka);
variants dense / hierarchical / ... via CLI. Launch with `torchrun` and `PYTHONPATH=src` (see its docstring).
Run `recipes/data/fix_marker_embeddings.py` on the base checkpoint first (`records/document-chunked-marker-embeddings.md`).

The Beaker `*-5task-32k-nocpt-SFT.py` family and `_docchunk_5task_32k_nocpt_common.py` that this mirrors are
not part of the clean branch. Checkpoints/logs: node-local `/data/prasann/...`.
