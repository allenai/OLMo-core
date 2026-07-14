# sft_docchunk/ — document-chunked 5task SFT (Beaker + local)
Docchunk mask family on the canonical 5task mix: dense/compressive/landmark masks, hierarchical
dilated (hier, hierK25 = dilation_cycle rotation), randomdoc ablation. All import
`_docchunk_5task_32k_nocpt_common.py` (shared config) — keep them in this folder.
`Qwen3-4B-docchunk-5task-local.py` is the local torchrun twin.
⚠ Train only from a marker-repaired base — see records/document-chunked-marker-embeddings.md.
