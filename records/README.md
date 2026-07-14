# Records

Standalone writeups that used to hang loose at the repo root: experiment diagnoses, task briefs
for agents, and setup notes. These are *reference documents* — still-valid knowledge, unlike
`deprecated/` (things that must not be used). New writeups of this kind go here, not the repo
root; the root keeps only README/CHANGELOG/CONTRIBUTING/CLAUDE.md/local_cluster.md/beaker.md.

| Doc | What it is |
|---|---|
| `document-chunked-marker-embeddings.md` | Diagnosis of Qwen3's untrained (bit-identical) marker-token embeddings + the `fix_marker_embeddings.py` repair. **Load-bearing**: read before any docchunk/landmark training from a fresh base (CLAUDE.md points here). |
| `instruction-tuning-setup.md` | Instruction-tuning / longctx SFT pipeline setup notes (weka-era; some pointers superseded by `local_cluster.md`). |
| `landmark-packing-cp-task.md` | GPU-agent task brief: landmark attention + sequence packing + context parallelism (done). |
| `landmark-sparse-decode-task.md` | GPU-agent task brief: make landmark top-k decode O(k·block) (open). |
