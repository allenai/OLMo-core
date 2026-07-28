# Records

Standalone writeups that used to hang loose at the repo root: experiment diagnoses, task briefs
for agents, and setup notes. These are *reference documents* — still-valid knowledge, unlike
`deprecated/` (things that must not be used). New writeups of this kind go here, not the repo
root; the root keeps only README/CHANGELOG/CONTRIBUTING/CLAUDE.md/local_cluster.md/beaker.md.

| Doc | What it is |
|---|---|
| `contradiction-data-and-base-hygiene.md` | **Load-bearing**: the running list of *silently wrong* contradiction shards, base checkpoints, sidecars and defaults — things that load, train, and yield a plausible-but-garbage number. Read before picking a base or a shard; check here before adding a new data build. |
| `document-chunked-marker-embeddings.md` | Diagnosis of Qwen3's untrained (bit-identical) marker-token embeddings + the `fix_marker_embeddings.py` repair. **Load-bearing**: read before any docchunk/landmark training from a fresh base (CLAUDE.md points here). |
| `multihop-gold-routing-experiment.md` | Experiment proposal: can a model use two gold docs that never directly attend to each other? (channel (c) = multi-hop routing across layers). Hop ladder + the leak-matched `hop∞` control. |
| `instruction-tuning-setup.md` | Instruction-tuning / longctx SFT pipeline setup notes (weka-era; some pointers superseded by `local_cluster.md`). |
| `landmark-packing-cp-task.md` | GPU-agent task brief: landmark attention + sequence packing + context parallelism (done). |
| `landmark-sparse-decode-task.md` | GPU-agent task brief: make landmark top-k decode O(k·block) (open). |
| `weka-checkpoint-cleanup.md` | The 2026-07-28 weka audit + cleanup (21.37 TB → 7.2 TB). Why `stepN/` dirs are redundant against the model-only `model_and_optim/` that eval reads, what was deleted, and the paths that must never be. Tooling in `debug/weka_cleanup/`. |
