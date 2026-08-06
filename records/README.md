# Records

Standalone writeups that used to hang loose at the repo root: experiment diagnoses, task briefs
for agents, and setup notes. These are *reference documents* — still-valid knowledge, unlike
`deprecated/` (things that must not be used). New writeups of this kind go here, not the repo
root; the root keeps only README/CHANGELOG/CONTRIBUTING/CLAUDE.md/local_cluster.md/beaker.md.

| Doc | What it is |
|---|---|
| `contradiction-data-and-base-hygiene.md` | **Load-bearing**: the running list of *silently wrong* contradiction shards, base checkpoints, sidecars and defaults — things that load, train, and yield a plausible-but-garbage number. Read before picking a base or a shard; check here before adding a new data build. Includes §4b, the FEVER/wiki filler leak into PubMed contradiction evals — **still OPEN for the CTC suite ladder** — and which eval bundle to use. |
| `POSSIBLE_BUG_SFT_DATA.md` | **OPEN**: dense and landmark 5-task SFT arms don't train on the same data. (A) every dense-at-32768 script upsamples contra 2.9 / oolong 1.3 vs the landmark arms' 2.0 / 1.0 — lists the mismatched pairs. (B) even with identical weights the two packers emit different epochs (256k pair: 8,971 vs 10,145 instances, +13.1%). Read before quoting any dense-vs-landmark delta. |
| `document-chunked-marker-embeddings.md` | Diagnosis of Qwen3's untrained (bit-identical) marker-token embeddings + the `fix_marker_embeddings.py` repair. **Load-bearing**: read before any docchunk/landmark training from a fresh base (CLAUDE.md points here). |
| `multihop-gold-routing-experiment.md` | Experiment proposal: can a model use two gold docs that never directly attend to each other? (channel (c) = multi-hop routing across layers). Hop ladder + the leak-matched `hop∞` control. |
| `instruction-tuning-setup.md` | Instruction-tuning / longctx SFT pipeline setup notes (weka-era; some pointers superseded by `local_cluster.md`). |
| `landmark-packing-cp-task.md` | GPU-agent task brief: landmark attention + sequence packing + context parallelism (done). |
| `landmark-sparse-decode-task.md` | GPU-agent task brief: make landmark top-k decode O(k·block) (open). |
| `weka-checkpoint-cleanup.md` | The 2026-07-28 weka audit + cleanup (21.37 TB → 7.2 TB). Why `stepN/` dirs are redundant against the model-only `model_and_optim/` that eval reads, what was deleted, and the paths that must never be. Tooling in `debug/weka_cleanup/`. |
