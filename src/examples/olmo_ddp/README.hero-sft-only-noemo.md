# SFT-only EMO ablation

Three trials from the original **EMO PT -> EMO MT -> EMO LC** final checkpoint,
with **EMO disabled only during SFT**. This is not the separately running branch
that disables EMO during both MT and LC.

- Source: original LC experiment 01M2CJ0D247DDJF8C5CV66SRZ5, step5961.
- LRs: 1e-5, 5e-5, 1e-4; two epochs / 1810 updates, saves at 905 and 1810.
- Each run: 8 B300 GPUs, urgent allocated, ai2/olmo3p5-training.
- Same data, assistant masks, seed1729, 524288-token global batch, 65536-token
  packed sequence, linear decay and 3% warmup, fresh optimizer, eager execution,
  block recomputation, FLA packed KDA, precision and architecture.
- Real-image config gate compares data and configs against the original sweep:
  only model router EMO changes, plus campaign identities/storage paths.
- One four-step GPU smoke verifies source weights, optimizer reset, a step2
  full-state restart, all eight ranks, and exact original first-batch hashes.
- Both epochs convert with the existing strict conversion/metadata gates.
- Epoch2 only: Math500, IFBench, HumanEval, AlpacaEval, unchanged generation/judge
  settings. Independent HF/vLLM qualification and a four-suite chat smoke gate
  full eval fanout. Failed qualification holds that checkpoint; no blind retries.
- Two resource-free Phobos watchers; conversion/eval workers are urgent allocated
  in ai2/OLMo-3-moe-experiments. Frozen source commits and durable submit-once intents.
- Separate checkpoint, upload, conversion and eval namespaces; no original jobs,
  outputs or source checkpoints modified or deleted.

Inference keeps the previously documented provisional fast-BF16 caveat. The
original EMO SFT outputs include off-topic answers; this ablation tests whether
changing SFT EMO improves them, without presuming their root cause.
