# moe-v2-core port validation

This directory contains the exact checkpoint gate for the mergeable
`akshitab/moe-v2-core` port. The historical result in `RESULTS.md` was produced
at `0cdcc8b813ab5ca582689edb80e4892891b03ae9`; the current migration reruns the
same gate at `f5376c18424e3f7329fa6e39312c63b84c5f845a` before any smoke or speed
tests.

The gate is intentionally ordered:

1. `beaker_strict_parity.yaml` loads the final optimal 275M Cx1 checkpoint in
   both codebases on one GPU. It requires a complete one-to-one checkpoint-main
   tensor mapping and bitwise-identical fixed-batch logits, every block output,
   and router weights/indices/counts/scores/logits.
2. Only after that passes should the separate training, continuation, eval,
   and systems smokes be submitted. `beaker_postgate.yaml` is retained as the
   historical candidate-control specification and is not submitted
   automatically.
3. The full control uses no in-loop evals. Its final validation is a separate
   post-training job, consistent with the v2 experiment rules.

The only config translations are:

- identical `attention_norm` and `feed_forward_norm` configs become the port's
  single `layer_norm` config;
- attention `d_attn / n_heads` becomes `head_dim` (128 for this family).
- serialized Muon-only optimizer fields are dropped only after requiring every
  source parameter group to have `use_muon=false`; these controls are absent on
  the older port branch and are inert for this AdamW recipe.

The full 275M control uses the historical optimum and exact production batch
geometry: LR `1.6e-3`, 262,144 tokens/global batch, 8,192-token sequences, two
GPUs, EP1, rank MB16, accumulation 1, Cx1. It writes under
`.../olmo-ddp/port-validation/0cdcc8b81/pretraining/`, with one rolling
ephemeral checkpoint every 500 steps and only the final permanent checkpoint.

The 1.2B smoke uses LR `4e-4`, the same 262,144-token global batch, eight GPUs,
EP8 `sync_1d`, MB cap 8/effective MB4, accumulation 1, no checkpoints, and no
in-loop evals. Compare tokens/sec, step time, and peak memory across branches;
the branches use different FLOP accounting, so their reported TFLOPs/MFU are
not directly comparable.

The strict-parity task does not automatically submit downstream work. This
keeps the latest-upstream checkpoint result as an explicit gate.

All three branch-comparison tasks use the unallocated queue: urgent priority,
`minRuntime: 0m`, and `autoResume: true`. The deprecated `preemptible` field is
omitted because Beaker rejects combining it with `minRuntime`/`autoResume`.
