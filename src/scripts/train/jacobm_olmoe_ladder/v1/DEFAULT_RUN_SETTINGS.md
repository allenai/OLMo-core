# Default run settings at cutover

This is the concise handoff for the legacy experiment regime. See `SETTINGS_AUDIT.md` for the full table of per-size/per-Cx systems settings and historical exceptions.

- Model rungs: 275M, 480M, 810M, and 1.2B active parameters.
- Data rungs: Cx1, Cx2, Cx4, and Cx8. Cx16 is diagnostic, not part of the main grid.
- Sequence length: 8,192 tokens for pretraining and midtraining.
- Optimizer batches: Cx1 = 262,144 tokens (32 sequences), Cx2 = 393,216 (48), Cx4 = 524,288 (64), and Cx8 = 786,432 (96).
- Expert parallelism: EP=1 by default at every size; minimize EP for future DDP runs as well. GPU count and microbatch can change for memory or throughput while the optimizer batch remains fixed.
- Data root: `s3://ai2-llm`.
- Tracking: W&B project `ai2-llm/jacobm-olmoe-ladder`; run names are semantic and resume-stable and exclude systems-only settings.
- LR selection: choose among completed full runs using final-window training CE, primarily the final 250M-token average. Start with factor-of-two brackets; accept a quadratic fit only inside the observed bracket.
- Checkpoint selection: publish the final optimal-LR pretrained/midtrained endpoint for each canonical cell. Do not publish smoke, stopped, eval-only, intermediate, diagnostic, or non-winning LR endpoints by default.
- Baseline architecture: 48 routed experts, top-4 routing, routed hidden size equal to `d_model`, one shared expert of width `d_model / 2`, one dense prefix layer, GQA with half as many KV heads as query heads, and mostly sliding attention with periodic full attention.

These are historical comparability defaults, not a claim that DDP launcher defaults have already been validated. New DDP configs must be smoke-tested before becoming canonical.
