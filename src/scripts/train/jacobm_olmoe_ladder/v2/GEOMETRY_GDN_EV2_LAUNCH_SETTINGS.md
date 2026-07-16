# 275M geometry + GDN expand_v=2 launch settings

This is the launch contract for the first geometry-matched experiment. The
model is the `geometry_only` profile in `models/geometry_matched_275m.py`:
`d_model=640`, 10 layers, GDN at layers 0–3 and 5–8, full attention at layers
4 and 9, `expand_v=2`, 8 Q / 4 KV heads, 664-wide experts, a 5,976-wide dense
first FFN, RoPE, no attention gate, and `init_std=0.01`.

Exact counts are 290,782,080 active, 226,556,800 active non-embedding, and
3,136,314,240 stored parameters. Total active parameters are 0.898% above the
current 288.19M hybrid. Active non-embedding parameters—and therefore the
Cx-derived token budgets—are 7.31% higher.

## Fixed scientific settings

- Train from initialization on `OLMo_mix_0925` with the Dolma 2 tokenizer.
- Sequence length is 8,192.
- Optimizer batches remain 32/48/64/96 sequences for Cx1/2/4/8.
- Use DDP on two Holmes B300s with EP/TP/PP/CP all equal to one.
- Use weight decay 0.1, betas (0.9, 0.95), the established no-decay and routed
  expert optimizer groups, and gradient clipping at 1.0.
- Use a 10%-of-token-budget linear warmup followed by cosine decay to 10% of
  peak LR.
- Disable all in-loop and final-step evaluators. Run validation after training
  in separate eval-only jobs.
- For full runs, write a rolling ephemeral checkpoint every 500 steps with
  `remove=ephemeral_only`; retain only the final checkpoint permanently.

| Cx | Target tokens | Global batch tokens | Target steps | Final-step tokens | Warmup tokens | Warmup steps |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 4,531,136,000 | 262,144 | 17,285 | 4,531,159,040 | 452,984,832 | 1,728 |
| 2 | 9,062,272,000 | 393,216 | 23,047 | 9,062,449,152 | 905,969,664 | 2,304 |
| 4 | 18,124,544,000 | 524,288 | 34,570 | 18,124,636,160 | 1,811,939,328 | 3,456 |
| 8 | 36,249,088,000 | 786,432 | 46,094 | 36,249,796,608 | 3,624,665,088 | 4,609 |

## Microbatch capacity study

The no-checkpoint smokes run 12 optimizer steps with compilation, W&B, the
speed/MFU callback, and Beaker progress reporting enabled. They write no model
or optimizer checkpoints and run no evaluators.

The largest legal no-accumulation shape was tested first at every Cx. The
validated production settings are:

| Cx | Global sequences | GPUs | Validated MB | Accumulation | Active memory | Mean TFLOPs/GPU | W&B |
|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 32 | 2 | 16 | 1 | 197.3 GiB | 368.0 | `4zeb0iah` |
| 2 | 48 | 2 | 12 | 2 | 156.8 GiB | 413.2 | `hj0ip22r` |
| 4 | 64 | 2 | 16 | 2 | 197.3 GiB | 387.4 | `adpjvm8b` |
| 8 | 96 | 2 | 16 | 3 | 197.3 GiB | 471.1 | `rl5kz2u5` |

Each promoted shape completed a compiled forward/backward dry run and 12 real
optimizer steps with exit code zero. TFLOPs/GPU is the arithmetic mean of the
11 reported step-level samples; these very short runs are capacity tests, not
stable throughput benchmarks. MB24, MB32, and MB48 all produced genuine CUDA
OOMs during the compiled dry run. Because Cx2 MB24 and Cx8 MB24 have identical
per-rank tensor shapes, the former also rules out the latter. The first smoke
attempt was stopped after it exposed a missing `BEAKER_TOKEN`; it is an
infrastructure-canceled attempt, not a memory result. No smoke wrote a model or
optimizer checkpoint.

## LR sweep

All four Cx batch shapes passed. Launch these four inherited LRs at every
Cx: `4e-4`, `8e-4`, `1.6e-3`, and `3.2e-3`. This is 16 tasks and 32 B300s at
maximum concurrency. The full manifest contains the smoke-validated
microbatches above.

This intervention's smokes and LR sweep have an explicit scheduling exception:
urgent priority with `minRuntime: 0m` and `autoResume: true`. They are therefore
unallocated, immediately interruptible jobs and do not use the workspace's
allocated concurrency slots. Do not apply this exception to other experiment
families without a new decision.
