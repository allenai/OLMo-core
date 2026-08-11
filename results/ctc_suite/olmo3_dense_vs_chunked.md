# OLMo-3-7B — dense vs document-chunked, CTC ladder

Base: **OLMo-3-7B**. Both arms fine-tuned from the same base on the same 10k-example joint
2k-32k uniform-mix ladder; `chunked-mix` = document-chunked attention with the p 0.8→0.0
mask-mixing curriculum. In OLMo-3 only the 8 full-attention layers are chunked (the 24 sliding-window
layers are left as pretrained), so this is the same partial-chunking setting as Qwen3.5.

All cells `eval_size=500`, `parse_rate=1.0`, `skipped_too_long=0` — no truncation losses.
± values are binomial SE at eval_size=500; these are set-valued F1s so treat the SE as approximate,
and note it captures eval noise only, not run-to-run seed variation.

## contradiction (N² comparison)

Metric: `set_f1`.

| rung | dense (full) | chunked-mix | gap |
|---|---|---|---|
| 2k | 0.829 | 0.767 | **+0.063** ±0.025 |
| 4k | 0.742 | 0.649 | **+0.093** ±0.029 |
| 8k | 0.646 | 0.449 | **+0.197** ±0.031 |
| 16k | 0.426 | 0.194 | **+0.232** ±0.028 |

## qdmatch_hpqa (N·M comparison)

Metric: `pair_f1`.

| rung | dense (full) | chunked-mix | gap |
|---|---|---|---|
| 2k | 0.997 | 0.987 | **+0.010** ±0.005 *(n.s.)* |
| 4k | 0.996 | 0.790 | **+0.206** ±0.018 |
| 8k | 0.985 | 0.448 | **+0.536** ±0.023 |
| 16k | 0.962 | 0.216 | **+0.746** ±0.020 |

## hotpotqa (N retrieval)

Metric: `gold_id_f1`.

| rung | dense (full) | chunked-mix | gap |
|---|---|---|---|
| 2k | 1.000 | 0.998 | **+0.002** ±0.002 *(n.s.)* |
| 4k | 0.998 | 0.997 | **+0.001** ±0.003 *(n.s.)* |
| 8k | 0.997 | 0.993 | **+0.004** ±0.004 *(n.s.)* |
| 16k | 0.987 | 0.975 | **+0.012** ±0.009 *(n.s.)* |

## Summary: the gap tracks task structure, not context length

Same checkpoints, same harness, same rungs at 8k — only the task's comparison structure differs.

| task | class | dense | chunked | gap @8k |
|---|---|---|---|---|
| contradiction | N² comparison | 0.646 | 0.449 | **+0.197** ±0.031 |
| qdmatch_hpqa | N·M comparison | 0.985 | 0.448 | **+0.536** ±0.023 |
| hotpotqa | N retrieval | 0.997 | 0.993 | **+0.004** ±0.004 |

Chunked attention matches dense on O(N) retrieval (gap within ~2 SE at every rung up to 16k,
both arms at ceiling) and collapses on tasks that require comparing documents against each other.

### Provenance

| field | value |
|---|---|
| contradiction / qdmatch ckpts | trained + evaluated on AI2 Beaker (jupiter), native backend |
| hotpotqa ckpts | trained on Beaker, evaluated on Berkeley H200s (job 3378560), native backend |
| contradiction max_length | 32768 (raised — the default rung+2048 silently truncated 23–39% of prompts) |
| eval_size | 500 per cell |

Superseded files carrying the truncation bug are kept alongside as
`results/ctc_suite/contradiction/olmo3-7b-maxlen-truncated_*` — do not use them.
