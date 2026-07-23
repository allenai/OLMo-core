# MoE Ladder Settings Audit

Last refreshed: 2026-06-12.

This file is the current source of truth for optimizer-batch and systems settings
used by the MoE A0 ladder and first ablations. Older docs may still mention
`mid_480m`, `midpoint`, or Cx2 `b512k`; those are historical unless repeated
here.

Unless noted otherwise:

- Sequence length is 8192 tokens.
- Total batch tokens = `global_batch_size_seq * 8192`.
- Expert parallelism is EP=1.
- Data root is `s3://ai2-llm`.
- LR and architecture decisions currently use final-window training CE only.
- Validation/eval metrics are tracked but observational.

## Naming

Canonical model-size names are active-parameter labels:

| Canonical name | Compatibility / historical names | Notes |
| --- | --- | --- |
| `275m` | tiny 275M | Architecture-search scale; old run names often include `tiny-275m`. |
| `480m` | `mid_480m`, midpoint, `m480` run prefix | `mid_480m` remains a code alias; new docs should say `480m`. |
| `810m` | 810M | Larger baseline rung. |
| `1p2b` | 1.2B | Larger baseline rung. |

## Canonical Settings Going Forward

Use these settings for new comparable baseline and ablation runs unless Jacob
explicitly approves a special-case systems change.

Cx2 is now canonical at 393,216 tokens / 48 sequences (`b384k`) for all model
sizes. Earlier 275M `b256k`, expert-granularity `b512k`, 480M `b512k`, 810M
`b512k`, and planned 1.2B `b512k` Cx2 references are diagnostic or stale.

| Model | Cx | Batch tokens | `global_batch_size_seq` | GPUs | EP | Microbatch | Current status / notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 275M | 1 | 262,144 | 32 | 1-2 | 1 | 16 | Completed canonical family uses `gpu2-ep1mb16`; future one-GPU is acceptable if systems are healthy. |
| 275M | 2 | 393,216 | 48 | 2 | 1 | 8 | Canonical repaired `b384k` family; completed for baseline and expert granularity. |
| 275M | 4 | 524,288 | 64 | 4 | 1 | 16 | Completed canonical family. |
| 275M | 8 | 786,432 | 96 | 8 | 1 | 4 | Future policy; historical baseline completed on 4 GPUs / mb8. |
| 275M | 16 | 1,048,576 | 128 | 8 | 1 | 16 | Reserve for important/unclear cases only. |
| 480M | 1 | 262,144 | 32 | 4 | 1 | 8 | Completed canonical family. |
| 480M | 2 | 393,216 | 48 | 4 | 1 | 4 | Canonical repaired `b384k` family; old `b512k` is diagnostic. |
| 480M | 4 | 524,288 | 64 | 4 | 1 | 8 | Completed canonical family. |
| 480M | 8 | 786,432 | 96 | 8 | 1 | 4 | Completed canonical family. |
| 480M | 16 | 1,048,576 | 128 | 8 | 1 | 4 | Not a default target; reserve for important/unclear cases. |
| 810M | 1 | 262,144 | 32 | 8 | 1 | 4 | Some completed runs used 4 GPUs; future policy prefers 8 GPUs. |
| 810M | 2 | 393,216 | 48 | 8 | 1 | 2 | Canonical repaired `b384k` family; old `b512k` is diagnostic. |
| 810M | 4 | 524,288 | 64 | 8 | 1 | 4 | Completed canonical family. |
| 810M | 8 | 786,432 | 96 | 16 | 1 | 4 | Future policy; historical completed baseline used 8 GPUs. |
| 810M | 16 | 1,048,576 | 128 | 8-16 | 1 | 4 | Avoid for now unless ambiguity makes it worthwhile. |
| 1.2B | 1 | 262,144 | 32 | 8 | 1 | 2 | Completed canonical family. |
| 1.2B | 2 | 393,216 | 48 | 8 | 1 | 2 | Should be launched soon; canonical `b384k`. |
| 1.2B | 4 | 524,288 | 64 | 16 | 1 | 2 | Future policy; historical completed runs used 8 GPUs. |
| 1.2B | 8 | 786,432 | 96 | 8 | 1 | 4 | Preferred replacement policy after 32-GPU / mb1 underperformed. |
| 1.2B | 16 | 1,048,576 | 128 | 16-32 | 1 | 2 | Avoid for now unless ambiguity makes it worthwhile. |

## Baseline Historical Mismatches

These runs may still be useful for diagnostics, but should not be silently merged
into canonical U-curves.

| Area | Historical setting | Current interpretation |
| --- | --- | --- |
| 275M Cx1 | Multiple families, including original 2M batch and EP=8 sanity checks | Use canonical `gpu2-ep1mb16` for ladder plots. EP=8 sanity runs are diagnostic only. |
| 275M Cx2 | Old baseline `b256k`; old expert-granularity `b512k` | Superseded by `b384k` repair. |
| 275M Cx8 | Baseline completed on 4 GPUs / mb8; expert-granularity uses 8 GPUs / mb4 | Optimizer batch is matched; systems differ and should be documented. |
| 480M Cx2 | Initial `b512k` family | Diagnostic only after `b384k` repair. |
| 810M Cx1 | Main sweep used 4 GPUs; one sentinel used 8 GPUs | Same optimizer batch and microbatch; future policy prefers 8 GPUs. |
| 810M Cx2 | Completed old `b512k` family | Diagnostic only after `b384k` repair. |
| 810M Cx8 | Completed on 8 GPUs | Valid completed baseline; future launches may prefer 16 GPUs for wall clock. |
| 1.2B Cx4 | Completed on 8 GPUs | Valid completed baseline; future policy may use 16 GPUs. |
| 1.2B Cx8 | Initial 32-GPU / mb1 jobs had poor throughput | Keep completed `4e-4` as systems comparison; prefer 8-GPU / mb4 replacements. |

## Expert Granularity Settings

Serious variants:

| Variant | Experts | `top_k` | `moe_hidden_size` | Role |
| --- | ---: | ---: | ---: | --- |
| `eg24e2k` / `coarse_24e_top2` | 24 | 2 | `2 * d_model` | Serious coarse endpoint. |
| baseline `eg48e4k` / `baseline_48e_top4` | 48 | 4 | `d_model` | Current A0 control. |
| `eg96e8k` / `fine_96e_top8` | 96 | 8 | `d_model / 2` | Serious fine endpoint. |

Diagnostic curiosities:

| Variant | Experts | `top_k` | `moe_hidden_size` | Current policy |
| --- | ---: | ---: | ---: | --- |
| `eg192e16k` / `extreme_192e_top16` | 192 | 16 | `d_model / 4` | Diagnostic only; Cx1 is noisy and not curve-fit cleanly. |
| `eg384e32k` / `ultra_384e_top32` | 384 | 32 | `d_model / 8` | Diagnostic only. |

Canonical 275M expert-granularity systems:

| Cx | Batch tokens | `global_batch_size_seq` | GPUs | EP | Microbatch notes |
| ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 262,144 | 32 | 1 | 1 | Coarse mb16; fine mb8; extreme mb4; ultra mb2. |
| 2 | 393,216 | 48 | 2 | 1 | Repaired `b384k`; coarse/fine mb8. |
| 4 | 524,288 | 64 | 4 | 1 | Coarse mb16; fine mb8. |
| 8 | 786,432 | 96 | 8 | 1 | Coarse/fine mb4. |
| 16 | 1,048,576 | 128 | 8 | 1 | Not launched by default. |

## Total Sparsity Settings

Total sparsity is paused only because expert granularity is currently higher
priority under compute limits. If additional compute opens up, this axis can be
queued again.

Variants:

| Variant | Experts | `top_k` | `moe_hidden_size` | Role |
| --- | ---: | ---: | ---: | --- |
| baseline `sp48e4k` | 48 | 4 | `d_model` | Current A0 control. |
| `sp96e4k` / `high_total_96e_top4` | 96 | 4 | `d_model` | More total capacity at fixed active routed compute. |
| `sp192e4k` / `huge_total_192e_top4` | 192 | 4 | `d_model` | Aggressive sparse point. |

Do not run `sp24e4k` in the first wave unless Jacob explicitly asks for a
low-total diagnostic.

Canonical first-wave 275M systems:

| Cx | Batch tokens | `global_batch_size_seq` | GPUs | EP | Microbatch |
| ---: | ---: | ---: | ---: | ---: | ---: |
| Smoke | 262,144 | 32 | 4 | 1 | 4 |
| 1 | 262,144 | 32 | 4 | 1 | 4 |
| 4 | 524,288 | 64 | 4 | 1 | 4 |
| 8 | 786,432 | 96 | 8 | 1 | 4 |

Only `sp96e4k` Cx1 `1e-3` is known to have completed from the first partial
wave; most other tracked sparsity jobs were manually canceled.

## External Dense-Ladder Reference

The external `ai2-llm/olmo-hybrid-pe` workspace motivated the Cx2 repair because
its comparable 260M Cx2 runs used an intermediate batch size rather than our old
`b256k` or the old expert-granularity `b512k`. The local canonical choice is now
`b384k` for consistency across model sizes.
