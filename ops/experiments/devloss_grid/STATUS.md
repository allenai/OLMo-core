---
experiment: devloss_grid
title: Dev-loss (answer-CE) grid of task-general soft-token keep schemes x the CTC suite rows (+cpt80), 2k/8k/32k, frozen dense Qwen3.5-4B checkpoints
status: done
updated: 2026-10-01
branch: prasann/landmark
launcher: research/token_routing/devloss_grid/ctc_devloss_grid.py
records:
  - records/devloss-grid-underperformers.md
  - records/learned-token-router.md
  - records/outlier-saliency-preview-probe.md
  - records/trained-parity-winning-arms.md
results:
  - /accounts/projects/berkeleynlp/prasann/projects/OLMo-core/debug/devloss_grid/grid.csv
  - /accounts/projects/berkeleynlp/prasann/projects/OLMo-core/debug/devloss_grid/grid.json
  - /accounts/projects/berkeleynlp/prasann/projects/OLMo-core/debug/devloss_grid/results_skip/
  - /accounts/projects/berkeleynlp/prasann/projects/OLMo-core/debug/devloss_grid/results_rand20p8/
wandb_group:
next_step: None for the grid itself. It is now the scoring harness for learned_router; any new keep scheme is added as a grid row
---

# devloss_grid

## Question
Is there one task-general token-selection rule ("a unified algorithm that could work for any kind of
data", Prasann 2026-09-21) whose soft/compacted construction keeps answer CE at parity with full context
on every CTC task? Cells are teacher-forced answer CE on frozen dense 4B suite checkpoints, with no training.

## Decisions
- 2026-09-21 (Prasann): use the standard olmo-eval CTC suite. Rows and rendering come from the vendored `ctc` package (HF `PrasannSinghal/ctc-suite-eval`). Checkpoints are the dense 4B suite exports (`ctc-4b-<task>-full`).
- 2026-09-21 (Prasann): gold information is allowed if gold documents are used uniformly across tasks. All schemes are header-free, with K counted from doc_start, so nothing is task-specific.
- 2026-09-21 (Prasann): added the extra column `cpt80` (amandab's CPT setting). Loss is on the last 20% of a long document, with the first 80% as pooled 512-token pseudo-docs, on the base model.
- 2026-09-21 (agent): dropped columns: hpqa, qdmatch_fiqa/nq, outlier_fixedM. xabsence is VOID (paraphrase-era checkpoint on EXACT rows). Rows are 16/16/8 at 2k/8k/32k.
- 2026-09-21 (agent): a pure-fraction budget pools document ids on short docs. Fix: `p8`, a uniform 8-token real prefix per document. With it, `gold_fl20p8` is the leading uniform rule.
- 2026-09-21 (agent): the frozen screen ranks INFERENCE usability, not training value. Example: rand20 is worst here but sd20 wins in CPT (see softdetach). Do not pick training arms from this grid.
- 2026-09-29 (agent): the grid becomes the scoring harness for the learned router (`results_router_*`). The router uses `gold_fl20p8_noslot` as "the bar".

## Current best numbers
Median ΔCE vs full, over the 17 non-VOID suite tasks (15 at 32k). ⚠ eval_size 16/16/8 rows per cell, by design.

| arm | task / rung | metric | eval_size | ±SE | cot | source |
|---|---|---|---|---|---|---|
| `gold_fl20p8` (gold whole + 8-tok prefix + first_last 20%, cent_cmean slot) | 17 tasks / 2k | ΔCE≤0.05 on 13/17; median +0.002; compaction x0.49 | 16 ⚠ eval_size=16 | per-cell paired | no-cot (TF CE) | debug/devloss_grid/grid.csv |
| `gold_fl20p8` | 17 tasks / 8k | 13/17; median −0.000; x0.37 | 16 ⚠ | paired | no-cot (TF CE) | grid.csv |
| `gold_fl20p8` | 15 tasks / 32k | 10/15; median 0.000; x0.32 | 8 ⚠ eval_size=8 | paired | no-cot (TF CE) | grid.csv |
| `gold_rand20p8_noslot` | 17 / 2k, 8k; 15 / 32k | 12/17, 12/17, 10/15; medians ≈0 (position choice barely matters once gold + ids kept) | 16/16/8 ⚠ | paired | no-cot (TF CE) | grid.csv |
| gold-blind `fl20` | 2k / 8k / 32k | 3/17, 5/17, 3/15; median +0.40 / +0.69 / +0.70 | 16/16/8 ⚠ | paired | no-cot (TF CE) | grid.csv |
| `gold_fl20p8` + skip odd layers (all attention + half GDN) | 2k / 8k / 32k | median extra ΔCE +0.003 / +0.001 / +0.015 (≈ free) | 16/16/8 ⚠ | paired | no-cot (TF CE) | grid.csv (results_skip) |
| `gold_fl20p8` + skipeven / skipgdn2 | 2k / 8k / 32k | +0.04 / +0.16 / +0.34 and +0.06 / +0.12 / +0.40 (GDN not skippable) | 16/16/8 ⚠ | paired | no-cot (TF CE) | grid.csv |

Misses for `gold_fl20p8`:
- absence and reorder, where every document is load-bearing;
- outlier_amzn;
- the 32k cells of textgroups, rerank, fiqa and obliq.

See `records/devloss-grid-underperformers.md` for the per-cell audit (verdict codes a-d).

## Next step
Nothing is queued on the grid itself. The follow-on question, whether a *learned* rule can beat the bar,
is the `learned_router` experiment. The CPT column's training follow-up is `softdetach`.

## Open issues / traps
- Gold-blind rules at x0.2-0.3 (fl20, rule20, grad20 oracle, attnrow20, rand20, k0) all cost +0.5 to +1.5 nats. On this suite, gold is the lever, not token selection.
- `gold_fl64` does not compress on short-document columns (compaction 0.94-1.00 on niah, strmatch, contradiction, absence, obliq, oolong), so its "parity" there is trivial.
- The niah 8-token prefix is the header ids, not the fragment layout. Every construction that hides neighbour ids fails.
- Wrong-checkpoint cells: xabsence uses the PARAPHRASE checkpoint (`memory:xabsence-exact-vs-paraphrase-split`), and obliq is retrieval-graded.
- Grid artifact: https://claude.ai/code/artifact/5f22d13d-544b-4907-b31d-6be6afb537af. The older f1 grid is artifact 7f8d198f (`research/token_routing/ds64/scheme_grid/`).

## Verified vs unverified
- Verified 2026-10-12 by recomputing from `debug/devloss_grid/grid.csv`, excluding xabsence and cpt80:
  - `gold_fl20p8` pass counts 13/13/10 and compaction 0.49/0.37/0.32. Memory said "13/18, 13/18, 10/16", which differs only in whether xabsence is counted.
  - `gold_rand20p8_noslot` 12/12/10, fl20, and the skip-layer medians.
- Corrected: memory said skipeven/skipgdn2 cost "+0.04-0.26". The CSV medians reach +0.34/+0.40 at 32k.
- UNVERIFIED: the cpt80-column screen ranking is not re-derived here (see softdetach).
