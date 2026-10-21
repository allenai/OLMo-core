---
experiment: setA_pareto
title: Dense vs bar (gold_fl20p8_noslot token drop) on the full 11-task setA mix — is bar Pareto-better in FLOPs and wall-clock on every task?
status: active
updated: 2026-10-20
branch: prasann/flopdrop_clean
launcher: research/token_routing/setA_pareto/launch_pareto.py
records:
  - records/flopdrop-final-figures-plan.md
results:
artifacts:
  - https://claude.ai/artifact/RFpSfNnqnz8M9ygi8kK1U5
wandb_group: setA-pareto
next_step: Ladders ON HOLD -- tokdrop_lr shows the gold-whole keep rule teaches a gold shortcut (bar contradiction 8k 0.07 vs dense 0.89 at every LR). Smokes (sAp-dense-smoke, sAp-bar-smoke) still pending on jupiter; let them run for the throughput check. Pick a shortcut-free global rule first, then size the ladder.
---

# setA_pareto

## Question
Prasann (2026-10-20): "a training run with the best heuristic on the setA data to see if we're
pareto optimal across tasks with just the bar heuristic", in both FLOPs and wall-clock, at every
budget, on all 11 setA tasks.

## Design
- Same trainer (`train_ctc_suite.py`), base (`q35-4b-dense-256k-fix/step2385`), data, optimizer and
  node shape (1×8 H100, jupiter, urgent, minRuntime 1h) for both arms; both PACK whole examples.
  - dense: 32,768-token windows × 32 per step (1,048,576 tokens; the canonical setA batch).
  - bar: 131,072-token windows × 8 per step (same original tokens = same examples per step). The
    model unpacks each window, applies gold-whole + 8-token prefix + first/last 20% per example,
    and re-packs the compacted examples (segment mode, commit a656a6e6e).
- Budgets: separate `--max-tokens` runs, each with its full schedule. Dense at ⅛, ¼, ½, 1 epoch
  of setA-32k (925M tokens); bar at matched FLOPs, which needs ~2.5–3× setA's rows → fresh-seed
  expansion (seeds 43, 44).
- FLOPs: `flop_meter/actual_pflops` with per-example accounting (each example charged at its own,
  for bar compacted, length). The old meter charged every token attention at `--seq-len`, which
  overstates dense packed attention ~7× on setA (examples average 4,555 tokens).
- Eval: olmo-eval `launch_ctc_suite.py --arm full --doc-markers`, 11 tasks × 2k–32k, 500 rows.

## Decisions
- 2026-10-20 (Prasann): "Yes, can you handle all of these fixes?" — stall fix, packing, gold
  sidecars, expansion.
- 2026-10-20 (agent): rows > 32,768 tokens are dropped (as in the canonical setA-32k arm).
- 2026-10-20 (agent): flash_2 for both arms (the soft path is validated on it).

## Runs
| run | what | job | state |
|---|---|---|---|
| setA-pareto-data-gold-s42(-r2) | setA s42 re-tokenized with gold sidecars, byte-checked, merged ≤32k → `setA32k_s42` | 01M5DQNYVDHSG5KGXGP6MXXBFK | running (first try compared against the wrong dir) |
| setA-pareto-data-expand-s43 / -s44 | fresh setA rows (2k–32k), sidecars, overlap vs s42, merge | 01M5DPBRWZ06Q8D7R91VA4S268 / 01M5DPC1SG7W7K109DP9RMADKQ | running |

## Current best numbers
None yet.

## Next step
See `next_step`.

## Open issues / traps
- Bar arm wall-clock: fast8k Lambda runs stalled 25–75 s on most steps from FLA re-autotuning at
  each new compacted width (measured: 22–51 s per unseen length on an A100). Fixed in b8e7c71f4
  (freeze NB key + `--st-pad-multiple 256`); not yet re-measured in a training run.
- Supply-bound pools (contradiction pairs, rerank, qdmatch_nq) cannot give fresh gold material at
  a new seed; `reports/overlap_s<seed>.json` says how much of each expansion repeats s42.
- The canonical setA-32k checkpoint (step888) used a different trainer (BFD-packed setA recipe,
  flash_3); it is a reference point, not a ladder point.

## Verified vs unverified
- Verified 2026-10-20: packed segment-mode loss == padded per-example loss on a GDN hybrid
  (src/test/nn/pooled_soft_token_packed_test.py, H200); 31 soft-token tests pass.
- UNVERIFIED: bar throughput on packed 128k windows at 4B; the bar/dense FLOP ratio on setA.
