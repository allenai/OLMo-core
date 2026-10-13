---
experiment: softdetach
title: Soft-detached pooling for CPT (sd20 = random 20% of 512-token blocks whole, rest one detached slot) vs dense CPT, Qwen3.5-4B
status: done
updated: 2026-09-22
branch: prasann/landmark
launcher: src/scripts/train/memexpress/cpt/softdetach/launch_softdetach_cpt.py
records:
  - records/softdetach-cpt-plan.md
  - records/softdetach-cpt-crossover.md
  - records/eval_launches/2026-09-21_sdcpt-q35-4b_ctcbench.yaml
results:
  - src/scripts/train/memexpress/cpt/softdetach/softdetach_devloss.csv
  - src/scripts/train/memexpress/cpt/softdetach/ctc_eval_results.json
  - src/scripts/train/memexpress/cpt/softdetach/LAUNCH_LEDGER.tsv
  - src/scripts/train/memexpress/cpt/softdetach/EVAL_LEDGER.tsv
wandb_group: https://wandb.ai/prasanns-allen-institute-for-ai/memory-networks/groups/sdcpt-q35-4b
next_step: None queued. Verdict reached (cheap-end only); the drop-CPT experiment (ffndrop) reuses its cpt_u1B shard and dev-loss evaluator
---

# softdetach

## Question
Does CPT on soft-detached pooled contexts (most 512-token blocks collapsed to one detached, untrained slot)
reach dense CPT's dev loss with fewer training FLOPs? Does that advantage persist as the FLOP budget grows?

## Decisions
- 2026-09-21 (agent): screened ≥4x-compaction rules on the frozen-base cpt80 column of the dev-loss grid, then trained dense plus three soft arms (sd20, sfl20, lslot20) at 16M-128M tokens.
- 2026-09-21 (agent): the frozen screen ranked rand20 WORST, but trained sd20 WON. Pick training arms with a short trained ladder, not the frozen screen.
- 2026-09-21 (Prasann): requested dense 4M/8M/16M anchors so that cheap-end claims are measured, not extrapolated (`memory:ds64-scaling-campaign` lesson).
- 2026-09-21 (Prasann): requested a downstream raw-CPT CTC bench (contra/nq/oolong, eval_size 500/rung) to check that downstream ranks like dev loss.
- 2026-09-22 (agent): extended sd20 to 256M/512M/1B on the 1B shard and found the crossover. Diagnosed it (normalisation ruled out; regime trade). Two fixes, sd20mix and sd20p32, both land ON the dense curve. Verdict: cheap-end technique only.

## Current best numbers
Dev = 32 held-out 64k rows (`cpt_dev`, part 28), full-attention dev CE (nats/token).
PF = `flop_meter/actual_pflops`. ⚠ eval_size=32 rows (~2M tokens); paired per-row SEs 0.001-0.007.

| arm | task / rung | metric | eval_size | ±SE | cot | source |
|---|---|---|---|---|---|---|
| sd20-64M (383 PF) vs dense-32M (1518 PF) | cpt_dev / 64k | CE 1.279 vs 1.283: parity at 0.25x FLOPs (paired Δ −0.004) | 32 rows ⚠ eval_size=32 | 0.004 | no-cot (TF CE) | softdetach_devloss.csv + records/softdetach-cpt-plan.md |
| sd20 vs dense at equal PF (193/383/763) | cpt_dev | 1.291/1.279/1.272 vs 1.314/1.303/1.296 (≈ −0.024 each) | 32 ⚠ | ≤0.005 | no-cot (TF CE) | softdetach_devloss.csv |
| sd20 high-FLOP (1523/3043/5942 PF) | cpt_dev | 1.2725/1.2634/1.2585 vs dense interp. → −0.010 / +0.007 / +0.013 (crossover ~1.5k-3k PF) | 32 ⚠ | unpaired interp. | no-cot (TF CE) | records/softdetach-cpt-crossover.md |
| sd20p32 256M/512M (2076/4148 PF) | cpt_dev | 1.2655 / 1.2575 → −0.005 / +0.007 vs dense interp. (on curve) | 32 ⚠ | seed ±0.004 | no-cot (TF CE) | records/softdetach-cpt-crossover.md |
| sd20mix 256M/512M (1945/5122 PF) | cpt_dev | 1.2668 / 1.2527 → −0.006 / +0.005 (on curve) | 32 ⚠ | seed ±0.004 | no-cot (TF CE) | records/softdetach-cpt-crossover.md |
| sd20 own construction | cpt_dev | own CE 1.50 vs full 1.27: saving is TRAINING-only | 32 ⚠ | — | no-cot (TF CE) | softdetach_devloss.csv |
| raw-CPT downstream, nq 3k | sd20-64M / dense-32M / dense-64M | f1 0.730 / 0.484 / 0.642 | 500 | 0.020 / 0.022 / 0.021 | no-cot (raw prompt) | ctc_eval_results.json |

## Next step
None queued. The CPT verdict is closed: a 4x FLOP saving up to roughly where dense saturates short
context (~1-2k PF, ~32M dense tokens on this 4B), and not a dense replacement at scale. The open tail is the
downstream oolong bench, which never landed (below).

## Open issues / traps
- Shard switch: the ≤128M points used `cpt_u128M` (parts 0-7) and the ≥256M points used `cpt_u1B` (parts 0-27). The flat step at 128M→256M was a SHARD artefact, but the crossover survives on one shard (12.8σ at 6k PF).
- The sd20p32 "own" CE is scored under plain sd20 compaction (`header_extra_tokens: 0` is hard-coded in the eval), so only its full-attention CE is meaningful.
- Raw-CPT downstream is noisy and format-limited:
  - contradiction is 0.000-0.007 for EVERY arm because of the raw prompt format;
  - all oolong jobs and all base-model jobs FAILED or never landed (null in `ctc_eval_results.json`);
  - dense nq is non-monotone in budget (16M 0.734 > 64M 0.642 > 128M 0.374);
  - do not read the ranking as more than weak support.
- The CPT fork died on a session rate limit before writing the high-FLOP and downstream results into the plan record. They live in the crossover record and the JSON.
- Learned-slot-only (lslot20, frozen backbone) never moves the backbone: its full CE stays at the base's 1.320.

## Verified vs unverified
- Verified 2026-10-12 against `softdetach_devloss.csv`: all ≤128M dense, sd20, sfl20 and lslot20 CE and PF values. The run and eval Beaker ids are in the CSV.
- Verified against `ctc_eval_results.json`, at eval_size 500/rung:
  - nq numbers for dense 16M/32M/64M/128M, sd20 64M/128M and sfl20 64M;
  - contra ≈0 for all arms;
  - oolong and base entries are null or FAILED.
- Not re-derived from a primary file: the 256M-1B sd20, sd20mix and sd20p32 CE values are only in `records/softdetach-cpt-crossover.md`. The weka devloss JSONs are not accessible from here.
- Conflict: memory devloss-grid-task (2026-09-22) says downstream "ranking agrees with dev loss at the cheap end". The JSON shows dense nq non-monotone in budget, so that agreement is weak.
- UNVERIFIED — confirm with Prasann: are the oolong raw-CPT evals still wanted?
