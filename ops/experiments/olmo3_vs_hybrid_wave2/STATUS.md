---
experiment: olmo3_vs_hybrid_wave2
title: OLMo-3-7B (noswa-CPT, full attention) vs Olmo-Hybrid-7B (3:1 GDN:full) — dense CTC-suite backbone comparison
status: done
updated: 2026-08-31
branch: prasann/landmark
launcher: debug/ctc_olmo_hybrid/launch_wave2_runs.sh
records:
  - records/ctc-final-suite.md
results:
  - results/ctc_suite/olmo3_vs_hybrid.md
  - results/ctc_suite/_crossfamily_pull/
  - debug/ctc_olmo_hybrid/LAUNCH_LEDGER.tsv
  - debug/ctc_olmo_hybrid/hybrid_vllm/results/
  - paperdraft/figures/data/ctc_hybrid_pertask_data.csv
wandb_group:
next_step: None required for the verdict. Optional — re-run the oolong full arm with a stop condition and natively re-grade the hotpotqa 32k full arm before quoting either cell as final
---

# olmo3_vs_hybrid_wave2

## Question
With data, shards, sequence length, LR and steps held fixed, does replacing 3 of every 4 attention
layers with Gated-DeltaNet (Olmo-Hybrid-7B vs OLMo-3-7B) hurt high-CTC tasks more than low-CTC tasks?

## Decisions
- 2026-08-17 (Prasann): roster swap. rerank, msmarco and textgroups are cancelled mid-queue in favour of qdmatch_fiqa and xabsence. The final set is nq, oolong, hotpotqa, qdmatch_hpqa (low/parity block) and outlier, qdmatch_nq, qdmatch_fiqa, xabsence, contradiction (high). Dense/full arms only.
- 2026-08-17 (Prasann): the protocol extends to the full 2k-32k ladder. Training uses a 10k subsample capped at 33k est. tokens, seq 40960, lr 5e-5, gbatch 8, 1 epoch.
- 2026-08-17/18 (Prasann): the OLMo-3 arm is `7b-noswa` (SWA disabled), for a minimal full-vs-hybrid attention comparison. It is repaired by a ~500-step / 130M-token long-text CPT (`olmo3ns-7b-cptlong32k`), because raw noswa plateaus.
- 2026-08-18 (Prasann): vLLM for every eval of full arms. Hybrid stays native (no vLLM mapping at the time). Beaker runs hybrid arms only.
- 2026-08-18 (agent): every outlier number graded before 2026-08-18 11:40 used a stale pre-fix "shrunk" ladder and is VOID (quarantined `*.STALE-shrunk-ladder`). Re-run on the fixed v2 ladder.
- 2026-08-19 (agent): the oolong hybrid "2-4x collapse" was an eval artifact (no stop condition, 93-97% repetition loops). It was regraded on the first line and moved to parity.
- 2026-08-20 (agent): a 10-task hybrid column is added (`outlier_amzn` pair etc., ledger). The 32k-protocol contradiction noswa arm was CANCELLED by the user on 2026-08-21, so the contradiction pair stays at the 16k protocol.

## Current best numbers
All cells are eval_size 500. Full arms are vLLM-graded; hybrid arms are native-graded. Values are 32k unless noted.

| arm | task / rung | metric | eval_size | ±SE | cot | source |
|---|---|---|---|---|---|---|
| full / hybrid | qdmatch_nq / 32k (2k) | pair-F1 0.598 / 0.000 (2k: 0.991 / 0.032) | 500 | ±0.022 / 0 | unknown | results/ctc_suite/olmo3_vs_hybrid.md |
| full / hybrid | xabsence / 32k | set-F1 0.519 / 0.003 | 500 | ±0.022 / ±0.002 | unknown | same |
| full / hybrid | outlier (fixed v2) / 32k | set-F1 0.392 / 0.210 | 500 | ±0.022 / ±0.018 | unknown | same |
| full / hybrid | contradiction_iid / 16k | set-F1 0.919 / 0.830 | 500 | ±0.012 / ±0.017 | unknown | same |
| full / hybrid | nq / 32k | gold-ID F1 0.888 / 0.900 | 500 | ±0.014 / ±0.013 | unknown | same |
| full / hybrid | hotpotqa / 32k | gold-ID F1 0.846 / 0.931 (inversion, flagged) | 500 | ±0.016 / ±0.011 | unknown | same |
| full / hybrid (regraded) | oolong / 32k | partial credit 0.618 / 0.594 | 500 | ~±0.02 | unknown | same |

Verdict: the hybrid collapses or trails on every high-CTC pair that has headroom. qdmatch_nq never
learns; xabsence is about 5x worse; the outlier gap widens from +0.03 to +0.18 with length. The hybrid
holds parity on retrieval and oolong. qdmatch_hpqa sits at ceiling for both arms.

## Next step
Nothing is queued. If these cells go into the paper as final:
1. Re-run the oolong full arm with a stop condition. The full arm saved no generations, so the row is cleaned-native vs as-run-vLLM.
2. Natively re-grade the hotpotqa 32k full arm (0.846 < hybrid 0.931, unexplained).

## Open issues / traps
- The native evaluator has no stop condition, so it grades repetition loops (memory:native-eval-repetition-loop-bug). Watch the loop rate per arm. outlier loops at 80-94% but parses the first set, so it moves ≤0.001.
- oolong's loss covers ~3.6 answer tokens per example. Per-step CE cannot resolve small gaps.
- The SWA-kept `olmo3-7b` (fixmark base) and `olmo3-7b-noswa-cpt` are different models. Never mix them in one column. qdmatch_hpqa's full arm is SWA-kept.
- GDN `o_norm` eps: training used 1e-6, while HF/vLLM serve at 1e-5. Aggregates agree within 1 SE, but per-example analyses across backends are invalid (memory:gdn-onorm-eps-train-eval-match).
- The results-hub local clone was left diverged (`wave2-local-backup`, `wave2-ingest`). The memory says "Prasann to reconcile". The current state was not checked.

## Verified vs unverified
- Verified 2026-10-12: the table against `results/ctc_suite/olmo3_vs_hybrid.md` (2026-08-19); the ledger tail (last row 2026-08-21, a cancellation); the paper figure CTC-HYBRID-ALL reading `paperdraft/figures/data/ctc_hybrid_pertask_data.csv` (2026-08-31); git `a539b8da7` (2026-08-21, hybrid vLLM exporter and eps fix).
- Status conflict resolved: the CLAUDE.md memory index line says "ACTIVE", but the memory file itself (2026-08-19) and the results doc both say COMPLETE. There is no new ledger row after 2026-08-21. Marked `done`.
- UNVERIFIED — confirm with Prasann: whether the results-hub divergence (bf79101 / local backup branches) was ever reconciled. Also whether the 10-task hybrid column added 2026-08-20 is complete. Its numbers are in `ctc_hybrid_pertask_data.csv`; they were not cross-checked cell by cell.
