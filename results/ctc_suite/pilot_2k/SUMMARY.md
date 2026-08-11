# CTC-suite Stage 3 — 2k full-attention validation sweep

**Gate:** does *full attention* learn each task at short context (2k rung)? Measured by train CE
dropping; eval f1 reported where an eval set is staged. Model: Qwen3.5-0.8B, 1 epoch.
**Result: 27 tasks, 26 PASS / 1 FAIL.** `contra_cmix` excluded (mask-mix arm, not full-attention →
deferred to Stage 4, per decision 2026-07-19).

Pools: Berkeley H200 (jsteinhardt cubbins/mooney) + lambda A100 (6 tasks). All runs have a full
per-step loss curve in `curves/<task>.json`.

## Table (worst-first)

| task | pass | CE start→end | eval | metric | eval_size | pool |
|---|---|---|---|---|---|---|
| **groups4** | **FAIL** | 1.39→0.863 | **0.000** | cycle_f1 | 500 | mooney-1gpu |
| mathmatch | ⚠pass | 1.21→0.793 | 0.002 | set_f1 | 500 | H200-1gpu |
| strmatch | ⚠pass | 1.15→0.732 | 0.005 | set_f1 | 500 | H200-1gpu |
| textgroups | ⚠pass | 0.82→0.401 | 0.015 | textgroups_f1 | 500 | H200-1gpu |
| qdmatch_nq | ⚠pass | 1.07→0.474 | 0.037 | pair_f1 | 500 | H200-1gpu |
| nq | ⚠pass | 1.77→0.465 | 0.074 | gold_id_f1 | 500 | H200-1gpu |
| xabsence | pass | 0.85→0.570 | 0.079 | set_f1 | ⚠300 | H200-1gpu |
| helmet_qa | pass | 3.78→3.734 | 0.125 | token_f1 | 500 | H200-1gpu |
| outlier_amzn | pass | 1.02→0.450 | 0.202 | set_f1 | 500 | H200-1gpu |
| outlier | pass | 1.64→0.130 | 0.217 | set_f1 | 500 | H200-8gpu |
| contradiction | pass | 1.70→0.007 | 0.684 | set_f1 | 500 | cubbins |
| cycle | pass | 1.42→0.050 | 0.786 | cycle_f1 | 500 | mooney-1gpu |
| hotpotqa | pass | 1.81→0.148 | 0.861 | gold_id_f1 | 500 | H200-8gpu |
| qdmatch_hpqa | pass | 0.82→0.002 | 0.985 | pair_f1 | 500 | H200-1gpu |
| niah | pass | 1.44→0.002 | 0.988 | gold_id_f1 | 500 | H200-1gpu |
| helmet_summ | pass | 2.30→1.722 | — | — | — | H200-1gpu |
| reorder | pass | 1.05→0.897 | — | — | — | H200-8gpu |
| rerank | pass | 1.17→0.781 | — | — | — | H200-8gpu |
| oolong | pass | 1.90→0.521 | — | — | — | H200-8gpu |
| grouping | pass | 0.74→0.504 | — | — | — | H200-8gpu |
| grouping_labeled | pass | 0.77→0.285 | — | — | — | mooney-1gpu |
| obliq_retrieval | pass | 1.01→0.321 | — | — | — | lambda-1gpu |
| msmarco_rerank | pass | 0.76→0.301 | — | — | — | lambda-1gpu |
| absence_gutenberg | pass | 1.69→0.113 | — | — | — | lambda-1gpu |
| msmarco | pass | 1.78→0.099 | — | — | — | lambda-1gpu |
| fiqa | pass | 0.72→0.062 | — | — | — | lambda-1gpu |
| scifact | pass | 1.55→0.021 | — | — | — | lambda-1gpu |

## Read / caveats (do NOT scale blindly on "26 PASS")

The pass gate is *training* (CE drop). Full attention learns to fit essentially every task at 2k.
The **eval** column tells a different, more important story:

- **groups4 — FAIL, real.** CE dropped but eval=0.000: the model emits an identical constant
  prediction for all 500 examples (mode collapse under the 1-epoch / 2500-example budget). Not the
  maxlen-truncation trap. Needs more data/epochs or LR before it's usable.
- **NQ-family ≈0 evals — DIAGNOSED (2026-07-19): mode collapse, not an eval/data bug.** Inspected
  generations: **nq** emits only 2 distinct IDs over 500 examples (`[12]`×402, `[2]`×98) — and 12 is
  NOT a frequent gold, so it's a degenerate positional attractor, not even the label prior. f1 0.074
  ≈ chance. **qdmatch_nq** likewise outputs a near-constant pair-set (`[[1,11],[2,17],[4,12]]`×204 + tiny
  permutations). Harness + gold labels are correct. Root cause = the NQ-sourced tasks use p10 hard
  (cross-encoder-filtered) negatives that are too hard to discriminate in 1 epoch/2500 ex, so the model
  learns the FORMAT and emits a constant. Decisive tell: **qdmatch_hpqa 0.985 vs qdmatch_nq 0.037 —
  same task, only the source (easier HotpotQA negatives) differs.** Same collapse class as groups4;
  mathmatch/strmatch/textgroups (also ≈0) are likely the same. FIX = more epochs/data or easy→hard
  curriculum for the hard tasks; nothing to fix in eval/data.
- **helmet_qa 0.125** — low but task-appropriate (free-form NarrativeQA generation; its flat CE was
  expected, CE gate uninformative for generative tasks).
- **⚠ xabsence eval_size=300** (<500) — smaller eval, wider error bar; do not over-read 0.079.
- **11 tasks are train-only (no eval f1 yet):** the 6 lambda tasks (eval not wired on the air-gapped
  cluster) + reorder/rerank/oolong/grouping/helmet_summ (no eval_rungs staged / eval outran walls).
  They validate that full attention *learns*, but their eval story is still open.

## Infra bugs found + fixed mid-sweep (reusable for Stage 4/5)

1. **jsteinhardt 8-GPU/user cap** serialized full-scale (NGPU=8) jobs → dropped to NGPU=1 for
   concurrency (slower per task, but 8 run at once).
2. **Eval node-pinning bug:** eval submitted with `-w cubbins,mooney` could land on the node that
   didn't train the ckpt (node-local `/data`) → silent FileNotFoundError. Fixed by pinning eval to
   the training host parsed from the log banner.
3. **Eval-namespace collision:** many shard-tasks share one evaluator task (niah/nq/msmarco/… →
   `retrieval`); added per-shard `--out-root` isolation.
4. **lambda seq_len sizing:** max example length varies per task (msmarco 2070, fiqa 3660, absence
   4082) → seq_len must be ≥ task max; padding to 4096 on A100 timed out a 40-min wall (use ≥90 min).
   Re-run with a **fresh run-name** to avoid the trainer auto-resume/fingerprint crash.
5. **lambda NFS quota** hit 100% → blocked staging; freed 230G, came online.
