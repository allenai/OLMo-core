---
title: Plan — empirical token-dropping CPT screen on OLMo 3 7B (stage 2 → 32k, longmino pool)
date: 2026-10-21
updated: 2026-10-21
experiment: olmo3_cptdrop
area: token_routing
kind: plan
status: current
superseded_by:
artifact: https://claude.ai/artifact/4tZKEXtpSpg5ax6B1YeoLk
summary: Short trained screen from the OLMo 3 7B stage-2 checkpoint at 32k on the Dolma 3 Longmino pool — dense vs length curriculum vs two input-level token-drop rules, scored by full-context dev CE at three matched FLOP budgets (~73 H200-h).
---
# Plan — empirical token-dropping CPT screen on OLMo 3 7B

Agreed with Prasann on 2026-10-21. State lives in `ops/experiments/olmo3_cptdrop/STATUS.md`; code in
`research/token_routing/olmo3_cptdrop/`.

## Why this design

- Every CPT token-drop result so far (softdetach, dropstruct, ffndrop) used a Qwen3.5-4B base that had
  already been through long-context training, so dense was near-saturated. softdetach's `sd20` won about
  4× at small budgets and then lost to dense near 2k PF; the deficit was in long-range positions and grew
  with budget (`records/softdetach-cpt-crossover.md`).
- OLMo 3 7B publishes every stage. `stage2-step47684` is midtrained but has only seen 8k contexts: the
  exact state OLMo's own long-context stage started from. That makes the long-range regime the thing being
  learned, which is where dropping has failed before.
- Prasann dropped the frozen dev-loss screen (2026-10-21): frozen-model rankings may not track training
  value (softdetach: `rand20` ranked worst frozen, best trained). This plan selects on short trained runs only.

## Setup (identical for every arm)

| item | value |
|---|---|
| base | `allenai/Olmo-3-1025-7B@stage2-step47684`, converted to OLMo-core (`convert_base.sbatch`, converter's own HF-parity validation) |
| context | 32,768. YaRN ×4 (8,192 → 32,768) on the full-attention layers only, from step 0. 64k is the follow-up. |
| data | Dolma 3 Longmino pool (`allenai/dolma3_longmino_pool`), 2^15 / 2^16 buckets, synthetic folders excluded. Row = EOS + one 32,767-token window of one document, ≤2 windows per document, globally shuffled. No marker tokens (blocks are positional). ~43k rows = 1.41B tokens (`build_pool_data.py`) |
| batch | 16 rows × 32k = 512k tokens/step |
| optimizer (dense) | OLMo 3 stage-3 recipe as is: peak LR 2.0712e-4, linear decay to 0, AdamW β (0.9, 0.95), wd 0.1 (none on embeddings), z-loss 1e-5, clip 1.0. Warmup scaled to the same fraction of the run (200/11,921 ≈ 1.7%) |
| optimizer (drop arms) | same LR by default (option A). Option B (1e-4 on the better drop arm, ~20 H200-h) only if a drop arm is unstable or lands just short of dense |
| budgets | 60M / 125M / 250M dense-token equivalents (≈3.8k / 7.9k / 15.7k PF at ~5.3e10 FLOPs/token), from one warmup-stable run per arm with LR-decay branches |
| dev | 256 rows: first 32k tokens of documents from the 2^17+ buckets (never in the 7B stage-3 mix), round-robin over the 22 topics. Full-context CE (no dropping at eval), binned 0–4k / 4–8k / 8–16k / 16–32k |
| guard | 256 × 4k rows of short documents from one stage-3 shard (`dolma3_longmino_0625/000500`), to catch forgetting |
| FLOPs | `flop_meter/actual_pflops`; GPU-hours logged next to every point |

## Arms

| arm | strategy | FLOPs/token vs dense |
|---|---|---|
| D | dense, 32k | 1.0 |
| C | 8k windows for the first half of the run, then 32k; one run per budget (Prasann 2026-10-21), the three in parallel | ~0.9 |
| I-rand | input drop: random 20% of 512-token blocks kept whole, the rest removed (no pooled slot) | ~0.2 |
| I-fl | input drop: first+last 25% of every 512-token block kept (first 64 + last 64) | ~0.25 |

Token exit and layer skipping are out of scope for this round (Prasann, 2026-10-21). Drop arms use the same
16 rows/step, so they take 3–5× more (noisier) steps for the same FLOPs.

## Selection rule (fixed in advance)

m = (FLOPs the dense curve needs to reach the arm's dev CE, log-linear interpolation) ÷ (arm's FLOPs).
An arm wins only if m > 1 at all three budgets, beyond the dense seed-to-seed noise. The 16–32k bin says
whether a win is long-range or short-range only. The short-document guard must not regress beyond noise.

## Steps

1. Experiment page, this record, the artifact.
2. Stage-2 (and stage-3 reference) checkpoints converted on a node's /data, validated against HF.
3. Data build (train32k / dev32k / short4k), then upload to `s3://ai2-llm/checkpoints/prasanns/_transfer/olmo3_cptdrop/`
   as the hub for Berkeley nodes, weka (Beaker) and Lambda node /tmp.
4. Sliding-window mask by ORIGINAL position for compacted sequences (the 24 SWA layers otherwise reach
   ~4× further back after dropping), with a keep-all parity check.
5. Positional 512-token blocks in the input-drop path (today it finds blocks from marker tokens).
6. OLMo 3 dev-CE eval by position bin; fails on an unknown model family. Sanity: stage 3 beats stage 2 at 8–32k.
7. Trainer: YaRN factor option (the CTC factory hard-codes ×8), LR-decay branches, arm C's 8k → 32k switch,
   FLOP-meter check.
8. Smoke test of all four arms (berkeleynlp, `--qos=normal`, debug only).
9. Dense curve at the recipe LR (one run to 250M + decay branches) + a second seed at 60M; dense must beat
   stage 2 on dev (a check, not tuning).
10. C, I-rand, I-fl to 15.7k PF with decay branches at all three budgets.
11. Score, `ops result add --check`, compute m, verdict into STATUS and the artifact.

After round 1: a winning drop arm gets its keep fraction (0.1 / 0.5) and a switch-to-dense-at-50% run
(~60 H200-h), then 64k. If nothing beats dense, the position bins decide whether token exit is next.

## Compute

~73 H200-h through step 11 (dense curve + seed ~25, three arms ~55 minus the dropped LR pilot). Venues:
jsteinhardt H200 (single 8-GPU node), Beaker jupiter (urgent, minRuntime 1h, ai2/flex2), Lambda A100
(within the 8 preemptive_high + 8 preemptive fair-share cap; Prasann OK'd Lambda 2026-10-21). At 32k a 7B
fits 8 GPUs with full activation checkpointing, no context parallelism.

## Known traps to watch

- Compacted sequences and sliding windows (step 4): without the fix every drop number on OLMo 3 is wrong.
- `throughput/total petaflops` is ~10× wrong on compacted arms; use `actual_pflops`.
- `--max-tokens` below one step's tokens silently runs one step.
- Branching decay runs into a new save folder can silently auto-resume (`memory:trainer-silent-autoresume-trap`).
- The pool's length-bucket labels are approximate in the OLMo 3 tokenizer (a "32k–64k" stage-3 shard held
  documents up to 133k tokens), so the builder measures token lengths itself.
- The short-document guard comes from stage-3 training data; stage 2 may have seen similar midtraining text.
  It is a relative check across arms, not an absolute one.
