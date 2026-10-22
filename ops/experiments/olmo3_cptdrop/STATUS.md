---
experiment: olmo3_cptdrop
title: Token-dropping CPT screen on OLMo 3 7B — stage2 → 32k on the Dolma 3 Longmino pool; dense vs length curriculum vs input-drop (random-20% blocks, first+last 25%), full-context dev CE at 3 matched FLOP budgets
status: active
updated: 2026-10-21
branch: prasann/flopdrop_clean (uncommitted: research/token_routing/olmo3_cptdrop/)
launcher: research/token_routing/olmo3_cptdrop/
records:
  - records/olmo3-cptdrop-screen-plan.md
results:
artifacts:
  - https://claude.ai/artifact/4tZKEXtpSpg5ax6B1YeoLk
wandb_group:
next_step: Merge o3cd/swa-position-window (eae16ae0b) + o3cd/positional-blocks (2a0d30ef8) + o3cd/devce-eval (a151f921d) with the uncommitted trainer flags (Prasann OK pending); rebuild short4k from several stage-3 files; decide dev size (256 rows: paired SE ~0.006 per bin vs 0.046 total stage2->stage3 headroom at 16-32k); then smoke-test all four arms.
---

# olmo3_cptdrop

## Question
Starting from a base that has never seen long context (OLMo 3 7B stage 2, 8k), does input-level token
dropping during long-context CPT beat dense at every matched FLOP budget — not only at the cheap end, where
softdetach won on an already-long-trained Qwen3.5-4B before dense overtook it near 2k PF?

## Decisions
- 2026-10-20 (Prasann): CPT from a raw OLMo 3 checkpoint, longmino pool as data.
- 2026-10-21 (Prasann): no frozen dev-loss screen (may not correlate with training value); an empirical
  screen of strategies on short training runs instead.
- 2026-10-21 (Prasann): arms D (dense), C (8k → 32k curriculum), I-rand, I-fl only; token exit and layer
  skipping deferred. Keep the H200-hour count low (plan ≈73 H200-h).
- 2026-10-21 (Prasann): 32k first, 64k later.
- 2026-10-21 (Prasann): dense uses the OLMo 3 stage-3 optimizer recipe as is (peak LR 2.0712e-4); drop arms
  may use a different LR. Agent default: same LR for drop arms, warmup scaled to 1.7% of the run.
- 2026-10-21 (Prasann): Beaker and Lambda are both allowed where useful. Agent: S3 is the hub
  (`s3://ai2-llm/checkpoints/prasanns/_transfer/olmo3_cptdrop/`).
- 2026-10-21 (agent): start = `stage2-step47684` (the state OLMo's own long-context stage started from);
  stage3-step11921 converted too, as a reference for the eval sanity check. Blocks are positional (no
  marker tokens, so no marker-embedding repair). Dev from the 2^17+ buckets (not in the 7B stage-3 mix).
- 2026-10-21 (agent): build node mooney — cubbins is stuck in `completing` (other users' jobs hung in epilog).
- 2026-10-21 (agent): trainer flags added to `recipes/train/research_trainer/train_ctc_suite.py` (uncommitted):
  `--weight-decay`, `--z-loss`, `--warmup-fraction`, `--scheduler {linear,wsd}`, `--wsd-decay-fraction`,
  `--branch-from` (resume model+optim+trainer state = decay branch); `--rope-yarn-factor` is passed INTO
  `olmo3_7B_ctc_swa(yarn_factor=...)` (olmo3_configs.py) instead of stacking a second with_rope_scaling.
  Verified on the built config: x4 on the 8 full-attention layers (3,7,...,31), SWA layers plain RoPE,
  default unchanged at x8. Budgets as steps 120 / 240 / 480 (16 rows x 32k), `--save-interval 108`
  gives the 0.9 x budget branch points 108 / 216 / 432.
- 2026-10-21 (Prasann): arm C runs separately per budget (60M / 125M / 250M), the three in parallel. Recipe
  per budget B steps (512k tokens/step both phases): phase 1 = `--data data/c8k --seq-len 8192
  --global-batch 64 --scheduler wsd --warmup-fraction 0.017 --max-tokens B*524288 --max-steps B/2`
  (stable LR, full checkpoint at the hard stop); phase 2 = `--data data/train32k --seq-len 32768
  --global-batch 16 --branch-from <phase1>/step{B/2} --branch-optim-only --scheduler wsd --warmup-fraction 0
  --wsd-decay-fraction 0.2 --max-tokens (B/2)*524288` (decay = last 10% of B). `data/c8k` = train32k rows
  39,160-42,999 cut into 15,360 x 8,192 (`build_c8k.py`); phase 2 reads from row 0, so phases share no text.
- 2026-10-21 (session positional-512-token-blo, commit 2a0d30ef8 on branch claude/focused-lovelace-abac0f, not
  merged): positional blocks for the input-drop path. Arm flags: I-rand = `--variant softtoken --st-block-size 512
  --st-gold-blind --st-drop-slots --st-keep-prob 0.2` (kept 0.2007, sd 0.048 per row); I-fl = same with
  `--st-keep-prob 0.0 --st-keep-token-rule first_last --st-keep-token-k 0.25` (exactly 0.25: first 64 + last 64
  per 512-block). Also fixed: random block draw identical on every positional row (now hashed with row tokens),
  and the per-example FLOP meter charging every EOS-led row as length 1 (affected dense too). Adds
  `flop_meter/kept_label_frac`; logged train CE on drop arms = true CE x kept fraction (divisor counts labels
  before dropping). Pass `--skip-marker-audit` (shards have no markers).
- 2026-10-21 (agent): converter's built-in validation is skipped in favour of `check_conversion.py`
  (fp32 + bf16 parity incl. a 6,144-token row so the sliding window is exercised).

## Runs
| what | id | state |
|---|---|---|
| builder smoke (200 train / 22 dev / 16 short rows) | slurm 4012901 + 4012903 (mooney) | PASSED after a User-Agent fix for olmo-data.org |
| data build train32k / dev32k / short4k | slurm 4012904 (mooney) | DONE 19.5 min: train 43,000 x 32,768 = 1.409B tokens (32,382 docs); dev 256 rows (23 topics, 0 doc overlap with train); short 256 x 4,096 |
| convert stage2 HF -> OLMo-core | slurm 4012907 (mooney) | checkpoint SAVED; converter's bf16 self-check "failed" (atol 1e-4 vs bf16 noise). fp32 parity (4012938) PASS: max logit diff 4e-5 (random 512) / 1.4e-3 (real 6,144, SWA exercised), CE identical; bf16 CE diff 3e-4. 4012906 failed earlier: converter needs a local HF dir |
| convert stage3 + parity | slurm 4012946 (mooney) | DONE: fp32 parity PASS (max logit diff 4e-5 random / 4e-4 real 6,144, CE identical; bf16 CE diff 3e-5) |
| upload c8k to S3 | slurm 4012971 (mooney) | DONE |
| upload data + both bases to S3 hub (`s3_upload.sbatch`) | slurm 4012955 (mooney) | DONE 5.5 min: data/{train32k,dev32k,short4k,c8k} + base/{stage2-step47684,stage3-step11921} at s3://ai2-llm/checkpoints/prasanns/_transfer/olmo3_cptdrop/ (4012954 failed at once: `--wrap` runs sh) |
| dev-CE sanity (`eval_devce.py`, branch `claude/vigorous-agnesi-20ecb0` @ a151f921d): stage2 none/x4, stage3 x8 on dev32k + short4k | slurm 4012959 (mooney, 1 H200) | DONE: 6 CPU tests pass; per checkpoint load ~30 s + dev32k 372 s (1.45 s/row) + short4k 40 s; JSONs `/data/prasann/olmo3_cptdrop/devce/` (mooney) |

## Current best numbers
| arm | task / rung | metric | eval_size | ±SE | cot | source |
|---|---|---|---|---|---|---|
| stage3 x8 (reference) | devce_dev32k 16k-32k | CE | 256 rows ⚠<500 (4.19M tok) | Δ vs stage2-x4 −0.0455 ±0.0063 paired | no-cot | /data/prasann/olmo3_cptdrop/devce/stage3-yarn8_dev32k.json |
| stage2 x4 (CPT start) | devce_dev32k overall / 0-4k / 4-8k / 8-16k / 16-32k | CE | 256 rows ⚠<500 | — | no-cot | 1.9307 / 1.9698 / 1.9489 / 1.9289 / 1.9173 (stage2-yarn4_dev32k.json) |
| stage3 x8 | same bins | CE | 256 rows ⚠<500 | Δ −0.026±.006 / −0.004±.006 / +0.016±.010 / −0.018±.007 / −0.046±.006 | no-cot | 1.9051 / 1.9661 / 1.9653 / 1.9113 / 1.8718 |
| stage2 no YaRN | same bins | CE | 256 rows ⚠<500 | vs x4: −0.014 / −0.019 at 0-8k, +0.151 / +0.359 at 8-32k | no-cot | 2.1437 / 1.9555 / 1.9297 / 2.0798 / 2.2763 |
| short4k (synthetic math, 1 file) | devce_short4k | CE | 256 rows ⚠<500 | stage3 vs s2x4 +0.0230 ±0.0009 | no-cot | s2 none 0.4643, s2 x4 0.4718, s3 x8 0.4948 |

## Next step
See `next_step` above.

## Open issues / traps
- FIXED in eae16ae0b (branch o3cd/swa-position-window, not merged): compacted rows used to apply the 4,096
  sliding window in compacted index space. SWA layers with gapped position_ids now use a FlexAttention mask by
  ORIGINAL position (built once per forward, ~1 ms at 32k); full-attention layers and contiguous rows stay on
  flash. Keep-all parity <=1e-5 fp32 (bf16 flash bit-identical); leak test; dense-reference outputs + grads;
  361 existing attention/kv_route/soft-token/block-skip tests pass. 32k fwd+bwd: dense unchanged, keep 25%/50%
  ~3% faster. Follow-up: the `block_keep` / kv_route attention path still ignores sliding windows -- fix before
  any OLMo 3 layer-skipping arm.
- RESOLVED (uncommitted trainer change): `olmo3_7B_ctc_swa` took YaRN x8 only; it now takes `yarn_factor`,
  set via `--rope-yarn-factor 4`.
- ~~The eval harnesses silently build Qwen3-4B for an unknown model family~~ FIXED on branch `claude/vigorous-agnesi-20ecb0` (`ctc_devloss_grid.build_cfg` raises); OLMo 3 dev CE = `research/token_routing/olmo3_cptdrop/eval_devce.py` + `compare_devce.py` (paired per-row SE).
- short4k is one preprocessed file (`dolma3_longmino_0625/000500.npy`): LLM-written step-by-step math, CE ~0.47. A narrow forgetting guard; rebuild from several stage-3 files (already in next_step). `ops result` auto-fills a binomial SE (0.031) on those rows: meaningless for CE, use the paired SE.
- Pool length-bucket labels are approximate in the OLMo 3 tokenizer; the builder measures lengths itself.
- Lambda: /data not writable, use node /tmp; sbatch exit codes lie; fair-share 8 preemptive_high + 8 preemptive.
- 7B is not in train_ctc_suite's `_LARGE_SCALES`: pass `--shard-degree 8 --activation-checkpointing full`
  or every GPU holds full fp32 params + Adam state (~117 GB) and OOMs.
- Train topic mix follows the pool (45% science_math_and_technology, 13 of 22 topics sampled); dev is
  topic-balanced over 23 topics. Same for every arm, so arm comparisons are unaffected; absolute CE is not
  comparable to OLMo's own stage-3 numbers.
- Arm C's two phases use different shards: phase 2 must load model + optimizer only (`--branch-optim-only`),
  never trainer state, or the data loader resumes at a position in the wrong shard.

## Verified vs unverified
- Verified 2026-10-20/21: HF has `stage2-step47684` (8k, no rope scaling) and `stage3-step11921` (65,536,
  YaRN ×8, `sliding_window` 4096 on 3 of 4 layers), 14.6 GB bf16 each; olmo-checkpoints.org serves the
  OLMo-core `model_and_optim` for both. Pool layout: 24 topic folders × 8 length buckets (2^13–2^20), jsonl.zst
  with `id`/`text`; ~490 docs per 2^15 file. Two stage-3 training shards measured: 41% of tokens in docs >8k,
  max doc 132,838 tokens, rows packed to 65,536.
- Verified 2026-10-21: shards (counts, leading EOS, mask, max id, decoded text, train/dev disjoint by doc
  id); stage-2 conversion parity (above).
- UNVERIFIED: everything about training.
