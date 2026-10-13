---
experiment: hybridish_sft
title: Hybridish attention-ratio ladder 7:1 vs 4:1 (SSMax-matched) — in-domain CTC SFT, graded on the olmo-eval CTC suite
status: paused
updated: 2026-09-19
branch: prasann/ctc-sft-hybridish
launcher: src/scripts/train/hybrid-small-suite/launch_sft.sh
records:
  - /accounts/projects/berkeleynlp/prasann/projects/OLMo-core/records/hybridish-ctc-sft.md
results:
  - /accounts/projects/berkeleynlp/prasann/projects/OLMo-core/debug/hybridish_sft/results/ctc_suite_sweep.json
  - /accounts/projects/berkeleynlp/prasann/projects/OLMo-core/debug/hybridish_sft/results/sft4to1_fixed.json
  - /accounts/projects/berkeleynlp/prasann/projects/OLMo-core/debug/hybridish_sft/results/sft7to1_fixed.json
  - /accounts/projects/berkeleynlp/prasann/projects/OLMo-core/debug/hybridish_sft/sweep_ids.txt
wandb_group:
next_step: Find a parameter-matched pair, or quote non-embedding params beside every number. 7:1 (2.1B) has +62% non-embedding params over 4:1 (1.4B), so the 7:1 win is not attributable to the attention ratio yet.
---

# hybridish_sft

## Question
After SFT on a CTC mix, does a 7:1 GDN:full hybrid beat a 4:1 hybrid on high-CTC tasks more than on
low-CTC tasks? The base models are at the floor on the suite without SFT.

## Decisions
- 2026-09-16/17 (Prasann): finetune the hybridish ratio ladder on a CTC SFT mix. Start with smaller scales. Use two mixes (2k-4k and 2k-32k). Omit reorder and grouping_labeled from eval.
- 2026-09-17 (agent): the matched pair is 7:1 `tanushy/v0.0.1-seven_to_one_hybrid_ratio` vs 4:1 `yashasbls/v0.0.1-ssmax`. Their override-layer configs are byte-identical, both with SSMax. The `dolma3p5_v2` 4:1 tree lacks SSMax and must not be used.
- 2026-09-17 (Prasann): follow their recipe, not ours: SkipStepAdamW lr 1e-4 to 8e-4, wd 0.0, linear warmup 3%, 2M-token batch, 2 epochs, seq 32768.
- 2026-09-17 (agent): train on EOS (it was masked before). `retrieval` is drawn from nq, not MS MARCO, and this is recorded in the mix manifest.
- 2026-09-18 (agent): the first results table is RETRACTED. A shard/src positional-zip bug scored 459/470 probes against another row's gold. A gold self-check is now enforced.
- 2026-09-19 (agent): use self-contained checkpoints (SSMax-patched modeling + `auto_map`, `trust_remote_code`), with SSMax computed from `cache_position` so KV-cache decode is correct. Evaluate with olmo-eval `prasann/ctc-suite-grader-fixes` @`69729fa`: 8 tasks x 2k-32k, 500 per rung, 16 Beaker jobs.

## Current best numbers
Held-out olmo-eval CTC suite, eval_size 500 except absence (499/494/493 at 2k-8k, 146 at 16k).
Parse rate is 1.0 except 4:1 absence r16k (0.986). Arms: 4:1 = 1.4B, 7:1 = 2.1B (branch README).

| arm | task / rung | metric | eval_size | ±SE | cot | source |
|---|---|---|---|---|---|---|
| 4:1 / 7:1 | qdmatch_nq / r2k | 0.333 / 0.959 | 500 | ±0.021 / ±0.009 | no-cot | debug/hybridish_sft/results/ctc_suite_sweep.json |
| 4:1 / 7:1 | qdmatch_nq / r32k | 0.012 / 0.299 | 500 | ±0.005 / ±0.020 | no-cot | same |
| 4:1 / 7:1 | strmatch / r32k | 0.009 / 0.213 | 500 | ±0.004 / ±0.018 | no-cot | same |
| 4:1 / 7:1 | contradiction / r32k | 0.224 / 0.389 | 500 | ±0.019 / ±0.022 | no-cot | same |
| 4:1 / 7:1 | outlier / r32k | 0.024 / 0.047 | 500 | ±0.007 / ±0.009 | no-cot | same |
| 4:1 / 7:1 | nq / r32k | 0.862 / 0.918 | 500 | ±0.015 / ±0.012 | no-cot | same |
| 4:1 / 7:1 | oolong / r32k | 0.481 / 0.491 | 500 | ±0.022 / ±0.022 | no-cot | same |
| 4:1 / 7:1 | xabsence / r32k | 0.014 / 0.008 | 500 | ±0.005 / ±0.004 | no-cot | same |
| 4:1 / 7:1 | absence / r16k | 0.941 / 0.989 | ⚠ 146 | ±0.020 / ±0.009 | no-cot | same |

Verdict, provisional and confounded: 7:1 wins large on high-CTC pairwise tasks (qdmatch_nq,
strmatch, contradiction). It wins modestly on nq. It ties on oolong, absence and xabsence, where xabsence is at the
floor for both. This is the expected direction, but the arms are not parameter-matched.

## Next step
Resolve the parameter confound: a param-matched pair, or a smaller-scale replicate (275M-Cx8 was the
original plan). The 2k-4k mix arm in the original two-mix plan was never reported.

## Open issues / traps
- **Not parameter-matched.** Both arms have identical attention (4 full layers, 199.25M attention params). 7:1 adds 12 GDN layers, +62% non-embedding params.
- The stock `mainline_ladder` transformers plugin silently drops `ssmax_scale` (loaded as UNEXPECTED, `MISSING: []`). Our olmo-core has no `scalable_softmax`. Train only from `prasann/ctc-sft-hybridish`.
- SSMax must be applied after qk-norm. Scaling at the `q_proj` output measures identical to no SSMax.
- absence/xabsence 32k rungs realise up to 52k/42k tokens. At seq 32768 with `exclude`, 15.9% of tokens are dropped, concentrated in those two tasks (7:1 parse collapses on the older in-house table).
- Shards are a subsequence of src rows. Grade only via `src_index.json` plus a gold self-check (memory:shard-src-positional-zip-bug).
- The olmo-eval HF provider is slow (~5-8k prompt tok/s on these models). Use the vLLM path for anything bigger (see `ctc_suite_eval`).

## Verified vs unverified
- Verified 2026-10-12: every table value against `debug/hybridish_sft/results/ctc_suite_sweep.json` (2026-09-19 06:05); the arm sizes and the +62% figure against `src/scripts/train/hybrid-small-suite/README.md` on `origin/prasann/ctc-sft-hybridish` (tip `9a0ab011b`, 2026-09-19).
- Verified: `apply_ssmax_to_plugin.py` uses `cache_position` (the KV-decode SSMax fix).
- Verified (branch README): both graded checkpoints trained on `ctc_hybridish_sft/shards_long32k`. That is 15,183 instances, 8 tasks, 2k-32k, dolma2 tokenizer, ONE epoch. `launch_sft.sh` defaults to lr 4e-5 and 1 epoch. That departs from the 2026-09-17 "their recipe" decision (2 epochs, lr 1e-4 to 8e-4). UNVERIFIED — confirm with Prasann which LR the graded runs used and whether the 1-epoch / 4e-5 departure was intended.
- Conflict: `records/ctc-fast-suite-eval.md` describes this sweep as 0.23B/0.29B models. The branch README says 1.4B/2.1B. UNVERIFIED — confirm with Prasann.
- No memory file covers this experiment. The record (2026-09-19) is the only prior summary. No activity found after 2026-09-19.
