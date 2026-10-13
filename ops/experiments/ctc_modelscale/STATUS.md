---
experiment: ctc_modelscale
title: Model scale x family x CTC class (Qwen3.5 0.8B/2B/4B, OLMo-3-7B, Llama) + dense length generalization past the 32k training ceiling
status: paused
updated: 2026-09-14
branch: prasann/landmark
launcher: recipes/train/research_trainer/beaker_ctc_suite.py
records:
  - /accounts/projects/berkeleynlp/prasann/projects/OLMo-core/records/ctc-model-scale-plan.md
results:
  - /accounts/projects/berkeleynlp/prasann/projects/OLMo-core/debug/ctc_modelscale/LAUNCH_LEDGER.tsv
  - /accounts/projects/berkeleynlp/prasann/projects/OLMo-core/debug/ctc_modelscale/scalefam_grid_status.md
  - /accounts/projects/berkeleynlp/prasann/projects/OLMo-core/debug/ctc_modelscale/results/
  - /accounts/projects/berkeleynlp/prasann/projects/OLMo-core/debug/ctc_crossfamily/launches/
  - /accounts/projects/berkeleynlp/prasann/projects/OLMo-core/debug/ctc_crossfamily/results/
  - /accounts/projects/berkeleynlp/prasann/projects/OLMo-core/debug/ctc_llama8b/LAUNCH_LEDGER.tsv
  - /accounts/projects/berkeleynlp/prasann/projects/OLMo-core/results/ctc_suite_llama_iid/all_results.jsonl
  - /accounts/projects/berkeleynlp/prasann/projects/OLMo-core/paperdraft/figures/data/ctc_scale_data.csv
  - /accounts/projects/berkeleynlp/prasann/projects/OLMo-core/paperdraft/figures/data/ctc_length_generalization_data.csv
wandb_group:
next_step: Re-grade the contradiction length-gen rungs (rung_65674/rung_131342) with gold pairs canonicalised. ~53% of their gold pairs are unsorted, so the paper's high-CTC retention numbers for contradiction are depressed by construction.
---

# ctc_modelscale

The `ctc_crossfamily` key is merged here. The OLMo-3-7B and Llama-3B/8B "family" arms are columns of
the same scale x family grid (`scalefam_grid_status.md`), launched from the same ledgers. They are not
a separate campaign. The OLMo-3 vs Olmo-Hybrid backbone pair is its own page, `olmo3_vs_hybrid_wave2`.

## Question
(1) Does model scale (0.8B→2B→4B), or a different family, close the chunked-vs-full gap? Is there a
minimum viable scale for high-CTC tasks? (2) When a dense model trained to 32k is pushed to 64k/128k,
do high-CTC tasks degrade more than low-CTC tasks?

## Decisions
- 2026-08-12 (Prasann): tasks are contradiction, reorder, hotpotqa and qdmatch_nq at 0.8B/2B/4B x {dense, chunked-mix}. hotpotqa is the O(N) anchor.
- 2026-08-12 (agent): `TransformerConfig.qwen3_5_2B` is added (2.390B params, strict load OK). Its marker audit passes, so no fixmark is needed. The evaluator stays vLLM (0.079 f1 native-vs-vLLM drift on this family).
- 2026-08-12 (agent): length-gen rungs are built by `expand_ctc_rung.py`. They are nested supersets with measured token targets, and 64k/128k use native RoPE (no YaRN).
- ~2026-08-21 (agent): the grid widens to 10 tasks x {Qwen 0.8B/2B/4B, OLMo-3-7B (SWA-kept fixmark base), Llama-3B}. OLMo-3 runs on Beaker (moved to saturn 2026-08-22). Llama is not expanded past contradiction/hotpotqa/qdmatch.
- 2026-08-15 to 2026-08-31 (agent/Prasann): figures CTC-SCALE-ROW, CTC-SCALE-BARS (full attention only, reorder replacing contradiction), CTC-FAMILY(-BARS), CTC-SCALE-LENGTHGEN and CTC-RETENTION are committed in `paperdraft/figures/`.
- 2026-08-25 (agent): the OOLONG chunked arm was graded `cot_mode=plan` against a `none` comparator. Re-run cot-matched, the mask-mixing gain flips sign (+0.577 → −0.037).

## Current best numbers
All rows are dense/full-attention Qwen3.5 unless named, with eval_size 500 per cell (`paperdraft/figures/data/*.csv`).

| arm | task / rung | metric | eval_size | ±SE | cot | source |
|---|---|---|---|---|---|---|
| 0.8B / 2B / 4B full | reorder / 8k | τ 0.0068 / 0.0041 / 0.2618 | 500 | ±0.004 / ±0.003 / ±0.020 | no-cot | ctc_scale_data.csv |
| 0.8B / 2B / 4B full | qdmatch_nq / 32k | pair-F1 0.000 / 0.663 / 0.686 | 500 | 0 / ±0.021 / ±0.021 | no-cot | ctc_scale_data.csv |
| 2B / 4B chunked-mix | qdmatch_nq / 32k | pair-F1 0.180 / 0.104 | 500 | ±0.017 / ±0.014 | no-cot | ctc_scale_data.csv |
| 0.8B / 4B full | hotpotqa / 32k | gold-ID F1 0.940 / 0.943 | 500 | ±0.011 / ±0.010 | no-cot | ctc_scale_data.csv |
| 4B full, length-gen | qdmatch_nq / 32k→64k→128k | 0.686 → 0.223 → 0.017 | 500 | ±0.021 / ±0.019 / ±0.006 | no-cot | ctc_length_generalization_data.csv |
| 4B full, length-gen | hotpotqa / 32k→66k→131k | 0.943 → 0.718 → 0.408 | 500 | ±0.010 / ±0.020 / ±0.022 | no-cot | same |
| 4B full, length-gen | contradiction_iid / 32k→66k→131k | 0.946 → 0.278 → 0.061 ⚠ VOID-SUSPECT | 500 | ±0.010 / ±0.020 / ±0.011 | no-cot | same (see traps) |
| 4B full, length-gen | oolong / 32k→64k→128k | 0.573 → 0.573 → 0.575 | 500 / 668 / 669 | ~±0.02 | no-cot | same |

Verdict as captioned in the paper figures README:
- high-CTC tasks have a minimum viable scale (reorder is at chance below 4B; qdmatch_nq is at zero at 0.8B);
- low-CTC tasks move by ≤0.086 across scales;
- the chunked relative gap narrows with scale on contradiction (0.40→0.35→0.26 at 32k) but widens on qdmatch_nq;
- at 128k, low-CTC tasks keep 35-100% of their 32k score and high-CTC tasks keep 3-6%.

## Next step
Re-grade the contradiction length-gen cells with canonicalised gold pairs. olmo-eval `5f165a65`
canonicalises them. Alternatively run `debug/ctc_fast_suite/sort_gold_pairs.py` on the rung files and
re-score. Then re-render CTC-LENGTHGEN, CTC-RETENTION and CTC-SCALE-LENGTHGEN. Secondary: an LR sweep
for the 2B nq/oolong cells before quoting them.

## Open issues / traps
- **Unsorted contradiction gold in the length-gen rungs.** Checked 2026-10-12 on `cubbins:/data/prasann/ctc_suite_staged/eval_rungs_lengthgen/contradiction_iid/rung_65674.jsonl` (built 2026-08-12): 48 of 90 gold pairs in the first 30 rows are unsorted. Set-intersection scoring cannot match them (memory:contradiction-xlong-unsorted-gold-pairs). The 0.278 and 0.061 numbers predate the fix, so the contradiction high-CTC retention line is at least partly mechanical. qdmatch is not affected: its pairs are ordered.
- 2B shares the 0.8B hyperparameters (lr 5e-5, gbatch 8, 1 epoch). The 2B nq and oolong cells are non-monotone, so they are under-tuned. Do not quote them as a scale trend. outlier_amzn 2B is a clean win on the same hyperparameters.
- xabsence / olmo3-7b full reads 1.000 at 2k and 4k. That is either saturation or the known shipped-data shortcut (memory:shipped-data-shortcuts-strmatch-xabsence-hotpotqa). Verify before use.
- Length-gen rungs scale N with tokens, so longer means a bigger corpus. Only the excess drop beyond each task's own 2k-32k N-trend is a length effect. absence length-gen is RETRACTED: injected filler satisfies absence gold.
- Llama-3.1-8B (`debug/ctc_llama8b/`, 2026-08-14) was graded only at contradiction 2k. chunked-mix f1 is 0.0016 (eval_size 500). The full arm has `full_gens.json` but no summary. It was never followed up, and it is not in any figure.
- Every chunked run in the ledger is chunked-mix, so the "chunked" column includes the mask-mixing curriculum (TABLE-MASKMIX).
- `beaker_ctc_suite.py` used to block on `launch(follow=True)`. Pass `--no-follow` when batch-launching. It also stores weights twice (memory:beaker-launcher-missing-nofinalckpt).

## Verified vs unverified
- Verified 2026-10-12:
  - the table values against `paperdraft/figures/data/ctc_scale_data.csv` (2026-08-15) and `ctc_length_generalization_data.csv` (2026-09-14);
  - the grid against `debug/ctc_modelscale/scalefam_grid_status.md` (2026-08-25);
  - the last ledger rows: modelscale 2026-08-25 AC-full retries, crossfamily 2026-08-25 olmo3-7b evals, llama8b 2026-08-14;
  - the unsorted-pair sample on the real rung file.
- UNVERIFIED — confirm with Prasann: which evaluator graded the contradiction length-gen cells, and therefore whether the full file's unsorted share (sample ~53%) maps to roughly half-recall. Also whether the paper text still quotes 3-6% high-CTC retention.
- UNVERIFIED: whether the cells marked `train-run`/`eval-run` in the 2026-08-25 grid ever completed (olmo3-7b oolong/outlier-scaleK, 0.8B cmix). The grid has not been regenerated since.
- The memory file `ctc-model-scale-lengthgen` (2026-08-12, "ACTIVE") is a stale launch-day snapshot.
