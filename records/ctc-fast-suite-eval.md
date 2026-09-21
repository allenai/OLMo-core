# A one-hour CTC suite: what it costs, and what has to change

**Goal** (prasann, 2026-09-21): a full CTC-suite evaluation up to the **256k** rung that finishes in
**~1 hour on 8 GPUs**, splittable into 8 single-GPU Beaker jobs, plus the two held-out OOD rows
(`contra_fever`, `outlier_review`) that the results-hub tracks but the suite does not contain.

Scope decided with prasann up front: **dense / full-attention arms only**, served through vLLM. The
mask-variant arms (docchunk / landmark / compressive / summtoken) have no HF or vLLM mapping and run
through the native `olmo_core` evaluator at batch size 1; they need their own budget and are out of
scope here.

## The size of the thing

Counted from the olmo-eval roster (`prasann/ctc-suite-grader-fixes`), 22 tasks, every rung at or
below 256k, at the eval sizes the roster declares:

| span | cells | examples | prompt tokens |
|---|---|---|---|
| figure ladder 2k–32k | 108 | 51,127 | **0.63B** |
| everything ≤256k | 153 | 67,254 | **2.52B** |
| everything ≤256k at 100/rung | 153 | 14,472 | **0.82B** |

Per rung, at 100/rung: 2k 0.005B · 4k 0.009B · 8k 0.018B · 16k 0.036B · 32k 0.066B · 64k 0.098B ·
128k 0.197B · **256k 0.393B**. So **≥128k is 72% of the bill** even after subsampling — the xlong
rungs are what the budget is actually spent on, not the figure ladder. Cutting examples at 2k–32k
saves almost nothing; cutting them at 256k is the whole lever.

Eight GPU-hours is 28,800 GPU-seconds. 100/rung therefore needs **~28.5k prompt-tokens/s sustained
on one GPU**, decode included.

## Why the current harness cannot get there

The 2026-09-19 hybridish sweep (`debug/hybridish_sft/sweep_ids.txt`) is the measured baseline for
olmo-eval's **HF provider**: 16 Beaker jobs, 8 tasks × 5 rungs (2k–32k) × 500 examples, one GPU each.

| arm | per-task wall-clock | implied throughput |
|---|---|---|
| 4:1 (0.23B params) | 46–97 min (mean 66) | ~7.8k prompt-tok/s |
| 7:1 (0.29B params) | 71–164 min (mean 103) | ~5.0k prompt-tok/s |

That is **4x short of the target on a model 17x smaller than a 4B**. The binding constraint is the
**provider**, not the sample count: no subsetting closes a gap that large, and a 4B model through
the same path is roughly 60x off. Any plan that keeps `provider.kind=hf` is not a one-hour plan.

⚠ **olmo-eval pins `vllm[runai]==0.19.1`** (`pyproject.toml`), which long predates Qwen3.5 and GDN.
Its vLLM provider therefore cannot serve this model family as shipped — the version has to be
overridden to 0.25.1, or execution has to happen outside olmo-eval. The provider does forward
`**engine_kwargs` to `LLM(...)`, so once the version is right, `hf_overrides` and
`limit_mm_per_prompt` are reachable from `-o provider.*`.

## Two load facts worth keeping

* **Stock `Qwen/Qwen3.5-4B` loads in vLLM 0.25.1 on jupiter with none of the 7-piece serving-copy
  recipe** — vLLM reports `All limits of multimodal modalities ... set to 0, running in text-only
  mode` on its own. That recipe ([[qwen35-4b-vllm-load-recipe]]) is needed only for *our*
  olmo-exported text-only checkpoints, which declare a config vLLM resolves to the multimodal class.
* **`CUDA_HOME` must point at the pip `nvidia/cuda_nvcc` component.** Omitting it kills the engine
  at init with `Could not find nvcc and default cuda_home='/usr/local/cuda' doesn't exist` — *after*
  the model has loaded, so it reads as a model problem. This is the step `run_pipeline.sh` preserves
  and this benchmark's first attempt dropped.

## The OOD rows

Neither is in the 22-row suite; both exist only as 4-rung (2k/3k–32k) ladders in the in-house eval
bundle. Both are now built to **2k–256k**, in `/scratch/users/prasann/ctc_ood_ladders/`.

### `contra_fever` — built by expansion

Grades with the `contradiction` spec; schema is byte-compatible with `contradiction_iid`
(1-indexed gold pairs, verified). Built with `expand_ctc_rung.py` from the eval bundle's
`contradiction_eval_fever_plain_n100/n1642` files, which are already a proper v2 ladder (the same
599 questions at every rung, only distractor count varying — checked by comparing gold pair *texts*
across the family).

Expansion is legitimate here and would not be for outlier: contradiction gold is **pairwise**, so an
injected document does not become an unlabelled true positive. `expand_ctc_rung` refuses structural
gold for exactly this reason, and the ladder that ignored the rule went 0.428 @32k → 0.069 @65k.

| rung | docs/ex | realized prefill p50 | vs label |
|---|---|---|---|
| 2k | 67 | 2,042 | −0.3% |
| 4k | 146 | 4,143 | +1.1% |
| 8k | 310 | 8,216 | +0.3% |
| 16k | 640 | 16,759 | +2.3% |
| 32k | 1,298 | 33,809 | +3.2% |
| 64k | 2,615 | 68,941 | +5.2% |
| 128k | 4,411 | 131,378 | +0.2% |
| 256k | 8,796 | 261,884 | −0.1% |

599 examples per rung. Gold pair texts identical to the source on **599/599 rows at every rung**.
⚠ the 256k rung's p90 is 264,294, past Qwen3.5's 262,144 position ceiling — it needs a **YaRN
factor-2** serving copy, exactly as the suite's own r256k rungs do.

Two tools had to be written for this:

* **`shrink_ctc_rung.py`** — `expand_ctc_rung` can only grow, and contra_fever's smallest shipped
  build already measures p50 2,958, so calling it the 2k rung would have put a +44% label error on
  the x-axis ([[ctc-rung-labels-not-tokens]]). Dropping non-gold documents is sound for exactly the
  gold semantics injection is sound for, so it reuses expand's own task table and refuses the rest.
* **`dedup_rung_fillers.py`** — expand draws fillers from documents that are non-gold in *every*
  example, which includes documents the example being grown already holds, so fillers land on top of
  existing text. The shipped ladders carry 0.07% (contradiction) to 0.5% (nq) duplicates this way;
  contra_fever's small pool made it 1.7–2.6%. Duplicates never touch gold, but they quietly shrink
  the haystack. Each duplicate is replaced **in place** by an unused pool document, so counts and
  every index field are untouched. After the pass: **0.0000% duplicates, gold still 599/599**.

### `outlier_review` — regenerated natively

⚠ **It is not the same task as `outlier_amzn`, and `outlier_amzn` is not one task.** Both come from
`generate_review_outlier_data.py` over Amazon reviews, but the axis differs: `outlier_amzn` is
`--rating-ratio 0.5`, and its shipped rungs are **exactly 250 `review_outlier_rating` + 250
`review_outlier_category` at every rung from 2k to 32k** — one suite row averaging a star-rating
task and a category task under one metric. `outlier_review` is the pure category construction at
K=3, which is the domain-shift control for a model trained on wiki-category `outlier`.

Regeneration, not expansion: outlier gold is **structural** ("the documents from the least common
category"), so injected fillers satisfy the gold condition without being labelled.

n per rung mirrors outlier_amzn's grid (20/40/80/160/320/596/1207/2429) so the two Amazon rows share
an x-axis; 500 examples per rung to 128k and 125 at 256k, per the suite's eval_size policy.

## Still open

1. The vLLM cost model at 2k–32k, then 64k–256k — the measurement this file exists to record.
2. eval_size policy + LPT-balanced 8-way shard assignment (`size_the_suite.py`).
3. Registering the two OOD rows in olmo-eval. They should go in a **separate `OOD_ROSTER` and a
   `ctc:ood` suite**, not appended to `ROSTER`: `ROSTER` is the roster frozen 2026-08-12, and
   appending would silently redefine `ctc:figure` (108 → 124 runs), `ctc:low` and `ctc:high`.
4. Staging the new rung files where Beaker can read them (weka, or uploaded to the public HF dataset
   — the latter needs prasann's HF token).

Related: [[ctc-final-suite-22-tasks]], [[ctc-1m-ladders-olmo-eval]], [[ctc-public-repo-release]],
[[eval-wallclock-sanity-check]], [[eval-size-and-error-bars]], [[beaker-qwen35-vllm-cracked]].
