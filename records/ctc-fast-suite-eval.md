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

## Measured: what a cell actually costs

Qwen3.5-4B (stock) through vLLM 0.25.1, one H100 on jupiter, 100 examples/cell, 33 cells over 7
tasks x 2k-32k. `debug/ctc_fast_suite/results/bench_2k32k.json`; the runner is
`run_bench_beaker.sh`, the fit and shard packing are `size_the_suite.py`.

**Prefill throughput is FLAT in context length** -- the result the whole budget turns on:

| rung | prefill tok/s (aggregate, excl. grouping) |
|---|---|
| 2k | 30,972 |
| 4k | 40,086 |
| 8k | 45,802 |
| 16k | 46,983 |
| 32k | 46,459 |

There is no quadratic term, which is what a GDN hybrid should do -- only the full-attention layers
pay for length, and they are a minority of the stack. The climb from 2k to 8k is batch efficiency,
not attention. Fitting `t = P/R_prefill + G/R_decode` over all 33 cells (two global parameters,
justified by that flatness) gives **prefill 51,067 tok/s, decode 5,352 tok/s**, rms residual 10.8 s.

For scale: the HF provider measured ~7.8k tok/s on a **0.23B** model. vLLM is ~6.5x faster in
absolute terms on a model **17x larger**.

⚠ **Decode, not prefill, is what varies between tasks.** The stock base model never emits EOS, so
every cell ran to its full `max_new_tokens`, and olmo-eval passes no decode-time stop strings by
design. `grouping` (budget 4096) costs 157 s per 100 examples at 16k against `fiqa`'s 26 s -- 6x,
at identical prompt length. An SFT checkpoint will stop earlier, so these are upper bounds, but
`grouping` and `reorder` (2048) are the cells that decide a shard's wall-clock.

The fit's worst residual is `grouping:r16k`, 49 s under-predicted: one global decode rate cannot
capture how concurrency falls when 100 sequences each generate 4k tokens. Treat decode-heavy cells
as optimistic in the table below.

### The budget

24 rows (the 22-row suite + the two OOD rows), every rung <=256k, LPT-packed into 8 single-GPU
jobs, 300 s per job for model load and GDN JIT. **Rungs above 32k are priced at the 32k rate --
that extrapolation is not yet measured, and it is ~80% of the bill.**

| policy | GPU-h | slowest shard |
|---|---|---|
| **A** 100/rung throughout | 5.4 | **46 min** |
| **B** A, assuming an SFT checkpoint stops at ~40% of budget | 5.2 | 44 min |
| **C** 500/rung to 32k, 100 at 64k/128k, 50 at 256k | 7.4 | **61 min** |
| **D** 500 to 32k + 125 at xlong (the roster's own sizes) | 10.7 | 85 min |

**C is the interesting one**: the full 500-example figure ladder survives, paid for entirely out of
the xlong rungs. That works because at a uniform 100/rung the >=64k rungs are 80% of the cost, so
the cheap rungs are nearly free and the expensive ones are where subsetting buys anything.

## Found on the way: the shipped contradiction xlong rungs are unscoreable

The benchmark died on `contradiction_iid:r64k` with the vendored spec's own gold guard:

    ValueError: gold pair [1359, 251] is not sorted low-high. Predicted pairs are sorted, and
    scoring is a set intersection, so this pair could never be matched and would silently cost
    recall on every example that contains it.

It is not one bad row. Every rung **above 32k** carries it, and the figure ladder carries none:

| rung | rows affected | unsorted pairs |
|---|---|---|
| 2k–32k | 0 | **0 / 1500** |
| 64k | 435 / 500 | 744 / 1500 |
| 128k | 442 / 500 | 764 / 1500 |
| 256k | 114 / 125 | 192 / 375 |
| 512k | 109 / 125 | 186 / 375 |
| 1M | 114 / 125 | 185 / 375 |

`expand_ctc_rung.py` shuffles documents and remaps every index, but never re-sorted the pair — so a
source pair (low, high) comes out in whatever order the shuffle left it. Roughly **half the gold is
unmatchable**, and the recall it costs lands *exactly* on the rungs where "contradiction collapses
at long context" is the expected, publishable-looking result. A mechanical defect wearing the shape
of the headline finding.

How it surfaces depends on the evaluator, which is the part worth internalising: the grader-fixes
branch **refuses** the file outright (that guard is why this was caught at all), while any older
evaluator scores it silently at half recall.

Fixed at source — sort after remap — and `debug/ctc_fast_suite/sort_gold_pairs.py` repairs files
already built. Sorting is meaning-preserving: a contradiction pair is unordered. **`qdmatch` is
deliberately not touched** — its `gold_pairs` are (query, doc), ordered by construction, and sorting
them would corrupt them; it is named explicitly rather than inferred for that reason.

⚠ The **contra_fever ladder built above inherited this** (same tool) and has been repaired: all 8
rungs now pass the spec's gold guard. **The shipped rungs on the HF dataset have NOT been repaired**
— that needs the fix re-run and a re-upload. Until then, treat every contradiction number at 64k and
above as void.

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
