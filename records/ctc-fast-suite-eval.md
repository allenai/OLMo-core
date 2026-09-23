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

### Corrected once the xlong rungs were measured

⚠ **The flat extrapolation above was optimistic by ~40%.** Prefill IS flat to 32k, but it decays
past it. `outlier` gives a clean series (same spec, same 256-token decode budget at every rung, so
rung-to-rung is pure prefill):

| rung | prefill tok/s | vs previous |
|---|---|---|
| 32k | 48,337 | — |
| 64k | 42,526 | 0.880x |
| 128k | 33,870 | 0.796x |
| 256k | ~26,975 *projected* | 0.796x assumed — **not measured**, it needs a YaRN copy |

That is the full-attention layers' quadratic term finally showing; the GDN layers keep it to ~20%
per doubling rather than 4x, which is still the reason any of this is affordable, but it is not
free. Repricing the same policies:

| policy | GPU-h | slowest shard |
|---|---|---|
| **S** 200/rung to 32k; 100 at 64k; 50 at 128k; 25 at 256k | 4.9 | **42 min** |
| **R** 500/rung to 32k; 50 at 64k+128k; 25 at 256k | 7.7 | **63 min** |
| A 100/rung throughout | 7.9 | 65 min |
| C 500 to 32k; 100 at 64k/128k; 50 at 256k | 10.0 | 80 min |

**S fits an hour with room; R buys the full 500-example figure ladder for three minutes over.**
Assuming an SFT checkpoint stops early barely moves either, because above 32k the cost is prefill,
not decode -- so the eval_size at 128k/256k is the only lever that really matters.

⚠ **The 128k MAX_LENGTH in the repo's own table (146,227) is too small.** `fiqa:r128k` has prompts
of at least 146,228 tokens and was rejected outright. Any 128k launch needs a larger cap, and a cap
that merely *looks* generous silently skips the tail.

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

⚠ The **contra_fever ladder built above inherited this** (same tool) and has been repaired.

**Resolved in the GRADER instead of the data (prasann's call, and the better one).** `parse()`
already sorts predicted pairs, on the stated grounds that "1 contradicts 4" and "4 contradicts 1"
are the same claim. Gold was never canonicalised the same way, and *that asymmetry is the whole
defect*. So `_check_gold` now canonicalises rather than raises:

* it fixes every affected file at once, including ones nobody has audited, and any future builder
  that reintroduces the same bug;
* it needs **no dataset re-upload**, which was the one blocker requiring an HF token;
* ordered-pair tasks cannot be caught by it -- they go through `parse_qd_pairs` and never reach
  this function.

The signal is kept: module counter `UNSORTED_GOLD_SEEN` records what it canonicalised, so a builder
emitting unsorted pairs stays visible even though the scores are now correct. Verified: unchanged on
already-sorted gold; the shipped `rung_65536` scores 50/50 without raising (73 pairs canonicalised);
and the **gold-answer control on that same corrupt file is f1 1.0000**, which it could not reach
before.

⚠ **The fix is in the VENDORED copy and must go upstream.** The vendor tree's own rule is
fix-upstream-then-re-vendor, so the same change belongs in the `ctc` package (OLMo-core branch
`prasann/ctc` / the public repo) before this diverges.

## The recommended config (S+)

200 at each of 2k/4k/8k/16k/32k, 100 at 64k, 50 at 128k, 50 at 256k -- **26,627 examples over 169
cells, 6.11 GPU-h, ~51 min** across 8 single-GPU shards.

| rung | n per row | rows | examples | per-row SE @f1~0.7 |
|---|---|---|---|---|
| 2k / 4k / 8k / 16k | 200 | 24 | ~4,725 each | ±0.032 |
| 32k | 200 | 22 | 4,326 | ±0.032 |
| 64k | 100 | 17 | 1,700 | ±0.046 |
| 128k | 50 | 17 | 850 | ±0.065 |
| 256k | 50 | 17 | 850 | ±0.065 |

Per-rung totals fall short of `n x rows` because some rows cap themselves lower -- `scifact` is 300
and `obliq_twitter` 126 at every rung, and the roster declares 125 at 256k+.

Why these numbers and not others:

* **The xlong tail is the only lever.** Above 32k the cost is prefill, so thinning the short rungs
  buys almost nothing and assuming an SFT checkpoint stops generating early buys almost nothing
  either. 128k->256k at 50 each costs 9 minutes over the 25/25 version and halves the 256k error
  bar; taking 128k to 100 costs another 7 minutes and does not move 256k at all.
* **Grouping and reorder stay in.** They are the two largest decode budgets (4096 and 2048) and the
  2026-09-19 sweep dropped them for speed -- but that was under the HF provider. Under vLLM they are
  7% of the total (16.9 + 7.1 min of 367) and are not on the critical path; dropping them saves
  ~0 wall-clock because the 256k cells decide it. Grouping is also the only O(NM) clustering row at
  the base rungs.
* **Shard per CELL, not per row.** Same work, 51 min vs 59.5 min -- 24 lumpy rows do not divide into
  8 jobs evenly, and the row-level split leaves the shard holding the 256k-heavy rows on the
  critical path alone.

⚠ **Every rung is below the 500 floor**, so every number from this config carries an inline
`eval_size` and error bar per [[eval-size-and-error-bars]]. ±0.032 at the short rungs is fine for
ladder *shape*; a 0.03 difference between two arms at a single rung is not resolvable. This is the
**iteration** config -- score every checkpoint, watch the shape, catch regressions. Anything
publication-facing re-runs the specific rows at 500 (125 at 256k, the roster's own size), which is
the ~3-hour config.

### What carries the xlong half

Only 17 of 24 rows reach 64k, and **6 of the 7 that stop short are high-CTC**:

| row | tops out | why |
|---|---|---|
| `absence`, `reorder` | 16k | real ceiling -- one contiguous book, ~250k max as constructed |
| `grouping`, `strmatch`, `textgroups`, `xabsence`, `scifact` | 32k | merely UNBUILT -- regen-only, the 2026-08-14 push ran out of time |

So the high-CTC class drops from 11 rows at <=32k to 6 at 64k+, which is what makes the pooled
high-CTC statistic the binding constraint on the error bars up there (gap SE ±0.037 at 50/row,
±0.053 at 25/row). The long-context story rests on `contradiction`, the three `qdmatch` rows,
`outlier` and `contra_fever`. Extending even two of the five unbuilt rows would materially
strengthen it -- `grouping`'s OpenAlex pool alone is reported to support ~1M.

## SFT training sets (2026-09-22)

Builder: `src/scripts/data/hybridish/build_ctc_sft_sets.sh <set-a|set-b> <out-root>`.

**Set A is the one being built** (prasann: "just use set A"). **13 tasks**:
`nq hotpotqa qdmatch_nq outlier oolong contradiction xabsence absence reorder rerank strmatch
textgroups grouping_labeled`.

* **Token-balanced buckets, not example-balanced.** Every context bucket gets the same token budget
  (default 20M), so examples fall as the bucket grows -- ~9,765 at 2k, ~610 at 32k. Equal examples
  per bucket would spend ~99% of the budget above 32k.
* **`query_position=both`**, applied at `convert_ctc_to_sft_completion.py`, NOT at `ctc-data build`
  (rung files are raw unified JSONL; the prompt is rendered at tokenisation). It must match the eval
  flag or the mismatch reads as a capability gap.
* **`qdmatch_hpqa` dropped** so the qdmatch spec is trained from `qdmatch_nq` alone and
  `qdmatch_hpqa`/`qdmatch_fiqa` stay clean probes.
* Set B (CTC-BENCH-10, `fiqa`->`rerank`, `qdmatch_fiqa`->`reorder`) is defined in the script but
  **not being built**.

⚠ **Six roster members have no `ctc-data` generator** -- `msmarco`, `niah`, `obliq_twitter`,
`qdmatch_fiqa`, `outlier_amzn`, `outlier_fixedM`. That is why "as many of the 22 as possible" is 13,
not 20. `outlier_amzn`/`outlier_fixedM` would come from `generate_review_outlier_data.py`.

⚠ **`ctc-data` the console script is unusable**: it is shebanged to a python without
`huggingface_hub`, and `--pool auto` fetches seed pools from the Hub, so every build dies at the
first task. Drive it as `python -m ctc.data.cli` under the `corpus-reasoning-olmo` env with
`PYTHONPATH=<newolmocore>/OLMo-core/ctc/src`. The script now does this.

⚠ **Training `outlier_amzn` would gut the `outlier_review` probe** -- the former is half
category-axis Amazon reviews and the latter IS category-axis Amazon reviews. `TASK_SPEC` in
`build_ctc_sft_mix.py` now declares this (it previously listed only 6 ladders, so the
two-sources-per-spec guard could not see `hotpotqa`, `niah`, `obliq`, `qdmatch_fiqa` or any
`outlier` variant -- and `DEFAULT_ROSTER` ships `msmarco`+`hotpotqa`, a collision it was blind to).

### In flight at handoff

**The build runs on BEAKER, CPU-only** (prasann: "CPU heavy just use lots of CPUs, do this on beaker
so all data / training is fully on beaker"). The 65 (task, bucket) builds are independent, so they
run under `xargs -P`: ~6 h serially on an 8-core login node, ~20 min on a 180-core Beaker box. A
serial warm-up pass builds one bucket per task first, so 65 processes do not race to fetch the same
seed pools onto a cold HF cache.

    gantry run --name ctc-sftdata-setA-c -w ai2/flex2 -b ai2/oe-other \
      --cluster ai2/jupiter-cirrascale-2 --gpus 0 --cpus 64 --priority urgent \
      --weka oe-training-default:/weka/oe-training-default --branch prasann/landmark \
      --env SET=set-a -- bash src/scripts/data/hybridish/run_ctc_sft_build_beaker.sh

Output: `/weka/oe-training-default/ai2-llm/checkpoints/prasanns/ctc_sft_sets/setA_max20/`, so
training reads it with no staging hop. Last run: `prasanns/ctc-sftdata-setA-c`
(`01M33VRDV1HCXHMTJZGJN1T02M`).

⚠ **`uv pip install`, never `pip` or `python -m pip`.** gantry builds its runtime with uv, and a uv
venv ships WITHOUT pip, so both die with `No module named pip` in ~70 ms -- which reads as a network
or bad-URL failure and is neither. Cost two launches to find. Applies to ANY gantry job that installs
something at runtime.

⚠ **The `ctc` package is not in `prasann/landmark`.** It is installed from the public release repo
(`git+https://github.com/PrasannS/corpustaskcomplexity.git#subdirectory=ctc`). If that ever becomes
flaky, vendor the source into this branch instead and drop the network dependency.

**Before tokenising, check the merged per-task row counts** (~18k per task over 5 buckets). That is
where the bucket-clobbering bug above would hide, and it hides well -- the build reports success
throughout.

### Next: the two SFT runs (NOT launched)

prasann wants **dense attention** and **compressive landmark**, ~1B tokens each, >=500 steps.
Step arithmetic is comfortable: 1B tokens at seq 40960 and global batch 8 is ~3,000 steps; even at
global batch 32 it is ~760. Two pre-flight checks that are silent failures if skipped:

1. **The base checkpoint needs repaired marker embeddings** before ANY landmark/document-chunked
   training -- and re-repaired if the fix predates 2026-07-14, because the first version fixed the
   marker cosine but not the norm and flatlines training at CE ~0.79 for *every* mask, which reads
   as "the mask is too restrictive" when it is not. See [[marker-embedding-norm-bug]].
2. **The run name must carry the variant.** The docchunk/landmark eval path infers its emitter from
   the run name (`*compressive*`/`*landmark*` -> landmark emitter, else dense). A mismatched emitter
   produces garbage, not a low score.

Watch the loss curve in the first few hundred steps: a wrong emitter, an unrepaired base and a bad
shard all produce a plausible-looking or flat curve rather than an error.

## Still open

1. The vLLM cost model at 2k–32k, then 64k–256k — the measurement this file exists to record.
2. eval_size policy + LPT-balanced 8-way shard assignment (`size_the_suite.py`).
3. ~~Registering the two OOD rows in olmo-eval.~~ **DONE** (olmo-eval `a150b926`, branch
   `prasann/ctc-absence-low-ctc`): `OOD_ROSTER` holds `ctc_contra_fever` and `ctc_outlier_review`,
   registered as `ctc:ood` / `ctc:ood:figure` / `ctc:ood:xlong` and deliberately NOT folded into
   `ctc`/`ctc:figure`/`ctc:low`/`ctc:high`, so an already-published figure-ladder number still
   refers to the roster frozen 2026-08-12. Ask for OOD explicitly.
   Note from that roster: `contra_fever` r256k p90 is 264,294 — past the 262,144 position ceiling,
   so it needs a YaRN factor-2 serving copy exactly as the suite's own r256k rungs do.
4. Staging the new rung files where Beaker can read them (weka, or uploaded to the public HF dataset
   — the latter needs prasann's HF token).

Related: [[ctc-final-suite-22-tasks]], [[ctc-1m-ladders-olmo-eval]], [[ctc-public-repo-release]],
[[eval-wallclock-sanity-check]], [[eval-size-and-error-bars]], [[beaker-qwen35-vllm-cracked]].
