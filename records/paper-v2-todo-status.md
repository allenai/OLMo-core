# Paper v2 — experimental TODO list: take-stock + execution status

Owner: prasann + Claude. Opened 2026-08-04. Live document — update as items land.

Source: the 12-item TODO list for the next paper iteration. This file organises them, records what
was **already known** in the repo before this effort started (several items were open bugs with
partial diagnoses), and tracks execution.

Two standing constraints for this effort, from the user (2026-08-04):
- **Debug small first.** Validate at the 2k rung with a small eval before running a full ladder or
  a retrain. Applies to every item below.
- **Node-local only.** No `/scratch` for anything — data, caches, checkpoints, logs. `/scratch` is
  NFS at ~5 MB/s and sits at 88% of a 500G per-user quota; a flashinfer JIT compile there is what
  killed the first contradiction eval attempt (`No space left on device`, job 3422671). Use the
  target node's `/data` and the `node_local_env.sh` preamble, which pins `HOME`,
  `FLASHINFER_CACHE_DIR`, `HF_HOME`, `TRITON_CACHE_DIR` to `/data`.

## Status board

| # | Item | State | Blocking finding |
|---|---|---|---|
| 1 | Contradiction train/eval sanity + FEVER mixing | **fix built, re-eval running** | leak is EVAL-ONLY; rung labels also mis-calibrated |
| 2 | Add 2 more qdmatch settings | design constraint identified | existing qdmatch is lexically anchored → doesn't stress O(N²) |
| 3 | NarrativeQA / GovReport training | root cause known, fix exists but is OFF | `truncate_generic` newline bug → 61.8% empty preds |
| 4 | OOLONG low numbers | not started | shards predate the `--item-regex` fix; no dense arm; eval_size 100 |
| 5 | Grouping low numbers | **root-caused; chunked re-grade running** | 2 parser bugs; published table carries superseded numbers |
| 6 | xabsence atomic-op difficulty ablation | not started | also needs the A/B-block asymmetry fixed |
| 7 | Outlier no-scale-k ablation | not started | — |
| 8 | Model scaling rerun | not started | should wait on items 1/5 data fixes |
| 9 | More model families | partly done | OLMo-3-7B clean; Llama-3.2-3B contra dense arm broken |
| 10 | Bigger suite without mask-mixing | not started | should wait on items 1/5 data fixes |
| 11 | Manual data validation + co-author doc | not started | this file is its seed |
| 12 | Upload eval sets to HF | **DEFERRED by user** | "10M" = 10M-context-token rungs; do not start until told |

---

## 1. Contradiction — FEVER filler leak + rung mis-calibration

**Confirmed the leak.** `debug/xlong_5task/audit_contra_fever_leak.py` on the live CTC-suite ladder:

| rung | 2k | 4k | 8k | 16k | 32k | 128k |
|---|---|---|---|---|---|---|
| % FEVER/wiki fillers | 92.2 | 96.4 | 98.3 | 99.2 | 99.6 | 29.9 |

Gold is 100% PubMed. Visible on inspection — one biomedical claim among Wikipedia trivia
(*"West Virginia borders Maine to the southwest."*), so the task degenerates to "find the
biomedical sentences."

**The training data is CLEAN.** Decoded the tokenized 4B train shard directly
(`ctc_suite_staged/shards/contradiction_train`, 19366 instances): every claim in the inspected
examples is PubMed. So this is **eval-only** — the model trained on all-PubMed corpora and was
scored on ~99%-Wikipedia ones, i.e. a train/eval distribution shift *on top of* the shortcut.
**No retraining is needed**; the eval ladder had to be rebuilt.

**Second bug found while rebuilding: the rung→`n_docs` map was calibrated on the contaminated pool.**
FEVER trivia tokenizes at ~22.8 tok/doc, real PubMed claims at ~43. Re-running the same `n` against
the fixed pubmed-only glob overshoots every label by ~1.8× (measured: `n=77` → 3413 tokens, not
2048; `n=1423` → 61461, not 32768). Refit on the clean pool:

    tokens = 170 + 42.8 * n_docs        →  n = {44, 92, 187, 379, 762}

This lands within a few docs of the **original** BUILD_MATRIX row-16 ladder (40/88/190/385/765),
which was calibrated on real PubMed and was correct all along — "FIX 2" only looked necessary
because the pool was contaminated. Added as the `contra_ctc` config in
`src/corpus_reasoning/data/build_v2_eval_ladders.py`.

**Rebuilt ladder verified** (eval_size 500 at every rung):

| rung | n_docs | tok p50 | tok p95 | FEVER/wiki |
|---|---|---|---|---|
| 2k | 44 | 1925 | 2142 | 0.00% |
| 4k | 92 | 3933 | 4315 | 0.00% |
| 8k | 187 | 8052 | 8376 | 0.00% |
| 16k | 379 | 16074 | 17121 | 0.00% |
| 32k | 762 | 32397 | 33802 | 0.00% |

Staged node-local at `cubbins:/data/prasann/ctc_suite_staged/eval_rungs/contradiction_clean/`.

**Every contradiction number in `results/ctc_suite/dense_vs_chunked_table.md` is measured on the
contaminated ladder** and must be re-measured. Also check whether the Llama-3.2-3B and OLMo-3-7B
contradiction rows read the same rungs.

⚠ **p95 exceeds the rung label at 16k/32k** — eval `max_length` must be sized from the measured
prompt distribution, not `rung + 2048` (the known truncation trap; `run_rung_eval` already does
this as of 53b52101d).

### First clean measurement (2k, Qwen3.5-4B dense) — the leak did NOT inflate the short rung

| ladder | n_docs | f1 | exact_match | eval_size |
|---|---|---|---|---|
| contaminated (92.2% FEVER) | 77 | 0.8491 | — | 500 |
| **clean (0.00% FEVER)** | 44 | **0.8427** | 0.560 | 500 |

Delta **−0.006** against a binomial SE of **±0.016** at f1≈0.85, eval_size 500 — **not significant**.
Same checkpoint, same harness, parse_rate 1.0 on both.

This is a genuine negative result at 2k and should not be spun. Two honest readings, not yet
separated:
- **The shortcut's value scales with corpus size.** At n=44 with 3 gold pairs the search space is
  small enough that "restrict to the biomedical sentences" saves little. Contamination was 92.2% at
  2k but 99.2%/99.6% at 16k/32k where the corpus is 379/762 docs, so any real effect should appear
  there. This is the reading the memory note predicted ("the shortcut strengthens as n grows").
- **The comparison is not perfectly controlled.** The clean 2k rung is n=44 and the contaminated one
  n=77 — both are "2k tokens" by measured prompt length (1925 vs 1797), which is the comparison the
  figure's x-axis makes, but they are not the same corpus size. A same-n control would isolate the
  filler domain from the doc count.

### Full clean dense ladder — the leak DEPRESSED the numbers, worst at 32k

The 2k null was misleading, and the prediction written above it ("if the leak was doing work, clean
comes in below") had the **sign backwards**. The contamination was not a helpful shortcut; it was a
harmful train/eval domain shift (train 100% PubMed, eval up to 99.6% Wikipedia), and its cost grows
with corpus size:

| rung | contaminated | **clean** | delta | ~SE |
|---|---|---|---|---|
| 2k | 0.849 | 0.843 | −0.006 | ±0.016 |
| 4k | 0.766 | **0.803** | +0.037 | ±0.018 |
| 8k | 0.690 | **0.744** | +0.054 | ±0.020 |
| 16k | 0.619 | **0.664** | +0.045 | ±0.021 |
| 32k | 0.335 | **0.559** | **+0.224** | ±0.022 |

Qwen3.5-4B dense, eval_size 500, parse_rate 1.0 at every rung. The 32k delta is ~10 SE.

**Cross-checked against results-hub** (user directive). The hub's clean-bundle contradiction ladder
(`_eval_bundle_eval500_v2_clean`, eval_size 500) uses n = {100, 190, 385, 765}; under the
recalibration above those are near-identical corpora to this ladder's n = {..., 187, 379, 762}:

| corpus size | results-hub dense (`q4b-dense-5task`) | this ladder, clean (`ctc-4b-contradiction-full`) |
|---|---|---|
| n≈190 | 0.769 | 0.744 |
| n≈385 | 0.684 | 0.664 |
| n≈765 | 0.611 | 0.559 |

Different checkpoints (5-task mix vs single-task CTC), so exact agreement isn't expected — but the
clean ladder now sits within 0.02–0.05 of an independently produced clean measurement at every
corpus size, whereas the contaminated 32k cell (0.335) was 0.28 away from it. That is the check
that confirms the rebuild is right.

**Consequence for the paper:** the published "contradiction dense collapses at 32k"
(0.619 → 0.335) is an artifact of the contaminated ladder. The real curve is a graceful decline
(0.664 → 0.559). Contradiction is the O(N²) anchor, so this changes the headline figure.

⚠ **The hub's own contradiction rung LABELS are wrong in the same way FIX-2 was**: its "2k" row is
n=100 ≈ 4450 measured tokens. Its 8k/16k/32k labels are accurate. Worth fixing at the source so the
two ladders can be plotted on one x-axis.

### Both arms, clean ladder, COMPLETE (Qwen3.5-4B)

| rung | dense | chunked-mix | abs gap | ratio |
|---|---|---|---|---|
| 2k | 0.843 | 0.402 | 0.441 | 2.10× |
| 4k | 0.803 | 0.331 | 0.472 | 2.43× |
| 8k | 0.744 | 0.277 | 0.467 | 2.69× |
| 16k | 0.664 | 0.234 | 0.430 | 2.84× |
| 32k | 0.559 | 0.191 | 0.369 | 2.93× |

eval_size 500, parse_rate 1.0 throughout, both arms on `contradiction_clean`.

⚠ **The absolute gap does NOT widen with context — it narrows slightly (0.441 → 0.369).** Only the
*ratio* grows (2.1× → 2.9×). The plan's T1 criterion for an O(N²) task is "gap grows ≥2× from
smallest to largest rung", and on the clean ladder contradiction **fails that criterion**: both arms
decay roughly proportionally. The previously published widening was partly an artifact of the
contaminated ladder collapsing the dense arm at 32k (0.335, now 0.559).

Chunked is not at floor (0.19 at 32k vs ~0 for random pair guessing), so this is a real
proportional-decay result, not a saturation artifact. Which lens (absolute gap vs ratio) the paper
should use for the CTC claim is now an open question that needs deciding explicitly rather than
implicitly.

**Open sub-item:** `COMPLEXITY_VERDICTS.md` notes the ladder never samples the paper's Fig-5
"chunked starts near full" regime (N≈20) — the old 2k rung was already N=77. The clean 2k rung is
N=44, still above it. Consider adding a sub-2k rung at N≈20 so the small-N end is observable.

## 5. Grouping — the low numbers are a grader artifact (already root-caused; re-grade unfinished)

`COMPLEXITY_VERDICTS.md` (2026-07-22) found and fixed **two compounding grader bugs**:

- **Bug A — `parse_partition` mangles the ramble.** grouping's stop rule is `eos`, so after emitting
  the correct single-line answer the model rambles more `{"groups": ...}` objects. The greedy
  `re.finditer(r'\{[\s\S]*\}')` spanned first-brace..last-brace across all of them → invalid JSON →
  a digit-scrape fallback that lumped every doc_id into one giant cluster. On the saved dense-2048
  gens: parse(full ramble) = **0.4395**, parse(first valid object) = **0.8419**.
- **Bug A layer 2 (chunked only).** The `-cmix` model emits a stray `</think>` right after the
  answer's opening `{"groups": [{"doc_ids": [`, so `run_vllm_eval`'s `split("</think>",1)[1]` threw
  the opening away and the stored response began mid-array. Fixed by re-attaching the primer and
  retrying. Chunked @2k: **0.44 → 0.816**.
- **Bug B — output truncation.** Driver default `MAX_NEW_TOKENS=256`, but the gold partition grows
  with N (~228 tok @8k, ~493 @16k, ~1011 @32k). Re-runs use 1300.

Both fixes are present in the repo today (`raw_decode` + primer retry in both
`src/corpus_reasoning/lib/eval_tasks.py` and `src/scripts/ctc_eval/lib/eval_tasks.py`).

**What was left undone:** the dense ladder was re-run to completion, the **chunked ladder was not**,
and `results/ctc_suite/dense_vs_chunked_table.md` (2026-07-27) still carries the broken numbers for
both arms. Corrected dense, already on disk under
`debug/ctc_vllm_validation/node_local_results/`:

| rung | 2k | 4k | 8k | 16k | 32k |
|---|---|---|---|---|---|
| grouping — published "dense" (wrong) | 0.439 | 0.358 | 0.186 | 0.043 | 0.011 |
| grouping — **corrected dense** | **0.842** | **0.705** | **0.529** | **0.285** | 0.007 |
| grouping_labeled — published "dense" (wrong) | 0.439 | 0.365 | 0.231 | 0.054 | — |
| grouping_labeled — **corrected dense** | **0.827** | **0.697** | **0.532** | **0.092** | 0.005 |

The 32k dense cell stays ~0 even after the fix and was verified REAL (complete well-formed
partition of all 176 docs, just wrong) — a genuine 4B capability limit.

**SCOPE NARROWED (user, 2026-08-04): finish `grouping_labeled` only.** The two tasks share the same
eval rungs and differ only in the prompt, so they were near-duplicate grid rows. The chunked
re-grade now runs `grouping_labeled` 4k/8k/16k/32k (job 3422770); plain `grouping` is dropped.
Because plain `grouping` will therefore have a corrected dense arm but an uncorrected chunked arm,
its row should be **removed** from the table rather than left half-stale.

Validated at 2k before fanning out (job 3422753):

| task | arm | 2k f1 before | 2k f1 after | ARI before | ARI after | coverage after |
|---|---|---|---|---|---|---|
| grouping | chunked | 0.4386 | **0.8158** | 0.002 | **0.733** | 0.993 |
| grouping_labeled | chunked | 0.4388 | **0.8203** | 0.002 | **0.767** | 0.990 |

So at 2k the corrected arms are dense 0.827 vs chunked 0.820 — **no chunked collapse at small N**,
confirming the headline correction rather than the published "identical decline" story.

⚠ **`WEIRD_NUMBERS_VERDICTS.md` §7 contains the superseded WRONG verdict** ("both arms decline in
lock-step → genuine O(N·M) scaling, no artifact"). The lock-step was itself the artifact signature:
both arms were deflated to ~0.44 by the *same* parser bug. Anyone reading that section without
`COMPLEXITY_VERDICTS.md` will re-adopt the wrong conclusion. Mark it superseded.

**Also note for reporting:** `pairwise_f1` has a large non-zero floor — a degenerate prediction
scores ~0.44 while ARI ≈ 0. Never report grouping `pairwise_f1` without ARI beside it.

## 2. qdmatch — design constraint for the 2 new settings

`COMPLEXITY_VERDICTS.md` §4 established that **qdmatch_hpqa does not stress O(N²)**: dense is flat
(0.999 → 0.981 across 2k→32k) because the matches are **lexically anchored** — the query's anchor
entity appears verbatim in the gold document, so the match is retrievable in O(N) rather than
requiring exhaustive pairwise comparison. The grader is sound (the same grader shows real variation
on the chunked arm).

So the two new qdmatch settings should be chosen to **break lexical anchoring** — queries whose
match to the gold document is semantic/inferential rather than surface-overlapping. Otherwise they
will reproduce the same flat dense curve and add rows without adding evidence.

## 3. NarrativeQA / GovReport — why training "isn't working"

`WEIRD_NUMBERS_VERDICTS.md` §5 on helmet_qa (dense, 2k, token_f1 = 0.033):

1. **61.8% of predictions are literally empty.** `run_vllm_eval.py:truncate_generic` strips
   `<think>…</think>` then returns `text.split("\n",1)[0]` for `stop_rule=="newline"`. The
   post-`</think>` text usually *starts* with a newline, so the first line is empty and the real
   answer on the next line is discarded. Same class as the known Qwen3.5 think/newline-stop trap.
2. Among the non-empty 38%, mean token_f1 is only 0.086 — genuinely weak free-form QA.

A `NEWLINE_ROBUST=1` opt-in guard exists in `truncate_generic` but is **OFF by default** (kept off so
already-validated grid tasks stay byte-identical). Enabling it for the helmet tasks is step 1.

Even a perfect recovery lifts the ceiling to only ~0.09, so there is a second, real problem: the
checkpoint is retrieval-tuned on the CTC mix, and free-form QA / ROUGE summarization were never
really trained as such. Both need a look at the training target, not just the grader.

## 4. OOLONG

Three separate problems, none yet addressed:
- Shards built before 2026-07-26 hit the `--item-regex '||'` leak bug (bare alternation matched every
  line → the preamble was wrapped as chunks with FREE bridges). All such shards need rebuilding.
- There is **no dense arm at all** — the table's oolong row is chunked-only.
- `eval_size = 100/rung`, below the 500 floor, so every number needs an inline ±SE flag until rebuilt.

## 9. More model families

- **OLMo-3-7B: done and clean.** contradiction / qdmatch_hpqa / hotpotqa, both arms, 2k–16k,
  eval_size 500, parse_rate 1.0. Shows the intended pattern (gap tracks task structure, not length).
- **Llama-3.2-3B: the contradiction dense arm is broken.** Seed-0 full scores 0.0002–0.0008 (dead);
  the seed-1 rerun reaches only 0.330/0.268/0.200/0.112 at 2k–16k, versus ~0.83 at 2k for both Qwen
  and OLMo-3. Its retrieval arm is fine (0.977 dense / 0.956 chunked at 2k). So this is
  contradiction-specific, not a broken family port. Needs diagnosis before it can be reported.
- At least one further family is still owed.

## Cross-cutting: two evals that are not well-posed for their complexity class

Recorded in `COMPLEXITY_VERDICTS.md`, relevant to any figure that groups tasks by class:
- **cycle** — every instance is a *single* length-3 cycle at every rung; only distractors scale. A
  len-3 loop is locally findable, so chunking doesn't break it and both arms sit at ~1.0. To probe
  O(N³) the cycle length and/or count must scale with N.
- **qdmatch_hpqa** — see item 2 above.

Neither is a bug; both are task-construction limits that should be either fixed or stated.

## Chunked-eval speedup investigation — CLOSED, with a negative result

Chunked eval runs ~7x slower than dense on the same prompts (48 ex, contradiction clean 2k:
**123.5 s vs 17.4 s**). Two hypotheses were tested and **both were wrong**; recording them so nobody
re-derives them.

**Hypothesis 1 (WRONG): the per-step chunk_ids rebuild is the cost.** `_build_chunk_ids_for_batch`
re-scans every request's full token stream and Python-loops over every doc start, on every decode
step, even though chunk_ids are invariant during decode. Real, but tiny — CPU microbenchmark
(`debug/chunked_eval_speedup/bench_chunk_ids.py`):

| case | ms/call | projected s / 500 ex | measured total | share |
|---|---|---|---|---|
| contra 2k (n=44) | 2.23 | 4.5 | 1199 s | **0.4%** |
| contra 32k (n=762) | 16.29 | 66 | — | small |

(Within that, the Python `for s in starts` loop is 96% of it; `flatnonzero` 0.57 ms, alloc 0.08 ms.)

**Hypothesis 2 (WRONG): `create_block_mask` isn't compiled.** It is — vLLM defines
`create_block_mask_compiled = torch.compile(create_block_mask, fullgraph=True, mode="reduce-overhead")`
and our patch calls it directly.

**Measured attribution** (`CHUNK_PROFILE=1`, job 3422856, 48 ex @2k, total 123.5 s):

| stage | seconds | share |
|---|---|---|
| build_block_mask | 11.27 | 9.1% |
| orig_build | 7.07 | 5.7% |
| build_chunk_ids | 2.36 | 1.9% |
| make_mask_mod | 0.01 | 0.0% |
| **kernel + model forward (unaccounted)** | **102.8** | **83.2%** |

So **the entire patch's bookkeeping is 17%**, and the chunk_ids caching originally proposed would
have bought ~2%. The cost is the FlexAttention kernel — dense uses vLLM's optimized attention
backend, chunked must use FlexAttention to carry a custom `mask_mod`.

**Hypothesis 3 (also WRONG): the forced BLOCK_N=32 is the cost.**
`_patch_flex_kernel_options_pow2` clamps the triton kernel's KV tile to the largest power-of-2
divisor of the page size; Qwen3.5 pads the attention page to **288** to align with the mamba page,
and 288 = 2^5·9, so BLOCK_N is forced to **32** (a pure-softmax model would get 128+). Sweep
(job 3422884):

| BLOCK_N | gen time (48 ex @2k) |
|---|---|
| 32 (default clamp) | 123.5 s |
| 16 | 125.5 s |
| 64 | crash |

Halving the tile costs **1.6%**. The sensitivity is far too low for tile size to explain a 7x gap,
so raising it to 128 would not recover the gap either. (BLOCK_N > 16 crashed with
`Q and KV block size must be divisible by BLOCK_M and BLOCK_N` — the per-call q/kv block size is 16
on the decode path, so a blanket override is invalid; only the value the stock clamp computes
per-call is safe. `CHUNK_BLOCK_N` is therefore a diagnostic knob only, never a config.)

### Verdict: no cheap speedup exists. Thread closed.

The ~7x is **inherent to FlexAttention**: dense uses vLLM's optimized attention backend, and a
custom `mask_mod` requires FlexAttention, which is simply slower. Every accessible knob has been
measured and none is worth taking. Recovering the gap would mean a bespoke chunked-attention kernel
— a real project, not an optimization, and not justified by the current eval volume. Plan chunked
eval time at ~7x dense and parallelise across nodes instead.

Instrumentation (`CHUNK_PROFILE`, `CHUNK_BLOCK_N`) is **off by default** — no CUDA syncs and no
behaviour change when unset, so validated numbers are untouched.

⚠ There are TWO copies of the patch. The vLLM driver imports
**`corpus_reasoning.lib.vllm_chunked_patch`**, not `scripts/ctc_eval/lib/`. Editing the wrong one
is a silent no-op.

## Item 3 update — helmet: eval artifact is real but small; the model is the problem

| task | baseline | NEWLINE_ROBUST=1 |
|---|---|---|
| helmet_qa (NarrativeQA, token_f1) | 0.033 | **0.0437** |
| helmet_summ (GovReport, rouge1_f) | 0.329 | **0.329** (unchanged) |

The newline fix helps QA only (summarization isn't a newline-stop task). Two findings from the
generations:

1. **48% of helmet_qa predictions are STILL empty** (down from 61.8%, not solved). Job 3422888
   re-runs with `DUMP_RAW_GENS=1` to classify the residue: immediate-EOS vs unclosed `<think>` vs
   whitespace. Because the metric averages the empties in, the true per-answered f1 is ~0.084.
2. **The non-empty answers are fluent, on-format, and wrong** — and at least one answers from a
   *different book* ("He leaves Satis house", i.e. Great Expectations, for a question about another
   story). That is parametric recall, not reading the context: a training/task-fit problem, not a
   grader problem.

Provisional read: **helmet_summ is not broken, it is uninformative** — ROUGE-1 ~0.33 flat across
every rung is in the normal band for GovReport and does not move with context, so it cannot serve as
a CTC probe. **helmet_qa is genuinely failing** (HELMET NarrativeQA F1 for competent models is
~0.2–0.3, not 0.04).

## Item 4 update — oolong reference from results-hub

The CTC row is chunked-only. results-hub's clean v2 bundle (eval_size 500) has the full picture:

| arm | 8k | 16k | 32k |
|---|---|---|---|
| full (dense) | 0.682 | 0.676 | 0.640 |
| doc_chunked (leak-fixed) | 0.220 | 0.193 | 0.140 |
| **random_doc-trained chunked** | **0.654** | **0.647** | **0.615** |
| compressive_landmark | 0.664 | 0.653 | 0.608 |

**A model trained with RANDOMLY grouped chunks scores ~3x higher than one grouped by true document.**
That is not a plausible capability ordering, and it lines up with `CHUNK_LEAK_AUDIT.md`: oolong's
training convert wraps the instruction/question/header as their own chunks (question duplicated at
both ends) while eval keeps the preamble FREE — so the `-cmix` model is evaluated on a prompt layout
it never trained on. Random grouping presumably scrambles that structure less destructively. Fixing
the train/eval layout mismatch is the concrete next step, and it requires a retrain.

(The oolong eval rungs are eval_size **500**, not 100 — the earlier 100/rung note is stale.)

## Compute notes

- **lambda is nearly full: 1.23 T / 1.30 T = 94.4%**, ~70 G free. Checkpoints are ~12 G each and need
  ~24 G transient, so lambda can host at most 2–3 runs before risking a write failure at 100%.
  Deletes there are classifier-blocked → **the user has to free space** before lambda is useful for
  this effort. Not used tonight for that reason.
- **lorax is NOT a drop-in eval node**: its node-local conda env lacks `cached_path`, so
  `build_prefills_any.py` dies at import (job 3422804 failed in 26 s). sneetches and cubbins are
  complete.
- Beaker `beaker experiment get <id>` does not resolve gantry job ids; use
  `beaker job list --author prasanns --cluster ai2/jupiter-cirrascale-2`. Also: in zsh an unquoted
  `$c` does not word-split, so loop over beaker subcommands inside `bash -c`.
- The gantry launcher runs with `follow=True` and never exits. Killing the client does **not** cancel
  the submitted job.

## Execution log

- 2026-08-04 — Audited CTC contradiction ladder: 92–99.6% FEVER confirmed. Train shard decoded and
  confirmed clean PubMed. Rung map recalibrated; `contra_ctc` config added; clean ladder built and
  verified (0.00% leak, token-accurate); staged node-local to cubbins. First eval attempt on horton
  (3422671) died on `/scratch` flashinfer JIT — relaunched node-local on cubbins (3422760).
- 2026-08-04 — Grouping root cause traced to the already-documented parser bugs; confirmed the
  published table carries superseded numbers and that the chunked re-grade never ran. Launched the
  2k chunked validation for grouping + grouping_labeled (3422753) to reproduce the documented 0.816
  before committing the full ladder.
- 2026-08-04 evening — Clean contradiction DENSE ladder complete (0.843/0.803/0.744/0.664/0.559);
  cross-validated against results-hub. Chunked arm running. Llama-3.2-3B clean-ladder re-eval
  launched (3422861) after measuring Llama-tokenizer rung lengths: max ~= p95 at every rung, so no
  `--allow-short-max-length` and no dropped examples (the contaminated 16k rung had p50 29k / max
  101k, which is why the old Llama runs had to drop the tail). grouping_labeled chunked ladder
  running (3422770). oolong DENSE arm launched on sneetches (3422815) after lorax failed on a
  missing `cached_path`. helmet 2k robust eval done (qa 0.033->0.0437, summ unchanged); raw-gen
  diagnostic queued (3422888). Chunked-eval speedup investigated and CLOSED as no-cheap-win after
  three refuted hypotheses. Beaker: pure-chunked (no mask-mix) contradiction 4B submitted to
  jupiter at urgent, jobs 01KZ83A33Q2P6GX6DNK1ABGKV8 / 01KZ83A37F3FZP57BZYPT9Q8DM (pending on
  capacity) -- serves item 10 and completes the contradiction arm set on the clean ladder.

---

# Overnight results — 2026-08-05

## Item 9 — Llama-3.2-3B: the leak was NOT the explanation (hypothesis refuted)

Full arm re-scored on the clean ladder (eval_size 500 every rung):

| rung | contaminated | clean | delta |
|---|---|---|---|
| 2k | 0.330 | 0.362 | +0.032 |
| 4k | 0.268 | 0.255 | −0.013 |
| 8k | 0.200 | 0.194 | −0.006 |
| 16k | 0.112 | 0.154 | +0.042 |
| 32k | — | 0.063 | — |

Essentially unchanged. **Llama-3.2-3B is genuinely far weaker at contradiction** than Qwen3.5-4B
(0.843) or OLMo-3-7B (0.829) at 2k on the identical clean task — not a data artifact and not a
broken port (its retrieval arm is 0.977/0.956). I earlier suggested the leak likely explained this;
that is now refuted. Whether a 3B model simply cannot do N² pair-finding, or the Llama SFT recipe
needs work, is the open question — a 3B-vs-4B capability gap this large deserves one control before
being reported as a family effect.

The chunked-mix arm failed on all rungs with `--variant: invalid choice: 'doc_chunked'` (the flag
takes `{dense,chunked}`); fixed and relaunched as job 3423126.

## Item 4 — OOLONG: the dense checkpoint is DEGENERATE, and that is the headline

The dense arm finally ran (job 3422815). It is **worse than chunked at every rung**, which inverts
both the CTC thesis and results-hub:

| rung | 2k | 4k | 8k | 16k | 32k |
|---|---|---|---|---|---|
| CTC dense (`ctc-4b-oolong-full`) | 0.232 | 0.231 | 0.180 | 0.139 | 0.136 |
| CTC chunked (`-cmix`) | 0.628 | 0.524 | 0.390 | 0.297 | 0.297 |
| results-hub dense (clean v2) | — | — | **0.682** | 0.676 | 0.640 |

Inspecting the generations settles it: **the dense checkpoint is in repetition collapse** —
`"1\n1\n1\n1..."`, `"positive\npositive\n..."`, `"numeric value\nnumeric value\n..."` for 300/300
sampled examples (0% empty, 100% degenerate). It never emits the templated `answer:` the oolong stop
rule keys on, so it runs to max_new_tokens emitting one token forever.

So the grid's "oolong = CHUNKED-ONLY (no -full ckpt)" was closer to the truth than this new number:
a serving copy exists, but the run behind it is broken. **Do not report CTC oolong dense.** Use
results-hub's oolong ladder, which has a healthy dense arm, and retrain the CTC oolong arms after
fixing the preamble layout mismatch (`CHUNK_LEAK_AUDIT.md`) — the mismatch and the `random_doc`
anomaly are still the substantive open issue.

## Item 3 — helmet_qa: the empties are MODEL degeneration, not a harness bug

`DUMP_RAW_GENS=1` (job 3422888, 120 ex) resolves the residual 48–49% empties:

    truncated-empty = 59/120 (49.2%)
      raw is 256 tokens of pure "\n"  = 59/59

Every empty is the model emitting **nothing but newlines** until it hits max_new_tokens. (My
diagnostic labelled these "immediate EOS" because `.strip()` on all-newlines is empty — the label is
wrong, the classification is not: none were unclosed `<think>`.)

Combined with the earlier finding that the *non*-empty answers are fluent but answered from
parametric memory ("He leaves Satis house" for a question about a different story), item 3's verdict
is: **the eval harness is now fine; helmet_qa training genuinely failed.** Same degeneration class as
the oolong dense checkpoint above — two of the CTC checkpoints collapsed into repetition, which is
worth checking across the whole roster before trusting any near-floor cell.

## Item 5 — grouping_labeled chunked: 4k/8k landed, 16k/32k timed out

| rung | published (broken) | corrected chunked |
|---|---|---|
| 2k | 0.4388 | **0.8203** (ARI 0.767) |
| 4k | 0.3702 | **0.6790** (ARI 0.627) |
| 8k | 0.2259 | **0.4750** (ARI 0.445) |
| 16k | 0.0511 | stale — rerunning |
| 32k | 0.0179 | stale — rerunning |

Job 3422770 hit its 4h wall mid-16k (chunked eval is ~7x dense and these carry
MAX_NEW_TOKENS=1300). Relaunched as 3423127 with an 11h budget.

Against the corrected dense arm (0.827/0.697/0.532/0.092) the picture at 2k–8k is dense ≈ chunked
(0.827 vs 0.820, 0.697 vs 0.679, 0.532 vs 0.475) — i.e. **grouping shows no chunked collapse at
small/mid N**, confirming the 2026-07-22 headline correction rather than the published table.

## Cross-cutting: check every near-floor cell for repetition collapse

Two independent checkpoints (oolong dense, helmet_qa dense) are in degenerate repetition, and in
both cases the number looked like a plausible "hard task" result rather than a broken run. Before
any near-floor cell in the grid is reported as a capability finding, its generations must be
inspected. This is the same lesson as `goldgrad-eval-maxlen-truncation-bug` but for a different
failure mode.
