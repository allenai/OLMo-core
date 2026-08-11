# HELMET NarrativeQA + GovReport: why the CTC numbers were floored, and the repair

**Date:** 2026-08-08 · owner: prasann + Claude · branch: `prasann/landmark`
**Code:** `debug/ctc_helmet_v2/` · **Supersedes:** `debug/ctc_vllm_validation/HELMET_OOLONG_FIX.md` §2–3

The grid read `helmet_qa` token_f1 ≈ 0.044 and `helmet_summ` rouge1_f ≈ 0.33, both essentially flat
across 2k→32k. The earlier writeup concluded the QA number was "the true ceiling of this
checkpoint". That conclusion was wrong: three independent defects — two in the data, one in the
metric — floor the number before the model gets a say.

---

## 1. The reference numbers we should be near

HELMET's published results sheet
(`docs.google.com/spreadsheets/d/1LBt6dP4UwZwU_CjoYhyAd_rjKhQLvo0Gq4cYUnpi_CA`), NarrativeQA,
columns *ROUGE* (rougeL_f1) / *judge* (GPT-4o fluency×correctness) / *F1* (DrQA token-F1), all ×100:

| model | 8k | 16k | 32k |
|---|---|---|---|
| Llama-3.2-1B-Inst | 12.7 / 14.3 / 11.8 | 14.3 / 16.7 / 13.7 | 13.1 / 18.7 / 11.9 |
| Qwen2-7B-Inst | 17.2 / 19.7 / 16.6 | 14.4 / 24.0 / 14.3 | 9.9 / 26.3 / 10.2 |
| Llama-3.1-8B-Inst | 22.3 / 20.7 / 21.0 | 26.5 / 28.7 / 25.4 | 29.1 / 33.0 / 27.3 |

Two things to take from this. First, **our 4.4 is 3–5× below the weakest model HELMET reports.**
Second, **the reference numbers rise with length** (the judge column monotonically, for every model
listed) — because more of the story fits. Ours was flat. A flat curve where the reference rises is
the signature of a model that is not reading the context at all.

Summarization for scale: HELMET's Multi-LexSum judge-F1 runs 16.4–46.4 and its ROUGE 21–25. Our
GovReport ROUGE-1 of 0.33 is not obviously broken — but ROUGE is not the metric HELMET reports, so
the number was never comparable to anything.

---

## 2. Defect 1 (data) — NarrativeQA contexts are boilerplate, not story

`generate_helmet_qa_data.py` puts `document.text` straight into the context. For NarrativeQA that
field is the **raw scrape**: the IMSDb HTML page for movie scripts, the Project Gutenberg licence
header for books. Head-truncating to 2k tokens therefore keeps page chrome and nothing else. From
the dumped prompt of `eval_rungs/helmet_qa/rung_2048.jsonl`, example 0 in full:

```
Document: <html>
<head><title>Jacob's Ladder Script at IMSDb.</title>
<meta name="description" content="Jacob's Ladder script at the Internet Movie Script Database.">
... [meta tags, a movie-index table, a yellbox chat iframe, a comment form] ...
Question: What does Louis tell Jacob that hell burns away?
```

All 2762 tokens are the web page. There is no story in the context at all. Measured over the eval
set: boilerplate stripping fires on **296/500 HTML documents and 260/500 Gutenberg headers**.

## 3. Defect 2 (data) — the gold answer is usually not in the retained context

Measured over 200 sampled examples per rung (`debug/ctc_helmet_v2/reports/diag_helmet_qa.json`):

| rung | gold answer verbatim in context | gold-answer token coverage (soft ceiling) |
|---|---|---|
| 2048 | **9.5%** | 31.8% |
| 4096 | 18.0% | 56.3% |
| 8192 | 24.5% | 68.8% |
| 16384 | 30.0% | 80.7% |
| 32768 | 32.5% | 84.7% |

At 2k, **nine of ten examples are unanswerable**. The metric there is a floor, not a measurement.
`BUILD_MATRIX.md` row 5 flagged the risk ("at 2k the gold answer may lie beyond the truncation")
but nobody measured the rate.

HELMET avoids this by keeping **only documents longer than 131072 Llama-2 tokens** (`data.py::
load_narrativeqa`) and never running below 8k. Our generator applied no length filter and added
2k/4k rungs.

**The same defect is in the training set** — it was built by the same generator with the same
truncation. A model trained on data where the answer is absent ~85% of the time learns to answer
from parametric memory rather than from the context. That is exactly what the earlier generation
dumps showed: an answer about *Great Expectations* returned for a question about a different book.
This, not a capacity limit, is why the ladder is flat.

## 4. Defect 3 (data) — the GovReport ladder is not a ladder above 8k

Measured median context, Qwen3.5 tokens, per rung: **1556 / 3104 / 5961 / 8765 / 8765**. The 16k and
32k rungs are *the same inputs*. Length audit of the validation split (973 docs): p50 8895, p90
18162, p99 38066; fraction reaching each rung — 4k: 88.4%, 8k: 54.5%, 16k: 13.0%, **32k: 1.3%**.

Only 13 validation documents reach 32k. Any claim about GovReport behaviour between 16k and 32k in
the existing grid is an artifact of evaluating identical data twice.

## 5. Defect 4 (metric) — neither task used HELMET's metric

HELMET explicitly rejects n-gram overlap for these two tasks and scores them with GPT-4o:

* **LongQA** — one call returning `{"fluency": 0|1, "correctness": 0..3}`; example score is the
  product; reported as mean/3×100.
* **Summarization** — three calls (fluency; recall against key points decomposed from the reference;
  precision of the prediction's sentences against the reference), scored
  `fluency × 2·rec·prec/(rec+prec)`.

We reported token-F1 and ROUGE-1. Both are legitimate numbers; neither is comparable to HELMET.

## 5b. Defect 6 (metric) — the GovReport "ROUGE" numbers were never ROUGE

`_eval_summarization` wraps its rouge import in `except ImportError` and, on failure, computes
token-F1 — but still writes it under the keys `rouge1_f` / `rougeL_f`. On sneetches, `rouge_score`
was installed and its dependency **`absl-py` was not**, so the import failed on every run. On mooney
`rouge_score` was absent entirely.

**Every `helmet_summ` "rouge1_f" in the CTC grid is therefore token-F1 wearing a ROUGE label.** The
tell is in the grade JSON: `rouge1_f == rougeL_f` to sixteen decimal places. ROUGE-1 (unigram
overlap) and ROUGE-L (longest common subsequence) are never equal at that precision over 500
examples. It also means the earlier writeup's "ROUGE-1 F ~0.33 is normal for summarization; not
floored" compared a token-F1 value against ROUGE norms.

Fixed by installing `absl-py`/`rouge_score`/`nltk` into the node-local conda env on both nodes, and
by making the fallback print a loud warning instead of failing silently
(`src/corpus_reasoning/eval/evaluate.py`). `regrade_ngram.sbatch` re-grades from saved generations.

## 5c. Defect 7 (eval) — GovReport generations were truncated to a quarter of the reference

The shared driver defaults to `--max-new-tokens 256`, while GovReport reference summaries average
**715 Qwen3.5 tokens** (p90 924). Every generation was cut off well before it could cover the
reference, capping recall structurally. Raising the budget to 1024 (HELMET gives Multi-LexSum 400
and InfBench Sum 1200) moved the 2k rung from **0.329 → 0.447** on the same checkpoint and the same
data — a 36% relative gain from generation length alone, with no modelling change.

## 6. Defect 5 (eval) — half of the QA generations were empty

`TASK_CFG["qa"]["stop"] == "newline"`, and this checkpoint emits a leading blank line, so vLLM's
`stop=["\n"]` returned EMPTY for 61.8% of examples. `NEWLINE_ROBUST=1` (already in
`run_vllm_eval.py`, opt-in) cuts that to 47.8%; the residual are generations that are pure newlines
to the token limit. Separately, the QA instruction asks for `Answer: [answer]` while the training
target (`data_format.py:1056`) is the **bare** answer string — an instruction/target mismatch the
model also saw during training.

---

## 7. The repair

All new code lives in `debug/ctc_helmet_v2/`.

| file | what it does |
|---|---|
| `diag_helmet_data.py` | the measurements in §2–4 (answerability, context length, exact prompt dump) |
| `helmet_judge.py` | HELMET's longqa + summarization rubrics **verbatim**, judged by a local Qwen3-8B instead of GPT-4o; generates GovReport key points (HELMET ships pre-computed ones for its own two sets) |
| `helmet_native_repro.py` | rebuilds HELMET's own NarrativeQA (test split, >131072-Llama-2-token filter, HELMET prompt, 2-shot, greedy, 100 new tokens) and scores it with HELMET's own metric code |
| `build_helmet_v2_data.py` | v2 ladders: boilerplate stripping, exact-tokenizer truncation, long-document filter, per-example answerability flags, and an evidence-preserving truncation mode |
| `make_judge_rows.py` | keeps the conda (dataset/grading) and vLLM (judge) environments apart |
| `eval_ladder.sbatch` / `rejudge.sbatch` | ladder generation graded twice — n-gram *and* judge; rejudge reuses saved generations |

### Deliberate deviations from HELMET, all recorded in the judge output

1. Judge is Qwen3-8B, not `gpt-4o-2024-05-13`. `repro_gpu.sbatch` measures the disagreement against
   the published GPT-4o column on identical generations — do not quote a judge number without it.
2. The summarization rubric's domain clause is adapted to GovReport (HELMET's are written for "a
   civil lawsuit" / "a novel"). Rubric, worked examples and output schema are unchanged.
3. GovReport key points are generated by the judge model and cached, so recall is computed against
   an identical key-point set across every rung and arm.
4. `--truncation evidence` (window guaranteed to contain the answer) is **ours, not HELMET's**. It
   exists so the ladder measures long-context retrieval instead of the chance that a book's opening
   happens to contain the answer. Report `head` as the HELMET-comparable arm and `evidence` as the
   well-posed one — never blend them.

### v2 datasets

* `helmet_qa_head` / `helmet_qa_evidence` — narrativeqa test split, 500 examples, docs ≥32768
  Qwen3.5 tokens so every rung 2k→32k is a nested truncation of the same story.
* `helmet_qa_train_evidence` — the matching training pool, so the SFT model sees answerable contexts.
* `helmet_summ_long` — GovReport docs ≥32768 tokens pooled from validation+train (document hashes
  reserved so the SFT pool can hold them out). Genuine 2k→32k ladder; small eval_size, quote the
  error bar.
* `helmet_summ_wide` — GovReport docs ≥8192 tokens, validation only, rungs capped at 8k,
  eval_size 500. Full power over the range GovReport actually supports.

## 7b. Calibration result — the harness reproduces, the judge is biased

`repro_gpu.sbatch`, Qwen2-7B-Instruct on HELMET's own NarrativeQA, eval_size 100 (HELMET's
`max_test_samples`), 2-shot, greedy, 100 new tokens:

| L | metric | ours | HELMET | delta | ~SE | delta in SE |
|---|---|---|---|---|---|---|
| 8192 | ROUGE (rougeL_f1) | 19.5 | 17.2 | +2.3 | 3.8 | 0.6 |
| 8192 | token-F1 | 17.8 | 16.6 | +1.2 | 3.7 | 0.3 |
| 8192 | **judge** | **32.3** | **19.7** | **+12.6** | 4.0 | **3.2** |
| 16384 | ROUGE | 15.2 | 14.4 | +0.8 | 3.5 | 0.2 |
| 16384 | token-F1 | 13.8 | 14.3 | −0.5 | 3.5 | −0.1 |
| 16384 | **judge** | **30.0** | **24.0** | **+6.0** | 4.3 | **1.4** |
| 32768 | ROUGE | 15.5 | 9.9 | +5.6 | 3.0 | **1.9** |
| 32768 | token-F1 | 14.2 | 10.2 | +4.0 | 3.0 | **1.3** |
| 32768 | **judge** | **34.7** | **26.3** | **+8.4** | 4.4 | **1.9** |

**Conclusion 1 — the harness is validated at 8k and 16k.** Every n-gram metric there is inside
0.6 SE of HELMET's published value. Data construction, prompt, truncation, generation settings and
metric code agree with upstream.

**Conclusion 1b — 32k does NOT reproduce, and the cause is unresolved.** We read +5.6 ROUGE
(1.9 SE) and +4.0 token-F1 (1.3 SE) above HELMET. What is ruled out: the run needed **zero**
document truncations at 32k (the driver reports the count and printed none), because Qwen2's
tokenizer is more compact on English than Llama-2's, so a 32768-Llama-2-token context fits inside
Qwen2-7B's 32768 window with room for generation — our model saw the full context, untruncated.
The example set is also very unlikely to differ, since the same 100 (document, question) pairs feed
all three lengths and 8k/16k both match.

What remains: HELMET's own sheet shows Qwen2-7B-Instruct falling off a cliff at 32k (ROUGE
17.2 → 14.4 → **9.9**) while other models in the same table rise with length. Our run shows no such
cliff (19.5 → 15.2 → 15.5). A collapse confined to one model at exactly its context limit looks
more like a context-window edge effect in the upstream run than a property of the model. **Treat
HELMET's 32k Qwen2-7B point as suspect rather than treating ours as wrong — but do not claim either
without a direct check.** This does not affect anything downstream: our ladders top out at 32k on
Qwen3.5, a different model with a different window.

**Conclusion 2 — the local judge is systematically generous.** Qwen3-8B scores the SAME generations
**+12.6 / +6.0 / +8.4** points above GPT-4o at 8k/16k/32k (3.2 / 1.4 / 1.9 SE) — generous at every
length, mean offset ≈ +9 points. That is a bias, not noise. Therefore:

* Judge scores are valid for **relative** comparisons — dense vs chunked, rung vs rung, v1 vs v2 —
  where a consistent offset cancels.
* Judge scores are **NOT comparable to HELMET's published judge column**. Reading our 31.3 against
  their 19.7 would suggest we beat Qwen2-7B-Instruct by 60%, while the n-gram metrics show we are
  measuring the same thing they are.
* Every judge number must be published with the judge model named and this offset cited. A future
  swap to a stricter or larger judge must re-run this calibration before its numbers are mixed with
  these ones.
* The offset points the wrong way for the v1 verdict, which makes that verdict safe: the v1
  narrativeqa checkpoint scored 1.0/100 on the *generous* judge, so GPT-4o would score it lower.

## 7c. A judge bug that produced a confident, fabricated number — and the guard against it

The first summarization judge run reported 12.1 / 13.9 / **70.8** at 2k / 4k / 8k. The jump was not
a finding. Per-example inspection:

| rung | reported judge-F1 | examples actually judged | unparseable |
|---|---|---|---|
| 2048 | 12.121 | **3** | 497 |
| 4096 | 13.889 | **3** | 497 |
| 8192 | **70.751** | **2** | 498 |

Every number was an average over two or three of 500 examples. Cause: the key-point prompt asked
for an exhaustive decomposition and produced **26.3 key points per reference**; HELMET's own shipped
sets are ~5–7. The recall rubric asks the judge to reason step-by-step over every key point, which
overran the 1024-token generation budget before any JSON was emitted, so `parse_json` found nothing
and the example was dropped. The aggregate then averaged whatever survived and printed it with no
indication that 99.6% of the eval set was missing.

Three fixes:
1. `--max-keypoints` (default 10), applied to newly generated sets **and** to anything loaded from
   cache, so a stale 26-point cache cannot re-introduce the failure.
2. `--judge-max-tokens` now defaults per task — 1024 for helmet_qa, **3072** for helmet_summ.
3. **`MIN_JUDGED_FRACTION = 0.8`.** If fewer than 80% of examples yield a parseable judgement,
   `metric_value` is set to `None` and an `invalid_reason` is written into the result JSON. A metric
   computed over 0.4% of an eval set is worse than no metric, because it is indistinguishable from
   a real one.

The QA judge was unaffected (0 unparseable at every rung) — its rubric emits a two-field JSON after
short reasoning, well inside the budget.

## 7d. v2 datasets as built (all measured, not assumed)

**narrativeqa**, eval_size 500 per rung, answer coverage (the achievable ceiling):

| rung | v1 | v2 `head` (HELMET rule) | v2 `evidence` (lexical locator) |
|---|---|---|---|
| 2048 | 31.8% | 34.7% | **77.8%** |
| 4096 | 56.3% | 48.8% | 79.3% |
| 8192 | 68.8% | 61.2% | 81.9% |
| 16384 | 80.7% | 71.9% | 83.5% |
| 32768 | 84.7% | 80.7% | **85.3%** |

`evidence` has zero head fallbacks at every rung. Training pool: 20k examples (4000 × 5 rungs),
coverage 76.2% → 84.7%, 11–12 head fallbacks per 4000. Note the verbatim `answer_in_context` rate
stays near 30% throughout — NarrativeQA answers are abstractive, so verbatim is a lower bound and
coverage is the honest ceiling. Both are recorded per example; quote both.

**govreport**, median context tokens per rung:

| rung | v1 | v2 `long` (≥32768-token docs) | v2 `wide` (≥8192-token docs) |
|---|---|---|---|
| 2048 | 1556 | 1648 | 1648 |
| 4096 | 3104 | 3696 | 3696 |
| 8192 | 5961 | 7792 | 7792 |
| 16384 | 8765 | **15984** | — |
| 32768 | **8765** (identical to 16k) | **32368** | — |

`long` is eval_size **298** (⚠ below the 500 floor, binomial SE up to ±0.029 — quote inline) and is
100% genuinely truncated at every rung; documents come from validation+train with their hashes
reserved in `reserved_doc_sha1.json` so the SFT pool can exclude them. `wide` is eval_size 500,
capped at 8k, validation only.

The v1 ladder's 16k and 32k columns landing at rouge1_f 0.5553 and 0.5564 — a 0.001 difference — is
the metric-side confirmation that those two rungs were the same inputs.

## 7e. v1 checkpoint results, measured correctly (dense arm, eval_size 500 per rung)

**narrativeqa** (`ctc-4b-helmet_qa-full`, v1 eval data):

| rung | token_f1 | judge | answer-in-context (the ceiling) |
|---|---|---|---|
| 2048 | 0.0329 | 1.00 | 9.5% |
| 4096 | 0.0364 | 1.40 | 18.0% |
| 8192 | 0.0392 | 1.07 | 24.5% |
| 16384 | 0.0397 | 1.40 | 30.0% |
| 32768 | 0.0408 | 1.07 | 32.5% |

The ceiling more than triples across the ladder; the model does not move. token_f1 rises 0.033 → 0.041,
which at eval_size 500 is inside 1 SE (±0.009) — statistically flat. This is the whole diagnosis in
one table: a model trained on contexts that usually did not contain the answer learned not to read
them. HELMET's reference models score 11.8–27.3 token-F1 over the same lengths.

**govreport** (`ctc-4b-helmet_summ-full`, v1 eval data), after the ROUGE and judge repairs:

| rung | rouge1_f | judge-F1 | judged/500 | recall | precision |
|---|---|---|---|---|---|
| 2048 | 0.5024 | 27.08 | 437 | 0.398 | 0.660 |
| 4096 | 0.5332 | 33.90 | 443 | 0.477 | 0.705 |
| 8192 | 0.5466 | 37.68 | 459 | 0.515 | 0.735 |
| 16384 | 0.5553 | 38.54 | 462 | 0.533 | 0.740 |
| 32768 | 0.5564 | 39.16 | 459 | 0.531 | 0.756 |

Summarization is healthy and rises with length, driven by recall (0.398 → 0.533) as more of the
report fits. It plateaus at 16k in **three independent measurements** — median context (8765 = 8765),
ROUGE (0.5553 vs 0.5564) and judge recall (0.533 vs 0.531) — because the v1 16k and 32k rungs are the
same inputs. Against HELMET's Multi-LexSum judge-F1 span of 16.4–46.4, and discounting our judge's
~+9 generosity, this lands around 18–30: a normal number, where "ROUGE 0.33" was uninterpretable.

Both assembled into `results/ctc_suite/helmet_v2/helmet_{qa,summ}__full__v1.jsonl`.

## 7f. v2 retrain results — the fix works, but does not close the gap to HELMET

Both arms trained from the repaired pool, 4B, 1 epoch, identical shards, `VARIANT` the only
difference. Evaluated on `helmet_qa_evidence`, eval_size 500 per rung.

| rung | v1 dense token_f1 / judge | **v2 dense** | **v2 chunked** | dense−chunked (judge) |
|---|---|---|---|---|
| 2048 | 0.0329 / 1.00 | **0.0722 / 5.40** | 0.0539 / 3.60 | +1.80 |
| 4096 | 0.0364 / 1.40 | 0.0615 / 4.33 | 0.0570 / 3.53 | +0.80 |
| 8192 | 0.0392 / 1.07 | 0.0651 / 4.40 | 0.0594 / 3.20 | +1.20 |
| 16384 | 0.0397 / 1.40 | 0.0575 / 3.80 | 0.0568 / 2.87 | +0.93 |
| 32768 | 0.0408 / 1.07 | 0.0591 / 4.33 | 0.0588 / 3.33 | +1.00 |

**What the retrain bought.** Repairing the training data roughly **doubles token-F1** (0.033–0.041 →
0.058–0.072) and lifts the judge score **3–5×** (1.0–1.4 → 3.8–5.4). That is a large effect from a
pure data change, and it confirms §3's claim that the unanswerable training contexts, not model
capacity, were holding the task down.

**What it did not buy.** The absolute numbers are still far below HELMET's reference models
(11.8–27.3 token-F1 at these lengths), and remember our judge reads ~+9 points generous, so the true
judge score is lower still. Something beyond the data is limiting this task. The most likely
candidate is visible in the shard stats: the pool carries **164k supervised tokens across 246M total
(0.07%)** — 20k examples × ~7-token answers. One epoch over that is a very thin gradient signal for a
4B model. Worth testing before any further data work: more epochs, a higher LR, or pairing
narrativeqa with a task that supervises more tokens per example.

**Dense vs chunked.** Dense leads at every rung by 0.8–1.8 judge points, and the gap is flat rather
than widening with length — the O(N) shape the plan predicts for this task. Treat the magnitude
cautiously: at eval_size 500 the judge's per-rung noise is comparable to the gap, so the *consistent
sign* across five rungs is the finding, not any single rung's value.

Rows: `results/ctc_suite/helmet_v2/helmet_qa__{full,chunked}__v2evidence.jsonl`.

## 7f. Traps worth remembering

* **`sbatch` snapshots the script at submit time.** Editing a launcher after queueing a job that uses
  it changes nothing for that job. This bit twice tonight: once nearly writing a training pool
  concurrently from two jobs, once re-running a known-broken `add_dummy_visual` invocation.
* **A slurm dependency is not a validation.** A failing step inside an sbatch script still exits 0
  unless the script propagates it; `sacct` reported `COMPLETED 0:0` for a conversion that produced
  zero shards, and `afterok` duly launched training on nothing. Validate the artifact, not the exit
  code — `launch_v2_training.sh` now checks shard count, token count, loss-mask density and marker
  presence before committing GPUs.
* **Split expensive work from cheap work.** Generation (20 min of GPU) died to a missing scoring
  import three separate times until generation was made to persist its output before scoring, and
  the vLLM venv and conda env were each given only the job their dependencies support.

## 8. Reporting rules for these two rows

* Quote the **judge** metric as the headline; keep token-F1/ROUGE alongside for continuity with
  `all_results.jsonl`.
* Print the **answer-in-context rate** next to every narrativeqa number. Without it an
  unanswerable-example floor is indistinguishable from a model failure — which is precisely the
  mistake this record corrects.
* GovReport `long` runs below the 500-example floor: give `eval_size` and its error bar inline.
* The HELMET reproduction uses HELMET's own `max_test_samples=100`; at those values the binomial SE
  is ≈4 points, so it is a sanity anchor, not a precision measurement.
