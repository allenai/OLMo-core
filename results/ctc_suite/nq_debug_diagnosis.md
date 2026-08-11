# NQ debug diagnosis (CTC-suite Figure-4)

Status: **CLOSED — root-cause analysis complete, no bug found.** The whole effort pivoted to 4B
for all tasks; nq + qdmatch_nq are now owned by the 4B fan-out coordinator as regular tasks (4B /
20k IS the decisive budget+scale test). The 4 exploratory 0.8B runs I had queued
(3338986/3338987/3338988 on berkeleynlp, 109753 on lambda) were **CANCELLED** on the pivot so they
don't hold queue slots — no 0.8B decisive run was needed once 4B/20k became the plan.

### HANDOFF TO THE 4B COORDINATOR — reusable 20k CE-clean shards already staged
I rsynced the verified-correct 20k CE-clean nq + qdmatch_nq shards to lambda, **overwriting the
stale 2500-example pilot stubs** that were there. Reuse these directly (do NOT rebuild):
- **lambda**: `/accounts/projects/sewonm/prasann/ctc_suite/data/nq_train`
  (`num_instances=19967`) and `/accounts/projects/sewonm/prasann/ctc_suite/data/qdmatch_nq_train`
  (`num_instances=20000`) — both `marker_set=qwen3_5`, verified post-transfer.
- **Berkeley source of truth** (for any node-local staging on jsteinhardt/berkeleynlp):
  `/scratch/users/prasann/ctc_suite_staged/shards/nq_train` and
  `/scratch/users/prasann/ctc_suite_staged/shards/qdmatch_nq_train`.
- Underlying CE-clean NQ jsonl (if a rebuild at a different tokenizer/scale is ever needed):
  `/scratch/users/prasann/nq_p10_20k/nq_train_k25-202_clean.jsonl` (hard_ratio 0.097, CE-filtered).

⚠ These shards were tokenized with the **Qwen3.5-0.8B-Base** tokenizer (vocab 248320,
doc_start_id=248049, doc_end_id=248050, eos=248044). The Qwen3.5 tokenizer is shared across the
0.8B/4B hybrid family, so they are reusable at 4B **as long as the 4B base uses the same Qwen3.5
tokenizer/marker set** — confirm that before reuse (the old `/scratch` q4b base was a different
vocab; the correct hybrid 4B base is the q35-4b on cubbins /data per the session handoff).

## TL;DR

Nothing in the eval harness, the data generator, or the doc-numbering/off-by-one path is broken.
Every place that has burned us before on this exact task (`nq-eval-retrieval-3k-gotcha`,
`contra-outlier-v2-rootcause-fix`, `goldgrad-eval-maxlen-truncation-bug`, thinkstrip, no-cot
full-path) was checked directly against this data and came back clean. The failure signature —
predictions collapse to 2-3 CONSTANT positional answers regardless of the actual gold — is the
signature of **underfitting at the 2500-example training budget on a genuinely hard
discrimination task**, not a harness/data bug: the model learns the output *format* (emit one
`[N]`, or emit ~3 `[q,d]` pairs at roughly the right block boundary) but never learns to condition
the choice on content. The clean A/B is `qdmatch_hpqa` (.985, learns fine) vs `qdmatch_nq` (.037,
collapses) — **identical harness, identical scorer, identical prompt template, only the source
corpus differs** — which rules out a shared eval-code bug outright and points at NQ-sourced
content + budget.

The decisive test (full 20k CE-clean nq + qdmatch_nq, both arms) is queued (see below); it has not
yet produced a single training step, so **this is not yet closed** — it is the standing hypothesis
with everything-else-ruled-out behind it, not a confirmed finding.

## What was checked (Track 2, in order)

### 1. Eval harness
- `run_rung_eval.py` / `eval_lc_native_docchunk.py --task retrieval` (nq's canonical alias) —
  raw eval JSON (`results/ctc_suite/pilot_2k/raw_eval/nq/retrieval/qwen3.5-0.8b_full/rung_2048.raw.json`)
  shows `skipped_too_long: 0` — no maxlen truncation ([[goldgrad-eval-maxlen-truncation-bug]] is
  NOT in play here; the rung-2048 prompt is short and well inside the `rung+2048` max-length guard).
- Dumped and read `rung_2048.generations.json` (500 examples) directly — a well-trained model at
  exactly f1=0 would show empty generations ([[goldgrad-eval-maxlen-truncation-bug]] /
  [[eval-lc-native-nocot-fullpath-bug]] signature); it does NOT. Predictions are syntactically
  valid, non-empty `[N]` ids:
  ```
  idx0 pred=[2]  gold=[9]    idx1 pred=[2]  gold=[2]  (correct, f1=1.0)
  idx2 pred=[12] gold=[10]   idx3 pred=[12] gold=[4]   idx4 pred=[12] gold=[1]
  ```
  Distribution over all 500: **prediction ∈ {[12]×402, [2]×98}** — literally 2 unique values.
  Gold distribution over the same 500 is roughly uniform across ids 1–15 (53,48,40,36,35,35,35,
  30,30,30,29,24,23,17,12 — no dominant id), so the eval SET is not positionally skewed; the
  MODEL is the thing outputting a near-constant answer.
- No thinkstrip issue ([[contradiction-eval-thinkstrip]]): predictions are the raw `[N]` id, not
  an empty string from a newline-stop landing inside `<think>`.
- No doc-ID format mismatch: predicted/gold ids are both small integers in `[N]` bracket form,
  parsed identically by `compute_retrieval_metrics_single`.

### 2. Off-by-one / doc-numbering (the exact bug class that hit contra/outlier, [[contra-outlier-v2-rootcause-fix]])
- Train and eval render documents through the **same** function
  (`src/olmo_core/data/corpus_reasoning_prompts/_data_format.py:_format_documents`, `task in
  ("retrieval", ...)` branch): `doc_id = i + 1` — 1-indexed by render position.
- The answer/label side (`_build_retrieval_ids`, same file) does `f"[{g + 1}]"` over
  `gold_doc_indices` (0-indexed) — i.e. the identical `+1` convention on both sides.
- The eval scorer (`compute_retrieval_metrics_single`,
  `src/corpus_reasoning/eval/evaluate.py:984`) does the same `gold_ids = {g + 1 for g in
  gold_doc_indices}` conversion before comparing to the parsed prediction.
- Train (`convert_unified_to_document_landmark.py` → `segment_prompt_to_chunks` →
  `build_prompt`) and eval (`eval_lc_native_docchunk.py:build_prefill` → the same
  `segment_prompt_to_chunks` → `build_prompt`) call through the **identical single source of
  truth**, both with `query_position="both"` (train's CLI default and eval's hardcoded value
  agree). There is no train/eval divergence in doc numbering or query placement — confirmed by
  reading both call sites, not by assumption.

### 3. Data generator (`generate_nq_training_data.py`)
- `_assemble_example` does `rng.shuffle(tagged)` over `[(gold, "gold")] + hard + pool` before
  assigning positions — gold position is genuinely randomized per example, not fixed by
  construction. No positional leak in the generator.
- **Independently re-audited** the actual CE-clean 20k source file myself (not just trusting the
  earlier session's audit): sampled 2852/19967 examples from
  `/scratch/users/prasann/nq_p10_20k/nq_train_k25-202_clean.jsonl` —
  **mean hard_ratio = 0.0973** (matches the ~0.10 target for the p10 pipeline,
  [[nq-pipeline-10pct-hardneg-cefilter]]), **0% substring false-negatives** in the sample (no
  non-gold doc contains an answer string). This corroborates, from a fresh independent check, that
  the CE-filter genuinely ran and the p10 ratio genuinely holds — it is not a mislabeled/stale file.

### 4. Cross-task control: `qdmatch_hpqa` vs `qdmatch_nq`
This is the strongest single piece of evidence. Both run through the byte-identical qdmatch
harness/scorer/prompt template (`generate_qdmatch_data.py` + the `qdmatch` branches of
`_format_documents`/`_build_task_query` + `_eval_qdmatch`) — the ONLY difference is which
single-query source file (hotpotqa vs nq) it was derived from.
- `qdmatch_hpqa`: f1 **0.985** (learns essentially perfectly).
- `qdmatch_nq`: f1 **0.037** (collapses).

Dumped `qdmatch_nq`'s `rung_2048.generations.json` (500 examples) — same collapse signature as
plain nq, but even more explicit because the prediction is multi-field ([query_id, doc_id] pairs):
```
top predictions (500 total): [[1,11],[2,17],[4,12]] x204   [[1,11],[2,12],[4,17]] x129
                              [[1,11],[2,12],[4,18]] x110   (+ 3 more near-identical variants, x57 combined)
```
The model has learned "predict ~3 pairs, query positions ~{1,2,3,4}, doc positions ~{11,12,17,18}"
— i.e. it learned the STRUCTURAL prior (M queries then N docs in `layout=separate`, so doc ids
start right after the query block, and ~3 pairs are expected) but not which specific query matches
which specific doc. This is the textbook signature of a model that has fit the output-format loss
to near-zero and is sitting at the token-level prior for the content-dependent digits — exactly the
mechanism already documented for the nq_ceconfirm collapse ("CE plateaus ~0.46-0.55 = perfect
format tokens + digit stuck at prior entropy", see [[nq-pipeline-10pct-hardneg-cefilter]]).

Since the harness is byte-identical between the two and only ONE of the two sources collapses,
**this rules out a shared eval-code bug** — whatever is wrong is either (a) something specific to
NQ-sourced text (its gold passages are generic Wikipedia paragraphs that merely contain an answer
string, vs. HotpotQA bridge documents which are much more lexically distinctive/entity-linked and
thus easier to discriminate from distractors), or (b) simply that NQ is a harder discrimination
task and 2500 examples isn't enough signal to move past the positional prior — most likely both,
compounding.

### 5. Loss curve shape
`results/ctc_suite/pilot_2k/curves/nq.json` (312 steps, nq pilot full-attn run): NOT perfectly
flat as the task brief assumed — early-run avg CE ≈ 0.85, late-run avg CE ≈ 0.45, so the model is
learning *something* (mostly format) over the run, it just never breaks through to real
per-example retrieval before running out of the 2500-example/1-epoch budget. Consistent with "not
enough budget for this task," not "broken/no gradient signal."

## What this rules OUT

- Eval max-length truncation (goldgrad-style bug) — `skipped_too_long: 0`, generations non-empty.
- thinkstrip / newline-stop swallowing the answer — predictions are the literal digit, not empty.
- Doc-ID off-by-one (contra/outlier-style bug) — verified both directions of the +1 convention,
  train and eval share one code path.
- CE-filter not actually applied / stale source file — independently re-audited hard_ratio ≈0.097,
  0% false-negatives on a fresh sample.
- A bug shared by the qdmatch harness generally — qdmatch_hpqa on the SAME code learns to 0.985.
- Eval-set positional skew making a constant guess look artificially good — gold ids are
  ~uniform 1–15 across the 500 nq eval examples; a constant-[12] guesser would NOT score 0.074 by
  luck alone if gold were actually spread that evenly (0.074 ≈ chance rate for guessing one fixed
  id out of ~15, which is the exact 1/13 ≈ 0.077 ballpark — consistent, not suspicious).

## What remains open (genuinely unresolved until the 20k run reports)

Whether the **budget** (2500 → 20000, 8x more, matching how hotpotqa went from failing at pilot
scale to f1=0.861 at 20k) is sufficient to move nq/qdmatch_nq off the positional prior, or whether
NQ retrieval among generic Wikipedia distractors is hard enough that even 20k isn't enough and this
becomes a genuine task-difficulty finding for Figure 4 (an O(N) task that, unlike other O(N) tasks
in the suite — niah .988, hotpotqa .861 — doesn't scale cleanly). Both are legitimate outcomes; the
decisive run is the only thing that resolves it.

## Exploratory 0.8B runs (submitted, then CANCELLED on the 4B pivot)

I had queued four exploratory 0.8B runs on the verified 20k CE-clean shards to test the
budget-fixes-it hypothesis. When the effort pivoted to 4B-for-all-tasks (the 4B/20k run being the
real decisive test, owned by the 4B coordinator), these were **cancelled** to free queue slots —
they never reached a training step. Recorded here only for provenance:

| job id | run name | compute | task | variant | seq_len | final state |
|---|---|---|---|---|---|---|
| 3338986 | ctc-s5-nq-full-08b-loc | berkeleynlp/lorax | retrieval (nq) | full | 33280 | CANCELLED (was PENDING) |
| 3338987 | ctc-s5-nq-cmix-08b-loc | berkeleynlp/lorax | retrieval (nq) | chunked-mix | 33280 | CANCELLED (was PENDING) |
| 3338988 | ctc-s5-qdmatchnq-full-08b-loc | berkeleynlp/lorax | qdmatch (qdmatch_nq) | full | 33792 | CANCELLED (was PENDING) |
| 109753 | ctc-s5-qdmatchnq-cmix-08b-lambda | lambda | qdmatch (qdmatch_nq) | chunked-mix | 33792 | CANCELLED (was PENDING) |

The intended config (for the 4B coordinator to mirror at 4B): `EPOCHS=1`, `ACT_CKPT=full`
(single-GPU + ~33k seq needs activation checkpointing), `GLOBAL_BATCH=8`, `MICRO_BATCH=1`,
`LR=5e-5` — unchanged from the known-good hotpotqa/contradiction 20k configuration. seq_len sized
to each shard's `max_example_len` (nq 32904 → 33280; qdmatch_nq 33398 → 33792).

## Eval-size note

All eval numbers cited above (nq f1=0.074, qdmatch_nq f1=0.037, qdmatch_hpqa f1=0.985) are on
**eval_size=500** — meets the ≥500 directive, no flag needed. Binomial SE at f1≈0.07 with n=500 is
≈±0.011; the near-constant-prediction signature is visible in the raw per-example predictions
directly (2-3 unique values out of 500), which is a much stronger signal than the aggregate f1
alone and does not depend on the error bar.

## Next steps (now owned by the 4B fan-out coordinator)

The 4B/20k nq + qdmatch_nq runs ARE the decisive budget+scale test. When they land:
1. Reuse the shards already staged (see the HANDOFF box at the top) — do NOT rebuild.
2. Watch `train_ce` per-step and capture the full curve into the result JSON (per
   [[record-loss-curves]]); watch specifically whether CE breaks BELOW the ~0.45 floor seen at the
   2500-example 0.8B pilot — that is the tell for "budget+scale fixes it" vs "still stuck."
3. Once a checkpoint exists, dump generations and check prediction diversity (unique-value count)
   directly, not just the aggregate f1 — that was by far the most legible signal in this whole
   investigation (2 unique preds / 500 = collapse; a healthy model shows hundreds).
4. If the 4B/20k run ALSO collapses to a small constant set of predictions: this is a genuine
   Figure-4 finding (an O(N) task that does not scale like its siblings niah/hotpotqa) — report
   with the full loss curve + generation samples, not the bare metric.
