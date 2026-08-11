# Shared-corpus ("efficient") v2 eval suites — plan + overnight schedule

Created 2026-08-06. Owner: prasann + Claude. Scope: **EVALUATION ONLY** — no training data, no
shards, no retraining. Every training file stays exactly as it is.

## 1. Goal

Today every v2 eval rung is 500 examples × 500 *independent* corpora. At the 32k rung that means
500 full 32k prefills per (task, arm, ckpt): the measured cost of one such rung is
**8,885 s ≈ 2.5 h on one H200** (`results/ctc_suite/oolong/qwen3.5-4b_full/rung_32768.json`).

Target: **1–10 unique corpora per 500 queries**, so a rung prefills its corpus a handful of times
and reuses the KV across every query that shares it.

Two things must both be true, and the second is the actual deliverable:

1. **Efficiency** — the shared prefix must be a real, byte-identical prefix so a KV cache can be
   reused. Reported as **shared-token fraction**, not corpus count (see §3 note).
2. **Fidelity** — the efficient rung must give the *same score* as the independent rung on the same
   checkpoint. This is what tonight measures at 32k.

## 2. Task set and eval-only structure

The 5 canonical tasks (`src/scripts/train/memexpress/sft_5task/`): **contradiction, nq, outlier,
rerank, oolong**. They split into two families by whether the task has a discriminative query.

### Family A — query-multiplexed (nq, rerank, oolong): 100% shared, no position leak

These have a real per-example query, so the corpus can be **completely** shared: identical
documents, different question. Gold documents sit at arbitrary positions in the shared corpus, so
there is *no* recency shortcut to worry about. This is the clean case.

| task | construction | fidelity risk |
|---|---|---|
| oolong | Pure regroup — the OOLONG source files already carry **25 questions per context** (`oolong_test_synth_ctx32768_spliteval.jsonl`: 400 rows over **16 contexts**). The v2 rung scattered these (500 rows / 215 contexts). Regrouping changes **no bytes**, only row order. | **None.** Content-identical. Doubles as the correctness gate for the cache runner in §5. |
| nq | Pool Q examples' (gold + own hard-negs) into one corpus, pad with filler to the rung's doc count; remap `gold_doc_indices` / `hard_neg_indices` per row. At 32k (~200 docs, 4 docs/query) Q=50 → **10 corpora / 500 queries**. | Distractor mix shifts: other queries' golds are topically unrelated (easier) while each query keeps its own CE-filtered hard negs (the difficulty that matters). Measured. |
| rerank | Same pooling. **MRR@10 needs only the rank of the gold**, so pooled distractors need no CE rescoring — no cross-encoder job. (NDCG would; we report MRR@10, as the existing grid does.) | Same as nq. Measured. |

### Family B — prefix + tail (contradiction, outlier): critical docs at the end

Neither task has a discriminative query — contradiction's `queries` field is literally `[]`, and
outlier's is a fixed generic string — so the only way to vary the answer across queries sharing a
corpus is to vary a per-query **tail**. That is what creates the recency shortcut, and it is the
one real fidelity risk in this whole plan.

**contradiction — one member of each pair in the shared prefix (per user directive).** Each example
has exactly 3 gold pairs, verified disjoint (0/500 examples share a member across pairs). For each
pair we put **one member in the shared prefix and its partner in the per-query tail**. The model
therefore gets at most *half* of each answer from recency and must still search the full corpus for
the partner. Other queries' prefix-resident half-pairs are inert distractors for this query (their
partners live in a different query's tail), so no spurious contradictions are introduced — asserted
by the verifier.

**outlier — tail-only golds, no way around it.** The task is "find the topic with the fewest
documents"; an outlier trio placed in the shared prefix would be an outlier for *every* query
sharing that prefix, so gold-in-prefix is structurally impossible here. Dilution is the only
mitigation, and dilution filler must come from topics that are already well represented in the
prefix — otherwise a filler doc forms its own 1-doc "topic" that is *smaller* than the true trio and
the task becomes ill-posed.

Per-doc topic labels are stripped from the eval files (`title: None`) and the wiki100w article pool
would need a pyserini/Lucene build (a known-hostile dependency — JVM hangs on jsteinhardt compute
nodes, see `[[obliq-gen-infra-jvm-nfs-traps]]`). Instead we **recover labels by TF-IDF clustering**
of a source example's 217 majority docs, and *validate* the clustering against `meta.category_distribution`,
which gives us the exact multiset of topic sizes. Any example whose recovered histogram doesn't
match is skipped rather than guessed at.

### Tail sweep (both Family-B tasks)

Per the answered question, two points, plus controls:

| variant | tail | shared-token fraction | notes |
|---|---|---|---|
| `tail05` | 5% of corpus | ~95% | aggressive |
| `tail25` | 25% of corpus | ~75% | conservative |
| control `guess-tail` | — | — | analytic: score of "answer with random docs from the tail". At tail05 outlier this is high, so a shared-run score near baseline (not near the control) is what makes the variant usable. |
| control `head` | golds in the *first* tail-sized block instead of the last | — | isolates recency from "the golds are in a contiguous block" — run only if `tail05` and the baseline disagree. |

## 3. File format — deliberately unchanged

The shared variants are written in the **exact same schema** the graders already read, plus three
additive fields: `corpus_id`, `shared_prefix_len`, `shared_prefix_sha1`. Each row still carries its
full `documents` list.

Consequences: every existing evaluator, parser and metric works untouched; independent-vs-shared is
a pure data swap (`--eval-jsonl`), so the alignment comparison has no confounds from new inference
code; and a cache-aware runner is a pure optimisation that must reproduce the same numbers.

Invariant, enforced by `verify_shared_corpus.py`: for all rows sharing a `corpus_id`,
`documents[:shared_prefix_len]` is byte-identical and `sha1` matches.

> Note on "1–10 corpora": for Family A that count is literal. For Family B `tail25`, the honest
> number is "N shared prefixes + 500 unique tails", so the plan reports **shared-token fraction**
> as the efficiency metric throughout. `tail25` is a ~4× prefill saving, `tail05` ~20×.

Output root: `/scratch/users/prasann/ctc_suite_staged/eval_rungs_shared/<task>/rung_<N>_<variant>.jsonl`
(same convention as the existing `eval_rungs/`; scripts, logs and reports live in the repo).

## 4. Validation matrix — 32k, Qwen3.5-4B, full-attn checkpoints

All five checkpoints exist node-local and were inventoried this session:

| task | ckpt | node | partition | baseline metric (existing grid) |
|---|---|---|---|---|
| contradiction | `ckpts_4b/ctc-s5-contra-full-4b` | horton | berkeleynlp | set_f1 0.335 @32k |
| outlier | `ckpts_4b/ctc-4b-outlier-full` | horton, mooney | both | set_f1 0.428 @32k |
| oolong | `ckpts/ctc-4b-oolong-full` | sneetches | jsteinhardt | partial_credit @32k |
| nq | `ckpts_4b/ctc-4b-nq-full` | mooney, sneetches | jsteinhardt | gold_id_f1 0.864 @8k |
| rerank | `ckpts_4b/ctc-4b-rerank-full` | mooney, sneetches | jsteinhardt | mrr@10 0.960 @8k |

Checkpoint placement splits cleanly across the two GPU quotas (contradiction+outlier on
berkeleynlp/horton, the other three on jsteinhardt) — both pools can run at once.

**Baselines are re-run this session rather than quoted from the grid.** The grid numbers come from
a different git commit and, for oolong, from a run with `skipped_too_long: 81` (81/500 examples
scored 0 because `--max-length` was undersized — the `[[goldgrad-eval-maxlen-truncation-bug]]`
failure mode). Re-running under one commit with the driver's prompt audit enabled is the only way
the comparison means anything.

**Rungs:** contradiction / outlier / oolong validate at **32k** as asked. nq and rerank have no 32k
rung (their canonical CE-filtered pools cap out at 48 and 100 docs), so they validate at **8k**,
where a baseline exists and a run is ~30–40 min rather than 2.5 h. Worth noting as a side effect:
multiplexing *creates* the 32k rung for nq and rerank for free, because the corpus is now assembled
from many queries' pools — that unblocks two cells the grid has had to leave blank, but it is a
follow-up, not tonight's validation.

**Decision rule.** Binomial SE at eval_size=500 is ±0.021 at f1≈0.7 and ±0.010 at f1≈0.95, and
seed/run variation sits on top of that. A variant is called **aligned** if |shared − baseline| ≤ 2 SE
*and* the sign is consistent across the two tail settings; anything larger is reported as a real
divergence with the direction called out, not averaged away.

## 5. Cache-reusing runner

`src/scripts/eval/ctc_suite/run_shared_corpus_eval.py`: group rows by `corpus_id`, prefill the
shared prefix once per corpus, then for each query run only `tail + query` against the reused KV.

Correctness gate before any speedup number is quoted: run it on the **oolong regroup**, whose
content is byte-identical to the independent file, and require **exactly** the baseline score.
A content-identical variant that changes the score means the cache path is wrong, full stop.

This is staged deliberately: §4's alignment results come from the existing proven native driver and
land regardless of whether the cache runner works. The runner is developed while those evals occupy
the GPUs.

## 6. Overnight schedule

| # | step | where | est. |
|---|---|---|---|
| 1 | builder + verifier; build oolong regroup, nq, rerank variants | CPU | 45 m |
| 2 | **Wave 1**: 5 baselines + oolong/nq/rerank shared | 8 GPUs, both pools | ~2.5 h |
| 3 | contradiction pair-split builder; outlier TF-IDF label recovery + builder | CPU, during wave 1 | 1.5 h |
| 4 | **Wave 2**: contradiction tail05/tail25, outlier tail05/tail25 | horton | ~2.5 h |
| 5 | cache runner + bit-identity gate | CPU/1 GPU, during waves | 2 h |
| 6 | speedup benchmark + report | 1 GPU | 45 m |

Total wall ~6–7 h with the two GPU pools running in parallel.

## 6b. Findings during the build (2026-08-06)

**The staged contradiction "32k" rung is really a ~64k rung.** The driver's prompt audit on
`eval_rungs/contradiction/rung_32768.jsonl` reports **p50 = 63,749 tokens, max = 214,446** against a
32,768 label. This is the downstream consequence of the FEVER-filler calibration already recorded in
`[[contra-fever-filler-leak]]`: the n=1423 ladder was fit against a pool that was 92–99% one-line
Wikipedia claims (~22.8 tok/doc), and real PubMed claims are ~43 tok/doc, so every contradiction rung
in the current grid overshoots its label by ~1.8×. `build_v2_eval_ladders.py` already carries the
corrected `contra_ctc` map (32k → n=762); only the *staged* file predates it.

Handled both ways rather than picking one:
- the mislabeled pair still runs (`full-indep` vs `full-shared-tail05/25`) — both sides are the same
  size, so it is a valid alignment test, just at ~64k rather than 32k;
- a **recalibrated, PubMed-only** ladder was rebuilt this session (`eval500_v2_contra_ctc`, n=762,
  81,250 distinct fillers) and its own indep/tail05/tail25 triple queued (`*-ctc-*` arms). Those are
  the numbers to quote: correct length label *and* free of the FEVER filler contamination.

**rerank needs its `ce_scores` realigned, not dropped.** Pooling documents from other queries leaves
them with no CE score, and dropping the array entirely makes the prompt builder raise
(`NotImplementedError: DEPRECATED binary rerank format`). The correct fix is the format's own
existing affordance: carry each query's own scores across to their new positions and mark foreign
documents `None`, which `_rerank_reference_order` already handles as "unscored random fill". Caveat
to quote with the result: the independent file CE-scores all 70 documents whereas the shared one
scores ~8 (gold + hard negs), so the *reference ranking* is coarser even though MRR@10 — which needs
only the gold's rank — is unaffected.

**oolong needed no new data at all.** Its source split already stores 25 questions per context
(16 contexts at 32k), so the shared file is the same rows in a different order.

### ⚠ The main finding: which tasks can actually be made cheap is the OPPOSITE of the §2 split

A byte-identical *document* prefix is necessary but not sufficient for KV reuse — the prompt
template decides whether that block is still a **token** prefix. These checkpoints are trained with
`query_position="both"` (the converters' own default, `convert_longctx_tasks_to_sft.py:407`;
`segment_prompt_to_chunks` defaults the same way), which renders

    ### Input:\n{questions}\n\n{documents}\n\n{questions}\n\n### Response:\n

So on any task with a real per-example question, the question is emitted **before** the shared
corpus and the corpus stops being a prefix. Measured directly
(`debug/shared_corpus_eval/measure_lcp.py`, longest common token prefix within a corpus group):

| task | reusable token prefix | max prefill saving | why |
|---|---|---|---|
| contradiction | **95.0%** | 20× | `queries` is empty — nothing precedes the documents |
| outlier | **94.7%** | 18.7× | query is one fixed generic string, identical across all rows |
| nq | 0.5% | 1.0× | per-query question precedes the corpus |
| rerank | 0.9% | 1.0× | ditto |
| oolong | 0.1% | 1.0× | ditto |

This inverts §2's framing. **The two tasks that were hard to make shareable are the two that
benefit** — and for the same underlying reason: having no discriminative query is what forced the
prefix+tail construction *and* what leaves the corpus at the front of the prompt. The three
"clean" tasks share 100% of their documents and 0% of their prefill.

Family A's shared files are still worth having — they are what makes a 32k rung possible for nq and
rerank at all (their canonical CE-filtered pools cap at 48/100 docs per query, and a pooled corpus
has no such cap), and they cut unique corpora ~30×, which matters for staging and storage. They
just do not, on their own, make the *forward pass* cheaper.

The one available fix is `query_position="after"`, which makes the corpus a true prefix — at the
cost of an off-distribution prompt. Rather than assume that cost, it is being measured: job 3426850
runs the same 500 nq queries three ways (indep+both = reference, indep+after = format change alone,
shared+after+cache = the configuration that would actually be cheap). An opt-in
`--query-position` flag was added to the evaluator for this; **its default stays `both`** and no
headline number uses anything else unless the probe says the change is free.

### Cache path is verified bit-exact

`debug/shared_corpus_eval/smoke_cache_parity.sh` on 2 nq corpus groups: **16/16 generations
identical** with and without `--shared-corpus-cache`. Note this run exercised the fallback path
(reusable prefix 0.5% < the 64-token floor), so it proves the plumbing and the no-op path; the
reuse path itself is exercised by the contradiction/outlier runs, where the prefix is 95%.

## 7. Known issues inherited, not introduced

- **contradiction filler is 92–99% FEVER/wiki**, not PubMed, in the staged CTC ladder
  (`[[contra-fever-filler-leak]]`, still open). The gold *pairs* are genuine PubMed contradictions —
  verified this session. Because every shared variant is derived from the same rung file, the
  alignment measurement is unaffected; absolute contradiction numbers inherit the contamination
  exactly as the current grid does.
- **oolong eval_size at 32k is 400, not 500** (16 contexts × 25 questions in the eval split; the
  train split's questions are held out and must stay held out). Flagged inline wherever quoted:
  `eval_size=400 (SE ±0.024)`.
