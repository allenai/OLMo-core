# Contradiction: the eval was never IID with the training data — and that is most of the gap to the paper

Opened 2026-08-11. Companion to `records/paper-v2-todo-status.md` §1 (the FEVER filler rebuild) and
`records/contradiction-data-and-base-hygiene.md`.

**One-line version.** Every contradiction number we have measured — CTC suite *and* results-hub v2 —
scores a model trained on `realistic`-mode gold pairs against an eval built from `both`-mode gold
pairs. They are different generators. Fixing only that, with no retraining and the same checkpoints,
moves dense from **0.559 → 0.946 f1** at n=762 and reproduces the paper's curve.

## 1. What the mismatch is

The contradiction generator (`generate_pubmed_contradiction_data.py`) has a `--mode` flag:

| mode | what the gold pair looks like | measured gold-pair word-Jaccard |
|---|---|---|
| `both` | half simple polarity flip ("reduced"→"increased"), half subtle numeric edit | median 0.375, **mean 0.501**, p90 0.943, p99 1.000, 38.1% of pairs > 0.5 |
| `realistic` | full rephrase, contradiction TYPE assigned per pair, near-duplicate backstop | median 0.306, mean 0.307, p90 0.459, p99 0.500, **0.0% > 0.5** |

`both` is substantially a near-duplicate-detection task: 38% of its gold pairs share more than half
their words, and its p90 is 0.943. `realistic` cannot be solved that way.

**Training is `realistic`. Every eval we run is `both`.**

Provenance, from `s3://ai2-llm/checkpoints/prasanns/single_task_ladders_v2/contradiction/metadata.json`:

    "input_jsonl": [".../single_task_ladders_20k/contradiction/gen_full/
                     contradiction_train_pubmed_realistic_n50-950_k3.jsonl"]
    "cot_mode": "none", "query_position": "both", "num_instances": 20091

That same file is the CTC suite's training data *and* the source of the results-hub 5-task shards, so
this is one mismatch affecting both result sets, not two independent ones.

⚠ **Consequence for the earlier cross-check.** `paper-v2-todo-status.md` treats the results-hub
ladder as an independent confirmation of the CTC clean rebuild. On the FEVER-filler question it is
(different filler pools, different rung construction). On the *mode* question it is not: same
training file, same `both`-mode eval family. Their agreement was two runs sharing one mismatch.

## 2. Outlier has the same disease on a different axis — and it is worse

`single_task_ladders_v2/outlier/metadata.json` → `outlier_wiki100w_contin_n14-220_k3_20000.jsonl`.
Corpus size lines up (train n ∈ [14,220] covers the rungs), and `num_outliers = 3` in both. **M does
not.** `build_v2_outlier_ladder.py` is scale-K by construction — it grows majority topics with n:

| rung (n) | num_categories, TRAIN | num_categories, EVAL |
|---|---|---|
| 22 | median 3.5 (max 5) | 3 |
| 55 | median 3.5 (max 5) | 7 |
| 110 | median 4.0 (max 6) | 13 |
| 220 | median 6.5 (**max 10**) | **25** (min 23, max 28) |

At the top rung the eval poses a 25-category problem where training's maximum, over the sample, was
10. M is precisely the axis the O(NM) claim rests on, so the outlier ladder's decline conflates
growing N with an M the model never trained on. **Not yet fixed.**

(Train stats: 373-example head sample, n interleaved 14–220. Eval stats: truncated reads,
208/83/41/20 examples per rung. The n=220 eval row is the thinnest but its range 23–28 is tight.)

## 2b. The full 5-task mix audit (2026-08-11) — 2 of 5 broken, 3 clean

Every component of the canonical 700M 5-task mix, checked against the eval it is scored on. Weights
are the dense launcher's `_W` (sum 8.2).

| task | weight | train source | eval | verdict |
|---|---|---|---|---|
| contradiction | 2.9 (35%) | `..._realistic_n50-950_k3.jsonl` | `..._both_n{100,190,385,765}_k3` | ❌ **OOD** — different perturbation generator |
| outlier | 1.5 (18%) | `outlier_wiki100w_contin_n14-220_k3_20000.jsonl` | `outlier_wiki100w_n{22,55,110,220}_k3_eval_600` | ❌ **OOD** — M axis (§2) |
| rerank | 1.5 (18%) | `msmarco_trainhn_train_k20-315_20000.jsonl` | `msmarco_trainhn_eval_k{20,100}_500` | ✅ clean |
| oolong | 1.3 (16%) | `oolong_ladder_train_combined.jsonl` | `oolong_test_synth_ctx*_spliteval` | ✅ clean |
| nq | 1.0 (12%) | `nq_train_k25-202_clean.jsonl` (p10) | `nq_validation_k{20,100,200}_600` | ✅ clean |

⇒ **~53% of the training mix by weight was OOD for its own eval**, and contradiction — the single
most upweighted task — was the worst of it.

Measured signatures for the three that passed, so nobody re-opens them:

- **nq**: hard-negative fraction 0.100 (train) vs 0.105 (eval). Both are the p10 pipeline; the eval
  was NOT built with the old 98%-hard negatives. Hypothesis refuted. Residual: eval k=20 sits just
  below the train minimum k=25.
- **rerank**: hard-negative fraction 0.100 vs 0.105. Same builder lineage, eval k inside train range.
- **oolong**: built by *two different scripts* (`generate_oolong_ladder_data.py` for train,
  `generate_oolong_data.py` for eval) yet distributionally identical — same set of all 10 source
  datasets with no set difference either way, matching task_group and answer_type mixes, same schema
  and query template, item density 0.0187 vs 0.0206, num_items 145 vs 156 in the matched context
  band. Two generators, one distribution. Hypothesis refuted. (Separate issue: the train file
  predates 2026-07-26 so it may carry the `--item-regex` leak bug — a leak, not an iid problem.)

**Outlier fix — SHIPPED (data), 2026-08-11.** `generate_wiki_outlier_data.py` gained
`--majority-mode articles` (default-off), which fills the corpus with whole articles of
`U[--min-run, --max-run]` chunks until full and lets K emerge — *literally the rule
`build_v2_outlier_ladder.py` uses*, rather than fitting `--chunks-per-article` to the eval's observed
K. Sharing the construction is the point: re-deriving it through a second mechanism is how this class
of bug arises. Rebuilt 20k file at `mooney:/data/prasann/outlier_iid/outlier_iid_n14-220_k3_20000.jsonl`,
audited on 4000 examples:

| rung | train K (min–max) | eval K (min–max) | train docs/maj | eval docs/maj |
|---|---|---|---|---|
| 22 | 3 (2–6) | 3 (3–5) | 8.3 | 9.5 |
| 55 | 7.0 (5–10) | 7 (5–11) | 8.7 | 8.7 |
| 110 | 13.0 (9–18) | 13 (10–17) | 8.8 | 8.9 |
| 220 | 24.0 (19–29) | 25 (23–28) | 8.9 | 9.0 |

against the old file's K = 2–10 (median 6.5) and ~40 docs/topic. Residuals: train K dips to 2 at
n=22 where the eval floor is 3, and docs/maj is 8.3 vs 9.5 there. Tooling: `debug/outlier_iid_rebuild/`.
**Retraining on it has NOT been done** — the data is the deliverable so far.

Shards: `mooney:/data/prasann/outlier_iid/tokenized/` — 20,000 instances, 0 skipped, 340.9M tokens,
median_len 17,014 (original: 19,981 / 19 skipped / 340.1M / 16,987). Drop-in for the hub lineage
(Qwen3-0.6B tokenizer, eos 151643, 40960 window, query_position both).

⚠ **BEFORE EVALUATING ANYTHING TRAINED ON THIS: raise outlier's generation budget.** The outlier
target enumerates EVERY majority topic ("Most passages are about *A, B, C, …* and the outliers are
about *X*. Outlier documents: [i], [j], [k]") — so answer length scales with K, and the doc IDs the
grader parses come LAST. That is why `num_loss_tokens` doubled (1.09M → 2.20M, 55 → 110 per
example): a consequence of the M fix, not a defect. Measured answer tokens vs the eval's
`max_new_tokens=200` (`eval_lc_native.py`, LADDERS entry `("outlier", _eval_outlier, "f1", 200)`):

| n band | old target | rebuilt target |
|---|---|---|
| ~22 | med 34, max 60 | med 43, max 49 |
| ~55 | med 40, max 59 | med 68, max 80 |
| ~110 | med 45, max 71 | med 95, max 125 |
| ~220 | med 63, max 102 | **med 160, max 182** |

At the top rung the rebuilt target needs ~160 tokens median against a 200 ceiling, on an 8-example
sample — across 20k, examples WILL exceed it, truncating before the IDs and scoring ~0. That reads
as "the rebuild broke outlier" or "long-context collapse". Use 512 for models trained on this data.
Raising it globally would also affect existing runs (it can only help a truncated model, but gives a
rambling one more room), so verify before touching published numbers. Same family as
`goldgrad-eval-maxlen-truncation-bug`.

## 3. Building an IID eval — and the flag that had to be recovered from the data

The generator gives a disjoint train/eval split for free when both come from ONE run (a single
cursor walks the shuffled pair pool, train first, eval second). **The training run asked for none**
— its log shows `TAG=full NUM_TRAIN=28000 POOL=40000 SEED=42` and `Building eval n50-950 k3: 0it` —
and it exhausted its pool (28,000 requested → 20,114 produced, ~28% lost to the validity filter).

Two things do not survive in any log and had to be recovered:

- **Generator model = `Qwen/Qwen2.5-7B-Instruct`, served locally by vLLM on port 8766.** Evidence:
  `gen_full/vllm_serve.log` beside the training file. NOT gemini-2.5-flash (the script default).
  Generating the eval with gemini would have introduced a fresh mismatch.
- **`--max-overlap 0.5`.** Recovered from the data, not a log. The training file's gold-pair Jaccard
  has p99 = 0.500, max = 0.500 and *nothing* above — a hard ceiling at the cap value, which is the
  fingerprint of the realistic-mode near-duplicate backstop. A first attempt at the default 1.0
  produced median 0.366 / mean 0.390 / max 1.000 with 20.6% of pairs above 0.5, i.e. including
  near-verbatim "contradictions" — closer to the `both` ladder it was meant to replace than to
  training. That run was discarded. `max_overlap` DROPS the offending pair (no retry), so the cap
  acts as pure truncation and reproducing it reproduces the distribution.

Build: `debug/ctc_contra_iid_eval/gen_iid_eval.sbatch` (mooney, 1 GPU, ~9 min).

**Disjointness is enforced, not assumed** (`filter_disjoint.py`). A separate run re-shuffles, so it
can redraw claims the training run already consumed. Measured: **21.3% of freshly generated eval
examples had a gold sentence the model was trained to pair** and were dropped. This is the same
contamination that makes the pre-existing `contradiction_eval_pubmed_realistic_n100_k3.jsonl`
unusable as a control (it shares gold with training, and is 300 examples, not 500).

Validation of the final ladder:

| | median | mean | p90 | p99 | max | >0.5 |
|---|---|---|---|---|---|---|
| train (target) | 0.306 | 0.307 | 0.459 | 0.500 | 0.500 | 0.000 |
| **iid ladder** | **0.324** | **0.317** | **0.462** | **0.500** | **0.500** | **0.000** |
| old `both` ladder | 0.388 | 0.501 | 0.943 | 1.000 | 1.000 | 0.381 |

`eval_size` 500 every rung, parse_rate 1.00, zero out-of-range gold, and the same 500 examples
(identified by gold *sentences*, not indices) present in the same order at every rung. n at
92/187/379/762 is identical to `contradiction_clean`, so those four rungs isolate the mode as the
only change. The bottom rung moves n=44 → 56 because 44 sat below the training minimum of 52.

## 4. Results — dense (Qwen3.5-4B, `ctc-4b-contradiction-full`, nothing retrained)

| n | **iid f1** | iid EM | both-mode f1 | both-mode EM |
|---|---|---|---|---|
| 56 / 44 | **0.9895** | 0.962 | 0.843 | 0.560 |
| 92 | **0.9843** | 0.948 | 0.803 | 0.504 |
| 187 | **0.9760** | 0.924 | 0.744 | 0.454 |
| 379 | **0.9652** | 0.890 | 0.664 | 0.370 |
| 762 | **0.9463** | 0.830 | 0.559 | 0.270 |

eval_size 500, parse_rate 1.00, SE ≈ ±0.010.

- **Dense does not collapse with corpus size.** 13.6× more documents costs 0.043 f1. The both-mode
  ladder lost 0.284 over the same span. The "graceful decline 0.664 → 0.559" that replaced the
  contaminated "collapse 0.619 → 0.335" is *itself* mostly mismatch. The real curve is 0.965 → 0.946.
- **The paper's dense points sit on this curve.** Paper N=100: f1 0.982 / EM 0.945 — here n=92 gives
  0.9843 / 0.948. Paper N=500: 0.962 / 0.885 — here n=379 gives 0.9652 / 0.890.
- Therefore the training differences catalogued in `paper-v2-todo-status.md` (LoRA-per-N specialist
  vs one joint full-FT model, 3 epochs vs 1, 2k examples vs 20k over n∈[50,950]) do **not** explain
  the gap to the paper at these corpus sizes. The eval did.
- The "realistic is just intrinsically easier" reading is disfavoured: both *matched* pairings
  (realistic→realistic here, both→both in the paper) land at ~0.98, produced by different recipes;
  only the crossed pairing is low. The untested cell is a both-trained model on a realistic eval.

## 5. Chunked — IN PROGRESS, and it does not follow dense

n=56: **f1 0.8613 / EM 0.624** (both-mode at n=44: 0.402). Higher, but the paper reported chunked
EM 0.908 at N=20 and 0.838 at N=100 — this is 0.624 at n=56, well below.

Dense reproduces the paper once the eval is iid; chunked does not. That asymmetry is what you would
expect if the paper's chunked arm was **under-masked**: its runs used `backend: chunked-sdpa` on a
75%-GDN hybrid, where the 4D mask does not reach the GDN layers — forbidden for hybrids per
CLAUDE.md. Independent evidence, not proof. Re-scoring the paper's chunked checkpoint through the
in-tree hybrid-aware vLLM patch would settle it.

Remaining rungs are running (`eval_iid_ladder.sbatch`, MODE=chunked); the dense–chunked gap is the
actual O(N²) claim and is **not yet answerable**.

## 6. What this does not fix

- **Nothing has been RETRAINED.** Both fixes are data-side. Every published number still comes from
  a model trained on the old mix, and the outlier rebuild only pays off once something trains on it.
- Every published contradiction number — CTC grid, results-hub v2, the paper's own figures for our
  runs — is on the `both` ladder and is a cross-pairing measurement.
- The n-axis mismatch is only partly addressed: training samples n ~ U[50,950] while the eval sits at
  5 points. All rungs are now inside the support, but a per-rung-specialist comparison is still a
  different experiment.

## Artifacts

- `debug/ctc_contra_iid_eval/` — `gen_iid_eval.sbatch`, `filter_disjoint.py`, `audit_iid_rungs.py`,
  `stage_rungs_to_s3.sbatch`, `eval_iid_ladder.sbatch`
- rungs: `mooney:/data/prasann/contra_iid_eval/rungs/`, staged to
  `s3://ai2-llm/checkpoints/prasanns/_transfer/contra_iid_rungs/` and
  `cubbins:/data/prasann/ctc_suite_staged/eval_rungs/contradiction_iid/`
- results: `cubbins:/data/prasann/ctc_suite_vllm_results{,_chunked}/contradiction_iid/`
- provenance: `disjointness_report.json`, `iid_rung_audit.json` (both staged to S3 alongside)
