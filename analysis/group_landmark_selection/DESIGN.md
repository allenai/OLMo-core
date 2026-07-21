# GQA-aware landmark selection (`group_landmark_selection`)

## Problem

`FastCompressiveLandmarkAttention` / `DocumentCompressiveLandmarkAttention` already support GQA
(`n_kv_heads < n_heads`) in the sense that K/V projections and the KV cache use `n_kv_heads`, but
GQA is implemented purely as a cache/compute-expansion trick: `repeat_kv` duplicates each KV group's
K/V across its `n_rep = n_heads // n_kv_heads` query heads *before* any landmark scoring runs
(`landmark_fast.py:884-887`, `__init__.py:1569-1574`). From that point on the landmark math treats
every (duplicated) query head independently.

That means hard top-k landmark retrieval at decode time (`_compressive_decode_probs`,
`landmark_compressive.py`) computes `lm_scores.topk(top_k, dim=-1)` **per query head**. Two heads in
the same KV group share identical (repeated) landmark keys but have different queries, so they can
retrieve *different* blocks. This silently defeats the point of GQA at decode time: the whole reason
to share KV across a group is to share the memory traffic of reading it, but if each head in the
group retrieves a different block set, decode still has to touch the union of every member's chosen
blocks.

## Methods considered

All operate on `lm_scores` of shape `(B, H, 1, n_lm)` (`H = n_heads`, already expanded), reshaped to
`(B, n_kv_heads, n_rep, 1, n_lm)` for aggregation, before `topk`:

1. **Mean** of the `n_rep` heads' scores. No new parameters, cheapest, differentiable.
2. **Max** (or a smooth logsumexp) over the group. "If any head wants this block, keep it" — more
   inclusive than mean, still parameter-free.
3. **Pooled group query**: mean-pool the `n_rep` query vectors *before* the QK dot product, scoring
   once per group instead of `n_rep` times. Cheaper (fewer dot products actually needed for
   *selection*), but changes what "score" means, so it likely needs fine-tuning rather than being a
   drop-in eval-time change.
4. **Learned aggregation**: a small softmax-weighted combination of the `n_rep` scores. More
   expressive than mean/max, but requires new trained parameters.
5. **Vote / Borda count**: count how many heads independently rank a block in their own top-k.
   Robust to score-scale differences but a step function (not differentiable) — inference-only at
   best.

**Implemented: 1 and 2** (`group_landmark_selection = "mean" | "max" | None`). They're free of new
parameters, work on an existing checkpoint with no retraining, and are cheap to A/B directly. 3-5 are
documented here as follow-ups if 1/2 turn out to hurt eval quality — no need to build them
speculatively.

## Why inference-only

The codebase already establishes the convention that hard top-k landmark retrieval is an eval/decode-
only mechanism: training and prefill always use the dense soft gate over every block (see the
`FastCompressiveLandmarkAttention` module docstring, `landmark_compressive.py:30-37`), and top-k
(plus the `nonselected_landmark_mass` fudge) is a decode-time approximation deliberately kept out of
the training loss. `group_landmark_selection` slots into the same convention:

* it's gated behind `_eval_top_k` / `set_landmark_eval_decode`, exactly like `top_k` itself;
* only the **selection** (which blocks are eligible) is shared across a group — the gate softmax and
  within-block softmax that actually weight the output still read each head's own (real, not
  aggregated) scores, so the per-head attention output for whichever blocks end up selected is
  unchanged;
* the systems payoff (shared KV bandwidth across a group) only exists at decode; training's dense
  forward already attends every block per head regardless of this setting, so there is nothing to
  gain (or change) by touching training;
* it requires zero kernel/backward changes and zero retraining — an existing checkpoint can be
  A/B'd directly, which is the fast way to find out whether losing per-head-independent retrieval
  costs anything before investing further.

## Implementation

* `FastCompressiveLandmarkAttention._group_landmark_scores` (`landmark_compressive.py`) does the
  reshape/aggregate/broadcast; called from `_compressive_decode_probs` only for the `topk` ranking,
  not for the gate/within-block softmax.
* `group_landmark_selection` constructor kwarg on `FastCompressiveLandmarkAttention` and
  `DocumentCompressiveLandmarkAttention` (the latter borrows the Fast class's decode methods via
  class-attribute assignment, same pattern as `nonselected_landmark_mass`), plus an eval-time
  override on `set_landmark_eval_decode`.
* `AttentionConfig.group_landmark_selection` (`__init__.py`) threads it through config-driven builds,
  restricted to the two compressive attention types (mirrors `nonselected_landmark_mass`'s
  validation).
* `GenerationConfig.landmark_group_selection` (`generate/generation_module/config.py`) threads it
  through `TransformerGenerationModule._set_landmark_eval_decode` for eval-time sweeps without
  rebuilding the model.

### Bug found and fixed en route

`FastCompressiveLandmarkAttention.set_landmark_eval_decode` used a zero-arg `super().set_landmark_eval_decode(...)`
call. That method is shared onto `DocumentCompressiveLandmarkAttention` via class-attribute assignment
(not inheritance — the two classes are in unrelated hierarchies), and zero-arg `super()` closes over
`__class__ == FastCompressiveLandmarkAttention` at compile time, so it raised `TypeError: super(type,
obj): obj must be an instance or subtype of type` whenever called with `self` bound to a
`DocumentCompressiveLandmarkAttention` instance. This was a **pre-existing, 100%-reproducible bug**
(not specific to `group_landmark_selection` — it broke `nonselected_landmark_mass` overrides too),
just never exercised by an existing test. Fixed by calling the known leaf implementation explicitly:
`FastLandmarkAttention.set_landmark_eval_decode(self, ...)`.

## Eval plan

Checkpoint: `/weka/oe-training-default/ai2-llm/checkpoints/prasanns/q4b-compressive-5task-32k-nocpt-fixdata/step8550`
(the plain "compressive" variant — `FastCompressiveLandmarkAttention`, not the docchunk/Document
variant, per `variant_from_run_name` inferring `variant=compressive` from the run name).

Compare `group_landmark_selection in {None (baseline), "mean", "max"}` on:

* **RULER** — via `olmo-cookbook-eval` / oe-eval-internal, `--model-args group_landmark_selection=...`
* **HELMET** — via `ai2-helmet`, `OLMO_CORE_LANDMARK_GROUP_SELECTION` env → CLI flag
* **SFT 5-task ladder** (contra/nq/rerank/outlier/oolong) — via
  `src/scripts/train/sft/singletask_ladder/run_q4b_beaker_multirung_eval.py`

See `results/` for baseline numbers already on file for this checkpoint (`ruler_nsm_results.csv`,
`helmet_nsm_results.csv`, `singletask_ladder_sft_5task.csv`); new runs are suffixed to avoid
collision with those.
