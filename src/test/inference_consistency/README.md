# Inference-consistency tests

Does the cross-entropy a model assigns to a continuation under a **single teacher-forced forward
pass** match the distributions it produces for that same continuation **while generating it**?

Those are supposed to be the same conditional distribution computed two ways. Where they disagree,
one of the two numbers a run is judged on is describing a model that does not exist — either the loss
curve is measuring a function the eval harness never serves, or the eval harness is serving a
function that was never trained.

## Why the decode loop is forced

If generation free-runs, its context diverges from the forward pass's at the first token where the
model's argmax differs from the gold continuation, and every later position is conditioned on
different history — so a mismatch no longer tells you the code paths disagree. The harness therefore
patches token selection: the real logits are recorded at each step, then the **gold** token is fed in
regardless of what the model would have picked. Everything else in `generate_batch` — prefill, KV
cache, landmark prompt construction, top-k retrieval — runs untouched.

## What each variant owes

Not all five owe the same agreement, and asserting otherwise would either fail on correct code or
widen the tolerance until it stops catching real regressions.

| variant | expect | why |
| --- | --- | --- |
| `dense_nocache` | identical | No cache at all — the harness's own control. Runs on CPU. |
| `dense` | identical | Prefill + cached decode must reproduce the batched forward. |
| `document_chunked` | identical | Generated tokens are FREE; roles are rebuilt from the token stream on both paths. |
| `summary_token` | identical | Serving mask is fixed per-run, so both paths should agree. |
| `sparse_landmark_no_topk` | identical | Top-k off ⇒ dense gating over all past blocks, as the forward does. Runs on CPU. |
| `compressive_landmark_no_topk` | identical | Same, for the compressive variant. |
| `sparse_landmark` | **gap** | Production config: hard top-k retrieval runs on decode steps only. |
| `compressive_landmark` | **gap** | Same, plus a separate decode routine for the compressive summary term. |

`gap` variants are not asserted equal — the divergence is recorded and held under a budget, because a
decode path that has stopped attending to the prompt also produces a "gap".

## The landmark gap decomposes into exactly two causes

`GenerationConfig.landmark_top_k_fraction` **defaults to 0.1**, so landmark eval runs top-k retrieval
unless you turn it off. Measured on the tiny sparse-landmark model (`block_size=16`, 51-token
landmark prompt):

| config | max KL | top-1 agreement |
| --- | --- | --- |
| top-k ON (default), 12 gold, in-block | 6.8e-03 | 0.25 |
| **top-k OFF, 12 gold, in-block** | **4.2e-07** | **1.00** |
| top-k OFF, 18 gold, past final block | 1.7e-02 | 0.78 |

1. **Hard top-k retrieval** — applied on single-query decode steps, never during the batched
   prefill, so decode attends to a strict subset of the blocks the forward gates over. Turning it
   off closes the gap completely: the cached decode reproduces the eager forward to float32 noise.
   This is the assertion `sparse_landmark_no_topk` makes, and it is the first real correctness check
   on the landmark KV-cache decode — with the default config every landmark comparison was
   contaminated by top-k and could only ever be *measured*.
2. **Landmark drift over generated tokens** — landmark slots are fixed by absolute position
   (`pos % block_size == block_size - 1`), and `generate_batch` never inserts landmark tokens among
   generated tokens (its decode modes treat the continuation as one growing local block). Once the
   continuation reaches the next landmark slot, the eager forward reads that generated *content*
   token as a landmark and decode does not. Survives with top-k off; bounded by
   `landmark_drift_gold_budget()`.

   The break is one step *after* the slot — a query at the slot still reaches its own block locally;
   only queries looking back *through* the spurious landmark diverge. Verified across both decode
   modes, which have different prompt lengths and so different predicted break points:

   | decode mode | P | first landmark slot among generated | predicted break | observed |
   | --- | --- | --- | --- | --- |
   | `extend_last_block` | 51 | pos 63 | gold=15 | gold=15 ✓ |
   | `generation_only` | 64 | pos 79 | gold=18 | gold=18 ✓ |

   This also rules out the simpler "query crossed a block boundary" explanation: under
   `generation_only` queries enter a new block at gold=2, yet agreement holds through gold=17.

   `eval_lc_native_docchunk.py` already works around this in its own decode loop by feeding a real
   landmark token after every `mem_freq` generated tokens, explicitly *"so the periodic `is_mem`
   structure … matches"*. `generate_batch` has no such injection — so for landmark models evaluated
   through it, a long generation drifts from the periodic structure the model trained under.

### How often does drift actually bite?

Measured on 180 real generations pulled from completed `fast_landmark` / `compressive_landmark`
eval jobs (`mem_freq=63` ⇒ `block_size=64`, `decode=extend_last_block`, the default). Under
`extend_last_block` the prompt keeps a partial trailing block, so the distance to the first landmark
slot a generated token occupies is ~uniform on `{1..64}` across prompts; `P(drift)` is the resulting
per-generation probability, and "always" counts generations longer than any possible budget.

| task | n | median tokens | p90 | max | P(drift) | always drifts |
| --- | --- | --- | --- | --- | --- | --- |
| outlier_review | 18 | 179 | 200 | 200 | 83% | 67% |
| rerank | 18 | 175 | 256 | 256 | 100% | 100% |
| oolong | 18 | 155 | 200 | 200 | 67% | 67% |
| outlier | 24 | 131 | 200 | 200 | 91% | 75% |
| contra_fever | 18 | 123 | 400 | 400 | 78% | 67% |
| contra | 24 | 82 | 400 | 400 | 70% | 50% |
| fiqa | 18 | 64 | 67 | 67 | 69% | 11% |
| scifact | 18 | 64 | 64 | 64 | 65% | 0% |
| nq | 24 | 34 | 64 | 65 | 50% | 0% |
| **ALL** | **180** | **64** | **256** | **400** | **74%** | **48%** |

⚠ `eval_size=180` generations, and they are the "first 6 sampled" lines each job logs rather than a
random draw — treat the per-task rates as indicative, not precise. The direction is not in doubt:
**roughly three quarters of real landmark generations run past a landmark slot, and about half are
long enough that no prompt alignment could avoid it.** Short-answer tasks (`nq`) are the least
affected; long-form tasks (`rerank`, `outlier_review`) essentially always drift.

Switching to `landmark_decode_mode="generation_only"` changes this: it pads the prompt to end on a
real landmark, so the budget is a full `block_size + 1 = 65` tokens every time instead of a coin
flip — which would move `nq`, `scifact` and `fiqa` to near-zero drift.

`test_landmark_gap_decomposes_into_topk_and_block_drift` asserts this decomposition, so a genuine
KV-cache regression cannot hide inside a gap everyone had learned to expect.

`train_serve_gap_test.py` is the other half: both paths are teacher-forced forwards differing only in
`model.training`, which isolates *serving the wrong mask* (the summary-token failure mode) from
*computing the mask wrongly*.

## Running

```bash
# CPU: the control, the sparse-landmark gap, and every train/serve test
pytest -v src/test/inference_consistency/

# GPU: adds dense, document_chunked, summary_token (their KV cache needs flash attention)
pytest -v src/test/inference_consistency/
```

The GPU-only variants skip cleanly without CUDA/flash-attn, so a green CPU run does **not** mean all
five were checked — read the skip list.

## On a real checkpoint

The tiny models above verify the code paths agree in principle. To check an actual trained
checkpoint against real eval data:

```bash
python -m corpus_reasoning.eval.run_inference_consistency \
    --model-path /weka/oe-training-default/.../step10000 \
    --data data/contradiction_eval_pubmed_both_n100_k3.jsonl \
    --task contradiction \
    --tokenizer Qwen/Qwen3.5-0.8B \
    --family qwen3_5 \
    --eval-size 500 \
    --out consistency.json
```

The variant is detected from the built model, not from a flag, so the same command works for every
arm of a sweep — and a checkpoint that is not the variant its run name claims shows up as a detection
mismatch rather than a plausible-looking number.

`--family` is required for landmark checkpoints (it supplies the landmark/pad token ids from
`RESERVED_IDS`). Over-budget prompts abort by default rather than being silently dropped; pass
`--skip-too-long` to exclude them and have the count reported. The JSON reports `eval_size`, and a
sub-500 run is flagged inline.

For a landmark checkpoint, run it **both ways** — they answer different questions:

```bash
# As real evals serve it (top-k on): how far does eval-time behaviour sit from the trained function?
python -m corpus_reasoning.eval.run_inference_consistency ... --out topk_on.json

# Top-k off: a genuine correctness check on the KV-cache decode. Should come back ~identical.
python -m corpus_reasoning.eval.run_inference_consistency ... --no-topk --out topk_off.json
```

If the `--no-topk` run does *not* come back near-identical, that is a real bug in the landmark
cached decode — not a modelling result. Keep `--max-gold-tokens` small enough that continuations stay
inside the prompt's final block, or cause (2) above will contribute a gap of its own.
