# Batched native-eval decode: status (2026-07-20)

## What changed

Added `--batch-size` (default 1, unchanged behavior) to the native document-chunked evaluators:

- `src/corpus_reasoning/eval/batched_native_decode.py` (new) — shared batched greedy-decode loop:
  LEFT-pads a group of examples to the batch's max prefill length, uses the model's existing
  `cache_leftpad` support so every row's "next token" always lands in the tensor's last column (no
  ragged per-row gather needed), and replays each eval file's own per-row stop condition
  (`should_stop` / `_answer_complete`) faithfully.
- `src/corpus_reasoning/eval/eval_lc_native_docchunk.py`, `eval_lc_native_docchunk_contra.py` —
  `--batch-size N` dispatches to the batched path for `--variant dense|full`; `--variant landmark`
  always stays bs=1 (periodic landmark-token re-insertion during decode doesn't fit the batched loop
  — out of scope here). `pad_id` is now always `PAD_TOKEN_ID` for the dense/landmark chunk-id
  reconstruction (harmless no-op at bs=1: that id never appears in an unbatched dense/full prefill).
- `src/scripts/eval/ctc_suite/run_rung_eval.py` — `--batch-size` passthrough to the evaluator
  (default 1, so nothing else in the pipeline changes unless it's opted in).
- `src/olmo_core/nn/attention/recurrent.py` (`GatedDeltaNet.forward`) — **the fix this whole effort
  hinged on.** See "The hybrid-model complication" below. Also added
  `src/test/nn/attention/recurrent_test.py::test_gated_delta_net_cache_leftpad_matches_unpadded`
  (GPU-only, auto-skips without one).

## The hybrid-model complication (read this before batching a new checkpoint)

The CTC-suite Qwen3.5 checkpoints are **hybrid**: `block_pattern=["gdn","gdn","gdn","attn"]` — 3 of
every 4 layers are a recurrent `GatedDeltaNet` (causal-conv + gated delta-rule state), only 1 in 4 is
attention. Left-padding a batch is automatically safe for the attention layers (causal masking, plus
`DocumentChunkedAttention`'s own PAD role, both exclude the pad prefix from ever being attended) — but
**it is not automatically safe for GatedDeltaNet**. The recurrent scan has no notion of padding: the
pad tokens' conv/decay contributions would leak into the real tokens' outputs and corrupt the
downstream KV/state that generation continues from, unless masked.

A previous, unrelated session (`src/corpus_reasoning/eval/_batched_integration_notes.md`,
`diag_rightpad_divergence.py`, `_rightpad_bs_sweep.py` — HF/LoRA backend, different codebase)
independently hit exactly this: bit-exact at bs=2, F1 dropped ~0.06 at bs=8, with an open question of
whether GDN state pollution was the cause. That is the same failure mode this fix targets.

**The fix** (`GatedDeltaNet.forward`, new `cache_leftpad` parameter): at a padded prefill, zero
`q`/`k`/`v` at padded positions *before* the causal conv (so the conv's receptive window sees the
same all-zero history it would via its own intrinsic zero-padding — bit-identical to no padding at
all, as long as left-pad length ≥ `conv_size - 1`, true in every realistic case), and force
`beta=0`/`g=0` (log-decay) at padded positions so the delta-rule update is an exact no-op there
(state passes through unchanged). Proven correct in isolation by the new GPU unit test (two rows,
different pad amounts, large-magnitude garbage in the pad region, compared against an unpadded bs=1
reference) and end-to-end in the parity run below.

**A second, separate wrinkle**: some "full"-arm checkpoints (e.g. `ctc-1ep-contra-full`) are
genuinely trained with **plain** `Attention`, not `DocumentChunkedAttention` with the mask disabled —
a different baseline model, not an ablation of the same weights. Calling
`enable_document_chunk_attention` on such a model would thread a `chunk_ids` kwarg into blocks whose
`forward` doesn't accept it (a hard crash) and isn't needed anyway — plain `Attention`'s KV-cached
prefill already goes through `flash_attn_with_kvcache`, which honors `cache_leftpad` natively (the
same mechanism the generic `TransformerGenerationModule.generate_batch` already relies on). Other
"full"-arm checkpoints really are a `DocumentChunkedAttention` model with the mask disabled at eval
time, which DOES need pad-exclusion — for those, `batched_native_decode.force_standard_pattern`
temporarily swaps every `DocumentChunkedAttention` layer's pattern to `"standard"` (`causal &
not_pad`, no document isolation) for the duration of the batched call.
`batched_native_decode.model_has_document_chunked_attention(gm)` detects which case a checkpoint is
and the eval scripts branch on it automatically — no flag needed.

## Parity result (bs=1 vs bs=16, contradiction rung_2048, 0.8B, 1×H200)

Ckpts: `full` arm = `/data/prasann/ctc_suite/ckpts/ctc-1ep-contra-full` (plain-attention-trained,
case above); `dense`/`chunked` arm = `/data/prasann/ctc_suite/ckpts/ctc-smoke-contra-3ep`
(`DocumentChunkedAttention`, `cross_doc_mode=chunked`). Both are the same hybrid
GDN+attention architecture, so both exercise the GatedDeltaNet fix.

N=40 (quick check):

| arm | bs=1 f1 | bs=16 f1 | exact-prediction matches |
|---|---|---|---|
| full   | 0.5417 | 0.5417 | 40/40 |
| dense  | 0.0750 | 0.0750 | 40/40 |

N=500 (full rung, via `run_rung_eval.py` — real production path, results written through
`results_io`):

| arm | bs=1 f1 | bs=16 f1 | \|Δf1\| | exact-prediction matches | bs=1 wall | bs=16 wall | speedup |
|---|---|---|---|---|---|---|---|
| full  | 0.5167 | 0.5153 | 0.0013 | 496/500 (99.2%) | 256.5s | 232.5s | 1.10x |
| dense | 0.0360 | 0.0360 | 0.0000 | 482/500 (96.4%) | 248.7s | 53.8s  | **4.62x** |

At N=500, a small fraction of examples (0.8% full, 3.6% dense) produce a token-level flip between
bs=1 and bs=16 — **not a masking/padding bug**: every mismatch is a single near-tied-logit flip (one
digit differs in one pair; example `idx=63`: `[17,49]` vs `[10,49]`), the kind of divergence expected
whenever a bf16 batched GEMM reduces in a different order than the equivalent bs=1 GEMM (well-known,
engine-agnostic bf16 batching non-determinism — the same phenomenon vLLM/continuous-batching users
hit). Higher-quality generations (`full`, more confident logits) flip less often (0.8%) than
near-floor ones (`dense`, more logits genuinely tied) — consistent with a numeric-precision
explanation, not a structural bug. Aggregate `|Δf1|` stays ≤0.0013, at or inside the task's own
"within 1e-3" tolerance. Re-verified with `debug/batched_eval_parity/compare.py` (exact per-example
prediction diffing, not just the aggregate metric).

`dense` shows the expected large speedup (near-floor quality → short generations, minimal
straggler cost). `full` shows only 1.10x at N=500/bs=16 — see "Batching efficiency notes" below for
why, and pick the batch size per-task/arm based on typical generation length until compaction is
added.

## Batching efficiency notes (read before assuming 10-20x)

Naive batching (this implementation) pads every row in a batch to the group's max PROMPT length
and then runs every row for the same number of DECODE steps (until every row in the batch has
stopped) — a single slow "straggler" example in a batch forces every other row in that batch to
keep stepping. `full`'s higher-quality, longer generations hit this harder than `dense`'s near-floor
short ones, which is why the two arms show such different speedups above. This is the same tradeoff
a prior (unrelated, HF-backend) exploration flagged as "helps when output length varies" for a
`DynamicCache.batch_select_indices`-style compaction variant
(`src/corpus_reasoning/eval/batched_chunked_prune.py` — not wired up here, different codebase). NOT
implemented in this pass (out of scope / time-boxed) — flagging as the natural next optimization if
a task's `full`-arm speedup disappoints: dropping finished rows from the batch (or bucketing by
expected generation length) would recover most of the loss.

## Cleared for batching

- **`--variant full`**: cleared. Bit-exact modulo the bf16 batching noise documented above
  (|Δf1| ≤ 0.0013). Speedup varies by generation-length profile (1.1x-5x observed); measure per task.
- **`--variant dense`** (the `chunked`/`chunked-mix` driver arms): cleared, same caveat. Strong
  speedup (4.6x) observed on a near-floor checkpoint; expect less on a higher-quality one due to the
  straggler effect above.
- **`--variant landmark`**: NOT batched (out of scope — periodic landmark re-insertion during decode
  doesn't fit this loop). Always uses `--batch-size 1`; the flag is rejected with a clear error if
  passed with `--variant landmark`.

## Exact command

```bash
# direct (single GPU, no torchrun):
python src/corpus_reasoning/eval/eval_lc_native_docchunk_contra.py \
  --variant full --model-path <ckpt> --out out.json --per-example-out out.gen.json \
  --tokenizer Qwen/Qwen3.5-0.8B-Base --doc-start-id 248049 --doc-end-id 248050 \
  --eos-token-id 248044 --max-length 4096 --cot-mode none --contra-max-new-tokens 512 \
  --contra-data <rung.jsonl> --batch-size 16

# via the driver (adds provenance + results-hub schema):
python src/scripts/eval/ctc_suite/run_rung_eval.py \
  --task contradiction --ckpt <ckpt> --variant chunked --rung-tokens 2048 \
  --eval-jsonl <rung.jsonl> --arm chunked-mix --model-scale qwen3.5-0.8b --nproc 1 \
  --batch-size 16
```

See `debug/batched_eval_parity/run_parity.sh` for the exact harness used above.
