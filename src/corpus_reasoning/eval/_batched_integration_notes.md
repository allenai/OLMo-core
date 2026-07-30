# Batched chunked eval — integration plan

## Files added

Shared:
- `scripts/eval/chunked_batch_helpers.py` — padding, chunk-id building,
  prefill-mask, position-ids, decode-mask, stop tester. All variants
  below are thin wrappers around these.

Variants (each is a drop-in replacement for the existing
`generate_hf` loop for HF chunked-family backends):

| File                                        | Strategy                                              | Expected win |
|---------------------------------------------|-------------------------------------------------------|--------------|
| `batched_chunked_generate.py`               | Left-pad batch; 4D prefill mask; eager decode loop    | B × (minus prefill amortization) |
| `batched_chunked_prune.py`                  | Baseline + drop finished examples via `DynamicCache.batch_select_indices` | Helps when output length varies |
| `batched_chunked_compile.py`                | Baseline + `torch.compile(model, dynamic=True)` on the decode step | +1.5–2× on top of baseline |
| `batched_chunked_flex.py`                   | Baseline with FlexAttention batched BlockMask prefill; eager decode | Wins when prefill dominates (long prompts); requires `attn_implementation="flex_attention"` at load time |

Harness:
- `scripts/eval/test_batched_chunked_generate.py` — runs the existing
  single-example path and the selected variants on the same prompts;
  reports per-example text agreement + wall speedup.

## Test plan (when GPUs free up)

Compare all three safe variants against the single-example baseline on
the n=12 chunked checkpoint (fast to load, enough tokens to be mask-
realistic):

```
python scripts/eval/test_batched_chunked_generate.py \
    --lora outputs/checkpoints/contradiction-pubmed-n12-k3-qwen08-lora-chunked-cot-enum \
    --base-model Qwen/Qwen3.5-0.8B-Base \
    --eval-data data/contradiction_eval_pubmed_both_n12_k3.jsonl \
    --n-examples 8 --batch-size 4 --max-new-tokens 400 \
    --variants baseline,prune,compile
```

For flex (requires model reload):

```
python scripts/eval/test_batched_chunked_generate.py \
    --lora ...same... \
    --attn-impl flex_attention \
    --variants baseline,flex
```

## Pass criteria

Per variant:
- `exact matches vs single ≥ 7/8` at T=0 greedy. Bf16 ties on late
  tokens are acceptable but flag anything earlier than the last ~5%.
- `speedup ≥ 1.3x` at batch_size=4 (higher for larger batch sizes).
  Compile variant needs a warmup run (first batch is slow while
  inductor compiles); the harness does a bs=1 warmup.

On a pass, integrate into `scripts/eval/evaluate.py` with a new
`--eval-batch-size` flag and `--eval-variant` flag; dispatch into the
right module. Keep the single-example path as the default.

## Deferred (not written in this pass)

- `StaticCache` + `torch.compile(..., mode="reduce-overhead")`. Needs
  explicit `cache_position` plumbing. Attempt only if the
  DynamicCache+compile variant shows substantial recompile overhead.
- Cross-batch persistent compile cache between eval runs (requires
  saving dynamic-shape artifacts). Not worth the complexity for a
  one-shot eval.
- vLLM kernels with a custom chunked mask. Out of scope — vLLM's
  attention path has no per-token mask_mod hook.
