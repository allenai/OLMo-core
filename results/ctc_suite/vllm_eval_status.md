# CTC-suite vLLM eval: status (2026-07-20)

STOPPED per coordinator directive (4B roster mostly failed training on a metadata bug;
eval deprioritized until relaunch completes). This records what's proven so far so the
next session doesn't re-debug the same infra.

## Parity table (task, arm, rung, native_f1, vllm_f1, |Δ|, PASS/FAIL)

| task | model | arm | rung | native_f1 | vllm_f1 | \|Δ\| | binomial SE (combined) | PASS/FAIL |
|---|---|---|---|---|---|---|---|---|
| contradiction | qwen3.5-0.8b | full | 2048 | 0.684 (n=500) | 0.763 (n=500) | 0.079 | ~0.028 | **FAIL** (~2.8x SE) |
| contradiction | qwen3.5-0.8b | chunked-mix | 2048 | 0.036 (n=500)\* | 0.049 (n=500) | 0.013 | ~0.013 | soft PASS (~1x SE, both near-floor) |
| contradiction | qwen3.5-4b | full | 2048 | 0.571 (n=500) | **not measured** | — | — | blocked (see below) |

\* No native chunked-mix reference existed for the current `rung_2048.jsonl` (the only
prior chunked-mix number was on the retired n20-pilot data). I ran it myself via the
existing native evaluator to get an apples-to-apples comparison; it's now recorded at
`results/ctc_suite/contradiction/qwen3.5-0.8b_chunked-mix/rung_2048.json` (overwriting the
stale pilot entry, as intended by the pipeline's per-rung-key design).

## Wall-clock (measured, 500 examples, contradiction rung 2048, 0.8B, 1 GPU)

| arm | native (bs=1) | vLLM | speedup |
|---|---|---|---|
| full | 551.2s | 104.8s (73.8s load + 31.0s gen) | **5.3x** (would be larger amortizing model load across multiple rungs in one process) |
| chunked | 247.6s | 1052.7s | **0.24x — 4.25x SLOWER** |

## Verdict

**vLLM-full: mechanically works, NOT statistically at parity with native, but not broken either.**
Both engines produce coherent, well-formed pair-list generations (parse_rate=1.0 on both
sides); spot-checking individual examples shows genuine token-level divergence (only
134/500 generations were byte-identical) rather than truncation/masking artifacts. The
gap is directionally consistent (vLLM higher every time I looked), not symmetric noise,
so likely a real cross-engine numerical difference — plausibly amplified by the hybrid
model's Gated-DeltaNet (linear-attention) layers, whose recurrent state compounds
differently between olmo-core's native kernel and vLLM's Triton/FLA prefill kernel, more
than a plain-attention forward would. **Root cause not diagnosed further per the stop
directive.** Recommendation: fine for fast relative/trend comparisons across
checkpoints/rungs; do not report vLLM numbers as interchangeable with native numbers in
results-hub without this caveat until the drift is understood.

**vLLM-chunked (FlexAttention custom mask): NOT VIABLE, drop it.** Independent of the
(roughly-passing) numeric parity, it is 4.25x *slower* than the already-slow native path
here — the opposite of the point. Root cause: the hybrid Qwen3.5 KV-cache page size (544
tokens, padded for mamba alignment) never matches vLLM's kernel block size, so
`vllm_chunked_patch`'s metadata-builder patch is *always* forced onto the O(num_actual_tokens
× total_cache_tokens) fallback BlockMask-rebuild path — confirmed via the patch's own
debug counters: `{'calls': 32399, 'applied': 32399, 'direct': 0, 'fallback': 32399}` (zero
uses of the fast direct-build path). **Keep chunked-arm eval on native bs=1**; do not
spend more time on the vLLM-chunked path for this model family unless vLLM's hybrid
FlexAttention direct-build path changes.

**4B (the actual target scale): parity unconfirmed, blocked on infra not the model.**
Export (olmo distcp → HF text-only, via the shadow-transformers workaround from
`convert_both.sh`) succeeded in ~35 min (`/data/prasann/ctc_suite/hf_exports_4b/ctc-s5-contra-full-4b`
on horton, mostly a one-time base-model network download). The next step — wrapping it
into a vLLM-servable VL config + pasting in the (unused but required-to-construct) vision
tower weights — failed because horton's local `Qwen3.5-4B-Base` mirror
(`/data/prasann/hf_models/Qwen3.5-4B-Base`) is metadata/tokenizer-only (config + tokenizer
+ preprocessor jsons but **no actual `.safetensors` weight shards**). Prefill-building
(tokenizer-only) succeeded fine against the same dir, confirming it's specifically the
weight shards that are missing, not the whole snapshot. The job was then preempted before
a fix could be tried. **Straightforward fix next time**: point `make_vl_weights.py
--base-snapshot` at any COMPLETE Qwen3.5-4B-Base HF snapshot (with real
`model.safetensors*` shards) — e.g. re-download via `snapshot_download`, or locate one on
a node that has the full weights — then rerun `debug/ctc_vllm_validation/pipeline_4b.sh`
(export step will skip, already done).

## Reusable infra (for whoever resumes)

- `debug/ctc_vllm_validation/pipeline.sh` (+ `.sbatch`) — the proven 0.8B validation
  gate: builds vLLM-servable copies, token-identical prefills (reusing the native
  evaluator's own `build_eval_prefill`), runs both vLLM arms, grades with the same
  `_eval_contradiction` the native path uses.
- `debug/ctc_vllm_validation/pipeline_4b.sh` (+ `.sbatch`) — same for 4B on horton; needs
  the base-snapshot fix above before it'll get past the serving-copy step.
- `debug/ctc_vllm_validation/build_prefills_generic.py`, `run_vllm_eval_generic.py`,
  `grade_responses_generic.py` — task-parametrized versions (any `TASK_CFG` key, not just
  contradiction) for the eventual multi-task sweep (grouping/hotpotqa/outlier/...). Written
  but **never run** — untested beyond code review. `run_vllm_eval_generic.py` only
  implements the `eos`/`newline` stop rules (covers retrieval/outlier/grouping); oolong's
  answer-line stop rule is not implemented.
- vLLM venv: `/data/prasann/ctc_vllm_venv` on **cubbins** (jsteinhardt) — vllm==0.25.1,
  built from scratch this session (none existed despite the run_vllm_eval.py docstring
  assuming one). Same version now also confirmed present at `/data/prasann/ctc_vllm_venv`
  on **horton** already (pre-existing, from an earlier session).
- Known env trap: the login shell/profile exports a stale `TRANSFORMERS_CACHE`
  (`/scratch/users/prasann/huggingface-cache/transformers`, empty) that silently shadows
  `HF_HOME` in transformers 4.57+ (deprecated-var precedence) and breaks
  `AutoTokenizer.from_pretrained("Qwen/...")` in offline mode with a confusing "outgoing
  traffic disabled" error even when the real cache is present and correctly pointed to.
  `unset TRANSFORMERS_CACHE` before setting `HF_HOME` (both scripts above already do this).
- Serving-copy gotcha (now fixed in `make_vllm_serving_copy.py`): vLLM 0.25.1 resolves
  *any* `Qwen3_5*` architecture string to the multimodal `Qwen3_5ForConditionalGeneration`
  class (`_normalize_arch`'s `ForCausalLM`→`ForConditionalGeneration` suffix rule), which
  still probes for an image/video processor even for a pure-text prompt — the text-only
  HF export has none, so the base snapshot's `preprocessor_config.json` /
  `video_preprocessor_config.json` must be symlinked in too (not just the vision-tower
  weights already handled by `make_vl_weights.py`).
