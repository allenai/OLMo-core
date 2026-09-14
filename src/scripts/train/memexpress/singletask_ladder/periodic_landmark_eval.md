# SFT-aligned periodic landmark output

Enable `--landmark-periodic-output` on
`src/scripts/train/memexpress/singletask_ladder/run_q4b_beaker_multirung_eval.py`
or directly on `src/scripts/ctc_eval/eval/eval_lc_native.py`. The shell runner accepts
`LANDMARK_PERIODIC_OUTPUT=1`. The generation API equivalent is
`GenerationConfig(landmark_decode_mode="periodic", ...)`.

Insertion follows the checkpoint's `mem_freq` and `landmark_mem_id`. For the block64
models, each block contains 63 content tokens and one landmark. The count starts at
the beginning of the tokenized prompt (including chat-template tokens) and continues
through output. A prompt ending 23 tokens after its last landmark allows 40 output
tokens before the next forced landmark. It is never padded at the input/output
boundary. This is the prefix of the actual SFT packer's input/output document layout.

The forced landmark is processed through the model's KV and recurrent caches before
the next content token is sampled. Attention uses its periodic per-block decode path:
completed output blocks become past blocks, and a landmark query uses the same
self-attention rule as a training landmark. The reserved landmark ID is suppressed
at content positions so it cannot be sampled out of place.

Returned completions, logits, and log probabilities contain only content positions.
Landmarks are removed before stop-string decoding and before the evaluator decodes
and scores responses. `max_new_tokens` remains a visible-content budget. In periodic
mode, a supplied `max_length` bounds physical positions **including landmarks**, even
when `max_new_tokens` is provided. An oversized requested completion fails before
prefill rather than truncating the prompt or silently shortening output. With only
`max_length`, the visible budget is derived from the available physical positions.
The legacy modes retain their existing length-budget behavior.

This option is independent of retrieval. Add `--landmark-disable-top-k` to use all
past blocks, or keep the existing top-k settings. The Python API disables retrieval
with both `landmark_top_k_blocks=None` and `landmark_top_k_fraction=None`.

The launcher automatically adds `periodic-lm` to the Beaker name and Weka output tag
(and `no-topk` when applicable). The shell runner also adds `periodic-lm` when invoked
directly. Result JSON records the periodic decode mode and retrieval settings.
Use a distinct user eval tag for each launch, as usual.

The new mode supports single-landmark models via `generate_batch`. Unpadded,
equal-length batches work; variable-length eval prompts use batch size 1. The ragged
batch API and multi-landmark geometry reject this mode explicitly. Existing modes
remain the default. No evaluation sweep is launched by enabling support in the code.

Tests cover the real SFT packer's layout, 23+40 alignment, full-forward versus cached
logits with and without chunked prefill, compressive attention at output block
boundaries with and without top-k, marker stripping, EOS/stop strings, length budgets,
and launcher tags. CPU tests can be run with `TORCH_COMPILE_DISABLE=1` where a local
PyTorch/Inductor compiler is unavailable.
