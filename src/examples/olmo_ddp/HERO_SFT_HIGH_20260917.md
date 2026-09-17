# Small hero SFT: GPT-OSS-120B high reasoning

Campaign: `olmo35-small-gptoss-high-sft-20260917`.

This experiment changes the SFT dataset and independently tests two and five
epochs. It does not change the model architecture, training kernels, precision,
packing/masking recipe, or evaluation protocol from the prior qualified SFT sweep.

## Data

- Dataset: `jacobmorrison/length-investigation-gptoss-120b-high`.
- Immutable revision: `2fa53f4df6e4e41f9202c31cc8e26b2a04bce027`.
- Local preparation: `/weka/oe-adapt-default/jacobm/olmoe3/olmo-ddp-migration/sft-data/prepare_hero_sft_high_20260917.py`.
- Output: `/weka/oe-adapt-default/jacobm/olmoe3/olmo-ddp-migration/sft-data/gptoss120b-high-olmo-thinker-20260917`.
- Same pinned Dolma2 tokenizer and Open Instruct `olmo_thinker_no_think_sft_tokenization`
  as the previous experiment. Assistant content, including reasoning and EOS, is
  supervised; prompts and padding are masked. No answer truncation.
- Require complete, stop-terminated valid conversations. Exclude 11,622 of
  108,023 source rows; reserve 1,024 unique prompt groups with seed 1729.
- 95,377 training records; the real-image preparation gate computes packed epoch
  size and records exact processed-token and step counts.
- Use authoritative document boundaries, not EOS scanning, for packed isolation.

## Runs

| PT lineage | MT/LC/SFT EMO | LR | Epochs | GPUs |
|---|---|---:|---:|---:|
| EMO | off | 5e-5 | 2 | 8 |
| Non-EMO | off | 5e-5 | 2 | 8 |
| EMO | off | 5e-5 | 5 | 8 |
| Non-EMO | off | 5e-5 | 5 | 8 |

The LC parents and paths are frozen in `olmoe3_hero_sft_plan.py`. The common LR is
the balanced prior-sweep choice, not a claim that every individual metric peaked
at this LR. Independent schedules matter: epoch two of a five-epoch schedule is
not equivalent to a fully decayed two-epoch run.

Use 524,288 global batch tokens, 65,536-token rank microbatches, one eight-GPU
Holmes node, BF16, packed-document FLA 0.5.2, eager execution, block recomputation,
fresh optimizer and data state, linear decay to zero, 3% warmup, zero weight decay.
Preserve the prior optimizer and all remaining settings. Training is urgent and
allocated in `ai2/olmo3p5-training`.

## Automation and checks

1. A one-GPU Holmes config/data gate verifies packing, masks, exact architecture,
   unchanged train-module recipe, and each run's epoch horizon.
2. Two eight-GPU four-update smokes verify native LC weight loading, fresh optimizer,
   finite losses/gradients, full-state restart at update two, and validation loss.
3. Only after both pass, a resource-free Phobos controller releases the four runs.
   Exact-name create intents prevent duplicate submissions. The native checkpoints
   are registered with the existing uploader under distinct run/epoch prefixes.
4. Final checkpoints are converted with the existing qualified converter, correct
   thinking template/tokenizer/generation metadata, and independent HF/vLLM checks.
5. A separate resource-free watcher launches Math500, IFBench, HumanEval and
   AlpacaEval for all four final checkpoints, following a real four-task smoke.
   Evals use the same temperature .6, top-p .95, 32,768 generated-token limit,
   seed 1234, thinking prefill, and final-answer extraction as the prior sweep.
6. Eval workers are urgent, allocated for six hours, auto-resumable, in the MoE
   workspace. Math/IFBench/Alpaca replay immutable completed generations after
   preemption; HumanEval starts a fresh isolated attempt/sandbox.

Only the four requested final models are exported/evaluated. Epoch-one native
checkpoints are recovery checkpoints. Existing campaigns and source checkpoints
are untouched. This comparison does not isolate data quality alone: reasoning
effort, response length, filtering outcomes and dataset contents also differ.
