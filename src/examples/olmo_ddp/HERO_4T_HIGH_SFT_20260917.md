# 4T descendants: high-reasoning SFT automation

This replaces the **not-yet-launched** old-data three-LR SFT sweep for the 4T
campaign. It does not alter the running MT jobs, the LC recipe, any 2T SFT
experiments, or the base-model comparison watcher.

| PT lineage | Parent | SFT EMO | LR | Epochs | GPUs |
|---|---|---|---:|---:|---:|
| EMO | 4T decay → no-EMO MT → no-EMO LC, step5961 | off | 5e-5 | 2 | 8 |
| Non-EMO | 4T decay → no-EMO MT → no-EMO LC, step5961 | off | 5e-5 | 2 | 8 |

Dataset: `jacobmorrison/length-investigation-gptoss-120b-high`, revision
`2fa53f4df6e4e41f9202c31cc8e26b2a04bce027`, using the already-tokenized data at
`/weka/oe-adapt-default/jacobm/olmoe3/olmo-ddp-migration/sft-data/gptoss120b-high-olmo-thinker-20260917`.
The pinned tokenizer, thinker template, assistant-only masks, held-out split,
packing and document isolation match the existing high-reasoning experiment.

Keep its qualified recipe: 524,288 batch tokens, 65,536-token microbatches,
one eight-GPU Holmes node, BF16, eager packed FLA, block recomputation,
fresh optimizer/data state, linear decay to zero, 3% warmup, zero weight decay.
Save native checkpoints each epoch; convert/evaluate **only the final epoch**.

## Controllers

Branch: `codex/hero-4t-high-sft-20260917`.
Native campaign: `olmo35-small-4t-gptoss-high-sft-20260917`.
The existing durable `native-pipeline` and `eval-pipeline` state is reused.
Existing MT/LC workers remain pinned to `d031ab975`; existing base-eval workers
remain pinned to `ca8083701`. Their specs and submission names stay identical,
so completed/running jobs are adopted rather than duplicated.
The LC handoff checks immutable MT task arguments/identity: training progress
updates overwrite Beaker's human-readable description, so it is not parsed as
the original JSON plan.

Both controllers use Phobos, no resource requests, urgent/unallocated. Training
remains urgent/allocated in `ai2/olmo3p5-training`; post-training evals use the
MoE workspace, urgent/allocated with six-hour minimum runtime and auto-resume.

The new CPU configuration/data gate also validates all downstream Beaker specs.
Each LC lineage independently gates a four-update load/restart smoke and its
two-epoch SFT. Uploader registration remains on, with latest-two local retention.
After successful SFT exit, conversion installs/checks chat template, tokenizer
and generation metadata; then Math500, IFBench, HumanEval and AlpacaEval launch.
Numerical inference parity remains waived as previously requested for this 4T
campaign; structural, finite-tensor, checksum and metadata gates remain enabled.

Eval settings are unchanged: thinking prefill/final-answer extraction,
temperature 0.6, top-p 0.95, maximum 32,768 new tokens, seed 1234. Math/IFBench/
Alpaca reuse immutable completed responses after preemption. HumanEval gets a
fresh sandbox and attempt directory. Failure is visible; runtime errors do not
trigger uncontrolled retry loops.

Deployment stops only the two superseded CPU watchers. Running training and
eval jobs are not canceled. No checkpoint or dataset data is deleted by this
deployment, and no scratch artifacts go into Beaker result datasets.
