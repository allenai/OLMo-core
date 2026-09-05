# Small-model production integration, September 5

Status at 2026-09-05 11:48 UTC: both 100B integration arms are training successfully.
The production save/restore/eval/upload smoke, final combined A/A review, and
matched long-run startup audit all passed.
Frozen candidate: `core-docpool-top16-wgrad-rs`.
This is an experiment, not broad rollout approval. A 100B run cannot establish
14T stability. See `SMALL_OPTIMIZATION_SIGNOFF.md` for the evidence and caveats.

Submitted at 11:26 UTC from clean pushed commit
`107dfa3ff42f4ee4984d5c445d587ceb7db5e4f4`:

- Reference: https://beaker.org/ex/01M1RN3TK5Q4GCM5JYNJ20W1XZ
- Optimized: https://beaker.org/ex/01M1RN3NHFH32P2Z952BCR03YD
- Read-only start observer: https://beaker.org/ex/01M1RN5CGDCZGY23T10MHW7GTZ

Each training arm is urgent/allocated, 8x8 B300s in `ai2/olmo3p5-training`,
excluding Holmes hosts 485 and 516. Beaker replaced optimized replica5 after a
pre-start health-check failure on host555; the replacement on host493 is running.
No model error or configuration change was involved.

The start observer exited0 at11:48:22. At its final snapshot, reference had8 and
optimized57 finite, non-skipped updates. All64 initial-weight fingerprints and
all64 first-batch fingerprints match; both use the same source commit. Both
step-0 checkpoints are remotely verified on their first upload attempt, and both
current manifests are published. Weka has39.044TB free. Small evidence dataset:
`01M1RN5CGQCKZKB5NHMBQX6MXD`, local `profiling-analysis-20260905/integration-start-r1`.

WandB:

- Reference: https://wandb.ai/ai2-llm/olmoe3-production-profiling/runs/x7urehin
- Optimized: https://wandb.ai/ai2-llm/olmoe3-production-profiling/runs/p21bkf81

Launch success is established; long-horizon loss/eval assessment is still pending.

## Fixed comparison

| Setting | Both arms |
|---|---|
| Parameters |794,233,472 active;12,496,341,632 total |
| Architecture |16 layers;d1024;latent512;14KDA/2FA;512experts/top16;EMO16–512 |
| Attention |Q8/KV4;head128;expand_v2;per-head QK gains (PR855) |
| Hardware |64B300s;8Holmes nodes;urgent, allocated |
| Parallelism |PP1/EP1/DP64;MB4 sequences;8 accumulation microbatches |
| Batch / sequence |16,777,216 tokens/update;8192 tokens/sequence |
| Schedule |LR.00185;WSD warmup2000;decay1;6000updates=100.663296B tokens |
| Precision |BF16 model/compute;FP32 gradients, reductions and Adam states |
| Initialization / data seed |12536 /928543231;Dolma3.5;shared existing data-order cache |
| Checkpoints |Synchronous step0, then every250 updates;keep all25 |
| Held-out evaluation |OLMo v3 small perplexity validation every1000 updates and at finish |
| Uploading |Existing private pilot bucket, unique run prefixes;report_only;no deletion |
| Explicitly off |Activation checkpointing, MXFP8, shared EP buffers, fused attention, TBO |

Reference retains the same package versions, model and correctness fixes, with
profiling performance switches off. Candidate enables only the final qualified
bundle. The exact flags are written in `audit/session-*.json`, rather than inferred
from a mutable branch name. Both arms continue to use the existing cconv package.
The separate upstream CTA-policy PR does not change the pinned integration package.

## Gates and monitoring

1. Primitive and optimizer qualification plus same-allocation timing for each new
   feature. Keep rejected candidates default-off.
2.64GPU matched smoke: reference0->4->8, optimized0->4->8 in separate processes.
   Save step0/4/8, restore model/Adam/trainer, evaluate held-out data, and verify
   uploader manifests. No smoke artifacts/checkpoints are removed.
3. Compare all distributed initial-weight SHA256 hashes and first-batch hashes
   between arms. Retries verify existing fingerprints instead of replacing them.
4. Launch separate100B arms and confirm finite training updates. If allocation
   remains unavailable, leave queued only after the relevant smoke passes.
5. Review loss/held-out curves, skipped steps, norm outliers and wall-clock
   throughput at matched token counts. The paired activation A/A control had one
   skipped update in one repeat; retain that fact in the long-run assessment.

Artifacts and checkpoints live under
`/weka/olmo-3p5-checkpoints/production-integration/<run-name>`.
Beaker results must contain only small diagnostics, never the checkpoint tree.
The uploader discovers completed registered checkpoint directories by polling;
no trainer callback or uploader restart is needed. A matching enabled
`report_only` registration is required by the training audit before stepping.

## Entry points

- `olmoe3_small_integration.py`: model/training/eval/checkpoint policy and audit.
- `olmoe3_integration_node.py`: current-Beaker-job rendezvous and independent
  torchrun agents. Smoke mode runs both arms and their restores on one allocation.
- `olmoe3_profile_setup.sh`: frozen kernel-fun commit and runtime checks.
- `olmoe3_integration_start_check.py`: read-only CPU observer for the fixed final
  run IDs; requires matched source/weights/data, five non-skipped finite updates
  per arm, and complete, remotely verified step-0 checkpoints. Writes only small
  diagnostics to Beaker results, never to the checkpoint mount.

Set `OLMOE3_INTEGRATION_SMOKE=1` for the paired smoke. For each long run set
`OLMOE3_INTEGRATION_ARM=reference` or `optimized`, leave smoke disabled, and choose
the **qualified** `OLMOE3_INTEGRATION_POLICY`. Submit with the standard `launch`
subcommand on a clean pushed commit. Supply the same eligible Holmes host list,
excluding the known unhealthy485/516 hosts. Do not use unsupported policy strings
or change model/training settings between arms.

## Long-run assessment after launch

The registered run IDs are `olmoe3-small-16mi-100b-reference-r1` and
`olmoe3-small-16mi-100b-optimized-r1`. Both use WandB group
`small-production-integration-20260905`. The initial launch check requires all
eight workers to initialize, matching weight and first-batch fingerprints, and
several finite actual optimization updates. It does not mean the 100B comparison
has already passed.

For the subsequent assessment:

- Compare CE and norms by update/token count, including rolling means and tails,
  not by wall clock. Keep every skipped update and restart visible. A single skip
  is not automatically a regression: the earlier paired A/A control had one.
- Compare the same observed held-out subsets at each 1,000-update checkpoint and
  finish; do not average unmeasured/NaN subsets or mix partial and full evaluations.
- Use steady-state median TPS/GPU for kernel throughput. Separately report elapsed
  wall-clock tokens/sec including compilation, synchronous saves and validation;
  do not use the former as an end-to-end measurement.
- Confirm all 25 checkpoint records per run are complete and remotely verified,
  with no uploader deletion and no unexpected rewrite of an older checkpoint.
  Upload lag is allowed while local capacity remains ample; keep checkpoints.
- Investigate reproducible loss/held-out degradation or excess norm/skip behavior.
  Two short repeats measure an observed variability envelope, not a universal
  acceptance threshold. Do not describe close BF16 trajectories as bit-identical.
- Keep broader deployment approval separate from this experiment. In particular,
  small-model EP1 qualification does not establish medium/large/EP/PP correctness
  or performance, and 100B tokens do not establish 14T-run stability.
