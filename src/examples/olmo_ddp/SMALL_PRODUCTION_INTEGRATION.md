# Small-model production integration, September 5

Status: prepared; launch only after the feature qualification and combined smoke
gates in `SMALL_OPTIMIZATION_SIGNOFF.md`. This is an experiment, not broad rollout
approval. A100B run cannot establish14T stability.

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

Set `OLMOE3_INTEGRATION_SMOKE=1` for the paired smoke. For each long run set
`OLMOE3_INTEGRATION_ARM=reference` or `optimized`, leave smoke disabled, and choose
the **qualified** `OLMOE3_INTEGRATION_POLICY`. Submit with the standard `launch`
subcommand on a clean pushed commit. Supply the same eligible Holmes host list,
excluding the known unhealthy485/516 hosts. Do not use unsupported policy strings
or change model/training settings between arms.
