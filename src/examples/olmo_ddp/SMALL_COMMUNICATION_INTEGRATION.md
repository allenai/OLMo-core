# Communication follow-up qualification

The September 5 optimized 100B run is the new reference, not the original slower
reference arm. Its qualified source is
`107dfa3ff42f4ee4984d5c445d587ceb7db5e4f4`. Model, EMO, per-head QK scales,
kernel package, BF16/FP32 precision, 64 B300 GPUs, PP1/EP1/DP64, MB4, GA8,
16,777,216 tokens/update and LR/WSD/data seeds remain unchanged.

## Explicit selection

Use the existing `olmoe3_integration_node.py` / `olmoe3_small_integration.py`
with these environment settings in both arms:

```
OLMOE3_INTEGRATION_BASELINE=optimized100b
OLMOE3_INTEGRATION_POLICY=core-docpool-top16-wgrad-rs
OLMOE3_INTEGRATION_COMMUNICATION=deferred-lb
```

The final value is a candidate selector, not a sign-off: valid alternatives are
`none`, `deferred`, `lb-overlap`, `deferred-lb`. In the reference arm the new
collective flags are always zero, regardless of ambient environment. Both arms
retain CTA128, inverse-scatter, vectorized gradient addition, paired activation,
document pool, native-tie top16, rounded wgrad and direct reduce-scatter. The
original campaign remains the default; combining new collectives with that old
reference, or omitting part of the qualified bundle, fails early. Unit tests
cover all choices and stale ambient flags. The complete selection is saved in
each session's provenance.

## Gates before any longer run

1. Existing two-GPU numerical tests: global counts, routing, loss, gradients and
   three Adam steps; eager/compiled, ordinary routing/EMO, Gloo/NCCL.
2. Same-allocation 200-update A/A/B repeats from optimized step5750, separately
   testing overlap and combined deferred reductions. Compare clean steps31–200,
   all skipped steps, full CE/gradient-norm trajectories and baseline A/A spread.
   Previous step6000 restores shared an early transient even with both flags off;
   preserve that evidence, do not attribute it to a candidate or hide skipped steps.
3. Updated Nsight trace, excluding capture-start transitions when interpreting
   exposed communication. Instrumented spans are not unprofiled throughput.
4. Matched fresh initialization smoke: reference0→4→8 and candidate0→4→8,
   separate processes for resumes; identical initial weights and first batches,
   synchronous complete checkpoints, held-out evals, remote verified receipts.
5. If the clean speed gain remains worthwhile, a matched longer integration,
   using the same architecture/schedule/data and independently registered roots.
   Never relabel the old slow baseline as this follow-up reference.

## Uploader and cleanup

New registrations use `deletion_mode=inherit`, floor2 and grace3600s. The current
deployment default is guarded `apply`. Trainer checkpoint pruning stays disabled.
The uploader verifies remote payloads/receipts and a later resumable checkpoint,
protects the newest two verified local checkpoints and rechecks before deletion.
No remote deletion. Use `olmoe3_integration_collect.py --allow-cleanup` to audit
smokes: a removed step0 is accepted only with the published verified deletion
record, verified later successor and protected final local pair. Checkpoint
payloads stay on Weka/HF, never in the Beaker results dataset.

These switches are prepared for the performance/numerical gates; their presence
in this branch does not mean the follow-up integration has passed or been launched.
