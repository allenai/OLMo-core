# Small LR sweep, September 6

Source baseline: qualified optimized 100B bundle from `107dfa3ff`. No new model,
precision, recomputation, or experimental communication changes. The new branch
adds only sweep orchestration and imports the existing checkpoint-ready callback.

All GPU jobs: 64 B300 GPUs on Holmes, urgent/allocated (1h min runtime), workspace
`ai2/olmo3p5-training`. PP1/EP1/DP64, MB4 x8192, GA8, 16,777,216 tokens/update.
6000 steps =100.663296B tokens; first2000 steps warmup. Constant trunks retain
the undecayed step6000. LRs:3.25e-4,6.5e-4,1.3e-3,2.6e-3,5.2e-3.

| Decay | Parent step | Child updates | Endpoint |
|---|---:|---:|---:|
|5%|5700|300|6000|
|10%|5400|600|6000|
|20%|4800|1200|6000|
|30%|4200|1800|6000|

Full synchronous saves every300 updates, plus initial step0. Children restore
optimizer/trainer/data state, keep global steps, and retry from their own latest
checkpoint before falling back to the immutable parent fork. No re-warmup.
All eleven qualified held-out PPL datasets run every1000 steps and on finish.
No downstream category/MT/LC evaluation fan-out is part of this production sweep.

The single CPU Rhea controller is urgent/unallocated, empty Beaker result path.
It registers the27 roots (25 production trajectories plus two smoke trajectories),
validates all configs in the actual image, then runs an isolated64GPU three-pass
smoke: fresh trunk0->2, child2->4, and child-own-checkpoint restore4->6.
Only a successful gate launches the five trunks. Each independently successful
trunk triggers its four children after validating all forks. Durable launch
intents, a shared-filesystem lock, and exact-name reconciliation avoid duplicates.
Failed or ambiguous submissions are logged for review; no blind retry copies.

One existing uploader serves all roots, in the authorized private pilot bucket.
Each branch has its own run_id AND lineage_id, checkpoint root and HF prefix.
Explicit apply policies: trunks retain7; children retain1; grace3600s. Trainer
pruning stays disabled. The watcher NEVER reduces trunk retention. Do not extend
trunks past6000 or lower their retention until every child has an independently
verified uploaded restart checkpoint. No global policies or unrelated runs change.

Controller ledger: `/weka/olmo-3p5-checkpoints/uploader/automation/small-lr100b-20260906-r1/launch2`.
The first controller was stopped after a rejected GPU submission revealed that
exported Beaker specs must be recombined into one eight-replica task. No GPU
experiment was created by that rejected request. Its original ledger is preserved.
Checkpoints: `/weka/olmo-3p5-checkpoints/production-lr-sweeps/small-lr100b-20260906-r1`.
W&B group: `small-lr100b-20260906-r1`, project `olmoe3-production-profiling`.

Launch only from a clean, pushed `codex/small-lr100b-sweep`:

```bash
uv run --no-project --with beaker-py==2.7.2 python \
  src/examples/olmo_ddp/olmoe3_lr_sweep_launch.py \
  --output /absolute/path/to/launch-record --submit
```

The controller source commit is pinned in every child spec. Changing a spec
after a submission intent fails closed; use a reviewed revision/receipt migration
instead of editing a live ledger to bypass checks.

## Higher-LR extension

The September 6 extension adds **1.04e-2**, twice the previous upper LR of 5.2e-3.
The registry now contains six LRs / 30 production trajectories. Nothing in the
original 25 run definitions, training configuration, or submitted specs changes.
No lower-LR extension was submitted.

Leave the original controller pinned at `215df45915c871827cb096d60dd05ddf5a4e11a0`.
Launch the extension from the clean pushed branch with:

```bash
uv run --no-project --with beaker-py==2.7.2 python \
  src/examples/olmo_ddp/olmoe3_lr_sweep_launch.py \
  --extension 1p04em2 --output /absolute/path/to/extension-record --submit
```

This controller uses a separate `launch2/extension-1p04em2` ledger and lock, a
uniquely named actual-image config gate, and the already-passed save/restore smoke
(`01M1TW62KAPQ7GS5CQJDT2EKQN`, verified against its original source pin). It only
registers and submits the new trunk and its four decays. The new jobs pin the
extension commit; the original controller, receipts, and jobs remain untouched.
Its incremental free-space gate is 7 TB (above the new LR's <6 TB footprint even
without deletion). The existing uploader serves all five new lineages using
unchanged apply/grace policies: keep seven for the trunk, one for each decay.
The new run uses the same W&B group and is included in the combined U-plot tracker.
