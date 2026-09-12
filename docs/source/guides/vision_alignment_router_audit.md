# Vision-alignment router audit

This audit covers the s002 parent, the historical alignment recipe, and integration canary
`integration-canary-20260906-v4`. The canary tests checkpoint handoffs and resume; its eight
bridge updates, eight perception updates, and 32 joint updates do not establish convergence.

The measurements and proposed comparisons below describe that historical source snapshot,
not the current launch defaults. The [native recipe](vision_alignment.md) now disables LB
for bridge, uses CF8 with shared dispatch buffers, and preserves the parent's LB policy in
later phases unless explicitly restored or overridden. Neither native alignment nor
[mixed midtraining](mixed_midtraining.md) applies the historical router RMS repair.

## Settings and inheritance

The s002 checkpoint and historical alignment use the same routed-router settings:

| Setting | Value |
| --- | --- |
| Routed experts / active experts | 64 / 4 |
| Gating | Softmax |
| Selected-weight normalization | L1, then multiply by 4 |
| Load-balancing loss | 0.015, per instance |
| Router Z loss | 0.0001 |
| Expert-parallel degree | 8 |
| Destination-rank capacity factor | 2 for blocks 1–3; 1.1875 for later blocks |

Block 0 has shared experts only. The router linear projection and softmax use FP32. Disabling
the multiply-by-4 operation would change the pretrained expert contribution; it is not an
unintended amplification.

At this audit's source snapshot, a forced load-balancing coefficient of 0.015 was replaced
with inheritance of the parent value, including block-specific overrides and disabled
losses. This preserved the s002 recipe; the later bridge LB-off default is a separate choice.

Bridge and perception freeze `lm.blocks.*`, including the routers and their input norms.
The model remains in training mode: router auxiliary losses still backpropagate through those
frozen modules to trainable image inputs and the connector. Freezing parameters does not disable
these objectives. Whether to remove the frozen-LM auxiliary objectives is an experimental choice,
not an established implementation fix.

Alignment uses Adam epsilon `1e-6`, matching the historical alignment recipe. Pretraining and
the historical repaired 50B midtraining recipe use `1e-8`. A larger epsilon attenuates updates when the
second-moment scale is small, but no measurement here establishes that changing it improves
alignment. Keep this separate from the router-input experiment.

## Canary measurements

| Endpoint | Block-1 load imbalance | Dropped routes | Packing fill | Maximum buffer utilization |
| --- | ---: | ---: | ---: | ---: |
| Bridge 8 | 15.9309 | 14.2717% | 52.3285% | 100% |
| Perception 8 | 15.9008 | 12.7433% | 49.2468% | 100% |
| Joint 16 | 15.7266 | 0% | 14.9960% | 55.8716% |
| Joint 32 | 15.5576 | 0% | 14.5271% | 37.5092% |

Load imbalance is the rank-local maximum expert count divided by its mean, followed by a maximum
reduction across ranks. With 64 experts and top-4 selection its maximum is 16. A value of 15.93
means that at least one rank assigns approximately 99.6% of its valid tokens to its most-used
expert. It does not establish that all four assignments, or the busiest expert across ranks,
are identical.

Historical final W&B summaries already show substantial first-block concentration:

| Historical run | Block-1 load imbalance | Dropped routes | Packing fill |
| --- | ---: | ---: | ---: |
| `vision-alignment-bridge-real-canary-v1` | 15.6733 | 0.1368% | 49.8447% |
| `vision-alignment-bridge-real-v1` | 15.8931 | 2.4775% | 50.7803% |
| `vision-alignment-joint-v1` | 2.2154 | 0% | 14.4321% |

The later historical joint endpoint is not a step-matched comparison with the integration
canary.

### Padding and capacity

Rowwise dispatch allocates `ceil(capacity_factor * B * S * top_k)` rows per destination rank.
Allocation uses physical padded shape because symmetric buffer shapes must agree across ranks.
Padding is excluded from requested traffic, auxiliary losses, and the drop-rate denominator.
Unused padded slots therefore provide capacity headroom.

Joint's zero dropping does not demonstrate recovered routing: its lower fill supplies much more
headroom, while its load imbalance remains high. Drops refer to expert assignments, not deleted
image positions. Residual, attention, and shared-expert paths remain active. Current canary metrics
do not identify which modalities lost routes; they cannot establish that dropping caused the
real-image/blank-image loss gap to disappear.

## First-router input

Selective CPU reads of the bare s002 checkpoint produced these affine input-norm statistics:

| Block | Median absolute weight | 90th percentile absolute weight | RMS weight |
| --- | ---: | ---: | ---: |
| 1 | 0.00142963 | 0.01113225 | 0.19620846 |
| 2 | 1.76480067 | 4.18030882 | 2.55286098 |
| 3 | 1.92668843 | 4.13678646 | 2.56156921 |
| 12 | 0.88832492 | 1.19921184 | 0.89954388 |

Most first-block dimensions are strongly attenuated, with a few much larger dimensions. This is
inherited pretrained state, not evidence of a new canary initialization error.

The existing VA12k fixed-batch, zero-update probe compared the original affine router input with
parameter-free RMS of the pre-feed-forward residual. Mean top-4/top-5 logit margin increased from
`0.0001447775` to `0.04331566`; image-patch margin increased from `0.0001523444` to `0.04410800`.
The original all-token normalized entropy was `0.999999885`, with top-1 probability `0.01567093`
close to `1/64`. The repair leaves both routed and shared experts' affine-normalized inputs
unchanged. It was used in the historical midtraining overlay, not the alignment model or
the current native mixed-midtraining recipe.

These logits were measured on VA12k using multimodal Stage-1 inputs, not the bare pretrained
model on native text. The same intervention changed the top-four expert set for all 136,230
measured tokens. Global requested-load imbalance increased from 2.11783 to 5.23937, while
weighted response CE changed by only -0.0004333. A larger routing margin alone is not evidence
of better expert matching or balance. The intervention remains an architecture experiment,
not a demonstrated correction to checkpoint loading.

Load-balancing loss alone cannot diagnose weak routing. A CPU synthetic example with fixed router
weights produced imbalance 16, normalized entropy `0.999999881`, and unscaled balancing loss
`1.00000477`. It also confirmed nonzero upstream auxiliary gradients with frozen router weights.
This demonstrates metric semantics, not the canary's causal mechanism.

## Historical adaptation-validation plan

Preserve the parent's architecture and per-layer routing settings by default. No norm reset,
first-block exception, or fixed expert-count assumption is part of the integrated recipe.
The following bounded comparisons were proposed for the audited recipe. Reproducing them
requires its frozen source deployment; they are not current launch instructions:

1. Compare unpadded native text through the multimodal wrapper and its same loaded LM, using
   the same attention backend, precision and dispatch. Check logits and expert assignments.
   This tests wrapper equivalence, not independent checkpoint loading or Flash-versus-Flex
   equivalence. Read a small explicit subset of the parent's native data for this smoke test.
2. Compare the unchanged 500-step bridge with an otherwise identical bridge using
   `recipe.router_lb_loss_weight=0`. Router z-loss and CE remain enabled. The override is
   serialized in the model config: later phases must explicitly choose whether to re-enable
   balancing. In that snapshot, `None` preserved the checkpoint's value, including zero;
   the current native bridge interprets `None` as LB off.
3. Save first-step connector gradients before the optimizer update, plus batch identity, for
   the two arms. Their difference isolates the balancing contribution only when weights,
   inputs, RNG and other objectives agree. Do not treat `.grad` as the authoritative buffer:
   OLMoDDP can store reduced FP32 gradients separately.
4. Evaluate the fresh initialized model in a separate process with larger fixed dispatch
   capacity. With equal physical token counts on each EP rank, capacity factor equal to the
   EP degree bounds worst-case destination traffic. Configure this before buffer allocation;
   do not resize live symmetric buffers as an evaluation-time patch.

The optional native-text callback also records requested routing separately for text,
image-patch and image-structural tokens on one batch from each named validation evaluator.
Those histograms are pre-dispatch and must not be presented as accepted-route coverage.
The generic diagnostics infer layer names, expert counts and top-k from supported V2 routers;
other router implementations require an explicit output adapter. Dense models need no router
adapter. Neither the diagnostics nor the balancing override was enabled in the audited recipe.

The retired `bridge_validation/ablation.yaml` wrapper is preserved in
`/weka/oe-training-default/rustin/experiments/vision-moe/vision-alignment/artifacts/bridge-code-archive-20260911-v1/source.tar.gz`.
It runs these comparisons sequentially in one two-node allocation, with separate outputs and
the same immutable source snapshot. The replay-quota experiment is deliberately not combined
with this bridge comparison. Establish behavior on the next pretraining checkpoint too before
promoting a checkpoint-specific outcome to a general alignment default.

Do not scale symmetric capacity by rank-local valid-token counts. Do not interpret additional
padding or a larger capacity factor as a routing repair. Any capacity-only comparison must hold
the actual examples and token masks fixed.

If router normalization is eventually decoupled, preserve the initial function: clone the
existing affine scale into the new router norm, or fold it into the router matrix. Removing
the affine scale while retaining the old matrix does not preserve pretrained logits.

### Metrics persistence

The first 500-step validation stopped after completing step 100 because the metric saver
rejected the second write to that step's JSON file. The per-file overwrite fix replaces
metric snapshots atomically without enabling checkpoint overwrites. In the archived
`bridge-validation-20260906-v1` output, `metrics-final.json` contains the completed step-100
evaluations; `metrics_step100.json` still contains startup evaluation values. Do not use the
latter as step-100 evaluation results.

## Evidence and checks

- Parent: `/weka/oe-training-default/robertb/s002-step125500/config.json` and
  `model_and_optim`, selectively reading five small tensors on CPU, not a full model.
- Canary: `/weka/oe-training-default/rustin/experiments/vision-moe/vision-alignment/artifacts/integration-canary-20260906-v4/checkpoints/vision-align-canary-{phase}/metrics-stop{N}.json`.
- Historical W&B summaries: `/weka/oe-training-default/rustin/experiments/vision-moe/vision-alignment/checkpoints/{run}/wandb/wandb/run-*/files/wandb-summary.json`.
- Fixed-batch probe: `/weka/oe-training-default/rustin/experiments/vision-moe/vision-midtraining/artifacts/routing-probe-va12k-stage1-step2001-router-rms-ab.json`.
- Historical repaired production config: `/weka/oe-training-default/rustin/experiments/vision-moe/vision-midtraining/checkpoints/vision-midtraining-va12k-vt80-two-stage-count-replay-50b-mb4-v2/step47684/config.json`.
- Implementation: `src/olmo_core/nn/moe/v2/router.py`, `src/olmo_core/nn/moe/loss.py`,
  `src/olmo_core/nn/moe/v2/ep_no_sync_rowwise.py`,
  `src/olmo_core/nn/moe/v2/ep_no_sync_common.py`, and
  `src/olmo_core/train/train_module/transformer/multimodal_train_module.py`.
- Historical repair implementation: `configs/vision_moe/eval/vision_midtraining_checkpoint_suite_v10/training_overlay/`
  in the matching experiment deployment; it is not a dependency of the native recipes.

CPU checks:

```bash
PYTHONPATH=src CUDA_VISIBLE_DEVICES='' /opt/conda/bin/python -m pytest -q \
  src/test/nn/moe/router_v2_test.py \
  -k 'token_mask or restore_weight_scale or normalize_expert_weights or forward_shapes' \
  -p no:cacheprovider
PYTHONPATH=src CUDA_VISIBLE_DEVICES='' /opt/conda/bin/python -m pytest -q \
  src/test/internal/vision_alignment_test.py \
  -k router_load_balancing_is_inherited -p no:cacheprovider
```

The commands passed five router tests and three inheritance cases. The latter cover `None`,
zero, and a nondefault coefficient, a distinct per-block override, and all three phase handoffs.
No new GPU run was used for this audit.
