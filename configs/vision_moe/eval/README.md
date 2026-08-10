# Stage 2 step-200 evaluation

These are standalone, single-node EP8 Beaker specifications for the permanent Stage-2
step-200 checkpoint and its matched checkpoint comparisons. They target `ai2/holmes`,
which supplies B300 GPUs, with
`priority: urgent`, `minRuntime: 8h`, budget `ai2/oe-other`, and only Rustin's Beaker and
W&B secrets.

The existing specs are pinned to the immutable `vision-moe` commit that contains their
evaluator fixes. Submit each spec to the requested workspace:

```bash
beaker experiment create -w ai2/molmofication \
  -n s002-stage2-step200-olmes-fast \
  configs/vision_moe/eval/stage2_step200_olmes_fast.yaml
beaker experiment create -w ai2/molmofication \
  -n s002-stage2-step200-mmmu-pro-standard \
  configs/vision_moe/eval/stage2_step200_mmmu_pro_standard.yaml
beaker experiment create -w ai2/molmofication \
  -n s002-stage2-step200-mmmu-pro-vision \
  configs/vision_moe/eval/stage2_step200_mmmu_pro_vision.yaml
beaker experiment create -w ai2/molmofication \
  -n s002-stage2-fast-vision-checkpoint-comparison \
  configs/vision_moe/eval/stage2_fast_vision_checkpoint_comparison.yaml
```

The count discriminator compares candidate-normalized first-token NLL/top-1 over answers 2-10
with raw digit/EOS and response-prefix mass, while `point_count` CE measures the grounded
training format on the exact same source indices as basic pointing. The spec is pinned to
immutable evaluator commit `771c954772413c378e36fc01dc57a3409529eafe`:

```bash
beaker experiment create -w ai2/molmofication \
  -n s002-stage2-count-discriminator-checkpoint-comparison \
  configs/vision_moe/eval/stage2_count_discriminator_checkpoint_comparison.yaml
beaker experiment create -w ai2/molmofication \
  -n s002-stage1-parent-count-discriminator-retry \
  configs/vision_moe/eval/stage1_parent_count_discriminator_retry.yaml
```

Both specs use job-local `/results` for `TMPDIR`, avoiding shared-Weka temporary-file
failures. The single-task retry only repairs the missing Stage-1-parent measurement and
does not alter the Stage-2 checkpoints or constitute a corrective-training ablation. A
causal corrective ablation should resume from Stage 2 step 50 and compare at step 200
against the existing control; step 200 is not the causal branch point.

The grounded final-count discriminator isolates `point_count` and compares the exact same
512 examples across the Stage-1 parent, Stage-2 step 50, and Stage-2 step 200. Before
submission, replace every literal `REPLACE_WITH_IMMUTABLE_GIT_SHA` with the immutable
`vision-moe` commit containing the grounded final-count evaluator, then verify that no
placeholder remains:

```bash
beaker experiment create -w ai2/molmofication \
  -n s002-stage2-grounded-final-count-discriminator-checkpoint-comparison \
  configs/vision_moe/eval/stage2_grounded_final_count_discriminator_checkpoint_comparison.yaml
```

The predeclared GO decision requires all four gates: Stage-2 step 200 grounded
`point_count` CE must be at least 5% better than Stage 1 and no more than 5% worse than
Stage-2 step 50; its final-count top-1 accuracy must be within 3 percentage points of the
better of Stage 1 and step 50; and its final-count NLL must be within 0.10 nat of the better
baseline. These thresholds must not be changed after inspecting results.

Do not launch until the checkpoint health audit confirms that `step200` is permanent,
all 16 trainer states agree on step 200, and a native distributed load succeeds.

## Exact comparison artifacts

| Artifact | Path or immutable ID | Existing matched result |
|---|---|---|
| s002 text base | `/weka/oe-training-default/robertb/s002-step125500` | `evals/s002-step125500/olmes-fast-complete-rerun-20260804.json` |
| s002 Stage 1 parent | `/weka/oe-training-default/rustin/experiments/vision-moe/checkpoints/s002-stage1-corrected-clean-32k-b300-20260807/step32000` | `evals/s002-stage1-corrected-clean-32k-b300-20260807-step32000/` |
| s002 Stage 2 step 50 | `/weka/oe-training-default/rustin/experiments/vision-moe/checkpoints/s002-stage2-v9-pilot-bounded-errors-5a81c40c/step50` | not yet evaluated downstream |
| s002 Stage 2 step 200 | `/weka/oe-training-default/rustin/experiments/vision-moe/checkpoints/s002-stage2-v9-pilot-bounded-errors-5a81c40c/step200` | produced by these specs |
| Molmo2-4B Stage 1 | `/weka/oe-training-default/rustin/experiments/vision-moe/artifacts/molmo2-4b-pretrain/hf-bf16` | `evals/Molmo2-4B-Pretrain/lmms-mmmu-pro-letter-logits-complete-20260804.json` |
| Molmo2-4B released | `allenai/Molmo2-4B@042abfa7a38879a376cec03d949eff0aefaa0600` | `evals/Molmo2-4B/lmms-mmmu-pro-letter-logits-complete-20260804.json` |

All relative result paths above are under
`/weka/oe-training-default/rustin/experiments/vision-moe/evals`.

Existing letter-logit MMMU-Pro reference scores are:

| Model | Vision | Standard |
|---|---:|---:|
| s002 Stage 1 step 32000, document interface | 0.11792 | 0.12023 |
| Molmo2-4B Stage 1 | 0.14393 | 0.29249 |
| Molmo2-4B released | 0.21098 | 0.35029 |

The Stage-2 jobs use the model's trained `olmo3_chat` interface and eight crops per image.
The existing s002 Stage-1 numbers use its trained document interface and legacy shared crop
budget. This is the correct stage-matched comparison. A later 2-by-2 prompt ablation can run
both checkpoints under both interfaces to isolate weight updates from the interface change.

The fast-vision comparison deliberately holds the OLMo3-chat interface fixed across the
Stage-1 parent and both Stage-2 checkpoints. It therefore measures the combined effect of
learning that interface and updating the weights, rather than Stage-1's native document
interface. It also evaluates each checkpoint with its serialized MoE capacity factors, so
it is a native-checkpoint comparison rather than a capacity-normalized weights-only test.

For OLMES fast, compare all task-level metrics and report at least ARC Challenge and ARC Easy
accuracy, four-domain MMLU macro accuracy, six-domain Basic Skills macro accuracy, CopyColors
accuracy, HellaSwag BPB, Minerva Math 500 BPB, HumanEval BPB, and MBPP BPB. Existing summary
anchors are:

| Metric | s002 base | s002 Stage 1 step 32000 |
|---|---:|---:|
| ARC Challenge accuracy | 0.75683 | 0.77218 |
| ARC Easy accuracy | 0.90109 | 0.90572 |
| MMLU macro accuracy | 0.60434 | 0.60701 |
| Basic Skills macro accuracy | 0.83554 | 0.81445 |
| CopyColors accuracy | 0.89000 | 0.87000 |
| HellaSwag BPB | 0.69155 | 0.69950 |
| Minerva Math 500 BPB | 0.44159 | 0.47014 |
| HumanEval BPB | 0.30788 | 0.36680 |
| MBPP BPB | 0.47761 | 0.52112 |

OLMES fast is a completion-format language-retention evaluation. It does not apply the OLMo3
chat template and must not be described as an instruction-following benchmark.
