# Vision-MoE Stage-1 launch gates

These profiles exercise EP8 topology on one or two 8-GPU B300 nodes in
`ai2/holmes`. They use workspace `ai2/molmofication`, budget `ai2/oe-other`, and urgent
priority. Legacy real-data gates and pilots request no minimum runtime. Holmes supplies the
B300 hardware, so the profiles do not add a redundant GPU-type constraint. The 4k-to-8k
continuation, clean 32k run, and corrected 200-step topology gates request an eight-hour
minimum runtime.

The corrected profiles are topology and serialization baselines, not claims of byte-exact
released-recipe parity. Their 10% text source is the pinned OLMo 3 no-tools SFT dataset rather
than released Molmo2's `allenai/molmo2-tulu4-classified`, and response-only residual dropout
remains intentionally disabled pending an exact MoE-aware implementation and isolated test.

- `stage1_ep8_2node_synthetic_1step.yaml` checks distributed startup, native s002 loading,
  vision-weight loading, optimizer construction, forward/backward, and checkpoint writing
  without depending on the production datasets.
- `stage1_ep8_2node_real_1step.yaml` adds the production PixMoCap, pointing/counting, and
  Tulu4 input mixture and is the final one-step gate before a longer Stage-1 run.
- `stage1_ep8_2node_real_resume_2step.yaml` restores the real-data gate's full step-1
  checkpoint into a separate run folder, executes step 2, and verifies that model,
  optimizer, scheduler, data-loader, and trainer state are resumable.
- `stage1_ep8_1node_real_200step_micro8.yaml` and
  `stage1_ep8_1node_real_200step_micro16_recompute.yaml` form a controlled one-node
  comparison of two accumulated eight-sequence forwards against one sixteen-sequence forward.
  Both use standard OLMo per-block recomputation. The sixteen-sequence arm failed without
  recomputation but completed a full real-data local B300 optimizer-step gate with it enabled.
- `stage1_ep8_2node_real_200step_micro8.yaml` validates the same corrected recipe with two
  EP-DP replicas, which is the intended multi-node production topology.
- `stage1_ep8_2node_real_500step_pilot.yaml` runs an exact 500-step prefix of the 32,000-step
  production schedule with the native s002 router loss weights, FP32 gradient
  accumulation/reduction, and padding-excluded routed-expert traffic.
- `stage1_ep8_2node_real_resume_to8000_b300.yaml` restores the corrected step-4000 model,
  optimizer, scheduler, trainer, and data state, then continues to step 8000. It disables
  LM block recomputation on B300 while retaining vision and connector checkpointing.
- `stage1_ep8_2node_real_32k_b300.yaml` starts a clean 32,000-step run from the native s002
  language checkpoint and pinned pristine SigLIP2 tower. It uses the corrected data layout,
  released nine-crop packing constraint, 32k LR horizon, and the same B300 recomputation
  optimization.

Inspect the fully merged configuration without submitting:

```bash
PATH=/weka/oe-training-default/rustin/envs/olmo-core-vision/bin:$PATH \
python src/scripts/train/Molmo2-Stage1.py dry_run stage1-real-gate \
  --beaker-test-config=configs/vision_moe/stage1_ep8_2node_real_1step.yaml
```

After reviewing the dry run and receiving explicit submission approval, replace `dry_run`
with `launch`. Explicit CLI overrides come after the profile's training overrides and
therefore take precedence. Launch topology and target fields are taken from the profile.
