# Vision-MoE Stage-1 launch gates

These profiles exercise the production EP8 topology on two 8-GPU B300 nodes in
`ai2/holmes`. They use workspace `ai2/molmofication`, budget `ai2/oe-other`, and urgent
priority. Real-data and pilot profiles request no minimum runtime. Holmes supplies the
B300 hardware, so the profiles do not add a redundant GPU-type constraint.

- `stage1_ep8_2node_synthetic_1step.yaml` checks distributed startup, native s002 loading,
  vision-weight loading, optimizer construction, forward/backward, and checkpoint writing
  without depending on the production datasets.
- `stage1_ep8_2node_real_1step.yaml` adds the production PixMoCap, pointing/counting, and
  Tulu4 input mixture and is the final one-step gate before a longer Stage-1 run.
- `stage1_ep8_2node_real_resume_2step.yaml` restores the real-data gate's full step-1
  checkpoint into a separate run folder, executes step 2, and verifies that model,
  optimizer, scheduler, data-loader, and trainer state are resumable.
- `stage1_ep8_2node_real_500step_pilot.yaml` runs an exact 500-step prefix of the 31,000-step
  production schedule with the native s002 router loss weights, FP32 gradient
  accumulation/reduction, and padding-excluded routed-expert traffic.

Inspect the fully merged configuration without submitting:

```bash
PATH=/weka/oe-training-default/rustin/envs/olmo-core-vision/bin:$PATH \
python src/scripts/train/Molmo2-Stage1.py dry_run stage1-real-gate \
  --beaker-test-config=configs/vision_moe/stage1_ep8_2node_real_1step.yaml
```

After reviewing the dry run and receiving explicit submission approval, replace `dry_run`
with `launch`. Explicit CLI overrides come after the profile's training overrides and
therefore take precedence. Launch topology and target fields are taken from the profile.
