# Vision-MoE Stage-1 launch gates

These profiles exercise the production EP8 topology for one optimizer step on two
8-GPU B300 nodes in `ai2/holmes`. They use workspace `ai2/molmofication`, budget
`ai2/oe-other`, urgent priority, and a `1h` minimum guaranteed runtime. Holmes supplies
the B300 hardware, so the profiles do not add a redundant GPU-type constraint.

- `stage1_ep8_2node_synthetic_1step.yaml` checks distributed startup, native s002 loading,
  vision-weight loading, optimizer construction, forward/backward, and checkpoint writing
  without depending on the production datasets.
- `stage1_ep8_2node_real_1step.yaml` adds the production PixMoCap, pointing/counting, and
  Tulu4 input mixture and is the final one-step gate before a longer Stage-1 run.

Inspect the fully merged configuration without submitting:

```bash
PATH=/weka/oe-training-default/rustin/envs/olmo-core-vision/bin:$PATH \
python src/scripts/train/Molmo2-Stage1.py dry_run stage1-real-gate \
  --beaker-test-config=configs/vision_moe/stage1_ep8_2node_real_1step.yaml
```

After reviewing the dry run and receiving explicit submission approval, replace `dry_run`
with `launch`. Explicit CLI overrides come after the profile's training overrides and
therefore take precedence. Launch topology and target fields are taken from the profile.
