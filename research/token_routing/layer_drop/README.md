# layer_drop: train-time whole-block skipping

Model side: `src/olmo_core/nn/layer_drop.py` (`Transformer.enable_layer_drop`), driven by
`LayerDropCallback`; trainer flags `--train-skip-layers` / `--layer-drop-*` in
`recipes/train/research_trainer/train_ctc_suite.py` (the `layerdrop` modifier in
`recipes/train/variants.py`, composes with `--variant full` and `--variant softtoken`).
FLOPs: `FlopMeterCallback` charges only the blocks that ran.

- `smoke_layerdrop.sh`: 2-GPU wall-clock smoke, dense / dense+LayerDrop / tier67a50m32 /
  tier67a50m32+LayerDrop on the fast8k mt4s488 shard. Logs and run dirs stay on the node's
  `/data/prasann/layerskip/` (not in git).

## Smoke 2026-10-22 (sneetches, 2x H200, jsteinhardt preemptive; slurm 4017466 + 4017497)

Qwen3.5-4B `q35-4b-base-markerfix`, `mt4s488`, seq 10240, gb 16, micro 2, unpacked, no compile,
lr 5e-6, seed 0, 12 steps (`--max-steps 12`). s/step = mean over steps 2-12. FLOPs from
`flops.json` (`per_example`). The LayerDrop draw is seeded from (seed, step), so every LD arm with
the same flags skips the same blocks (e.g. step 10 of the p=0.25 arms: [1,2,3,4,5,9,27,29]).
CE is the training loss of the skipped network after a handful of steps: a smoke, not a quality result.

| arm | s/step | wall x dense | actual PF | FLOPs x dense | blocks run | CE step 1 / 12 | CE max | CE mean 7-12 |
|---|---|---|---|---|---|---|---|---|
| dense | 8.41 | 1.00 | 41.81 | 1.000 | 1 | 1.18 / 0.41 | 1.27 | 0.351 |
| dense + LD p=0.25 uniform | 6.69 | 0.80 | 31.64 | 0.757 | 0.721 | 2.79 / 3.40 | 14.00 | 2.650 |
| dense + LD p=0.4 linear, excl. first 4 / last 2 | 7.37 | 0.88 | 34.39 | 0.823 | 0.802 | 2.57 / 0.88 | 2.57 | 1.080 |
| dense + fixed skip 20,24,28 (GDN) | 7.92 | 0.94 | 38.61 | 0.924 | 0.906 | 1.33 / 0.42 | 1.42 | 0.347 |
| tier67a50m32 | 5.73 | 0.68 | 23.75 | 0.568 | 1 | 1.20 / 0.44 | 1.26 | 0.402 |
| tier67a50m32 + LD p=0.25 uniform | 4.51 | 0.54 | 17.99 | 0.430 | 0.721 | 2.77 / 3.34 | 14.03 | 2.624 |
| tier67a50m32 + LD p=0.4 linear, excl. 4 / 2 | 4.89 | 0.58 | 19.55 | 0.468 | 0.802 | 2.55 / 0.87 | 2.55 | 1.082 |

- Wall-clock tracks FLOPs: skipping saves real time (dense+LD: 0.80x wall at 0.76x FLOPs; the gap
  is the un-skippable embedding / LM head / optimizer step / data path).
- Uniform p=0.25 on the pretrained 4B wrecks the loss (drops early blocks 1-5): exclude the early
  blocks. Fixed skip of three late GDN blocks costs nothing in CE here (0.347 vs 0.351) for 0.92x
  FLOPs. Whether any of this keeps the eval score is the screen's job, not this smoke's.
- Caveat (pre-existing, all arms alike): `examples` = 208 for 12 x 16 rows, i.e. the trainer's dry-run
  batch is charged to step 1 by the per-example meter.
