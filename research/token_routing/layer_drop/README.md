# layer_drop: train-time whole-block skipping

Model side: `src/olmo_core/nn/layer_drop.py` (`Transformer.enable_layer_drop`), driven by
`LayerDropCallback`; trainer flags `--train-skip-layers` / `--layer-drop-*` in
`recipes/train/research_trainer/train_ctc_suite.py` (the `layerdrop` modifier in
`recipes/train/variants.py`, composes with `--variant full` and `--variant softtoken`).
FLOPs: `FlopMeterCallback` charges only the blocks that ran.

- `smoke_layerdrop.sh`: 2-GPU wall-clock smoke, dense / dense+LayerDrop / tier67a50m32 /
  tier67a50m32+LayerDrop on the fast8k mt4s488 shard. Logs and run dirs stay on the node's
  `/data/prasann/layerskip/` (not in git).
