# Mixed midtraining

Run `src/scripts/train/Mixed-Midtraining.py` after the joint phase of
[vision alignment](vision_alignment.md). It uses the same internal experiment runner,
dataclass configuration and dotted overrides. The implementation is split between
`olmo_core.internal.vision_midtraining` (training) and
`olmo_core.internal.vision_midtraining_data` (sources and loss allocation).

## Configure and launch

Inspect a configuration without loading weights, allocating text sources or replaying data:

```bash
python src/scripts/train/Mixed-Midtraining.py dry_run mixed-midtraining local \
  --recipe.parent_checkpoint=/path/to/alignment-joint/stepN
```

Text receives 90% of expected supervised-loss mass by default. The same recipe supports
text-only training:

```bash
python src/scripts/train/Mixed-Midtraining.py dry_run text-midtraining local \
  --recipe.parent_checkpoint=/path/to/alignment-joint/stepN \
  --recipe.text_loss_share=1
```

The text-only path does not construct visual sources or forward dummy image crops. It keeps
the same text allocation, model, LM/connector optimizer settings and schedule, while freezing
vision with zero vision LR and no vision checkpointing. In mixed runs, an all-text local
batch still forwards dummy crops to keep vision collectives aligned across ranks.

Once the code is available in the remote repository commit, launch with the standard CLI:

```bash
python src/scripts/train/Mixed-Midtraining.py launch mixed-midtraining ai2/holmes \
  --recipe.parent_checkpoint=/path/to/alignment-joint/stepN
```

Defaults are two eight-GPU nodes, EP8, urgent priority, an eight-hour minimum runtime,
32 GiB shared memory and workspace `ai2/molmofication`. These are ordinary `--launch.*`
settings. There is no wall-clock training deadline. `allow_dirty` does not upload local or
untracked changes: the standard launcher clones a remote commit. `prep` prepares datasets
and is not a mandatory pre-launch audit. `train` runs under an existing distributed allocation.

## Training defaults

| Setting | Default |
| --- | --- |
| Parent | Completed joint-alignment checkpoint; model-only handoff |
| Trainable components | Vision encoder, connector and LM, including full input/output embeddings |
| Context / global sequences / microbatch per GPU | 8,192 / 128 / 2 |
| Gradient accumulation | Four microbatches per GPU on 16 GPUs |
| Token-position budget | 50B, rounded up to 50,000,297,984 (47,684 steps) |
| Learning rates | LM `1e-5`; connector `2e-5`; vision `1e-6` (zero vision weight decay) |
| Schedule | 200-step warmup, cosine decay to 10% at the token budget |
| Activation checkpointing | LM blocks, vision encoder and connector |
| Packing | 48-example buffer, 16-crop packed-sequence limit, eight loader workers |
| Image preprocessing | Eight local detail crops plus one overview per image |
| Checkpoints | Every 10k steps, keeping two; latest temporary checkpoint every 500 steps |

The model architecture, tokenizer and router coefficients come from the alignment parent.
No router-input RMS repair is installed. Execution uses FlexAttention, EP8 dispatch and
block recomputation. Packing is not yet the bridge's 64-crop configuration; that change
requires its own throughput, exposure and quality comparison.

For a frozen-vision control, use the ordinary component overrides:

```bash
--train_module.freeze_params='["vision.*"]' \
--train_module.vision_activation_checkpointing=false
```

Frozen parameters are excluded from the optimizer. No separate arm profile is required.

`recipe.max_tokens` controls both the text allocation and scheduler horizon. The budget is
independent of `recipe.text_loss_share`; shares control sampling, not dataset construction.
For a short canary, override `trainer.max_duration` without changing the schedule:

```bash
python src/scripts/train/Mixed-Midtraining.py dry_run mixed-canary local \
  --recipe.parent_checkpoint=/path/to/alignment-joint/stepN \
  --trainer.max_duration.unit=steps --trainer.max_duration.value=2
```

A fresh handoff resets optimizer, loader and trainer state. Resuming this recipe in its
existing output folder restores full state, including packing and RNG state. Output folders
must be separate from the alignment parent. Historical joint checkpoints are accepted when
their model and tokenizer metadata are compatible; archived mixed-run full-state resumes
still require their original frozen runtime. No existing checkpoint is modified.

## Sources and allocation

Text uses `OLMo3-32B-midtraining-modelnamefilter.yaml`: the existing 61-source mixture,
including its instruction sources, Dolma2 tokenization and repetition filter. It is not
inferred from alignment's pretraining replay. An explicit `recipe.text_dataset` supports
another compatible fixed-length text dataset; a different parent tokenizer requires this.

The [source inventory](../../../configs/vision_moe/vision_midtraining/README.md) lists the
seven visual sources. Grounded count, scalar-answer replay, point styles, document formatting
and prepared train/validation exclusions are retained. The visual recipe does not add Tulu.

`recipe.text_loss_share` accepts values in `(0, 1]` and sets the outer supervised-loss allocation.
Within vision, `recipe.visual_loss_shares` reserves conditional loss shares for OCR and audited
alignment.
The remaining visual mass preserves `recipe.visual_example_weights`. Sampling uses
`target_loss_mass / mean_loss_weight`; these are neither input-token nor image-count shares.

Sources are ordinary configs in `dataset.sources`. To add or replace a source, supply its
config, calibrated `dataset.mean_loss_weight.SOURCE`, and its membership in one of the two
visual weight mappings. The groups must be disjoint and cover the visual sources. Final loss
targets and training diagnostics are derived from these settings after component overrides.
Use the text-share knob rather than overriding `dataset.target_loss_mass` directly.

Calibration is bounded sampling, not a full corpus replay:

```python
from olmo_core.data.multimodal.alignment import MultimodalMixtureConfig

dataset = MultimodalMixtureConfig.from_file("visual-sources.yaml")
means = dataset.estimate_mean_loss_weights(samples_per_source=128, seed=6198)
```

The input config must use the selected model's tokenizer/revision and source serialization.
Unit targets suffice when measuring per-source means. Default means are supplied for the
documented visual recipe; changes require explicit calibrated means. Unmasked 8K text has
8,191 next-token labels; masked mixed text needs its own measured mean. The reserved grounded
count validation split remains separate from scalar-count diagnostics.

## Validation

```bash
pytest -q src/test/internal/vision_midtraining_test.py \
  src/test/internal/vision_midtraining_data_test.py \
  src/test/data/multimodal/text_only_collator_test.py
```

Configuration and CPU tests do not validate distributed training stability. Before a full
run, use a short optimizer/checkpoint/resume canary from the selected alignment parent.
Training callbacks can be extended through the standard configuration API; decoded vision
and text benchmark suites remain separate evaluations.
