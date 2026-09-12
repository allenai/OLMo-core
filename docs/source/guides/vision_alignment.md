# Vision alignment

Use `src/scripts/train/Vision-Align.py` with the standard internal experiment CLI. The recipe
is defined in `olmo_core.internal.vision_alignment`; source defaults are in
`olmo_core.internal.vision_alignment_data`. Configuration uses ordinary dataclasses and
dotted overrides, as in the other training recipes.

## Stages and defaults

Vision alignment runs bridge, perception, then joint training. [Mixed midtraining](mixed_midtraining.md)
is a separate stage that consumes the resulting aligned checkpoint.

| Phase | Trainable components | Steps | Training context | Global sequences / microbatch per GPU |
| --- | --- | ---: | ---: | ---: |
| Bridge | Connector and six image-token input rows | 500 | 8,192 | 128 / 4 |
| Perception | Above plus vision encoder; LM body frozen | 4,000 | 2,560 | 128 / 4 |
| Joint | Above plus LM body; output projection frozen | 16,000 | 8,192 | 128 / 1 |

The bridge uses eight local detail crops per image, a 64-crop packed-sequence budget, a
48-example packing buffer, and eight loader workers. Its connector learning rate is
`2e-4`, with 100 warmup steps and a cosine horizon of 250 steps. It then remains at the
10% learning-rate floor (`2e-5`) through step 500. Bridge validation sources retain their 2,560-token
serialization cap; the evaluator pads those examples to 8,192 tokens for execution.

Bridge expert dispatch uses capacity factor 8 and shared dispatch output buffers with EP8.
Load-balancing loss is disabled by default **only for bridge**. Router z-loss, the pretrained
router architecture and model weights are otherwise preserved; no router RMS repair is used.
Perception and joint retain their existing phase defaults and inherit the parent's model
configuration, including its load-balancing coefficients.

The LM architecture and tokenizer configuration come from the pretrained checkpoint, not a
fixed model size. Phase parents preserve the tokenizer revision. The current bootstrap supports
OLMoDDP LMs, Molmo2-compatible tokenizers, reserved image-token rows, and untied input/output
embeddings. Dense bootstrap and composable text replay are not supported by this recipe.

## Configure and launch

Inspect a bridge configuration without loading model weights or replaying the corpus:

```bash
python src/scripts/train/Vision-Align.py dry_run align-bridge local \
  --recipe.pretraining_checkpoint=/path/to/pretraining/stepN
```

`dry_run` reads checkpoint, tokenizer, vision architecture and prepared source metadata.
`prep` constructs/prepares the data loader and can be expensive; it is not a mandatory
pre-launch audit. `train` runs inside an existing distributed allocation under `torchrun`.
The process count must match the configured parallelism.

Once the source is available in the remote repository commit, submit through the standard CLI:

```bash
python src/scripts/train/Vision-Align.py launch align-bridge ai2/holmes \
  --recipe.pretraining_checkpoint=/path/to/pretraining/stepN
```

Bridge launch defaults are two nodes with eight GPUs each, EP8, urgent priority, an eight-hour
minimum runtime, and 32 GiB shared memory. There is no wall-clock training deadline. Later
phases retain their existing launch settings; these remain ordinary `--launch.*` overrides.

The standard launcher clones a remote git commit. **`allow_dirty` does not upload local edits
or untracked files.** A dry run in this checkout does not establish that the remote commit
contains the same implementation. Use an explicitly frozen source deployment when evaluating
local changes that are not available remotely.

## Checkpoint handoffs and resume

Use a separate output folder for each phase and the actual completed parent checkpoint:

```bash
python src/scripts/train/Vision-Align.py dry_run align-perception local \
  --recipe.phase=perception \
  --recipe.parent_checkpoint=/path/to/align-bridge/step500

python src/scripts/train/Vision-Align.py dry_run align-joint local \
  --recipe.phase=joint \
  --recipe.parent_checkpoint=/path/to/align-perception/step4000 \
  --recipe.restore_pretraining_router_lb=true
```

A phase transition loads model weights and resets optimizer, data-loader and trainer state.
Resuming the same phase in its existing output folder restores full state, including packing
and RNG state. A restored step-zero checkpoint also skips fresh LM/vision initialization.
Historical parents are accepted when their saved phase, model and pretraining ancestry are
compatible.

An LB-off parent stays off in later phases unless explicitly changed.
`recipe.restore_pretraining_router_lb=true` restores the original pretrained per-layer
coefficients; it is appropriate when joint training should resume the pretrained balancing
policy. `recipe.router_lb_loss_weight=VALUE` instead sets a uniform coefficient. These options
are mutually exclusive and do not modify router z-loss or dispatch capacity.

Training duration and scheduler horizon are separate fields. For example,
`--trainer.max_duration.value=12000` stops joint training at 12k without changing its 16k
scheduler. Do not restart optimizer or loader state merely to extend a within-phase run.
Experiment-specific exposure-budget wrappers have additional endpoint checks; use their
explicit continuation path rather than changing a completed run's definition in place.

## Sources and calibration

`MultimodalMixtureConfig.sources` maps source names to ordinary dataset configs implementing
`build(tokenizer)`. `MultimodalSourceConfig` optionally applies prepared row selections.
Sources can be replaced or extended through configuration; there is no runtime source
allowlist. Bridge uses caption/transcript targets of 0.70/0.30. Perception adds pointing,
counting, OCR/document and audited alignment sources. The visual recipe does not include Tulu.

Targets are expected supervised-loss mass, not image counts, input tokens or fixed per-update
gradient fractions. Sampling probabilities are proportional to
`target_loss_mass / mean_loss_weight`. Default means describe the prepared populations,
tokenizer and phase serialization. Recalibrate when these change:

```python
from olmo_core.data.multimodal.alignment import MultimodalMixtureConfig

dataset = MultimodalMixtureConfig.from_file("my-sources.yaml")
means = dataset.estimate_mean_loss_weights(samples_per_source=128, seed=0)
```

Supply each source config, `dataset.target_loss_mass.SOURCE`, and
`dataset.mean_loss_weight.SOURCE` together. This is bounded sampling, not a full corpus replay;
the underlying dataset still incurs its ordinary preparation cost. Image/content disjointness
belongs in data preparation. Loader fingerprints protect resume identity; new training does
not require legacy source-code hashes or approval receipts.

Dataset `max_crops` limits local detail crops per image; one overview crop is added separately.
Loader `pack_max_crops` limits total crops per packed sequence. Packed examples have isolated
attention and independent positions. Set `recipe.sequence_length` to update coherent training
lengths; this does not change RoPE. Changes to source serialization, crops, packing or replay
splits require compatible calibration and a fresh data stream, not an incompatible resume.

## Native text replay

Joint uses `PretrainingReplayConfig(checkpoint=...)` to resolve the original LM's fixed-length
numpy dataset and saved text paths. It preserves storage dtype, file order, weighted mixtures,
repetition filtering and label-mask sidecars. Tokens are not retokenized, labels shift once,
and text examples carry no images. Storage URLs remain those recorded by the checkpoint;
the launch workspace must have access to them.

The default replay holdout reserves 1,024 windows with seed 6198 and excludes those windows
from joint replay. It is not a holdout from the original LM pretraining. Automatic splitting
requires unique, unweighted fixed-length paths; weighted or duplicate paths need an explicitly
disjoint validation dataset and `recipe.text_validation_size=0`. An explicit
`PretrainingReplayConfig.dataset` can replace checkpoint-derived replay. Masked replay requires
its own mean-loss-weight calibration.

Optional grouped loading separates per-update sequence allocation from objective weights:

```python
config.data_loader.global_batch_size = 128 * 8192
config.data_loader.source_groups = {
    name: "text" if name == "native_text_replay" else "vision"
    for name in config.dataset.sources
}
config.data_loader.group_sequence_quotas = {"text": 16, "vision": 112}
config.train_module.loss_group_weights = {"text": 0.35, "vision": 0.65}
```

For DP size 16, this allocates one text and seven visual sequence slots per rank. Quotas count
padded sequences, not supervised tokens. The CE objective is a 0.35/0.65 weighted sum of the
global text/vision weighted-mean losses; these are not gradient-norm shares. Groups pack
independently and retain calibrated source sampling within each group. Quotas must sum to the
global sequence count and be positive multiples of DP size. Each objective group needs positive
supervised mass on every update. Changing groups, quotas or DP size invalidates saved grouped
loader state. Grouped loading is optional, not a change to the default joint recipe.

## Evaluation and tests

The native callback measures deterministic held-out per-source CE with 64 examples/source for
bridge and 512 for later phases. Caption/transcript also use 64 correct/wrong-image pairs from
a bounded candidate pool, matched for crop geometry and pooling indices. Positive
wrong-minus-correct CE indicates image dependence; blank-image differences alone are
insufficient. Joint adds its native replay holdout. These checks do not replace decoded quality
or external benchmarks.

The experiment [index](../../../configs/vision_moe/vision_alignment/README.md) distinguishes
the seven-source decoded diagnostics from the six-task historical academic panel. Compare only
matching examples, prompts, generation limits and scoring definitions. Teacher-forced image
sensitivity is not decoded accuracy. Packing utilization alone is not a quality or speed result.

Focused tests cover the supported recipe and reusable components:

```bash
pytest -q src/test/internal/vision_alignment_test.py \
  src/test/internal/vision_alignment_bridge_test.py \
  src/test/internal/vision_alignment_data_test.py \
  src/test/data/multimodal/alignment_test.py \
  src/test/data/multimodal/pretraining_replay_test.py \
  src/test/data/multimodal/grouped_mixture_test.py \
  src/test/train/callbacks/multimodal_test.py
```

Use a short distributed canary for a new architecture or execution configuration, covering
initialization, frozen parameters, label masks, dispatch health and full-state resume. The
optional `train_module.trim_microbatch_image_padding` remains disabled by default; it requires
valid collator counts and zero vision dropout, and must be validated on the distributed path.

## Compatibility

`Vision-Alignment.py` and historical audit/evaluation adapters remain for saved configurations
and hash-bound evidence. They are not the entrypoint for new training. Shared experiment
modules containing serialized config/callback classes require their matching local or frozen
source deployment; those historical utilities are not included in the native recipe's source
distribution. Preserve them while their checkpoints are supported. The
[router audit](vision_alignment_router_audit.md) holds historical experimental evidence
separately from this API guide.
