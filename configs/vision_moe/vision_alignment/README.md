# Vision alignment continued pretraining

Vision alignment is a three-phase continued-pretraining recipe for attaching the Molmo2 vision
tower to the pretrained s002 language model. It is intentionally separate from the historical
Molmo2 Stage 1 and Stage 2 recipes: it owns its checkpoint namespace, optimizer state, data
cursor, and W&B run identity.

The implementation follows the same recipe shape as the other internal OLMo-core training
scripts:

- `src/scripts/train/Vision-Alignment.py` defines typed configuration, component builders,
  `build_config()`, `train()`, `launch()`, and `main()`;
- reusable datasets and mixture math live under `src/olmo_core/data/multimodal/`;
- checked-in YAML profiles contain only phase inputs and launch settings;
- Trainer owns same-run checkpoint resume and cross-phase model loading.

## Phases

| Phase | Parent | Trainable components | Sequence length | Default duration |
|---|---|---|---:|---:|
| `bridge` | bare s002 + pinned SigLIP2 | connector and six image-token input rows | 2,560 | 1,000 steps |
| `perception` | permanent bridge checkpoint | connector, vision tower, image-token rows | 2,560 | 4,000 steps |
| `joint` | permanent perception checkpoint | connector, vision tower, LM blocks/norms/routers, image-token rows | 8,192 | 16,000 steps |
| `joint` frozen-vision control | frozen-vision perception checkpoint | connector, LM blocks/norms/routers, image-token rows | 8,192 | 16,000 steps |

The joint phase still freezes the ordinary lexical embedding rows and the untied output
projection. Phase changes are model-only forks: they load model parameters from the pinned
parent while starting a fresh optimizer, trainer state, RNG stream, and data cursor. An
interruption within one unchanged phase resumes the newest checkpoint in that phase's own save
folder with full state.

## Profiles

The active profiles are:

- `bridge/synthetic_smoke.yaml`: versioned one-step code and topology smoke test;
- `bridge/real_bridge_v1.yaml`: production connector-only bridge;
- `perception/treatment_v1.yaml`: production vision-unfrozen perception phase;
- `joint/joint_v1.yaml`: production joint phase with native text replay.
- `joint/frozen_vision_control_v1.yaml`: joint control with identical data and the vision encoder
  frozen throughout the complete alignment lineage.

The joint trainability arm changes only the parent and vision-encoder trainability. The original
joint treatment remains the default, and its data and trainable contract hashes remain compatible
with the completed checkpoints.

Profiles use strict YAML parsing, may not repeat override destinations, and own their phase
selector. Production perception and joint profiles must be checked-in files directly under
their phase directory and do not accept additional command-line overrides. This keeps the
recipe reviewable without maintaining a second SHA allowlist for Git-tracked profile bytes.

Inspect a profile without submitting:

```bash
PYTHONPATH=src python src/scripts/train/Vision-Alignment.py dry_run \
  vision-alignment-bridge-real-v1 \
  --profile=configs/vision_moe/vision_alignment/bridge/real_bridge_v1.yaml
```

Replace `dry_run` with `launch` only after reviewing the merged configuration. Launches use
complete eight-GPU Holmes nodes in `ai2/molmofication`; the recipe rejects individual host
selection and cloud credentials.

## Artifact contract

Production profiles pin the artifacts that determine model and data semantics:

- the bare s002 checkpoint, config, data-path manifest, checkpoint marker, and DCP metadata;
- the Dolma2 tokenizer revision and fingerprint;
- the Molmo2 vision configuration and pristine SigLIP2 revision;
- the filtered PixMoCap root and serialized-source audit;
- the perception provenance manifest or joint visual projection;
- the joint native-text train/holdout manifests and their shared verification receipt;
- the exact parent checkpoint config and permanent checkpoint marker.

Runtime validation checks artifact content identities, parent phase/config/data/trainable
contracts, permanent checkpoint status, live dataset fingerprints, deterministic serialized
probes, and native replay cross-binding. Historical builder and promotion implementations are
not imported by the training process; their identities remain opaque metadata inside the pinned
artifacts. This keeps training packaging independent from one-off evidence producers while
preserving the artifacts' semantic hashes.

The mixture targets effective supervised-loss mass rather than raw example probability.
`VisionAlignmentMixtureConfig` converts each target using the audited mean loss weight.
Training reports delivered examples, tokens, supervised tokens, summed loss weight, realized
loss-mass share, and target error for every source.

## Resume and transition invariants

The recipe fails before training when any of these invariants drift:

- a phase's freeze patterns, image-row mask, optimizer groups, or schedulers;
- its source set, mixture targets, serialization, sequence length, or crop budget;
- the parent phase, checkpoint config SHA, data/trainable contracts, or permanent marker;
- a joint replay parent identity, manifest fingerprint, receipt, or train/holdout disjointness;
- an exact resume's saved data or trainable contract.

Same-phase checkpoints take precedence over a configured parent load path. Cross-phase loads
always set `load_optim_state=False` and `load_trainer_state=False`.

## Validation

Run the focused recipe and data tests with:

```bash
pytest -v \
  src/test/scripts/vision_alignment_test.py \
  src/test/data/multimodal/vision_alignment_mixture_test.py \
  src/test/data/multimodal/vision_alignment_perception_test.py \
  src/test/data/multimodal/vision_alignment_perception_provenance_test.py \
  src/test/data/multimodal/vision_alignment_joint_sources_test.py \
  src/test/data/multimodal/vision_alignment_joint_provenance_test.py \
  src/test/data/multimodal/native_text_replay_test.py
```

Post-hoc checkpoint evaluation and promotion evidence are separate workflows. They are not
runtime dependencies of this training recipe.
