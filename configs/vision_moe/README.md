# Vision-MoE launch profiles

Every Beaker submission for this project, including manual CPU, GPU, training, and evaluation
jobs, must target workspace `ai2/molmofication`. Direct YAML/spec submissions must use
`python src/scripts/beaker_submit_vision_moe.py SPEC.yaml`; direct use of
`beaker experiment create` is prohibited for this tree. The supported Python training launchers
independently fail closed unless their final workspace is exactly `ai2/molmofication`.

## Vision alignment (new continued-pretraining recipe)

The new [`vision_alignment/`](vision_alignment/README.md) tree is independent from the
historical Molmo2 Stage 1/2 recipes below. It starts from the bare s002 pretrained LM, uses
native document serialization, separates connector/vision/joint unfreezing into model-only
phase forks, and replaces Tulu retention data with fingerprinted native `OLMo-mix-0925`
replay when the LM is unfrozen. Its mixture is calibrated by supervised-loss mass and its
profiles select the Holmes cluster without pinning individual nodes.

Only a synthetic bridge smoke profile is currently launchable. Real bridge, perception, and
joint profiles remain intentionally absent until their exact data audits and manifests are
pinned.

## Stage 1

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
- `stage1_ep8_2node_real_resume_to32000_micro8.yaml` resumes that selected two-node arm from
  its latest complete checkpoint, currently step 4,000, through step 32,000 in the original save
  folder. It restores
  model, optimizer, scheduler, trainer, and packed-loader state and resumes W&B run `sdgbbjmz`.
  PR806's corrected point/count formatting applies from step 201, so the state transition is
  exact but the data serialization after resume is intentionally not byte-identical to steps 1-200.
  Held-out caption and fast vision evaluation run every 2,000 steps. Four language-retention
  sentinels run every 4,000 steps at one instance per rank and stop after 30 batches per task,
  preserving complete 10-choice Basic Skills documents;
  these deterministic partial sentinels are health trends, not full OLMES benchmark scores.
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

Inspect the selected full-state continuation with its exact original run name:

```bash
PATH=/weka/oe-training-default/rustin/envs/olmo-core-vision/bin:$PATH \
python src/scripts/train/Molmo2-Stage1.py dry_run \
  s002-stage1-rerisk-200-micro8-2node-20260809-2db297f \
  --beaker-test-config=configs/vision_moe/stage1_ep8_2node_real_resume_to32000_micro8.yaml
```

## Stage 2 pilot

The initial Stage 2 pilot is split into two profiles for one logical run:

- `stage2_ep8_2node_image_only_v9_to50.yaml` starts from the canonical completed Stage 1
  `step32000` checkpoint and trains through step 50.
- `stage2_ep8_2node_image_only_v9_resume_to200.yaml` resumes the same run and trains through
  step 200. Use exactly the same run name so the trainer finds the latest checkpoint in the
  existing save folder before considering the Stage 1 fallback path.
- `stage2_ep8_2node_image_only_v9_resume_to400.yaml` is the bounded continuation of the
  audited run `s002-stage2-v9-pilot-bounded-errors-5a81c40c`. It resumes that run's complete
  step-200 model, optimizer, scheduler, trainer, packed-loader, RNG, and W&B state and stops
  at step 400 for another performance gate. Its required-run-name guard rejects any other
  positional run name, save folder, or W&B identity before launch.

All profiles use two 8-GPU Holmes nodes, EP8, urgent priority, an eight-hour minimum runtime,
and the approved workspace and budget. They run the complete 43-source `image-only-v9`
mixture with OLMo 3 chat serialization, 16k sequences, MoE capacity factor 2, sigma factor 12,
and diagnostics every 10 steps. The 50-step and 200-step profiles evaluate held-out vision
tasks every 50 steps. The bounded step-400 continuation evaluates them every 200 steps. The
scheduler keeps the full 30,000-step horizon in every phase. Isolated malformed rows are
skipped deterministically, reported through `data/errors total`, and bounded to at most 10
consecutive or 1,000 cumulative errors per rank so a broken source still stops training. No
inline language evaluator is enabled.

The checkpointer saves permanent milestones at steps 50 and 200, keeps both, and maintains one
rolling ephemeral checkpoint every 25 steps for preemption recovery. It does not write a
pre-train checkpoint. The first command assumes the run's save folder is empty. Repeating it
with an existing checkpoint intentionally resumes that run because save-folder checkpoints
take precedence over `trainer.load_path`.

The step-400 continuation adds a permanent step-400 milestone, retains all three permanent
checkpoints, and writes one rolling recovery checkpoint at step 300. Although its Stage 1
fallback flags remain weights-only for the original transition, a checkpoint in the guarded
same-run save folder always takes precedence and is restored with full trainer and optimizer
state by the trainer's resume path. Its `all_non_permanent` removal strategy preserves the
fixed steps while allowing the new step-100 cadence to retire the old ephemeral step 175 after
step 300 is safely written.

Inspect the fresh profile without submitting:

```bash
PATH=/weka/oe-training-default/rustin/envs/olmo-core-vision/bin:$PATH \
python src/scripts/train/Molmo2-Stage2.py dry_run s002-stage2-image-only-v9-pilot \
  --beaker-test-config=configs/vision_moe/stage2_ep8_2node_image_only_v9_to50.yaml
```

After step 50 is complete, inspect the continuation with the same run name:

```bash
PATH=/weka/oe-training-default/rustin/envs/olmo-core-vision/bin:$PATH \
python src/scripts/train/Molmo2-Stage2.py dry_run s002-stage2-image-only-v9-pilot \
  --beaker-test-config=configs/vision_moe/stage2_ep8_2node_image_only_v9_resume_to200.yaml
```

Inspect the guarded step-200 to step-400 continuation with its exact run name:

```bash
PATH=/weka/oe-training-default/rustin/envs/olmo-core-vision/bin:$PATH \
python src/scripts/train/Molmo2-Stage2.py dry_run \
  s002-stage2-v9-pilot-bounded-errors-5a81c40c \
  --beaker-test-config=configs/vision_moe/stage2_ep8_2node_image_only_v9_resume_to400.yaml
```

Only replace `dry_run` with `launch` after reviewing the merged configuration and receiving
explicit submission approval.

The standalone post-step-200 health and performance jobs are documented in
[`eval/README.md`](eval/README.md). They remain unlaunchable until their `GIT_REF` placeholder
is replaced with the committed `vision-moe` evaluator revision.
