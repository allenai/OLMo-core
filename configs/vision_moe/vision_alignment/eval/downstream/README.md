# Dense SSMax downstream fast suite

This is the checkpoint-native, image-multiple-choice comparison for the two 1.4B Cx8 SSMax
Vision-Alignment arms. One paired run evaluates one exact matched phase/checkpoint step. Repeat it
at assembled bridge step 0, bridge step 500, perception steps 500/1000/2000/3000/4000, and joint
steps 4000/8000/12000/16000
to obtain the trajectory needed for magnitude-routing and downstream-signal correlation. Stop at
the joint boundary until the mid-training interface is selected:

- `ssmax_blink_jigsaw`: all 150 BLINK Jigsaw validation examples.
- `ssmax_mathvista_geometry_mc`: the 203 MathVista `testmini` examples satisfying
  `metadata.task == "geometry problem solving"` and `question_type == "multi_choice"`.

The runner scores the next token only over the valid answer letters in each example. The local
MathVista scorer accepts one valid letter, maps it to the selected choice, and implements the
pinned scorer's string-equality semantics locally. The task adapter does not import MathVista's
evaluator module, whose class initialization constructs an LLM server, and never calls its
answer-extraction/chat path. The metric is therefore `mathvista_geometry_mc_acc`, not
`llm_as_judge_eval`, and no OpenAI or other judge API key is needed. The suite does not run video
or open-ended tasks. These two tasks are an efficient paired ranking signal, not a substitute for
the eventual complete Molmo image suite.

## Exact software and data contract

- OLMo-core checkpoint format: native DCP; no HF conversion is performed. The checkpoint pin is
  the bridge `checkpoint_identity`: SHA-256 over every DCP state file and every trainer-rank state
  file, plus `config.json`, `.metadata.json`, DCP metadata, step, file counts, and absolute path.
  The evaluator recomputes that full identity before a strict model-state load and records the
  complete inventory receipt in its result.
- lmms-eval: `cb45ac4d4a667ea5ef89c7a148bff69b3489b981`.
- datasets / pyarrow: `3.6.0` / `19.0.1`.
- BLINK dataset: `BLINK-Benchmark/BLINK@a3666eb249237ba3d5eca8db21176cc47967e040`.
- MathVista dataset: `AI4Math/MathVista@2b6ad69445fbb5695c9b165475e8decdbeb97747`.
- Both pinned datasets are public and the checked-in task YAML loads them without `token: true`.
  The launch validator intentionally rejects `HF_TOKEN`, `HUGGING_FACE_HUB_TOKEN`, and judge API
  variables. Its exact secret surface is only
  `{name: BEAKER_TOKEN, secret: RUSTINS_BEAKER_TOKEN}`, which Gantry itself requires.
- Tokenizer: the checkpoint must retain the repository-pinned Dolma2 tokenizer revision and
  fingerprint used by Vision-Alignment.
- Prompt/crops: native Vision-Alignment `document` layout and eight high-resolution crops shared
  across all images in a request. Pinned Jigsaw rows expose contiguous `image_1` through
  `image_4`, with trailing `image_4` null, so their three actual images receive `[3, 3, 2]` crops;
  a single-image MathVista row receives all eight. The adapter sorts numeric suffixes and rejects
  missing slots or a non-null image after a null slot.

The runner uses one B300 process per model. A 1.4B dense model fits without sharding, and a
single-process load avoids collective-order hazards from variable numbers of images. The paired
Beaker spec launches the two arms concurrently.

## Materialize and validate one paired trajectory point

The checked-in
`ssmax_joint_fast_pair.yaml.template` is an intentionally non-launchable joint example. Copy it
once per paired trajectory point. For bridge or perception, change both literal `joint` phase
values to the same desired phase. Replace every placeholder with:

1. the same clean, pushed 40-hex OLMo-core commit for both tasks;
2. the same declared global step for both arms and each exact checkpoint path ending in that
   `stepN`;
3. each full `identity_sha256`, plus its `config_sha256`, `marker_sha256`, and
   `dcp_metadata_sha256`; and
4. two distinct result paths below
   `/weka/oe-training-default/rustin/experiments/vision-ssmax-molmofication/vision-alignment/evals`.

Compute the full checkpoint identity (this reads and hashes every state/trainer file):

```bash
python -c 'import json,sys; from pathlib import Path; from olmo_core.eval.vision_alignment_ssmax_bridge import checkpoint_identity; print(json.dumps(checkpoint_identity(Path(sys.argv[1]), workers=8), indent=2, sort_keys=True))' \
  CHECKPOINT
```

Validate the untouched placeholder template now:

```bash
python src/scripts/beaker_submit_vision_ssmax_eval.py \
  configs/vision_moe/vision_alignment/eval/downstream/ssmax_joint_fast_pair.yaml.template \
  --validate-only --allow-placeholders
```

Validate a materialized spec (this does not submit):

```bash
python src/scripts/beaker_submit_vision_ssmax_eval.py MATERIALIZED_SPEC.yaml --validate-only
```

The submission wrapper has no workspace option and accepts only `ai2/scaling-ladders`. It also
requires exactly two tasks (one QK-norm and one no-QK-norm), the same declared phase/global step,
the same evaluator commit, full checkpoint identities, `ai2/oe-other`, Holmes-only placement,
urgent priority, at least eight hours of minimum runtime, one GPU per arm, and the exact evaluator
argument and secret surfaces. Once the materialized spec has been reviewed, the launch command is:

```bash
python src/scripts/beaker_submit_vision_ssmax_eval.py MATERIALIZED_SPEC.yaml \
  --name vision-ssmax-PHASE-stepN-downstream-fast-v1
```

When one matched checkpoint finishes first, its exact task may be staged without bypassing the
submission validator. Materialize a spec containing only that task and name the expected arm
explicitly:

```bash
python src/scripts/beaker_submit_vision_ssmax_eval.py MATERIALIZED_SINGLE_ARM_SPEC.yaml \
  --single-arm ssmax_head_qknorm \
  --name vision-ssmax-PHASE-stepN-head-downstream-fast-v1
```

Submit the other exact arm the same way when its checkpoint is complete. This changes only job
timing: the comparator still requires both complete, same-phase, same-step results and their full
checkpoint identities.

For a local one-example mechanics smoke on a B300, use the exact command from either task and add
`--limit 1`. Any limited output is marked partial and the comparator rejects it.

## Compare the two complete results

Pin the two result files by SHA-256, then create the paired receipt:

```bash
python src/scripts/eval/vision_alignment_ssmax_downstream_compare.py \
  --qknorm-result QKNORM.json \
  --expected-qknorm-result-sha256 QKNORM_SHA256 \
  --no-qknorm-result NO_QKNORM.json \
  --expected-no-qknorm-result-sha256 NO_QKNORM_SHA256 \
  --output COMPARISON.json
```

The receipt records the matched phase/global step and both full checkpoint identities. It reports
task-level accuracy, chance accuracy, prediction histograms, maximum answer share (a useful
answer-collapse fingerprint), matched outcome counts, exact two-sided McNemar p-values, and the
equal-task macro point difference. The field `observed_point_ranking` is descriptive only;
`inference.conclusion` remains `inconclusive` because no same-step cross-task winner rule was
preregistered. It refuses mismatched sample IDs, targets, choice counts, task definitions, crop
settings, partial coverage, or model variants.

## Compare adaptation trajectories against assembled step 0

Absolute same-step performance does not isolate adaptability from starting capability. For each
candidate phase/step, compare its exact pair with the exact assembled bridge-step-0 pair:

```bash
python src/scripts/eval/vision_alignment_ssmax_downstream_trajectory_compare.py \
  --baseline-qknorm-result STEP0_QKNORM.json \
  --expected-baseline-qknorm-sha256 STEP0_QKNORM_SHA256 \
  --baseline-no-qknorm-result STEP0_NO_QKNORM.json \
  --expected-baseline-no-qknorm-sha256 STEP0_NO_QKNORM_SHA256 \
  --candidate-qknorm-result CANDIDATE_QKNORM.json \
  --expected-candidate-qknorm-sha256 CANDIDATE_QKNORM_SHA256 \
  --candidate-no-qknorm-result CANDIDATE_NO_QKNORM.json \
  --expected-candidate-no-qknorm-sha256 CANDIDATE_NO_QKNORM_SHA256 \
  --output TRAJECTORY_COMPARISON.json
```

For each task and the equal-task macro, this reports
`(QK_t - QK_0) - (noQK_t - noQK_0)`. It requires identical source IDs/targets/choice counts across
all four hash-pinned results. Its 95% interval uses exactly 10,000 deterministic nonparametric
row-paired bootstrap draws: a sampled row keeps both arms and both time points together; BLINK and
MathVista are sampled independently and then weighted equally. The preregistered superiority rule
requires both task DIDs to have the same strict sign and the macro interval to exclude zero in that
direction. Practical equivalence requires each task's 95% interval to fit inside +/-3 accuracy
points and the macro interval inside +/-2 points. Otherwise the conclusion is `inconclusive`.

The primary pilot adaptability endpoint is the joint step-16000 DID against assembled bridge
step 0. Earlier checkpoints are trajectory/early-signal measurements, and final same-step macro
accuracy is the secondary practical-capability endpoint. A primary directional or equivalence
result is conditional on all phase hard invariants, native-text noninferiority, and answer-
distribution collapse checks passing. It ranks only these two exact 1.4B Cx8 artifacts under one
fixed adaptation seed; it is not a general QK-norm causal claim and cannot answer whether 810M or
less pretraining is sufficient.

## Known boundary before mid-training

These configs are exact for the current Vision-Alignment document interface through joint. The
eventual mid-training recipe may introduce a chat template or other response serialization. Do
not reuse this prompt contract for a post-mid-training checkpoint until that recipe is selected
and its native interface is added and tested. Likewise, broader Molmo image-suite evaluation
remains a later expansion; video and open-ended judge evaluations are intentionally outside this
fast gate.
