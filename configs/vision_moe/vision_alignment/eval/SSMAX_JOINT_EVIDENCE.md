# Descriptive SSMax joint evidence

This protocol starts only after one exact treatment checkpoint for each SSMax lineage has an
explicitly approved perception gate. Historical paired perception-v1 and perception-v2 evidence
produce v5 and v6 gates. The direct, no-control perception protocol produces a v7 gate bound to its
single lineage, authorizing amendment, training Git ref, and evidence Git ref. Validation dispatches
by the exact gate version and reopens that version's referenced report and manifest; no version can
authorize another protocol's candidate. A separately authorized v8 gate may admit a rejected v7
report only to exploratory SSMax joint alignment when every deviation is a permitted short-prefix,
source-level visual diagnostic; v7 remains rejected and unchanged. This protocol stops at the end
of joint alignment; it
neither selects nor starts a mid-training recipe. The historical s002 joint receipts remain
historical evidence and are not accepted by these model-variant-aware tools.

The two checked-in `.json.template` files under `eval/joint/` are deliberately non-runnable. Their
manifest-spec schema and the finalized manifest/receipt schema are version 2.
For the current direct program, fill one copy per lineage only after its strict v7, exploratory v8,
or exploratory-waiver v9 gate and reviewed joint profile exist, together with the projection,
source audit, pairings, output root, and clean evidence commit. Do not replace a missing artifact
with a placeholder path.
Historical v5/v6 gates remain accepted only through their original paired validators. The v8
admission rules and issuance commands are documented in
`SSMAX_PERCEPTION_EXPLORATORY_JOINT.md`.

The version-2 spec pins the raw `config.json` SHA-256 independently at every retained step. This is
intentional: both completed lineages crossed reviewed exact-resume launches, so their raw configs
do not all have identical bytes. Finalization validates every saved config against its step pin.
It also performs a narrowly named *structural* comparison by removing only `launch.name` and
`launch.git.ref` and comparing canonical-JSON SHA-256 values. This structural comparison cannot
alias JSON booleans, integers, and floats, and it does not claim that Git refs are semantically
interchangeable. The manifest preserves every raw config SHA, launch name, branch, and training Git
ref in `training_resume_lineage`.

The observed source schedule is asymmetric: head-QK uses `7cc97a77` at steps 0/4,000,
`e53e8ee6` at 8,000, and `26eebf08` at 12,000/16,000; no-QK uses `7cc97a77` through step 8,000 and
`26eebf08` at 12,000/16,000. The finalized manifest therefore classifies the cross-arm schedule as
`asymmetric_code_transition`, its causal interpretation as `confounded`, and its decision scope as
`descriptive_only`. The comparison must not be presented as a clean causal estimate of QK norm.

The separately pinned `evidence_git` is the clean commit used to build the manifest and launch the
evaluator. Checkout attribution is derived from the imported manifest-builder module, never from a
caller-supplied recipe path. The recipe and profile must resolve inside that exact repository, and
the manifest pins the builder's repository-relative path, evidence Git ref, and raw source SHA.
Recipe/profile references retain their build-time raw artifact pins plus repository-relative paths;
an evaluator resolves the latter under its own imported-module checkout and proves the same blobs,
so the build worktree path is never treated as the Beaker clone. The manifest also pins the
finalized input spec by absolute path, raw SHA, and canonical semantic SHA, then rebinds all
spec-derived finalized fields to that live-pinned spec during validation.

Each finalized manifest binds all bytes of permanent steps 0, 4,000, 8,000, 12,000, and 16,000,
the clean recipe/profile/git identity, the approved perception candidate, the fixed eight-source
matched/wrong population, a 992-row manifest-order native holdout prefix, and the fixed 32-row
joint SSMax attention probe.

The native prefix uses 992 of the existing 1,000 held-out replay windows because 992 is the largest
prefix divisible by the fixed 16-rank HSDP world. No row is resampled or substituted. Visual
evaluation uses 496 exact-geometry matched rows per source: the live projection has respectively
511, 510, 512, 512, 508, 510, 509, and 508 eligible exact-geometry rows in the source order above,
so 496 is the largest common population divisible by 16. No row is padded or substituted.
GatedDeltaNet inputs remain unpacked (`pack_sequences=False`), contain no example/subsegment IDs,
and have exactly one response branch. Multi-annotation sources first pass through the shared
deterministic SSMax single-response projection at training data seed 95818; evaluation fixes its
selection epoch to zero and the projection recalibrates loss weights to the exact single-branch
convention. The independent matched-donor selection seed is 6198; the two seeds are never
interchanged or conflated.

The resulting eight live pairings are under
`/weka/oe-training-default/rustin/experiments/vision-ssmax-molmofication/vision-alignment/artifacts/ssmax-joint-pairings-v1/`.
Their manifest raw SHA-256 is
`d47cab3193ddadb20dec31ae8713186dfc78dda57786c943d9c7f6631f8f1cc0` and semantic
SHA-256 is `b461222e87791aeec58ec5ef48ec1524005ff36c54f3e11ee67b0088574af88d`.

The original bridge attention probe is not interchangeable with the joint population: a live
preflight found bridge indices outside the 512-row joint selection. The dedicated joint probe was
derived with `build_ssmax_joint_attention_probe_manifest.py` from the pinned projection. Its raw
SHA-256 is `f34e7bdc944d71826aff3ee2c963da9675487a9f2d40d5b7199a1fe0124a687c`; all 32 live
sample/content IDs, valid lengths, input IDs, token types, and loss masks were re-materialized with
zero mismatches.

Finalize a manifest from the completed run:

```bash
PYTHONPATH=src python src/scripts/eval/vision_alignment_ssmax_joint_manifest.py \
  --spec /absolute/path/to/filled-ARM-joint-manifest-v1.json \
  --output /weka/oe-training-default/rustin/experiments/vision-ssmax-molmofication/vision-alignment/evidence/joint-v1/ARM/manifest.json
```

Record `sha256sum` of the manifest. For each retained step, first render and then submit the fixed
2x8 Holmes job. The launcher fixes workspace `ai2/scaling-ladders`, urgent priority, an eight-hour
minimum runtime, the Weka mount, and a clean immutable git checkout.

```bash
PYTHONPATH=src python src/scripts/beaker_launch_vision_ssmax_evidence.py dry_run joint \
  ssmax-ARM-joint-stepSTEP -- \
  --manifest /weka/oe-training-default/rustin/experiments/vision-ssmax-molmofication/vision-alignment/evidence/joint-v1/ARM/manifest.json \
  --expected-manifest-sha256 MANIFEST_SHA256 \
  --step STEP \
  --work-dir /weka/oe-training-default/rustin/experiments/vision-ssmax-molmofication/vision-alignment/evidence/joint-v1/ARM/stepSTEP/work \
  --output /weka/oe-training-default/rustin/experiments/vision-ssmax-molmofication/vision-alignment/evidence/joint-v1/ARM/stepSTEP/evaluation.json

PYTHONPATH=src python src/scripts/beaker_launch_vision_ssmax_evidence.py launch joint \
  ssmax-ARM-joint-stepSTEP -- \
  --manifest /weka/oe-training-default/rustin/experiments/vision-ssmax-molmofication/vision-alignment/evidence/joint-v1/ARM/manifest.json \
  --expected-manifest-sha256 MANIFEST_SHA256 \
  --step STEP \
  --work-dir /weka/oe-training-default/rustin/experiments/vision-ssmax-molmofication/vision-alignment/evidence/joint-v1/ARM/stepSTEP/work \
  --output /weka/oe-training-default/rustin/experiments/vision-ssmax-molmofication/vision-alignment/evidence/joint-v1/ARM/stepSTEP/evaluation.json
```

One GPU receipt contains the strict generic DCP load, full checkpoint identity, exact hashes of
the frozen lexical input rows and output projection relative to step 0, per-example correct/wrong
CE for first-1/8/32/all response windows across all eight sources, per-row native-text summed CE
and token counts plus aggregate CE/PPL, and the manifest-bound attention report. At every retained
step, native-text retention is a deterministic 10,000-sample row-paired bootstrap against step 0;
the upper 95% bound on relative CE increase must be at most 2%. The attention report captures
attention-logit magnitude, normalized
entropy/effective context, and probability mass plus argmax routing to image, prompt, and response
keys. It uses the same collector as bridge; no second attention implementation is introduced.

For every retained step, run the CPU cursor replay. It reads and validates the resume-safe,
hash-chained `ssmax_health_ledger` callback state from all 16 raw trainer states, requires identical
reduced event chains, and derives skip/non-finite/data-error counts without accepting W&B history
or hand-authored summaries. It also reports delivered loss mass for all nine training sources.

```bash
PYTHONPATH=src python src/scripts/eval/vision_alignment_ssmax_joint_health.py \
  --manifest MANIFEST.json \
  --expected-manifest-sha256 MANIFEST_SHA256 \
  --step STEP \
  --work-dir /weka/oe-training-default/rustin/experiments/vision-ssmax-molmofication/vision-alignment/evidence/joint-v1/ARM/stepSTEP/health-work \
  --output /weka/oe-training-default/rustin/experiments/vision-ssmax-molmofication/vision-alignment/evidence/joint-v1/ARM/stepSTEP/health.json
```

Build and independently audit one report per arm by supplying both receipt types for every step:

```bash
PYTHONPATH=src python src/scripts/eval/vision_alignment_ssmax_joint_report.py \
  --manifest MANIFEST.json \
  --evaluation-receipt 0=step0/evaluation.json \
  --evaluation-receipt 4000=step4000/evaluation.json \
  --evaluation-receipt 8000=step8000/evaluation.json \
  --evaluation-receipt 12000=step12000/evaluation.json \
  --evaluation-receipt 16000=step16000/evaluation.json \
  --health-receipt 0=step0/health.json \
  --health-receipt 4000=step4000/health.json \
  --health-receipt 8000=step8000/health.json \
  --health-receipt 12000=step12000/health.json \
  --health-receipt 16000=step16000/health.json \
  --output ARM-joint-trajectory.json

PYTHONPATH=src python src/scripts/eval/vision_alignment_ssmax_joint_report.py \
  --audit ARM-joint-trajectory.json
```

Compare the two audited reports:

```bash
PYTHONPATH=src python src/scripts/eval/vision_alignment_ssmax_joint_compare.py \
  --left ssmax_head_qknorm-joint-trajectory.json \
  --right ssmax_no_qknorm-joint-trajectory.json \
  --output ssmax-joint-paired-description.json
```

The report hard-fails on a non-strict/incomplete load, frozen lexical-surface mutation,
checkpoint/data/cursor drift, a failed raw receipt, any optimizer-guard skip, non-finite/data-error
counter, or failure of the native-text noninferiority bound. It reports
visual gap, correct CE, retention, native-text regression, loss-mass delivery, and attention
trajectories without converting thresholds into promotion or a winner. The paired comparison
directly compares the two embedded raw attention reports at every retained step, including step 0.
For every visual source and first-1/8/32/all window it also reports a paired 95% normal interval
for the same-step arm difference and the step-0-normalized difference-in-differences (DID) in gap
and correct CE. Native CE/PPL receive both direct and DID summaries. A visual adaptation signature
is labeled directional only when the strict interval direction is identical at all four
post-baseline checkpoints; all other signatures are explicitly inconclusive, and even a
consistent signature remains descriptive rather than a promotion decision.
BLINK Jigsaw and MathVista geometry are run through the separately pinned downstream fast-pair
template; this protocol does not duplicate that evaluator.
