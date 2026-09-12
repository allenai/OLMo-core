# Vision alignment

The supported entrypoint is `src/scripts/train/Vision-Align.py`, backed by
`olmo_core.internal.vision_alignment` and the standard experiment CLI. See the
[API guide](../../../docs/source/guides/vision_alignment.md) for sources, replay, checkpoint
handoffs, configuration and launch requirements.

## Native recipe

| Phase | Default training configuration |
| --- | --- |
| Bridge | 8K context, pack64, eight local crops/image, global 128, MB4, 500 steps |
| Perception | 2,560 context, global 128, MB4, 4,000 steps; existing phase recipe |
| Joint | 8K context, global 128, MB1, 16,000 steps; existing phase recipe |

Bridge trains the connector and six input image-token rows with LB loss off, CF8 and shared
dispatch buffers. Its connector LR is `2e-4`, warmup 100, cosine horizon 250 and floor `2e-5`.
Validation sources retain a 2,560-token serialization cap with 8K padded execution. Perception
additionally trains the vision encoder; joint additionally trains the LM body while retaining
a frozen output projection. No router RMS repair is applied. Later phases inherit the parent's
LB policy unless explicitly restored with `recipe.restore_pretraining_router_lb=true` or overridden.

```bash
python src/scripts/train/Vision-Align.py dry_run align-bridge local \
  --recipe.pretraining_checkpoint=/path/to/pretraining/stepN
```

Bridge launch defaults: two eight-GPU nodes, EP8, urgent priority, minimum runtime 8h and
32 GiB shared memory, without a wall-clock training deadline. Later-phase launch settings remain
unchanged. The remote git commit must contain the requested implementation; `allow_dirty` does
not deploy local edits or untracked files. Direct experimental Beaker specs use
`src/scripts/beaker_submit_vision_moe.py` and target `ai2/molmofication`.

## Experiment utilities

These historical utilities support explicit comparisons and compatibility; they do not
define new production defaults. Paths below are relative to
`configs/vision_moe/vision_alignment/` in their matching experiment deployment. The utilities
remain local or in frozen runtime snapshots; they are not required by the native recipe or
included as part of its source distribution.

| Location | Purpose |
| --- | --- |
| `decoded_comparison/` | Frozen decoded panel and historical academic evaluation wrappers |
| `context_quality/` | Step/exposure/schedule comparisons and full-state perception continuations |
| `context_validation/` | Bounded context, packing, crop and shared-dispatch screens |
| `packing_probe/` | CPU crop/packing geometry and annotation-retention checks |
| `loader_speed/`, `loader_training/` | Loader measurements and controlled training-speed screens |
| `joint_validation/`, `joint_continuation/` | Joint objective, quota, resume and continuation checks |
| `perception_validation/`, `perception_continuation/` | Perception comparisons and checkpoint continuations |

The decoded panel contains 64 examples each for counting, grounding, four OCR/document sources
and captioning. It reports task scores, positive/empty grounding, repetition and termination;
it is not an official benchmark suite. With the matching frozen deployment in its configured
two-node allocation, set `EVAL_ROOT` and `PANEL_CHECKPOINT`, then run:

```bash
bash configs/vision_moe/vision_alignment/decoded_comparison/run.sh CHECKPOINT LABEL
```

The wrapper uses `EVAL_ROOT/panel.json` and writes `EVAL_ROOT/results/LABEL/`.

The separate historical academic panel uses 512 examples/task for VQAv2, TextVQA, DocVQA, ChartQA,
AI2D and A-OKVQA MC, each with correct/shuffled/blank images. Its reference checkpoint is the
original VA12k run's **perception step4000**, not the joint VA12k checkpoint. Frozen cached
baseline results can be reused when the full panel and scoring definitions match. Historical
training-overlap annotations do not establish disjointness from a new checkpoint's training data.

## Artifacts and compatibility

Results, manifests, checkpoints and immutable runtime snapshots live under
`/weka/oe-training-default/rustin/experiments/vision-moe/vision-alignment/`:

- `artifacts/alignment-decoded-comparison-20260907-v1/panel.json`: frozen decoded panel.
- `evals/joint-v1-external-academic-v1/`: historical academic panel and baseline receipts.
- `artifacts/bridge-code-archive-20260911-v1/source.tar.gz`: recovery archive for retired
  bridge-only launch profiles, diagnostic wrappers and their tests.
- `artifacts/historical-readmes-archive-20260911-v1/history.md`: original experiment
  documentation, including run links, measurements and superseded procedures;
  `readmes.tar.gz` in the same directory preserves the exact README files.

Keep archived runtime inputs referenced by queued/running experiments unchanged.
`Vision-Alignment.py` and the legacy audit adapters remain intact because historical consumers
import their config classes and validate their exact bytes. Saved experimental callback/config
classes under `context_quality/`, `context_validation/` and `decoded_comparison/` require the
matching source deployment; a clean native checkout does not include those utilities.
Retiring launch wrappers does not authorize deleting datasets,
checkpoints or evidence. Detailed experimental interpretation belongs in saved reports and the
[router audit](../../../docs/source/guides/vision_alignment_router_audit.md), not the native recipe.
