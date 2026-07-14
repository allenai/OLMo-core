# Jacob MoE Checkpoint Conversion Plan

Status: publication scope confirmed and all 13 restartable family jobs launched.
Durable per-family receipts under `converted-checkpoints/_family_status/` are the
source of truth for live completion state.

## Proposed selection rule

Use one canonical final checkpoint per `(family, model size, data multiple)` cell: the finished run with the best observed final-250M-token training loss under the existing canonical-batch plotting policy. This selects the winning LR rather than every endpoint from an LR sweep.

For midtraining, select the canonical completed endpoint recorded in `RUN_TRACKER.md`. Do not convert intermediate, stopped, smoke, eval-only, or diagnostic checkpoints by default.

## Optimizer and trainer state

Conversions are model-only. The existing converter reads only the legacy FP32 model-master tensors and writes only the target model tensor schema. It does not copy optimizer moments, optimizer steps, scheduler state, dataloader state, or trainer state. Each conversion manifest records `optimizer_state_included: false` and `trainer_state_included: false`.

The output subdirectory is still named `model_and_optim` because that is the distributed-checkpoint convention used by the loader; its name does not mean optimizer state is present.

## Pretraining inventory

Exact source paths and selected LRs are in `JACOBM_DDP_PRETRAIN_CANDIDATES.tsv`.

| Family | Ready | Missing |
| --- | ---: | ---: |
| Baseline 48E/top4 | 16 | 0 |
| Expert granularity: 24E/top2 | 16 | 0 |
| Expert granularity: 96E/top8 | 16 | 0 |
| Total sparsity: 96E/top4 | 12 | 4 |
| Total sparsity: 192E/top4 | 12 | 4 |
| No shared expert, active matched | 16 | 0 |
| Dense0 + shared | 16 | 0 |
| Dense2 + shared | 16 | 0 |
| Dense4 + shared | 16 | 0 |
| Qwen-like active-matched 4.5d | 16 | 0 |
| Qwen-like true-3d + depth | 16 | 0 |
| Integration wide 256E/top8 | 16 | 0 |
| Integration deep 256E/top8 | 16 | 0 |
| **Total** | **200** | **8** |

The only absent pretraining cells are 1.2B Cx1/2/4/8 for each of the two total-sparsity families. These eight runs were intentionally skipped after the smaller-scale sparsity results were sufficiently clear; they are not unfinished migration work.

## Midtraining inventory

Exact source paths are in `JACOBM_DDP_MIDTRAIN_CANDIDATES.tsv`.

| Family | Completed selected endpoints | Other state |
| --- | ---: | --- |
| Baseline | 10 | Larger-model Cx2/Cx4: six stopped runs, excluded |
| Integration wide | 7 | Larger models have only Cx8; 275M has Cx1/2/4/8 |
| Integration deep | 7 | Larger models have only Cx8; 1.2B Cx8 finished at step 127157 |
| **Total** | **24** | **0 running** |

Of the 24 completed endpoints, `mt-275m-baseline-cx8-lr2e-4-r1/step95368` already has a conversion that passed the original exact-logits protocol. It must still pass the newer exhaustive strict-tensor protocol. The publication set therefore contains **224 checkpoints**: 200 pretraining and 24 midtraining.

Midtraining has not been run for the expert-granularity, total-sparsity, shared-expert, dense-schedule, or Qwen-like grids. Those are not conversion candidates unless new midtraining runs are completed later.

## Explicit default exclusions

- Non-winning pretraining LR sweep endpoints.
- The six non-selected 275M baseline midtraining LR endpoints at Cx1/Cx8.
- Stopped larger-model baseline midtraining checkpoints.
- Intermediate and ephemeral saves such as `step95000`, `step63500`, and `step125000`.
- Smoke, sanity, eval-only, and HF-export directories.
- The 275M integration-wide top-16 diagnostic and its midtraining checkpoint.

## Family-job conversion contract

Launch one restartable batch job per family: **13 jobs total**. Each family job
processes its pretraining checkpoints followed by any midtraining checkpoints,
one model at a time. The baseline test job therefore contains 26 models (16
pretraining and 10 midtraining). Each model has its own permanent local output
directory and stage reports. Converted checkpoints are not automatically
deleted after upload.

Each family job requests one GPU on `ai2/holmes`, uses urgent priority in the
`ai2/OLMo-3-moe-experiments` workspace, and sizes host RAM for the family's
largest checkpoint. Conversion and strict tensor verification are serial, so
expert parallelism is not used.

On restart, the driver checks the recorded result for every required stage of a
model in order. It skips a stage only when its report says it succeeded and the
report's source/config/output identities still match the manifest. This permits
an upload-only retry without repeating conversion or verification, while
avoiding unsafe skips based only on the presence of a file or directory.

A model is accepted only after:

1. Conversion completes atomically and records source/config metadata.
2. An independent verifier reloads both checkpoints and requires exact keys, shapes, dtypes, element counts, and bitwise-equal tensor values after reversing the layout mapping.
3. The converted checkpoint loads through the OLMoDDP training/eval path.
4. Legacy and converted full-vocabulary logits and captured intermediates match bitwise with zero tolerance.
5. The manifest confirms optimizer and trainer state are absent.
6. GCS object count, sizes, and CRC32C values match locally, after which `_SUCCESS.json` is written last.

The local outputs remain under:

`/weka/oe-adapt-default/jacobm/olmoe3/olmo-ddp-migration/converted-checkpoints/(pretraining|midtraining)/<source_run>/step<step>/`

The GCS destination for every checkpoint is:

`gs://ai2-llm/checkpoints/jacobm/olmoe3/olmo-ddp-converted/v1/(pretraining|midtraining)/<family>/<model_size>/cx<data_multiple>/`

The source step is deliberately absent from the directory name. Each published directory includes a generated `README.md` that records the source run, source step, architecture and training settings, conversion provenance, and verification hashes.


## Family launch record

The baseline family uses one GPU for 26 models. The other 12 families contain
198 models and use one one-GPU job each. All jobs were submitted on Holmes at
urgent priority. A failed family can be requeued independently; its durable
stage reports ensure already converted, verified, and uploaded cells are not
repeated.

The generated launch plan and family specs live under
`src/scripts/beaker/generated/olmo_ddp_conversion_families/`. Regenerate them
from the publication manifest with:

```bash
python src/scripts/prepare_olmo_ddp_family_jobs.py
```

The launch driver is read-only by default. This command validates the prepared
files and prints Beaker commands without submitting them:

```bash
python src/scripts/launch_olmo_ddp_family_jobs.py
```

Any future retry requires explicit submission, an exact aggregate GPU-count
confirmation, an independent retry receipt, and `--allow-resubmit`:

```bash
python src/scripts/launch_olmo_ddp_family_jobs.py \
  --family <family> \
  --receipt <retry-receipt.json> \
  --allow-resubmit \
  --submit \
  --confirm-gpu-count 1
```

After every successful submission the driver atomically records the experiment
ID in the selected receipt. The original launch receipt is
`converted-checkpoints/_family_status/_launches/remaining_families_v1.json`;
retries use separate receipts in the same directory.

## Loading published checkpoints from GCS

OLMo-core's distributed checkpointer accepts `gs://` checkpoint paths directly.
It can load a checkpoint under a different distributed topology and either
stream each rank's required ranges from GCS or pre-download them into its local
work directory. The published checkpoint root is passed as the load path; the
loader recognizes the `model_and_optim/.metadata` file beneath it.

These conversions are model-only initialization checkpoints, not exact training
resumes. A downstream `TrainerConfig` should therefore use the GCS root with
`load_trainer_state=False` and `load_optim_state=False`, for example:

```python
TrainerConfig(
    save_folder=save_folder,
    load_path=(
        "gs://ai2-llm/checkpoints/jacobm/olmoe3/olmo-ddp-converted/v1/"
        "midtraining/baseline/275m/cx8/"
    ),
    load_strategy=LoadStrategy.always,
    load_trainer_state=False,
    load_optim_state=False,
    checkpointer=CheckpointerConfig(
        pre_download=False,
        load_thread_count=8,
    ),
)
```

`pre_download=False` reads the needed distributed-checkpoint ranges directly;
set it to `True` when repeated local access is worth the temporary cache space.
The runtime must have GCS read credentials (or anonymous access if the objects
are public).
