# First real run through the migrated `sft.py` / `cpt.py`

2026-08-11, horton, 1x H200. Task #21 of the migration. **Both entry points train end to end.**

The point of this was mechanics, not model quality: does a run launched through the migrated
entry points load shards, build the masked model, step the optimizer, and write a checkpoint that
carries its format fingerprint. It found three defects that the 1067-test CPU suite structurally
could not catch.

## What was run

`mathmatch` — chosen deliberately. It is pure-synthetic (no corpus, no network, reproducible from a
seed), and its train and eval halves come from the *same* generator with the *same* knobs, so the
train/eval in-domain question cannot confound a mechanics test. Two of the five main SFT tasks were
found OOD on 2026-08-11 (contradiction's eval, outlier's train), so any real task would have needed
that argued first.

```bash
# 1. data (login node; synthetic, CPU only)
ctc-data build --task mathmatch --out debug/train_smoke/data --rungs 2k --split train --train 512

# 2. shards
python src/scripts/ctc/convert_to_shards.py --task mathmatch --emit dense --chunk-by document \
    --marker-set qwen3_5 --tokenizer <Qwen3.5-4B-Base> --query-position after --seq-len 4096 \
    --input-jsonl debug/train_smoke/data/mathmatch/train.jsonl \
    --out-dir debug/train_smoke/shards/mathmatch
# -> 512 instances, p50 1342 tok, 0 dropped, fingerprint written

# 3. train  (see smoke_sft.sbatch / smoke_cpt.sbatch)
```

## Result

| run | budget | loss | exit |
|---|---|---|---|
| SFT, qwen3_0_6B from scratch, `--arch chunked` | 20 steps | 12.75 -> 7.825 -> 7.220 | 0 |
| CPT, same model and shards | 200k tokens (98 steps) | 12.72 -> 8.204 -> 6.462 -> ~4.5-5.2 | 0 |

`ctc_format_fingerprint.json` is written into **every** checkpoint (`step0`, `step10`, `step20`),
beside `model_and_optim/`. That is the loop closing: the converter writes the fingerprint next to
the shards, the callback collects it into the checkpoint, and `ctc-eval` can then refuse to grade a
checkpoint against a format it was not trained on. Previously only unit-tested.

## Defects found, and why the suite missed them

### 1. `run.py` handed the loader a config instead of a built source  🔴

`data_loader.build(dataset)` where `build()` takes `*sources: InstanceSource`. Died as
`TypeError: object of type 'PadToLengthInstanceSourceConfig' has no len()` from inside
`ComposableDataLoader.__init__`, *after* the model was on the GPU.

**Every local training launch would have hit this**, on any task and any architecture. It survived
because `test_every_architecture_and_mode_assembles` asserts the returned config's *type* and never
builds it. Fixed to `data_loader.build(dataset.build(work_dir))`, and
`test_the_instance_source_actually_builds_from_shards` now builds a source from tiny synthetic
shards on CPU. Note that test pins the *contract* -- a source config must build into something with
a length -- not `run.py`'s call site, so it would catch a config that cannot build but not a repeat
of this exact miswiring.

### 2. `init_distributed()` was NCCL-only  🔴

Async checkpointing needs a CPU-capable process group; with NCCL alone the run dies in the
checkpointer's `pre_train` with *"a CPU-capable backend is required for async checkpointing"*.
olmo-core's own `prepare_training_environment` defaults to `backend="cpu:gloo,cuda:nccl"` for
exactly this reason. `run.py` now passes the same string.

### 3. `ephemeral_save_interval >= save_interval` failed late  🟡

olmo-core rejects it, but only once the trainer config is assembled -- after a GPU model build.
Lowering `--save-interval` for a short run while leaving the ephemeral default at 500 is the natural
way to hit it. Now checked in `options.__post_init__`, for free, before anything is scheduled.

## Trap worth remembering (launcher-side, not code)

The first attempt printed its startup banner and then sat silent for seven minutes. Cause: the
sbatch set `PY=` to the node-local interpreter and then called **bare `torchrun`**, which resolves
off `PATH` to `/usr/local/linux/miniforge-3.13` -- on NFS. The worker parked in `D` state on
`rpc_wait_bit_killabl` at ~2% CPU, which reads exactly like a slow model load.

Always `$PY -m torch.distributed.run`, and echo the interpreter first so the log proves which one
ran. Diagnose with `srun --jobid=<id> --overlap ps -u <user> -o pid,stat,wchan:20,pcpu,etime,args`.

## Still owed

- Multi-rank. Both runs were world_size 1 because only 1 of horton's 8 GPUs was free and the
  node-local env exists only there. FSDP wrapping and the `reduce_metrics` path are unexercised.
- A run from a **real base checkpoint** rather than `--from-scratch`. That exercises the marker
  embeddings and the base-loading path, and needs a marker-repaired Qwen3 base staged node-local.
- Closing the eval loop: `ctc-eval` against one of these checkpoints, to see the fingerprint guard
  accept a matching format and reject a mismatched one on real files.
