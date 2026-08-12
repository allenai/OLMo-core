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

---

# Multi-rank, from a real marker-repaired base

2026-08-12. Task #25. Prepared on horton, **run on mooney** (see the QOS section). This is the first
two items of "Still owed" above: **world_size 2**, so FSDP actually wraps and `reduce_metrics`
actually reduces, and **`--base`** instead of `--from-scratch`, so the base-loading path and the
marker embedding rows are exercised.

Everything below the "RESULT" divider is the finished run; everything above it is the preparation.

## Prerequisite: the base was POISONED, and the audit says so in two numbers

The candidate base is `/data/prasann/dcmix_base/model_and_optim` (horton). Reading the distcp
shard sizes identifies it without loading it: two 311 MB embedding shards is
`151936 x 1024 x fp32`, i.e. a **Qwen3-0.6B at Qwen3's 151936-row vocabulary**. That pins the run
to `--model qwen3_0_6B --tokenizer qwen3`.

`src/scripts/ctc/fix_marker_embeddings.py` audits and repairs it. (It was written for this run as
`debug/train_smoke/check_and_fix_markers.py` and promoted on 2026-08-12 — see "Migration gap
closed" below. The commands in this section are quoted with the new path.) It is the pre-migration
`src/scripts/data/fix_marker_embeddings.py` with two changes: a `--check-only` mode, so a base can
be audited before anyone decides to repair it, and reserved ids looked up by tokenizer family via
`reserved_ids()` rather than the module-level Qwen3 constants, since Qwen3 and Qwen3.5 do not share
a vocabulary.

Reporting **both** numbers is the point. Cosine alone would have called a pre-2026-07-14 repair
healthy; norm alone would have called the raw base healthy (0.481 is low but not absurd). It is the
pair that identifies the state.

| | cos(doc_start, doc_end) | marker norm | trained-row median | ratio | verdict |
|---|---|---|---|---|---|
| BEFORE | **+1.0000** | 0.5712 | 1.1874 | **0.481** | POISONED — bit-identical markers |
| AFTER | **+0.5061** | 1.2606 | 1.1874 | **1.062** | OK |

All six marker pairs were `cos = +1.0000` before (`doc_start`, `doc_end`, `landmark`, `pad` are one
vector), and all four rows sat at norm 0.5712. So this base had **never** been repaired — it is not
a pre-2026-07-14 repair, it is the raw Qwen3 state. After the donor-row repair (`«`, `»`, `§`, `¶`)
every pair is well separated and every norm is within 6% of the trained-row median. `+0.5061` for
`doc_start`/`doc_end` is expected and not a near-miss: `«` and `»` are genuinely related delimiters
and sit at 0.6013 in the base model.

The repaired copy was then re-audited **after** being written and read back, which is what proves
the repair survives the distcp round trip rather than only existing in the repairing process's RAM.

## Commands

```bash
# 1. audit + repair + rebuild shards   (CPU only -- holds no GPU while horton is contended)
sbatch debug/train_smoke/smoke_prep_base.sbatch        # job 3438512, COMPLETED

# inside it:
python src/scripts/ctc/fix_marker_embeddings.py --base /data/prasann/dcmix_base/model_and_optim \
    --check-only --model qwen3_0_6B --tokenizer qwen3
python src/scripts/ctc/fix_marker_embeddings.py --base /data/prasann/dcmix_base/model_and_optim \
    --out /data/prasann/ctc_smoke/base_fixmark --model qwen3_0_6B --tokenizer qwen3 \
    --hf-tokenizer $TOK
python src/scripts/ctc/convert_to_shards.py --task mathmatch --emit dense --chunk-by document \
    --marker-set qwen3 --tokenizer $TOK --query-position after --seq-len 4096 \
    --input-jsonl debug/train_smoke/data/mathmatch/train.jsonl \
    --out-dir /data/prasann/ctc_smoke/shards/mathmatch_qwen3
# -> 512 kept, 0 dropped, p50 1339 tok, doc_start_id 151648, marker_set qwen3

# 2. stage base + shards onto mooney's own /data, CPU only    # job 3438650, COMPLETED
sbatch debug/train_smoke/stage_to_mooney.sbatch

# 3. the run itself                                          # job 3438664, SUCCEEDED
sbatch debug/train_smoke/smoke_sft_multirank_base_mooney.sbatch
```

`smoke_sft_multirank_base.sbatch` is the horton/`berkeleynlp` twin. It is kept, and its `--base` and
`find` bugs are fixed, but it has never been run to completion — horton's QOS is why (below).

**The shards had to be rebuilt, and that is not incidental.** The 2026-08-11 smoke shards are
`--marker-set qwen3_5`, tokenized at vocab 248320 with `doc_start_id 248049`. Fed to a
151936-row Qwen3-0.6B they are not merely wrong, they make success criterion 4 unreadable: a base
that loaded correctly would still show a high step-0 loss on ids that mean nothing to it, which is
indistinguishable from a base that silently did not load. Same `mathmatch` task JSONL (task data is
tokenizer-independent), re-tokenized with `--marker-set qwen3`.

## Preflight: the whole run assembled on CPU, for free, while the GPU job queued

`debug/train_smoke/preflight_configs.py` builds the exact configuration and, crucially, **builds the
instance source** — the step whose absence let defect #1 above reach a GPU before failing. horton was
contended enough that a 2-GPU slot was worth minutes of waiting, so it is worth not spending one on a
`TypeError`. Job 3438592, `PREFLIGHT OK`:

```
model vocab_size      : 151936
document_chunk_attn   : {'doc_start_id': 151648, 'doc_end_id': 151649, 'eos_id': 151643, 'mode': 'chunked'}
dp shard_degree       : 2
trainer load_path     : /data/prasann/ctc_smoke/base_fixmark
trainer load_strategy : always
instances             : 512
```

`load_strategy: always` is the one to read twice. It means the run **fails loudly** if the base
cannot be found, rather than warning and training from random init — so the "silently reinitialized"
failure mode cannot happen quietly here. It can still happen *loudly-but-ignorably* the other way,
via the save folder: a save folder that already contains a checkpoint is loaded first and logs
`Ignoring load path (...) since checkpoint was found in save folder`. The launcher `rm -rf`s the save
folder for exactly that reason.

## Why it did not run on horton

**A QOS wall on horton.** Job 3438562 was the highest-priority pending job in `berkeleynlp` and still
sat in `(Resources)` with an estimated start ~10 h out. All eight of horton's cards were held by
`preemptive` and `preemptive_high_*` jobs, and on `berkeleynlp` we hold only `normal` and
`preemptive` — and `preemptive` preempts `normal` and nothing else. So the request could not preempt
anything actually on the node. Worse than slow: single-GPU jobs backfilled into each card as it
freed, so a 2-GPU request never saw two free at once. This is worth remembering as a rule — **on
`berkeleynlp`, a multi-GPU `preemptive` request starves rather than queues** when the node is held by
peers at the same or higher QOS.

The fix is `--partition=jsteinhardt --qos=preemptive_high` on cubbins or mooney, the two nodes other
than horton that carry the node-local conda env. `smoke_sft_multirank_base_mooney.sbatch` is that
launcher, written and ready: it stages the repaired base and the qwen3 shards from `/net/horton` onto
mooney's own `/data` (one NFS hop at submit time, never during training), **re-audits the markers on
the staged copy** rather than assuming the repair survived the copy, and is otherwise byte-for-byte
the same experiment.

**A permission wall.** The previous agent's `sbatch` was denied by the harness permission system.
Escalating QOS to preempt other users' jobs is exactly what that gate is for, and an instruction from
another agent is not consent for it. Prasann approved the runs directly, which is what unblocked the
section below.

## Migration gap closed

**`src/scripts/data/fix_marker_embeddings.py` was not ported into this tree.** It gates every
document-chunked and landmark run from a fresh base — skipping it produces chance-level results that
read as modeling findings — and CLAUDE.md still points at its pre-migration path.

Promoted on 2026-08-12 to **`src/scripts/ctc/fix_marker_embeddings.py`**, with the tensor logic split
out of the CLI (`marker_cosines`, `marker_norm_ratios`, `problems`, `repair_markers`) so it can be
tested without a checkpoint. `debug/train_smoke/check_and_fix_markers.py` is deleted; every launcher
here points at the new path, and so does the note in `src/olmo_core/data/document_chunk_landmark.py`.

Two behaviour changes:

- **`--check-only` now exits non-zero on a bad base**, so it works as a gate in a launcher under
  `set -e` rather than only as something a human reads. `smoke_prep_base.sbatch` audits a base it
  *expects* to be poisoned, so that one call carries `|| true`.
- The post-repair assertions became a returned list of problems rather than bare `assert`s, which is
  what lets the same rule serve the audit and the repair. Same thresholds: `|cos| < 0.9`, norm ratio
  in `0.5x .. 2x`.

`ctc/tests/train/test_fix_marker_embeddings.py` (12 tests) pins it on a synthetic 48x16 matrix.
**The load-bearing one is `test_cosine_fixed_but_norm_still_small_is_STILL_rejected`**: markers that
are mutually distinguishable but sit at 1/3.6 of a trained row's norm — the exact state the first
version of the script produced and called success — must fail. A test that only checked cosines
would have blessed it.

Because a unit test cannot exercise the distcp load or a real 151936-row embedding matrix, the CLI
was also run against both real checkpoints (`validate_fixmark_cli.sbatch`, job 3438673):

```
=== A. the RAW base -- expect POISONED and a NON-ZERO exit ===
VERDICT BEFORE: cos(doc_start, doc_end)=+1.0000  norm_ratio=0.481  -> markers doc_start/doc_end are
  indistinguishable (cos=+1.0000); ... marker pad norm is 0.481x the trained-row median -- out of
  distribution. This is the exact failure the donor-row init exists to prevent.
EXIT_RAW=1   (expected: non-zero)
=== B. the REPAIRED base -- expect OK and exit 0 ===
VERDICT BEFORE: cos(doc_start, doc_end)=+0.5061  norm_ratio=1.062  -> OK
EXIT_FIXED=0   (expected: 0)
```

## Traps met on the way

- **`TRANSFORMERS_CACHE` is set in the environment on horton and overrides `HF_HOME`.** With
  `HF_HUB_OFFLINE=1`, `AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base")` then raises
  `LocalEntryNotFoundError` — *"couldn't find them in the cached files"* — even though every file is
  present under `HF_HOME`. transformers warns about the deprecation and resolves it anyway. Pass the
  absolute snapshot directory instead of the repo id and the lookup cannot go wrong:
  `/data/prasann/hf-cache/hub/models--Qwen--Qwen3-0.6B-Base/snapshots/da87bfb6...`.
- **Identify a base's architecture from its distcp shard sizes before loading it.** Two 311 MB
  embedding shards is `151936 x 1024 x fp32` = Qwen3-0.6B at the Qwen3 vocab; the 4B base at
  `stable_bases/` shows 777 MB = `151936 x 2560 x fp32`. That is a two-second check that pins
  `--model` and `--tokenizer`, and getting it wrong is not a crash — it is a plausible wrong number.

---

# RESULT: the 2-GPU run from the real base (2026-08-12, mooney)

**All four success criteria met.** Job **3438664**, `jsteinhardt` / `preemptive_high` / mooney,
2 x H200, ~45 s of training. Two earlier submissions failed first (below), and both failures were
real defects rather than infrastructure noise.

| # | criterion | evidence |
|---|---|---|
| 1 | exits 0, loss decreasing | `=== SFT EXIT=0 ===`; CE `3.208 -> 1.124 -> 0.8379 -> 0.7710` over 30 steps |
| 2 | world_size 2, FSDP wraps | `Built 1D device mesh with shape (dp=2,)` and `Applied FSDP to the model with 1D device mesh with shape (dp=2,)`, both logged twice (once per rank) |
| 3 | fingerprint in checkpoints | `ckpt_sft_mr/step20/ctc_format_fingerprint.json` and `.../step30/ctc_format_fingerprint.json` |
| 4 | base genuinely loaded | `Loading checkpoint from '/data/prasann/ctc_smoke/base_fixmark/model_and_optim'...` → `Checkpoint successfully loaded`; step-1 CE **3.208** vs **12.75** from random init on 2026-08-11 |

### 2 — this is the point of the run

`reduce_metrics` is a real collective now, not a no-op. It shows up by name in the training loop:

```
olmo_core.distributed.parallel:246  INFO  Built 1D device mesh with shape (dp=2,)
olmo_core.train.train_module.transformer.common:137  INFO  Applied FSDP to the model with 1D device mesh with shape (dp=2,)
olmo_core.train.trainer:1361  INFO  Waiting for bookkeeping ops to finish: 'reduce_metrics' (1 ops)...
```

The banner confirms the run agreed: `budget 30 steps at 2 GPU(s), 4,096 tok/step` — 2048 seq_len x 2
ranks, i.e. both ranks are carrying an instance, not one rank running twice.

### 4 — "loaded" is established two ways, and neither alone would do

**Positively, from the loss.** Step-1 CE is **3.208**. The 2026-08-11 `--from-scratch` run on the
same task and the same architecture opened at **12.75**. 3.2 is a pretrained model meeting an unseen
task format; 12.7 is `ln(vocab)`-ish, a model that knows nothing. A silent reinit cannot produce 3.2.

This is why the shards had to be rebuilt for the qwen3 tokenizer first (see above). Against a
151936-row model, qwen3_5-tokenized ids are meaningless, and a correctly-loaded base would *also*
have shown a high step-0 loss — the criterion would have been unreadable, not merely noisy.

**Structurally, from the config.** `load_strategy: always` makes a missing base a `FileNotFoundError`,
not a warning — so "silently reinitialized" is not a failure mode this configuration has. That is not
hypothetical: it is exactly how submission #2 below died. The one way it *could* go wrong quietly is
the save folder winning over `--base` (`Ignoring load path ... since checkpoint was found in save
folder`); the launcher `rm -rf`s the save folder, and no such line appears in the log.

The audit ran once more on mooney's own copy, immediately before training:

```
=== marker audit on the STAGED copy ===
VERDICT BEFORE: cos(doc_start, doc_end)=+0.5061  norm_ratio=1.062  -> OK
```

### Loss curve, in full

Logged every 10 steps, so four points: `3.208` (step 1), `1.124` (10), `0.8379` (20), `0.7710` (30).

Worth naming so nobody re-derives it later: **0.771 is near the CE ≈ 0.79 that is the signature of
the marker-norm bug, and this is not that.** That signature is a *flatline* — 0.79 from step 0,
identical under every mask, including plain causal. This is a curve that starts at 3.2 and falls
monotonically on 512 mathmatch examples with a repaired base. Same number, different shape.

## Three defects found by running it

### 4. `--base` pointed at the parent of `model_and_optim` 🔴  (job 3438657)

```
FileNotFoundError: No checkpoint found in save folder ('/data/prasann/ctc_smoke/ckpt_sft_mr')
  or load path ('/data/prasann/ctc_smoke/base_fixmark')
```

`Checkpointer.dir_is_checkpoint` accepts a bare `.metadata`, or all of `train/rank0.pt` +
`model_and_optim/.metadata` + `.metadata.json`; `contains_checkpoint` additionally accepts a
directory of `stepN/` checkpoints. A directory that merely *contains* `model_and_optim/` — which is
precisely what `save_model_and_optim_state` writes, and therefore what the marker-repair script
produces — is none of those.

`load_strategy=always` did its job: loud failure, no silent random init. But it fired from inside
`trainer.fit()`, **after both ranks had built the model and FSDP had wrapped it** — the log shows
`Applied FSDP ... (dp=2,)` and then the traceback. Roughly two minutes of a 2-GPU allocation to learn
about a missing path component.

Fixed in two places: the launchers now pass `--base .../base_fixmark/model_and_optim`, and
`run.py` gained `_check_base_is_a_checkpoint()`, called before anything is built. If the path is a
local directory that olmo-core would not load, it exits immediately — and if
`<base>/model_and_optim` *would* load, the message says so and prints the exact argument to use.
Remote (`s3://`, unmounted weka) paths are left alone. Pinned by
`ctc/tests/train/test_base_checkpoint_guard.py`.

### 5. `find "$SAVE" | head -40` turned a successful run into `FAILED` 🟡

Job 3438664 printed `=== SFT EXIT=0 ===` and a complete `step30/` checkpoint, and `sacct` reported
**FAILED**. `head` closes the pipe after 40 lines, `find` dies on SIGPIPE (141), and `set -o
pipefail` propagates that as the script's exit status.

This is the "sbatch reports FAILED on a run that fully succeeded" trap in `lambda_cluster.md` —
except self-inflicted, and reproducible rather than mysterious. It matters more than it looks: the
standing advice is to distrust the exit code and read the loss curve, which is correct, and a
launcher that manufactures false failures makes that advice load-bearing for no reason. Both
launchers now use `find "$SAVE" -maxdepth 2` — bound the walk, not the output.

### 6. `rsync -a src/ dst/` does not create intermediate directories 🟡  (job 3438645)

```
rsync: [Receiver] mkdir "/data/prasann/ctc_smoke/shards/mathmatch_qwen3" failed:
  No such file or directory (2)
```

rsync creates only the final component of the destination. `base_fixmark/` (one level) worked;
`shards/mathmatch_qwen3/` (two) did not, because `shards/` did not exist. Both the staging job and
the GPU launcher now `mkdir -p` first. Cheap, but inside the GPU launcher this would have burned a
2-GPU allocation on a failed `mkdir`.

## Staging, and why it is a separate job

`stage_to_mooney.sbatch` (CPU only, `qos=normal`) pulls the 2.0 GB base and the shards from
`/net/horton` onto mooney's `/data` **before** the GPU job is submitted. The GPU launcher still
stages — it must be self-contained — but that rsync is then a no-op instead of ~7 minutes of two
H200s waiting on a ~5 MB/s NFS link. It also means the marker audit on mooney's own copy can be
*read* before deciding to submit, rather than discovered mid-run.

Verified byte-for-byte after staging: identical `(size, path)` inventories on both nodes, and the
audit re-run on mooney's copy returns `cos +0.5061 / ratio 1.062 -> OK`. `du -sh` disagrees between
the two nodes (2.0 G vs 1.9 G) — that is block allocation, not missing data.

## What this run still does not cover

- **`ctc-eval` against one of these checkpoints.** The fingerprint is written; whether the eval-side
  guard accepts a matching format and rejects a mismatched one has only been unit-tested.
- **More than one node.** `dp=2` on one node exercises FSDP and `reduce_metrics`; it does not
  exercise multi-node rendezvous or HSDP.
- **CPT at world_size 2.** Only SFT was re-run here.
- **A real task.** `mathmatch` is synthetic and in-domain by construction — deliberately, since this
  is a mechanics test — so nothing here says anything about model quality.
