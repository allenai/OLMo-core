# Hybridish SFT → CTC-suite eval

Finetune a `mainline_ladder` hybrid checkpoint on a prebuilt CTC SFT set, then score it on the real
CTC suite (held-out, 500 examples per rung) via olmo-eval. This is the path that produced the
4:1 vs 7:1 comparison.

Everything below has actually been run. Where a step exists only to work around a trap, the trap is
named — skipping those steps does not fail loudly, it produces numbers that look fine and are wrong.

Everything needed to run this lives on **this branch** (`prasann/ctc-sft-hybridish`): the SFT
script here, and the export/eval tooling in `debug/hybridish_sft/`. Training must run from this
branch — `prasann/landmark`'s `olmo_core` has no `scalable_softmax`, so training there loads the
checkpoint and silently drops a trained component.

The CTC suite tasks are a separate repo (`allenai/olmo-eval` @ `prasann/ctc-suite-grader-fixes`)
which the eval launcher clones for you. Eval data is the **public** HF dataset
`PrasannSinghal/ctc-suite-eval` — no token, no weka.

Rebuilding the SFT data from scratch is out of scope here; the dataset below is prebuilt. If you do
need to regenerate it, the builders are on `prasann/landmark` under `src/scripts/data/hybridish/`
(`build_ctc_sft_mix.py` → `convert_ctc_to_sft_completion.py`).

## The dataset

One path. This is the exact SFT set both published checkpoints were trained on — **15,183
instances**, 8 CTC tasks, 2k–32k, dolma2 tokenizer, one epoch.

```bash
DATA=/weka/oe-training-default/ai2-llm/checkpoints/prasanns/ctc_hybridish_sft/shards_long32k
```

It contains `token_ids_part_*.npy`, `labels_mask_*.npy`, `metadata.json`, and `src_index.json`
(instance → source-row map; **any grader needs it** — see the warning under "In-house grader").

Other shard directories exist alongside it on weka from earlier iterations. They are not this
dataset and are not what anything here was trained or scored on — use the path above.

The two finetuned checkpoints sit next to it and are ready to evaluate as-is (SSMax-patched
modeling code and `auto_map` live inside each checkpoint, so `trust_remote_code=True` is the whole
integration):

```bash
CKPTS=/weka/oe-training-default/ai2-llm/checkpoints/prasanns/ctc_hybridish_sft   # sft_4to1_ml_hf, sft_7to1_ml_hf
```

## Quickstart

Three scripts, each one Beaker job that pins the branch it needs — no checkout to get right.

**Score a checkpoint that is already prepared:**

```bash
H=src/scripts/train/hybrid-small-suite
bash $H/launch_eval.sh sft_7to1_ml_hf smoke smoke      # ~3 min, proves the path end to end
bash $H/launch_eval.sh sft_7to1_ml_hf 7to1 full        # all 39 task x rung runs
python debug/hybridish_sft/harvest_sweep.py            # the comparison table
```

**Finetune your own base checkpoint and score it** — the whole loop:

```bash
H=src/scripts/train/hybrid-small-suite
W=/weka/oe-training-default/ai2-llm

bash $H/launch_sft.sh     my-run --model $W/path/to/your/checkpoint/step1234/
bash $H/prepare_ckpt.sh   $W/checkpoints/<you>/my-run/step1244  my_run_hf
bash $H/launch_eval.sh    my_run_hf  myrun  full
python debug/hybridish_sft/harvest_sweep.py
```

`--model` is an olmo-core checkpoint step directory on weka. **That is the only thing you specify.**
Depth, width, head count, where the full-attention layers sit and whether they use Scalable-Softmax
all come from the checkpoint's own `config.json`, so there is no geometry to get wrong. Add
`--dataset` to train on something other than the set below; `--preset 1.4b_4to1|1.4b_7to1`
substitutes a published checkpoint.

---

## 1. SFT

```bash
W=/weka/oe-training-default/ai2-llm
bash src/scripts/train/hybrid-small-suite/launch_sft.sh my-run --preset 1.4b_7to1 --dataset $DATA
bash src/scripts/train/hybrid-small-suite/launch_sft.sh my-run --model $W/path/to/your/checkpoint/step1234/
```

| flag | what |
|---|---|
| `--model` | weka path to the base checkpoint (olmo-core step dir, the one holding `model_and_optim/`) |
| `--preset` | shorthand for a published checkpoint (`1.4b_4to1`, `1.4b_7to1`) |
| `--dataset` | weka path to the shard dir (defaults to `$DATA`) |
| `--lr`, `--epochs` | defaults 4e-5, 1 |

**No architecture flags.** `build_ctc_model_config` reads the checkpoint's `config.json` and
rebuilds its `TransformerConfig` exactly — including `block_overrides`, which is where the
full-attention layers and their Scalable-Softmax setting live. The only thing overridden is the
attention backend, which is a property of the cluster, not the checkpoint. The resolved geometry is
printed before launch.

`launch_sft.sh` refuses to run from the wrong branch, and refuses if HEAD is unpushed — gantry ships
the remote commit, so both would otherwise be silently wrong. Weka mounts automatically on clusters
tagged `storage:weka` (e.g. jupiter).

Everything else — optimizer, schedule, FSDP, activation checkpointing, packing — comes from the
repo's standard SFT machinery. Packing uses block-diagonal masking, so packed training matches
example-level.

## 2. Prepare the checkpoint for eval

```bash
bash src/scripts/train/hybrid-small-suite/prepare_ckpt.sh <olmo-core step dir> <output name>
```

One Beaker job with weka mounted, doing all three prep stages in place so nothing is copied between
machines:

1. **export** olmo-core → HF (`examples/huggingface/convert_checkpoint_to_hf.py`)
2. **re-dialect** `olmo3_5_hybrid` → `mainline_ladder` — same weights, different key spellings and
   `model_type`; the plugin and every released hybrid checkpoint use the latter
3. **make self-contained** — copy the SSMax-patched modeling code into the checkpoint and set
   `auto_map`, so `trust_remote_code=True` is the entire integration for any consumer

Writes to `$W/checkpoints/prasanns/ctc_hybridish_sft/<output name>`; override with `OUT=`. The
re-dialect needs a released `mainline_ladder` checkpoint as a naming/config donor — a 1.4B one is
the default, override with `--ref`.

> ⚠ Stage 3 is not optional. The stock `mainline_ladder` plugin has **no Scalable-Softmax at all**:
> a checkpoint's `ssmax_scale` loads as UNEXPECTED and is silently ignored, so the model loads
> clean and scores while missing a trained component. The patched plugin also adds the
> `transformers>=5.13` cache shim the stock one lacks, without which any cached forward dies with
> `AttributeError`, and it takes decode positions from `cache_position` — a forward-hook
> reconstruction reads the current forward's length and scales the query **13x too small** at an 8k
> prompt, on exactly the tokens being graded.

Verify independently, with the plugin deliberately *not* importable:

```bash
python debug/hybridish_sft/test_plugin_ssmax.py                      # formula + cached-decode equivalence
python debug/hybridish_sft/verify_trust_remote_code.py --ckpt <ckpt>
```

## 3. Evaluate

```bash
bash src/scripts/train/hybrid-small-suite/launch_eval.sh sft_7to1_ml_hf 7to1_nq ctc_nq
```

Third argument: `smoke` (8 instances, proves the path), `full` (all 39 task × rung runs), or a task
name to shard one job per task — the 8-task ladder is hours on one GPU and sharding makes it
concurrent.

`CHUNK=8` for long-context tasks. `absence` OOMs at the default batch of 64: a single 20 GiB
allocation plus ~29 GiB lost to allocator fragmentation.

## 4. Harvest

```bash
python debug/hybridish_sft/harvest_sweep.py
```
Reads `sweep_ids.txt`, fetches **only** `metrics.json` from each result dataset (the full datasets
are ~130 MB of predictions apiece; sixteen of them exhausted our disk quota, after which the
harvester's own write left a 0-byte file while reporting success), and prints the arm comparison
per task × rung. Running, failed and missing jobs are reported separately and a partial table says
so — a table quietly covering 13 of 16 arms reads as complete.

---

## Bringing in artifacts from elsewhere

All four steps are Beaker jobs reading and writing weka, so nothing is copied between them.

You only need the staging scripts in `debug/hybridish_sft/` if you produce an artifact somewhere
Beaker cannot read — a checkpoint trained on another cluster, or shards built locally:

```bash
bash debug/hybridish_sft/stage_ckpts_to_s3.sh    # your machine -> S3
bash debug/hybridish_sft/sync_s3_to_weka.sh      # S3 -> weka, via gantry
```

S3 alone is not enough: a Beaker job reads weka, so skipping the second command surfaces as a
MISSING path at step 0. Same for shards (`src/scripts/data/hybridish/stage_shards_to_weka.sh` on
`prasann/landmark`).

## Reading the output

* **Check `parse_rate` before any score.** A low score at low parse rate is a decoding/format
  failure, not a capability one. Sub-1B hybridish checkpoints frequently cannot emit an `[id]` at
  all.
* **Quote `eval_size` next to any sub-500 cell.** The suite is 500/rung by design; `absence` tops
  out at r16k with fewer.
* **These arms are not parameter-matched.** 4:1 (1.4B) and 7:1 (2.1B) have *identical* attention:
  4 full-attention layers, 199,251,008 attention params each. The whole difference is 12 extra GDN
  layers (+62% non-embedding params). A 7:1 win is therefore not attributable to attention capacity,
  and calling it an attention-ratio result is wrong. A clean ratio ablation holds non-embedding
  params fixed and varies where the attention layers sit.

## In-house grader (`grade_ctc_mix.py`)

Also in this directory, for quick iteration against the *training* shards. It is **not** a
substitute for the suite: it grades train data, uses a continuous length mix rather than rungs, and
its SSMax is a reconstruction. Use it to check a pipeline runs; use olmo-eval for numbers. It
requires `src_index.json` and runs a gold self-check (score each shard's own gold against its mapped
example — must be 1.0, needs no generation) before grading.
