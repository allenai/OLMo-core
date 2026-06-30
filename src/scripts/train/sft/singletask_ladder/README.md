# Single-task length-ladder SFT (Qwen3-4B)

Fine-tune Qwen3-4B **per task** (not a mixed mixture) on a length ladder, comparing
**attention variants**, to see how each handles long context on 5 tasks:
`contra` · `nq` (retrieval) · `oolong` · `outlier` · `rerank`.

Each run fine-tunes **from that variant's own CPT base**, on a **50% (~10k of 20k)** subsample of
one task's continuous-length ladder, with **no CPT mix**. Eval is native (no HF/vLLM), 8-GPU, multi-rung.

## Variants (the 3-variant launcher)

| variant | attention | base checkpoint (`amandab/.../step2385`) | packing |
|---|---|---|---|
| `dense` | flash-attn 2 + YaRN (factor 2) | `q4b-base-dense-lr1.1e-4` | ConcatAndChunk (varlen EOS masking) |
| `landmark` | fast landmark (`mem_freq=63`) | `q4b-base-fast-landmark-lr1p1e-4` | landmark windowing |
| `compressive` | fast compressive landmark (`nonselected_mass=0.1`) | `qwen4b-base-compressive-lr1.1e-4` | landmark windowing |
| `docchunk_dense`¹ | document-chunked dense (`<box>` markers) | `q4b-base-dense-lr1.1e-4` | per-document, first-fit window packing |

¹ separate launcher (`Qwen3-4B-docchunk-singletask-ladder-10k-SFT.py`); uses `single_task_docchunk_v2/<task>_dense` data.

## Shared training config (in the 3-variant launcher)

- `SEQUENCE_LENGTH = 40960` (concat/chunk window; YaRN supports it)
- `EPOCHS = 1`
- `GLOBAL_BATCH_SIZE = 8 × SEQUENCE_LENGTH` → **batch = 8 windows via gradient accumulation**
  (`CP=8` ⇒ DP=1, so `rank_microbatch_size = SEQUENCE_LENGTH` and the trainer runs 8 accum microbatches/step — no extra memory)
- `LR = 2e-5`, `LinearWithWarmup` (3% warmup), `SkipStepAdamW`, HSDP bf16, `CP_DEGREE = 8` (Ulysses)
- `SUBSAMPLE_FACTOR = 0.5` (whole-document seeded sampling)
- ⇒ **~641 steps, ~3h/run** at ~2.1 s/step

## Data (weka)

`/weka/oe-training-default/ai2-llm/checkpoints/prasanns/single_task_ladders_v2/<task>/`
(`token_ids_part_*.npy` + `labels_mask_*.npy` + `metadata.json`) — **all no-CoT; `rerank` is CE-graded**.
⚠️ Data uploaded to `s3://ai2-llm/...` does **not** auto-populate the weka filesystem — stage it with a
gantry `aws s3 sync s3→/weka` job (creds from the `PRASANNS_AWS_*` beaker secrets). See the
`weka-s3-checkpoint-transfer` note.

## Scripts

**Current — launch + eval**
- `Qwen3-4B-singletask-ladder-32k-10k-3variant-SFT.py` — dense/compressive/landmark (Beaker; variant inferred from run name)
- `Qwen3-4B-docchunk-singletask-ladder-10k-SFT.py` — docchunk_dense (Beaker)
- `launch_singletask_10k_overnight.sh` — submits the variant×task matrix (`CLUSTER` defaults to `ai2/jupiter`)
- `launch_beaker_multirung_eval.sh` → `run_q4b_beaker_multirung_eval.py` → `run_beaker_multirung_eval.sh` — native 8-GPU multi-rung eval on Beaker (reads everything from the weka eval bundle)
- `upload_lc_eval_bundle.sh` — push eval code+data to the weka bundle

**Local (Berkeley H200) — validation path**
- `Qwen3-4B-singletask-ladder-SFT-local.py` — torchrun launcher (reads local `/data` or `/scratch`)
- `run_q4b_stl_{traineval,validation,eval,multirung_eval}.sbatch` — node-portable train→eval sbatches

**Legacy / superseded** (kept for reference)
- `launch_singletask_10k_3variant.sh` — older matrix driver (use `…_overnight.sh`)
- `Qwen3-4B-compressive-singletask-ladder-32k-SFT.py` — compressive-only predecessor the 3-variant launcher generalizes
- `run_q4b_singletask_ladder.sbatch` — earlier local sbatch

## Run

```bash
# from the repo root
bash src/scripts/train/sft/singletask_ladder/launch_singletask_10k_overnight.sh        # submit matrix (jupiter)
DRY=1 bash src/scripts/train/sft/singletask_ladder/launch_singletask_10k_overnight.sh   # dry-run
# evals fire per finished checkpoint:
bash src/scripts/train/sft/singletask_ladder/launch_beaker_multirung_eval.sh
```
Note: `launch` uses `follow=True` (blocks streaming logs). To submit many runs without blocking, background each
`python <launcher> launch <run> <cluster>` with a `timeout`; the Beaker job keeps running after the follower dies.
