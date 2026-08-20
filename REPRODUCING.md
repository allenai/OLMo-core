# Reproducing the CTC experiments

Every command here is **standalone and node-local**: a machine with GPUs, this branch, and an
internet connection for the first data fetch. Nothing below needs AI2 infrastructure — the same
training entry points accept `--cluster ...` to submit to Beaker instead, but that is an option,
not a requirement.

```bash
git clone -b prasann/ctc_public https://github.com/allenai/OLMo-core && cd OLMo-core
pip install -e '.[all]'        # olmo-core (training side); torch per your CUDA
pip install ./ctc              # data generation + evaluation
```

The pipeline is four steps, and the **format fingerprint** threads through all of them — eval
refuses to grade a checkpoint against a format it was not trained on:

```
ctc-data build  →  convert_to_shards.py  →  train/sft.py  →  ctc-eval
```

## 1. Data — no GPU, no index, no API key

Every corpus-backed task has a published **seed pool** (the expensive half of generation:
cross-encoder scores, BM25 negatives, LLM-mined pairs — precomputed) on the Hub at
[`PrasannSinghal/ctc-seed-pools`](https://huggingface.co/datasets/PrasannSinghal/ctc-seed-pools);
`--pool auto` fetches it. The synthetic tasks need nothing at all.

```bash
ctc-data list                                                        # every task and its knobs
ctc-data build --task contradiction --pool auto --train 18000 --out data/
ctc-data build --task nq            --pool auto --train 20000 --out data/
ctc-data build --task textgroups                --train 20000 --out data/   # synthetic

# rungs are open-ended past the calibrated 2k-32k table
ctc-data build --task textgroups --split eval --rungs 64k,1m,10m \
    --eval-size 125 --allow-small-eval --out data/xlong
```

Builds run in seconds (eval ladders) to minutes (20k-example train sets) and refuse to write
anything that fails the built-in audit. Per-task supply bounds (contradiction caps at
`--train 18000`; nq reuses queries with fresh distractors past ~9k and its build report says so)
are documented in `ctc/src/ctc/data/README.md` and on the seed-pool dataset card.

**Suite coverage note:** 18 of the frozen 22-row suite have generators in this tree. The other
four rows (`msmarco` retrieval-graded, `niah`, `obliq`, `qdmatch_fiqa`) were built with
pre-migration pipelines and are served ready-made in the public eval dataset
[`PrasannSinghal/ctc-suite-eval`](https://huggingface.co/datasets/PrasannSinghal/ctc-suite-eval).

## 2. Shards

```bash
PYTHONPATH=src:ctc/src python src/scripts/ctc/convert_to_shards.py \
    --input data/contradiction/train.jsonl --out shards/contradiction \
    --layout chunked --query-position after
```

`--layout` and `--query-position` are recorded in the shard's fingerprint; the reference runs use
`--query-position after`. Full flag reference: `src/scripts/ctc/README.md`.

## 3. Training

One recipe (`src/scripts/ctc/train/`), the experiment axes as flags. `run/train.sh` resolves the
cluster environment (node-local interpreter, caches, `torchrun` spelling) and execs the same
Python the tests exercise.

### 3a. The 22-task suite protocol (Qwen3.5, one model per task × arm)

The headline dense-vs-chunked grid trains **one model per (task, arm)** on that task's own train
set. Reference hyperparameters, recovered from the launch records: **lr 5e-5, 3 epochs, seq-len
40960, global batch 8 instances** — with 20k train examples and one instance per rank per step on
8 GPUs, 3 epochs is `--max-steps 7500`.

```bash
for ARCH in full chunked-mix; do
  CTC_NPROC=8 run/train.sh ctc-contradiction-$ARCH \
      --data shards/contradiction:1 --base BASE --arch $ARCH \
      --model qwen3_5_4B --tokenizer qwen3_5 --lr 5e-5 --max-steps 7500
done
```

Three arms matter, and the naming is load-bearing:

- `--arch full` — plain causal over the identical marker-bearing token stream.
- `--arch chunked-mix` — the chunked mask **plus the mask-mixing curriculum** (each example
  collapses to plain causal with probability p, annealed 0.80 → 0.0 over the run). **This is the
  arm published results tables call "chunked."**
- `--arch chunked` — the pure mask, p = 0 always. A different, stricter arm; do not relabel one
  as the other.

Bases: Qwen3.5 checkpoints need **no marker-embedding repair** (their box-marker ids sit inside
the trained vocabulary — audited at 0.8B/2B/4B/9B). Plain Qwen3 bases DO need
`src/scripts/ctc/fix_marker_embeddings.py` first; the trainer's `--base` error text says so.

### 3b. Model-scale sweep (0.8B / 2B / 4B × full / chunked-mix, 4 tasks)

Same protocol, **1 epoch** (kept comparable with the 4B reference fan-out), over contradiction,
reorder, hotpotqa and qdmatch_nq — complexity classes O(N²)/O(N²)/O(N)/O(N²), hotpotqa as the
low-tracking anchor. Factories: `qwen3_5_0_8B`, `qwen3_5_2B`, `qwen3_5_4B`. Per-task seq-len from
the shard's own max example: contradiction/reorder 40960, hotpotqa 26112, qdmatch_nq 33792.

```bash
CTC_NPROC=8 run/train.sh ctcms-contradiction-cmix-2b \
    --data shards/contradiction:1 --base BASE_2B --arch chunked-mix \
    --model qwen3_5_2B --tokenizer qwen3_5 --lr 5e-5 --max-steps 2500   # 1 epoch of 20k @ 8/step
```

### 3c. The 5-task mixed-SFT family (Qwen3-4B, packed 32k)

The older multi-task comparison: contradiction/nq/oolong/rerank/outlier mixed at weights
`2 : 1 : 1 : 1.5 : 1.5`, lr 1e-5, ~700M content tokens (1100 steps at 4 nodes for the
document-chunked arms; 1465 at 2 for dense-packed), YaRN factor 2 past 32k (applied
automatically), `--arch landmark --mem-freq 63` for the landmark arm.

```bash
CTC_NPROC=8 run/train.sh q4b-5task-chunked \
    --data shards/contradiction:2 --data shards/nq:1 --data shards/oolong:1 \
    --data shards/rerank:1.5 --data shards/outlier:1.5 \
    --base BASE_Q3_FIXMARK --arch chunked --model qwen3_4B --max-steps 1100
```

## 4. Evaluation

```bash
ctc-eval --ckpt runs/ctc-contradiction-full/step7500 --tasks contradiction \
    --bundle data/ --backend vllm
ctc-eval --ckpt runs/ctc-contradiction-chunked-mix/step7500 --tasks contradiction \
    --bundle data/ --attn chunked --backend native
```

`--bundle data/` grades the ladders you built in step 1. Two rules that keep numbers comparable:

- **Score both arms with the same backend.** The reference grid was vLLM-scored, and
  native-vs-vLLM drift has been measured at ~0.08 f1 on contradiction@2k — larger than the
  eval's standard error.
- Quote every number with its `eval_size` and standard error, and never across bundles.

vLLM specifics (including the serving-copy requirement for olmo-exported Qwen3.5 checkpoints)
are in `ctc/src/ctc/eval/README.md`.
