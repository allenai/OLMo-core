# Evaluate ANY checkpoint on the 5 long-context test sets (Beaker)

You have a Qwen3-4B checkpoint trained with one of our attention architectures
(**dense / landmark / compressive / docchunk**). This submits Beaker jobs that evaluate it on the
**5 tasks** — `contra` · `nq` · `oolong` · `outlier` · `rerank` — at **multiple context lengths**,
using **native olmo_core generate (no HF / no vLLM)** on **8 GPUs**. Everything (eval code, eval data,
your checkpoint) is read from **weka**; nothing is copied to the node.

---

## 1. What you need

- **A checkpoint on weka**: a standard olmo_core distcp **step dir** containing `config.json` and
  `model_and_optim/.metadata`, e.g.
  `/weka/oe-training-default/ai2-llm/checkpoints/<you>/<run>/step1234`.
  (The model architecture is read from that `config.json`, so it must be a checkpoint trained with
  one of our launchers.)
- **Its architecture**: `dense`, `landmark`, `compressive`, or `docchunk`.
- **Beaker access**: workspace `ai2/flex2`, budget `ai2/oe-other`; the conda env that has `gantry`
  (`corpus-reasoning-olmo`), with the beaker CLI authed.
- **The eval bundle on weka** (one-time — see §4). A bundle from prior runs is already there.

---

## 2. Quick start — one command

```bash
cd <OLMo-core repo>            # branch prasann/landmark
export PYTHONPATH=src

python src/scripts/train/sft/singletask_ladder/run_q4b_beaker_multirung_eval.py \
    my-eval-label  ai2/jupiter \
    --variant   landmark \
    --ckpt      /weka/oe-training-default/ai2-llm/checkpoints/<you>/<run>/step1234 \
    --task      all
```

| arg | meaning |
|---|---|
| `my-eval-label` (positional) | a name for this eval — used for the Beaker job name and the results dir |
| `ai2/jupiter` (positional) | a **weka-mounting** Beaker cluster with free 8-GPU slots (jupiter, neptune, …) |
| `--variant` | your architecture. `dense/landmark/compressive` use the standard native eval (model built from the checkpoint's `config.json`); `docchunk` uses the doc-chunked eval (**oolong only**) |
| `--ckpt` | **absolute weka path to your step dir** — this is what makes it work for any checkpoint |
| `--task` | `all` (one Beaker job per task) or a comma list, e.g. `--task contra,nq` |
| `--dry-run` | print the jobs without submitting |
| `--max-test` | examples per rung (default **600**); `--batch-size` (default 8); `--priority` |

Each task becomes **one 8-GPU Beaker job** running `torchrun --nproc_per_node=8 eval_lc_native.py`.
Re-running is safe (idempotent; results overwrite).

> Without `--ckpt`, the launcher instead globs the **latest complete step** under
> `checkpoints/prasanns/<run_name>/step*` — that's the path we use for our own runs, where `run_name`
> follows `q4b-<variant>-<task>-ladder32k-10k`. For *your* checkpoint, just pass `--ckpt`.

---

## 3. Rungs & metrics (`--max-test 600`)

| task | context rungs | metric |
|---|---|---|
| `contra` | 2k / 8k / 16k / 32k | F1 |
| `nq` | 3k / 8k / 16k / 32k | F1 (+ EM) |
| `outlier` | 3k / 8k / 16k / 32k | F1 |
| `oolong` | 8k / 16k / 32k | composite score |
| `rerank` | CE pools k20 / k50 / k100 | NDCG@10 + Kendall-τ |

## 3b. Where results land (on weka)

- per task: `checkpoints/prasanns/<my-eval-label>/eval/<task>_multirung.json`
- central collection: `checkpoints/prasanns/_eval_results/<my-eval-label>_<task>_*.json`

Read them on a weka-mounted node, or via `AWS_PROFILE=S3 aws s3 cp s3://ai2-llm/checkpoints/prasanns/_eval_results/...`.

---

## 4. One-time setup: the eval bundle on weka

The eval **code + data** live in a bundle on weka:
`checkpoints/prasanns/_eval_bundle` (corpus-reasoning `scripts/` + `data/`) and
`_eval_bundle_eval500` (the goal-rung ladder files). The on-node runner
`run_beaker_multirung_eval.sh` is served from there too.

If the bundle is missing or you changed eval code:

```bash
bash src/scripts/train/sft/singletask_ladder/upload_lc_eval_bundle.sh    # push code+data+runner to s3
```

> ⚠️ **Staging gotcha:** that uploads to `s3://ai2-llm`, and **s3 does not auto-populate the weka
> filesystem**. To make a fresh/updated bundle visible on `/weka`, run a one-off gantry
> `aws s3 sync s3→/weka` job (creds from the `PRASANNS_AWS_*` beaker secrets — same step we use to
> stage training data; see `README.md` and the `weka-s3-checkpoint-transfer` note). The existing
> bundle is already on weka, so you only need this if you refreshed the bundle.

---

## 5. How it's wired (for reference)

```
launch_beaker_multirung_eval.sh        # driver: loop over variants×tasks (our runs)
        │
        ▼
run_q4b_beaker_multirung_eval.py       # gantry launcher: 1 Beaker job per (label, task)
        │   mounts weka, 8 GPUs, torchrun=False, image=OLMoCoreBeakerImage.stable
        ▼
run_beaker_multirung_eval.sh           # on-node runner (lives in the weka bundle)
        │   resolves CKPT (your --ckpt > --step > latest-complete glob)
        ▼
torchrun --nproc_per_node=8 scripts/eval/eval_lc_native.py   # native, 8-way DP, per rung
```

Notes:
- **Native generate (no HF/vLLM)** — required so landmark/compressive decode correctly.
- The checkpoint must be a **complete** distcp step (`config.json` + `model_and_optim/.metadata`);
  an in-progress step is skipped by the latest-complete glob (and rejected if you point `--ckpt` at it).
- Pick a cluster with free 8-GPU slots; if Jupiter is saturated, try `ai2/neptune`.
