# Evaluate ANY checkpoint on the 5 long-context test sets (Beaker)

You have a Qwen3-4B checkpoint trained with one of our attention architectures
(**dense / landmark / compressive / docchunk**). This submits Beaker jobs that evaluate it on the
**5 tasks** — `contra` · `nq` · `oolong` · `outlier` · `rerank` — at **multiple context lengths**,
using **native olmo_core generate (no HF / no vLLM)** on **`--ngpu` GPUs (default 2)** per job — the 4B
fits on 2, so more evals run concurrently. Everything (eval code, eval data, your checkpoint) is read
from **weka**; nothing is copied to the node.

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
| `ai2/jupiter` (positional) | a **weka-mounting** Beaker cluster with free GPU slots — 2 per job (jupiter, neptune, …) |
| `--variant` | your architecture. `dense/landmark/compressive` use the standard native eval (model built from the checkpoint's `config.json`); `docchunk` uses the doc-chunked eval (**oolong only**) |
| `--ckpt` | **absolute weka path to your step dir** — this is what makes it work for any checkpoint |
| `--results-dir` | absolute weka dir for the result JSONs (default: `checkpoints/prasanns/<my-eval-label>/eval`) — point it anywhere on weka, e.g. next to your checkpoint |
| `--task` | `all` (one Beaker job per task) or a comma list, e.g. `--task contra,nq` |
| `--ladder-version` | `v1` (default, original per-rung files) or **`v2`** (cleaned ladders — see §3c) |
| `--ngpu` | GPUs per job (default **2**); each task runs `torchrun --nproc_per_node=$NGPU` |
| `--dry-run` | print the jobs without submitting |
| `--max-test` | examples per rung (default **600**); `--batch-size` (default **2**); `--priority` |

Each task becomes **one Beaker job** (`--ngpu` GPUs, default 2) running
`torchrun --nproc_per_node=$NGPU eval_lc_native.py`. Re-running is safe (idempotent; results overwrite).

> **`landmark` / `compressive` force `batch_size=1`** (their native decode can't be batched / left-padded);
> `dense` / `docchunk` keep the configured `--batch-size`.

> Without `--ckpt`, the launcher instead globs the **latest complete step** under
> `checkpoints/prasanns/<run_name>/step*` — that's the path we use for our own runs, where `run_name`
> follows `q4b-<variant>-<task>-ladder32k-10k`. For *your* checkpoint, just pass `--ckpt`.

---

## 3. Rungs & metrics (`--max-test 600`)

| task | context rungs | metric |
|---|---|---|
| `contra` | 2k / 8k / 16k / 32k | F1 |
| `nq` | 3k / 8k / 16k / 32k | doc-retrieval F1 (+ EM) |
| `outlier` | 3k / 8k / 16k / 32k | F1 |
| `oolong` | 8k / 16k / 32k | composite score |
| `rerank` | CE pools k20 / k50 / k100 | NDCG@10 + Kendall-τ |

> **NQ is doc-index retrieval** (the model outputs `Relevant Document: [N]`; scored by set-F1 over doc
> IDs), matching the training task — *not* answer-text QA. The rungs use the **p10 pipeline**:
> `nq_validation_k{20,50,100,200}_hn{2,5,10,20}_600.jsonl` = **10% hard negatives** (rest random) + a
> **cross-encoder gold-quality filter** (removes false negatives), all docs from `wikipedia-dpr-100w`.
> This matches the training-data negative distribution. (The old `hn19/hn49/hn99/hn199` files were 98%
> hard-neg and much harder — don't mix them in.) No `align` step (redundant: single-source docs).

## 3c. v2 evals (`--ladder-version v2`) — comparable rungs, ≥500 examples

The default (`v1`) rung files were each generated **independently**, so a task's questions differ
from rung to rung — length and question-set are confounded. Pass **`--ladder-version v2`** to use the
cleaned ladders where, within a task, **every rung shares the SAME ≥500 questions/answers and only the
distractor documents change** to hit each context length. This isolates length (and makes different
corpora comparable), which is the point of the ladder.

```bash
python src/scripts/train/sft/singletask_ladder/run_q4b_beaker_multirung_eval.py \
    my-eval-label ai2/jupiter --variant landmark --ckpt <weka step dir> \
    --task all --ladder-version v2
```

How v2 is built (`corpus-reasoning/scripts/data/build_v2_eval_ladders.py`, verified by
`verify_v2_eval_ladders.py`): one canonical ≥500-question set per task (the largest rung), with each
shorter rung a **nested prefix of the shuffled distractor pool** — gold docs / query / answer stay
byte-identical, indices remapped. contra tops up its 500-question base with offline-harvested PubMed
fillers; rerank keeps CE `ce_scores` aligned. Rungs:

| task | v2 rungs (docs/ex) | n/rung |
|---|---|---|
| `contra` | 2k/8k/16k/32k = n100/190/385/765 | 500 |
| `nq` | 3k/8k/16k/32k = k20/50/100/200 | 600 |
| `outlier` | 3k/8k/16k/32k = n22/55/110/220 | 600 |
| `rerank` | 3k/8k/16k = k20/50/100 (CE-graded; **no 32k** CE pool) | 500 |
| `oolong` | 8k/16k/32k (freshly synthesized, **disjoint from training**; v1's oolong eval overlapped its own train split) — same question-type + corpus-type distribution across rungs | ≥500 |

Data lives on weka at `_eval_bundle_eval500_v2` (staged by `upload_lc_eval_bundle.sh`). v1 files are
untouched. Note: rerank's CE passages are short, so its true token lengths (~1.6k/4k/8k) sit under the
3k/8k/16k labels — kept for continuity with v1.

## 3b. Where results land (on weka)

- per task: `<results-dir>/<task>_multirung.json` — `<results-dir>` defaults to
  `checkpoints/prasanns/<my-eval-label>/eval`, override with `--results-dir <abs weka dir>`
  (e.g. point it next to your own checkpoint).
- central collection (always): `checkpoints/prasanns/_eval_results/<my-eval-label>_<task>_*.json`

Read them on a weka-mounted node, or via `AWS_PROFILE=S3 aws s3 cp s3://ai2-llm/checkpoints/prasanns/_eval_results/...`.

---

## 4. Setup: eval CODE (in-repo) + eval DATA (on weka)

**Code** lives in the repo: `src/scripts/ctc_eval/` — a self-contained `ctc_eval` package (ported from
corpus-reasoning, pyserini/vllm made lazy). The Beaker job clones OLMo-core, so the eval code is
**version-controlled and always current** — read/modify it directly in the repo; there is no code bundle
to refresh.

**Data** lives on weka (it changes rarely): the test sets + ladder files under
`checkpoints/prasanns/_eval_bundle/data`, and the goal-rung files under `_eval_bundle_eval500`. The
runner reads them via `--root` + `EVAL500_ROOT`.

⚠️ **s3 does not auto-populate the weka filesystem.** If the eval data isn't on weka yet (or you added
new test sets), push to s3 then stage it onto weka:

```bash
bash src/scripts/train/sft/singletask_ladder/upload_lc_eval_bundle.sh    # push data to s3
# then a one-off gantry job: aws s3 sync s3://ai2-llm/checkpoints/prasanns/_eval_bundle{,_eval500}
#   -> /weka/oe-training-default/ai2-llm/checkpoints/prasanns/_eval_bundle{,_eval500}
#   (creds from the PRASANNS_AWS_* beaker secrets; see README.md + the weka-s3-checkpoint-transfer note)
```

---

## 5. How it's wired (for reference)

```
launch_beaker_multirung_eval.sh        # driver: loop over variants×tasks (our runs)
        │
        ▼
run_q4b_beaker_multirung_eval.py       # gantry launcher: 1 Beaker job per (label, task)
        │   mounts weka, --ngpu GPUs (default 2), torchrun=False, image=OLMoCoreBeakerImage.stable
        ▼
run_beaker_multirung_eval.sh           # on-node runner (IN-REPO; gantry clones it)
        │   resolves CKPT (your --ckpt > --step > latest-complete glob)
        │   PYTHONPATH=repo/src/scripts (ctc_eval) ; DATA via --root=weka bundle + EVAL500_ROOT
        ▼
torchrun --nproc_per_node=$NGPU src/scripts/ctc_eval/eval/eval_lc_native.py   # native DP over --ngpu GPUs (default 2), per rung
```

Notes:
- **Native generate (no HF/vLLM)** — required so landmark/compressive decode correctly.
- The checkpoint must be a **complete** distcp step (`config.json` + `model_and_optim/.metadata`);
  an in-progress step is skipped by the latest-complete glob (and rejected if you point `--ckpt` at it).
- Pick a cluster with free GPU slots (2 per job by default); if Jupiter is saturated, try `ai2/neptune`.
