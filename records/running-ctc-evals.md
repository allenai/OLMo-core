# Running CTC-suite evals on Beaker

For anyone who has a checkpoint and wants its numbers. You need a Beaker account in `ai2/flex2` and
a clone of this repo on the `prasann/ctc` branch. You do **not** need a GPU, a weka mount, the eval
data, or the checkpoint locally — everything runs on the node.

## The command

```bash
python src/scripts/ctc/eval_beaker.py MY_RUN_NAME                          # Qwen3 checkpoint
python src/scripts/ctc/eval_beaker.py MY_RUN_NAME \
    --tokenizer Qwen/Qwen3.5-0.8B-Base                                     # Qwen3.5 checkpoint
```

`MY_RUN_NAME` is a directory under `/weka/oe-training-default/ai2-llm/checkpoints/prasanns/`. The
latest complete `stepN` inside it is found on the node, so you don't have to look one up. Results
land in `<that directory>/ctc_eval/` as one JSON per task-rung.

That grades the **five in-distribution tasks** — contradiction, nq, outlier, rerank, oolong — over
every rung each one defines (18 rungs total). Common variations:

```bash
# the four held-out (OOD) ladders instead: fiqa, scifact, outlier_review, contra_fever
python src/scripts/ctc/eval_beaker.py MY_RUN --tasks ood

# a document-chunked checkpoint (must match how it was trained)
python src/scripts/ctc/eval_beaker.py MY_RUN --tasks main --attn chunked

# one task, two rungs, pinned step
python src/scripts/ctc/eval_beaker.py MY_RUN --tasks contradiction --rungs 2k,8k --step step1100

# a checkpoint that isn't under checkpoints/prasanns/
python src/scripts/ctc/eval_beaker.py --ckpt /weka/.../some/step2000
```

`--dry-run` prints the exact command the node will run without submitting. Do that first if you're
changing anything.

## Before your first launch

**gantry clones the pushed commit, not your working tree.** If you have local changes, commit and
push them or the job silently runs older code. This is the single most common way one of these jobs
produces a confusing result.

**Check the bundle once**, especially if someone has rebuilt eval data:

```bash
python src/scripts/ctc/eval_beaker.py --check-bundle
```

That submits a GPU-less job that only resolves every rung to a file and reports whether it exists.
It takes about a minute and is much cheaper than discovering a missing rung an hour into a sweep.

## What comes back

One JSON per task-rung in the results directory, named `<task>_<rung>_<attn>.json`. Each carries the
metric, its **standard error**, `eval_size`, `parse_rate`, the prompt-length distribution, and the
git commit that produced it. The job's log ends with a summary table of every rung it ran.

Three fields are worth reading before you trust a number:

- **`parse_rate`** — below 1.0 means some generations didn't parse. A parse failure and a wrong
  answer both score zero, so a drop here is a decoding or truncation problem, not a capability one.
  Read the generations (they're in the JSON) before concluding anything.
- **`eval_size`** — scifact's ladder is 299 examples, below the 500 floor. Quote the size and its
  error bar inline next to any number from it.
- **`warnings`** — where a format-fingerprint mismatch or a truncated prompt gets recorded.

## Things that will bite you

**`--tokenizer` must match the checkpoint's family, and the default is Qwen3.** The two families
have different vocabularies — Qwen3 is 151,936 and Qwen3.5 is 248,320 — so grading a Qwen3.5
checkpoint with the default gets every token id wrong, including EOS, and the run *completes* and
reports f1 = 0.000. It reads as a dead model. This is now a hard error naming the right id, but the
flag still has to be set:

```bash
--tokenizer Qwen/Qwen3-4B             # Qwen3   (vocab 151,936) — the default
--tokenizer Qwen/Qwen3.5-0.8B-Base    # Qwen3.5 (vocab 248,320) — one tokenizer, all sizes
```

A Qwen3.5 checkpoint is easy to spot in the load log: `vocab_size=248320` and a
`block_pattern=['gdn','gdn','gdn','attn']`.

**`--attn` must match how the checkpoint was trained.** `full` forces plain causal attention even on
a checkpoint whose config carries the document-chunked mask, and `chunked` enables it. Grading an
arm under the wrong mask produces a plausible number, not an error. The eval checks the checkpoint's
recorded format fingerprint and warns on a mismatch; if the checkpoint predates fingerprinting the
check can't run, and you'll need `--ignore-format-fingerprint` — the result then records that
compatibility was **unverified**.

**`--query-position` must also match, and the default is right.** These checkpoints trained with
`both`, which renders the question before *and* after the corpus. `after` collapses them (nq 0.860 →
0.074). Don't change it unless you know the checkpoint was trained that way.

**A rung label bounds corpus size, not prompt length.** Contradiction prompts at the "4k" rung have
been measured from 3,457 to 23,796 tokens. `--max-length` defaults to 40960 and is sized from the
model, not from `--rungs`. If prompts exceed it the run **stops** rather than skipping them — a
skipped prompt scores a clean 0.0, which is indistinguishable from a wrong answer, and that failure
mode once silently dropped 354 of 500 examples in both arms at once. Pass `--allow-truncated` only
if you want them counted and reported instead.

**`rerank`'s reliable ladder has no 32k rung** (its CE-filtered hard-negative pool caps at 100
documents per query) and `fiqa` stops at 16k. `--rungs all` handles this; an explicit `--rungs 32k`
skips those tasks with a note. The *fast* bundle does reach 32k on rerank — pooling makes the corpus
size a free parameter (`ctc/src/ctc/eval/bundles.py:373-385`).

## Bundles: which eval data, and why the name matters

`--bundle` picks the eval data. `ctc-eval --list-bundles` shows what's registered; a directory path
works too, for a staged local copy.

| bundle | kind | what it is |
|---|---|---|
| `v2_clean` | reliable | **default.** The v2 ladder with contradiction rebuilt against a PubMed-only filler pool and its rungs recalibrated. |
| `v2` | reliable | The original v2 ladder. Kept because existing results — including the 256k runs — were produced against it. |
| `fast` | fast | Shared-corpus rungs, 8k–1M, over all five in-distribution tasks (contradiction / nq / outlier / rerank / oolong). Cheaper on contradiction; **not comparable to a reliable number.** See below. |

**Bundles are not interchangeable, and this is the one thing to get right.** The same rung label
maps to *different files with different corpus sizes*: contradiction's 64k rung is `n=1602` in `v2`
and `n=1525` in `v2_clean`, because the clean rebuild recalibrated after the original's filler pool
turned out to be 92–99% FEVER/wiki rather than PubMed. A "64k contradiction" number from one bundle
is not comparable to one from the other. Every result file therefore records `bundle`,
`bundle_root` and `bundle_kind`.

`kind` distinguishes **reliable** (one independently sampled corpus per row, so 500 rows are 500
independent measurements) from **fast** (many queries share a corpus so a prefill can be reused).
Rebuilding an eval set to share corpora measurably moves scores, so the two kinds are separate
bundles rather than a flag on one.

### Ultra-long rungs (64k–2M) are opt-in

```bash
--rungs xlong            # 64k, 128k, 256k, 512k, 1M, 2M
--rungs 64k,256k         # or pick them individually
```

Deliberately excluded from `--rungs all`: one 256k rung is hours per task, and it should never start
by accident. Asking for `xlong` on a task that has none raises rather than returning an empty list —
a missing row reads as "scored nothing", not "never ran".

**Coverage differs by bundle**, verified against weka:

| | contradiction | nq | outlier | rerank |
|---|---|---|---|---|
| `v2` | 64k–2M | 64k–2M | 64k–2M | 64k–2M |
| `v2_clean` | 64k–2M | — | — | 128k, 256k, 1M, 2M |

So **for 64k+ on nq or outlier you must pass `--bundle v2`.** The default has no such files.

## Which data this runs on

The `_eval_bundle_eval500_v2_clean` bundle on weka. Every rung of a task grades the **same
questions** — only the distractor documents change — so a rung-to-rung difference is a length
effect and not eval-set resampling noise.

The bundle is `_clean` rather than the original `_v2` because contradiction's rungs in that one were
calibrated against a filler pool that turned out to be 92–99% FEVER/wiki rather than PubMed;
Wikipedia claims tokenize at ~22.8 tok/doc against PubMed's ~43, so every contradiction rung
overshot its label by about 1.8×.

The rung-to-file mapping lives in `ctc/src/ctc/eval/bundles.py` — one table, because it used to be
retyped in three drivers and the copies drifted.

## Running it without Beaker

`ctc-eval` is the same command the node runs, so a local GPU works too:

```bash
PYTHONPATH=src:ctc/src python -m ctc.eval.cli \
    --ckpt /data/prasann/ckpts/my-run/step1100 \
    --tasks main --attn chunked \
    --bundle /data/prasann/eval_bundle --out results/my-run
```

Point `--bundle` (or `$CTC_EVAL_BUNDLE`) at a staged copy of the bundle. On the Berkeley cluster
that copy must live on node-local `/data` — `/scratch` and `/accounts` are both NFS at ~5 MB/s and
will deadlock concurrent readers.

## The fast (shared-corpus) bundle

```bash
python src/scripts/ctc/eval_beaker.py MY_RUN --bundle fast --share-prefix \
    --tasks contradiction --rungs xlong
```

Many queries share one corpus, so the shared part is prefilled once and its KV reused.
`--share-prefix` is what turns the reuse on; without it you get the fast *data* at the reliable
cost. The reuse is measured and printed per rung rather than assumed, and falls back to plain
prefills when there is nothing to share.

**Coverage is 8k–1M, over all five in-distribution tasks:**

| task | construction | shared | eval_size |
|---|---|---|---|
| contradiction | prefix + per-query tail, 10% tail | 90% of the prefill | 500 |
| outlier | candidates planted in the prefix, decoys eliminated by the tail | 90% of the prefill | 500 |
| nq | query-multiplexed | 100% of the documents | 500 |
| rerank | query-multiplexed | 100% of the documents | 500 |
| oolong | already 25 questions per context | 100% of the documents | **100 ⚠** |

**oolong's 100 rows are a fifth of the 500 floor** — quote the size and its error bar (SE ≈ ±0.046
at 0.7) inline next to any oolong number from this bundle. It earns its place anyway: the build is
a pure regrouping of a split that already stores 25 questions per context, so the file's *content*
is identical to the independent one and only the row order changes. That makes it the correctness
gate for KV reuse — a content-identical variant whose score moves means the cache path is wrong.

### How much prefill you actually save depends on `query_position`

Shared documents are not shared tokens. Measured on the built files with the Qwen3 tokenizer
(`debug/fast_bundle/measure_reuse.py`, no GPU needed):

| rung | `both` shared prefix | `after` shared prefix | prefill left, `both` → `after` |
|---|---|---|---|
| nq @8k | 66 tok (0.9%) | 7,298 tok (99.6%) | 99.2% → **13.5%** |
| rerank @16k | 85 tok (1.0%) | 8,809 tok (99.5%) | 99.2% → **11.7%** |
| contradiction @8k | 14,514 tok (90.6%) | 14,432 tok (90.5%) | 11.0% → 11.0% |

`both` renders `{questions}\n\n{documents}\n\n{questions}`, so the per-query question sits ahead of
the corpus and the identical document block is not a token *prefix*. Under `after` the corpus is
the prefix and the multiplexed tasks go from **no saving at all to ~7–8×**. contradiction is
unaffected either way — its question block is empty, so it already had ~90%.

So the multiplexed half of this bundle is built for models trained with the question after the
context. The data is already correct for that; nothing needs rebuilding when those checkpoints
arrive. Grading a checkpoint trained with `both` under `after` still collapses it (nq 0.860 →
0.074), so `--query-position` must match training, as always.

### outlier is planted, not tailed — and it is the one task generated rather than transformed

The obvious shared build for outlier puts the answer's trio in the per-query tail, and that is
exactly where its measured **+0.215** came from: a scatter control held corpus contents fixed and
moved only the golds, and the rebuilt corpus was faithful (0.330 vs 0.320) while the placement was
the whole effect.

So it is built the other way round. K candidate topics are planted in the **shared prefix** with 3
documents each; each query's tail tops up every candidate *except its own* to 5, leaving exactly one
topic at 3. The answer therefore lives in the body of the context and never in the tail.

Two properties worth knowing:

- **Topping decoys to 5 rather than 4 is deliberate.** The real rungs' smallest majority topic is 4
  in 484/500 rows, so a decoy at 4 would sit exactly on the discrimination boundary and put ~28
  topics one document from the answer where the source has ~6. At 5 the decoys hide in the modal
  majority size and the natural size-4 topics remain the only competitors. The corpus gap stays 1,
  matching the reliable rung.
- **Absence from the tail is only a shortcut at the short rungs**, and it is measured per rung
  rather than argued. The answer's topic is the one candidate the tail does not top up, but every
  majority topic that donated no camouflage is also absent, so absence is diluted — by how much
  depends on how many topics the corpus holds:

  | rung | 8k | 16k | 32k | 64k | 256k | 512k | 1M |
  |---|---|---|---|---|---|---|---|
  | guess-among-absent | **0.203 ⚠** | 0.090 | 0.048 | 0.024 | 0.006 | 0.003 | 0.0015 |

  **Use 32k and above.** At 8k there are only ~11 topics, so absence narrows the field to a handful
  and the heuristic scores 0.203 against ~0.09 chance — close enough to a real score to contaminate
  it. Never quote an 8k outlier number from this bundle without that control beside it.

Because the construction needs to know each document's topic — and the eval files strip that — the
rungs are generated from the wiki100w article pool. They match the reliable rungs in corpus size but
**share no documents with them**. Every other fast task is a transform of existing v2 data; this one
is not, and that is worth stating on a results row.

### A fast number is a different measurement

Not a cheaper route to the same number. Against the independent rungs: **outlier +0.215/+0.261,
contradiction −0.102…−0.175**. A scatter control that held corpus contents fixed and moved only
the gold documents attributed outlier's entire shift to *placement* — the rebuilt corpus was
faithful (0.330 vs 0.320); putting the golds at the end was not.

So use the fast bundle to compare **arms against each other** at a length you otherwise could not
afford, and never to fill a cell in a table whose other cells came from `v2_clean`. Every result
file records `bundle`, `bundle_kind` and `share_prefix`, so which one produced a number is always
answerable after the fact.

The KV-reuse machinery itself (`ctc.eval.prefix_cache`) is score-preserving by construction and was
verified byte-identical against the plain path in the pre-migration tree — but **not yet re-verified
on a GPU in this repo**. Treat the first fast numbers as unconfirmed until that parity run exists.

### Rebuilding it

```bash
python src/scripts/ctc/build_fast_bundle.py                      # print the matrix, submit nothing
python src/scripts/ctc/build_fast_bundle.py --tasks contradiction --from-bundle v2_clean --submit
```

One CPU-only Beaker job per task-rung, reading the reliable bundle on weka and writing
`_eval_bundle_eval500_v2_fast` beside it. Nothing is staged through this machine. contradiction is
built from `v2_clean`; nq and rerank from `v2`, because `v2_clean` has no ultra-long rungs for
them — each output file's `.manifest.json` records the source it actually came from.
