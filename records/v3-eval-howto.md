# Running the v2 and v3 eval ladders

**Short version:** same launcher, add `--ladder-version v3`. Run v2 for everything, and v3 for
**contra and outlier only** — the other three tasks are the same files in both bundles, so running
them twice burns GPU hours to produce identical numbers.

## What v3 is

`_eval_bundle_eval500_v3` is `_eval_bundle_eval500_v2_clean` with two tasks rebuilt:

| task | v2 | v3 | comparable? |
|---|---|---|---|
| **contra** | `both`-mode gold (polarity flip + numeric edit) | `realistic`-mode, matching the training generator | ❌ **no** |
| **outlier** | xlong rungs with **K pinned at 25** while n grew 32× | true scale-K (K ≈ n/9.5) at every rung, majority-vs-outlier gap ≥ 2 | ❌ **no** (xlong only) |
| nq / rerank / oolong | — | *directory symlinks to v2_clean* | ✅ identical files |

Why it exists: the 5-task training mix is `realistic`-mode contradiction, so scoring it on `both`-mode
gold measured a task the model was never trained for. On a CTC checkpoint that mismatch alone was
worth **0.559 → 0.946 f1** at n=762. Outlier had the analogous problem on the M axis — the O(NM)
claim needs M to scale with N, and the shipped xlong rungs held M fixed.

**v3 numbers go in their own column. Never put a v3 contra number beside a v2 one.**

v3 tops out at **1M** (v2 reaches 2M). The 2M outlier rung was deliberately left out rather than
carried over, because the v2 file is the K-frozen build — an absent rung is loud, a wrong one isn't.

## The commands

From the `corpus-reasoning-olmo` env, with the branch pushed (gantry checks out a remote ref).

```bash
export PY=/data/prasann/conda/envs/corpus-reasoning-olmo/bin/python   # node-local; see local_cluster.md
L=src/scripts/train/memexpress/singletask_ladder/run_q4b_beaker_multirung_eval.py
CKPT=/weka/oe-training-default/ai2-llm/checkpoints/prasanns/<run>/<step>
```

**Pass A — v2, all 9 tasks (5 in-distribution + 4 OOD), base ladder:**

```bash
PYTHONPATH=src python $L <run-name> ai2/jupiter-cirrascale-2 \
    --task all --ckpt $CKPT \
    --query-position after --tokenizer Qwen/Qwen3.5-0.8B
```

**Pass B — v3, contra + outlier only, base ladder:**

```bash
PYTHONPATH=src python $L <run-name> ai2/jupiter-cirrascale-2 \
    --task contra,outlier --ckpt $CKPT --ladder-version v3 \
    --query-position after --tokenizer Qwen/Qwen3.5-0.8B
```

**Pass C — xlong.** Add `--xlong --xlong-rungs`, split by YaRN group, one checkpoint per submission:

```bash
# native RoPE, no YaRN copy needed
... --xlong --xlong-rungs 64k,128k --ckpt $CKPT
# 256k/512k need factor 2, 1M needs factor 4 -- build the copies first:
#   python debug/ctx_ceiling_4b/make_yarn_copy.py --src $CKPT --factor 2
... --xlong --xlong-rungs 256k,512k --ckpt ${CKPT}_yarn2
... --xlong --xlong-rungs 1M        --ckpt ${CKPT}_yarn4
```

Add `--ladder-version v3` to any of these for the v3 ladder. 256k+ needs an 80GB GPU
(`ai2/jupiter-cirrascale-2`); the runner sets `PREFILL_CHUNK_SIZE` itself there.

## Things that will bite

**`--query-position` must match the training shards.** `after` for anything trained on
`xlong5_2k256k_qwen35_qafter`; `both` (the default) for everything before 2026-08-11. Evaluating a
query-after model with `both` hands it a second copy of the ask it never saw, which reads as a
capability gap rather than a prompt mismatch.

**Use `Qwen/Qwen3.5-0.8B` for Qwen3.5 models.** The runner now infers the family from the
checkpoint's `config.json`, but passing it explicitly always wins and costs nothing.

Two separate things here, which an earlier version of this doc ran together:

- **Family must match.** A Qwen3 tokenizer on a Qwen3.5 model produces an all-zeros result *while
  the job reports success* — this cost an overnight sweep of 27 jobs. This is the dangerous one.
- **Base vs Instruct, not 0.8B vs 4B.** Any `-Base` repo sets `eos == pad == <|endoftext|>`
  (248044), which trips olmo_core's `pad_token_id and eos_token_id must be different` check and
  **crashes at startup** — loud, harmless, before any GPU work. Instruct repos keep them distinct.
  The whole family shares one vocab, so borrowing the 0.8B tokenizer for a 4B model is safe: every
  id is identical.

Known wart, not worth blocking on: training terminates instances with `<|endoftext|>` (248044) per
the `qwen3_5` marker set, while the instruct tokenizer's `eos` is `<|im_end|>` (248046). Decoded
output is still trimmed correctly (the trim loop breaks on `pad_token_id`, which *is* 248044), so
scores are unaffected — but generation does not early-terminate, and only contradiction sets
`stop_strings`. Outlier/nq/rerank/oolong therefore run their full token budget every row.

**Result files no longer collide, as of this commit.** v2 writes `<task>_multirung.json` and v3
writes `<task>_multirung_v3.json`. Before the fix both wrote the same filename, so a v3 run
overwrote the v2 result in place — same run dir, same task, silently different eval set. If you have
results from before this change, check the `ladder_version` field inside the JSON rather than
trusting the filename.

**Outlier does NOT need a raised `max_new_tokens`** — an earlier version of this doc said to raise it
to 512, which was wrong. The outlier answer is one sentence plus three IDs:

```
Most documents are about various topics and the outliers are about Lucy Diakovska.
Outliers: [111], [158], [296]
```

That is ~112 characters (~30 tokens) on the n=451 rung, and only the ID digits grow with n — about
35 tokens at n=7209. The 200-token default is ~6× the requirement. Scale-K raises the number of
*majority* topics, but the model never enumerates them; it names the outlier topic and lists three
IDs. (The ≥512 budget is a **contradiction**-with-CoT concern — enumerate-CoT there needs ~2200 at
large n — and it does not transfer to outlier.)

**Cluster choice is about scheduling, not just GPUs.** jupiter is strict-priority with
unallocated-only backfill, so it can sit pending when full; saturn/neptune/ceres backfill eagerly.
Dense Qwen3.5 needs H100 (FlashAttention-3 is Hopper-only) — that rules out A100 pools.

## Before quoting a number

- `eval_size` is the eval-set size; `n` is the corpus size. Never write `n=488` for an eval set.
- Anything under 500 examples gets its size and error bar inline, next to the number.
- ≥256k runs must be labeled with their YaRN factor in `other_notes`, or the number is unusable —
  nobody can tell later whether it was an in-ceiling measurement.
- Write the ledger entry at submission time (`records/eval_launches/`), including the Beaker
  experiment id. `pull-evals` is what flips a job to `done`.
