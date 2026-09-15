# Accuracy conditioned on gold output length

Does a model's score depend on **how much it has to emit**? The 256k dense and 256k
compressive-landmark runs are scored by the same multirung ladder, so a single scalar per task
is comparable — but it hides where a model loses. A compressed-KV model plausibly degrades as
the gold answer gets longer, because every additional emitted item is another independent read
out of the compressed context.

This slices every per-example score by gold answer length and reports both models side by side.

## What it reads

The `<task>_multirung.generations.jsonl` sidecars that
`src/scripts/ctc_eval/eval/eval_lc_native.py` already wrote next to each checkpoint on weka.
No GPU, no checkpoint load, no re-decoding. The only reason it needs a Beaker job is that weka
is not mounted at Berkeley.

Gold comes out of each record's `detail` dict, which is populated by the task's eval function
in `src/scripts/ctc_eval/eval/evaluate.py`:

| ladder task | eval fn | gold key in `detail` | per-example metric |
|---|---|---|---|
| contradiction, contra_fever | `_eval_contradiction` | `gold_pairs` | `f1` |
| nq, fiqa, scifact | `_eval_retrieval` | `gold_ids` | `f1` |
| outlier, outlier_review | `_eval_outlier` | `gold` | `f1` |
| oolong | `_eval_oolong` | `gold` (gold_list) | `score` |
| rerank | `_eval_rerank` | `gold` | `ndcg@10` → `mrr@10` |

Multi-query retrieval examples aggregate away the per-query gold and are dropped — counted and
reported in the audit section, never dropped silently.

## Two length axes

- **`gold_items`** — how many things the gold contains (pairs / ids / list entries). The primary
  axis: it is the natural "how much must be produced".
- **`gold_tokens`** — tokens in the canonical rendering of the gold answer, via the Qwen3.5
  tokenizer staged on weka. Falls back to a whitespace word count if that is unreachable, and
  the report always says which was used.

## Reading the output

Three things are load-bearing and printed before the headline table:

1. **Gold-length variation per task.** Several of these tasks are generated at a fixed `k`
   (`..._k3` rungs), so their gold length may not vary at all. A task with one distinct
   `gold_items` value contributes a single bucket and says nothing about length sensitivity —
   but it still moves the macro-average in whatever bucket it lands in. The table flags those
   with **NO (single bucket)**.
2. **Pairing.** Only `(task, rung, idx, eval_tag)` keys present in *both* models enter the
   headline table, so a bucket never compares one model's examples against a different subset
   of the other's. `eval_tag` is part of the key because `eval_xlong256k` and the `eval_yarn2-*`
   dirs re-run the same short rungs under different serving configs. If the two models ever
   disagree on the gold for a shared key, the job says so loudly — that means the join is
   misaligned and the numbers are void.
3. **`⚠ eval_size=N`.** Slicing by length makes cells small fast. Anything under 500 examples
   carries its size inline (repo rule); read the error bar, not the third decimal.

`rerank` is excluded from the macro-average by default: its output is a fixed top-10 ranking
regardless of how many documents are relevant, so its gold length does not measure emitted
length the way the others do. `--include-fixed-output-tasks` folds it back in.

Averaging over tasks is **macro** — task mean first, then the unweighted mean over tasks — so a
task contributing 600 examples to a bucket cannot outvote one contributing 40. The
example-weighted pooled mean sits next to it in `summary.json`.

## Running it

```bash
# On the cluster (weka mounted). Commit AND push first — gantry clones your pushed HEAD.
src/scripts/train/memexpress/singletask_ladder/gold_length/launch_gold_length_gantry.sh

# Include the YaRN-served dirs as separate configs
EVAL_DIRS='eval,eval_xlong,eval_xlong256k,eval_yarn2-256k,eval_yarn2-512k,eval_yarn4-1M' \
  NAME=gold-len-with-yarn src/scripts/train/memexpress/singletask_ladder/gold_length/launch_gold_length_gantry.sh

# Short rungs only, to separate "long output" from "long context"
RUNGS='2k,3k,8k,16k,32k' NAME=gold-len-short-rungs src/scripts/train/memexpress/singletask_ladder/gold_length/launch_gold_length_gantry.sh
```

The full report goes to stdout as well as to `$OUT_DIR/report.md`, so it can be pulled from the
beaker logs without a second weka job:

```bash
beaker experiment results <experiment-id>
```

Outputs: `report.md` (markdown tables), `summary.json` (machine-readable buckets + coverage),
`per_example.jsonl` (one row per paired example — re-bucket or plot without rerunning the job).

Locally, against any directory laid out like a checkpoint root:

```bash
python src/scripts/train/memexpress/singletask_ladder/gold_length/gold_length_conditioned_accuracy.py \
  --model dense=<root> --model compressive=<root> --tokenizer '' --out-dir /tmp/out
```
