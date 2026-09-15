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

## Three length axes, and only one of them is the right one

**`gold_output_words` (primary).** Words in the full gold answer string as
`_build_output` (`src/scripts/ctc_eval/lib/data_format.py`) builds it — the single definition of
what the model was trained to emit. For outlier that is a chain-of-thought sentence *plus* the id
line:

```
Most passages are about Geology, Volcano, Mineral and the outliers are about Jazz.
Outliers: [3], [8], [12]
```

The CoT grows with the number of majority topics, so the emitted length varies a lot even though
the payload is always 3 ids. This axis is the one that answers "does accuracy depend on how much
has to be produced".

**`gold_items` (secondary).** How many things the answer names — 3 pairs, 3 ids, 1 entry. On these
ladders it is *constant* for most tasks, and it is **not** the length of the output. Kept because
it needs nothing but the sidecar, so it still works when the bundle is unreachable.

**`gold_payload_tokens` (secondary).** Tokens in the payload rendering only. Same caveat.

### Where the gold text comes from, and why it is trustworthy

`_record_gens` saves only the prompt tail, not `expected_output`, so the gold answer is rebuilt
here from the same rung file through the same steps: `load_jsonl`, then the same seeded
subsample (`random.seed(42)`, `random.sample`) when the file exceeds `MAX_TEST`, then
`_build_output`. `ladder_paths.py` holds the rung→file map transcribed from `eval_lc_native.py`.

That reconstruction is **verified, not assumed**. Every rebuilt example's gold is compared against
the gold the eval already recorded in the sidecar, and a task where even one example disagrees is
dropped from the emitted-length axis and named in the report. A stale path, a changed bundle, or a
`MAX_TEST` mismatch therefore surfaces as "task DROPPED" — never as a plausible-looking wrong
number. The payload axes never read the bundle, so they survive such a failure untouched.

`--cot-mode` selects the reasoning prefix (`label` is the library default and what the eval's own
loader uses). **`COT_MODE=none` is the control**: rebuild the same golds without the CoT and see
whether the effect survives. If it does, the effect is about the answer; if it vanishes, it was
about the reasoning prefix.

## Reading the output

Three things are load-bearing and printed before the headline table:

1. **What actually varies, per task.** Several tasks are generated at a fixed `k` (`..._k3`
   rungs), so their *payload* never varies — but their emitted length can still vary through the
   CoT prefix, and the table shows both plus `%CoT`. A task that is constant on the axis being
   plotted contributes a single bucket and says nothing about length sensitivity, while still
   moving the macro-average in whatever bucket it lands in; those are flagged
   **NO (single bucket)**.
   A second table compares gold emitted length against each model's own median generation length
   — a model answering far outside the trained format changes what the score means.
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

# CONTROL: same golds with the CoT stripped. If the effect survives, it is about the answer;
# if it vanishes, it was about the reasoning prefix.
COT_MODE=none NAME=gold-len-nocot src/scripts/train/memexpress/singletask_ladder/gold_length/launch_gold_length_gantry.sh
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
