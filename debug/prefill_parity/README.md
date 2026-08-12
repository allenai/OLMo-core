# Does `ctc-eval --share-prefix` change anything but the cost?

**Verdict: NO — it is not score-preserving, and it is not reliably faster.** It rewrote 26.4% of the
generations on a 500-row rung and was 0.94x (i.e. slower) in the one regime it is built for. See
"Results". Jobs: 3438531 (parity), 3438595 (localization).

`ctc-eval --share-prefix` prefills each corpus group's shared token prefix once and reuses its KV
across that group's queries (`ctc/src/ctc/eval/prefix_cache.py`). The claim is that this is *purely*
a speed optimisation. Both the module docstring and `records/running-ctc-evals.md` said the
GPU-level check existed only in the pre-migration tree, and `NativeBackend._generate_with_shared_prefixes`
said in as many words: *"no scored run has gone through this path in this repo."* This directory is
that run.

## What is being compared, and what it is not

The question is whether the **cache path** preserves scores. That is a comparison of one code path
against another **over one file**: both arms read the same rows, build the same prompts, tokenize to
the same ids, and differ only in where the KV of the prompt came from. Nothing about the eval-set
construction enters it.

That is a different question from *"does the fast bundle's data measure the same thing as the
reliable bundle's data"*, for which oolong is the only clean gate (it is a pure regrouping, so the
content is identical and only the row order changes; every other fast task's construction genuinely
moves the data). **This directory answers the first question and says nothing about the second.**

The fast oolong rungs live on weka, which this cluster does not mount; see "Reachability" below.
Since both arms of a path-vs-path comparison read the *same* file, the construction that built that
file is irrelevant to it, and the locally-available fast rungs are as valid a vehicle as oolong's.

## Design

One H200 on horton, one checkpoint, three levels of evidence:

1. **Logit-level** (`parity_probe.py`) — one process, one loaded model. For each row it captures the
   logit vector at the last prompt token from a full prefill and from the reuse path, and compares
   them elementwise, plus the greedy argmax and the fully decoded text. The prefill produces exactly
   one thing, that logit vector; if it is bit-identical the reuse path handed the decoder the state a
   full prefill would have, and no downstream difference is possible.
2. **End-to-end** — `ctc-eval` itself, run twice over the same rung with everything else fixed,
   `--tag plain` / `--tag shared` keeping the result files apart (the filename encodes task, rung and
   mask, but neither the checkpoint nor this flag; `ctc-eval` refuses to overwrite a result whose
   identity differs).
3. **Cost** — wall clock for both arms plus the prefilled-token counts the reuse path reports.

Configurations probed:

| data | construction | query_position | why |
|---|---|---|---|
| `contradiction/rung_2048_tail50.jsonl` | prefix + 50% per-query tail, 39 corpora × 13 queries | `both` | the layout the checkpoint trained with; ~50% of the prompt is a genuine shared token prefix, so the cache path actually engages |
| `nq/rung_8192_mux.jsonl` | query-multiplexed, 61 corpora | `both` | the near-zero-reuse regime: the question precedes the corpus, so the shared *token* prefix is ~1% and the path is expected to fall back |
| `nq/rung_8192_mux.jsonl` | same | `after` | the ~99% reuse regime: the corpus becomes a true prefix. Scores are meaningless (this checkpoint trained with `both`), which does not matter — the arms are compared against each other |
| `contradiction/rung_2048_tail50.jsonl` | " | `both`, `--attn chunked` | the suffix forward sees only the suffix's tokens; if the chunked mask's chunk-id reconstruction needs the whole stream, this is where it breaks |

Checkpoint: `/data/prasann/ctc_suite/ckpts_4b/ctc-s5-contra-full-4b` (Qwen3.5-4B hybrid, so the GDN
recurrent-state snapshot/restore in `prefix_cache.py` is exercised, not just the KV cursor).
Tokenizer `Qwen/Qwen3.5-0.8B-Base`.

## Reachability of the fast bundle

The fast bundle root is `/weka/.../\_eval_bundle_eval500_v2_fast`, and weka is not mounted on the
Berkeley cluster. Locally reachable substitutes were searched for on every node's `/data` via `/net`:

- `debug/fast_bundle/out/` in this repo holds locally built fast rungs for **contradiction, nq,
  outlier and rerank** — these are what this comparison uses, staged node-local to `/data`.
- `/net/{cubbins,lorax,sneetches}/data/prasann/ctc_suite_staged/eval_rungs/oolong/` holds only the
  **old independent** oolong ladder (500 rows, 500 distinct contexts, one query each, no
  `corpus_id`), which shares no prefix and cannot exercise the path.
- No `oolong/*_mux.jsonl` and no `oolong_validation_synth_ctx*.jsonl` source split was found.

## Files

- `parity_probe.py` — the logit/text comparison.
- `run_parity.sbatch` — the job: stages data node-local, runs the probes, then the two `ctc-eval`
  passes per configuration.
- `out/` — probe reports and copies of the `ctc-eval` result files.

## Results

_Pending._
