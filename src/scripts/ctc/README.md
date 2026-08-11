# `src/scripts/ctc/` — the training-side layer

Everything that writes or reads **olmo-core's** formats. The `ctc` package deliberately stops at
task JSONL and stays dependency-free; the moment a step needs `numpy`, `torch` or `olmo_core`, it
belongs here.

| script | what it does |
|---|---|
| `convert_to_shards.py` | unified task JSONL → `.npy` SFT shards (dense or landmark document-chunked layout), plus `metadata.json` and the shard's format fingerprint |

## The pipeline, end to end

```
ctc-data build            task JSONL          (ctc package, no deps)
  ↓
convert_to_shards.py      .npy shards + format_fingerprint.json
  ↓
train                     FormatFingerprintCallback collects the shards'
                          fingerprints into every checkpoint
  ↓
ctc-eval                  refuses to grade a checkpoint against a format
                          it was not trained on
```

The fingerprint is the thread running through all four stages, and the converter is where it
enters. A shard built without one produces a checkpoint whose eval format cannot be verified —
which downgrades the guard to a warning, and a warning is what let a `query_position` mismatch cost
two GPU runs to find by bisection.

## Things that must match between convert and eval

These are converter flags whose value has to be reproduced at eval time, because the model was
trained on the layout they produce and nothing at eval time can infer it:

`--query-position` · `--use-titles` · `--free-pad-repeat` · `--repeat-doc-text` ·
`--summary-every-k` · `--marker-set`

The first and the last are recorded in the fingerprint and checked automatically. The others are
recorded in `metadata.json` but not compared — they change the token stream, so treat a mismatch as
a real risk rather than a formality.

`--attn` is **not** in this list. Full-vs-chunked is a mask, not a token layout: both arms run over
identically tokenized shards, box markers and all, which is exactly what makes the comparison a
comparison of attention.
