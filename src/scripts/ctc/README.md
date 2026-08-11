# `src/scripts/ctc/` — the training-side layer

Everything that writes or reads **olmo-core's** formats. The `ctc` package deliberately stops at
task JSONL and stays dependency-free; the moment a step needs `numpy`, `torch` or `olmo_core`, it
belongs here.

| script | what it does |
|---|---|
| `convert_to_shards.py` | unified task JSONL → `.npy` SFT shards (dense or landmark document-chunked layout), plus `metadata.json` and the shard's format fingerprint |
| `train/sft.py` | reference SFT run — loss on answer tokens, one padded example per window |
| `train/cpt.py` | reference CPT run — loss on every token, documents packed |
| `train/options.py` | the arguments, and nothing that needs a GPU |
| `train/recipe.py` | the one parameterised recipe both entry points and both clusters share |
| `train/run.py` | run here, or hand the same options to Beaker |

## Training

Two entry points, one recipe, both clusters. The pre-migration tree had 161 launchers, most
differing only in architecture, learning rate and node count — 20 CPT scripts of ~210 lines each.
Those axes are now arguments.

```bash
# local: Berkeley H200, torchrun, no weka/Beaker
PYTHONPATH=src:ctc/src torchrun --nproc-per-node=8 src/scripts/ctc/train/sft.py my-run \
    --data /data/prasann/ctc/shards/contradiction:2 \
    --data /data/prasann/ctc/shards/nq:1 \
    --base /data/prasann/bases/q4b-dense-fixmark --arch chunked --max-steps 1100

# Beaker: same options, plus where to run
PYTHONPATH=src:ctc/src python src/scripts/ctc/train/sft.py my-run \
    --cluster ai2/jupiter-cirrascale-2 --nodes 4 \
    --data /weka/.../contradiction:2 --data /weka/.../nq:1 \
    --base /weka/.../step2385/model_and_optim --arch chunked --max-steps 1100
```

`--data DIR[:WEIGHT[:LABEL]]` is repeatable and weights are ratios, so `a:2 b:1` mixes 2:1.
`--arch` is one of `full` / `chunked` / `hierarchical` / `landmark`, and must match how the shards
were converted — which the format fingerprint checks at step 0 rather than at checkpoint one.

SFT and CPT expose the *same* options and differ in exactly three things: SFT pads one already
chunked example per window and computes loss on answer tokens only, CPT packs documents and trains
on every token, and their budgets are naturally steps and tokens respectively.

Two defaults chosen to fail loudly rather than quietly:

- **`--base` is required** unless you pass `--from-scratch`. Training from random init when you
  meant to fine-tune produces a run that looks healthy and means nothing. The error also reminds
  you that a fresh Qwen3 base needs its marker embeddings repaired first — those rows are
  bit-identical out of the box, so the model cannot tell an open-document marker from a close one.
- **The format fingerprint is written by default.** Turning it off leaves a checkpoint whose eval
  format cannot be verified later.

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
