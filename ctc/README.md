# ctc

Corpus-reasoning **task generation** and **evaluation**. Independent of olmo-core.

```bash
pip install ./ctc              # generators + graders. No GPU, no CUDA, no compiler.
pip install './ctc[vllm]'      # + vLLM backend      (fastest)
pip install './ctc[hf]'        # + transformers backend
pip install './ctc[native]'    # + olmo-core backend (grade a training checkpoint directly)
```

```bash
ctc-eval --list-backends              # what this install can run
ctc-eval --list-tasks
ctc-eval --ckpt CKPT --task contradiction --rungs 2k --backend vllm

ctc-data list
ctc-data build --task contradiction --rungs 2k,4k,8k --out /data/ctc/v3
ctc-data build --task fiqa --split eval --out /data/ctc/v3    # held-out ladders are eval-only
ctc-data build --task cycle --split eval --rungs 64k,1m,10m \
    --eval-size 125 --allow-small-eval --out /data/ctc/xlong  # rungs are open-ended past 32k

ctc-data build --task nq --pool auto --out /data/ctc/v3   # seed pool from the Hub: no GPU, no
                                                          # Lucene index, no LLM mining -- a
                                                          # 20k-example build in about a minute
ctc-data pool export --task nq --out seeds/               # publish side: the expensive load, once

ctc-fingerprint show CKPT           # what format was this checkpoint trained on?
ctc-fingerprint check --ckpt CKPT --task contradiction --query-position both
```

`--task` means the same thing to both `ctc-data` and `ctc-eval`. The per-task command table, and
what each task's corpus needs, is in `src/ctc/data/README.md`; grading — backends, the vLLM
specifics (including the Qwen3.5 serving-copy requirement), bundles and the eval guards — is in
`src/ctc/eval/README.md`; training entry points are in `src/scripts/ctc/README.md` at the repo
level.

From inside the repo, `run/data.sh` → `run/convert.sh` → `run/train.sh` → `run/eval.sh` cover the
whole pipeline and resolve the cluster environment first (node-local interpreter, caches, and this
checkout at the front of `PYTHONPATH`; see `run/_env.sh`).

## Why this is a separate package

`ctc` holds the contract that data generation and evaluation must agree on — prompt templates,
document/chunk serialization, the task registry, gold-index conventions, answer parsers, metrics.
Both halves read one definition, so a prompt-format change that breaks grading fails a test rather
than quietly shifting a number.

It imports no olmo-core. Only `ctc.eval.backends.native`, `ctc.eval.masking.native` and
`ctc.train` may, and the first two sit behind the `native` extra.

## The format guard

A checkpoint is only meaningful against the format it was trained on, and every way that can go
wrong produces plausible output and a plausible score. So the format is **recorded** — beside the
shards at tokenize time, and inside every checkpoint at train time via
`ctc.train.FormatFingerprintCallback` — and eval **refuses** to grade a mismatch.

The record is per-task, because a mix trains several; eval asks "was *this* task trained, in *this*
layout". Absence and mismatch are different answers: an untrained task is an out-of-distribution
eval, which the guard can only decline to verify, while a mismatch is an error in both modes.

This is not hypothetical. Reproducing one pre-migration number took two extra runs to find that
`query_position` differed, because nothing recorded it. It is now a fingerprint field.

## Layout

```
src/ctc/
├── format/   the shared contract   (pure python)
├── data/     generators, ladders, audits   — JSONL in, JSONL out
└── eval/     backends, masking, tasks, runner
```

Tokenizing task JSONL into olmo-core training shards is *not* here — it writes olmo-core's format,
so it lives in `src/scripts/ctc/tokenize/` on the training side. Task JSONL is the boundary.
