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
```

From inside the repo, `run/eval.sh` and `run/data.sh` wrap these and resolve the cluster
environment first (node-local interpreter and caches; see `run/_env.sh`).

## Why this is a separate package

`ctc` holds the contract that data generation and evaluation must agree on — prompt templates,
document/chunk serialization, the task registry, gold-index conventions, answer parsers, metrics.
Both halves read one definition, so a prompt-format change that breaks grading fails a test rather
than quietly shifting a number.

It imports no olmo-core. Only `ctc.eval.backends.native` and `ctc.eval.masking.native` may, and
both sit behind the `native` extra.

## Layout

```
src/ctc/
├── format/   the shared contract   (pure python)
├── data/     generators, ladders, audits   — JSONL in, JSONL out
└── eval/     backends, masking, tasks, runner
```

Tokenizing task JSONL into olmo-core training shards is *not* here — it writes olmo-core's format,
so it lives in `src/scripts/ctc/tokenize/` on the training side. Task JSONL is the boundary.
