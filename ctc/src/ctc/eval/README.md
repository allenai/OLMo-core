# `ctc.eval` — grading a checkpoint

One command, one checkpoint, any number of tasks and rungs:

```bash
pip install './ctc[vllm]'      # or [hf] / [native]; bare install still parses and scores

ctc-eval --list-backends                       # what this install can run
ctc-eval --ckpt CKPT --tasks main --backend vllm --out results/
ctc-eval --ckpt CKPT --tasks contradiction --rungs 2k,8k --attn chunked
ctc-eval --ckpt CKPT --tasks main --rungs xlong    # 64k-2M, opt-in: one 256k rung is hours
```

`--tasks main` is the five in-distribution ladders, `ood` the four held-out ones, `all` both, or
a comma list. `--backend` selects how text gets generated and **nothing else**: prompt, parser and
scorer come from `ctc.format` and are shared, so a score that moves when only `--backend` moves is
a backend bug (`ctc/tests/eval/test_backend_parity.py` holds that line). Every result row carries
`eval_size`, its standard error, and `parse_rate`.

## The full loop from a bare clone (no cluster, no weka)

```bash
ctc-data build --task contradiction --pool auto --split eval --out /data/ctc/mydata
ctc-eval --ckpt CKPT --tasks contradiction --bundle /data/ctc/mydata --backend vllm
```

`--bundle` names where the eval JSONL lives: a registered bundle (`v2_clean` is the default and
lives on AI2 weka — see `--list-bundles`), **or any directory of `ctc-data build` output**, which
is recognized by its `<task>/eval_<rung>.jsonl` layout and graded as-is. `$CTC_EVAL_BUNDLE` sets
the default. Numbers from different bundles are not comparable — the same rung label can be a
different corpus size (contradiction 64k is n=1602 in `v2` and n=1525 in `v2_clean`).

## The vLLM backend

Fastest by an order of magnitude on prefill-heavy rungs; the reference numbers for full-attention
arms are measured on it. Three things to know before pointing it at a checkpoint:

1. **It is fed token ids, not prompt strings.** The prefill (including the document-marker
   scaffold) is built host-side by the same code the native backend uses; letting vLLM tokenize a
   prompt string would silently grade a different input.
2. **Stop conditions are enforced host-side over the decoded text.** vLLM's own `stop` is only an
   early-exit optimisation. This is what makes vLLM/HF/native numbers comparable at all.
3. **An olmo-exported Qwen3.5 needs a serving copy, not the raw export.** `export_olmo_to_hf.py`
   writes a text-only checkpoint, and vLLM resolves every `Qwen3_5*` architecture to a multimodal
   class that dies reading `vision_config` at construction. The backend detects a raw export and
   refuses with instructions rather than letting the `AttributeError` surface. Building a serving
   copy takes three scripts (wrapper config, `model.*` → `model.language_model.*` rename,
   synthesized `visual.*` params); plain Qwen3 and HF-native checkpoints need none of this. See
   `backends/vllm.py::build` for the whole recipe, including the architecture override and
   `limit_mm_per_prompt` it applies for `model_family="qwen3_5"`.

One honest limitation: `--attn chunked` under vLLM grades the chunked **token stream** — markers
and all — but vLLM has no way to install our attention masks, so it runs full attention over it
unless the chunked-vLLM patch is in play. The native backend applies the real mask; a
chunked-vs-full *mask* comparison belongs there.

## Guards you will meet (all deliberate, all bypassable by name)

- An over-long prompt **stops the run** (`--allow-truncated` to count-and-continue instead): a
  skipped prompt scores a clean 0.0, indistinguishable from a wrong answer.
- A checkpoint whose recorded training format does not match the eval format is refused
  (`--ignore-format-fingerprint` to proceed; the mismatch is then stamped into the result).
- A result file written by a different checkpoint, bundle or query position is refused before the
  model loads (`--overwrite` to replace).
- `--query-position` defaults to `both` and **must match training**: the shipped checkpoints
  trained with `both`, and `after` collapses them (nq 0.860 → 0.074).

## Layout

```
ctc/eval/
├── cli.py         ctc-eval
├── bundles.py     which file grades which task at which rung — the one copy of that mapping
├── runner.py      the task loop, backend-agnostic and testable against a fake generate()
├── prefill.py     example -> token ids (plain, or the document-marker scaffold)
├── stopping.py    host-side stop rules; four historical truncation bugs live here as tests
├── backends/      vllm / hf / native, all behind one interface
├── masking/       the chunked/landmark masks, for the backends that can install them
└── tasks/         per-task adapters
```
