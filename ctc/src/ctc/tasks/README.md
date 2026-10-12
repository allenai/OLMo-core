# Adding a task

A task is one directory. Nothing outside it needs to change except one line in `__init__.py`.

```
ctc/tasks/<name>/
    __init__.py     registers the spec — the only file the loader imports
    spec.py         the EVAL contract: instruction, parse, score, rungs, gold index base
    generate.py     the DATA contract: source corpus → task JSONL
    sources/        one module per corpus, only when the task has more than one
```

## Why spec and generate are separate files

They change for different reasons, and the difference matters:

- Editing **`spec.py`** reprices every result that already exists. The instruction string is hashed
  into the format fingerprint, so changing it correctly invalidates every checkpoint trained under
  the old wording.
- Editing **`generate.py`** only affects data built afterwards. Existing shards and the checkpoints
  trained on them are untouched.

When both lived in one file it was easy to touch one while meaning to touch the other. The
fingerprint would tell you *that* the format moved, but not which half moved it.

## Steps

1. `mkdir ctc/src/ctc/tasks/<name>` and write `spec.py`. The `TaskSpec` fields are documented in
   `ctc/format/registry.py`; the ones people get wrong:
   - `gold_index_base` — **1 for contradiction, 0 for outlier / rerank / nq.** An off-by-one here
     reads as a weak model, not as a bug.
   - `rungs` — the task's *own* ladder. There is no global set: nq runs 2k–8k, the xlong ladders
     reach 512k.
   - `max_new_tokens` — too small truncates a correct answer into a parse failure, which also reads
     as a capability limit.
   - `primary_metric` — name it, so a results table can't quietly switch between `f1` and
     `exact_match`.
2. Write `generate.py` exposing `build(source, *, rung, seed, **opts) -> Iterator[dict]`.
3. `__init__.py`: `registry.register(SPEC)`, nothing else.
4. Add the name to `TASK_MODULES` in `ctc/tasks/__init__.py`.
5. Add golden cases to `ctc/tests/fixtures/generate_golden.py` if the task needs a serializer or
   parser that doesn't exist yet.

## What does NOT go in a task directory

Anything two tasks share. Pair parsing alone is used by contradiction, redundancy, mathmatch,
matching_ngram and strmatch — and five copies of it is exactly how the copies drifted apart and
produced the grading bugs recorded in `ctc/format/parsing.py`. Shared things live in `ctc/format/`
and are referenced from the spec.

The test for whether something belongs here: *is this true of only this one task?*

## Chain of thought

Task specs carry no CoT builders. The pre-migration `data_format.py` had ~25 `_build_*_cot`
builders and a `cot_mode` knob threaded through prompt assembly; they were dropped in the port
because a check of the whole repo found:

- 150 of 150 CTC-suite result rows are `no-cot`, and `cot_mode` was never even recorded in them.
- Every non-`none` `--cot-mode` in the tree is docstring, `--help` text, or a validation *error
  message*. All 59 real invocations pass `none`.
- The pipeline that would have consumed CoT data (`build_combined_unified.py`, "24 of 184 files are
  CoT variants") was never written.

**The builders were not entirely unused, though.** Three `*_cotmix_*` files were built on
2026-06-13 and still sit in `/net/horton/data/prasann/corpus-reasoning/data/` — contradiction
(`enumerate`), outlier (`template`) and reorder (`successor`), each row tagged `_task` /
`_cot_mode`. Nothing in the reported results consumes them, so they read as an abandoned branch
rather than live data, but rebuilding those specific files would need the builders back from git
history.

Results still carry a `cot_label` field, per the project's labelling convention. If a task needs CoT
targets later, it is one builder in that task's own directory rather than a branch in shared code.
