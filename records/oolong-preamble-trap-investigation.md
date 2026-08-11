# Trap 3 — "oolong preamble train/eval layout mismatch"

**VERDICT: ALREADY FIXED in code. Trap 3 is the `--item-regex '||'` bug, misfiled as a separate
live mismatch. The still-real part is on DISK: the CTC oolong shard predates the 2026-07-26 fix
and must be rebuilt — the converter itself needs no change before the port.**

Investigated read-only in the pre-migration repo (`prasann/landmark`). Paths below are relative to
`/accounts/projects/berkeleynlp/prasann/projects/OLMo-core` unless stated.

---

## 1. Train and eval call the shared segmenter with identical arguments (VERIFIED by reading code)

Both sides go through `segment_prompt_to_chunks`
(`src/olmo_core/data/document_chunk_landmark.py:438`), which for `chunk_by="line"` does exactly one
thing: `_wrap_item_lines(prompt, re.compile(item_regex), ...)` at line 517. There is no
train-only or eval-only branch inside it. So a mismatch can only come from the arguments.

Argument-by-argument diff for oolong, TRAIN
(`src/scripts/data/convert_unified_to_document_landmark.py`, as invoked by
`src/scripts/data/ctc_suite/convert_ctc_p0_dense_cubbins.sbatch:64-109`) vs EVAL:

| arg | TRAIN | EVAL (ladder / rung driver) | EVAL (standalone driver) | match? |
|---|---|---|---|---|
| `task` | `oolong` | `oolong` | `oolong` | ✓ |
| `chunk_by` | `line` (sbatch `CHUNK=line`) | `"line"` (`..._ladder.py:178`) | `TASK_CFG["oolong"]["chunk_by"]="line"` (`eval_lc_native_docchunk.py:73`) | ✓ |
| `item_regex` | `r"\|\|"` — argparse default, `convert_unified_to_document_landmark.py:248`; sbatch sets `EXTRA=""`, i.e. **no override** | `r"\|\|"` (`..._ladder.py:178`) | `r"\|\|"` hardcoded (`eval_lc_native_docchunk.py:180`) | ✓ |
| `query_position` | `both` (default, `:239`) | `"both"` (`..._ladder.py:337`) | `--query-position` default `both` | ✓ |
| `cot_mode` | `none` (sbatch passes `--cot-mode none`; confirmed in the shard's `metadata.json`) | `"none"` (`..._ladder.py:337`; `run_rung_eval.py:123`, default `none` at `:457`) | **`"plan"`** (`TASK_CFG["oolong"]["cot"]`) unless `--cot-mode none` | ✓ on the CTC path, ⚠ see §4 |
| `use_titles` | `False` (flag absent) | not passed → `False` | not passed → `False` | ✓ |
| `free_pad_repeat` / `repeat_doc_text` / `summary_every_k` | `0 / 1 / 0` | not passed → `0 / 1 / 0` | same | ✓ |
| `include_answer` | `True` | `False` | `False` | by design (the prompt is asserted to be a prefix, `:550`) |

Every argument that can change the wrapping is identical. **There is no live train/eval preamble
layout difference in the current code.** Grep confirms every one of the 8 call sites in the repo
passes the escaped `r"\|\|"`; no launcher passes a bare `'||'` any more.

## 2. What the 2026-08-05 note was describing (VERIFIED)

The `||` bug and "trap 3" are the same defect, described twice. Three independent confirmations:

- `debug/ctc_vllm_validation/CHUNK_LEAK_AUDIT.md` — the source of the "layout mismatch" phrasing —
  measures 2019 inter-chunk FREE tokens (~5/example) in `oolong_train` and describes exactly the
  `||` symptom (instruction / question / header wrapped as their own chunks, free `\n\n` between).
- The converter's own docstring (`convert_unified_to_document_landmark.py:22-28`) says so verbatim:
  the bare `'||'` "is exactly the oolong train/eval layout mismatch measured in
  `CHUNK_LEAK_AUDIT.md`".
- The regression test `src/test/data/document_chunk_item_regex_test.py` asserts the good regex wraps
  only the 3 data items with 0 inter-chunk free chars, and that the bare `'||'` reproduces the leak.

The startup guard is at `convert_unified_to_document_landmark.py:365-382` (rejects any
`--item-regex` matching the empty string). Fix commit: `8bd79c0ec`, 2026-07-27.

`records/paper-v2-todo-status.md:262-267` already states the correct diagnosis ("shards built before
2026-07-26 hit the `--item-regex '||'` leak bug"). The later note at `:394-398` re-states it as a
live code mismatch; that framing is what created trap 3.

## 3. Empirical confirmation — already run, both directions (VERIFIED, this session)

Scanned the raw uint32 token stream directly (first 5M tokens of part 0, EOS-aware, the same metric
as `debug/xlong_5task/validate_train_shard_leak.py`):

| shard | built | chunks | examples | inter-chunk FREE gaps |
|---|---|---|---|---|
| `/net/cubbins/data/prasann/ctc_suite/shards/oolong_train` | **2026-07-19 (pre-fix)** | 86459 | 253 | **1269 gaps / 1269 tokens (~5/example, all single token id 271)** |
| `/net/cubbins/data/prasann/xlong5/shards_chunked/oolong_train` | 2026-07-27 (post-fix) | 80302 | 1604 | **0** |

The ~5-per-example rate is the `||` signature (preamble lines only). If the escaped regex were the
problem you would see a gap at *every* item boundary (~340/example here), not 5. Eval side is
independently clean: `debug/ctc_vllm_validation/CHUNK_LEAK_rung{2048,8192,32768}.json` record oolong
`inter_chunk_free_total: 0`, `wrap_short_ex: 0`, chunks 6–905.

So: the code is fixed, the post-fix shard is provably clean, and the CTC shard still on disk is the
bad one. That is the whole of trap 3.

Scope note: the defect was `--chunk-by line`-only. `--chunk-by document` goes through
`_wrap_documents` (`document_chunk_landmark.py:317`), which never consults `item_regex`, and all 24
document tasks measured 0 in both audits.

## 4. Two real gotchas found on the way (VERIFIED — worth carrying into the new repo)

1. **`metadata.json` does not record `item_regex` (or `query_position`).** The written dict is
   `convert_unified_to_document_landmark.py:506-530`; `item_regex` is threaded into `tok_kwargs`
   (`:457`) but dropped from the metadata. You therefore **cannot tell a good oolong shard from a
   bad one by reading its metadata** — you must scan the token stream. Adding `item_regex` and
   `query_position` to that dict during the port is a cheap, worthwhile change.
2. **`TASK_CFG["oolong"]["cot"] = "plan"`** in `src/corpus_reasoning/eval/eval_lc_native_docchunk.py:73`
   while every converter builds oolong with `--cot-mode none`. The CTC ladder / rung drivers hardcode
   `none` so the CTC numbers are unaffected, but anyone running that standalone driver on an oolong
   checkpoint **without** `--cot-mode none` evaluates a different preamble than the model trained on.
   This is a live train/eval *prompt-text* difference (not a wrapping one) and is a plausible second
   contributor to oolong's odd numbers. Consider making the default `none` in the ported copy.

## 5. The empirical check to re-run (if you want it independently)

CPU-only, no GPU, ~1 min. Point it at a shard directory tree:

```bash
cd /accounts/projects/berkeleynlp/prasann/projects/OLMo-core
python debug/xlong_5task/validate_train_shard_leak.py \
    --root /net/cubbins/data/prasann/ctc_suite/shards --parts 1
# expect: oolong  leak_gaps>0  -> BAD (pre-fix shard)

python debug/xlong_5task/validate_train_shard_leak.py \
    --root /net/cubbins/data/prasann/xlong5/shards_chunked --parts 1
# expect: oolong  leak_gaps 0  -> CLEAN
```

(`--root` takes the parent of the `*_train` dirs. `/net/...` is the slow NFS view of a node's local
disk — fine for reading one 50–100 MB part, do not point a job at it.)

Eval-side counterpart, for the prefill layout:

```bash
python debug/ctc_vllm_validation/validate_chunk_leak.py \
    --eval-root <eval_rungs dir> --rung 8192 --max-samples 40 --tasks oolong
```

## 6. Bottom line for the port

- Port `convert_unified_to_document_landmark.py` **as-is** — the guard at `:365-382` and the escaped
  default at `:248` are the fix, and they are already in the file you are porting.
- Do **not** carry over the CTC `oolong_train` shard from `ctc_suite/shards` (2026-07-19). Rebuild
  oolong, or reuse the already-clean `xlong5/shards_chunked/oolong_train` build recipe.
- Any oolong result computed from a pre-2026-07-26 shard (including every CTC oolong `-cmix` number
  in `paper-v2-todo-status.md`) is measured on a defective shard and needs a retrain, not a re-eval.
- Optional, cheap: record `item_regex`/`query_position` in `metadata.json`, and flip the standalone
  evaluator's oolong `cot` default to `none` (§4).
