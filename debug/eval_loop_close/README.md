# Closing the train -> eval loop: `ctc-eval` grades a checkpoint this repo trained

2026-08-12, mooney, 1x H200. Task #26, the item `debug/train_smoke/README.md` left owed: *"`ctc-eval`
against one of these checkpoints, to see the fingerprint guard accept a matching format and reject a
mismatched one on real files."*

Both directions are now demonstrated on real files. Getting there needed **four defects** fixed,
three of which made the loop impossible to close rather than merely awkward — and none of which the
1165-test CPU suite could reach, because each lives exactly where a unit test supplies by hand the
thing the real caller forgets.

## The result, in one table

| arm | command | expected | got |
|---|---|---|---|
| **A accept** | `--attn chunked --query-position after` | runs | see numbers below |
| **B reject** | `--attn chunked --query-position both` | refused | `query_position: train='after' eval='both'` |
| **C reject** | `--attn landmark --query-position after` | refused | `chunk_layout: train='wrap_documents' eval='landmark_documents'` |
| **D override** | B `+ --ignore-format-fingerprint` | runs, UNVERIFIED | recorded in `warnings` |
| **E accept** | `--attn full --query-position after` | runs | see numbers below |

## What was run

```bash
# 1. eval data: mathmatch, the task the checkpoint was trained on, same generator and knobs
ctc-data build --task mathmatch --split eval --rungs 2k --eval-size 500 \
    --out debug/eval_loop_close/data
# -> 500 rows, doc_id_range (1, 48) -- identical to the training half's

# 2. train + grade                                    # job 3438729
sbatch debug/eval_loop_close/close_the_loop.sbatch
# 3. grade again after the decode fix                 # job 3438743
sbatch debug/eval_loop_close/grade.sbatch
```

Each arm is one `ctc-eval` invocation:

```bash
PYTHONPATH=src:ctc/src python debug/eval_loop_close/ctc_eval_mathmatch.py \
    --data /data/prasann/ctc_eval_loop/data/mathmatch/eval_2k.jsonl \
    --ckpt /data/prasann/ctc_eval_loop/ckpt/step30 --tasks mathmatch \
    --backend native --tokenizer /data/prasann/ctc_eval_loop/tok --max-length 4096 \
    --out /data/prasann/ctc_eval_loop/results \
    --attn chunked --query-position after --tag accept
```

`ctc_eval_mathmatch.py` is a **six-line shim, not a fork**: it registers one `BundleTask` row for
`mathmatch` and then calls `ctc.eval.cli.main` unchanged, so planning, the collision check, backend
load, the runner, the guard and the result writing are all the shipped code. It exists because of
defect 0.

The checkpoint is a Qwen3-0.6B trained for 30 steps on 512 synthetic examples. **The score is
expected to be poor** — this tests machinery, not a model.

## Defect 0: `ctc-eval` has no ladder for `mathmatch` 🟡

`bundles.BUNDLE` carries the nine CTC-suite ladders and nothing else, so `--tasks mathmatch` exits
with *"unknown task(s) mathmatch"* before anything runs. `mathmatch` is what every smoke checkpoint
in this repo is trained on, and it is chosen deliberately (pure-synthetic, train and eval from one
generator, so train/eval domain cannot confound a mechanics test).

Not fixed in the source tree, on purpose: `GROUPS["all"]` is `tuple(BUNDLE)`, so an entry there
joins every `--tasks all` sweep and resolves to a weka file that does not exist. The shim registers
it per-run instead. If smoke ladders become routine, the fix is a *group* the `all` shorthand does
not include, not a row.

## Defect 1: no checkpoint this repo trains carries `config.json` 🔴

`recipe.py` attaches `ConfigSaverCallback`, but nothing ever sets its `config`, so it wrote nothing
and said so once per save:

```
olmo_core.train.callbacks.config_saver:62 WARNING  Config not set on ConfigSaverCallback, doing nothing
```

That is not lost provenance. `TransformerGenerationModule.from_checkpoint` **reads
`<ckpt>/config.json`** to rebuild the model, so the native eval backend could not load any
checkpoint this repo has ever produced. The 3438664 checkpoints on mooney have `model_and_optim/`,
a fingerprint, and no `config.json`; they are ungradable as they stand.

Fixed in `src/scripts/ctc/train/run.py` (`_record_config`), which hands the callback
`{model, train_module, dataset, data_loader, trainer}` as config dicts, matching olmo-core's own
experiment layout. Verified: job 3438743's checkpoint has `config.json` and loads.

## Defect 2: the guard refused every correctly-matched chunked checkpoint 🔴

The eval side built its fingerprint as `spec.fingerprint(query_position=...)` and nothing else. Of
the eleven compared fields, three are **not knowable from the spec**, and an unset field is not
neutral — it takes the dataclass default, which describes a plain marker-free build:

```
  chunk_layout: train='wrap_documents' eval='none'
      must be identical; the checkpoint was trained under a different chunk_layout
```

So the ACCEPT direction was impossible: grading a document-chunked checkpoint against exactly the
data it was trained on was refused, in the same words a real mismatch uses. A guard that refuses
everything gets switched off — `--ignore-format-fingerprint` would have become routine, and then
the real mismatches travel with it.

The unit tests could not catch this because they supply the field the evaluator forgets:
`test_convert_to_shards.py:218` asserts `recovered.require_compatible(spec.fingerprint(chunk_layout="wrap_documents", ...))`,
hand-passing what `runner.py` never passed.

Fixed by `ctc.eval.runner.eval_fingerprint`, which states:

* `chunk_layout` from the mask and the spec's `chunk_by` — every backend here wraps items in
  `<|box_start|>`/`<|box_end|>` (`ctc.eval.prefill`), in the `full` arm too, because `full` is a
  *mask*, not a token layout. `landmark` is the one mode whose tokens really differ, which is
  what arm C exercises.
* `doc_id_range` **measured** from the rows being graded, which also revives the containment rule —
  the digit-range failure the fingerprint module cites (training capped at 697, eval reaching 1423)
  is invisible unless eval states its range.

`marker_token_ids` and `tokenizer` are still unset, deliberately: both are backend properties, and
both compare with `_optional_exact`, so omitting them weakens the check rather than falsifying it.
**A marker-set mismatch (qwen3 vs qwen3_5) is therefore still not caught by the guard** — the native
backend derives its own marker ids from the embedding height, which is a different protection, not
this one.

## Defect 3: the accepted arm then died at the first decoded token 🔴

With the guard fixed, arms A/D/E got past it and into generation, where every one of them raised:

```
RuntimeError: 'TorchAttentionBackend' doesn't support KV caching
```

An `AttentionConfig` that leaves `backend` unset resolves to `torch` (`attention/__init__.py:1087`),
and the recipe never sets one — the loaded config shows `backend=None`. `TorchAttentionBackend`
refuses to build a KV cache manager, so `prepare_inference_cache` raises at the first decode step,
*after* the checkpoint has loaded and every prompt has been built. Again: this is the state of every
checkpoint this repo trains.

`TransformerGenerationModule.from_checkpoint` already accepts an `attention_backend` override, but
`TransformerGenerationModuleConfig.build` — the only door the native backend goes through — did not
forward it. Fixed in both places:

* `src/olmo_core/generate/generation_module/transformer/config.py`: `build(..., attention_backend=None)`,
  forwarded. Additive, default unchanged.
* `ctc/src/ctc/eval/backends/native.py`: `_decode_backend` picks the first installed backend that
  can KV-cache (flash_2, then flash_3, then te), prints which, and leaves the checkpoint's own
  backend alone if none can — so olmo-core's error survives rather than being replaced by a guess.

Overriding is sound: under `chunked` the prefill mask is computed by `DocumentChunkedAttention`
itself (FlexAttention or a materialized mask, neither of which is a backend) and decode is plain
causal over the cache. The backend supplies the kernel, not the mask.

## `--attn chunked` on a GPU: the 67bace8ed fix holds

`enable_document_chunk_attention(**self._document_chunk_ids())` is exercised here for the first time
on a GPU. The model loads in 17 s under `--attn chunked` and the ids it derives
(`doc_start 151648 / doc_end 151649 / eos 151643`) are the ones the training fingerprint records.
Arms B and C prove the load specifically: both loaded the model under `--attn chunked` /
`--attn landmark` and were refused afterwards, by the guard, not by the loader.

## Still not covered

* **Marker-set (qwen3 vs qwen3_5) mismatch** — not compared, see defect 2.
* **A backend that plain-tokenizes.** `eval_fingerprint` derives `chunk_layout` from the mask, which
  is right for every backend as configured here (`hf` defaults `structural=True`); an `hf` run built
  with `structural=False` would claim `wrap_documents` over a marker-free stream.
* **`strict=True` never runs.** `runner.py` always calls `check_or_explain_missing(..., strict=False)`,
  so a checkpoint with *no* fingerprint, or one graded on a task it never trained, is a warning and
  not an error. That is deliberate (the 4B suite checkpoints predate fingerprinting) but
  `--ignore-format-fingerprint`'s help — *"grade even when the checkpoint's training format does not
  match **or was never recorded**"* — describes a strictness the code does not have. Only a real
  mismatch is hard.
* **The other four smoke tasks and any real ladder.** One task, one rung, one checkpoint.
