# Closing the train -> eval loop: `ctc-eval` grades a checkpoint this repo trained

2026-08-12, mooney, 1x H200. Task #26, the item `debug/train_smoke/README.md` left owed: *"`ctc-eval`
against one of these checkpoints, to see the fingerprint guard accept a matching format and reject a
mismatched one on real files."*

Both directions are demonstrated below on real files. Getting there took **five defects**, four of
them blocking, and none reachable by the CPU suite — each lives exactly where a unit test hands the
code the thing the real caller forgets.

## The result

| arm | flags | expected | got |
|---|---|---|---|
| **A accept** | `--attn chunked --query-position after` | runs | `f1=0.003 ±0.002  eval_size=500  parse_rate=1.000`, no warnings |
| **B reject** | `--attn chunked --query-position both` | refused | `query_position: train='after' eval='both'` |
| **C reject** | `--attn landmark --query-position after` | refused | `chunk_layout: train='wrap_documents' eval='landmark_documents'` |
| **D override** | B `+ --ignore-format-fingerprint --limit 50` | runs, UNVERIFIED | `f1=0.013 ±0.016  ⚠ eval_size=50 only`, warning recorded |
| **E accept** | `--attn full --query-position after` | runs | `f1=0.005 ±0.003  eval_size=500  parse_rate=1.000` |

The checkpoint is a Qwen3-0.6B trained for **30 steps** on 512 synthetic examples. **The score is
supposed to be near zero** — this tests machinery. What says the machinery works is `parse_rate=1.000`
with `truncated=0`: every generation was well-formed (`[[1, 13], [2, 23], [3, 31]]` against a gold of
`[[2, 48], [11, 40], [19, 32]]`), so the model is answering in the right shape and getting the
arithmetic wrong, which is what 30 steps buys.

A@chunked 0.003 vs E@full 0.005 is **not** a mask effect: both standard errors are ~0.003 and the
model is untrained. Result files (with generations) are in `results/`.

## What was run

```bash
# 1. eval data: mathmatch, the task the checkpoint was trained on, same generator and knobs
ctc-data build --task mathmatch --split eval --rungs 2k --eval-size 500 \
    --out debug/eval_loop_close/data
# -> 500 rows; visible_doc_id_range (1, 48), identical to the training half's

sbatch debug/eval_loop_close/close_the_loop.sbatch   # 3438729  train + first grading pass
sbatch debug/eval_loop_close/grade.sbatch            # 3438743  all five arms after the decode fix
sbatch debug/eval_loop_close/probe_mask_applied.sbatch  # 3438825  is the mask real?
sbatch debug/eval_loop_close/regrade_full.sbatch     # 3438830  the full arm, after defect 5
```

One arm is one `ctc-eval` invocation:

```bash
PYTHONPATH=src:ctc/src python debug/eval_loop_close/ctc_eval_mathmatch.py \
    --data /data/prasann/ctc_eval_loop/data/mathmatch/eval_2k.jsonl \
    --ckpt /data/prasann/ctc_eval_loop/ckpt/step30 --tasks mathmatch \
    --backend native --tokenizer /data/prasann/ctc_eval_loop/tok --max-length 4096 \
    --out /data/prasann/ctc_eval_loop/results \
    --attn chunked --query-position after --tag accept
```

`ctc_eval_mathmatch.py` is a **shim, not a fork**: it registers one `BundleTask` row and calls
`ctc.eval.cli.main` unchanged, so planning, the collision check, backend load, the runner, the guard
and the result writing are all the shipped code. It exists because of defect 0.

## The reject direction, verbatim

```
[ctc-eval] loaded in 17s
[ctc-eval] (1/1) mathmatch@2k  /data/prasann/ctc_eval_loop/data/mathmatch/eval_2k.jsonl
[ctc-eval] FAILED mathmatch@2k: FormatMismatchError: eval format is incompatible with the checkpoint's training format (1 field(s)):
  query_position: train='after' eval='both'
      must be identical; the checkpoint was trained under a different query_position

This is a hard error on purpose. Every one of these failures produces plausible output and a
plausible-looking score, so continuing would generate a number that silently means nothing. Fix the
eval config, or pass --ignore-format-fingerprint if you have established the difference is benign.
mathmatch@2k failed. Fix it, or pass --keep-going to record the failure and continue the sweep.
```

Note *where* it fires: after the checkpoint has loaded (17 s) and before a single token is
generated. Arm C is the same shape with `chunk_layout: train='wrap_documents' eval='landmark_documents'`.

`--ignore-format-fingerprint` behaves as documented — arm D ran the same mismatched configuration to
a number, and the result file says so:

```json
"warnings": [
  "format fingerprint check DISABLED; train/eval format compatibility is UNVERIFIED",
  "limited to the first 50 examples; this is a preview, not a run",
  "eval_size=50 is below 500; quote it inline with the error bar"
]
```

## Defect 0: `ctc-eval` has no ladder for `mathmatch` 🟡 (not fixed, deliberately)

`bundles.BUNDLE` carries the nine CTC-suite ladders and nothing else, so `--tasks mathmatch` exits
with *"unknown task(s) mathmatch"*. `mathmatch` is what every smoke checkpoint here is trained on,
chosen because it is pure-synthetic and in-domain by construction.

Not fixed in the tree on purpose: `GROUPS["all"]` is `tuple(BUNDLE)`, so a row there joins every
`--tasks all` sweep and resolves to a weka file that does not exist. If smoke ladders become
routine, the fix is a group `all` does not include, not a row.

## Defect 1: no checkpoint this repo trains carries `config.json` 🔴

`recipe.py` attaches `ConfigSaverCallback` but nothing sets its `config`, so it wrote nothing and
said so once per save:

```
olmo_core.train.callbacks.config_saver:62 WARNING  Config not set on ConfigSaverCallback, doing nothing
```

Not lost provenance: `TransformerGenerationModule.from_checkpoint` **reads `<ckpt>/config.json`** to
rebuild the model, so the native eval backend could not load any checkpoint this repo has produced.
The 3438664 checkpoints on mooney have `model_and_optim/`, a fingerprint, and no `config.json` —
ungradable as they stand.

Fixed in `src/scripts/ctc/train/run.py` (`_record_config`), which hands the callback
`{model, train_module, dataset, data_loader, trainer}` config dicts, matching olmo-core's own
experiment layout. Verified: the checkpoint from 3438729 has `config.json` and loads in 17 s.

## Defect 2: the guard refused every correctly-matched chunked checkpoint 🔴

The eval side built its fingerprint as `spec.fingerprint(query_position=...)` and nothing else. Of
the eleven compared fields, three are not knowable from the spec, and an unset field is not neutral —
it takes the dataclass default, which describes a plain marker-free build:

```
  chunk_layout: train='wrap_documents' eval='none'
      must be identical; the checkpoint was trained under a different chunk_layout
```

So ACCEPT was impossible: grading a document-chunked checkpoint against exactly the data it was
trained on was refused, in the words a real mismatch uses. A guard that refuses everything gets
switched off — `--ignore-format-fingerprint` becomes routine, and the real mismatches ride along.

The tests could not catch it because they supply the field the evaluator forgets:
`test_convert_to_shards.py:218` asserts
`recovered.require_compatible(spec.fingerprint(chunk_layout="wrap_documents", ...))`, hand-passing
what `runner.py` never passed.

Fixed by `ctc.eval.runner.eval_fingerprint`, which states:

* `chunk_layout`, from the mask and the spec's `chunk_by`. Every backend here wraps items in
  `<|box_start|>`/`<|box_end|>` (`ctc.eval.prefill`), in the `full` arm too, because `full` is a
  *mask*, not a token layout. `landmark` is the one mode whose tokens really differ — arm C.
* `doc_id_range`, **measured** from the rows being graded, which revives the containment rule: the
  digit-range failure the module cites (training capped at 697, eval reaching 1423) is invisible
  unless eval states its range.

`marker_token_ids` and `tokenizer` stay unset: both are backend properties and both compare with
`_optional_exact`, so omitting them weakens the check rather than falsifying it. **A marker-set
mismatch (qwen3 vs qwen3_5) is still not caught by this guard.** Pinned by
`ctc/tests/eval/test_eval_fingerprint.py`, which tests `eval_fingerprint` — what the runner actually
presents — rather than `spec.fingerprint`.

## Defect 3: the accepted arm died at the first decoded token 🔴

Past the guard, every arm that got to generate raised:

```
RuntimeError: 'TorchAttentionBackend' doesn't support KV caching
```

An `AttentionConfig` with `backend` unset resolves to `torch` (`attention/__init__.py:1087`) and the
recipe sets none — the loaded config shows `backend=None`. `TorchAttentionBackend` refuses to build a
KV cache manager, so `prepare_inference_cache` raises at the first decode step, after the checkpoint
has loaded and every prompt has been built. Again: the state of every checkpoint this repo trains.

`from_checkpoint` already accepts an `attention_backend` override; `TransformerGenerationModuleConfig.build`
— the only door the native backend goes through — did not forward it. Fixed in both:

* `src/olmo_core/generate/generation_module/transformer/config.py`: `build(..., attention_backend=None)`,
  forwarded. Additive; default unchanged.
* `ctc/src/ctc/eval/backends/native.py`: `_decode_backend` picks the first installed backend that can
  KV-cache (flash_2, then flash_3, then te), prints it (`[ctc-eval] decoding with attention backend
  flash_2`), and leaves the checkpoint's own backend alone if none can, so olmo-core's error survives
  rather than being replaced by a guess.

Sound because the backend is not what the arms differ in: under `chunked` the prefill mask is built
by `DocumentChunkedAttention` itself (FlexAttention or a materialized mask, neither a backend) and
decode is plain causal over the cache.

## Defect 4: `--attn full` did not turn the chunked mask off 🔴

With A and E both running, they returned **500/500 byte-identical generations**. Too many to be a
null result, so `probe_mask_applied.py` checked the mechanism instead of the score:

```
1. model._document_chunk_attention = {'doc_start_id': 151648, 'doc_end_id': 151649, ...}
   markers in the prefill: {'doc_start_id': 48, 'doc_end_id': 48}  (prompt is 1318 tokens)
2. chunk_ids: 49 distinct values, min=-1 max=47
```

The mask was real and applied — in **both** arms. `NativeBackend._configure_attention` did

```python
disable = getattr(model, "disable_document_chunk_attention", None)
if disable is not None:
    disable()
```

and `Transformer` **had no such method**, so `--attn full` silently did nothing on any checkpoint
whose config carries the mask. The comment above that code said *"Forcing plain causal here is what
makes the 'full' arm mean full"*; the CLI help says the same. Both were false, and the failure is
invisible: a full-vs-chunked comparison was a run compared with itself.

(The shipped 4B checkpoints are not affected the same way — their `config.json` does not carry
`document_chunk_attention`, so nothing enables the mask and `full` is genuinely full. Everything this
repo trains does carry it.)

Fixed: `Transformer.disable_document_chunk_attention()` (`src/olmo_core/nn/transformer/model.py`),
and `_configure_attention` now **raises** if a mask-carrying model cannot be disabled instead of
skipping. After the fix, on the same checkpoint and data:

```
3. max|logit(chunked) - logit(full)| = 5.250000
identical generations chunked vs full: 2 / 500     (was 500 / 500)
```

Pinned by two tests in `ctc/tests/eval/test_backend_parity.py`.

## `--attn chunked` on a GPU: 67bace8ed holds

First GPU exercise of `enable_document_chunk_attention(**self._document_chunk_ids())`. The model
loads in 17 s under `--attn chunked`; the derived ids (`doc_start 151648 / doc_end 151649 /
eos 151643`) are the ones the training fingerprint records; the probe confirms 48 marker pairs in
the prefill become 48 chunks plus FREE. Arms B and C also loaded the model before the guard refused
them, so the loader works under `chunked` and `landmark` alike.

## Still not covered

* **Marker-set (qwen3 vs qwen3_5) mismatch** — not compared; see defect 2.
* **A backend that plain-tokenizes.** `eval_fingerprint` derives `chunk_layout` from the mask, right
  for every backend as configured here (`hf` defaults `structural=True`); an `hf` run with
  `structural=False` would claim `wrap_documents` over a marker-free stream.
* **`strict=True` never runs.** `runner.py` always calls `check_or_explain_missing(..., strict=False)`,
  so a checkpoint with *no* fingerprint, or one graded on a task it never trained, is a warning
  rather than an error. Deliberate (the 4B checkpoints predate fingerprinting), but
  `--ignore-format-fingerprint`'s help — *"grade even when the checkpoint's training format does not
  match **or was never recorded**"* — describes a strictness the code does not have.
* **One task, one rung, one checkpoint**, and a model too small and too briefly trained to say
  anything about scores.
