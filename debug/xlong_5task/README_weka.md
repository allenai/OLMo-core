# 5-task SFT mix, 2k → 256k short-skewed (Qwen3.5)

Built 2026-07-27. Extends the canonical 5-task mix (contradiction / nq / oolong / rerank / outlier)
from its 32k ceiling to a 2k–256k ladder, skewed toward short contexts.

```
/weka/oe-training-default/ai2-llm/checkpoints/prasanns/xlong5_2k256k_qwen35/
├── shards_chunked/         # CHUNKED arm  -- WITH <|box_start|>/<|box_end|> markers
│   ├── contradiction_train/  nq_train/  oolong_train/  outlier_train/  rerank_train/
├── shards_full/            # STANDARD arm -- NO markers (plain full attention)
│   ├── contradiction_train/  nq_train/  oolong_train/  outlier_train/  rerank_train/
└── eval/                   # eval ladders
    ├── contra/  nq/  outlier/  rerank/     # 64k/128k/256k rungs, eval_size=500
    └── oolong/                              # 2k/4k/64k/128k/256k rungs
```

## Two shard sets: chunked vs standard

`shards_chunked/` and `shards_full/` are built from the **same pools** with the same `build_prompt`, the same
chat template, the same tokenizer and the same EOS. They differ **only** by the 2 document-boundary
tokens per document, so the arms are a controlled pair:

| task | inst (chk / std) | tok (chunked) | tok (standard) | delta | 2×n_docs | marker ids (std) |
|---|---|---|---|---|---|---|
| contradiction | 19,965 / 19,988 | 362.0M | 351.9M | 10.12M | 16.01M | **0** |
| nq | 19,990 / 19,997 | 354.9M | 352.3M | 2.63M | 4.46M | **0** |
| oolong | 19,994 / 20,000 | 354.7M | 342.7M | 11.98M | 12.43M | **0** |
| outlier | 19,992 / 19,997 | 356.2M | 352.6M | 3.55M | 4.85M | **0** |
| rerank | 19,941 / 19,962 | 366.7M | 364.2M | 2.47M | 7.91M | **0** |

`delta` is smaller than `2×n_docs` because the two effects run opposite ways: removing the markers
subtracts 2 tokens per document, but the shorter marker-free instances then FIT under the 262,144
cap, so the standard build retains a handful of near-cap examples the chunked build dropped. The
residual divided by the extra retained instances comes out at 256,020 / 260,649 / 260,108 / 259,173
tokens for contradiction / nq / outlier / rerank — i.e. right at the cap, which is the confirmation.
(oolong is the exception at ~75k per extra instance: its drops are line-based and include
reserved-token collisions, not only length. That is 0.13% of the task's tokens.)

`metadata.json` records `doc_markers: true|false`. Use `shards_chunked/` for the chunked arm (it needs
the boundaries to derive chunk ids) and `shards_full/` for the standard arm.

## Loader compatibility

Both sets are standard olmo-core SFT shards (`token_ids_part_*.npy` uint32 + `labels_mask_part_*.npy`
bool + `metadata.json`, EOS-separated, completion-only loss), so they drop into the composable path
used by `src/scripts/train/memexpress/local_4b/*-local.py`:

```python
NumpyDocumentSourceConfig(
    source_paths=[f"{root}/token_ids_part_*.npy"],
    label_mask_paths=[f"{root}/labels_mask_*.npy"],
)
```

Use `TokenizerConfig.qwen3_5()` (vocab 248320, EOS/BOS/pad 248044). As in the Qwen3 launchers, pass
`replace(tokenizer_config, bos_token_id=None)` as the *document* tokenizer config so EOS-based
document splitting works (Qwen3.5 also ties bos == eos == 248044).

## Tokenization

`--emit dense --marker-set qwen3_5 --seq-len 262144 --cot-mode none`, tokenizer
`Qwen3.5-4B-Base`. Markers `<|box_start|>`=248049 / `<|box_end|>`=248050, EOS=248044.
`--use-titles` OFF, `--free-pad-repeat` 0, `--repeat-doc-text` 1.

The standard arm is built with `--no-doc-markers`, which renders the identical prompt with the
boundary strings empty. (The CTC-suite convention of running both arms off ONE marker-wrapped shard
set via a train-time `--variant` flag is deliberately **not** used here — the standard arm gets
genuinely marker-free data.)

## Realized context-length distribution (measured from the chunked shards, not the plan)

| task | 2–4k | 4–8k | 8–16k | 16–32k | 32–64k | 64–128k | 128–256k | instances | tokens |
|---|---|---|---|---|---|---|---|---|---|
| contradiction | 5466 | 5027 | 4045 | 2810 | 1510 | 668 | 350 | 19,965 | 362.0M |
| nq | 5708 | 5055 | 4035 | 2742 | 1446 | 612 | 315 | 19,990 | 354.9M |
| oolong | 5926 | 5023 | 3984 | 2734 | 1413 | 607 | 307 | 19,994 | 354.7M |
| outlier | 5481 | 5257 | 4101 | 2753 | 1469 | 611 | 311 | 19,992 | 356.2M |
| rerank | 4964 | 5311 | 4082 | 2883 | 1597 | 711 | 351 | 19,941 | 366.7M |

Every task clears the required floor of **≥300 examples at 128–256k**. Max instance length 261,771;
nothing exceeds the 262,144 cap.

## Qwen3.5 calibration (measured, 300 examples/task, MAPE 1.2–3.3%)

`tokens = intercept + tok_per_doc · n_docs`

| task | intercept | tok/doc |
|---|---|---|
| contradiction | 188.1 | 42.41 |
| nq | 26.1 | 156.54 |
| outlier | −5.4 | 144.33 |
| rerank | 13.8 | 85.23 |

Within 1–3% of the Qwen3 values, so eval rung `n` values are valid for both tokenizers.

## Validation

- **Chunk-leak scan: ALL SHARDS CLEAN** — 0 inter-chunk FREE tokens across 12.6M chunks
  (`debug/xlong_5task/validate_train_shard_leak.py`). A "leak" is any token between the end of one
  chunk and the start of the next; such a token bridges supposedly-isolated documents.
- **Source pools audited**: nq hard-negative ratio 0.097 (the banned 98%-hard build is ~0.99);
  gold indices in range for every task; rerank `ce_scores` present on 300/300 sampled.
- **oolong train/eval overlap: checked, and there is none.** A prefix signature (query + first 200
  chars of the packed body) initially flagged 108/57/18 examples in the 8k/16k/32k rungs. Comparing
  the FULL body and answer for every flagged pair showed **0 exact duplicates and 0 shared bodies** —
  all 183 were *prefix-only* collisions, because independently synthesized oolong examples share a
  preamble and opening item by construction. Verified by
  `debug/xlong_5task/verify_contamination.py`; do not re-derive contamination from a prefix
  signature alone.
- Two consequences of that false positive, both harmless but worth knowing: 236 oolong training
  examples (1.18%) were removed before the final tokenization and did not need to be, and the oolong
  2k/4k eval rungs were over-filtered (still 1730 / 1649, well above 500).

## ⚠ Known-bad variants this build avoids

| trap | status |
|---|---|
| oolong `--item-regex '\|\|'` — a bare `\|\|` is an empty-branch alternation matching *every* line, so instruction/question/header became their own chunks with FREE `\n\n` bridges | **Fixed 2026-07-26.** Converter now rejects any `--item-regex` matching the empty string. **Every oolong shard built before that date has this defect.** |
| `expand_example` is grow-only — with `keep_all=True`, targets below the source's natural size left examples long, making the whole 2k–16k range unreachable (nq had 40 examples at 2–4k instead of 6000) | Fixed with a shrink step keeping gold → hard negatives → distractors |
| FREE-token id/title wrapping leak | Fixed in `abccf2837`, in HEAD |
| NQ 98%-hard (`hn49`/`hn99`/`hn199`/`ladder64k`) | Not used — p10 pool only |
| `free_pad_repeat` / `repeat_doc_text` (verbatim repetition, void knobs) | Left at defaults |

## ⚠ Caveats to carry into any reported number

- **rerank above ~32k**: the added negatives are random non-gold documents, not CE-mined hard
  negatives — `ce=None`, which the grader treats as gain 0 and excludes from the Kendall-tau
  reference. This is the *same* approximation the eval rung makes, so train and eval agree, but
  "rerank @128k" measures surfacing CE-relevant docs among *more* noise, not *harder* noise.
- `gold_doc_indices` index base is **per task**: contradiction is 1-indexed, nq/outlier/rerank are
  0-indexed. Applying one convention everywhere fabricates an off-by-one defect.
- Qwen3.5 is a GDN hybrid with `n_kv_heads=4` on its full-attention blocks → **context parallelism
  is hard-capped at CP=4** (ring CP is unavailable; `GatedDeltaNet.apply_cp()` raises).

## Provenance

Build scripts: `debug/xlong_5task/` in `PrasannS/OLMo-core` (branch `prasann/landmark`).
Source pools: contradiction/outlier/rerank/oolong from `ctc_suite_data/*_pool/`, nq from
`nq_p10_20k/nq_train_k25-202_clean.jsonl`.
