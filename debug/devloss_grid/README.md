# devloss_grid — answer-CE of a frozen checkpoint under token-selection schemes (22 CTC rows + cpt80)

`ctc_devloss_grid.py` scores ONE checkpoint on ONE (task, rung): every row is rendered once, then
each scheme is a different *input construction* for the same model, and the teacher-forced CE of the
gold answer tokens is recorded (plus CE on digit tokens, top-1 agreement and KL against `full`, the
compaction ratio T2/T, and kept-document counts). One GPU, one JSON per (task, rung).

```
python debug/devloss_grid/ctc_devloss_grid.py --task ctc_nq --rung 8k --rows 48 \
  --ckpt /data/prasann/ctc_suite/vllm_serving_4b/ctc-4b-nq-full --ckpt-format hf \
  --data-root /data/prasann/ctc_1m_ladders --schemes full,k0,first16,fl64,gold_fl64 --out out/nq_8k.json
python debug/devloss_grid/ctc_devloss_grid.py --task cpt80 --rung 32k --rows 32 --cpt-source <dir> ...
```

Staging (checkpoints, rung files, `manifest.json`) and launching are NOT in this file's scope.

## Rendering: the suite's own token layout, not olmo-eval's plain-text path

Rows are rendered with `olmo_core.data.document_chunk_landmark.segment_prompt_to_chunks(...,
query_position="both", include_answer=True)` — the single source of truth the suite's training
shards (`convert_unified_to_document_landmark.py --emit dense`) AND its native/vLLM eval driver
(`debug/ctc_vllm_validation/general/build_prefills_any.py` → `eval_lc_native_docchunk.build_eval_prefill`)
both call. That means: the same `build_prompt` the olmo-eval branch vendors (`use_alpaca=False`,
`use_titles=False`), wrapped in the Qwen chat template, each document wrapped in
`<|box_start|>`/`<|box_end|>` (`RESERVED_IDS["qwen3_5"]`), the gold answer as the assistant turn,
EOS appended. The loss tokens are the assistant-turn tokens (same mask the shards carry).

**Deviation from the olmo-eval harness, on purpose.** olmo-eval's `CTCSuiteTask.format_request`
uses `spec.build_prompt(example, query_position="both")` = the *alpaca-wrapped*, marker-free text
("near but not bit-identical to the historical grid", its own docstring). The 4B suite checkpoints
were trained and graded on the chat-template + marker layout, and the soft-token compaction needs the
markers, so the driver uses that layout. `smoke_cpu.py` asserts that, after stripping the markers,
the user-turn text is byte-identical to the vendored `spec.build_prompt(..., use_alpaca=False,
use_titles=False)` — i.e. the only differences are the wrapper and the markers.

The roster (`ROSTER`), segmentation config (`SEG_CFG`: chunk_by / cot) and gold conventions
(`GOLD_CONVENTION`) are copied into the driver so it imports no eval module; at startup it asserts
them against `eval_lc_native_docchunk.TASK_CFG` and the vendored `ctc` registry at
`~/projects/olmo-eval/.../ctc_suite/_vendor` (skipped with a log line when not importable).
Note oolong renders with `cot="plan"` (a one-line "Reasoning: …" before `Answer:`) because that is
what TASK_CFG / the shards use; its CE therefore covers plan + answer.

## Schemes (names are fixed)

All gold-blind, keep 0 whole documents, `cent_cmean` slot, **no header stop id** (K counts from the
document's first token — the one header rule that is identical for every task) unless stated.

| name | construction | via |
|---|---|---|
| `full` | uncompacted row, plain causal (reference) | `model.eval()`, no holder |
| `k0` | every document → one slot | `keep_prob=0` |
| `first16` / `first64` | first K body tokens real per doc + slot | `header_extra_tokens=K` |
| `fl16` / `fl64` | first K//2 + last K−K//2 real + slot | `keep_token_rule="first_last"` |
| `idf16` / `idf64` | top-K body tokens by IDF real (original positions) + slot | `keep_token_rule="rule"`, `weights={"idf":1.0}` (all other features 0) |
| `rand33` | seeded random whole docs, keep 1/3, plain `mean` slot (grid scheme B) | `resolve_keep_docs(keep_prob=1/3)` |
| `gold_rand33` | gold docs real + random whole docs to 1/3 of all docs, plain slot (scheme A) | explicit `(1, n_docs)` mask → `PooledDocKeepHolder` |
| `gold_fl64` | gold docs real (whole) + `fl64` on every other doc, `cent_cmean` (scheme G) | mask + `first_last` |

Every construction goes through `Transformer._compact_pooled_soft_tokens` (the training code path,
driven via `model._pooled_soft_tokens` keys and `model._pooled_keep_holder`), and CE is taken at the
answer positions mapped into the compacted sequence with `logits_to_keep` — exactly
`debug/pooled_kv/trained_parity_check.py::run_rung`. `enable_pooled_soft_tokens(detach_soft_kv=True,
keep_prob=0.0)` as there; the checkpoint's own `pooled_projector` is kept (identity for every
dense checkpoint).

**Tables built from the scored rows, not a training shard** (documented difference from the ds64
arms, which built them from the head of the training shard):
* `cent_cmean` stop set = `build_slot_stop_ids(all row ids, top_k=100, extra=(markers, eos,
  landmark, pad), decode=tok)` → top-100 ids + every non-alphanumeric piece + markers.
* IDF = `build_token_idf(all row ids, vocab)` (−log p, add-one). Piece tables from
  `tok.convert_ids_to_tokens(range(vocab))`.

## Gold documents

`gold_docs(spec, example)` → 0-based positions in `documents` order (== marker-span order for
`chunk_by="document"`; the driver warns when span count ≠ document count):

| spec | field / base | note |
|---|---|---|
| retrieval (fiqa, nq, hpqa, msmarco, scifact, obliq, niah), outlier (×3), absence, xabsence | `gold_doc_indices`, 0-based | |
| rerank | `gold_doc_indices` 0-based ∪ every doc with `ce_scores > 0` | the CE-positive set the metric scores |
| contradiction, strmatch, textgroups | nested pairs/triples, 1-based | flattened |
| qdmatch (×3) | `gold_pairs` (query, doc) over the shared 1-based item index | both sides kept |
| oolong, reorder, grouping | **no gold subset** | `gold_*` schemes degrade to their gold-blind twin (`rand33` / `fl64`); rows listed under `gold_degenerate_rows` |

## `cpt80` (continued pretraining, amandab's setup)

Rows = long pretraining documents from `--cpt-source` (or `manifest.json:cpt_source`): a file or
directory of `.jsonl`/`.json(.gz)` (field `text`), `.parquet`, `.txt`, or a saved HF dataset. Keep
documents with ≥ rung tokens, truncate to the rung length L; the first 80% is cut into 512-token
pseudo-documents each wrapped in markers (`--cpt-block`), the last 20% stays real and is the loss
region; EOS appended, no chat template. All schemes apply; no gold (gold schemes degrade).

## Checkpoints

`--ckpt-format distcp` = an olmo-core `model_and_optim` dir (or its parent / `step*` parent).
`--ckpt-format hf` = a Qwen3.5 text export or a vLLM **serving copy** (`model.language_model.*`
keys, `visual.*` tower, VL wrapper config with `text_config`): the driver reads the safetensors
itself and converts with `olmo_core.nn.hf.convert.convert_state_from_hf(model_type="qwen3_5_text")`
(which normalises the prefixes and drops the vision keys), because the env's transformers 4.57 has
no `qwen3_5_text` class for `AutoModelForCausalLM`. Model = `TransformerConfig.qwen3_5_4B(
vocab_size=248320, attn_backend=torch)`, bf16 on GPU. Tokenizer: `--tokenizer` (default
`Qwen/Qwen3.5-0.8B-Base`; a local snapshot path works offline).

## Output JSON

`summary[scheme]` = mean and SE over rows of `ce, ce_digit, top1, kl, compaction, kept_docs,
n_docs`; `per_row[scheme]` the lists; `eval_size`, `row_lens`, `gold_degenerate_rows`, `meta`
(subset/spec/seg_task/span mismatches), `git_commit`, `argv`. Progress is logged at rows 1, 2, 5,
10 then every 10 with ETA and running per-scheme CE@compaction.

## Smoke test

`HF_HUB_OFFLINE=1 python debug/devloss_grid/smoke_cpu.py` — two real nq rows through the real
tokenizer, vocabulary remapped onto a 2-layer random `olmo2_190M` on CPU, every scheme; asserts
compaction ordering (`k0 < first16 < first64 < 1`, K-matched rules identical, gold masks keep
every gold doc) and the prompt-text equality above. Last run: OK (compaction k0 0.049, first16
0.147, first64 0.441, rand33 0.36, gold_rand33 0.383, gold_fl64 0.491 on nq@2k).

No shared library file was modified.

## Launch state (2026-09-21) and caveats

* Launcher: `run_grid_local.sbatch` (one GPU, `ONLY_TASKS`/`RUNGS`/`ROWS`/`SCHEMES`/`ATTN_BACKEND`/`RES_DIR`
  env overrides; resumable; bootstraps checkpoints from `sneetches:/data/prasann/devloss_grid` on any
  other node, refusing nodes with <300 GB free on `/data`). Group jobs: `GROUP_JOBS.tsv`. Results:
  `debug/devloss_grid/results/<task>_<rung>.json` (repo, so the four group jobs on different nodes
  share one skip-existing check); failures in `results/FAILED.tsv`.
* `--attn-backend flash_2` is the default (what the ds64 soft arms trained with): on nq@2k the CE
  matches the torch backend to <=0.005 on every scheme and a row takes ~2.3 s instead of ~23 s.
* fiqa r2k row 29 has a document with EMPTY text; `segment_prompt_to_chunks` renders it without
  markers, which would shift every later gold index by one. The driver DROPS any row whose marker-span
  count differs from its document count (`meta.span_count_mismatch_rows_dropped`); fiqa is the only
  row affected in the smoke logs (1 of 48 at 2k/8k/32k each).
* `idf{K}` on id-answer tasks pools the `Document [N]:` header (it is the lowest-IDF span in every
  document), so the id the answer names is never real -- expect idf ~ k0 there; the tokens it keeps
  are real content words (`check_idf_cpu.py`).
* `cpt80`: the loss region starts at the SECOND tail token -- the first is predicted from the last
  pseudo-document's `<|doc_end|>`, which every compaction pools away.
* Short-document tasks (strmatch: every document <= 64 tokens; oolong lines nearly so) make the K=64
  schemes degenerate -- `first64`/`fl64`/`idf64`/`gold_fl64` keep everything real (compaction ~1.0 /
  0.89). Read the K=16 rows there; the quick pass runs `first16,fl16,idf16` on strmatch/oolong/
  textgroups/grouping at 2k/8k as a follow-up job once the 32k cells are in.

## Learned router schemes (2026-09-23)

`router_[nogold_|noemb_]l<λ>[_samp]` / `router_l0.2_nomark` (`sel="router"`, `keep="none"`, no slots,
markers kept via `keep_markers`) load per-task weights from `debug/learned_router/weights/<task>/`; results in
`results_router/` (now a default `collect_grid.py` root). `gold_only_noslot` = gold docs whole, every other
document dropped outright. `load_model` now calls `freeze_fla_length_autotune()` (speed only, bit-identical):
without it every new compacted length re-autotunes FLA (a 16-row cell took 603 s instead of ~1 min).
See `debug/learned_router/README.md` and `records/learned-token-router.md`.
