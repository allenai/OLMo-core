# loss_bench — train/val loss benchmark across 8 SFT checkpoints

Computes mean cross-entropy loss (label/completion tokens only) for the 8 checkpoints behind 4
results-hub comparison pairs (sparse-landmark vs fast-landmark; dense vs fast-landmark/compressive;
summary-token causal/decay/p50; doc-chunk vs dense), on:

- **train loss** — each model's own actual SFT training-data mix (see `models.py` `DATA_GROUPS`),
  sampled once and shared by every model that trained on the same mix.
- **val loss** — the shared v3 eval bundle (contradiction/nq/outlier/rerank/oolong), sampled once
  and shared by every model.

Both are broken down by context-length bucket (2k/4k/8k/16k/32k/64k/128k/256k for train; the
ladder's own rung labels for val), capped per model at its own trained context window (a policy
decision, not a bug — see `models.py`'s module docstring for the pair that this makes asymmetric).

## Design decisions (from the clarifying-question round that produced this)

1. **Context range: cap each model at its own trained window.** Short-window models (32k) simply
   have fewer bucket rows than the 256k models; nothing is extrapolated past training.
2. **Train data: each model's own actual training mix**, not one shared mix. The two "landmark vs
   full attn" checkpoints therefore do NOT train on the same data — flagged loudly in `models.py`.
3. **Loss scope: label/completion tokens only**, matching this repo's SFT train-CE convention
   (prompt tokens are masked out), not full-sequence LM loss.

## Pipeline

```
build_train_manifest.py  --\
                             >--  compute_loss.py --model-key <one of 8>  -->  results/<key>.json
build_val_manifest.py    --/
```

Stage 1 (`build_train_manifest.py` + `build_val_manifest.py`) runs ONCE, CPU only, and writes to
`models.WORK_DIR` on weka:
- `train_manifests/<data_group>.npz` + `.index.json` — 50 real documents per in-range bucket,
  materialized (token ids + label mask) straight from the same raw per-task shards the training
  launchers mix from. 5 distinct data groups, not 8 — several checkpoints share a group verbatim.
- `val_manifest.json` — 50 sampled example INDICES per (task, rung) into the v3 bundle's JSONL
  files (not materialized: each model tokenizes its own copy, since tokenizer and query_position
  differ by model).

Stage 2 (`compute_loss.py --model-key <key>`) runs once per checkpoint, on GPU, and writes
`results/<key>.json` (per-bucket / per-(task,rung) mean CE, token counts, example counts, and
per-example detail).

## Known caveats / open risks

- **`q4b-dense-5task-32k-nocpt-fixdata` step number**: `step2000` does not exist on weka (verified
  by gantry job `01M1D7VQREFVN437R4Q1PTDM9Y`, 2026-08-31) — only `step10700` does, and that's what
  `models.py` points at. The doc-chunk comparison is therefore NOT step-matched (docchunk=step2000,
  dense=step10700).
- **`dense_fixdata`'s exact training launcher is unconfirmed.** The data recipe (Qwen3,
  `single_task_ladders_v2` + p10 NQ, no Dolci) is shared verbatim by every
  `dense-5task-32k-nocpt*` Qwen3 launcher in `sft_5task/`, so it's high-confidence, but the literal
  script that produced this specific run name could not be located.
- **Train-doc sampling does not replay the training mixture's ratios or packing/windowing** — it
  pools raw documents across all tasks in a group uniformly (see `build_train_manifest.py`'s
  docstring). Per-task provenance is kept in the manifest for transparency but bucket population is
  not reweighted to match e.g. contra's 2x sampling weight.
- **No CP for the three 256k models.** `compute_loss.py` runs a single-GPU, no-CP forward, while
  those checkpoints trained with Ulysses CP across multiple GPUs. Existing generation evals already
  do a full un-cached 256k forward on one 80GB GPU (see `records/v3-eval-howto.md`), which is
  precedent this fits, but it's not proven for this exact code path. Will fail loudly (CUDA OOM) if
  it doesn't, not silently misreport.
- **`--query-position` / mask-mode assumptions**: every model here is assumed trained with
  `query_position="both"` (none of these 8 runs postdate the 2026-08-11 qafter switch or use the
  `_qafter` data root) and summary-token models are scored with the causal mask-serving arm (the
  project default per `records/summary-eval-mask-training-gated.md`-adjacent fix, commit
  `6e3a4e309`). Both are set explicitly in `models.py` — change there if wrong for a given run.

## Running it

From the `olmo-core` conda env, branch pushed (see `beaker.md`'s golden rules).

**Stage 1** (one CPU gantry job covers both scripts — cheap, no GPU):

```bash
gantry run --name loss-bench-manifests -w ai2/flex2 -b ai2/oe-other \
  --cluster ai2/neptune --cluster ai2/ceres --cluster ai2/saturn --cluster ai2/jupiter \
  --gpus 0 --priority urgent \
  --beaker-image tylerr/olmo-core-tch291cu128-2025-11-25 \
  --weka oe-training-default:/weka/oe-training-default \
  --install true --allow-dirty --timeout 0 --yes -- bash -c '
    cd src/scripts/train/memexpress/loss_bench
    PYTHONPATH=/olmo-core/src python build_train_manifest.py
    PYTHONPATH=/olmo-core/src python build_val_manifest.py
  '
```

(`--install true` because this needs `olmo_core` importable, unlike the plain-copy one-off template
in `beaker.md`.)

**Stage 2** (one GPU job per model — `ai2/jupiter` for the 256k models, which need an 80GB H100):

```bash
for KEY in sparselm_32k fastlm_32k dense_xlong5_256k fastlm_33344_datamatch \
           summtok_causal summtok_decay summtok_p50 docchunk_bs128 dense_fixdata; do
  gantry run --name "loss-bench-$KEY" -w ai2/flex2 -b ai2/oe-other \
    --cluster ai2/jupiter --gpus 1 --priority urgent \
    --beaker-image tylerr/olmo-core-tch291cu128-2025-11-25 \
    --weka oe-training-default:/weka/oe-training-default \
    --install true --allow-dirty --timeout 0 --yes -- bash -c "
      cd src/scripts/train/memexpress/loss_bench
      PYTHONPATH=/olmo-core/src:/olmo-core/src/scripts python compute_loss.py --model-key $KEY
    "
done
```

Pull results from `models.RESULTS_DIR` (`.../loss_bench_2026-08-31/results/<key>.json`) once each
job finishes — `beaker experiment logs` / `beaker experiment get --format json` per `beaker.md`'s
monitoring section.
