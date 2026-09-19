# Hybridish SFT → CTC-suite eval: reference recipe

End-to-end path used to produce the 4:1 vs 7:1 comparison: build an SFT mix from CTC-suite tasks,
finetune a `mainline_ladder` hybrid checkpoint, and score it on the real CTC suite (held-out,
500 examples per rung) via olmo-eval.

Everything below has actually been run. Where a step exists only to work around a trap, the trap is
named — skipping those steps does not fail loudly, it produces numbers that look fine and are wrong.

## Two repos, two branches

| what | repo / branch |
|---|---|
| data build, export, eval tooling (this recipe) | `allenai/OLMo-core` @ `prasann/landmark` |
| the SFT training script | `allenai/OLMo-core` @ `prasann/ctc-sft-hybridish` |
| the CTC suite tasks | `allenai/olmo-eval` @ `prasann/ctc-suite-grader-fixes` |

The training branch is separate because it carries the real Scalable-Softmax implementation
(`Attention.scalable_softmax` / `ssmax_scale`) that these checkpoints are trained with.

Eval data is the **public** HF dataset `PrasannSinghal/ctc-suite-eval` — no token, no weka.

---

## The dataset

One path. This is the exact SFT set both published checkpoints were trained on — **15,183
instances**, 8 CTC tasks, 2k–32k, dolma2 tokenizer, one epoch.

```bash
DATA=/weka/oe-training-default/ai2-llm/checkpoints/prasanns/ctc_hybridish_sft/shards_long32k
```

It contains `token_ids_part_*.npy`, `labels_mask_*.npy`, `metadata.json`, and `src_index.json`
(instance → source-row map; **any grader needs it** — see step 2 of the appendix for why).

Other shard directories exist alongside it on weka from earlier iterations. They are not this
dataset and are not what anything here was trained or scored on — use the path above.

The two finetuned checkpoints sit next to it and are ready to evaluate as-is (SSMax-patched
modeling code and `auto_map` live inside each checkpoint, so `trust_remote_code=True` is the whole
integration):

```bash
CKPTS=/weka/oe-training-default/ai2-llm/checkpoints/prasanns/ctc_hybridish_sft   # sft_4to1_ml_hf, sft_7to1_ml_hf
```

## Quickstart

**Already have the checkpoints and only want the numbers** — skip to step 7:

```bash
bash debug/hybridish_sft/run_olmo_eval_ctc.sh sft_7to1_ml_hf smoke7to1 smoke   # ~3 min, proves the path
bash debug/hybridish_sft/run_olmo_eval_ctc.sh sft_7to1_ml_hf 7to1_nq ctc_nq    # one task, 5 rungs
python debug/hybridish_sft/harvest_sweep.py                                    # the comparison table
```

**Want to retrain** — start at step 4, pointing the trainer at `$DATA`. Building the mix and shards
is the appendix; you only need it to change the task roster, the length band, or the tokenizer.

---

## 4. SFT

From a checkout of `prasann/ctc-sft-hybridish`:

```bash
PYTHONPATH=src python src/scripts/train/hybrid-small-suite/sft_ctc.py \
  launch hyb-7to1-1p4b ai2/jupiter-cirrascale-2 --arm 1.4b_7to1
```

Model geometry is read from the base checkpoint's own `config.json` — never restate it. These are
custom architectures whose depth and attention-override layers differ per arm; a hand-copied
geometry that disagrees with the checkpoint silently trains a different model.

## 5. Export to HF, in the dialect the plugin speaks

```bash
python src/scripts/convert_checkpoint_to_hf.py --checkpoint-input-dir <step> --huggingface-output-dir <hf>
python debug/hybridish_sft/redialect_to_mainline.py --src <hf> --ref <released mainline ckpt> --out <ml_hf>
```

The converter emits `olmo3_5_hybrid`; the plugin and every released hybridish checkpoint use
`mainline_ladder`. Same weights, different key spellings and `model_type`.

## 6. Make the checkpoint self-contained ← **do not skip**

```bash
python debug/hybridish_sft/apply_ssmax_to_plugin.py --src <upstream plugin> --out debug/hybridish_sft/plugin_ssmax
python debug/hybridish_sft/make_self_contained_ckpt.py --ckpt <ml_hf> --plugin debug/hybridish_sft/plugin_ssmax
```

> ⚠ **The stock `mainline_ladder` plugin has no Scalable-Softmax at all.** A checkpoint's
> `ssmax_scale` loads as UNEXPECTED and is silently ignored — the model loads clean and scores, just
> without a trained component. `apply_ssmax_to_plugin.py` adds it (plus the `transformers>=5.13`
> cache shim the plugin lacks, without which any cached forward dies with `AttributeError`).
>
> Unlike olmo-core's version — which *raises* on KV caching — the port takes positions from
> `cache_position`, so decoding is correct. A forward-hook reconstruction that reads the current
> forward's sequence length computes `log(2)` per decode step where the truth is
> `log(prompt_len + step)`: at an 8k prompt that scales the query **13× too small** on every SSMax
> layer, during exactly the tokens being graded.

`make_self_contained_ckpt.py` copies the patched modules into the checkpoint and sets `auto_map`, so
`trust_remote_code=True` becomes the entire integration — for olmo-eval or any other consumer.
`scalable_softmax` is derived from the weights, never a flag, so a non-SSMax model cannot be
mislabelled.

Verify (with the plugin deliberately *not* importable):
```bash
python debug/hybridish_sft/test_plugin_ssmax.py            # formula + cached-decode equivalence
python debug/hybridish_sft/verify_trust_remote_code.py --ckpt <ml_hf>
```

## 7. Evaluate

The two published checkpoints are already on weka — go straight to the run:

```bash
bash debug/hybridish_sft/run_olmo_eval_ctc.sh sft_7to1_ml_hf 7to1_nq ctc_nq
```

Only if you trained your OWN checkpoint, stage it first (Beaker cannot read your disks):

```bash
bash debug/hybridish_sft/stage_ckpts_to_s3.sh    # your machine -> S3
bash debug/hybridish_sft/sync_s3_to_weka.sh      # S3 -> weka, via gantry
```

Third argument: `smoke` (8 instances, proves the path), `full` (all 39 task × rung runs), or a task
name to shard one job per task — the 8-task ladder is hours on one GPU and sharding makes it
concurrent.

`CHUNK=8` for long-context tasks. `absence` OOMs at the default batch of 64: a single 20 GiB
allocation plus ~29 GiB lost to allocator fragmentation.

## 8. Harvest

```bash
python debug/hybridish_sft/harvest_sweep.py
```
Reads `sweep_ids.txt`, fetches **only** `metrics.json` from each result dataset (the full datasets
are ~130 MB of predictions apiece; sixteen of them exhausted our disk quota, after which the
harvester's own write left a 0-byte file while reporting success), and prints the arm comparison
per task × rung. Running, failed and missing jobs are reported separately and a partial table says
so — a table quietly covering 13 of 16 arms reads as complete.

---

## Reading the output

* **Check `parse_rate` before any score.** A low score at low parse rate is a decoding/format
  failure, not a capability one. Sub-1B hybridish checkpoints frequently cannot emit an `[id]` at
  all.
* **Quote `eval_size` next to any sub-500 cell.** The suite is 500/rung by design; `absence` tops
  out at r16k with fewer.
* **These arms are not parameter-matched.** 4:1 (1.4B) and 7:1 (2.1B) have *identical* attention:
  4 full-attention layers, 199,251,008 attention params each. The whole difference is 12 extra GDN
  layers (+62% non-embedding params). A 7:1 win is therefore not attributable to attention capacity,
  and calling it an attention-ratio result is wrong. A clean ratio ablation holds non-embedding
  params fixed and varies where the attention layers sit.

## In-house grader (`grade_ctc_mix.py`)

Also in this directory, for quick iteration against the *training* shards. It is **not** a
substitute for the suite: it grades train data, uses a continuous length mix rather than rungs, and
its SSMax is a reconstruction. Use it to check a pipeline runs; use olmo-eval for numbers. It
requires `src_index.json` and runs a gold self-check (score each shard's own gold against its mapped
example — must be 1.0, needs no generation) before grading.

---

# Appendix: rebuilding the SFT data

Only needed to change the roster, the length band, or the tokenizer. Otherwise use
`$DATA` above.

## 1. Build the SFT mix

`ctc-data build` writes `<root>/<task>/train.jsonl` per task. Combine them into one tagged,
shuffled mix:

```bash
python src/scripts/data/hybridish/build_ctc_sft_mix.py \
  --root /path/to/ctc_builds --band 2k-32k --per-task 2000 \
  --tasks msmarco qdmatch_nq contradiction outlier oolong xabsence absence strmatch \
  --out /path/to/mix_long.jsonl
```

The roster is enforced, not advisory: held-out ladders (`fiqa`, `scifact`, `outlier_review`,
`contra_fever`, `redundancy`) are refused, and a grading spec may be drawn from only one source
(retrieval → msmarco, qdmatch → qdmatch_nq) so the rest stay clean generalisation probes.

## 2. Tokenize to shards

```bash
python src/scripts/data/hybridish/convert_ctc_to_sft_completion.py \
  --input-jsonl /path/to/mix_long.jsonl \
  --out-dir /path/to/shards_long32k \
  --tokenizer allenai/dolma2-tokenizer --max-seq-len 32768 --verify
```

Emits `token_ids_part_*.npy`, `labels_mask_*.npy`, `metadata.json`, and **`src_index.json`**.

> ⚠ **`src_index.json` is load-bearing.** Rows over `--max-seq-len` are dropped, so the shards are a
> *subsequence* of the source JSONL — not a copy. Any grader that pairs shard instance `i` with
> source row `i` scores each generation against a **different example's gold** from the first drop
> onward. On our 32k mix that was 817 of 16,000 rows (5.1%), first drop at row 11, and it produced a
> complete 8-task × 2-arm results table that was pure noise. For shards built before the sidecar
> existed, reconstruct it with `debug/hybridish_sft/build_src_index.py` (verified exact by
> element-wise token-length match).

Keep `--verify`: it scores each gold target with the evaluator's own parser and refuses to write
shards whose targets do not parse.

## 3. Stage shards where the trainer can see them

```bash
bash src/scripts/data/hybridish/stage_shards_to_weka.sh /path/to/shards_long32k shards_long32k
```
S3 first, then a gantry job syncs S3 → weka. Pushing to S3 alone does nothing for a Beaker job; its
absence shows up as a MISSING path at step 0.

