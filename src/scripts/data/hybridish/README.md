# CTC SFT mixes for base models (hybridish)

Two versions of one SFT mix, built from the CTC suite, for **base** checkpoints that score near the
floor on the suite out of the box:

| version | rungs | what it is for |
|---|---|---|
| **short** | `2k,4k` | does SFT move the suite at all, at a length the model is comfortable at |
| **long** | `2k,4k,8k,16k,32k` | the same tasks across the ladder the suite is actually measured on |

Built for Yashas' hybridish models (`mainline_ladder`, 4:1 linear:full attention, dolma2 tokenizer,
`max_position_embeddings` 65536), but nothing here is specific to them — point `--tokenizer` at any
HF tokenizer and the shards match that model.

## Why these scripts and not `convert_unified_to_sft.py`

That converter targets instruction-tuned Qwen: it wraps every instance in the Qwen3 chat template
and hardcodes Qwen's EOS (`151643`) and landmark id. Base models have a different tokenizer and
generally no chat template, and — the part that actually matters — **the ctc-suite evaluator prompts
a base model with a bare alpaca completion string**, so a chat-wrapped instance trains a format the
evaluator never emits.

`corpus_reasoning_prompts.build_prompt(..., use_alpaca=True)` was verified byte-identical to the
evaluator's `spec.build_prompt` (8841/8841 chars on a qdmatch_nq row), so training input and scoring
input are the same string by construction. Every tokenizer-specific id is read from `--tokenizer`.

## The roster, and why it is restricted

`build_ctc_sft_mix.py` enforces two source restrictions, because they are what the mix *measures*:

* **retrieval → MS MARCO only.** `nq`, `hotpotqa`, `fiqa` and `scifact` are all graded by the same
  `retrieval` spec. Training on several makes every retrieval row in-distribution and leaves nothing
  to generalise to.
* **qdmatch → NQ only**, for the same reason against `qdmatch_hpqa`.

Held-out ladders (`fiqa`, `scifact`, `outlier_review`, `contra_fever`, `redundancy`) are refused
outright rather than warned about — by the time a warning is read, the checkpoint is trained.

⚠ **`msmarco` is not yet a ported retrieval ladder.** MS MARCO currently exists in `ctc` only as
`rerank`. Adding it needs a *measured* ladder calibration, not a copy of rerank's — rerank's own
table is flagged "widest uncertainty band here, re-measure before quoting". Until that lands, a mix
requesting `msmarco` fails loudly.

## Build

Three steps. The first two need no GPU, no index and no LLM (`--pool auto` fetches the seed pool).

```bash
CTC=/path/to/ctc/src          # the ctc package
ROOT=/data/ctc_hybridish

# 1. per-task data, banded by rung
for T in qdmatch_nq nq; do
  PYTHONPATH=$CTC python -m ctc.data.cli build --task $T --split both \
      --rungs 2k,4k --train 8000 --eval-size 500 --pool auto --out $ROOT/short
done

# 2. one tagged, budgeted, shuffled mix
python src/scripts/data/hybridish/build_ctc_sft_mix.py \
    --root $ROOT/short --tasks qdmatch_nq nq --band 2k-4k \
    --out $ROOT/mix_short.jsonl

# 3. tokenize to olmo-core SFT shards
PYTHONPATH=src:$CTC python src/scripts/data/hybridish/convert_ctc_to_sft_completion.py \
    --input-jsonl $ROOT/mix_short.jsonl \
    --tokenizer allenai/dolma2-tokenizer \
    --max-seq-len 6144 --verify \
    --out-dir $ROOT/shards_short
```

Swap `--rungs 2k,4k,8k,16k,32k`, `--band 2k-32k` and `--max-seq-len 40960` for the long version.

## `--verify` is not optional

It feeds every gold target back through the **evaluator's own parser and scorer** and aborts if any
scores below 1.0. A target the grader cannot parse trains the model to write unparseable answers,
and the result reads as a capability failure rather than a data bug — which is exactly how four
rows of the suite came to report near-zero for a model that was answering correctly (see
`debug/ctc_suite_tractability/REPORT.md`).

## Measured output (qdmatch_nq, 2k–4k, dolma2)

```
num_instances        8000 / 8000     skipped: 0 (build, length, bad — all zero)
num_tokens           24,024,314      num_loss_tokens 144,000
token_len            min 1963  p50 3005  p90 3971  max 4209
verified             500             verify_mean_score 1.0
```

⚠ **18 loss tokens per example.** The answer is a short id list, so the per-step CE is computed over
a very thin signal and is too noisy to resolve small differences — it can rule out a gross failure
to fit and little else. Judge these runs on the graded suite metric, not the loss curve. (The same
trap already produced a fake 2–4x result on oolong; see `records/` and the wave-2 notes.)

## Length distribution (dolma2, 2k-32k mix)

Five of seven tasks fit under 32768; `absence` and `xabsence` overshoot their own 32k rung label
(max 52024 / 42009). At `SEQUENCE_LENGTH=32768` that drops 5% of instances but **15.9% of tokens**,
concentrated in the longest examples. Shards are tokenized at `--max-seq-len 40960` so the training
seq-len choice stays open. Check `metadata.json` -> `token_len` before picking one.

## Rung labels are not token counts

`ctc.data.ladders` calibrates documents-per-rung per task, several entries tokenizer-measured, but
some are fits and some extrapolate. The converter therefore bands on **measured** length
(`--min-tokens` / `--max-seq-len`) and `metadata.json` records the realized distribution. Always
read `token_len` before quoting a context length for these shards.
