# Landmark tokens as the soft-token slot — eval-only probe (2026-09-14)

**Verdict: NO-GO.** Both stopping conditions from the brief fire at once.

1. **The PASS-2 CONTROL is already far from FULL.** A compressive-landmark-trained checkpoint
   cannot run under plain causal attention: exact match 0.583 -> 0.000 at 2k on the content-only
   sequence, answer CE 0.271 -> 2.188 at 32k on the landmark sequence. Every compaction arm is
   therefore measuring the damage from switching attention mechanism, not the quality of the slot.
2. **The landmark slot is not better than the mean slot.** Paired over the same rows, at matched
   keep, `lmkv - meanemb` in answer CE is `+0.0000 +- 0.0071` (2k, k=0), `+0.0055 +- 0.0123`
   (2k, k=1/8), `-0.0119 +- 0.0074` (32k, k=0), `-0.0293 +- 0.0167` (32k, k=1/8) -- |t| <= 1.8
   everywhere. Against a FULL-MEAN gap of 0.81-1.03 that is **-0.6% to +2.9% of the gap closed**.
   The GO bar was ">= 50% on >= 2 rungs".

The block-local pass-1 variant -- the only one that would save training FLOPs -- is *worse* than the
already-uninformative full pass 1 (`lmkv-blocklocal - lmkv` = `+0.178 +- 0.012`, t=14.7 at 32k k=0),
so the secondary GO condition fails too.

## What was run

`debug/pooled_kv/landmark_probe/landmark_slot_probe.py` (committed on `prasann/landmark`,
5d4628e), one 1-GPU Beaker job per checkpoint, reading the distcp checkpoint straight off weka.

Two passes, both in **landmark token space**, so no position ever moves:

* **PASS 1** — `FastCompressiveLandmarkAttention` (`src/olmo_core/nn/attention/landmark_compressive.py`,
  `_attn_core` at 877-892, `is_mem` stride at 890) runs the landmark-structured sequence. Content is
  padded to a multiple of `mem_freq` with the tokenizer pad id and a landmark token is inserted after
  every `mem_freq` content tokens — `_insert_landmark_tokens`
  (`src/olmo_core/generate/generation_module/transformer/generation_module.py:63`), the exact routine
  the native landmark eval uses at prefill, and the same layout
  `LandmarkPackingInstanceSource` produces at training time. Every attention layer's **post-RoPE**
  K and V are stashed (a wrapper around `Attention._prepare_qkv`, as in
  `debug/pooled_kv/oracle_meankv_probe.py`) and reduced per block to (a) the K/V at the landmark
  position and (b) the block's K/V mean.
* **PASS 2** — a plain-`Attention` twin built from the SAME state dict runs the COMPACTED sequence:
  kept blocks verbatim, each pooled block replaced by ONE slot at the landmark's ORIGINAL position,
  every surviving token keeping its original position id (`position_ids`, threaded at
  `model.py:1139` on HEAD). The slot's K/V is overridden per layer via `soft_kv_override_layers`
  with `pos=0` (RoPE at 0 is identity; the captured keys are already rotated at their absolute
  position).

Two model instances are required because `FastLandmarkAttention.forward`
(`landmark_fast.py:822`) accepts neither `position_ids` nor `soft_kv_override`.

### Arms

| arm | what |
|---|---|
| `full` | PASS 1 — compressive landmark attention (reference) |
| `causal-lmseq` | plain causal over the SAME landmark sequence, no compaction |
| `causal-dense` | plain causal over the content-only sequence (no landmarks, no pads) |
| `meanemb k=` | compaction; slot input embedding = mean of the block's content embeddings (the current scheme) |
| `lmkv k=` | compaction; slot K/V = the block's LANDMARK K/V from pass 1 |
| `blockmeankv k=` | compaction; slot K/V = the MEAN of the block's K/V from pass 1 (control: is the landmark special?) |
| `lmkv-blocklocal k=` | `lmkv`, but pass 1 ran BLOCK-LOCAL (each block attends only to itself) — the only variant that would save training FLOPs |

`k` is the fraction of poolable blocks that stay real; all slot arms share the same keep set per row.

### Setup

* **Checkpoint:** `/weka/oe-training-default/ai2-llm/checkpoints/q4b-comp-block128-5task-dolci25-nocpt/step8550`
  — Qwen3-4B, `fast_compressive_landmark=True`, `mem_freq=127` (block 128),
  `nonselected_landmark_mass=0.1`, 5-task(75%)+Dolci-Instruct-SFT(25%) no-CPT SFT, ~700M tokens.
  Reported contradiction f1 0.905 / 0.852 / 0.777 / 0.627 at 2k / 8k / 16k / 32k
  (results-hub rows `78d69a61c1`…`3779fd683f`).
  **Path correction:** the brief gave an `amandab/` prefix; the checkpoint has **no owner prefix**
  (verified on weka, job `01M2GW3M2HF79A531DT7JKQEMY`). `amandab/q4b-comp-block128-…` does not
  exist; `prasanns/q4b-comp-block128-…` holds only an `eval_refix128k` directory. Same for block64.
* **Data:** contradiction eval rows from the v3 eval bundle
  (`_eval_bundle_eval500_v3/contra/contradiction_eval_pubmed_realistic_n{100,765}_k3.jsonl`),
  tokenized in-job with the ladder40k recipe — Qwen3 chat template, `query_position="both"`,
  `cot_mode="none"` (`src/scripts/data/convert_longctx_tasks_to_sft.py`, whose
  `build_contradiction_instance` / `render_chat` the probe imports directly).
  **Not** `convert_unified_to_document_landmark.py`: that emitter wraps documents in
  `<|box_start|>`/`<|box_end|>`, which this checkpoint's training data never contained. The
  block-128 SFT read plain SFT shards (`cptmix_data_ladder40k/<task>`) through
  `LandmarkPackingInstanceSourceConfig`, i.e. periodic landmarks and no document markers at all.
* **eval_size = 48 rows per rung.** ⚠ Well under the 500-example floor; every number below carries
  its per-row standard error and should be read at that resolution.

## Documents per landmark block (the landmark ≠ document trap)

| rung | claim len (tokens) mean / median / p90 | docs per 127-token block | blocks with ≥2 docs | blocks/row (poolable) |
|---|---|---|---|---|
| 2k | 39.8 / 35 / 65 | 3.92 | 96.3% | 34.3 (31.4) |
| 32k | 40.8 / 36 / 66 | 3.98 | 99.0% | 253.9 (250.9) |

A landmark block is **not** a document: at the 2k rung almost every block straddles ~4 claims. Any
"landmark = document summary" intuition is wrong for this geometry — the landmark summarises a
127-token window that cuts claims apart.

## Results

_Answer-position metrics, teacher-forced. `top1` = agreement with `full`'s argmax; `KL` =
KL(full ‖ arm); `exact` = the greedy teacher-forced answer matching the gold answer exactly;
`compact` = compacted length / landmark-sequence length._

### 2k rung (eval_size = 48; binomial SE on `exact` ≈ 0.071)

| arm | answer CE | ±SE | top1=full | KL | exact | compact |
|---|---|---|---|---|---|---|
| full | **0.0790** | 0.0196 | 1.0000 | 0.000 | **0.583** | 1.000 |
| causal-lmseq | 0.1862 | 0.0242 | 0.9701 | 0.149 | 0.271 | 1.000 |
| causal-dense | 0.8076 | 0.0474 | 0.8079 | 0.785 | **0.000** | 0.977 |
| meanemb k=0 | 0.8909 | 0.0123 | 0.6828 | 0.880 | 0.000 | 0.093 |
| lmkv k=0 | 0.8909 | 0.0106 | 0.6801 | 0.878 | 0.000 | 0.093 |
| blockmeankv k=0 | 0.8942 | 0.0128 | 0.6808 | 0.885 | 0.000 | 0.093 |
| lmkv-blocklocal k=0 | 0.9003 | 0.0121 | 0.6718 | 0.889 | 0.000 | 0.093 |
| meanemb k=0.125 | 0.9731 | 0.0202 | 0.6906 | 0.968 | 0.000 | 0.208 |
| lmkv k=0.125 | 0.9786 | 0.0195 | 0.6907 | 0.977 | 0.000 | 0.208 |
| blockmeankv k=0.125 | 0.9779 | 0.0198 | 0.6907 | 0.973 | 0.000 | 0.208 |
| lmkv-blocklocal k=0.125 | 0.9879 | 0.0221 | 0.6831 | 0.984 | 0.000 | 0.208 |

### 32k rung (eval_size = 48; binomial SE on `exact` ≈ 0.051)

| arm | answer CE | ±SE | top1=full | KL | exact | compact |
|---|---|---|---|---|---|---|
| full | **0.2711** | 0.0293 | 1.000 | 0.000 | **0.146** | 1.000 |
| causal-lmseq | 2.1880 | 0.0369 | 0.612 | 2.099 | 0.000 | 1.000 |
| causal-dense | 2.7115 | 0.0239 | 0.464 | 2.580 | 0.000 | 0.990 |
| meanemb k=0 | 1.1407 | 0.0160 | 0.595 | 1.021 | 0.000 | 0.019 |
| **lmkv k=0** | **1.1289** | 0.0154 | 0.601 | 1.006 | 0.000 | 0.019 |
| blockmeankv k=0 | 1.2116 | 0.0151 | 0.583 | 1.095 | 0.000 | 0.019 |
| lmkv-blocklocal k=0 | 1.3066 | 0.0136 | 0.565 | 1.185 | 0.000 | 0.019 |
| meanemb k=0.125 | 1.2974 | 0.0246 | 0.591 | 1.174 | 0.000 | 0.142 |
| lmkv k=0.125 | 1.2681 | 0.0201 | 0.601 | 1.143 | 0.000 | 0.142 |
| blockmeankv k=0.125 | 1.2813 | 0.0275 | 0.595 | 1.168 | 0.000 | 0.142 |
| lmkv-blocklocal k=0.125 | 1.3601 | 0.0223 | 0.565 | 1.237 | 0.000 | 0.142 |

### Paired comparisons (same rows, so the SE is on the per-row difference)

| rung | keep | FULL CE | MEAN CE | gap | lmkv − meanemb | t | gap closed | blockmeankv − meanemb | lmkv-blocklocal − lmkv |
|---|---|---|---|---|---|---|---|---|---|
| 2k | 0 | 0.0790 | 0.8909 | 0.812 | +0.0000 ± 0.0071 | +0.00 | −0.0% | +0.0033 ± 0.0049 | +0.0094 ± 0.0079 |
| 2k | 1/8 | 0.0790 | 0.9731 | 0.894 | +0.0055 ± 0.0123 | +0.45 | −0.6% | +0.0048 ± 0.0075 | +0.0093 ± 0.0166 |
| 32k | 0 | 0.2711 | 1.1407 | 0.870 | −0.0119 ± 0.0074 | −1.61 | +1.4% | +0.0709 ± 0.0098 | **+0.1777 ± 0.0121** |
| 32k | 1/8 | 0.2711 | 1.2974 | 1.026 | −0.0293 ± 0.0167 | −1.76 | +2.9% | −0.0161 ± 0.0104 | +0.0919 ± 0.0193 |

(Negative = the landmark slot is better. It is better by at most 0.03 nats, never at |t| > 1.8, on a
gap of ~0.9 nats.)

## Reading

1. **`full` validates the setup.** CE 0.079 / exact 0.583 at 2k and CE 0.271 / exact 0.146 at 32k
   are what a model with contradiction f1 0.905 / 0.627 should give under teacher forcing (exact
   match on the whole JSON pair list is strictly harder than pair-F1, and it falls with rung in the
   same way the f1 does). The in-job tokenization, the landmark layout, the block/alpha config and
   the checkpoint load are all right.
2. **The PASS-2 CONTROL is the blocker.** Run the *same tokens* under plain causal attention and the
   model degrades immediately — 2k: exact 0.583 → 0.271 on the landmark sequence, → 0.000 on the
   content-only sequence; 32k: CE 0.271 → 2.188 → 2.712. `top1=full` never exceeds 0.69 for any
   compaction arm, on a reference whose own answer is right 58% (2k) / 15% (32k) of the time.
3. **Landmark ≈ mean, on both rungs and both keeps.** See the paired table: the landmark K/V slot
   never separates from the mean input-embedding slot by more than 0.03 nats, at |t| ≤ 1.8. The
   landmark token is not a privileged summary of its block — consistent with what the compressive
   kernel actually does (`landmark_compressive.py` module docstring): the landmark sets the block's
   *gate*, while the block's value contribution is spread over all 128 tokens by the within-block
   softmax, so the landmark's own value carries roughly its 1/128 share.
4. **Block-mean K/V is a clean control and it also says "nothing special".** At 32k, k=0 the
   landmark beats the block K/V mean by 0.083 nats (t≈7), which is the one place the landmark shows
   any structure at all — but it is still level with the far cheaper mean *embedding*, so the
   structure buys nothing.
5. **Block-local pass 1 costs real accuracy** (+0.178 nats at 32k, k=0, t=14.7). Since the only
   reason to want a block-local pass 1 is training FLOPs, and the arm it would feed is already at
   the mean-slot baseline, that question never becomes live.
6. Worth remembering: at 32k the **compacted** arms (CE ≈ 1.13–1.36) beat the **uncompacted**
   plain-causal arms (CE ≈ 2.19–2.71). Plain causal attention over 32k is what breaks this model;
   over the ~630-token compacted row (compaction 0.019, i.e. **52×**) it is merely mediocre. Note
   also that keeping the landmark tokens in the sequence helps plain causal a lot at 2k
   (0.186 vs 0.808) and somewhat at 32k (2.19 vs 2.71).

## What would make the question testable

The probe is sound; the checkpoint is the wrong substrate. Two routes:

* **Run pass 2 under compressive-landmark attention too.** Not possible with the fused kernel as it
  stands: it derives `is_mem` from a fixed stride (`landmark_compressive.py:890`), so a compacted
  row — where blocks have collapsed to single slots — no longer has landmarks at period-128
  positions. It needs an `is_mem`/variable-block variant of the kernel.
* **Probe a model that is fluent in plain causal attention and still emits landmarks**, e.g. a
  mask-mixing SFT that alternates plain-causal and landmark forwards. Then the pass-2 control would
  be near `full` by construction and the slot comparison would mean something.

Either is a training-side change, so the brief's "eval-only test before training" gate is answered:
**do not take this to training as specified.**

## Artifacts

* Probe: `debug/pooled_kv/landmark_probe/landmark_slot_probe.py`
* Ledger: `debug/pooled_kv/landmark_probe/LAUNCH_LEDGER.tsv`
* Result JSON: `/weka/oe-training-default/ai2-llm/checkpoints/prasanns/_eval_results/landmark_slot_probe/q4b-comp-block128_2k-32k.json`
  (also the job's `/results/landmark-slot-probe.json` Beaker dataset)
* Jobs: `01M2GW3M2HF79A531DT7JKQEMY` (weka path check),
  `01M2GWTZQMS0KYHRY01DG4357G` (failed — RoPE `position_ids`/`cu_doc_lens` conflict),
  `01M2GXHJ7FCQT48BYE8ZF9C5M6` (the run above)
* `q4b-comp-block64-…/step8550` (block 64, mem_freq 63) exists on weka and was **not** run: the
  kill-switch arm fired on block128, and nothing about block 64 changes the pass-2 control.
