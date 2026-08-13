# `sft_summtoken` — SummTokenSFT

Per-document **summary tokens** with a causal/summary-only **mask mixture**.

Each context document in an SFT example is followed by a run of `N` (default 5) `<|summ|>` tokens.
On a **masked** example a document reads only itself plus the summary runs of strictly earlier
documents, and the trailing query/answer reads the summary runs but **no raw document content**. On a
**causal** example the mask is plain causal. Which examples are which is decided per forward by the
mask-mixture schedule — that is the only axis the five arms vary.

| arm | mixture | what it is |
|---|---|---|
| `summ-only` | (none) | Floor: every example masked |
| `summ-p25` | `standard_mix_prob=0.25` | a static 25% causal fraction |
| **`summ-p50`** | `standard_mix_prob=0.5` | **50% masked / 50% causal throughout** |
| `summ-step50` | `step`, `0.0 → 1.0` at 50% | a hard phase switch |
| `summ-anneal` | `linear`, `0.0 → 0.5` | a partially rising causal fraction |
| **`summ-decay`** | `linear`, `0.0 → 1.0` | **100% masked → 0% masked; ends fully causal** |
| `summ-causal` | `standard_mix_prob=1.0` | **The control** |

`p` throughout is P(**causal**), so "100% mask mixing decaying to 0%" is `mix_start_p=0.0 →
mix_end_p=1.0`. At `summ-p50` the two readings coincide, so there is no direction to get wrong there.

`summ-causal` is the control, **not** the existing dense run: it holds data, summary tokens, layout
(packed or not) and base fixed and varies only the mask. `records/POSSIBLE_BUG_SFT_DATA.md` (open) records that dense
and landmark arms have historically differed in both mixture weights *and* packer epochs, which is
why every arm here comes off one shared builder.

## Two standing caveats

**Only 8 of 32 layers carry the mask.** Qwen3.5-4B is a hybrid
(`block_pattern=["gdn","gdn","gdn","attn"]`); the 24 GatedDeltaNet layers ignore the roles and are an
unrestricted cross-document channel. This is deliberate — the intervention is on attention only — but
it bounds the claim. These runs support *"the attention layers communicate across context documents
only through summary tokens"*, **not** *"documents communicate only through summary tokens"*. The
realized split is logged at build time and saved into `config.json`; check it rather than assuming.

**Train only from a summary-repaired base.** `<|summ|>` (248210 for Qwen3.5) is an untrained row in
the embedding matrix's padded region. Untrained rows are bit-identical *and* out of distribution in
norm, and RMSNorm amplifies a low-norm row into a full-strength meaningless vector at every
occurrence — which flatlines training at CE ≈ 0.79 for **every** mask including plain causal. That
reads as "the mask is too restrictive", so it manufactures a clean false null. See
`records/document-chunked-marker-embeddings.md` and `records/n100-chunked-marker-position-bug.md`.

## Pipeline

```bash
# 1. Build the shards by inserting <|summ|> runs into the existing Qwen3.5 box-marker shards.
#    Documents stay byte-identical to the doc-chunked arms, so the families remain comparable.
#    (From raw JSONL instead: convert_unified_to_document_landmark.py --emit summary.)
#    --num-summary-tokens MUST equal the model's n_summary_tokens: roles are derived by counting
#    summary RUNS, so a mismatch silently renumbers every document.
src/scripts/data/build_summary_token_shards_gantry.sh

# 2. Audit the base (runs anywhere -- no model construction, no triton).
#    Expect cos(summary, pad) ~ 1.0 on an unrepaired base: that IS the untrained-row signature.
python src/scripts/data/fix_marker_embeddings.py --audit-only --family qwen3_5 \
    --marker-set doc_start,doc_end,summary,pad \
    --base .../q35-4b-dense-256k-fix/step2385/model_and_optim --audit-json audit_before.json

# 3. Repair. Building a Qwen3.5 model needs triton (GDN), so this is a gantry job, not a laptop
#    script. It audits before, repairs, then RE-AUDITS THE WRITTEN COPY and fails if that does not
#    pass both gates. Gate every launch on it.
src/scripts/data/fix_marker_embeddings_gantry.sh

# 4. Measure the realized mixture. dry_run does NOT build the dataset, and the curriculum arms
#    refuse to launch without the instance count rather than guessing it.
S=src/scripts/train/memexpress/sft_summtoken/Qwen3.5-4B-summ-p50-5task-SFT.py
PYTHONPATH=src python $S launch_prep q35-4b-summ-prep ai2/jupiter-cirrascale-2
export SUMMTOK_N_INSTANCES=<the "MixingInstanceSource: N instances" line>
# Also read the realized instance LENGTHS from that job: at 262,144 per window, short rungs of the
# 2k->256k ladder are almost entirely padding. Check the waste before committing GPU time.

# 5. Launch.
for arm in summ-p50 summ-decay; do
  PYTHONPATH=src python src/scripts/train/memexpress/sft_summtoken/Qwen3.5-4B-$arm-5task-SFT.py \
      launch q35-4b-$arm-5task ai2/jupiter-cirrascale-2 \
      --launch.follow=false --launch.step_soft_timeout=null
done
```

## Verifying the mask on real data

Before launching an arm, check the mask on real Qwen3.5-tokenized windows rather than trusting the
unit tests, which only cover synthetic layouts:

```bash
src/scripts/train/memexpress/sft_summtoken/verify_summary_mask_gantry.sh
```

One Beaker CPU node (no GPU, no triton — no model is built). It takes real document-chunked windows
off weka, runs the **production** emitter and role builder over them, and asserts at probe positions
spread through the context that a document token sees its own document but no other's, a summary
token relays from earlier runs, and the query sees every summary run but **no raw document content** —
then prints a picture of the mask. Non-zero exit if any window fails, so it works as a launch gate.

It ends with a **negative control** (`--summary-visible-tokens 0`) that is *expected to fail*; if that
run passes, the assertions are not binding and the main result is void.

Typical output:

```
=== window 0: T=262,144  documents=102  summary_tokens_each=5 ===
    roles: pad=10,050  instruction=30  doc_content=251,472  summary=510  query=82

    probe                   pos   instruction   own doc     other docs   summaries    query
    doc 51 content       124655         30/30   2519/2519     0/121852     255/255          -
    doc 51 summary       125499         30/30   3358/3358     0/121852     260/260          -
    query start          252012         30/30           -     0/251472     510/510        1/1
    answer end           252093         30/30           -     0/251472     510/510      82/82

    analytic block mask: 0.0308 of 2048x2048 blocks kept
    ==> window 0: OK
```

The `0/251472` column is the load-bearing one: the query is causally able to reach a quarter of a
million document tokens and reaches none of them, while reaching all 510 summary tokens.

## Knobs

All overridable by environment variable, and all propagated into the Beaker job (the job **rebuilds**
this config on the node, so anything resolved from the launch host must be forwarded or the rebuild
silently falls back to defaults):

| var | default | notes |
|---|---|---|
| `SUMMTOK_SEQ_LEN` | `262144` | see the 256k note below |
| `SUMMTOK_N_SUMMARY` | `5` | must match the shards' `num_summary_tokens` |
| `SUMMTOK_NUM_NODES` | `2` | 2 x 8 GPUs |
| `SUMMTOK_CP_DEGREE` | `4` | Ulysses only; DP = 16 / CP = 4 |
| `SUMMTOK_DATA_ROOT` | `.../amandab/summtoken_5task_xlong` | built by `build_summary_token_shards_gantry.sh` |
| `SUMMTOK_BASE` | `.../amandab/q35-4b-dense-256k-summfix/model_and_optim` | must be repaired |
| `SUMMTOK_MAX_STEPS` | `2240` | |
| `SUMMTOK_LR` | `4e-5` | `1e-5 * sqrt(GBS / 65,536)`, as in `sft_xlong256k` |
| `SUMMTOK_N_INSTANCES` | (unset) | **required by the curriculum arms** |
| `SUMMTOK_PACKING` | `0` | `1` packs several examples per window; see below |

⚠ `SUMMTOK_N_INSTANCES` feeds `mix_total_forwards`, which is divided by **`DP_DEGREE`, not
`WORLD_SIZE`**: under CP the four ranks of a DP group process the *same* instance. Using the world
size would make the anneal finish a quarter of the way through the run and sit pinned at its endpoint
for the rest — a silently different experiment. `derive_curriculum` asserts the anneal lands.

## On 256k

These runs are configured at 256k (2 nodes, Ulysses CP=4, DP=4, GBS 1,048,576, 2240 steps), matching
`sft_xlong256k`. The mask machinery is 256k-capable: the block mask is built analytically (~0.6 s and a few MB at
262,144, against the ~760 GiB and ~17 minutes that `create_block_mask` would need), Ulysses CP is
wired through `sdpa`, and the flex path is GQA-native.

What is *not* settled is padding waste. The default layout puts one example per window, and over a
2k→256k ladder that pads the short rungs almost entirely. Decide from the `launch_prep` numbers:
**turn on packing** (below), restrict the mixture to the long rungs, or stay at 32k. **Do not launch
256k blind.**

## Packing (`SUMMTOK_PACKING=1`)

`PackingInstanceSource` fits several EOS-terminated examples into one window instead of padding a
single example out to it. The mask keeps them fully separated: roles carry an `example_id`, `doc_id`
restarts per example, each example gets its own query region, and `same example` is a conjunct of the
whole rule — so **a packed window is exactly the block diagonal of the windows each example would
have got on its own**, which is the property
`src/test/nn/attention/summary_mask_packing_test.py` asserts directly.

Three things change when you turn it on:

| | unpacked | packed |
|---|---|---|
| instances per epoch | one per example | far fewer — **re-measure `SUMMTOK_N_INSTANCES`** |
| mixture arm | one coin per instance | one coin per **example** |
| over-long examples | truncated by `PadToLength` | dropped whole (`long_doc_strategy=exclude`) |

⚠ **Re-run `launch_prep` after switching.** `mix_total_forwards` is derived from the realized
instance count; reusing the unpacked number would end the anneal early — exactly the failure
`derive_curriculum` raises on. ⚠ **Do not compare a packed arm against an unpacked one**: they take
different numbers of optimizer steps over the same data, which is a confound stacked on top of the
intended contrast.

The mask does *not* get denser: block-diagonal structure means a packed window stays well under half
the density of plain causal attention over the same window.

Ring/zigzag CP is rejected outright: rank-local rows are a non-contiguous permutation and the kernel
understands only `causal + cu_seqlens`, which cannot express this mask. Ulysses only.

## Where the pieces live

| what | where |
|---|---|
| the mask rule and its levers | `olmo_core/nn/attention/summary_mask.py` |
| the kernel path, analytic block mask, CP | `olmo_core/nn/attention/summary_token.py` |
| roles + mixture wiring | `Transformer.enable_summary_token_attention` |
| the schedule | `olmo_core/nn/attention/chunked_mask.py` (`mask_mix_standard_prob`) |
| data emitter | `olmo_core/data/document_chunk_landmark.py` (`emit_document_chunk_summary`) |
| eval | `corpus_reasoning/eval/eval_lc_native_docchunk.py --variant summary` |
