# 256k contradiction SFT — feasibility notes (Qwen3-4B vs Qwen3.5-4B)

Captured 2026-07-23 from the CP-feasibility investigation. Grounds the plan presented to the user
after the 2k sanity evals.

## Context parallelism IS supported in olmo-core (no new infra needed)
- `TransformerTrainModuleConfig.cp_config: TransformerContextParallelConfig` (train_module/transformer/config.py).
  Constructors: `.ulysses(degree)`, `.zig_zag(degree)` / `.llama3(degree)` (ring).
- CP shards ACTIVATIONS/sequence only; params/optimizer sharded by `dp_config` shard_degree (independent lever).
- Working precedent: `sft_longctx/Qwen3-4B-dense-longctx-SFT.py` — 64k, `ulysses(degree=8)`, HSDP shard_degree=1,
  budget(0.7) AC, 1 node (8×H100). `cpt/debug/Qwen3-4B-long-context.py` — 64k, ulysses(8), 4 nodes.
- Beaker launcher already exists: `ctc_suite/beaker_ctc_suite.py` (jupiter, urgent) — but has NO cp flag yet.
  Extending it + train_ctc_suite.py's build_train_module_config to accept `--cp-degree` is the code change.

## THE binding constraint: Ulysses CP degree caps (n_kv_heads)
- Ulysses shards heads → requires `n_kv_heads % cp_degree == 0`.
- **Qwen3-4B dense**: n_heads=32, n_kv_heads=8 → Ulysses cap = **8**. To exceed 8 (needed for 256k) must use
  **ring CP** (`zig_zag`/`llama3`), which only needs seq divisibility — BUT ring needs the `ring-flash-attn`
  pip package importable in the Beaker image (UNVERIFIED — check `OLMoCoreBeakerImage.stable`).
- **Qwen3.5-4B hybrid**: full-attn layers n_heads=16, n_kv_heads=4 → Ulysses cap = **4**. Ring CP is
  **NOT available** — `GatedDeltaNet.apply_cp()` raises NotImplementedError for ring, and CP style is applied
  model-wide. So the hybrid is HARD-CAPPED at **CP=4**. block_pattern = [gdn,gdn,gdn,attn] (1/4 full-attn).

## Memory ceiling
- SkipStepAdamW fp32 master+moments = ~64GB static param/optim footprint at shard_degree=1, independent of
  seq len / CP. This dominates 80GB H100. Must ALSO enable FSDP shard_degree>1 (needs dp_world_size>1 after CP)
  + full-block AC. CP alone is insufficient at 256k.

## Ballpark node counts for 256k (seq_len 262144)
- Reference that FITS: dense 64k, ulysses(8), shard_degree=1, budget(0.7) AC, 1 node → local chunk 8k tok/GPU.
- **Qwen3-4B dense**: local-chunk-parity needs CP≈32 → ring CP=32 + ring-flash-attn, ~4 nodes (32 GPU) for one
  CP group, + more for optimizer sharding → **4–8 nodes**, full AC.
- **Qwen3.5-4B hybrid**: CP capped at 4 → local chunk 64k tok/GPU (8× the proven chunk!) but only 1/4 layers
  O(N²). Needs CP=4 × HSDP dp_shard (dp_world_size=4–8 on 16–32 GPU) + full AC → **2–4 nodes**.

## Implications / open decisions for the user
1. The two models need DIFFERENT parallelism (dense=ring-32, hybrid=ulysses-4) → not a clean "matched config"
   comparison; it's matched-TASK/matched-context, different parallel strategy (unavoidable, arch-driven).
2. ring-flash-attn availability in the Beaker image must be verified for the dense run.
3. 256k is genuinely expensive: ~2 runs × multi-node × slow (bs=1, 5000 seqs of 256k). Consider whether to
   ladder up (32k→64k→128k→256k) and/or reduce example count at the longest rung.
4. Data build: n≈11200 docs, 5000 examples, 2 tokenizers, seq_len 262144 → ~2.6B tokens tokenization + a
   256k eval rung (staged rungs only go to 32k). Heavy but CPU-only; token calibration must be re-probed at n≈11200.

## Data/eval status (2k sanity, done/in-flight)
- Shards: /scratch/users/prasann/ctc_qwen_compare/contra_2k_{qwen3,qwen35}_n77_5k (5000 ea, verified).
- Bases: qwen35 = ctc_suite_lambda_stage/q35-4b-base-modelonly; qwen3 = ctc_qwen_compare/bases/qwen3-4b-base-trainedmark.
- Source pool for scaling: contradiction_train_pubmed_recomb_n50_k3.jsonl (7421 ex); expand via
  generate_pubmed_contradiction_data.py --expand-from-train --num-docs N (PubMedQA fillers, cached).

## Launch config decision (user, 2026-07-23)
- **Qwen3-dense**: `--num-nodes 2` (16 GPU), CP=8 → dp=2 → FSDP shard_degree=2 (~28GB static, fits
  80GB) + full AC + YaRN factor=8 + `--pack`. 2 nodes chosen BOTH for ~2x speed AND to shard the
  optimizer (1 node/CP=8 → dp=1 → no param sharding → ~56GB static → OOM risk). global_batch=16.
- **Qwen3.5-hybrid**: `--num-nodes 1` (8 GPU), CP=4 → dp=2 → shard_degree=2 + full AC + `--pack`
  (YaRN skipped — hybrid named-block limitation). global_batch=8. Cheaper (~1/4 layers full-attn).
