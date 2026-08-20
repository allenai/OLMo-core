# SSMax adaptation diagnostics

This probe measures the mechanism implicated by the QK-norm comparison. It is deliberately
separate from the downstream outcome suite: BLINK jigsaw and MathVista geometry remain the main
ranking signals, while this report asks whether an arm develops large attention logits or loses
effective context during adaptation.

The fixed probe is pinned in `ssmax_attention_probe_v1.yaml`. Its 32 PixMo-caption validation rows
come from the existing content-disjoint validation-manifest v3. The probe manifest pins the upstream
manifest bytes, source row/content identities, live Arrow identity, tokenizer contract, tokenized
valid prefixes, and exact sampled query positions. Every query selection is SHA-256-priority based.

## What is collected

Only the four global `Attention` layers with Scalable-Softmax are instrumented; GatedDeltaNet layers
are intentionally out of scope. For every global layer, query head, and `all` / `image` / `prompt` /
`response` category, the collector reports:

- Q RMS immediately before SSMax and after QK normalization (when present). RoPE preserves this
  per-head magnitude.
- Q RMS after multiplying by `ssmax_scale[head] * log(causal_prefix_length)`.
- K RMS after QK normalization and RoPE.
- the learned per-head SSMax scale and sampled effective multipliers;
- signed and absolute pre-softmax logit distributions;
- normalized entropy, effective context, effective-context fraction, maximum attention
  probability, and visible-key count;
- post-softmax probability mass routed to image, prompt, and response keys, together with total
  mass over allowed keys;
- the share of sampled queries whose maximum-probability key is an image, prompt, or response key;
- maximum softmax-normalization and key-category-partition residuals as integrity checks.

GQA is reconstructed with `kv_head = query_head // (n_heads / n_kv_heads)`. Allowed keys exactly
follow `(causal | bidirectional_image) & subsegment & example & valid_padding`. Query logits are
computed in bounded chunks, and distribution quantiles use a deterministic priority reservoir.
Exact counts, moments, minima, and maxima do not use the reservoir. Image, prompt, and response
key masks exactly partition every allowed key. The mean of each `argmax_key_is_*` indicator is the
reported argmax-key share, so its three means must sum to one. Likewise, the three
`attention_mass_to_*_keys` means must sum to `attention_mass_to_allowed_keys` (normally one).

The JSON report remains the additive `ssmax_attention_diagnostics` version-1 schema. Reports with
routing metrics identify their measurement protocol as
`fixed-multimodal-ssmax-attention-diagnostics-v2`. Rank-local states declare an exact
`metric_schema`; the finalizer still accepts pre-routing version-1 states as the legacy protocol,
but rejects mixtures of legacy and routing-aware states. Comparisons of two legacy reports remain
valid and simply omit `key_routing`; comparisons of two routing-aware reports add it to each
layer/head/query-category record. Reports from different protocols cannot be compared.

## Checkpoint runner integration

The collector intentionally does not own model loading. Parent checkpoints and alignment
checkpoints have different strict native-DCP contracts, and silently treating them as the same is
unsafe. A runner must do the following:

1. Build the generic `MultimodalLM` from the relevant Vision Alignment config with compilation and
   activation checkpointing disabled. Do not install or use a KV cache.
2. Use the permanent alignment `step0` save as the scientifically defined parent-composite
   baseline. Its lineage must prove the exact bare-parent LM prefix load, pinned SigLIP load, and
   deterministic connector/image-row initialization. A bare text parent does not define image
   features and therefore is not reported as an image-conditioned baseline. The strict step-0
   identity check remains tied to the exact parent checkpoint named by the recipe.
3. For `step0` and every later generic alignment DCP, use the ordinary strict
   `Checkpointer.load(...)` path with
   trainer and optimizer state disabled. Never use the partial parent-load path for a resume save.
4. Reconstruct batches with `iter_ssmax_probe_batches(...)`. It materializes exactly the manifest's
   `population.selected_dataset_indices` at epoch 0, checks the source/index/content-derived sample
   IDs, and partitions samples disjointly by DP-rank ordinal.
5. Execute them with `capture_ssmax_probe_batches(...)`, whose callback must perform exactly one
   native no-cache eval forward. The underlying capture is equivalent to:

   ```python
   with collector.capture_batch(
       sample_ids=sample_ids,
       input_ids=batch["input_ids"],
       token_type_ids=batch["token_type_ids"],
       loss_masks=batch["loss_masks"],
       valid_tokens=batch["router_token_mask"],
       subsegment_ids=batch.get("subsegment_ids"),
       example_ids=batch.get("example_ids"),
   ):
       model(**model_batch)
   ```

6. Gather the returned compact `collector.export_state()` from every rank and call
   `SSMaxAttentionDiagnosticsCollector.finalize_states(...)` on rank zero. The merge rejects missing,
   duplicated, or unexpected probe samples and differing layer metadata.

The executable `src/scripts/eval/ssmax_attention_diagnostics.py` validates manifests, finalizes
rank states, and compares reports. `build_ssmax_attention_probe_manifest.py` reproduces the fixed
manifest from a checkpoint config.

## Comparisons

The strongest design is a trajectory comparison within each arm (parent composite, bridge saves,
perception saves, joint saves), followed by a cross-arm comparison at matched steps. Default flags
mark a normalized-entropy drop of 0.10, a 2x contraction in effective-context fraction, 2x absolute
logit q99 growth, or 2x post-SSMax Q-RMS growth. These are triage heuristics, not success criteria;
they should be correlated with the matched downstream trajectories rather than interpreted alone.
Each per-layer/head/query-category comparison also retains baseline, candidate, and delta values
for destination probability mass and argmax-key share. These routing deltas are descriptive rather
than independently gated: they are intended to test whether changes in magnitude-influenced
routing track BLINK-jigsaw and MathVista-geometry changes over the adaptation trajectory.
