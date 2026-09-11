# Rubric: Phase-0 pack-occupancy finding (Molmo2 Stage-2)

Target 8.0 · Max rounds 4 · Fixed before round 0, never edited mid-loop.

The artifact under review is an *empirical claim plus a proposed next experiment*,
not code. It will be used to decide whether to spend scarce B300 GPU hours.
A claim that is directionally right but quantitatively overstated is a FAILURE
here, because the magnitude is what justifies the spend.

## GATES (all must pass; any failure caps total at 4)

- [ ] G1 The reported sweep numbers reproduce from the committed script and cached data.
- [ ] G2 Existing test suite passes (src/test/data/multimodal, src/test/nn/vision),
      modulo the known-pre-existing parity_test.py::test_clip_parity.
- [ ] G3 All four stated claims are explicitly addressed by the critique.

## DIMENSIONS (weights sum to 100)

### D1 Soundness of the quantitative claim (weight 35)
Does "~3.2x LM forward+backward compute" survive scrutiny of what actually scales
with sequence length in this model?
  3 — Accepts 3.2x from the padded-token ratio alone; no per-component analysis.
  6 — Notes that FlexAttention skips fully-masked blocks so attention does not
      scale with padding, but does not quantify the resulting correction.
  9 — Decomposes LM cost into the parts that DO scale with padded positions
      (QKV/MLP/norms/RoPE/embedding/LM-head) and the parts that do not
      (block-diagonal attention), uses Molmo2-4B's actual dims, and states a
      corrected multiplier with its derivation. Names which term dominates.

### D2 Confound identification (weight 25)
Would acting on this claim change what the model trains on, not just how fast?
  3 — Treats the crop budget as a pure efficiency knob.
  6 — Notes that GLOBAL_BATCH_INSTANCES counts packs, so examples/step moves.
  9 — Traces the actual loss divisor in multimodal_train_module.py and states
      whether per-example gradient weighting is preserved; identifies that an
      examples/sec comparison at fixed gb is confounded by a ~3x change in
      effective batch size, and says what to hold fixed to decorrelate them.

### D3 Does the proposed experiment settle it (weight 25)
Is the recommended GPU A/B the one that would actually answer the question?
  3 — Endorses the A/B as stated with no changes.
  6 — Notes the memory constraint but does not specify the controlled comparison.
  9 — States the specific configs to run (what is held fixed vs varied), predicts
      the failure mode that would invalidate the result, and names a cheaper
      or more decisive alternative if one exists. Flags any lever that must be
      measured on B300 rather than screened off-cluster.

### D4 Honesty about what is not yet known (weight 15)
  3 — Presents the finding as settled.
  6 — Generic caveats ("needs GPU validation").
  9 — Enumerates specific residual risks: sampling adequacy (n=3000, order
      prefix), packer CPU cost at higher max_crops in dataloader workers, whether
      any prior telemetry already measured this, and convergence/quality risk
      from changed batch composition.

## SCALE
1-3 fails the objective visibly · 4-5 happy path only · 6-7 meets it with nameable
rough edges · 8-9 meets it with none · 10 only if no further change would help.
