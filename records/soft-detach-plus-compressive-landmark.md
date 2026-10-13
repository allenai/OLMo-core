---
title: Soft detaching + compressive landmark — can they be combined?
date: 2026-09-12
updated: 2026-09-12
experiment: 
area: landmark
kind: plan
status: design note
superseded_by: 
summary: Whether soft-detached pooling and compressive landmark can be combined.
---
# Soft detaching + compressive landmark — can they be combined?

Design note, 2026-09-12 (Prasann's idea, raised while the ds64 wave was blocked on Beaker capacity).
No runs yet: jupiter was at 0/1022 free slots.

## What each mechanism actually is

**Soft-detached document pooling** (`--variant softtoken`, `enable_pooled_soft_tokens`): a pooled
document's whole KV collapses to ONE token — the mean input embedding through an identity projector
at the document's centre position. Slot K/V is detached, and the GDN write channels are detached too
(commit 96f2fba11, asserted by `src/test/nn/pooled_soft_token_gdn_test.py`), so a slot contributes
**zero gradient to any parameter**. The stack then runs on a **compacted** sequence. Inference is
ordinary full attention, so the saving is **training-only**.

**Compressive landmark** (`FastCompressiveLandmarkAttention`, `landmark_compressive.py`): in ordinary
landmark attention a block's landmark token is only a *gate* — it sets the block's weight and then
contributes nothing to the output (multiplied by zero). The compressive variant keeps the gate but
includes the landmark token in the within-block softmax, so **its value enters the output as a
learned, compressed summary of its block**. At decode with hard top-k retrieval, non-selected blocks
are not dropped: their landmarks collectively retain a fixed fraction `nonselected_landmark_mass`
(alpha) of the attention mass. So the saving is **train AND inference**.

Stated plainly: both replace a span with one summary token. Soft-detach's summary is **fixed and
gradient-free**; compressive landmark's is **learned and contributes its value**. That is the whole
difference, and it is why the combination is interesting rather than redundant.

## Is it composable today? No — two blockers, one shallow, one real

1. **Shallow:** `--variant` in `train_ctc_suite.py` is a single required choice; `softtoken` and
   `sparselandmark` are mutually exclusive. `fast_compressive_landmark` / `document_compressive`
   exist on `TransformerConfig` but are not exposed on the trainer CLI.
2. **Real:** `enable_pooled_soft_tokens` explicitly rejects the chunked-mask path —
   *"mutually exclusive with enable_document_chunk_attention (the compacted sequence is plain
   causal)"*. Compaction throws away the document structure that any landmark/chunked pattern needs.

The second blocker is not fundamental, though: after compaction the sequence still *has* documents —
the kept real documents plus one slot per pooled document. `chunk_ids` could be **recomputed on the
compacted sequence** (kept docs keep their chunk, each slot becomes a 1-token chunk). Note
`DocumentLandmarkAttention` already derives its structure from `<|doc_start|>`/`<|doc_end|>` boundary
tokens — the *same* tokens the soft-token path keys off — so the document-granularity landmark is the
natural partner here, not the block-stride `sparselandmark` (which additionally needs landmark-packed
data, a data-layout conflict with compaction).

## The strongest version of the idea

**Replace the soft token's fixed, detached mean-embedding slot with a learned compressive
document-landmark token.** Identical compression ratio — one token per pooled document, so the same
FLOP saving — but the summary is trained through the loss and its value reaches the output.

Why this attacks the failure we actually measured:
- The campaign's central weakness is that a mean slot cannot support cross-document comparison:
  outlier collapses beyond 2k, and every task degrades with context length
  (`records/ds64-handoff.md`). The eval-side probe put it exactly: *"a mean-embedding slot may stand
  in for a document's free text but never for the tokens the question matches exactly"*
  ([[header-real-pooling-parity]]). A learned summary is the obvious way to carry what a mean cannot,
  and it is the same direction as the header-real fix that reached parity.
- The pooled-KV rounds already probed the fidelity axis and found something that **supports** this:
  oracle (perfect, static) slots *hurt* — v23 0.809 vs v21 0.857 — because "cached slots freeze in
  the base frame and go stale while projector slots track training" ([[pooled-doc-kv-task]]). The
  axis that worked was trainability/tracking, and a compressive landmark sits further along it.
- It converts a **training-only** saving into a **train + inference** saving, which is worth more:
  today's soft-token arms give 0.2-0.4x training FLOPs but full-price inference.

Honest counter-arguments:
- The detach was **deliberate**, to stop slot co-adaptation and to keep slots gradient-free. Making
  the summary learned gives part of the training saving back (backward through slots) and re-opens
  the co-adaptation risk that detaching closed.
- Several neighbouring knobs are settled negatives: "no logit bias (every bias variant lost)", plus
  keep policies, slot RoPE position and slot bias at keep 0.
- Compressive landmark's inference win depends on `nonselected_landmark_mass`; the sparse-landmark
  arms have their own failure history (qdmatch_nq fails architecturally, 15x CE gap).

## Minimal first experiment (when capacity returns)

Cheapest decisive probe, on **contradiction** (not outlier): contradiction is the task where pooling
already works and where a summary-quality change should show cleanly, and it has a dense ladder to
compare against. Outlier is the wrong first target — its problem is the gold shortcut
(`kvgb`, in flight), a different variable that would confound this one.

1. Expose `fast_compressive_landmark` / `document_compressive` on the trainer CLI.
2. Add a `softtoken`-derived variant that recomputes `chunk_ids` on the compacted sequence and runs
   `DocumentLandmarkAttention` over it, with the per-document landmark **trainable** (do NOT detach
   it) while non-kept document bodies stay pooled.
3. One arm at keep 1/3, 32M and 64M tokens, against `hdr33` at the same budgets — same rows/step so
   FLOPs and accuracy stay comparable. Success = beating `hdr33`'s 0.810/0.855 at equal or lower PF.

Do NOT start this until the `kvgb` wave reports: if gold-blind alone fixes outlier, the priority
order changes.
