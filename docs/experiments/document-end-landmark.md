# Document-end compressive landmarks

Development branch: `amandab/doclandmark`.

## Implemented reference

- `emit_document_end_landmark(segments, mem_id=...)` preserves segment order and
  inserts one loss-masked landmark after each context document, without padding.
- `AttentionType.document_end_compressive_landmark` builds an eager reference layer
  with inherited projections, RoPE, QK normalization, and GQA expansion.
- `model.enable_document_end_landmark_attention(doc_start_id=...,
  doc_end_id=..., landmark_token_id=..., eos_id=..., pad_id=...)` reconstructs
  roles using the existing summary-role schema. Save these arguments in
  `config.document_end_landmark_attention` to restore this setup when loading a checkpoint.
- Exact causal attention: local content and instruction tokens compete with earlier
  document landmarks in the gate softmax. Each document gate is redistributed over
  its content plus landmark. The landmark query reads its document but not itself.
- Packed examples are isolated by example ID; padding outputs are zero.

For a llama-like config, set
`config.block.sequence_mixer.name = AttentionType.document_end_compressive_landmark`
before building. No `mem_freq` is needed. Inputs must already have landmarks;
this mode does not insert them during model forward.

The layout contract is complete, ordered, marked documents, each immediately
followed by one landmark, with instruction/free spans and a trailing question/answer.
Use a dedicated padding ID. Preserve whole documents when truncating. Arbitrary
malformed layouts and partial-document streaming are not supported or fully validated.
The ordinary summary-token attention mask is not used: completed document content
remains accessible through its landmark gate.

## Remaining work

1. Replace the Python-loop reference with a tiled variable-length implementation
   using document offsets. The reference materializes dense scores and is intended
   only for short tests; it is not appropriate for long-context training.
2. Replace reference cached decoding with an optimized span-based implementation.
3. Add document-level top-k with retrieved-token accounting, followed by CP support.
4. Integrate launcher configuration and run GPU validation before training.

Dropout, sliding windows, and mask mixing are rejected. No GPU kernel, training
launcher, job, or benchmark is included yet.

## Validation

`PYTHONPATH=src python -m pytest -q
src/test/nn/attention/landmark_document_end_test.py
src/test/data/document_chunk_landmark_test.py`

Tests cover emission, equal-block forward/gradient parity against existing
compressive grouped softmax, unequal document lengths with analytically known
weights, finite-difference gradients, packed-example isolation, causality, padding,
GQA layer integration, and model role reconstruction in training and evaluation.

## Conversion and generation

The training converter accepts `--emit document_end_landmark`. It validates
complete marked documents and immediate closing landmarks, drops examples that
exceed the sequence budget, and records `landmark_placement: document_end`,
`landmarks_per_document: 1`, and tokenizer-specific IDs in metadata. It rejects
marker-free input and synthetic summary-span insertion for this mode.

The native eval entry point `corpus_reasoning.eval.eval_lc_native_docchunk` accepts
`--variant document_end_landmark --landmark-token-id <id>` and verifies boundary
IDs against the loaded checkpoint. Use batch size 1 and exact retrieval. The
existing `--doc-start-id`, `--doc-end-id`, and `--eos-token-id` must match the
checkpoint. This entry point still requires CUDA; no GPU run has been performed.

Cached attention supports a single unpadded prompt and continuation in the trailing
query/answer region. Prompt roles are retained across steps; generated tokens are
query tokens, with no periodic landmark insertion. Resetting the ordinary KV cache
also replaces role metadata at the next prefill. Packed prompts and continuation
inside an unfinished context document are rejected. The continuation API is for
answers, not streaming new context documents.

Additional tests cover train/eval emitter parity, malformed layouts, model-config
serialization, cached/full-logit equivalence with RoPE, and cache resets. These are
CPU correctness tests; the eager Python-loop implementation remains unsuitable
for long contexts.
