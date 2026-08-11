# Why is dense weak at SHORT context in the 256k runs?

**Status: two leading hypotheses ruled out with evidence; one structural difference identified as
the prime suspect; the discriminating experiment is cheap and not yet run.** 2026-08-11.

## The runs

| arm | checkpoint | sequence mixer |
|---|---|---|
| dense | `amandab/q35-4b-dense-xlong5-dolci25-256k` | `default` (full attention) |
| compressive landmark | `amandab/q35-4b-fastcomplm-xlong5-dolci25-256k` | `fast_compressive_landmark`, `mem_freq=63`, `num_landmarks=1` |

Both at `sequence_length = 262144`, both HSDP, both step560. There is also a `-ep1` variant of the
complm run. Everything below is read from the runs' own saved `config.json`, not inferred.

## Ruled out

**1. RoPE asymmetry — refuted.** This was the leading hypothesis and it is wrong. Both arms carry
`model.block.attn.sequence_mixer.rope.theta = 10000000` and neither has any `rope_scaling`. And
10,000,000 is Qwen3.5-4B's *factory native* theta (`TransformerConfig.qwen3_5_like`,
`rope_theta: int = 10_000_000`), so **neither arm received any context extension at all** — no YaRN,
no NTK theta bump.

The suspicion was well-founded and still worth recording, because the asymmetry is real *elsewhere*
in this codebase and would produce exactly this symptom if it applied:

- The `sft_longctx` launcher family applies `with_rope_scaling(YaRN factor=2)` to the dense arm and
  carries an explicit "No YaRN" comment on both landmark arms.
- `train_ctc_suite.py` **silently ignores** `--rope-yarn-factor` for `qwen3_5`, warning that
  `with_rope_scaling` refuses hybrid named-block models.
- YaRN's attention rescale is `0.1·ln(factor) + 1`, a **constant** applied at every sequence length:
  1.069 at factor 2, 1.208 at factor 8. It does not know the sequence is short, so a
  long-context-extended dense arm would attend more sharply than its base was trained for *at 2k*,
  while a no-YaRN landmark arm would not.

None of that is in play for these two checkpoints. If a future 256k run does extend RoPE on one arm
only, this is the first thing to check.

**2. Different training data — refuted.** Both arms read the *identical* shards:

```
prasanns/xlong5_2k256k_qwen35/shards_full/{contradiction,nq,oolong,rerank,outlier}_train
amandab/dolci-instruct-sft/qwen35                       (ratio 0.25 against the 0.75 five-task mix)
```

Note the landmark arm reads `shards_full` — the **dense-emit** tokenization — and inserts landmark
tokens at pack time via `LandmarkPackingInstanceSourceConfig` (`mem_id=248200`, `pad_id=248044`).
So there is no separate landmark tokenization lineage to drift, which is a real improvement over the
docchunk family where the two arms read separately-built shards.

**3. Intra-document masking — correct in both, by different mechanisms.** At a 256k window with
2k-long examples, packing puts ~100 examples in one window, so cross-example attention leakage would
hurt *short* examples specifically and spare long ones — a good fit for the symptom. But both arms
mask, deliberately:

- dense: `data_loader.generate_doc_lengths = True` → EOS-derived `doc_lens` → varlen attention.
- complm: `generate_doc_lengths = False`, because landmark attention **requires block-aligned**
  `doc_lens` from its own packer and raises if handed EOS-derived ones
  (`nn/attention/landmark.py:62`, `nn/attention/__init__.py:1915`).

## The prime suspect: the two arms start from different CPT bases

```
dense   trainer.load_path = .../q35-4b-dense-256k-fix/step2385/model_and_optim
complm  trainer.load_path = .../q35-4b-fastcomplm-256k-fix/step2385/model_and_optim
```

Two separately continued-pretrained bases. This is *necessary* — a landmark model needs a
landmark-adapted base — but it means the SFT comparison inherits whatever the CPT stage did, and a
short-context deficit created during 256k CPT would look exactly like a property of dense attention.

There is independent reason to think these bases vary a lot in quality: amandab's own RULER numbers
for the `dolma3-longmino` CPT bases at step2385 differ substantially between variants
(`partial_rope` 0.916 vs `block_local` 0.879 at 4k, widening to 0.355 vs 0.166 at 128k).

**The discriminating experiment, and it is cheap:** grade **both CPT bases at step2385, pre-SFT**, at
the short rungs (2k–8k). If dense is already behind there, the cause is upstream in CPT and neither
the SFT nor the eval is implicated. If both bases are level and the gap only appears after SFT, the
cause is in the SFT stage and the next suspect is the one below.

## Secondary: the arms did not see the same effective data

`dataset[0].long_doc_strategy` is **`exclude` on the dense** packing source, while the complm arm's
sources carry `truncate`. `exclude` **drops** documents that exceed the window; `truncate` keeps a
prefix. At a 262144 window this should bite rarely, but the `xlong5` ladder runs to 256k by
construction, so it is exactly the data most likely to cross the line — and it is dropped for one arm
and kept for the other. Worth counting before quoting any dense-vs-complm headline.

This affects the *long* end, so it does not explain a short-context deficit on its own.

## Do not conflate this with the nq@3k artifact

results-hub carries `nq @3k f1 = 0.003` for the 32k-trained dense run
(`q4b-dense-5task-v2-32k-nocpt`), which looks like the same phenomenon and is not. That one is a
**known, diagnosed eval-file artifact**: the 3k rung alone used a 50-query multi-query file, and the
scorer's `is_multi` branch demanded all 50 answer sets in one 64-token generation. With the
single-query k20 file it scores 1.0 (12/12). See `[[nq-eval-retrieval-3k-gotcha]]`.

## Method note

The config diff was taken by flattening both JSONs and comparing key paths. The two dataset classes
nest differently (`dataset[0].sources[…]` for dense vs `dataset[0].source[…]` for complm), so
**nested keys show as present/absent rather than compared**, and per-source values that differ only
in path were not diffed. The top-level differences reported above are reliable; a per-source
comparison needs a path-normalising diff.

Probes used (Beaker, GPU-less, pinned to a pushed commit because gantry clones the pushed ref):
`ctc-probe-256k-rope` (`01KZS0402ECHC2864JHBZH0N6R`), `ctc-diff-256k` (`01KZS0C2A76HPZ3ZGFCKECWWH1`).
