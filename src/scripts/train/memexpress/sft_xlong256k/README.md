# sft_xlong256k/ — 256k-context SFT on the xlong5 2k→256k ladder (Qwen3.5)

SFT of the Qwen3.5-4B dense 256k CPT base on 75% the xlong5 2k→256k 5-task mix / 25%
`allenai/Dolci-Instruct-SFT`, at a 262,144 window. Beaker only, 2 nodes.

**A controlled pair on query position.** Both arms are built by
`_qwen35_xlong5_dolci25_256k_common.py` from one set of constants, so everything except the
5-task shard root is shared by construction. Verified: the two configs' `dry_run` dumps differ
only in the five data paths and the run name — batch, LR, steps, base checkpoint, weights,
ratios, seed and parallelism produce no diff. **Add an arm by adding a row to `_ARMS`, never by
forking the common file.**

| Script | Arm | 5-task data root | Role |
|---|---|---|---|
| `Qwen3.5-4B-dense-xlong5-qboth-dolci25-256k-SFT.py` | `qboth` | `xlong5_2k256k_qwen35/shards_full` | **Control.** Original build (`--query-position both`). |
| `Qwen3.5-4B-dense-xlong5-qafter-dolci25-256k-SFT.py` | `qafter` | `xlong5_2k256k_qwen35_qafter/shards_full` | **Treatment.** `--query-position after` rebuild of the same pools. |

Shared by both: base `q35-4b-dense-256k-fix/step2385` (weights-only, strict), 75/25 blend,
within-mix weights 2.0/1.5/1.5/1.0/1.0, 2 nodes × CP=4 → DP=4 → 1,048,576 tok/step, LR 4e-5,
2,240 steps = 2.35B tokens, seed 34521.

## Why `qboth` exists rather than reusing the legacy run

`src/scripts/train/sft/amanda-landmark/Qwen3.5-4B-dense-xlong5-dolci25-256k-SFT.py` (run
`q35-4b-dense-xlong5-dolci25-256k`) already trained the `qboth` data from the same base
checkpoint — but at 560 steps × 4.19M tok/step and LR 1e-5. It differs from the qafter arm in
**batch and LR as well as data**, roughly 16× the path length through parameter space, so it
cannot be the control. `qboth` reproduces its data at this family's batch and LR; the pair above
differs in data alone.

The legacy dense/landmark 256k pair remains a valid landmark-vs-dense comparison with each other.

## Reading the pair

- **Exclude outlier.** Per the qafter tree's README, outlier's converter branch has no positioned
  query and was already query-after; its shards were rebuilt only so the root is self-contained.
  Four of the five tasks carry the contrast.
- **Eval flag.** Both arms read `xlong5_2k256k_qwen35/eval/` (the qafter root ships no `eval/`,
  since query position is an eval-time rendering flag). Pass `--query-position after` for the
  qafter arm, the default for qboth. Mismatching this makes a run read as a collapse.
- **The one unmatched axis.** The qafter rebuild tightened the instance cap 262,144 → 250,000,
  so `qboth` holds 112 of 99,944 instances (0.11%, ~1.8% of tokens) that `qafter` never sees,
  concentrated in the 128–256k band. Not fixable in config — the window must be a power of two —
  so name it if a long-rung delta comes out small. The 112 reconcile per-task as contra 27, nq 22,
  oolong 7, outlier 20, rerank 36.

## Measured pool sizes (from each task's `metadata.json`, 2026-08-11)

| | tokens | instances | longest example |
|---|---|---|---|
| `qboth` | 1.764B | 99,944 | 262,072 |
| `qafter` | 1.732B | 99,832 | 249,950 |

Neither reaches the 262,144 window, so `LongDocStrategy.exclude` drops nothing on either arm —
asserted in the common file rather than left to a comment.
