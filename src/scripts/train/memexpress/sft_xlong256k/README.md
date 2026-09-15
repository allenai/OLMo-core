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

## Contradiction-only SFT from end-of-CPT (2026-09-14)

| Arm | Script | CPT run / checkpoint | Control |
|---|---|---|---|
| Dense | `Qwen3.5-4B-dense-contradiction-3ep-256k-SFT.py` | `fq3brt27` / `q35-4b-dense-256k-fix/step2385` | Compressive |
| Compressive | `Qwen3.5-4B-compressive-contradiction-3ep-256k-SFT.py` | `2brjoa8r` / `q35-4b-fastcomplm-256k-fix/step2385` | Dense |

Shared builder: `_qwen35_contradiction_256k_common.py`. Three **loader epochs** of only
`xlong5_2k256k_qwen35/shards_full/contradiction_train`, query position `both`, no task
mixing. Dense window 262144; compressive window 266368 (4162 blocks, content capacity
262206). Both use BFD, LR 4e-5, 3% warmup, seed 34521, two 8-GPU nodes, CP=4 / DP=4,
strict weights-only CPT loading and unique checkpoint folders under `amandab`.
Compressive incurs landmark compute overhead; its CPT had 1/64 fewer content tokens.
The loader drops the final incomplete global batch each epoch; prep records the count.
`prep_contradiction_256k.py` checks checkpoint files and builds both real datasets.


CPU prep passed on [Beaker 01M2H8QC6M1MN5EP910NPB1ST1](https://beaker.org/ex/01M2H8QC6M1MN5EP910NPB1ST1).
Both arms retain all 19,988 tokenized examples / 351,891,821 content tokens (longest
262,072); the original converter had excluded 12 oversized source examples.

| Arm | Packed windows | Steps/epoch | Three-epoch steps | Tail windows dropped/epoch |
|---|---:|---:|---:|---:|
| Dense | 1,343 | 335 | 1,005 | 3 (0.22%) |
| Compressive | 1,345 | 336 | 1,008 | 1 (0.07%) |

Epochs use the native loader's shuffled, full-batch convention. This is data-controlled,
with the small tail discrepancy above; it is not an equal-compute comparison.
Model-token budgets: dense 1,053,818,880; compressive 1,073,995,776, including padding.
The full corpus before batch-tail dropping has 1,055,675,463 content tokens over three passes.

`launch_contradiction_256k.py` submits detached jobs with exactly two replicas, eight GPUs
each, urgent priority, and `minRuntime: 3600000000000` (one hour). It requires a full pushed commit SHA:

```bash
python src/scripts/train/memexpress/sft_xlong256k/launch_contradiction_256k.py dense --ref <sha>
python src/scripts/train/memexpress/sft_xlong256k/launch_contradiction_256k.py compressive --ref <sha>
```

Pass `--dry-run` to inspect the Beaker spec without submitting, or `--run-name` to set a
fresh output namespace. The default run name includes a timestamp.


### Submitted jobs (one-hour minimum runtime)

Training source commit: `ed819780fadb1f1b3759074567f1c3e3c5cc9ea9` on
`amandab/contradiction-only-256k-20260914`.

| Arm | Run name | Beaker |
|---|---|---|
| Dense | `q35-dense-contra-3ep-256k-min1h-20260914` | [01M2H90PG60ST3S7MGPCQF27QN](https://beaker.org/ex/01M2H90PG60ST3S7MGPCQF27QN) |
| Compressive | `q35-compressive-contra-3ep-256k-min1h-20260914` | [01M2H90YY680NHJTJGS4Q88BQY](https://beaker.org/ex/01M2H90YY680NHJTJGS4Q88BQY) |

Checkpoints: `/weka/oe-training-default/ai2-llm/checkpoints/amandab/<run-name>/`.
Both replicas of each submitted job were verified as 8 GPUs, urgent, minRuntime=1h.
The earlier minRuntime=0 submissions (`01M2H8XM0BKSANRMJHE8N2SPBD` and
`01M2H8XRVAKKZQBB500VB1FN34`) were canceled before starting, following the runtime change.


### Compressive startup fix (2026-09-15)

The first compressive job failed before loading weights: the shared builder passed
`num_landmarks=1` to `fast_compressive_landmark`, which rejects that option because it
fixes the count at one internally. Removed the model option; the packer still explicitly
inserts one landmark every 63 content tokens. CPT geometry, data, epochs, parallelism,
LR, and strict weights-only loading are unchanged.

Validation now actually constructs both attention modules on the meta device. The
regression test is `src/test/scripts/contradiction_256k_config_test.py`; CPU prep now
performs this construction too. The old `dry_run` only counted parameters and therefore
missed the invalid model option. Dense was preempted after one hour, automatically
resumed, and reached step 414/1005 in the inspected logs; it needs no replacement.


Replacement compressive job: [01M2JAD9PZDYKQDKJ287FA866D](https://beaker.org/ex/01M2JAD9PZDYKQDKJ287FA866D),
run name `q35-compressive-contra-3ep-256k-min1h-20260915-r2`, pinned to fix commit
`1204d1b693e651c1520899c1427e2e7c65444be5`. Verified two 8-GPU replicas and minRuntime=1h.
Both attention-construction regression tests pass; full GPU training awaits scheduling.
