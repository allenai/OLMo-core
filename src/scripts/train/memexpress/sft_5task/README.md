# sft_5task/ — canonical 5task 32k SFT (Beaker)
The fixed 700M-token 5task mix (see sft-canonical-mix memory / results-hub): dense, compressive,
fast-landmark × {32k-nocpt, fixnq, cptmix-32k, cptmix-64k}. The `-fixnq` variants use the p10
hard-neg NQ rebuild — the only valid NQ (never the old 98%-hard files).

## Landmark data-vs-compute ablation (Qwen3.5)

Two arms in `_qwen35_5task_dolci25_32k_nocpt_common.py`, both `fast_landmark` against the
`dense` arm (`q35-4b-dense-5task-dolci25-32k-nocpt`, Beaker `01KZAF94DPA971G2J7YCESFC74`). Both
run at `sequence_length=33344` (521 blocks of 64, content capacity 32,823 ≥ dense's 32,768), with
`LandmarkPackingStrategy.best_fit_decreasing` and the **dense** 2.9/1.3 sampling weights, so they
differ from the baseline only in the landmark geometry — and from each other only in duration.

| launcher | arm | matched on | steps |
|---|---|---|---|
| `Qwen3.5-4B-fast-landmark-5task-dolci25-33344-datamatch-SFT.py` | `fast-landmark-datamatch` | original data (32.86% of the epoch) | from measured `LANDMARK_ABLATION_INSTANCES` |
| `Qwen3.5-4B-fast-landmark-5task-dolci25-33344-tokenmatch-SFT.py` | `fast-landmark-tokenmatch` | token budget (701.2M) | 10,515 |

The datamatch arm needs a CPU-only `launch_prep` first to measure the landmark instance count; it
refuses to build until `LANDMARK_ABLATION_INSTANCES` is set. Background:
`records/POSSIBLE_BUG_SFT_DATA.md`.

## Qwen3 shared-vector 32k arms

Both use `_qwen3_sharedvec_33344_common.py`: block64/vec32, 33344-slot BFD packing,
dense task weights, p10 NQ, shared-vector CPT step2385, 10515 steps / 701.224M slots.

| launcher | mixture | training RoPE | comparison |
|---|---|---|---|
| `Qwen3-4B-sharedvec-5task-33344-tokenmatch-SFT.py` | five tasks only | native | Earlier pilot recipe |
| `Qwen3-4B-sharedvec-5task-dolci25-33344-tokenmatch-SFT.py` | 75% five tasks / 25% Dolci Qwen3 | Native CPT RoPE, theta1000000 | Dense `q4b-dense-5task-dolci25-32k-nocpt/step10700` recipe and token budget |

Token matching includes landmarks and padding; it does not imply equal data exposure or FLOPs.
The Dolci repetition ceiling is 8 (dense saved config: 1); CPU preparation must verify that
Dolci is downsampled so the higher ceiling is inactive. Use fresh run names.

The Dolci25 arm preserves shared-vector CPT positional encoding by explicit user request;
its RoPE therefore differs from the dense reference, which trained with YaRN2.
