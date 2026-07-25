# Length-mix scaling law — STATUS

**Updated 2026-07-24 22:30 PDT.** Autonomous overnight loop (dynamic self-pacing).
Scope per user: **hybrid Qwen3.5-4B only** (no dense variant).

## The question
Holding the long-context training pool **fixed**, does adding progressively more short-context data
improve **f1@32k**? And does that beat the uniform production mix at **equal token cost**?

## Baseline (do not re-measure)
`ctc-s5-contra-full-4b`, uniform 2k–32k mix, **vLLM**: 2k 0.849 / 8k 0.690 / **32k 0.335**.
⚠ vLLM ONLY. The native backend reads 0.571/0.219/0.038 on the *same* checkpoint — a degraded
harness, not a model property. Mixing backends fabricates results.

## Round 1 — TRAINING COMPLETE (9/10, B2 rerunning)

| arm | long tok | short tok | total | steps | train CE (last10) |
|---|---|---|---|---|---|
| A0 | 35.2M | 0 | 35.2M | 67 | 1.0807 |
| A1 | 35.2M | 20.0M | 55.2M | 105 | 0.0425 |
| A2 | 35.2M | 41.5M | 76.7M | 146 | 0.0170 |
| A3 | 35.2M | 84.5M | 119.7M | 228 | 0.0095 |
| A4 | 35.2M | 148.7M | 183.9M | 351 | 0.0019 |
| B0 | 20.2M | 0 | 20.2M | 39 | 1.0729 |
| B2 | 20.2M | 21.5M | 41.7M | 80 | *rerunning* |
| B4 | 20.2M | 84.5M | 104.7M | 200 | 0.0085 |
| C3 | uniform | — | 119.7M (=A3) | 228 | 0.0238 |
| C4 | uniform | — | 182.9M (≈A4) | 349 | 0.0119 |

⚠ **Train CE is NOT comparable across arms** and must not be reported as a result. Each arm's CE is
measured on its *own* data distribution — short-heavy arms are dominated by easier short examples —
and arms differ in optimizer-step count by design (67 vs 351). The only comparable numbers are
**f1 at fixed eval rungs**.

## ⚠ vLLM venv: torch-pin trap (first build is suspect, corrected build in flight)
`lm-vllm-venv` completed but ended at **torch 2.13.0+cu130**, with pip itself warning
`vllm 0.25.1+cu129 requires torch==2.11.0 ... which is incompatible`. Cause: flashinfer drags torch
upward, and my version-detection read the version *after* that drift, so it re-pinned to the
drifted 2.13.0 instead of the 2.11.0 vLLM needs. vLLM's compiled extensions are built against the
2.11.0 ABI, so imports succeed but model load is expected to fail.

The proven recipe prevents this with a **`PIP_CONSTRAINT` file** applied before installing
flashinfer — which I omitted. Corrected build `lm-vllm-venv-fix` →
`shared/vllm_venv_fix`, pinning the cu129 triad (torch 2.11.0 / torchvision 0.26.0 /
torchaudio 2.11.0) and asserting the pin held. **If the pilot fails at model load, point
`eval_arm_beaker.sh` at `vllm_venv_fix`.**

## Round 1 — EVAL: in progress
Chain per arm: olmo distcp → HF export → vLLM serving copy → generate @2k/8k/32k → graded f1.
- shared vLLM venv on weka: **building** (`lm-vllm-venv`, forcing the cu129 torch triad)
- eval inputs on weka: **ready** (rung_2048/8192/32768 + base config/tokenizer)
- **pilot arm A0**: queued behind the venv build — validating the untested chain on ONE arm before
  spending 10 GPUs
- remaining 9 arms: launch after the pilot proves the chain

## Working config (hard-won; do not change casually)
`seq_len 65536` · `cp_degree 4` · `budget-AC 0.3` · compile ON · `--pack` · `global_batch 8` · 1 node.
Two failed launch rounds established this:
1. `--pack` requires a **power-of-2 `--seq-len`** (SegmentTree assertion). 40960 dies ~15 min in on
   rank 0 only, leaving 7 ranks hanging to a 900s gloo timeout — a bad argument that costs a
   node-hour and reports itself as a comms failure. Guard added in `train_ctc_suite.py` (uncommitted).
2. `full-AC + CP + compile` → `CheckpointError: recomputed metadata mismatch`. **budget-AC avoids
   it** (and requires compile anyway). This exact shape is proven by diagnostic arm P.

## Infra notes
- **Cordoned nodes** (`SXid error code 12028`) killed 3 jobs today. Always check `canceledFor`
  before assuming a config bug; relaunch is the fix.
- gantry `launch(follow=True)` blocks; a 2-min client timeout does NOT kill the job. Check the API
  before relaunching or you double-submit.
- JIT caches must be container-local — concurrent arms sharing one cache dir deadlock on the lock.

## Next iterations
1. Pilot A0 eval → validate chain.
2. Launch remaining 9 evals in parallel.
3. `fit_law.py` → f1@32k(N_S) saturating fit; marginal value, saturation point, Row A vs B
   (substitute vs complement), Row C (equal-cost vs uniform).
4. Refine: add S/L points where the curve is steep, extend past saturation, or add a third
   long-pool level.
