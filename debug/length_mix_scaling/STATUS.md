# Length-mix scaling law — STATUS

**Updated 2026-07-24 22:30 PDT.** Autonomous overnight loop (dynamic self-pacing).
Scope per user: **hybrid Qwen3.5-4B only** (no dense variant).

## ★ ROUND 1 RESULT (2026-07-25 00:05) — an interior optimum, and a length-specific collapse

f1@32k, vLLM, eval_size 500 (binomial SE ±0.022 at f1≈0.5, ±0.019 at ≈0.25):

| arm | short tok | 2k | 8k | **32k** |
|---|---|---|---|---|
| A0 | 0 | 0.000 | 0.000 | **0.000** |
| A1 | 20.0M | 0.912 | 0.739 | **0.452** |
| A2 | 41.5M | 0.929 | 0.794 | **0.512** |
| A3 | 84.5M | 0.921 | 0.794 | **0.514** |
| A4 | 148.7M | 0.921 | 0.796 | **0.249** |
| C4 | uniform (=A4 budget) | 0.934 | 0.803 | **0.447** |

**Findings:**
1. **Short data is necessary.** A0 (long-only, 1500 ex, 67 steps) scores 0.000 at *every* rung with
   parse_rate 1.0 — it learned the output format but not the task. Genuine undertraining, not a
   harness bug: index conventions and converter args were verified, and C4 on the identical harness
   scores 0.934/0.803/0.447.
2. **There is an interior optimum at S/L ≈ 1–2** (A2 0.512, A3 0.514 — tied within SE), NOT a
   monotone saturation. `fit_law.py`'s saturating-exponential form does **not** fit (SSE 0.047);
   report argmax + degradation instead.
3. **Overshooting destroys long-context ability.** A3→A4 (84.5M→148.7M short) drops 0.514→0.249,
   ~9 SE. And it is **length-specific**: A4's 2k/8k are as good as any arm (0.921/0.796); only 32k
   collapses. Every Row A arm sees the identical long pool exactly once (351 steps × 524k tok =
   184M = 1 epoch), so the sole variable is the short data piled on top → clean interference.
4. **Equal-cost vs uniform:** A4 (0.249) LOSES to C4 (0.447); A3 (0.514) BEATS C4 (0.447). C3 (the
   properly matched control for A3) still running.

**Still running:** B0/B2/B4 (substitute-vs-complement) and C3.

## The question
Holding the long-context training pool **fixed**, does adding progressively more short-context data
improve **f1@32k**? And does that beat the uniform production mix at **equal token cost**?

## Baseline (do not re-measure)
`ctc-s5-contra-full-4b`, uniform 2k–32k mix, **vLLM**: 2k 0.849 / 8k 0.690 / **32k 0.335**.
⚠ vLLM ONLY. The native backend reads 0.571/0.219/0.038 on the *same* checkpoint — a degraded
harness, not a model property. Mixing backends fabricates results.

## Round 1 — TRAINING COMPLETE (10/10)

| arm | long tok | short tok | total | steps | train CE (last10) |
|---|---|---|---|---|---|
| A0 | 35.2M | 0 | 35.2M | 67 | 1.0807 |
| A1 | 35.2M | 20.0M | 55.2M | 105 | 0.0425 |
| A2 | 35.2M | 41.5M | 76.7M | 146 | 0.0170 |
| A3 | 35.2M | 84.5M | 119.7M | 228 | 0.0095 |
| A4 | 35.2M | 148.7M | 183.9M | 351 | 0.0019 |
| B0 | 20.2M | 0 | 20.2M | 39 | 1.0729 |
| B2 | 20.2M | 21.5M | 41.7M | 80 | 0.4833 |
| B4 | 20.2M | 84.5M | 104.7M | 200 | 0.0085 |
| C3 | uniform | — | 119.7M (=A3) | 228 | 0.0238 |
| C4 | uniform | — | 182.9M (≈A4) | 349 | 0.0119 |

⚠ **Train CE is NOT comparable across arms** and must not be reported as a result. Each arm's CE is
measured on its *own* data distribution — short-heavy arms are dominated by easier short examples —
and arms differ in optimizer-step count by design (67 vs 351). The only comparable numbers are
**f1 at fixed eval rungs**.

## Eval chain: what the pilot has proven so far (5 attempts, 5 distinct gaps)
Piloting ONE arm before the 10-way fan-out has paid for itself repeatedly — every failure below
would have been 10 simultaneous failures. Each was strictly further along the chain:

| # | failure | fix |
|---|---|---|
| 1 | `No module named dataclass_extensions` | gantry `--no-python` skips `pip install -e .` → install the package for the export step |
| 2 | torch pin drift (2.13.0 vs required 2.11.0) | `PIP_CONSTRAINT` + explicit triad pin |
| 3 | `no step<N> checkpoint dir` | CKPT was built from the Beaker **experiment name** (`<run>-<hash>`); the save folder is just `<run>` → resolve by glob, use post-fit `model_and_optim` |
| 4 | `assert has_fla()` | GDN blocks need `flash-linear-attention` to *export* (vLLM inference does not — why the load recipe never hit it) |
| 5 | `model type qwen3_5` unknown | container transformers is 4.57.x → 5.14.1 **shadow** dir, scoped to export only |
| 6 | torch cu130 vs torchvision cu129 | version pin ignored the CUDA **tag**; pin explicit `+cu129` on all three and assert on tags |

**NOW WORKING:** export ✓, serving copy ✓, prefills ✓ (500 rows/rung), venv triad coherent
(`torch/torchvision/torchaudio 2.11.0+cu129`, `vllm import OK`). Untested: vLLM generation itself.

⚠ **A job graded 0/3 rungs and still exited 0** — the per-rung `continue` masked total failure.
Fixed: the script now tracks graded rungs and exits 5 if none succeeded. A silently-empty success
would have poisoned the fan-out with missing data that looked fine.

⚠ **32k rung prompts are much longer than the rung name suggests**: min/mean/max =
**31k / 59k / 214k** tokens. `run_vllm_eval` auto-raises `max_model_len`, but a 214k prompt on one
H100 is tight — that rung may need tensor-parallel or a bigger GPU even though 2k/8k are fine.

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
