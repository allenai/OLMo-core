# Qwen3 vs Qwen3.5 contradiction — 128k run — HANDOFF

**Last updated 2026-07-24 ~13:30 PDT** (supersedes the 2026-07-24 ~13:04 version)

## Goal
Compare **Qwen3-4B (dense)** vs **Qwen3.5-4B (GDN-hybrid)** on `contradiction`, full-attention SFT,
**long context**. Comparison axis = long-context capability. Branch `prasann/landmark`.
Work dir: `debug/qwen3_vs_qwen35_contra/`. Memory: `qwen3-vs-qwen35-256k-contra.md`.

Target is **128k** (changed from 256k). Dense needs only **factor-4** rope extension (32k→128k);
hybrid is native 262k so **no extension needed**.

## STATUS AT A GLANCE
| Piece | State |
|---|---|
| 128k data (both tokenizers) | DONE — S3 + weka + /scratch |
| weka sync | **DONE** (`WEKA_SYNC_128K_DONE`, 257 files per shard) |
| iso 64k NTK diagnostic | **VERDICT: extension learns** (caveat below) |
| Dense @128k (Beaker) | **LAUNCHED**, queued for scheduling — exp `01KYAWE23EE864R2XQX5NW4542` |
| Hybrid @128k (local) | **TRAINING** — job 3353323, CE 1.46→0.94, MFU 96%, 664 steps (~5.9h) |
| 128k eval rung | **BUILT + STAGED** (eval_size 500, n=2503) |
| Eval script | **WRITTEN + syntax-checked** — `eval_128k.sbatch` (both arms) |
| Eval run | NOT STARTED (gated on checkpoints) |

## Data (128k) — DONE
- Uniform mix `n~U[175,2700]` (~8k–125k tok), 10k examples, 0 dropped, max_example_len ~127k.
- S3: `s3://ai2-llm/checkpoints/prasanns/ctc_suite/shards/contra_mix_{qwen3,qwen35}_10k_128k`
- weka: `/weka/oe-training-default/ai2-llm/checkpoints/prasanns/ctc_suite/shards/contra_mix_{qwen3,qwen35}_10k_128k`
  — sync job `01KYAV93JTR4ZW3NRY962JDEE2` finished exit 0, **257 files each, verified** (not the
  silent-empty failure mode from `eval-bundle-weka-staging`).
- Hybrid shard also on /scratch: `/scratch/users/prasann/ctc_qwen_compare/contra_mix_qwen35_10k_128k`

## LIVE JOBS

### 1. Hybrid @128k — **job 3353323**, sneetches (RELAUNCH)
Config: TASK=contradiction VARIANT=full SCALE=4b MODEL_FAMILY=qwen3_5 SEQ_LEN=131072 EPOCHS=1
LR=5e-5 GLOBAL_BATCH=8 MICRO_BATCH=1 **CP_DEGREE=4** PACK=1 **ACT_CKPT=full** (no-compile, no rope ext).
8×H200, `--time=24:00:00`, qos preemptive_high. wandb group `q3-vs-q35-contra-128k`.
Log: `/scratch/users/prasann/ctc_suite_logs/train_q35-4b-contra-128k-local_3353323.log`

**CONFIRMED TRAINING** (20:35 UTC): passed the 128k forward/backward dry-run with **no OOM**, so
full-AC + no-compile + CP=4 is the right memory recipe at 128k. CE 1.461 → 1.295 → 0.9396,
**MFU 96.1%**, 4,116 TPS/device, **664 steps** total (10k examples packed into 5,319 sequences),
≈32 s/step → **~5.9 h** projected, well inside the 24 h limit.

**Why relaunched:** the previous attempt (3353237) died at 20:12 UTC with
`OLMoEnvironmentError: missing env var 'WANDB_API_KEY'` — see the new trap below. Its save folder
held only `provenance.json` (no step dir), so there was **no auto-resume risk** in reusing the run
name. Config verified identical via `provenance.json` before relaunch.

Relaunch command (works as-is):
```bash
export WANDB_API_KEY=$(awk '/machine[[:space:]]+api\.wandb\.ai/{f=1} f&&/password/{print $2; exit}' ~/.netrc)
sbatch --job-name=q35-4b-contra-128k-local --partition=jsteinhardt --qos=preemptive_high \
  --nodelist=sneetches --nodes=1 --gres=gpu:8 --cpus-per-task=32 --mem=256G --time=24:00:00 \
  --output=/scratch/users/prasann/ctc_suite_logs/train_%x_%j.log \
  --export=ALL,TASK=contradiction,VARIANT=full,SCALE=4b,MODEL_FAMILY=qwen3_5,SEQ_LEN=131072,\
EPOCHS=1,LR=5e-5,GLOBAL_BATCH=8,MICRO_BATCH=1,CP_DEGREE=4,PACK=1,ACT_CKPT=full,\
DATA_SRC=/scratch/users/prasann/ctc_qwen_compare/contra_mix_qwen35_10k_128k,\
BASE_SRC=/scratch/users/prasann/ctc_suite_lambda_stage/q35-4b-base-modelonly,\
RUN=q35-4b-contra-128k-local,WANDB_GROUP=q3-vs-q35-contra-128k,WANDB_API_KEY=$WANDB_API_KEY \
  src/scripts/train/memexpress/ctc_suite/run_ctc_local.sbatch
```

### 2. Dense @128k — **Beaker exp `01KYAWE23EE864R2XQX5NW4542`** (LAUNCHED 13:17 PDT)
Name `q3-4b-contra-128kmix-20260724T131752-0700-d92cbce5`, 2 nodes jupiter, urgent, preemptible.
**Uses `--rope-theta 4e6` (NTK), NOT `--rope-yarn-factor`** — see the RoPE section below.
wandb group `q3-vs-q35-contra-128k` (matched to the hybrid for a shared view).
```bash
python -u src/scripts/train/memexpress/ctc_suite/beaker_ctc_suite.py --task contradiction \
  --variant full --model-scale 4b --model-family qwen3 --run-name q3-4b-contra-128kmix \
  --wandb-group q3-vs-q35-contra-128k \
  --data-root /weka/.../ctc_suite/shards/contra_mix_qwen3_10k_128k \
  --num-nodes 2 --epochs 1 --seq-len 131072 --cp-degree 8 --rope-theta 4e6 --pack \
  --activation-checkpointing budget --ac-budget 0.3 --global-batch 8 launch
```
⚠ The launcher has `follow=True`, so `launch` **blocks streaming logs** and a 2-min tool timeout
kills the client, NOT the job. Do **not** re-run it on timeout — check the API for the experiment
first or you will double-submit:
```bash
TOK=$(python -c "import yaml,os;print(yaml.safe_load(open(os.path.expanduser('~/.beaker/config.yml')))['user_token'])")
curl -s -H "Authorization: Bearer $TOK" \
  "https://beaker.org/api/v3/workspaces/ai2%2Fflex2/experiments?limit=8&sortBy=created&order=descending"
```
(`beaker experiment list --workspace` is not a valid flag, and `beaker workspace experiments` times
out — the raw API is the reliable path.)

### 3. Iso 64k — job 3353222, horton — the NTK diagnostic. Still running, keep for more signal.

## ✅ RESOLVED (2026-07-24 17:00) — the dense plateau is the NON-NATIVE CONTEXT LENGTH

User's hypothesis, confirmed. Both of my hypotheses (compile+budget-AC, then CP degree) are **dead**.

| run | length | vs native | CP | config | result |
|---|---|---|---|---|---|
| **K** | 32k | **1× native** | 4 | full + no-compile | ✓ **0.578** |
| iso | 64k | 2× | 2 | full + no-compile | ✓ 0.277 @694 steps |
| P | 64k | 2× | 4 | budget + compile | ✓ 0.656 |
| G | 64k | 2× | 4 | full + no-compile | ✗ 1.07 |
| **J** | 128k | **4×** | **4** | budget + compile | ✗ **1.089** |
| dense 128k | 128k | 4× | 8 | budget + compile | ✗ 1.09 (cancelled) |
| dense 256k | 256k | 8× | 8 | YaRN, budget+compile | ✗ ~1.1 |

- **J** (`01KYB4C1XT1WWQ4BY8RGBZ0P8N`) plateaus at **1.089 with CP=4** — identical to CP=8 at the
  same token count ⇒ **CP exonerated**. (J ran 2 replicas: its 200 log lines are 100 real steps
  logged twice. Don't misread the repeat as oscillation.)
- **K** (`01KYB4NQRXZ272670709XZSAX3`) descends to **0.578** using the supposedly-broken config
  (full-AC + no-compile, CP=4) at native 32k, no rope extension ⇒ **config exonerated**.

**Conclusion: a length threshold.** Native 32k fine; 2× native mostly fine; **4× native fails under
every configuration tried**. NTK θ=4e6 does not successfully extend Qwen3-4B to 128k for this task.

**Unexplained residue:** G failed at 2× native while iso and P succeeded at the same 2×. At 2× the
outcome is config-sensitive; at 4× nothing works. A pure-length story does not cover G.

**Implication for the comparison:** the headline is now that the **GDN hybrid trains natively at
128k (CE 0.150) while the dense model cannot get there via NTK extension**. Before reporting that
as a modeling result, try **progressive/staged extension** (continue from the healthy 64k iso
checkpoint into 128k rather than jumping base→128k) — the standard long-context recipe, and the
obvious remaining candidate for making the dense arm work.

### Superseded hypotheses (kept so they are not re-litigated)
1. ~~compile + budget-AC causes the plateau~~ — REFUTED: it is the *healthiest* config (P → 0.656).
   The plan notes' "no-compile avoids the full-AC+CP+compile mismatch" guidance is **not** why these
   runs failed.
2. ~~CP degree ≥4 is the culprit~~ — REFUTED by J (CP=4 plateaus exactly like CP=8).

## Earlier framing (superseded by the above)

Dense 128k vs hybrid 128k on a **matched-token** axis (same task, same uniform-mix design, same
`global_batch=8`, same 1 epoch — much cleaner than the 64k-iso comparison):

| tokens | dense q3 (NTK 4e6) | hybrid q3.5 (native) |
|---|---|---|
| 10M | 1.911 | 0.671 |
| 20M | 1.503 | 0.523 |
| 40M | 1.146 | 0.505 |
| 60M | **1.090** | 0.385 |
| 80M | **1.084** | 0.321 |
| 88M | **1.097** | 0.288 |

Dense is **flat at ~1.09 over 28M tokens** — the same value the 256k YaRN run plateaued at.

**The confound.** Across all four long-context runs, "big rope extension" and
"compile + budget-AC + CP=8" are *perfectly* correlated:

| run | ext | method | CP | AC | compile | outcome |
|---|---|---|---|---|---|---|
| iso 64k | ×2 | NTK 2e6 | 2 | full | **no** | descends → 0.686 ✓ |
| hybrid 128k | native | — | 4 | full | **no** | descends → 0.288 ✓ |
| dense 128k | ×4 | NTK 4e6 | 8 | budget 0.3 | **yes** | flat ~1.09 ✗ |
| dense 256k | ×8 | YaRN | 8 | budget 0.3 | **yes** | flat ~1.1 ✗ |

Two arguments favor the **config** explanation over the rope explanation:
1. the hybrid learns fine at the *same* 128k context, so 128k itself isn't the problem;
2. factor-4 and factor-8 plateau at the *same* value — if extension severity were the cause,
   factor-8 should be clearly worse than factor-4.

Prior suspicion supports it: the hybrid was deliberately put on no-compile to dodge a
"full-AC + CP + compile recompute-metadata mismatch", and **budget AC requires compile**, so the
dense arm was forced onto the suspect path.

⚠ Not established — a 0.95 "plateau" on the iso earlier the same day turned out to be a shoulder
that broke through ~40 steps later. Dense has been flat ~28 steps.

### ✅ DIAGNOSTIC RESULT (2026-07-24 15:25 PDT) — hypothesis REFUTED, suspect is now **CP degree**

| steps | P: budget-AC + **COMPILE** (CP=4) | G: full-AC + **NO compile** (CP=4) | local iso: full-AC + no compile (**CP=2**) |
|---|---|---|---|
| 1–40 | 1.672 | 1.718 | 1.650 |
| 41–80 | 0.932 | 1.205 | 1.082 |
| 81–120 | 0.774 | **1.079** | 0.843 |
| 121–160 | 0.732 | **1.072** | 0.688 |
| 161–200 | **0.656** | **1.073** | 0.702 |

**compile + budget-AC is NOT the bug — it is the HEALTHY config** (P reaches 0.656, matching the
known-good iso). The *control* plateaued. This is the opposite of both my hypothesis and the plan
notes' "no-compile avoids the full-AC+CP+compile mismatch" guidance, which should be treated as
**wrong/obsolete** for the dense arm.

**What this bought:**
1. A **40-minute reproduction** of the ~1.09 plateau (G plateaus at 1.07 ≈ dense128k 1.09 ≈
   dense256k ~1.1) — iterate at 64k instead of hours at 128k.
2. **CP degree is the new prime suspect**, not rope and not compile:
   CP=2 ✓ · CP=4 works *only* with budget+compile · CP=8 fails even with budget+compile.
   Reads as "CP is subtly wrong for dense Qwen3 at degree ≥4; budget+compile dodges it at 4, not 8."
3. Rope extension is **exonerated** — every arm here used θ=2e6, the same setting that descends.

**Still open:** the dense 128k plateau is NOT explained — it ran arm P's config, which works at
CP=4. Remaining suspects: CP=8, seq_len 131072, extension factor ×4.

**Confound to close:** G vs the local iso differ in **both** CP (4 vs 2) **and** environment
(Beaker H100 vs local H200). "CP=4 breaks it" is not yet established over "Beaker breaks it."

**Proposed next two runs (~3 nodes, ~40 min):**
- **H:** CP=2, full-AC, no-compile, on Beaker → separates CP from environment. Descends ⇒ env
  exonerated, CP confirmed.
- **I:** CP=8, budget-AC, compile, on Beaker → does CP=8 break the *good* config? Plateaus ⇒
  dense 128k explained.

**Dense 128k CANCELLED** 2026-07-24 22:31 UTC (user call) after plateauing **1.08–1.09 from 60M to
148M tokens** — an 88M-token flat stretch, definitively not a shoulder. Full loss curve preserved
at `debug/qwen3_vs_qwen35_contra/curves/dense128k_curve.json` (281 points; no checkpoint existed,
mid-run checkpointing is off).

### The diagnostic as launched (2026-07-24 ~15:05 PDT)
Matched Beaker pair at 64k on the iso data, θ=2e6 (the setting *known* to descend to 0.686),
identical in everything but compile+AC. Run **in Beaker for both arms** so there is no
local-vs-Beaker environment confound, and both at **CP=4** so the result attributes to compile+AC
rather than to a three-variable package.

| arm | exp id | AC | compile |
|---|---|---|---|
| P (suspect) | `01KYB2FTVN56CXXWZMCW4KR1GY` | budget 0.3 | **on** |
| G (control) | `01KYB2H2PZRBPHSKCWQR554ZZC` | full | **off** |

Both verified from their logs (P: "Applied 'budget'" + "Applied torch.compile()"; G: "Applied
'full'", no compile line). Reference from the known-good iso: step 60 ≈ 0.95, 120 ≈ 0.74, 160 ≈ 0.69.

**If P plateaus at ~1.1 while G descends → rope exonerated, compile+budget-AC is the bug**, the
"extension plateaus at high factors" conclusion is wrong, and BOTH dense long-context runs must be
relaunched on the frugal (full-AC, no-compile) path.

Data staged for it: `s3://…/ctc_suite/shards/contra_iso64k_qwen3_2k` → weka (sync exp
`01KYB2DFZPV5XQ92TTPZ4NS7HV`, 143 MB, 8 s).

⚠ **`beaker_ctc_suite.py` had NO compile flag** — the trainer compiles by DEFAULT, so *every*
Beaker CTC run ever launched has been compiled with no way to opt out. Added a `--no-compile`
passthrough (uncommitted).

⚠ **Do not run multi-GPU on `lorax`** — `CUDA driver initialization failed`; the local launcher's
own comments record this ("lorax does not index-isolate like thidwick"). Cost one dead job today.

## RoPE extension: use `--rope-theta`, not YaRN
Commit `1367af0` added `--rope-theta` (NTK) and the trainer docstring says to **prefer it over
`--rope-yarn-factor` for large extensions** (YaRN writes per-layer `block_overrides`; theta is a
plain factory kwarg that unambiguously takes effect). The old handoff's YaRN-factor-4 command is
**stale** — the 256k YaRN run is what plateaued at ~1.1.

NTK-aware θ for Qwen3-4B (base θ=1e6, head_dim=128): `θ' = 1e6 · s^(128/126)`
→ **s=2 (64k) → 2e6 · s=4 (128k) → 4e6 · s=8 (256k) → 8e6**. Consistent with the values in use.

## Iso 64k verdict — extension LEARNS (confirmed; the early plateau was a shoulder)
Gate was "CE < 0.7". It first fired on a single noisy step (min 0.6970 at step 99) while the
windowed mean was still ~0.95, which looked marginal. **By step 161 the windowed mean itself is
below the gate**, so the verdict is now solid rather than borderline:

| steps | 1–20 | 21–40 | 41–60 | 61–80 | 81–100 | 101–120 | 121–140 | 141–160 |
|---|---|---|---|---|---|---|---|---|
| mean CE | 1.962 | 1.337 | 1.212 | 0.952 | 0.947 | 0.738 | 0.689 | **0.686** |

The 0.952/0.947 pair was an **SFT shoulder, not a ceiling** — descent resumed immediately after.
Final read: NTK θ extension learns at 64k, comfortably below the 256k YaRN plateau of ~1.1.
(Lesson: do not call a plateau off two 20-step windows.)

**The extension is definitely APPLIED** (this was the thing the iso existed to rule out) — verified
directly from `provenance.json`, not inferred: `rope_theta: 2000000.0`, `rope_yarn_factor: 0.0`.
Reading `provenance.json` is the general way to confirm a run's real config:
```bash
srun --overlap --jobid=<JOBID> cat /data/prasann/ctc_suite/ckpts/<RUN>/provenance.json
```

## 128k eval rung — BUILT
`src/corpus_reasoning/data/build_xlong_rungs.py` handles this (needs `PYTHONPATH` to include
`src`, `src/corpus_reasoning/data`, **and `src/scripts`** — the last one is missing from the
obvious invocation and gives `ModuleNotFoundError: ctc_eval`).

- Calibration: contra 128k → **n=2503 docs** (52.37 eff tok/doc). In-distribution vs the training
  mix's `n~U[175,2700]`. ✔
- Built at **`--count 500`**, not the 300 default — sub-500 would have to be flagged inline per
  `eval-size-and-error-bars`. Binomial SE at 500 is ±0.021 around f1≈0.7.
- File: `/scratch/users/prasann/cpt_data/eval500_v2/contra/contradiction_eval_pubmed_both_n2503_k3_xlong_128k.jsonl`
- **Staged** to `/scratch/users/prasann/ctc_suite_staged/eval_rungs/contradiction/rung_131072.jsonl` (252 MB)
- Validated structurally identical to the trusted 32k rung: 500 rows, 3 gold PAIRS/example, every
  pair length 2, **0 out-of-range gold indices**. (`queries`/`answers` empty is normal for
  contradiction — don't mistake it for a broken build.)

## Eval script — `eval_128k.sbatch` (written, syntax-checked, NOT yet run)
`debug/qwen3_vs_qwen35_contra/eval_128k.sbatch` — the 128k extension of `eval_2k.sbatch`, handling
**both** arms via `MODEL=qwen3|qwen35`. Chain: export → (qwen3.5 only) VL serving copy → prefills →
vLLM generate → grade. It bakes in the traps found today:
- passes the right `ROPE_ARGS` per arm (`--rope-theta 4e6` for dense, context bound for both);
- **hard-fails** if the exported config doesn't carry `max_position_embeddings ≥ 131072` / dense
  `rope_theta == 4e6` — the silent-corruption guard, not a formality;
- re-stamps `max_position_embeddings` on the qwen3.5 **serving copy**, because
  `make_vllm_serving_copy.py` rebuilds the wrapper config from the BASE snapshot and would
  otherwise undo the override;
- node-local `HOME`/flashinfer/triton caches (`flashinfer-cache-HOME-NFS-wedge`);
- prints eval_size + binomial SE and warns when `parse_rate < 0.5` (dump generations first).

**Run the timing probe before the full 500** (`MAXSAMP=20 TIMEONLY=1`) — 500×128k is ~64M prefill
tokens and cost is not linear in rung. Requires the node-local vLLM venv `/data/prasann/ctc_vllm_venv`
(present on **horton** and **sneetches**).

### vLLM eval needed a dense code path (added)
`run_vllm_eval.py` hardcoded `hf_overrides={"architectures":["Qwen3_5ForCausalLM"]}` +
`limit_mm_per_prompt` — the recipe for serving the **multimodal Qwen3.5 wrapper** as text-only. A
plain dense Qwen3 export has no VL wrapper, so those would name an architecture its config doesn't
declare. (`run_vllm_eval_generic.py` is no help — it forces `gdn_prefill_backend="triton"`, which
needs fla + causal_conv1d, absent in that venv.) Added `--model-family qwen3_5|qwen3` (default
`qwen3_5`, so **existing behavior is byte-identical**) plus `--tensor-parallel-size`. Verified by
AST that the qwen3.5 path still gets both overrides and `LLM()` has no duplicate kwargs.

## ⚠ NEW TRAP #1 — `export_olmo_to_hf.py` silently DROPS the trained rope_theta (FIXED, uncommitted)
`export_olmo_to_hf.py` built its HF config with `AutoConfig.from_pretrained(args.base_model)` —
verbatim from the **base**. So a run trained with NTK `--rope-theta 4e6` would export a config
saying `rope_theta=1000000, max_position_embeddings=32768`, and the rotary `inv_freq` buffer is
recomputed from that stale θ at `from_config()` time. The dense 128k checkpoint would then eval
with **RoPE that does not match what it trained with**, and vLLM would refuse/truncate past 32k.
That failure reads exactly like "dense collapses at long context" — a fake modeling result.

**Fixed** (working tree, not committed): added `--rope-theta` / `--max-position-embeddings`
overrides + `apply_rope_overrides()`, applied on **both** the dense and the qwen3_5 hybrid path,
before model instantiation. **When exporting the dense 128k checkpoint you MUST pass:**
```bash
--rope-theta 4e6 --max-position-embeddings 131072
```
The hybrid needs no θ override (native), but does want `--max-position-embeddings 131072`.

## ⚠ NEW TRAP #2 — missing WANDB_API_KEY kills a local run ~10 min in, looking healthy (FIXED)
olmo-core hard-requires the `WANDB_API_KEY` **env var**. `wandb login` writes the key to
`~/.netrc` but not the env, so a launch from a shell that never exported it dies with
`OLMoEnvironmentError: missing env var 'WANDB_API_KEY'`. What makes it nasty:
- it raises only **after** the base load + mesh build (~10 min at 128k), so it looks like a slow load;
- only **rank 0** raises — the other 7 ranks spin at **100% GPU** in the NCCL collective;
- **SLURM still reports RUNNING**, and the log keeps its mtime fresh.

So the node looks fully busy while doing nothing. This burned ~15 min of 8×H200.

**Fixed** in `run_ctc_local.sbatch` (working tree, not committed): backfill `WANDB_API_KEY` from
`~/.netrc` when unset, else **fail fast at submit** with a clear message (`NOWANDB=1` to opt out).

**Monitoring lesson:** the watcher that missed this grepped only for
`out of memory|Traceback|recompute|CUDA error|Killed` — `OLMoEnvironmentError` and
`Training failed due to` were not covered, and squeue said RUNNING, so silence read as health.
Any watcher on these runs must grep `Training failed due to|OLMoEnvironmentError|FATAL:` too.

## ✅ HYBRID 128k COMPLETED (2026-07-24 19:29 PDT)
`q35-4b-contra-128k-local` (slurm 3353323) finished **664/664 steps, final CE 0.0146** (min 0.0031).
Qwen3.5-4B GDN hybrid, seq_len 131072, CP=4, full-AC, no-compile, native RoPE (no extension),
gb=8, 1 node × 8×H200, ~5.5 h. Curve: `curves/hybrid128k_curve.json` (664 points).
Checkpoints on **sneetches** `/data/prasann/ctc_suite/ckpts/q35-4b-contra-128k-local/`:
`config.json` + `model_and_optim` (model-only export for eval) + `step250/500/664`.

This is the one completed arm of the comparison: **the hybrid trains to convergence at 128k
natively, while the dense model cannot reach 4× native by NTK extension at all.**

## ✅ CORRECTION — these runs DO write mid-run checkpoints (every 250 steps)
The claim below (carried from the earlier handoff) that "these runs have no mid-run checkpointing,
so a timeout loses everything" is **WRONG**. Verified empirically on both local runs:
- hybrid 128k → `step250`, `step500`, `step664`
- iso 64k (timed out at step 908) → `step250`, `step500`, **`step750`**, all with valid `.metadata`

So the iso's death by TIMEOUT did **not** lose its progress: `step750`
(`/data/prasann/ctc_suite/ckpts/q3-iso64k/step750/model_and_optim` on **horton**) is a healthy
64k-adapted checkpoint at θ=2e6, CE ≈ 0.25. **Progressive extension is available right now** —
continue from it into 128k with θ=4e6, no 64k rebuild needed.

(The `--time=24:00:00` advice still stands — a timeout costs you the tail of the run, just not
everything.)

## KEY TRAPS (carried forward)
- `run_ctc_local.sbatch` default `--time=06:00:00` killed the 256k hybrid and the 64k iso mid-run —
  ALWAYS pass `--time=24:00:00`. (Loses the tail, not the whole run; see the correction above.)
- Auto-restarting CTC jobs (screen `ctc`/`ctc2`) grab the jsteinhardt 8-GPU cap the instant it
  frees → cancel and resubmit **back-to-back** in one command, never as two steps.
- Data staging: `run_ctc_local` does `cp -un DATA_SRC/* node-local`; DATA_SRC must be COMPLETE on a
  shared FS. /scratch `du` under-reports (compression) — verify by S3 object count.
- Beaker: gantry clones the **pushed** commit; `--allow-dirty` ships nothing.
- 2k sanity PASSED: hybrid f1=0.865, dense f1=0.843 (eval_size 500) — pipeline is sound.

## NEXT STEPS
1. **Confirm hybrid 3353323 reaches a CE step** (watcher armed with full failure coverage).
2. **Confirm dense Beaker exp starts** — it was still `created` (scheduling) at last check.
3. Let iso 64k run further; if the mean breaks below ~0.85 the extension story is strong, if it
   sits at 0.95 for another few hundred steps, note it as a real limit of factor-2 NTK.
4. **On checkpoints:** export with the rope flags above, then eval at 128k via vLLM
   (`qwen35-4b-vllm-load-recipe`; native bs=1 too slow). Rung is staged at `rung_131072.jsonl`.
   Both arms are `full` attention, so the chunked-vLLM FlexAttention bug does not apply.
5. Commit the two fixes (`export_olmo_to_hf.py`, `run_ctc_local.sbatch`) — currently working-tree only.

## UNCOMMITTED CHANGES IN THIS SESSION
- `src/corpus_reasoning/train/export_olmo_to_hf.py` — rope_theta/max_position_embeddings overrides.
  **Verified**: the override propagates into the rotary `inv_freq` buffer with exactly the NTK
  ratio `4^(2/128)=1.02190`, and the no-op path leaves base values untouched.
- `src/scripts/train/memexpress/ctc_suite/run_ctc_local.sbatch` — WANDB_API_KEY backfill + fail-fast
- `debug/ctc_vllm_validation/run_vllm_eval.py` — `--model-family` (dense path) + `--tensor-parallel-size`
- `debug/qwen3_vs_qwen35_contra/eval_128k.sbatch` — NEW, the 128k eval for both arms
