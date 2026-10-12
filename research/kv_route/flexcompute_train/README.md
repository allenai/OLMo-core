# Does the flexcompute stack deliver a real training speedup on a task it still has to learn?

`research/kv_route/flexcompute_40x/` prices the three routers at **pinned** operating points: it forces a token
distribution over FFN rungs, KV keeps and block skips and times the step. That answers "how fast
*could* this be", and it answers it well — 5.65x at 4B/16k plain, 3.90x on the same cell packed x32.
It does not answer whether a model **trained** with those routers, learning them from a keep-all
init under a budget hinge, ever reaches that operating point, or whether the task survives when it
does.

This directory runs that experiment: **real contradiction SFT on Qwen3.5-4B, a ladder of arms, a
dense control at both ends of the same allocation**, then scores every arm on held-out
contradiction.

**The two headlines:**

1. **The budget hinge does not deliver an operating point in a 31-step epoch.** Block skip and the
   KV router never leave their keep-all initialisation, so the "moderate" and "aggressive" arms are
   in practice the same arm and both land at **1.10-1.14x**. The pinned 5.65x/3.90x numbers are not
   what a short SFT run realizes.
2. **Forcing the operating point works, and the wall-clock is real — but contradiction breaks
   before the big speedups arrive.** Fixed-capacity block skip gives a measured **2.82x at keep 1/2,
   5.05x at 1/4, 7.39x at 1/8**, all with a verified 1.000x drift bound. Only keep 1/2 still trains;
   1/4 and 1/8 diverge outright (CE 55 and 61). **The best speedup that preserves task quality is
   1.15x** (see the frontier table for the f1 evidence).

---

## Setup (copied from the established contradiction SFT recipe, not invented)

| | value | source |
|---|---|---|
| model | Qwen3.5-4B (`--model-family qwen3_5 --model-scale 4b`), 32 blocks, GDN x3 + attention x1 | `research/ffn_compute/flop_scaling/launch_grid35.py` |
| base | `q35-4b-base-modelonly` on cubbins; marker audit PASS (cos 0.226, norm ratio 0.65 / 0.95) | `convert_qwen35_bases.sbatch` |
| task / data | contradiction, short-heavy 2k-32k mix at 16M tokens (45/27/16/8/4 % at 2k/4k/8k/16k/32k), `realistic` mode, `pubmed_perturbation` | `research/ffn_compute/flop_scaling/build_shortheavy_data.sbatch` |
| tokenization | re-tokenized for Qwen3.5 (`marker_set qwen3_5`), query AFTER, no-CoT | `tokenize_contra_q35.sbatch` (this dir) |
| length handling | `--seq-len 65536 --pack` (≈32 documents per row) | `taskscale_lengthmix/launch_arms.sh` (prior campaign, not ported) |
| optimizer | `--lr 5e-6`, `--global-batch 8 --micro-batch-instances 1`, 1 epoch = **31 steps** | same |
| parallelism | FSDP, `shard_degree = world`, activation checkpointing `full` (the 4b default) | `train_ctc_suite.py` |
| hardware | **4** x H200 on cubbins, one slurm allocation per ladder | 8-GPU/user cap, shared with a concurrent benchmark job |

The one deliberate departure from the reference recipe is 4 GPUs instead of 8; every arm uses the
same 4 GPUs in the same allocation, so arm-to-arm comparisons are unaffected and absolute step
seconds are not comparable to an 8-GPU campaign run. The data is the same short-heavy JSONL the
Qwen3 FLOP-scaling build produced — only the tokenizer changed, because that build only ever emitted
Qwen3 ids (151648/151649/151643) and this model needs Qwen3.5's (248049/248050/248044).

## Measurement discipline

- **Every arm of a ladder ran inside ONE slurm job**, sequentially, so no comparison inside a ladder
  crosses a job boundary. This box drifts up to ~2x on an unchanged configuration
  (`research/kv_route/flexcompute_40x/README.md`), which is larger than most of the effects here.
- The dense control ran **first and last**: `dense` 19.53 s/step, `dense2` 19.53 s/step, and their
  final CE agrees to 3 decimals (0.039 / 0.039). **Drift bound over the 75-minute ladder: 1.000x.**
  Every speedup below is quoted against `dense`.
- The capacity sweep is a second job. It is bridged to the first by a matched pair —
  `aggr-nokv` (17.14 s, job 3523714) and `capbase` (16.98 s, job 3523800), which differ only in
  targets that both routers ignored — giving a **cross-job drift of 1.009x**. The cap arms'
  speedups are quoted against `dense` through that link and carry that 0.9% on top.
- Step seconds come from `throughput/device/BPS` (SpeedMonitor's own timer, which discards the first
  step), median over steps > 5, with p10/p90. `collect.py` cross-checks them against the wall-clock
  gaps between consecutive console-logger lines; the two agree to **1.000x** on every arm.
- FLOPs are `flop_meter/actual_pflops` from each run's `flops.json` — the method-aware meter, **not**
  `throughput/total petaflops`, which charges every routed arm the dense cost.

## Files

| file | what it does |
|---|---|
| `tokenize_contra_q35.sbatch` | Re-tokenizes the short-heavy contradiction JSONL arm for Qwen3.5. |
| `run_ladder.sbatch` | The ladder: every arm in `ARMS` trained back-to-back in one job on one allocation. |
| `eval_arms.sbatch` | Scores the trained checkpoints on `contradiction_iid` (500 examples per rung), routers re-enabled from each checkpoint's own `config.json`. |
| `collect.py` | Ladder logs -> the frontier table (step seconds, speedup, FLOP cut, conversion, realized router operating points, CE curve, f1). |
| `LAUNCH_LEDGER.tsv` | Every job launched from here, including the three that were no-ops and why. |

wandb group: `flexcompute-train-g1` —
https://wandb.ai/prasann-uc-berkeley-electrical-engineering-computer-sciences/memory-networks/groups/flexcompute-train-g1

---

## Finding 1: the budget hinge is a no-op at this run length

Two of the three routers never move. This is not a tuning miss, it is the initialisation:

| router | asked for | **realized at step 30** | why |
|---|---|---|---|
| block skip | 0.500 / 0.125 | **1.000 / 0.999** | sigmoid gate initialised at bias **+10** (`block_skip.py: KEEP_INIT_BIAS = 10.0`), where `d sigmoid/dz = 4.5e-5`. Over 31 steps block 0 moved 1.00 -> 0.98 and the other 31 blocks did not move at all. |
| KV keep | 0.500 / 0.250 | **1.000** | same `+10` keep-all init (`kv_route.py`) |
| FFN width | 0.100 / 0.010 | **0.058 / 0.059** = a 17x FFN cut | an argmax over 6 rungs, no saturating gate — it converges, in ~25 steps |

The keep-all init is deliberate and correct (an enabled-but-untrained model must reproduce its
base). What this measures is that the hinge cannot then *escape* it inside one epoch: to cross
sigmoid's midpoint the bias has to travel 10 logits, and under Adam at `--router-lr 1e-3` that is
~10^4 steps. The FFN router has no such barrier and does what it is supposed to.

Consequences worth carrying:

- **"moderate" and "aggressive" are the same arm.** The FFN router converged to mean cost 0.058 vs
  0.059 whether it was asked for 0.10 or 0.01; the other two routers did nothing in both. Their
  step times (17.51 / 17.76 s) differ by less than the p10-p90 spread.
- **`flops.json`'s `*_frac` fields are RUN AVERAGES, not operating points.** `ffn_cost_frac` reads
  0.43 because the first ~15 steps are still near keep-all; the end-of-run cost is 0.058. A longer
  run would show a much larger FLOP cut on the FFN axis than the 1.20x here.
- The routers are not free while they do nothing: they cost their prologue every block. That the
  arms are still 1.10-1.14x *faster* is entirely the FFN router's late-run 17x.

## Finding 2: forced capacity works, and converts better than 1:1

`--block-skip-capacity` replaces the threshold rule with a per-block top-k, so the keep fraction is
met exactly by construction. It binds from call 0 and **bypasses the anneal** — `block_skip.py` says
so — which is a real cost, priced below.

The capacity arms are the only ones that reach an operating point, and their wall-clock falls
**faster than their counted FLOPs** (conversion 1.06-1.17): compaction also shortens the GDN
sequential scan and removes kernel launches, and the FLOP meter credits neither.

## The frontier

`dense` is the control for every row. `job` is the slurm id; the two jobs are bridged as described
above. `FFN cost` / `block keep` / `KV keep` are the FLOP meter's **run averages**; the realized
end-of-run operating points are listed underneath.

<!--FRONTIER-->

Regenerate with (the logs are node-local on cubbins; `/net` is fine for reading them):

```
python research/kv_route/flexcompute_train/collect.py \
  /net/cubbins/data/prasann/joblogs/fxtrain_fxtrain_3523714.log \
  /net/cubbins/data/prasann/joblogs/fxtrain_fxtrain_3523800.log \
  --root /net/cubbins/data/prasann/flexcompute_train
```

## What this says about the ~6.5x target

**The 6.5x is real in wall-clock and false in practice.** `cap125` measured **7.39x** and `aggr-cap`
**6.79x**, both against a control whose drift bound is 1.000x, so the mechanism genuinely removes
that much time on a real FSDP training step at 4B/65k-packed. But both diverged: CE 61.3 and 61.3
against a dense CE of 0.041. A CE above `ln(vocab) = 12.4` is not "worse learning", it is a broken
optimisation, and the f1 column confirms what that costs.

## What the KV router actually costs

Measured here, with block skip inert in all three arms so the comparison is clean:

| arm | KV router | step s | vs dense |
|---|---|---|---|
| `aggressive` | on, target 0.25 | 17.76 | 1.100x |
| `aggr-nokv` | **off** | 17.14 | 1.140x |

**Removing the KV router is worth 1.036x** and costs nothing, because it kept 100% of keys anyway.
That is the same direction as `research/kv_route/flexcompute_40x`'s 5.650 -> 5.402x, measured on a training run
instead of a pinned benchmark. In the one arm where it did move (`aggr-cap`, where compaction
changes what the router sees) it reached keep 0.571 against a target of 0.250 — still not its target.

## A bug this uncovered: block-skip checkpoints could not be evaluated at all

Every block-skip arm died at eval in `olmo_core/nn/attention/kv_route.py::_compacted_block_mask`:

```
RuntimeError: The size of tensor a (19) must match the size of tensor b (20) at non-singleton dimension 2
```

`_masked_attention` sets `Kp = min(T, _round_up(max(1, counts.max()), 1024))`. The `min(T, ...)`
clamp makes `Kp == T` whenever few keys are dropped, and at eval `T` is an arbitrary prompt length
rather than a multiple of `FLEX_BLOCK_SIZE`. `k_first = pos_k[:, ::bs]` then has `ceil(Kp/bs)`
blocks while `k_last = pos_k[:, bs-1::bs]` has `floor(Kp/bs)`, and the two are AND-ed. In TRAINING
`Kp` is always block-aligned (`T = 65536`, and `round_up(., 1024)` is a multiple of 128), which is
why this has never fired before. It hits arms with the KV router **off** as well, because block
skipping routes its key mask through `kv_route_attention`.

Worked around, not fixed: `KV_ROUTE_BLOCKMASK=dense` selects the reference `create_block_mask`
builder. Its documented cost is an 8.6 GB mask at 65k x 131k, but at these eval lengths it is
~130 MB per layer and transient. A real fix belongs at the `Kp` assignment (pad the compacted key
run out to a `FLEX_BLOCK_SIZE` multiple and mark the tail slots invalid), not in the mask builder.

## Caveats

- **31 steps.** Long enough for a stable step time (26 timed steps, p90/p10 <= 1.02x on every arm
  except `aggressive`, whose p90 carries one 40 s outlier) and for a meaningful CE, but short enough
  that the hinge finding is *specific to short SFT*. A 1000-step run might well reach its targets;
  nothing here says it would not.
- **`contradiction_iid` is disjoint from a different training file.** Its 500 rows were filtered
  against `single_task_ladders_20k/contradiction/...`, not against the ctc-data seed-pool arm these
  runs train on. Both draw on PubMed perturbation pairs, so treat the absolute level as
  "held-out-ish". Every arm sees the identical training set and the identical eval, so the
  *cross-arm* comparison is unaffected.
- **eval_size = 500 per rung**, so the binomial SE is ±0.022 at f1 0.70 and ±0.007 at f1 0.97.
  Differences smaller than that are not differences.
- **Head gating's CE is lower than dense (0.017 vs 0.041) even though the mechanism is exact.** The
  train module uses `loss_reduction="sum"` with `loss_div_factor=batch_num_tokens_for_loss`, so the
  gathered path sums exactly the same rows and divides by exactly the same factor — the two are
  mathematically identical. The gap is the deliberate fused -> unfused kernel swap the gather
  triggers below `sparse_loss_unfused_max_rows`, whose ~1e-3 relative difference compounds over 31
  steps in a regime where CE is already ~0.03. The f1 column is the arbiter, not the CE.
- 4 GPUs, packed x32, activation checkpointing on: all three of these *reduce* the prize relative to
  `research/kv_route/flexcompute_40x`'s plain 4B/16k basis, and packing alone is worth 5.65x -> 3.90x there.
