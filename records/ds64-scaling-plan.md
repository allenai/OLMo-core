# Data-scaling campaign, short-heavy 2k–64k: soft tokens vs dense (Qwen3.5-4B, then 27B)

Brief (Prasann 2026-09-08 evening): for the best soft-token construction per task, get data-scaling
curves on 2k–64k contexts — performance vs training tokens, training FLOPs and wall-clock — against
dense attention retrained on the same mixes, to see whether soft-KV training is now Pareto-optimal.
Verify the realised speedup, not just the FLOP count. Contradiction and oolong keep their headers.
Everything (data, training, eval) on Beaker. 4B first; 27B queued behind it.

## What changed since the FLOP-scaling grid (records/flop-scaling-report-2026-09-02.md)

The mechanism is the same B1 soft token (mean input embedding, identity projector, no logit
bias, GDN sees the slot in the forward). Three things are new, all from 2026-09-08:
1. **Header-real pooling** (`--st-header-stop-id 25 --st-header-stop-count K`): the tokens the
   question matches exactly stay real (contradiction `Claim N:`, oolong `Date/User/Instance`).
2. **Leak-free neighbour runs** (`--st-neighbour-runs 1`): the document after each gold claim is
   real, random picks form the same runs.
3. **Slots play no role in training**: the GDN write channels (q/k/v pre-conv, beta, g) are
   detached like the attention K/V (`detach_soft_gdn`, default on) — the loss gradient at slot
   columns is exactly zero at every block input (GPU test). Every earlier KV arm had this leak.
4. nq/outlier gold-index bug fixed (their kv17/kv33 arms trained with the wrong gold).

## Arms (per task × budget), `debug/ds64/launch_ds64.py`

**Phase 1 (Prasann: "be careful about keep ratio, ablate with contradiction first"):** contradiction
trains dense + `hdr03/08/17/33` (headers, gold + 1/36 … 1/3) + `runs03/08` at every budget; the
other tasks train dense only. Their soft arms (`ohdr33/17/08`, `kv08/17/33`) are launched by
listing them in `debug/ds64/soft_arms.json` once the contradiction ablation says which keep
ratio survives training (the eval-side parity was at 1/36, but a tiny real set is also the
"real doc = gold" shortcut at training).

| task | dense | soft (best, eval-side parity) | aggressive |
|---|---|---|---|
| contradiction | packed 65536, 8 rows/step | `hdr36`: headers real, gold + 1/36 (~4.8x) | `runs36`: neighbour runs, 1/36 (~11x) |
| oolong | ″ | `hdr33`: headers real, 1/3 lines real, gold-blind (~1.4x) | — |
| nq | ″ | `kv08`: gold + 1/12 (~10x) | — |
| outlier | ″ | `kv08`: gold + 1/12 (~9.5x; gold-forced = "real doc is the answer" shortcut risk, flagged) | — |

Soft arms: unpacked seq 65536, global batch 16 × micro 2 (≈ dense's 524k tokens/step at the
mix's mean length), lr 5e-6, 1 epoch, detached slots (attention + GDN), torch attention backend
(`DS64_SOFT_BACKEND=flash_2` to switch; the 65536 throughput bench decides — job
01M226J7RBPPE8WS418AHER54H). Run names `ds64-<task>-<arm>-u<B>` (27B: `ds64-27b-…`).

## Data, `debug/ds64/build_ds64_data_beaker.sh` (CPU gantry jobs → weka `ds64/shards/<task>_u<B>`)

Per-rung train pools from `ctc-data build --pool auto` (seed pools on the HF Hub; the package is
`ctc/` on this repo's `prasann/ctc_public` branch — pip cannot clone github.com/PrasannS/ctc from
a job), rungs 2k/4k/8k/16k/32k/56k (56k so every example fits the 65536 packing window; 64k is
scored at eval), SHORT-HEAVY token shares 30/20/15/13/12/10 (the standing 45/27/16/8/4 shape
re-weighted so the long rungs carry learnable mass), nested prefixes at 16M/32M/64M/128M
(`compose_uniform_arms.py`; a budget a pool cannot cover is skipped, never shrunk — contradiction's
pair supply may cap it at 64M). Tokenized ONCE with Qwen3.5 markers (`--marker-set qwen3_5`, seq
65536, query after, gold sidecars); dense and soft arms train on the same shards. Job ids:
`debug/ds64/data_build_jobs.tsv` (first launch failed on the ctc-data install; relaunched).

## Evals (marker-aware docchunk evaluator, 2k/8k/16k/32k/64k, `max-test 500`)

contradiction v3 realistic n100/n190/n385/n765 + xlong 64k n1525; nq `outlier_lengthmix/eval_rungs/nq`
2048/8192/16384/32768/65536 (deep source); outlier `eval_rungs/outlier` 2048/8192/16384/32768 +
v2 xlong 64k n448 (random fillers — flagged); oolong v2_clean ctx 2048…65536. The 64k rungs have
300 questions (xlong) — quote ±0.026 at f1 0.8.

## Axes and outputs

Performance = mean f1 over rungs (and per-rung). x-axes: training tokens (nominal budget and
realised), training FLOPs priced per example at its real length (attention quadratic in the
example; soft arms metered on their compacted rows via `FlopMeterCallback`), and wall-clock
GPU-hours from the Beaker training logs (`debug/flop_scaling/collect_walltime.py --states
debug/ds64/orchestrator_ds64_state.json`). Realised speedup = dense seconds/token ÷ soft
seconds/token at the same budget. Orchestrator `debug/ds64/orchestrate_ds64.py` (state
`orchestrator_ds64_state.json`, log `orchestrator_ds64.log`); results `results/ds64/`.

## Status log

- 2026-09-08 21:20 first data builds launched (uniform 16k–56k) → all four died on the ctc-data
  install; 21:45 relaunched as short-heavy 2k–56k with the branch install.
- 2026-09-08 21:40 local smokes on sneetches (1 GPU, 24 rows, seq 65536, unpacked, torch backend):
  contradiction `hdr03`+runs and oolong `ohdr33` both train (4 steps, CE 0.02–0.06 / 0.2–0.65 from
  the trained dense bases, checkpoint saved). 65536 throughput bench (1xH100, Beaker
  01M226J7RBPPE8WS418AHER54H): dense/flash 9.70 s/step (6.8k tok/s); soft k=1/3 2.79 s (torch) /
  2.76 s (flash) = dense on a 22.7k row (2.70 s); soft k=1/12 0.85 s (torch = flash) = dense on a
  6.3k row (0.86 s). **Per-step speedup = the compaction factor (3.5x at 1/3, 11.4x at 1/12);
  backend irrelevant; the +log L bias path is 2.7x slower and is not used.** Peak memory 22.5 /
  20.6 GB vs 35.3 GB dense.
- 21:35 data relaunch #2: contradiction with POOL_2K=15000 (ctc-data refuses >~18k distinct 2k
  examples → its 128M budget is skipped), oolong tokenization with the conda interpreter
  (bare `python` in the job lacks numpy); nq/outlier pools still building (their first tokenize
  step will fail the same way; relaunch reuses the pools).
- 22:25 **batching fix.** First soft launch used 16 rows/step (sized for a 40k mean length); the
  short-heavy mix averages ~4.5k tokens/example, so hdr03-u16M ran 227 optimizer steps of ~72k
  tokens: FLOP meter 0.18x dense, wall-clock 1337 s vs dense 301 s (4.4x SLOWER — per-step
  overhead on tiny unpacked micro-batches), and 7x dense's optimizer steps (unfair the other way).
  Relaunched every soft arm at 128 rows/step (~576k tokens/step ≈ dense's 524k), micro-batch per
  arm by compaction (8 / 6 / 4 / 2 rows for keep 1/36 / 1/12 / 1/6 / 1/3), run names carry `-b128`;
  the gb16 runs were cancelled (the finished hdr03/runs03 16M ones stay in the state as a labelled
  reference). All four data builds are done (contradiction 16M/32M/64M; others 16M–128M).
- 22:52 **backend fix.** The `-b128` torch-backend arms ran ~70 s/step (dense 10 s) at 0.18x the
  FLOPs. Local test (sneetches, 1 GPU, 16 rows/step, one step): torch vs flash equal at micro 2
  (2 min each incl. load), but at micro 8 torch 234 s vs flash 71 s — PyTorch SDPA falls off the
  fused kernel on multi-row right-padded batches; the 65536 bench (B=1, no padding) could not see
  it. Soft arms relaunched on `flash_2` (exact for right-padded causal rows) as `-b128f`; the
  torch `-b128` runs cancelled (finished ones kept as accuracy-only reference). The keep-1/12 arms
  also needed micro 4 (128 rows/step must divide by micro x 8 GPUs).
- 23:35 **first accuracies.** Dense ladders healthy (contradiction 16M mean f1 0.93; nq 16M
  2k/8k/16k/32k 0.97/0.93/0.87/0.81; oolong 16M 0.85/0.58/0.53/0.53). The two gb16 soft runs at
  keep 1/36 COLLAPSE at full-attention eval: `hdr03` 0.24/0.15/0.07/0.04, `runs03`
  0.68/0.45/0.20/0.06 — the gold-forced shortcut ("real docs = answer") that the eval-side
  parity cannot see. Keep ratio is the load-bearing knob, as Prasann said; the ablation (hdr08 /
  hdr17 / hdr33, runs08) decides what survives training. Soft-arm wall-clock still ~65 s/step
  after the flash and vectorized-compaction fixes (per-row cost ~4 s under FSDP vs ~0.9 s on one
  GPU); diagnostics `hdr03gb` (no fingerprint hook) and `hdr03m1` (one row per micro-step) launched.
