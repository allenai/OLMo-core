# ds64 data-scaling campaign — handoff (state as of 2026-09-09 ~09:00 PDT)

Soft-token (pooled-document) training vs dense, on short-heavy 2k–64k mixes, Qwen3.5-4B, all on
Beaker. Goal (Prasann, 2026-09-08 evening): **is soft-KV training Pareto-optimal against dense
attention** on curves of accuracy vs training tokens / FLOPs / wall-clock, and **is the realised
speedup what the compaction ratio promises**? 27B is queued behind the 4B grid.

Plan + running status log: `records/ds64-scaling-plan.md`. This file is the "read me first".

---

## 1. The headline so far

**Contradiction: the header-real soft token at keep 1/3 (`hdr33`) BEATS dense at matched training
FLOPs** — the first clean matched-compute win on this task (the 2026-09-02 study had it at
"parity to slight loss", 1.21x). Measured FLOP meter, mean f1 over 2k/8k/16k/32k/64k
(eval_size 500/rung; the 64k contradiction rung skips 19/500 rows over the 70k generation limit,
identically for every arm):

| arm | 16M tokens | 32M tokens | 64M tokens |
|---|---|---|---|
| dense | 0.763 @ 758 PF | 0.868 @ 1539 PF | 0.922 @ 3056 PF |
| hdr33 (headers real, gold + 1/3 docs) | 0.660 @ 277 PF | 0.773 @ 521 PF | **0.839 @ 1005 PF** |
| hdr17 (… + 1/6) | 0.441 @ 299 PF | 0.649 @ 549 PF | 0.750 @ 1076 PF |
| hdr08 (… + 1/12) | 0.259 @ 236 PF | 0.379 @ 436 PF | 0.504 @ 853 PF |
| runs08 (neighbour runs, 1/12) | 0.266 @ 153 PF | 0.402 @ 287 PF | 0.492 @ 571 PF |
| runs03 (neighbour runs, 1/36) | 0.221 @ 134 PF | 0.296 @ 262 PF | 0.309 @ 521 PF |

Interpolating the dense curve in log-FLOPs, `hdr33-64M` is **+0.034 above dense at the same
1005 PF** (0.839 vs 0.805) — this one comparison sits INSIDE the measured dense range and is the
trustworthy one. `hdr33-32M` (+0.066) and `hdr33-16M` (+0.046) look better still but both require
*extrapolating* dense below its cheapest measured point (758 PF).
**→ First job for the next agent: dense anchors at 4M and 8M** (see §5), exactly as the earlier
FLOP-scaling study had to do. Without them the cheap end of the curve is not evidence.

**Keep ratio is the load-bearing knob and eval-side parity did NOT transfer.** The 2026-09-08
eval-side probe found keep 1/36 reproduces full attention on held-out rows; in *training* it
collapses (hdr03 mean f1 0.11 at 16M, runs03 0.22) because a tiny real set teaches "the answer is
in a real document". Accuracy is monotone in keep at every budget, and only 1/3 tracks dense.
Contradiction's honest compaction is therefore ~0.33x FLOPs, not the 0.09x the probe suggested.

**Other tasks: dense ladders complete, soft arms NOT yet run** (they were gated behind this
ablation and are launching now, §3). Dense mean f1 (2k/8k/16k/32k/64k):

| task | 16M | 32M | 64M | 128M |
|---|---|---|---|---|
| nq | 0.844 | 0.865 | 0.907 | 0.915 |
| oolong | 0.603 | 0.644 | 0.661 | 0.690 |
| outlier | 0.322 | 0.452 | 0.606 | 0.659 |
| contradiction | 0.763 | 0.868 | 0.922 | (pool caps at 64M) |

Note outlier's 64k rung is ~0.0 at every budget (its 64k eval file is the v2 xlong one with random
fillers) — treat outlier@64k as broken, not as a result.

---

## 2. Wall-clock: NOT yet answered, and it is the open risk

FLOPs are down 3x; **wall-clock is not**. Chronology of what was fixed and what remains:

1. **65536 microbenchmark (1 GPU, B=1, `debug/pooled_kv/bench_softtoken_throughput.py`, job
   01M226J7RBPPE8WS418AHER54H):** per-step time tracks compaction exactly — dense/flash 9.70 s,
   soft k=1/3 2.79 s, soft k=1/12 0.85 s, each equal to dense on a row of the compacted length.
   Backend irrelevant there; the `+log L` bias path is 2.7x slower (unused).
2. **On the real 8-GPU FSDP runs the soft arms were ~55–65 s/step** against dense ~10 s at 0.33x
   the FLOPs. Three causes found, two fixed:
   - *torch SDPA on padded multi-row micro-batches* is ~4x slower than flash (local test: micro 8,
     torch 234 s vs flash 71 s; equal at micro 1–2). → soft arms now run `--attn-backend flash_2`.
   - *per-document host syncs* in `compact_pooled_rows` (two `.item()` per pooled doc) stalled the
     FSDP collectives. → vectorised to one sync per row (commit 96f2fba11 lineage).
   - *padding waste inside a micro-batch*: compacted rows are padded to the longest member, so a
     56k example beside seven 2k ones wastes ~6x. Diagnostic: 8 rows/micro 55 s/step, 1 row/micro
     24 s/step, gold-blind (hook removed) 64 s/step → the keep hook is innocent.
     → **`microbatch_sort_pad_id`** added to `TransformerTrainModuleConfig`: each rank's batch is
     length-sorted before the micro-batch split, so micro-batches are length-homogeneous. Set
     automatically for `--variant softtoken`. Smoke-tested on 8 local GPUs; **its effect on Beaker
     step time is still unmeasured** — that is what the gen-3 runs will show.
3. Real fix if sorting is not enough: compact-then-**pack** the compacted rows with `cu_seqlens`
   (no padding at all). Not implemented.

**So: quote FLOPs today, do not quote wall-clock.** `results/ds64/results.csv` carries
`gpu_hours` per run, but every `-b128f2-` number in it is from the pre-sorting recipe and is
wall-clock-invalid (accuracy is fine). Only `-b128f3-` runs are wall-clock-faithful.

---

## 3. What is running right now

- **Orchestrator** (login node, restarted 08:51): `debug/ds64/orchestrate_ds64.py`, state
  `debug/ds64/orchestrator_ds64_state.json`, log `debug/ds64/orchestrator_ds64.log`. It launches
  training, then the 5-rung eval per finished run, retries once/thrice, harvests every 90 min.
  Restart with:
  `DS64_GEN=3 setsid nohup python debug/ds64/orchestrate_ds64.py >> debug/ds64/orchestrator_ds64.log 2>&1 &`
- **29 gen-3 runs relaunching** (`-b128f3-` names): contradiction hdr33/hdr17, oolong ohdr33/ohdr17,
  nq kv33/kv17, outlier kv33/kv17 at every budget. These are the wall-clock-faithful,
  keep-ratio-sane arms; the whole first launch of them died on an unpushed commit (§4, trap 1) and
  was reset at 08:51.
- Generations in run names: no tag = dense; `-b128-` torch backend (cancelled); `-b128f-` flash,
  per-doc syncs (cancelled); `-b128f2-` flash + vectorised syncs (**accuracy valid**, wall-clock
  not); `-b128f3-` + length-sorted micro-batches (**both valid**). Two very early `hdr03-u16M` /
  `runs03-u16M` runs carry no generation tag and used 16 rows/step — reference only.

---

## 4. Traps hit (all cost real time; do not repeat)

1. **An unpushed commit kills every Beaker job** with `fatal: remote error: upload-pack: not our
   ref <sha>`. It killed all 30 gen-3 launches and several evals. `git push` BEFORE any launch,
   and if a wave of jobs fails at once, check this first.
2. **Beaker data builds:** use `/opt/conda/bin/python` with `pip install -e . ./ctc` and gantry
   `--install false`. Gantry's uv venv has neither `pip` nor torch; the image's conda python lacks
   the repo. `ctc-data` must come from `origin/prasann/ctc_public:ctc/` (pip cannot clone
   github.com/PrasannS/ctc from a job). Run it as `python -m ctc.data.cli`.
3. **Batching arithmetic:** rows/step must be divisible by micro-batch × 8 GPUs, so micro ∈
   {1,2,4,8,16}. micro 6 dies with `global batch size ... must be divisible by`.
4. **Rows/step must be sized in TOKENS, not rows.** The short-heavy mix averages ~4.5k
   tokens/example, so the first soft launch at 16 rows/step gave the soft arm 7x dense's optimizer
   steps at 0.14x its tokens/step. 128 rows/step ≈ dense's 524k tokens/step.
5. **contradiction's pool caps at ~18k distinct 2k examples** (`ctc-data` refuses near-duplicates),
   so its grid stops at 64M and its 2k pool is built with `POOL_2K=15000`.
6. The collector used to cache eval logs mid-run and froze partial rung sets; it now caches only
   finished logs. If a run shows fewer than 5 rungs, delete `results/ds64/logs/eval_*.log`.
7. The orchestrator latches a `done` flag; after a full pass it exits immediately with `ALL_DONE`
   on restart. Clear `done` in the state JSON before restarting it for new work.

---

## 5. Next steps, in priority order

1. **Dense anchors at 4M and 8M tokens** for all four tasks (`--budgets` are nested prefixes, so
   this needs a `compose_uniform_arms.py` run at those budgets plus a tokenize, or simply
   `--max-tokens`-style short runs on the existing 16M shard). Without them, every matched-FLOP
   claim below 758 PF is extrapolation. This is the single highest-value missing piece.
2. **Measure gen-3 wall-clock**: compare a `-b128f3-` run's seconds/step with its `-b128f2-` twin
   and with dense (`debug/flop_scaling/collect_walltime.py --states debug/ds64/orchestrator_ds64_state.json`,
   or the `train_seconds`/`gpu_hours` columns of `results/ds64/results.csv`). If sorting did not
   close the gap, implement compact-then-pack with `cu_seqlens` (§2.3).
3. **Read out the other three tasks' soft arms** as the gen-3 evals land, and build the same
   FLOP-matched table. Oolong is the arm most likely to win (its 2026-09-02 result was the one
   matched-compute win, and its headers are half the line so its compaction floor is ~0.6).
4. **Plots**: performance vs tokens / vs FLOPs / vs GPU-hours per task.
   `debug/flop_scaling/make_axes_plots.py` is the template; nothing equivalent exists for ds64 yet.
5. **27B**: `DS64_SCALE=27b` on titan (B200), base `q35-27b-base-markerfix`, evals need ≥192 GB
   GPUs. Only worth starting once the 4B wall-clock story is settled.
6. Optional: a `hdr50`/`hdr66` arm — accuracy is monotone in keep and 1/3 is the cheapest keep that
   works, but nobody has checked whether 1/2 buys back the remaining gap to dense at lower cost
   than more tokens does.

---

## 6. Where everything lives

- **Code**: `debug/ds64/` — `build_ds64_data_beaker.sh` (per-rung `ctc-data` pools 2k…56k,
  short-heavy shares 30/20/15/13/12/10, nested arms, Qwen3.5 marker shards → weka
  `ds64/shards/<task>_u<B>`), `compose_uniform_arms.py`, `launch_ds64.py` (arm definitions,
  `ARM_EXTRA` / `ARM_MICRO` / `TASK_ARMS` / `TASK_BUDGETS`), `orchestrate_ds64.py`,
  `collect_ds64.py`, `harvest_ds64.sh`, `soft_arms.json` (gate for non-contradiction soft arms),
  `data_build_jobs.tsv`, `LAUNCH_LEDGER.tsv`.
- **Results**: `results/ds64/results.csv` (per-run per-rung f1, FLOP meter, wall-clock),
  `results/ds64/logs/` (cached Beaker logs).
- **Method code**: `mark_doc_headers_free` (`nn/attention/chunked_mask.py`),
  `enable_pooled_soft_tokens(header_stop_id=…, detach_soft_gdn=…)` (`nn/transformer/model.py`),
  `neighbour_runs` (`nn/attention/pooled_doc_kv.py`), `microbatch_sort_pad_id`
  (`train/train_module/transformer/{config,train_module}.py`), trainer flags
  `--st-header-stop-id/-count`, `--st-neighbour-runs`, `--st-no-detach-soft-gdn`, `--torch-profile`.
- **Method background**: `records/pooled-doc-kv-attention.md` (the eval-side probe, header-real
  parity, the GDN write detach), `records/soft-kv-slot-probe-handoff.md`,
  `records/flop-scaling-report-2026-09-02.md` (the previous campaign this supersedes for KV arms).

## 7. Method state (what "the soft token" is now)

Mean input embedding through an identity projector at the pooled document's centre position; slot
K/V detached; **GDN write channels also detached** (`q`/`k`/`v` pre-conv, `beta`, `g` — commit
96f2fba11) so a slot contributes zero gradient to any parameter, verified by
`src/test/nn/pooled_soft_token_gdn_test.py`; no logit bias (every bias variant lost); GDN still
*sees* the slot in the forward. Data-side: document headers stay real
(`--st-header-stop-id 25 --st-header-stop-count 1` contradiction / `3` oolong), keep set = gold +
a fixed fraction of random documents. This differs from the pre-2026-09-08 KV arms in the header
rule, the gold-index fix, and the GDN detach — old `kv17`/`kv33` numbers are not comparable.
