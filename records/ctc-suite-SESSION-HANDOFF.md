# CTC-suite — SESSION HANDOFF (2026-07-20 ~05:15)

## SESSION 2 UPDATE (resumed ~05:30) — user goal: FULL result set (chunked+dense, all ladders); **PIVOTED TO 4B**
- Stopped all 5 orphaned prior-session subagents (were idle-burning). Re-driving from main thread + finite agents.
- **SCALE PIVOT (user, explicit): 4B for ALL tasks, both arms (full+chunked-mix), all rungs. Deadline ~20h but SOFT (may slip — accepted).** Compute: **Beaker 1-NODE jobs** (not 2-node) + local H200 + lambda A100 (fit-test first) + VESSL (overflow, ask-first). The inherited 0.8B plan is SUPERSEDED (0.8B runs kept only as a free scale baseline).
- 4B facts: `beaker_ctc_suite.py --model-scale 4b --num-nodes 1 --priority urgent launch`; q35-4b HYBRID base already on weka (contra 4B uses it) — do NOT restage base, but each task SHARD needs cubbins/scratch→S3→weka relay. 4B@40960 fits 80GB w/ full-shard+AC (AC auto-fulls for 4b) → likely fits lambda A100 too (VERIFY). ⚠ correct base = q35-4b hybrid (NOT /scratch/.../q4b-base-modelonly = wrong old non-hybrid vocab 151936).
- **Beaker 4B contra** full=01KXYVRKZSHW5XGF7JRX9A78R2 cmix=01KXYVS8G8CT86T8ZYS1HATM1E: HEALTHY, ~step 750/1210 eta 40m (per-4B-run ~2h on 1-2 node H100). Do NOT relaunch contra.
- **Cancelled** all pending 0.8B (wave-2 3338955-3338985 + nq 3338986-88/109753). 2 running 0.8B left as baseline (hotpotqa 3338930/sneetches, msmarco 3338954/mooney) + lambda wave-1 (some running/pending).
- Agents: **4B fan-out coordinator** ad3e2b74f8bc205e3 (owns ALL 4B launches incl nq/qdmatch_nq; →results/ctc_suite/stage5_4b_launch_status.md), **vLLM wiring** a677c61709025c809 (parity-check Qwen3.5-hybrid vLLM eval; native stays default; →results/ctc_suite/vllm_parity.md). NQ debug CLOSED: **no bug — genuine underfitting at 2.5k budget** (qdmatch_hpqa .985 vs qdmatch_nq .037 through identical harness); 20k/4B run is the real test. nq/qdmatch_nq 20k CE-clean shards already on lambda (see nq_debug_diagnosis.md).
- Per-run 4B ≈ ~2h (1-node) but ~2× GPUs/run vs 0.8B; throughput now capacity-bound on H100/H200; lambda-4B (if fit) adds 4 nodes. eval (owned later) = native torchrun default; vLLM ~1.45x if parity holds.

Restart doc for a fresh session. Read this + `records/overnight-stage345-plan.md` (full rung policy,
roster, live log) + the memory files (`ctc-suite-scaling-effort`, `ctc-stage3-2k-validation-results`,
`nq-pipeline-10pct-hardneg-cefilter`, `lambda-cluster-ctc`, `always-urgent-priority`, `no-fable-agents-directive`).

## The goal (unchanged)
Build a large-scale **Figure 4** (TimeToPayAttention): full-attn vs document-chunked attn across a
~26-task corpus-reasoning suite, x-axis 2k→32k tokens, Qwen3.5-0.8B (contra→4B), chunked mask only on
non-GDN layers, curriculum mask-mix. One joint run per (task, arm) on a 20k mixed-n shard; eval per-rung
fixed-500. Results JSON → `results/ctc_suite/`. Rung cap: 2k-16k for reorder/cycle/groups4, else 2k-32k.

## WHERE THINGS STAND

### Stage 3 (2k full-attn validation) — DONE, with corrections
- Table: `results/ctc_suite/pilot_2k/SUMMARY.md` + per-task JSON + `curves/`.
- Clean high scorers: niah .988, qdmatch_hpqa .985, hotpotqa .861, cycle .786, contradiction .684,
  absence_gutenberg .813, msmarco .782, helmet_summ rouge1 .369 (all healthy, diverse preds).
- **Collapse cluster = a PILOT-BUDGET artifact, not per-task bugs.** groups4, nq (.074), qdmatch_nq
  (.037), obliq (.074), mathmatch/strmatch/textgroups all mode-collapse (constant/near-constant preds).
  Common cause: too few examples (~2500) in 1 epoch for hard tasks. hotpotqa/contradiction learned
  because their pilot ran at 20k. **The fix is the uniform 20k Stage-5 runs**, NOT ce-filter alone.
- **nq is OPEN (do NOT claim ce-filter fixed it):** a 2500-ex CE-CLEAN retrain (nq_ceconfirm) ALSO
  collapsed (f1 .034, flat CE ~.51). The clean source audits genuinely filtered (hard_ratio .098,
  false_neg .010). So ce-filter alone at 2500 ex does not fix nq — the 20k budget is the lever.
  Decisive test = a full-20k CE-clean nq run (NOT yet launched). A confirm-agent cross-encoder audit
  of the 4 nq files was mid-flight when this session ended — check `results/ctc_suite/` for its output.
- Still eval-BLOCKED: **fiqa, scifact** — their pilot eval rung files were 0 bytes; ckpts exist on
  lambda (fiqa-r3, scifact — KEPT during cleanup). Run these evals once the data agent's BEIR eval
  sets are built.
- obliq .074 is on an UNDERTRAINED step-44 ckpt at ~27k tokens (its "rung_2048" was 30 full ~27k docs)
  — needs a proper full-epoch run before believing anything.

### Stage 5 (fan-out) — PARTIALLY LAUNCHED, see compute below

## COMPUTE STATE (verify all with squeue/beaker before acting)

### Lambda (32× A100-80GB) — JUST UNBLOCKED, currently EMPTY
- Was blocked all session by the per-USER NFS quota (1.30T). User freed ~138G via the approved rm;
  **quota now 88.6% (1.15T/1.30T, ~168G free).** Queue is EMPTY (all prior jobs cancelled).
- **Checkpoints are ~12G each (model+optim distcp); ephemeral save-then-remove needs ~24G transient.
  Budget headroom: at 168G free you can run ~6-7 concurrent 0.8B runs' checkpoints safely. WATCH the
  quota each loop; if it climbs past ~92%, have the USER free more (deletes are classifier-blocked —
  user-only via `!`). Do NOT let it hit 100% (crashes checkpoint writes).**
- CANNOT harvest lambda compute-node-local /data (no ssh head→compute) and lambda is air-gapped
  (no S3/scratch) → **lambda runs MUST save to NFS (LROOT/ckpts), the SAVE_ROOT default.** Harvest =
  rsync ckpt from lambda NFS to a Berkeley node, eval there (Stage-3 pattern).
- Relaunch recipe (per task×arm), run from head after `rsync` of local src:
  `cd $LROOT/OLMo-core/src/scripts/train/memexpress/ctc_suite && sbatch --partition=lambda \
   --qos=preemptive_high --time=12:00:00 --gres=gpu:A100:8 -J <run> \
   --export=ALL,TASK=<t>,DATA_SRC=<t>_train,VARIANT=<full|chunked-mix>,SCALE=0.8b,EPOCHS=1,\
   SEQ_LEN=40960,RUN=<run>,ACT_CKPT=full,SHARD_DEGREE=8 run_ctc_lambda.sbatch`
  (ACT_CKPT=full + SHARD_DEGREE=8 REQUIRED — 40960-seq OOMs on 80G A100 without them.)
- Shards staged at `$LROOT/data/<task>_train` for: contradiction, grouping, hotpotqa, oolong, outlier,
  reorder, rerank (+ the data agent added many small ones: nq, msmarco, niah, qdmatch_*, etc — verify
  metadata.json + that they're the FULL 20k, not pilot stubs, before use). LROOT=/accounts/projects/sewonm/prasann/ctc_suite.
- KEPT ckpts (do NOT delete — eval pending): fiqa-full-lambda-r3, scifact-full-lambda,
  msmarco_rerank-full-lambda(+r2).

### Local jsteinhardt H200 — a launcher agent was mid-flight (DIES on session clear)
- Agent a5966c49 (sonnet) submitted ONE fit-validation job **3338930 (hotpotqa-full, mooney,
  NGPU=1 seq 40960 +AC)** and paused waiting on its monitor; it had NOT fanned out the other 11 and
  DIES on session clear. **New session: check job 3338930 — if it trains past step1 with OOM=0, the
  NGPU=1+AC config is validated for H200 local overflow. But since lambda is now FREE, prefer running
  the main fan-out on lambda; treat local as overflow.** Its intended runs: 6 tasks × 2 arms as
  `ctc-s5-<task>-<arm>-08b-loc`. Deliverable (may be incomplete): `results/ctc_suite/stage5_local_launch_status.md`.
- **New session: `squeue -u prasann | grep loc` — if it submitted jobs, KEEP them (harvestable) and
  DON'T also run those 6 on lambda (dedupe). If it submitted nothing, just run everything on lambda
  (now free + faster).** jsteinhardt = 8-GPU/user cap → NGPU=1 for 8-concurrent. berkeleynlp
  (horton/lorax) = separate group quota.

### Beaker jupiter — 4B contradiction, HEALTHY
- Two 4B contra runs RUNNING (started, not error-finalized, ~1h in):
  full=`01KXYVRKZSHW5XGF7JRX9A78R2`, cmix=`01KXYVS8G8CT86T8ZYS1HATM1E`. Reads base+shard from weka.
  Correct q35-4b base (cubbins /data + S3 + weka). Check loss descending next loop.

### Data agent (a52512b2) — COMPLETED
- Staged eval rungs (fixed-500) for the 7 launched tasks (contra/outlier/grouping/oolong 2k-32k;
  hotpotqa/rerank/reorder 2k-16k) + 20k train shards for ~20 tasks + Path-Y eval sets. Live table:
  `results/ctc_suite/stage5_data_status.md`. Eval files (unified JSONL):
  `/scratch/users/prasann/ctc_suite_staged/eval_rungs/<task>/rung_<tokens>.jsonl`.
  ⚠ oolong eval_size=100/rung (flag ±0.05 inline; 500-build was in flight). Its in-flight cubbins
  conversion/gen jobs (Dependency in squeue) are Berkeley-side, harmless.
- CE directive honored: nq train = /scratch/users/prasann/nq_p10_20k/nq_train_k25-202_clean.jsonl
  (CE-filtered, hard_ratio .097). nq eval = p10/CE. qdmatch_nq generated from CE-clean nq.
- Data-poor/deferred: xabsence (needs LLM rebuild — blocked), obliq ~1797, scifact ~4k,
  helmet_summ 17.5k, groups4 16k rung, hotpotqa/rerank 32k, nq 16k/32k (uniform-k200 CE regen).

## NEXT ACTIONS (priority order for the new session)
1. **squeue -u prasann | grep loc** + read stage5_local_launch_status.md → learn what local jobs exist.
2. **Fan out Stage-5 on lambda** (now free) for all tasks NOT already running on local: at minimum the
   ~13 tasks beyond the 7 launched, PLUS re-run any that only have collapsed pilot ckpts. **Include a
   full-20k CE-clean nq run + qdmatch_nq** (the decisive open test). Save to NFS. Watch quota.
3. **Beaker**: confirm 4B contra loss descending.
4. **Evals**: as each Stage-5 run finishes, eval per-rung (max-length ≥ rung+2048; DUMP GENERATIONS on
   any exact-0; node-pin eval to the training host for node-local /data). fiqa/scifact evals once
   their BEIR eval sets exist.
5. **Stage 4 gate** = the 2k/4k/8k rungs of the full+chunked Stage-5 runs: gap flat for O(N),
   widening for O(N²)/O(N³). Extract as those early rungs land.
6. Stage 6 = aggregate/plots/results-hub (later).

## HARD-WON TRAPS (do not re-hit)
- seq_len ≥ task max_example_len (else PadToLength drops long ex / trainer hard-fails). 40960 for all
  (reorder shard max 40957); eval caps reorder at 16k separately — DON'T rebuild reorder shard.
- Fresh run-name on every relaunch (auto-resume fingerprint crash).
- eval max-length ≥ rung+2048 (obliq/goldgrad truncation → empty gens → fake f1=0).
- Lambda: NFS-only saves (compute /data unreachable); ACT_CKPT=full + SHARD_DEGREE=8 for 40960 on A100;
  QOS=preemptive_high runs 4 nodes concurrent; SLURM may report FAILED despite success (verify via
  loss+ckpt); ssh head→compute is BLOCKED.
- Deletes over ssh are classifier-blocked → USER runs them via `!`.
- Subagents: SONNET default (opus only for hard verify), NEVER fable. Priority ALWAYS urgent on Beaker.
- `n` = CORPUS size only; eval-set size = `eval_size`; flag any eval_size<500 inline with its ±SE.
- Capture full per-step loss curve into every result JSON (wandb is offline-only on lambda).
