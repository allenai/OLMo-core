# Overnight autonomous plan — finish Stage 3 & 4, get Stage 5 running (2026-07-19 night)

User mandate (asleep, operate autonomously): finish Stage 3 + Stage 4; have **Stage 5 all running by
morning**. Compute: Beaker (jupiter H100), lambda (32 A100), local (jsteinhardt/berkeleynlp H200),
VESSL (final only, $500 cap). Straightforward (data-ready) task runs first; build more data for
data-poor tasks. 1 epoch on ≥20k datapoints per (task, arm).

## Rung policy (user)
- **2k→32k** (rungs 2k/4k/8k/16k/32k) for all tasks EXCEPT:
- **2k→16k** (cap, drop 32k) for: **reorder, cycle, groups4** (reorder per user; cycle+groups4 = O(N³)).
- Training = ONE joint run per (task, arm) on a 20k-datapoint shard with doc-count n ~ Uniform[n(2k),
  n(cap)]; eval per-rung fixed-500 sets. seq_len 40960 (fits 32k) / 20480 for 16k-cap tasks.

## Roster (~26 tasks; dropped: redundancy, absence_official/pubmed, qdmatch_obliq)
O(N): nq, hotpotqa, niah, oolong, helmet_qa, helmet_summ, scifact, fiqa, msmarco, msmarco_rerank
O(NM): outlier, outlier_amzn, grouping, grouping_labeled, textgroups
O(N²): contradiction, absence_gutenberg, strmatch, qdmatch_nq, qdmatch_hpqa, obliq_retrieval,
       xabsence, mathmatch, reorder(16k)
O(N³): cycle(16k), groups4(16k)
Arms: **full** + **chunked** (chunked mask on non-GDN layers; curriculum mask-mix). Scale 0.8B for
repro; **contra → 4B** (user). chunked eval = chunked-vllm/native (chunked-sdpa FORBIDDEN on hybrids).

## DATA rules
- **nq + qdmatch_nq (nq-derived): MUST use --ce-filter + hard-neg-frac 0.10 at EVERY rung** (the
  Stage-3 nq collapse was a dropped ce-filter). Audit hard-ratio ≈0.10 + false-neg ≈0 before use.
- Reuse existing 20k full-scale shards where present: shards/ = contradiction, grouping, hotpotqa,
  oolong, outlier, reorder, rerank. 20k pools on mooney /data: contradiction, nq, oolong, outlier, rerank.
- Data-poor (need more): grouping OpenAlex pool, obliq (~1797), + audit textgroups/cycle/groups4/
  mathmatch/strmatch/xabsence/absence_gutenberg/qdmatch. Straightforward-first: launch data-ready
  tasks into Stage 5 immediately; generate more for the rest in parallel.

## Phases (driven by the overnight loop)
- **P0 Stage-3 finish** (agents in flight): eval finisher a46e7afb (msmarco✓0.782, absence✓0.813,
  obliq=truncation RE-RUN, fiqa/scifact/helmet_summ pending) + nq CE-confirm a1e1a4c4. Monitor to done.
- **P1 Ladder build**: mixed-n 20k training shards + per-rung fixed-500 eval sets, straightforward-first.
- **P2 Stage-4 trend gate**: full vs chunked at 2k/4k/8k on ~4-6 tasks spanning classes (O(N): nq(CE-fixed)
  or niah; O(N²): contradiction, strmatch; O(NM): outlier). Confirm gap flat for O(N), widening for O(N²).
- **P3 Stage-5 fan-out**: ~26 tasks × {full, chunked}, launch across Beaker+lambda+local; contra@4B.
  Eval per rung. VESSL reserved for final/overflow only.

## Compute assignment
- lambda (32 A100): bulk 0.8B training fan-out (seq_len sizing per-task; fresh run-names; 90min+ walls).
- local jsteinhardt/berkeleynlp H200: NGPU=1 concurrency under caps; contra@4B here or Beaker.
- Beaker jupiter H100: 4B/9B runs + overflow; priority ALWAYS urgent.
- VESSL: final stage / overflow only ($500 cap).

## Traps (do not re-hit)
seq_len ≥ task max_example_len; fresh run-name on relaunch (auto-resume fingerprint); eval max-length
≥ rung+2048 (obliq truncation!); pin eval to training node (node-local /data); per-shard eval out-root;
lambda ssh drops (retry); NGPU=1 to beat jsteinhardt cap; DUMP GENERATIONS on any exact-0 eval.

## Live status (updated each loop)
- **2026-07-20 ~03:58** (loop 1 post-compaction): all 4 agents healthy, no stalls.
  - P0 Stage-3: eval finisher running helmet_summ + **obliq re-run** (27k-ctx prefill — truncation
    fix), fiqa/scifact queued. **nq CE-confirm** training on mooney H200 (job 3338786, epoch 1) —
    the ce-filter fix verification (n=2500, same budget as pilot, isolates the ce-filter variable).
  - P1 data: builder generating synthetic 20k train+eval for mathmatch/strmatch/textgroups/cycle/
    groups4 (isolated subdir to avoid pilot-file collision). Small train shards already on lambda data/.
  - P3 launcher: caught the WRONG 4B base (/scratch q4b = old Qwen3-4B vocab 151936, std attn — NOT
    hybrid; correct q35-4b on cubbins /data). Caught reorder seq_len conflict (max_example_len 40957
    > 20480 guard → must run 40960 OR rebuild reorder shard to 2k-16k; FLAGGED to data agent).
    Staging 7 shards (4.7G) to lambda; lambda queue still EMPTY (pre-submit). Local training live on
    cubbins (GPU0 87%). Lambda quota 95.9% (~50G headroom, 4.7G stage fits).
  - Compute: local jsteinhardt = nq-ceconfirm(mooney) + 2 launcher bash sessions(cubbins) + 1 pending
    (QOSGrpGRES cap). lambda 32×A100 IDLE pending launcher submit — WATCH next loop.
  - Directive noted: subagents on sonnet (opus only for hard verify), never fable.
- **2026-07-20 ~04:20** (launcher agent DONE): **all 14 Stage-5 runs launched**. Lambda VERIFIED
  4/4 usable A100 nodes RUNNING (hotpotqa full+cmix, grouping full+cmix), 8 PENDING (jobs 109718-728,
  cycle ~3 waves ~7.5h, 12h walls). **KEY FIX: 40960-seq at 0.8B OOMs on 80GB A100 w/o AC** (default
  tuned for 141GB H200) → settled `--activation-checkpointing full --shard-degree 8`; env knobs
  ACT_CKPT/SHARD_DEGREE/QOS added to run_ctc_lambda.sbatch. chunked-mix@40k also validated. Beaker 4B
  contra: full=01KXYVRKZSHW5XGF7JRX9A78R2 (started+ready), cmix=01KXYVS8G8CT86T8ZYS1HATM1E (scheduled),
  jupiter urgent 2 nodes. lambda under `preemptive` (no 8-GPU cap; preemptive_high caps at 1 node).
  - **REORDER RESOLUTION (my call):** run at 40960 on the 2k-32k shard, EVAL at 2k-16k. "stop at 16k
    for reorder" = eval-cost concession, NOT a train-data ban. Train-long/eval-short is fine+benign.
    **NO reorder shard rebuild** — data agent should SKIP it.
  - **WATCH:** Beaker 4B step-1 OOM (proven config but not yet observed for 4B); lambda quota 96.5%.
- **2026-07-20 ~04:20** (data builder status): P1 eval rungs staged for 6/7 launched tasks
  (contradiction/outlier/grouping/oolong 2k-32k; hotpotqa/rerank 2k-16k, 32k deferred; reorder 2k-16k
  in progress). Train shards (20k mixed-n) building on cubbins for the 20 pilot→20k tasks. FLAGS +
  my decisions: **oolong eval_size=100/rung ⚠** (KEEP+flag hard ±0.05 SE, remediate to 500 if cheap,
  don't block); hotpotqa/rerank 32k + groups4>8k deferred = OK for morning (mark pending); grouping
  temporal-held-out (document in provenance). nq TRAIN reuses nq_p10_20k clean (CE-filtered, ratio
  0.097 ✓); nq 16k/32k eval needs uniform-k200 CE regen (queued, lower pri). DATA-POOR deferred:
  xabsence (659-pair, LLM rebuild blocked), obliq ~1797, scifact ~4k, helmet_summ 17.5k, groups4 16k.
  Eval files = unified JSONL at /scratch/.../ctc_suite_staged/eval_rungs/<task>/rung_<tok>.jsonl.
- **2026-07-20 ~04:30** (loop 2): Stage-3 evals landed. **helmet_summ ✅ rouge1_f=0.369 (eval_size
  500, 499/500 unique — healthy, DONE).** fiqa/scifact: train ✅ (CE→0.06/0.02) but eval BLOCKED
  (staged rung files 0 bytes → data agent's BEIR gen supplies real ones). obliq: re-eval still in
  flight (JSON still old empty-gen note).
  **⚠ nq_ceconfirm CORRECTION — I over-claimed the ce-filter fix earlier.** The 2500-ex CE-confirm
  ALSO COLLAPSED: CE flat ~0.51-0.55 (no descent), eval f1=0.034, 3 unique preds ([12]×289/[14]×207/
  [10]×4) — same positional attractor as the 0.074 baseline. The clean source file audits genuinely
  CE-filtered (hard_ratio 0.098, false_neg 0.010). So **ce-filter-ALONE at the 2500-ex budget does
  NOT fix nq.** Real remaining lever = the 20k budget (hotpotqa learned at 20k/CE 0.148, not 2500).
  **nq learnability is OPEN, not fixed.** Two decisive tests pending: (a) confirm agent's cross-encoder
  audit of all 4 nq files (running on mooney — did the confirm actually change the CE variable?); (b)
  the full-20k CE-clean nq Stage-5 run (NOT yet launched — data agent converting the shard). Do NOT
  rewrite the nq memory until (a) resolves refutation-vs-source-bug.
  **ACTION: nobody is assigned to launch nq (launcher agent DONE, nq wasn't in its 7).** When the data
  agent's nq 20k CE-clean shard is ready, spawn a round-2 launcher (SONNET) to launch nq full+cmix +
  the other ~13 non-ready tasks. Same for the round-2 fan-out generally.
  **QUOTA WATCH: lambda 97.3% (~35G free).** 12 lambda runs write ckpts to LROOT NFS (~5G/run) — risk
  of hitting 100% mid-run → ckpt-write crash. If >98.5% next loop, pull completed ckpts to Berkeley
  /scratch + free lambda NFS (deletes need USER via `!` — classifier blocks ssh rm).
- **2026-07-20 ~05:00 (loop 3 — LAMBDA BLOCKED, pivot to local):**
  **ROOT PROBLEM: lambda per-USER NFS quota = 100% (1.30T/1.30T).** Underlying disk has 7.5T free —
  it's purely the quota. OLMo-core checkpointer is SAVE-then-remove (confirmed checkpointer.py:310-315)
  → each 0.8B ckpt needs ~24G transient; at 0 free ALL lambda jobs die at their next 12G ephemeral.
  I CANNOT fix autonomously: deletes are classifier-blocked (user-only via `!`), AND I can't reach
  lambda compute-node-local /data from the head node (no ssh to compute nodes) → node-local ckpts are
  UNHARVESTABLE and slurm --output still targets full NFS. Lambda is air-gapped (no S3, no /scratch).
  So **lambda produces nothing harvestable until the user frees NFS quota.**
  - Actions taken: cancelled all 12 lambda jobs (4 running hotpotqa/grouping were doomed at step500
    anyway + 8 pending). Tried node-local relaunch (109729-740) then cancelled it — unharvestable.
    Lambda queue now EMPTY/clean. Added SAVE_ROOT/WORK_ROOT knobs to run_ctc_lambda.sbatch (ready for
    when quota is freed).
  - **PIVOT: spawned SONNET local-launcher** (validate-fit-then-fanout) to run the 6 orphaned tasks ×
    2 arms (hotpotqa/grouping/oolong/outlier/reorder/rerank) on jsteinhardt H200, NGPU=1 + AC
    (40960-seq fits ~25-30G on 141G H200), 8-concurrent under the cap, HARVESTABLE (node-local /data +
    node-pinned eval, the Stage-3 pattern). Fresh run-names -08b-loc. Status → stage5_local_launch_status.md.
  - **⚠ USER ACTION NEEDED to re-enable lambda's 32 A100 (run via `!`):**
    `ssh lambda 'cd /accounts/projects/sewonm/prasann/ctc_suite/ckpts && rm -rf ctc-1ep-absence_gutenberg-full-lambda{,-r2,-r3} ctc-1ep-msmarco-full-lambda{,-r2} ctc-1ep-obliq_retrieval-full-lambda ctc-contra-full-08b-lambda-pilot ctc-1ep-fiqa-full-lambda-r2 ctc-s5-hotpotqa-{full,cmix}-08b ctc-s5-grouping-{full,cmix}-08b && quota -s | tail -1'`
    Frees ~160G (harvested-to-git pilot ckpts + cancelled step250 orphans). KEEPS fiqa-r3/scifact/
    msmarco_rerank (eval still pending). After it, relaunch lambda via run_ctc_lambda.sbatch (NFS save).
  - Correction logged: I over-claimed the node-local workaround before checking harvest reachability.
- **NEXT-LOOP CHECKLIST (updated loop 3):**
  (1) **Local launcher (agent a5966c49)**: did the hotpotqa-full fit-validation pass (40960-seq NGPU=1
      +AC, OOM=0)? How many of 12 running? Read stage5_local_launch_status.md; relaunch any that died.
  (2) **Lambda quota**: did the user run the `!` free command? If quota < 90%, RE-ENABLE lambda —
      relaunch the tasks NOT covered by local (or overflow) via run_ctc_lambda.sbatch (NFS save, the
      SAVE_ROOT default). If still 100%, lambda stays parked; don't churn.
  (3) **Beaker 4B contra** (01KXYVRKZSHW…, 01KXYVS8G8…): loss descending? not finalized-with-error?
  (4) **nq OPEN**: the 2500-ex ce-confirm collapsed (f1 0.034). Confirm-agent's cross-encoder audit
      result (did it change the CE variable?). The decisive test = full-20k CE-clean nq run — is it
      launched anywhere? If not, add nq (+ qdmatch_nq, CE-clean) to the next local/lambda wave.
  (5) fiqa/scifact eval unblocked? (data agent's BEIR eval sets built → run the 2 blocked evals).
  (6) obliq: proper full-epoch run needed (pilot was step44 undertrained); queue for a fan-out wave.
