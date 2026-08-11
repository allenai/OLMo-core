# CTC-suite Stage 5 — 4B fan-out launch status (2026-07-19 ~23:00 PDT)

Owner: 4B fan-out coordinator (this session). Pivot: **4B for ALL tasks, both arms** (was 0.8B).
Deadline ~20h, soft. This file is the single source of truth for what's running / what's left —
picks up where `stage5_wave2_launch_status.md` (0.8B, superseded) and `nq_debug_diagnosis.md`
(closed, handed off) left off.

## Actions taken this session
1. Read handoff docs, confirmed Beaker 4B contradiction (full+cmix) HEALTHY (started ~04:15,
   not yet finalized-with-error) — untouched, not relaunched.
2. Confirmed the correct **HYBRID 4B base** (`q35-4b-base-modelonly`, vocab 248320,
   `marker_audit.json` verdict **PASS**, cos/norm-ratio in range) is already on **weka**
   (`/weka/oe-training-default/ai2-llm/checkpoints/prasanns/ctc_suite/bases/q35-4b-base-modelonly`)
   and on **S3** (`s3://ai2-llm/checkpoints/prasanns/ctc_suite/bases/q35-4b-base-modelonly`) — this
   is the base the running contra-4B job already reads. Do NOT restage.
3. **Cancelled the 2 running wave-2 0.8B local jobs** (`3338930` hotpotqa-full/sneetches,
   `3338954` msmarco-full/mooney) per coordinator prompt's explicit permission, to free 2 full H200
   nodes for 4B (0.8B wave-2 is superseded by the 4B pivot; the 30 other wave-2 0.8B jobs from the
   HOLD were already gone from squeue before this session — never need cancelling).
4. **Beaker 4B launched (jupiter, 1-node, urgent, `--epochs 1`)** — shards+base already on weka:
   - `grouping` full: exp `01KXZ1MZ2R17R93S26XA02ZW7G`, seq_len 36864
   - `grouping` chunked-mix: exp `01KXZ1N0R98592VK36YJW32VND`, seq_len 36864
   - `outlier` full: exp `01KXZ1N026HV46ACZ26J7W4G8S`, seq_len 40960
   - `outlier` chunked-mix: exp `01KXZ1MZBP625XR7WSC15N8PQ8`, seq_len 40960
   - wandb: `https://wandb.ai/prasanns-allen-institute-for-ai/memory-networks/groups/ctc-suite-grouping`
     and `.../groups/ctc-suite-outlier`
5. **Local 4B launched** (jsteinhardt, freed nodes, NGPU=8, ACT_CKPT=full, SHARD_DEGREE=8) — used a
   `sbatch --wrap` (base pulled from S3 into node-local `/data` at job start, since `/scratch` is
   **100% quota-full**, see blocker below). ⚠ **First attempt (jobs 3339023/3339024) failed in <1s**
   — the wrapper scripts were written to `/tmp` on the head node, which is NOT shared with compute
   nodes (`bash: /tmp/....sh: No such file or directory`). Fixed by moving the wrapper scripts into
   the repo (`debug/ctc_suite_4b_fanout/local_launch_<task>.sh`, NFS-shared) and resubmitting — verify
   any future one-off `sbatch --wrap` local launch uses a repo/`/scratch`/`/accounts` path, never
   `/tmp`, for anything the wrapper references by path.
   - `hotpotqa` full: job **3339031** on **mooney** (RUNNING, confirmed pulling the 10.4G base from
     S3 at ~65 MiB/s), seq_len 26112, `RUN=ctc-4b-hotpotqa-full`
   - `oolong` full: job **3339032** on **sneetches** (RUNNING, same base pull), seq_len 40448,
     `RUN=ctc-4b-oolong-full`
   - Wrapper scripts: `debug/ctc_suite_4b_fanout/local_launch_hotpotqa.sh`,
     `debug/ctc_suite_4b_fanout/local_launch_oolong.sh`.
   - Logs: `debug/ctc_suite_4b_fanout/local_hotpotqa_full_3339031.log`,
     `debug/ctc_suite_4b_fanout/local_oolong_full_3339032.log`.
6. **Staged all 21 remaining task shards /scratch → S3** (background loop, **completed, all rc=0**,
   log `debug/ctc_suite_4b_fanout/s3_shard_sync_all.log`): hotpotqa, niah, oolong, helmet_qa,
   helmet_summ, scifact, fiqa, msmarco, rerank, outlier_amzn, grouping_labeled, textgroups,
   absence_gutenberg, strmatch, qdmatch_hpqa, qdmatch_nq, mathmatch, reorder, cycle, groups4,
   obliq_retrieval, nq. (hotpotqa/oolong were synced too even though already running locally, for
   S3-side completeness — harmless.)
7. **Kicked off the S3 → weka gantry sync** for all 21 of the above in one job (not yet confirmed
   done — CHECK BEFORE launching Beaker jobs for these tasks):
   `gantry` experiment **`01KXZ1PJDV3957SVGZX38AXTAM`** (`ctc4b-weka-sync-remaining`, jupiter,
   urgent, `--timeout 0`, not followed). Verify with:
   `beaker experiment get 01KXZ1PJDV3957SVGZX38AXTAM --format json` (check `jobs[-1].status` has
   `exited`+`exitCode:0`), then `beaker experiment logs 01KXZ1PJDV3957SVGZX38AXTAM | tail -50` and
   grep for `rc=0` on all 21 tasks / `ALL WEKA SYNCS DONE`.

## Blockers hit / notes
- **`/scratch/users/prasann` is at 100% quota (500G/500G)** — cannot write anything new there
  (deletes are classifier-blocked, user-only via `!`). This blocked the originally-planned
  cubbins-base → `/scratch` staging path; worked around by pulling the base **directly from S3**
  into each node's local `/data` inside the training job itself (compute nodes DO have outbound
  internet — confirmed via a quick `srun` + `curl` to S3, http 403 i.e. reached-but-unauthenticated-
  URL, then `aws s3 sync` with the `S3` profile succeeded). **User should free `/scratch` quota when
  free** — several other pipelines (eval rungs, data builds) will hit this same wall.
- Compute-node-to-compute-node `ssh` is blocked by `pam_slurm_adopt` unless you have an active job
  on BOTH ends — ruled out node-to-node rsync (e.g. cubbins → mooney) as a transfer path; the
  S3-pull-inside-the-job pattern above is the reusable workaround for any future local 4B/9B launch
  while `/scratch` stays full.
- **Lambda: NOT attempted this session** (explicit call, given usage constraints) — lambda quota was
  tight (**91.6%**, `1.19T/1.30T`) at that time. **Follow-up session (below, "### 4. Lambda") checked
  again and found quota had climbed further to 99.0% — bailed before any transfer.**
- `beaker workspace experiments` / `beaker experiment list` do not exist in this CLI version — no
  quick way to enumerate recent experiments; had to grep launcher stdout logs for the `Experiment:`
  line instead (all 4 Beaker launches confirmed submitted this way).

## Currently running / launched (4B, this session + carried over)

| task | arm | cluster | id | seq_len | shard_n | status |
|---|---|---|---|---:|---:|---|
| contradiction | full | Beaker jupiter | 01KXYVRKZSHW5XGF7JRX9A78R2 | 40960 | 19366 | RUNNING (pre-existing) |
| contradiction | chunked-mix | Beaker jupiter | 01KXYVS8G8CT86T8ZYS1HATM1E | 40960 | 19366 | RUNNING (pre-existing) |
| grouping | full | Beaker jupiter | 01KXZ1MZ2R17R93S26XA02ZW7G | 36864 | 20000 | job created |
| grouping | chunked-mix | Beaker jupiter | 01KXZ1N0R98592VK36YJW32VND | 36864 | 20000 | job created |
| outlier | full | Beaker jupiter | 01KXZ1N026HV46ACZ26J7W4G8S | 40960 | 19986 | job created |
| outlier | chunked-mix | Beaker jupiter | 01KXZ1MZBP625XR7WSC15N8PQ8 | 40960 | 19986 | job created |
| hotpotqa | full | local (mooney) | slurm 3339031 | 26112 | 20000 | RUNNING (base pull in progress) |
| oolong | full | local (sneetches) | slurm 3339032 | 40448 | 21000 | RUNNING (base pull in progress) |

Verify Beaker jobs with `beaker experiment get <id> --format json` (`jobs[-1].status`); verify local
with `squeue -u prasann` and `tail -f debug/ctc_suite_4b_fanout/local_<task>_full_<jobid>.log`.

## TODO — NEXT (post usage-reset)

### 1. Verify the weka sync finished
`beaker experiment logs 01KXZ1PJDV3957SVGZX38AXTAM | grep -E "rc=|ALL WEKA"` — expect 21× `rc=0` +
`ALL WEKA SYNCS DONE`. If any task shows non-zero, re-run just that task's sync (same one-liner,
single task) before launching it on Beaker.

### 2. Launch remaining Beaker 4B jobs (jupiter, `--num-nodes 1 --epochs 1 --priority urgent`)
Use `beaker_ctc_suite.py` exactly as in step 4 above (defaults already point at
`shards/<task>_train` and the hybrid base on weka — no `--data-root`/`--base-checkpoint` overrides
needed once weka sync #1 confirms). **hotpotqa/oolong only need the `chunked-mix` arm** (full is
already running locally); every other task below needs **both** arms.

seq_len table (ceil to nearest 512 of each shard's `max_example_len`; shard_n = num_instances):

| task | seq_len | shard_n | note |
|---|---:|---:|---|
| hotpotqa (cmix only) | 26112 | 20000 | full already local |
| niah | 35840 | 20000 | |
| oolong (cmix only) | 40448 | 21000 | full already local |
| helmet_qa | 40960 | 19762 | |
| helmet_summ | 32256 | 17500 | ⚠ data-poor (govreport cap) |
| scifact | 34816 | 4045 | ⚠ data-poor (800-query BEIR cap) |
| fiqa | 26112 | 20000 | |
| msmarco | 29696 | 20000 | |
| rerank | 31232 | 20000 | (= msmarco_rerank, launch ONCE, not both names) |
| outlier_amzn | 40960 | 19001 | |
| grouping_labeled | 37888 | 20000 | |
| textgroups | 40960 | 17022 | |
| absence_gutenberg | 40960 | 9610 | ⚠ small (Gutenberg pool cap) |
| strmatch | 19968 | 20000 | |
| qdmatch_hpqa | 29696 | 20000 | |
| qdmatch_nq | 33792 | 20000 | decisive nq test, see nq_debug_diagnosis.md |
| mathmatch | 25088 | 20000 | |
| reorder | 40960 | 19944 | train long / eval capped 2k-16k (documented decision, no rebuild) |
| cycle | 10240 | 20000 | eval cap 2k-16k (O(N³)) |
| groups4 | 12800 | 15000 | eval cap 2k-16k (O(N³)); ⚠ documented cap |
| obliq_retrieval | 40448 | 1736 | ⚠ data-poor (~1797 pool cap, smallest/fastest run) |
| nq | 33280 | 19967 | decisive test, CE-clean, see nq_debug_diagnosis.md |

Template per task/arm:
```
python -u src/scripts/train/memexpress/ctc_suite/beaker_ctc_suite.py \
  --task <task> --variant <full|chunked-mix> --model-scale 4b \
  --run-name ctc-4b-<task>-<full|cmix> --num-nodes 1 --epochs 1 \
  --seq-len <seq_len from table> --priority urgent launch
```
Run each in background (`nohup ... > debug/ctc_suite_4b_fanout/launch_<task>_<arm>.log 2>&1 &
disown`), then grep the log for `Experiment:` to get the ID — `launch_config.launch(follow=True)`
streams logs forever otherwise, and a plain `timeout N python ... | grep ...` pipe was unreliable
in this session (silently exceeded the tool's 120s window with no captured output more than once —
prefer the nohup-to-file pattern, it worked cleanly every time).

That's 20 tasks × ~2 arms − 2 (hotpotqa/oolong single-arm) = **38 Beaker launches** left.

### 3. Local 4B (jsteinhardt/berkeleynlp H200) — cmix arms + more capacity as it frees
- `hotpotqa`-chunked-mix and `oolong`-chunked-mix can go local too (same S3-pull-into-`/data`
  pattern as step 5 above) once mooney/sneetches free up, or route them through Beaker per the
  table above — coordinator's call, don't double-launch whichever way you pick.
- Any node showing free GPUs (`scontrol show node | grep -E "NodeName|AllocTRES"`) is fair game;
  reuse the `sbatch --wrap` pattern from this session (S3 base pull guarded by a `.metadata` check,
  `ACT_CKPT=full SHARD_DEGREE=8 NGPU=8`, `DATA_SRC=/scratch/users/prasann/ctc_suite_staged/shards/
  <task>_train` — reads from `/scratch` are fine, only writes are quota-blocked).

### 4. Lambda — fit-test then fan-out — **ATTEMPTED 2026-07-19 evening, BAILED at step 1 (quota)**

**Quota check (before any transfer):** `ssh lambda 'quota -s'` → **99.0% (1.29T/1.30T)**, not the
91.6% recorded at the end of the previous session — it climbed ~7.4 points since then (other lambda
work in the interim). This is already far past the ~93% stop-line in the task brief, before touching
the base at all. **Did not transfer the base, did not fit-test, did not fan out anything** — the
brief was explicit that a too-tight quota is a bail condition, and 99% with no margin for even one
running 4B checkpoint (~24G) is not a judgment call.

`du -sh` breakdown for context (nothing deleted — destructive cleanup needs the user via `!`, per
[[lambda-cluster-ctc]]):
- `/accounts/projects/sewonm/prasann/ctc_suite/` = 241G total: `ckpts/` **226G** (candidate for
  cleanup — likely stale 0.8B/2k-pilot checkpoints), `data/` 12G, `bases/` 3.8G.
- `/accounts/projects/sewonm/prasann/projects/` = 409G total: `corpus-reasoning/` 333G,
  `prasannfirstphd/` 70G.
- Underlying filesystem itself has headroom (`df`: 43% used, 7.5T avail on the pool) — this is
  purely the **per-user NFS quota** (1.30T cap), matching the documented lambda trap.

**Recommendation:** ask the user to free lambda quota (the 226G `ctc_suite/ckpts/` is the obvious
first target — old checkpoints, not code/data) before retrying. Once quota has real headroom (say
back under ~85%), re-run this same plan:
1. Transfer the 4B hybrid base (~13G) via tar-pipe cubbins → lambda (command unchanged from prior
   plan, see git history of this file) or head-mediated S3 pull if cubbins ssh is unavailable.
2. Fit-test: `obliq_retrieval` (seq_len 40448, shard_n=1736, cheapest), `--max-steps 20
   ACT_CKPT=full SHARD_DEGREE=8`, QOS=preemptive, 8×A100, `run_ctc_lambda.sbatch` with
   `BASE_SRC=$LROOT/bases/q35-4b-base-modelonly SCALE=4b`. A100/80G at 4B@40960 is still untested —
   this remains the open question.
3. If it fits: fan out remaining Beaker-queue overflow onto lambda's other 3 nodes.
   If it OOMs: report and stop — lambda can't do 4B.

**Nothing was launched on lambda this session.** No base transferred, no jobs submitted, no changes
to lambda's filesystem.

### 5. Once all launched
Aggregate this table with `stage5_wave2_launch_status.md`'s dead 0.8B rows removed/superseded, and
hand off to the eval agent (native torchrun per-rung eval, NOT this coordinator's job — do not run
evals here per the task brief).

## Wandb group links
- contradiction: `https://wandb.ai/prasanns-allen-institute-for-ai/memory-networks/groups/ctc-suite-contradiction`
- grouping: `https://wandb.ai/prasanns-allen-institute-for-ai/memory-networks/groups/ctc-suite-grouping`
- outlier: `https://wandb.ai/prasanns-allen-institute-for-ai/memory-networks/groups/ctc-suite-outlier`
- hotpotqa (local, personal entity): `https://wandb.ai/prasann-uc-berkeley-electrical-engineering-computer-sciences/memory-networks/groups/ctc-suite-hotpotqa`
- oolong (local, personal entity): `https://wandb.ai/prasann-uc-berkeley-electrical-engineering-computer-sciences/memory-networks/groups/ctc-suite-oolong`

## BEAKER 4B fan-out — all 42 remaining launches (created 2026-07-19T23:18, queued behind jupiter capacity, urgent)
| task_arm | beaker_id |
|---|---|
| absence_gutenberg_cmix | 01KXZ2HFHCMZMNXKXDFAVFZGRE |
| absence_gutenberg_full | 01KXZ2HEJYRC3MWAR6DWP4HBSG |
| cycle_cmix | 01KXZ2JKW8N86QYDHCB7EABSZX |
| cycle_full | 01KXZ2JGNY9G1RD8T76XF4EW3J |
| fiqa_cmix | 01KXZ2GECTWC5XJY5SYSBDDQNH |
| fiqa_full | 01KXZ2GBFKN8Q0P08VW7ZAZA3Q |
| grouping_labeled_cmix | 01KXZ2H3M0Q68B1RPAGAT4NNDF |
| grouping_labeled_full | 01KXZ2H0KT4JDRPKWBWQG9KV53 |
| groups4_cmix | 01KXZ2JV0AX543CV1AJ517KD7W |
| groups4_full | 01KXZ2JQAWD36A4N2BMR0NARZF |
| helmet_qa_cmix | 01KXZ2FRH7FDMB57X68RWXB96G |
| helmet_qa_full | 01KXZ2FRFG9KSA8TG7Y41C2SS2 |
| helmet_summ_cmix | 01KXZ2G15D22DJE3D35W9VMBYT |
| helmet_summ_full | 01KXZ2FXBF762T8P014AXJ23Y9 |
| hotpotqa_cmix | 01KXZ2K6ANC4Q1BVHVRHGCX8SZ |
| mathmatch_cmix | 01KXZ2J671TCGNHTNA4QXXWXC8 |
| mathmatch_full | 01KXZ2J3RPNS29V14SE14M7H13 |
| msmarco_cmix | 01KXZ2GKQZ212ABA9J2SVJPRZ6 |
| msmarco_full | 01KXZ2GGG90R0B6T6RNS09MVQT |
| niah_cmix | 01KXZ2FREXR8JJ02W630HEED6Y |
| niah_full | 01KXZ2FRRXHQQMRF6TKP2M49A7 |
| nq_cmix | 01KXZ2K2VP0J02R2X7QBPQH6F6 |
| nq_full | 01KXZ2K2AMZBT9H35C1N9M6122 |
| obliq_retrieval_cmix | 01KXZ2JZBWN886K34Y16YDTCGX |
| obliq_retrieval_full | 01KXZ2JXVC6XXNBP9P8P862JN9 |
| oolong_cmix | 01KXZ2K7MS6MN1FF81ECCSNQ1Z |
| outlier_amzn_cmix | 01KXZ2GY88J6YN70A38RKC3SAJ |
| outlier_amzn_full | 01KXZ2GVJ9GT16ZEFFDHV22D9D |
| qdmatch_hpqa_cmix | 01KXZ2HV8JWHJDM3QNN99PNT73 |
| qdmatch_hpqa_full | 01KXZ2HQC4F12V9CYFYYHGJ21F |
| qdmatch_nq_cmix | 01KXZ2J0M7WMHW1842AYMRPP6N |
| qdmatch_nq_full | 01KXZ2HXMGN91T1SFJNG33J8ZG |
| reorder_cmix | 01KXZ2JCFCZCFKEW1S0MVQNHH3 |
| reorder_full | 01KXZ2J9J5XEKNYNCT6PHN0MQ4 |
| rerank_cmix | 01KXZ2GRX6FVE48P3Q25ZJBNY4 |
| rerank_full | 01KXZ2GPECBGVW224KYGFDF1XE |
| scifact_cmix | 01KXZ2G8BYRXKJZQBSPWS54EDT |
| scifact_full | 01KXZ2G4X1N5EYQZJPW7EB2Z0D |
| strmatch_cmix | 01KXZ2HMXESZ12B1D2HF64NKCC |
| strmatch_full | 01KXZ2HHGATDMFTZG1B51EQR5Y |
| textgroups_cmix | 01KXZ2H982WMYPS5BYJKXVPJM6 |
| textgroups_full | 01KXZ2H5RFWV9Z79XTRH06PV4P |

## LAMBDA 4B MIGRATION (driven by main session after agent stalled)
- Fit-test VERDICT: **4B FITS on A100/80G** (obliq fit-test 109754 reached step 20/217, ~11s/step, no OOM; full-shard+AC).
- Base (19G) + full-20k shards for the whole roster are staged on lambda. Quota 94.9% (70G free) → ~3 concurrent 4B ckpts (~19G each) safe.
- LAUNCHED on lambda (QOS=preemptive, SCALE=4b, ACT_CKPT=full SHARD_DEGREE=8, BASE_SRC=$L/bases/q35-4b-base-modelonly):
  - 109755 ctc-4b-cycle-full-lam (seq 10240)
  - 109756 ctc-4b-cycle-cmix-lam (seq 10240)
  - 109757 ctc-4b-strmatch-full-lam (seq 19968)
- ⚠ Beaker copies of these 3 (cycle full/cmix 01KXZ2JGNY9G.../01KXZ2JKW8N8..., strmatch_full 01KXZ2HHGATD...) are KEPT QUEUED as backup (lambda=preemptible). CANCEL them once the lambda runs are confirmed stepping past compile.
- Lambda 4B is SLOW (~8h/run on A100 vs ~4h Beaker H100) + quota-limited to ~3 runs; 4th node idle for quota headroom. Free the 68G 0.8B KEEP ckpts (user `!`) to add more.

## LAMBDA 4B — PARKED (quota wall, reverted to Beaker)
- The 3 lambda 4B runs (cycle f/c, strmatch f) STEPPED fine (4B trains on A100) but the lambda code path writes PERIODIC checkpoints (~18G each): quota went 94.9%→99.2% by step 250. At ~180G/run, lambda 4B is NOT viable at current quota. CANCELLED 109755/756/757.
- NO work lost — their Beaker backups (cycle full/cmix, strmatch full) remain queued and will run on Beaker.
- To revive lambda 4B: (1) user frees quota (68G KEEP 0.8B ckpts + the cancelled jobs' partial ckpts, via `!`), AND (2) sync current train_ctc_suite.py to lambda so it writes model-only final ckpt (no periodic saves) — then a 4B run needs only ~19G.
- Until then: Beaker (full roster, draining at urgent) + local H200 (hotpotqa/oolong) carry everything.
