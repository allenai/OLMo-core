# CTC-suite Stage 5 — full-scale joint training launch status

Launched overnight 2026-07-19/20 (autonomous). Scope: the 7 tasks with ready ~20k mixed-n
qwen3_5 training shards at `/scratch/users/prasann/ctc_suite_staged/shards/<task>_train`, both
arms (`full` + `chunked-mix`), 1 epoch. 0.8B for all EXCEPT contradiction → 4B.

Run-name convention: `ctc-s5-<task>-<arm>-<scale>` (arm: `full` | `cmix`=chunked-mix). Fresh
save-folder per run (auto-resume trap).

## (2) Seq-fit / OOM validation result — RESOLVED

**40960-seq at 0.8B OOMs on 80GB A100 without activation checkpointing.** First validation run
(hotpotqa full, job 109705) OOM'd at step 1 — 77.7 GB allocated, 0 free. The trainer's default
for 0.8B is `shard_degree=1` / no-AC, which was tuned for **141 GB H200s**, not lambda's 80 GB
A100s. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (already set) did NOT save it — genuine
capacity, not fragmentation.

**Fix (settled config, now the lambda launcher default):** `seq_len=40960` +
`--activation-checkpointing full` + `--shard-degree 8` (full FSDP param sharding across the 8
A100s). Re-run (hotpotqa full, job 109706) applied full AC and trained past the OOM point cleanly
(OOM count = 0). Added `ACT_CKPT`/`SHARD_DEGREE`/`QOS` env knobs to
`run_ctc_lambda.sbatch` (defaults `full`/`8`).

- seq_len stays 40960 for ALL tasks (did NOT drop to 32768 — AC fixed it, and 4 shards have
  max_example_len ≈ 40957 so 32768 would trip the `seq_len ≥ max_example_len` guard anyway).
- ✅ chunked-mix arm at 40k ALSO confirmed to fit (hotpotqa-cmix 109718 + grouping-cmix 109720
  train past init, OOM=0; curriculum anneal reaches final_p≈0.0003≈0, hard-fail guard passed).
- Throughput (hotpotqa-full, 8×A100, full AC): MFU 36.4%, TPS 11,543/device (~92k/node),
  CE loss 0.44 & falling, PPL 1.55. → ~2.5 h/run for a 2500-step 0.8B run. 12 runs / 4 nodes ≈
  3 waves ≈ 7.5 h — all finish by morning, well inside the 12 h walls.

## (3) 4B base status for contradiction — READY

- `/scratch/.../cpt_mix_ckpts/q4b-base-modelonly` is the WRONG base (old Qwen3-4B: vocab 151936,
  standard attn — cannot represent qwen3_5 markers 248049/248050). Do NOT use it.
- Correct q35-4b base: cubbins node-local `/data/prasann/ctc_suite/bases/q35-4b-base-modelonly`
  (audited PASS: cos(box)=0.226, norm ratios 0.65×/0.95×). Also on S3
  `s3://ai2-llm/checkpoints/prasanns/ctc_suite/bases/q35-4b-base-modelonly` and synced to weka
  (smoke test). contradiction runs 4B on **Beaker jupiter** (base+shard read from weka).
- contradiction shard uploaded to S3 + S3→weka sync launched (exp
  `01KXYVKDQ6Z5K9GKD7M18KNFJ1`). 4B training launches once that sync lands.

## LDATA / lambda notes
- Lambda QOS `preemptive_high` caps a user at **8 GPUs = 1 node**. `preemptive` and `normal` have
  NO GPU cap → all 0.8B lambda runs submitted under `QOS=preemptive` to use all 4 usable nodes
  (hyperplane01/02/03/05; 04 is `down`). Physical max concurrency on lambda = 4 nodes → 12 runs
  cycle in 3 waves.
- All 7 shards staged to `LROOT=/accounts/projects/sewonm/prasann/ctc_suite/data/<task>_train`
  (verified num_instances: contradiction 19366, grouping 20000, hotpotqa 20000, oolong 21000,
  outlier 19986, reorder 19944, rerank 20000).

## ⚠ reorder seq_len conflict (FLAG)
Plan says reorder → seq_len 20480 (16k cap). But the staged reorder shard has
`max_example_len=40957` (built to the 2k–32k range, not 2k–16k), and the trainer hard-fails when
`seq_len < max_example_len`. So reorder is running at **seq_len 40960** to avoid a guard crash /
data loss. Training on up-to-32k contexts while eval is capped at 16k is benign, but if the 16k
policy must hold at train time, the **reorder shard needs a rebuild to n∈[n(2k), n(16k)]** (data
owned by another agent).

## (1) Launch table

| task | arm | scale | pool | jobid / exp | seq_len | run-name | status |
|---|---|---|---|---|---|---|---|
| hotpotqa | full | 0.8B | lambda-05 | 109706 | 40960 | ctc-s5-hotpotqa-full-08b | RUNNING (validated) |
| hotpotqa | cmix | 0.8B | lambda-01 | 109718 | 40960 | ctc-s5-hotpotqa-cmix-08b | RUNNING |
| grouping | full | 0.8B | lambda-02 | 109719 | 40960 | ctc-s5-grouping-full-08b | RUNNING |
| grouping | cmix | 0.8B | lambda-03 | 109720 | 40960 | ctc-s5-grouping-cmix-08b | RUNNING (cmix probe) |
| oolong | full | 0.8B | lambda | 109721 | 40960 | ctc-s5-oolong-full-08b | QUEUED |
| oolong | cmix | 0.8B | lambda | 109722 | 40960 | ctc-s5-oolong-cmix-08b | QUEUED |
| outlier | full | 0.8B | lambda | 109723 | 40960 | ctc-s5-outlier-full-08b | QUEUED |
| outlier | cmix | 0.8B | lambda | 109724 | 40960 | ctc-s5-outlier-cmix-08b | QUEUED |
| reorder | full | 0.8B | lambda | 109725 | 40960* | ctc-s5-reorder-full-08b | QUEUED |
| reorder | cmix | 0.8B | lambda | 109726 | 40960* | ctc-s5-reorder-cmix-08b | QUEUED |
| rerank | full | 0.8B | lambda | 109727 | 40960 | ctc-s5-rerank-full-08b | QUEUED |
| rerank | cmix | 0.8B | lambda | 109728 | 40960 | ctc-s5-rerank-cmix-08b | QUEUED |
| contradiction | full | 4B | Beaker jupiter | ex 01KXYVRKZSHW5XGF7JRX9A78R2 | 40960 | ctc-s5-contra-full-4b-20260719T211516 | SCHEDULED (2 nodes) |
| contradiction | cmix | 4B | Beaker jupiter | ex 01KXYVS8G8CT86T8ZYS1HATM1E | 40960 | ctc-s5-contra-cmix-4b-20260719T211538 | SCHEDULED (2 nodes) |

`*` reorder seq_len forced to 40960 (see conflict flag above).

Beaker 4B: global_batch=16 (2 nodes × 8), 1 epoch → 1250 steps. base+shard read from weka
(contradiction shard synced via exp 01KXYVKDQ6Z5K9GKD7M18KNFJ1, exitCode 0; q35-4b base already
on weka from the smoke test). wandb group:
https://wandb.ai/prasanns-allen-institute-for-ai/memory-networks/groups/ctc-suite-contradiction

## Live status (last poll 2026-07-20 ~04:17 UTC)
- **All 14 runs launched.** Lambda: 4/4 nodes RUNNING (hotpotqa full+cmix, grouping full+cmix —
  all healthy, CE loss falling, OOM=0), 8 QUEUED (oolong/outlier/reorder/rerank ×2) — cycle in as
  nodes free (~2.5 h/run). Beaker: 2× contra-4B SCHEDULED on jupiter (urgent).
- ⚠ TODO to confirm at next poll: the 2 Beaker 4B jobs pass step 1 without OOM (4B@40k, 2 nodes,
  FSDP full-shard + full AC — plan says proven, but not yet observed on jupiter this run).
- ⚠ lambda `preemptive` QOS runs CAN be preempted if another user submits `preemptive_high`;
  lambda is nearly idle so risk is low. Checkpoints are saved (`--save-checkpoint`); a preempted
  run resumes from its last checkpoint on requeue.
- Loss curves: full per-step curve to be log-parsed into each run's result JSON at run end (logs
  mirrored to /scratch/users/prasann/ctc_suite_logs/ on lambda; wandb offline on lambda so the
  log is the authoritative record — keep it).
