# CTC-suite 4B eval sweep: status (2026-07-20, in progress)

Batching is proven (`results/ctc_suite/batched_eval_status.md`). This tracks the actual sweep:
chunked-vs-dense f1 ladders for the 4B roster, using `run_rung_eval.py --batch-size 16` (8 for the
32768 rung, to bound KV-cache memory at 4B scale).

## Checkpoint availability (checked 2026-07-20)

| task | arm | ckpt status | location |
|---|---|---|---|
| contradiction | full | ready | `horton:/data/prasann/ctc_suite/ckpts_4b/ctc-s5-contra-full-4b` |
| contradiction | chunked-mix | ready | `horton:/data/prasann/ctc_suite/ckpts_4b/ctc-s5-contra-cmix-4b` |
| hotpotqa (task alias -> `retrieval` scorer) | full | ready | `mooney:/data/prasann/ctc_suite/ckpts/ctc-4b-hotpotqa-full` |
| hotpotqa | chunked-mix | **not trained yet** | Beaker `01KXZPJFFPHKMS0SRH2G9JB3P4` queued, not started |
| oolong | full | ready | `sneetches:/data/prasann/ctc_suite/ckpts/ctc-4b-oolong-full` |
| oolong | chunked-mix | **not trained yet** | Beaker `01KXZPJH8HBS9V5HJQH3CQZN4T` queued, not started |
| grouping | full + chunked-mix | **trained (exitCode=0), NOT harvested** | weka only: `ckpts/ctc-4b-grouping-{full,cmix}` |
| outlier | full + chunked-mix | **trained (exitCode=0), NOT harvested** | weka only: `ckpts/ctc-4b-outlier-{full,cmix}` |
| strmatch | full + chunked-mix | **trained (exitCode=0), NOT harvested** | weka only: `ckpts/ctc-4b-strmatch-{full,cmix}-07200501-...` |
| fiqa | full + chunked-mix | trained (exitCode=0) per weka, but **no eval-rung data staged at all** (`eval_rungs/fiqa/` empty on `/scratch`) | blocked on data staging, not just harvest |

**Blocker for grouping/outlier/strmatch**: weka is not reachable from Berkeley. Harvesting needs a
two-hop relay -- (1) a Beaker/gantry job that does a model-only resave of the weka checkpoint and
`aws s3 sync`s it to `s3://ai2-llm/checkpoints/prasanns/_transfer/<name>` (see `beaker.md`'s
weka-vs-S3 section + `debug/ctc_suite_4b_contra_eval/eval_contra_4b_local.sbatch`'s harvest step),
then (2) `aws s3 sync` from there to the target Berkeley node's local `/data`. Step (1) has only
been run for contradiction's two ckpts so far (confirmed via `aws s3 ls
s3://ai2-llm/checkpoints/prasanns/_transfer/` -- only `ctc-s5-contra-{full,cmix}-4b/` present).
Launching step (1) for grouping/outlier/strmatch needs a Beaker/gantry job (not just a local
sbatch) -- flagging rather than guessing at a new gantry recipe under sweep time pressure; this is
the next concrete action for whoever picks it up.

**fiqa** additionally needs its eval rungs generated (data-gen pipeline, not eval) before any harvest
even matters -- out of scope for this eval sweep.

## First-pass jobs launched (rungs 2048 / 8192 / 32768; hotpotqa capped at 16384 -- no 32768 staged)

All jobs use the native evaluators via `run_rung_eval.py`, `--batch-size 16` (8 at the 32768 rung),
`--tokenizer /scratch/users/prasann/hf_models/Qwen3.5-4B-Base`, Qwen3.5 marker ids
(248049/248050/248044). One 1×H200 job per (task, arm, rung); logs in
`debug/ctc_suite_4b_sweep/*.out`.

| task | arm | rung | node | job | status |
|---|---|---|---|---|---|
| contradiction | full | 2048 | horton | 3340773 | running/done — see table below once landed |
| contradiction | chunked-mix | 2048 | horton | 3340795 | running |
| contradiction | full | 8192 | horton | 3340796 | running |
| contradiction | chunked-mix | 8192 | horton | 3340798 | running |
| contradiction | full | 32768 | horton | 3340799 | queued (berkeleynlp 4-GPU cap) |
| contradiction | chunked-mix | 32768 | horton | 3340800 | queued (berkeleynlp 4-GPU cap) |
| hotpotqa | full | 2048 | mooney | 3340783 | running |
| hotpotqa | full | 8192 | mooney | 3340801 | running |
| hotpotqa | full | 16384 (cap) | mooney | 3340802 | running |
| oolong | full | 2048 | sneetches | 3340787 | running |
| oolong | full | 8192 | sneetches | 3340804 | running |
| oolong | full | 32768 | sneetches | 3340805 | running |

(First attempt at all 12 hit a `torchrun` default-master-port collision when >1 job landed on the
same node at once -- fixed in `debug/ctc_suite_4b_sweep/eval_one.sh` by deriving `--master-port`
from `$SLURM_JOB_ID`; resubmitted under the job IDs above.)

## Results (fills in as jobs complete)

| task | arm | rung | f1 (metric) | eval_size | eval_seconds | note |
|---|---|---|---|---|---|---|
| _pending_ | | | | | | first-pass jobs still running |

Every result also lands in `results/ctc_suite/all_results.jsonl` (per-line, with `git_commit`
provenance) via the same `results_io` path the driver already used for the 0.8B pilots -- no new
plumbing needed for this sweep, just point it at the 4B ckpts.

## Next actions

1. Let the 12 running/queued jobs finish; fill in the results table + ping with headline
   chunked-vs-dense numbers for contradiction/hotpotqa/oolong.
2. Second pass: rungs 4096/16384 for the same 3 tasks once first pass lands.
3. Someone with Beaker access needs to run the weka->S3 relay for grouping/outlier/strmatch (step 1
   above); once in `_transfer/`, harvesting to Berkeley + evaling is the same recipe already proven
   for contradiction.
4. fiqa needs eval-rung data generated before it can be evaluated at all.
5. Keep polling Beaker for hotpotqa-cmix / oolong-cmix (queued, not started) and pick them up once
   they land.

## EVAL PAUSED (main session, ~08:00) — GPU-STARVED, resume when GPUs free
- Batched native evaluator is BUILT + PROVEN (full arm bit-exact bs=1==bs=16; GatedDeltaNet cache_leftpad fix; --batch-size passthrough in run_rung_eval.py). Chunked arm may stay bs=1.
- BLOCKER: all Berkeley GPU nodes (horton/mooney/sneetches/cubbins/mcfuzz) saturated by OTHER labs → eval jobs preempted at `preemptive` QOS. Only contra full 2k evaluated (f1 0.571). Eval-sweep agent stopped (was burning budget wait-looping on preempted jobs).
- DONE ckpts awaiting eval: contra (both, on horton /data), hotpotqa/oolong (full, on mooney/sneetches /data), grouping/outlier/strmatch/fiqa (both, WEKA-only — need harvest weka→S3→Berkeley). More Beaker tasks completing hourly (all land on weka → need harvest).
- RESUME PLAN when GPUs free (Berkeley frees, or eval on Beaker/jupiter as training drains): harvest weka ckpts (model-only resave/`aws s3 sync` → S3 → Berkeley /data), then batched eval both arms, breadth-first (2k/8k/32k first). fiqa eval_rungs NOT staged (data gap).
- ALTERNATIVE if Berkeley stays starved: run eval as gantry jobs on jupiter (ckpts already on weka, no harvest; needs batched-eval code committed + eval sets on weka/S3).

## RESUMED (2026-07-20, GPUs freed) — fire-and-forget pass, no monitoring

GPUs freed up on horton/mooney/sneetches. Checked `squeue`: **the entire first-pass batch of 12
local jobs from the paused sweep never actually died** — they sat queued/preempted at
`preemptive`/`preemptive_high` QOS this whole time and started running (or already finished) on
their own the moment nodes freed. No relaunch was needed or performed for contradiction /
hotpotqa / oolong full-arm local ckpts — confirmed via `squeue -u prasann` + `sacct`:

| task | arm | rung | node | job | status (checked 2026-07-20) |
|---|---|---|---|---|---|
| contradiction | full | 2048 | horton | 3340773 | RUNNING (24m) |
| contradiction | chunked-mix | 2048 | horton | 3340795 | RUNNING (22m) |
| contradiction | full | 8192 | horton | 3340796 | RUNNING (22m) |
| contradiction | chunked-mix | 8192 | horton | 3340798 | **COMPLETED** — result in `all_results.jsonl` |
| contradiction | full | 32768 | horton | 3340799 | RUNNING (6m, just started) |
| contradiction | chunked-mix | 32768 | horton | 3340800 | PENDING (QOSGrpGRES — berkeleynlp 4-GPU cap still binding) |
| hotpotqa | full | 2048 | mooney | 3340783 | RUNNING (23m) |
| hotpotqa | full | 8192 | mooney | 3340801 | RUNNING (21m) |
| hotpotqa | full | 16384 (cap) | mooney | 3340802 | RUNNING (21m) |
| oolong | full | 2048 | sneetches | 3340787 | RUNNING (23m) |
| oolong | full | 8192 | sneetches | 3340804 | RUNNING (21m) |
| oolong | full | 32768 | sneetches | 3340805 | RUNNING (21m) |

No new local eval jobs launched this pass — all breadth-first (2k/8k/32k) cells for the
already-local ckpts (contra both arms, hotpotqa full, oolong full) were already in flight or done.
hotpotqa-cmix / oolong-cmix skipped per instructions (training not done). fiqa skipped (eval_rungs
not staged).

**Harvest job launched** (grouping/outlier/strmatch, both arms, weka → `s3://ai2-llm/checkpoints/prasanns/_transfer/`),
using the pre-built `debug/ctc_suite_4b_sweep/harvest_relay.sh` (dirnames already verified
exitCode=0 on Beaker as of today):

- gantry job: `ctc-harvest-relay-batch1-0a2b`, experiment `01KY00ZEA07FV7YW40X04151QG`
  (https://beaker.org/ex/01KY00ZEA07FV7YW40X04151QG), workspace `ai2/flex2`, priority `urgent`,
  `--gpus 0`, clusters neptune/ceres/saturn/jupiter.
- Covers: `ctc-4b-grouping-{full,cmix}`, `ctc-4b-outlier-{full,cmix}`, `ctc-4b-strmatch-{full,cmix}`
  (6 checkpoints, one `aws s3 sync` each).
- NOT launched: a Berkeley-side sbatch pull from S3→local /data — that's the next cycle's job,
  once this relay finishes (S3→weka relay was the only thing authorized/needed this pass; pulling
  the result down and launching those 6 eval cells is follow-up work).
- Did not re-audit the broader ~20-task Beaker roster for additional newly-finished cells beyond
  grouping/outlier/strmatch this pass (`beaker job list`/`experiment list` aren't usable
  off-node without `/etc/beaker/config.yml`; kept to the pre-verified three from
  `harvest_relay.sh`'s existing entries per the fire-and-forget budget). Next cycle: check for
  more done-on-weka tasks before the next harvest batch.

**Next cycle should**: (1) confirm the harvest relay finished (`aws s3 ls
s3://ai2-llm/checkpoints/prasanns/_transfer/` for grouping/outlier/strmatch dirs), (2) pull those
6 down to a free Berkeley node's local /data via the proven S3→local sbatch pattern
(`debug/ctc_suite_4b_contra_eval/eval_contra_4b_local.sbatch`), (3) launch their 2k/8k/32k eval
cells, (4) fill in results for the jobs above as they land in `all_results.jsonl`.

## HARVEST BLOCKER (bulk eval) — needs user, ~08:30
- FIRST-PASS eval (contra/hotpotqa/oolong — LOCAL ckpts, no harvest) is running/completing on the now-free Berkeley GPUs. Results landing in all_results.jsonl (qwen3.5-4b). These 3 tasks don't need harvest.
- BULK eval (grouping/outlier/strmatch + ~20 more Beaker-trained tasks) is BLOCKED on harvesting weka→Berkeley. The harvest MUST run as a gantry job (only Beaker reaches weka). `debug/ctc_suite_4b_sweep/harvest_relay.sh` has the correct ckpt list + sync loop, BUT:
  1. It failed exit 127 = gantry clones the PUSHED commit and the script is UNCOMMITTED → "No such file or directory". FIX: commit+push the script (I did NOT commit autonomously per the commit-only-when-asked rule), OR inline it.
  2. The script has no explicit AWS cred-file setup — unclear if the gantry env auto-configures creds (contra's harvest worked somehow; replicate that). Verify before trusting.
- USER OPTIONS when up: (a) approve committing harvest_relay.sh + re-run the gantry cmd in its header comment; or (b) switch bulk eval to run ON jupiter (ckpts already on weka, no harvest — needs the batched-eval code committed + eval sets on weka/S3). Batched evaluator is built + proven (full-arm bit-exact; GatedDeltaNet cache_leftpad fix) but UNCOMMITTED.
- NOTE: the batched-eval code (recurrent.py GatedDeltaNet fix + evaluator batching + run_rung_eval --batch-size) is valuable and uncommitted — worth committing regardless.

## HARVEST — EXACT DIAGNOSIS for user (stopped grinding ~09:35, budget)
- Attempt 2 (inline gantry, no commit): job 01KY04K6EJSGGNWHYH4DW8YV5Q ran but ALL 6 syncs failed with **`aws: command not found`** — the beaker-image `tylerr/olmo-core-tch291cu128-2025-11-25` has no aws CLI on PATH.
- FIX (trivial for you): re-run the weka→S3 harvest with EITHER (a) a beaker-image that has aws CLI (the OLMo-core release image / `OLMoCoreBeakerImage.stable` per the Dockerfile), OR (b) `pip install -q awscli` prepended in the job. Cred handling is still UNVERIFIED past that point — the inline wrote $AWS_CREDS/$AWS_CFG to ~/.aws/{credentials,config} and used `--profile S3`; confirm that matches your PRASANNS_AWS_CREDENTIALS/PRASANNS_AWS_CONFIG secret structure. The ckpt list (6 entries, grouping/outlier/strmatch both arms) + weka dirnames are in `debug/ctc_suite_4b_sweep/harvest_relay.sh`.
- Alternatively: point me at how the CONTRADICTION weka→S3 harvest was done (it succeeded — contra ckpts are on horton /data) and I'll replicate that exact recipe.
- STATE: first-pass eval (contra full+cmix ladders, hotpotqa/oolong full) landing on Berkeley GPUs — that's the autonomous partial deliverable. Bulk (~20 weka-only tasks) awaits this harvest fix. Batched evaluator built+proven+UNCOMMITTED (GatedDeltaNet cache_leftpad fix — worth committing).
