# CTC-suite Stage 5 Wave-2 — training launch status

Owner: wave-2 training launcher agent. Scope: the 16 remaining roster tasks NOT already covered
by the pre-existing lambda 6-task wave (grouping, hotpotqa/"retrieval", oolong, outlier, reorder,
rerank) and NOT owned by the separate nq-debug agent (nq, qdmatch_nq) or Beaker contra (4B).

**⏸ HOLD (coordinator, mid-launch):** user wants runs MOSTLY at 4B scale, not 0.8B — the 0.8B wave-2
plan may be changing. **All 32 jobs below were already submitted at 0.8B before the HOLD landed.**
Per coordinator instruction: do NOT cancel anything running; do NOT submit anything new until the
scale is confirmed. This file is accurate as of the HOLD; next action is a coordinator go/no-go.

## Skipped per roster rules
- **msmarco_rerank** — metadata is byte-identical to the already-launched `rerank` shard (same
  task field, same num_instances=20000, same num_tokens=306,740,752, same max_example_len=31135;
  `convert_ctc_final_dense_cubbins.sbatch` builds it from the exact same
  `rerank_pool/msmarco_trainhn_train_k20-315_20000.jsonl` input). Not distinct — not launched.
- **xabsence** — no shard exists (`xabsence_train/` missing on staged dir; LLM-paraphrase-pool
  rebuild still blocked per data-status). Blocked, not launched.
- **nq, qdmatch_nq, contradiction** — owned by other agents, untouched (confirmed still present
  in squeue under other job names, not mine).

## Shard readiness audit (done before launch)
All 16 below have `metadata.json` present, `marker_set=qwen3_5`. Several are below the "~18-21k"
rule-of-thumb but were confirmed to be the **real, final** shard (no generation job running to grow
them further; drops match documented pool caps), not 2500-instance pilot stubs — launched with a
size flag inline:

| task | num_instances | note |
|---|---:|---|
| msmarco | 20000 | clean |
| grouping_labeled | 20000 | clean |
| niah | 20000 | clean |
| scifact | 4045 | ⚠ data-poor (800-query BEIR cap, documented) |
| fiqa | 20000 | clean |
| outlier_amzn | 19001 | clean |
| textgroups | 17022 | 2978 dropped (>seq cap), real pool exhausted |
| absence_gutenberg | 9610 | ⚠ small (217 dropped; Gutenberg sentence-diff pool cap) |
| strmatch | 20000 | clean |
| qdmatch_hpqa | 20000 | clean |
| mathmatch | 20000 | clean |
| cycle (16k-cap task) | 20000 | clean |
| groups4 (16k-cap task) | 15000 | ⚠ documented cap (n900 infeasible) |
| helmet_qa | 19762 | clean |
| helmet_summ | 17500 | ⚠ documented cap (govreport train cap) |
| obliq_retrieval | 1736 | ⚠ data-poor (documented ~1797 pool cap; smallest/fastest run) |

## Compute-allocation finding (important for next wave)
`preemptive_high` QOS has a **cluster-wide per-user cap of gres/gpu=8** (confirmed via
`sacctmgr show qos`) — already fully consumed by the pre-existing `ctc-s5-hotpotqa-full-08b-loc`
job (job 3338930, running on sneetches since before this agent started). Plain `preemptive` QOS has
**no such cap** (confirmed empty `GrpTRES`/`MaxTRESPU`) and is available on both `jsteinhardt`
(`ACCOUNT=site`) and `berkeleynlp` (`ACCOUNT=site`) partitions — smoke-tested (job 3338951,
cancelled after confirming clean RUNNING start on mooney) and used for all 32 launches below.
Physical H200 capacity was checked via `scontrol show node` (AllocTRES/CfgTRES, no ssh needed):
at launch time only **cubbins had 1 free GPU and lorax had 3 free GPUs**; mcfuzz/mooney/sneetches/
horton were fully allocated (shared with many other lab users' jobs). So most of the 32 full-node
(8-GPU) requests below are genuinely PENDING on node availability, not a submission error — they
will start as soon as a full node frees (mine or another user's job finishing), or coordinator may
want a re-think toward partial-GPU footprints for faster fill (see note in report).

## Launch table (all submitted 0.8B, EPOCHS=1, fresh run-names, WANDB on)

| task | arm | cluster/partition | QOS | jobid | seq_len | act_ckpt | status (at HOLD) |
|---|---|---|---|---|---:|---|---|
| msmarco | full | jsteinhardt | preemptive | 3338954 | 29696 | full | RUNNING (mooney) |
| msmarco | chunked-mix | jsteinhardt | preemptive | 3338955 | 29696 | full | PENDING (Resources) |
| grouping_labeled | full | berkeleynlp | preemptive | 3338956 | 37888 | full | PENDING (Resources) |
| grouping_labeled | chunked-mix | berkeleynlp | preemptive | 3338957 | 37888 | full | PENDING (Priority) |
| niah | full | jsteinhardt | preemptive | 3338958 | 35840 | full | PENDING (Priority) |
| niah | chunked-mix | jsteinhardt | preemptive | 3338959 | 35840 | full | PENDING (Priority) |
| scifact | full | berkeleynlp | preemptive | 3338960 | 34816 | full | PENDING (Priority) |
| scifact | chunked-mix | berkeleynlp | preemptive | 3338961 | 34816 | full | PENDING (Priority) |
| fiqa | full | jsteinhardt | preemptive | 3338962 | 26112 | full | PENDING (Priority) |
| fiqa | chunked-mix | jsteinhardt | preemptive | 3338963 | 26112 | full | PENDING (Priority) |
| outlier_amzn | full | berkeleynlp | preemptive | 3338964 | 40960 | full | PENDING (Priority) |
| outlier_amzn | chunked-mix | berkeleynlp | preemptive | 3338965 | 40960 | full | PENDING (Priority) |
| textgroups | full | jsteinhardt | preemptive | 3338966 | 40960 | full | PENDING (Priority) |
| textgroups | chunked-mix | jsteinhardt | preemptive | 3338967 | 40960 | full | PENDING (Priority) |
| absence_gutenberg | full | berkeleynlp | preemptive | 3338968 | 40960 | full | PENDING (Priority) |
| absence_gutenberg | chunked-mix | berkeleynlp | preemptive | 3338969 | 40960 | full | PENDING (Priority) |
| strmatch | full | jsteinhardt | preemptive | 3338970 | 19968 | (auto) | PENDING (Priority) |
| strmatch | chunked-mix | jsteinhardt | preemptive | 3338971 | 19968 | (auto) | PENDING (Priority) |
| qdmatch_hpqa | full | berkeleynlp | preemptive | 3338972 | 29696 | full | PENDING (Priority) |
| qdmatch_hpqa | chunked-mix | berkeleynlp | preemptive | 3338973 | 29696 | full | PENDING (Priority) |
| mathmatch | full | jsteinhardt | preemptive | 3338974 | 25088 | full | PENDING (Priority) |
| mathmatch | chunked-mix | jsteinhardt | preemptive | 3338975 | 25088 | full | PENDING (Priority) |
| cycle | full | berkeleynlp | preemptive | 3338976 | 10240 | (auto) | PENDING (Priority) |
| cycle | chunked-mix | berkeleynlp | preemptive | 3338977 | 10240 | (auto) | PENDING (Priority) |
| groups4 | full | jsteinhardt | preemptive | 3338978 | 12800 | (auto) | PENDING (Priority) |
| groups4 | chunked-mix | jsteinhardt | preemptive | 3338979 | 12800 | (auto) | PENDING (Priority) |
| helmet_qa | full | berkeleynlp | preemptive | 3338980 | 40960 | full | PENDING (Priority) |
| helmet_qa | chunked-mix | berkeleynlp | preemptive | 3338981 | 40960 | full | PENDING (Priority) |
| helmet_summ | full | jsteinhardt | preemptive | 3338982 | 32256 | full | PENDING (Priority) |
| helmet_summ | chunked-mix | jsteinhardt | preemptive | 3338983 | 32256 | full | PENDING (Priority) |
| obliq_retrieval | full | berkeleynlp | preemptive | 3338984 | 40448 | full | PENDING (Priority) |
| obliq_retrieval | chunked-mix | berkeleynlp | preemptive | 3338985 | 40448 | full | PENDING (Priority) |

All 32 use: `SCALE=0.8b EPOCHS=1 GLOBAL_BATCH=8(default) NGPU=8(full node) BASE_SRC=default
(/scratch/users/prasann/cpt_mix_ckpts/q35-08b-base-modelonly) WANDB_API_KEY=<from ~/.netrc>`.
Run-name convention `ctc-s5-<task>-<full|cmix>-08b-loc`. wandb group:
`https://wandb.ai/prasann-uc-berkeley-electrical-engineering-computer-sciences/memory-networks/groups/ctc-suite-<task>`
(local Berkeley runs log under the personal entity, not the AI2 one — confirm from each job's log).

## Not launched (blocked / out of scope)
- msmarco_rerank (duplicate of rerank — see above)
- xabsence (no shard — LLM rebuild still blocked)
- nq, qdmatch_nq, contradiction (owned by other agents)

## NEXT (post-HOLD)
Waiting on coordinator's scale decision (0.8B vs 4B). If 4B is confirmed for wave-2 broadly:
- `beaker_ctc_suite.py` already supports `--model-scale 4b` cleanly (Beaker jupiter, urgent
  priority) — the natural path for 4B, NOT local H200 (4B needs the audited 4B base + weka staging,
  same two-step S3->weka relay already used for contradiction's 4B run).
- The 32 pending 0.8B jobs above should probably be cancelled (not by me — coordinator said don't
  cancel anything RUNNING; the 2 RUNNING ones, hotpotqa-full [pre-existing, not mine] and
  msmarco-full [mine], are explicitly to be left alone per the HOLD; the 30 PENDING-and-not-yet-
  started ones are a judgment call for the coordinator — cancelling a PENDING job loses no compute,
  only queue position).
