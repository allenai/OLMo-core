# CTC Suite — Stage 5 DATA readiness (per-rung eval + 20k mixed-n train shards)

Owner: data-build agent. Updated live. Launch agent reads staged shards from
`/scratch/users/prasann/ctc_suite_staged/shards/<task>_train/` and eval rungs from
`/scratch/users/prasann/ctc_suite_staged/eval_rungs/<task>/rung_<tokens>.jsonl`.
Node-local truth on **cubbins** `/data/prasann/ctc_suite/shards_final/`.

RUNG POLICY: 2k/4k/8k/16k/32k for all EXCEPT reorder/cycle/groups4 → cap 16k.
Eval = fixed-500/rung (contra/niah 488 ok). Train = one 20k mixed-n shard, n ~ Uniform[n(2k),n(cap)].

Legend: ✅ done · 🔄 building · ⏳ queued · ⚠ capped/flag · ❌ blocked.
Last update: 2026-07-20 ~05:40 UTC.

EVAL RUNGS STAGED (rung count, /scratch/.../ctc_suite_staged/eval_rungs/<task>/):
5 rungs (2k-32k): contradiction, grouping, oolong(⚠100→500 in flight), outlier, mathmatch,
  strmatch, textgroups, niah, outlier_amzn, qdmatch_nq, qdmatch_hpqa, helmet_qa, helmet_summ,
  scifact(⚠300/rung), fiqa(building). 4 rungs: cycle(cap16k), hotpotqa(2k-16k,32k-pending),
  msmarco(2k-16k), rerank(2k-16k). 3 rungs: nq(2k/4k/8k CE), groups4(cap8k), reorder(building 2k-16k),
  absence_gutenberg(building 2k-32k). 1: obliq_retrieval(488 fixed, n30, data-poor).

## Readiness table

| task | train_shard | staged? | eval_rungs | ce_ok | status |
|---|---|---|---|---|---|
| contradiction | ✅ 20k (S2) | ✅ | 2k/4k/8k/16k/32k ✅ | n/a | ✅ READY |
| grouping | ✅ 20k (S2) | ✅ | 2k/4k/8k/16k/32k ✅ (temporal, not-nested) | n/a | ✅ READY |
| hotpotqa | ✅ 20k (S2) | ✅ | 2k/4k/8k/16k ✅ (32k pending regen) | n/a | ✅ READY |
| oolong | ✅ 20k (S2) | ✅ | 2k/4k/8k/16k/32k ✅ ⚠ eval_size=100 (±~0.05) | n/a | ✅ READY (flag) |
| outlier | ✅ 20k (S2) | ✅ | 2k/4k/8k/16k/32k ✅ | n/a | ✅ READY |
| reorder | ✅ 20k (S2) | ✅ | 2k/4k/8k/16k 🔄 (gen 3338867) | n/a | 🔄 eval |
| rerank | ✅ 20k (S2) | ✅ | 2k/4k/8k/16k ✅ (32k pending k300 CE) | n/a | ✅ READY |
| mathmatch | ✅ 20k | ✅ | Path-Y 🔄 (3338869) | n/a | 🔄 eval |
| strmatch | ✅ 20k | ✅ | Path-Y 🔄 | n/a | 🔄 eval |
| textgroups | ✅ 20k | 🔄 conv | Path-Y 🔄 | n/a | 🔄 |
| cycle (16k) | ✅ 20k | ✅ | Path-Y 🔄 (2k-16k) | n/a | 🔄 eval |
| groups4 (16k) | ⚠ 15k (2k/4k/8k; n900 infeasible) | ✅ | Path-Y 🔄 (2k-8k) | n/a | ⚠ capped |
| nq | ✅ 20k (reuse clean) | 🔄 conv (3338823_0) | 2k/4k/8k 🔄 (3338885) | ✅ hr=0.097 CE | 🔄 |
| msmarco | ✅ 20k (reuse k20-315) | 🔄 stage | 2k/4k/8k/16k ✅ (=rerank) | n/a | 🔄 stage |
| msmarco_rerank | ✅ 20k (=rerank pool) | 🔄 conv (3338823_17) | 2k/4k/8k/16k ✅ (=rerank) | n/a | 🔄 |
| grouping_labeled | ✅ 20k (reuse grouping) | 🔄 stage | ⏳ (reuse grouping rungs) | n/a | 🔄 |
| outlier_amzn | ✅ 20k | 🔄 conv (3338880) | Path-Y 🔄 | n/a | 🔄 |
| helmet_qa | ✅ 20k | 🔄 conv (3338880) | ⏳ (batch C val, stage) | n/a | 🔄 |
| helmet_summ | ⚠ 17.5k src-cap | 🔄 conv (3338880) | ⏳ | n/a | ⚠ conv |
| niah | 🔄 20k gen (batch B) | ⏳ | Path-Y 🔄 | n/a | 🔄 gen |
| qdmatch_nq | 🔄 20k gen (CE src) | ⏳ | Path-Y 🔄 | ✅ CE-src | 🔄 gen |
| qdmatch_hpqa | 🔄 20k gen | ⏳ | Path-Y 🔄 | n/a | 🔄 gen |
| absence_gutenberg | 🔄 20k gen | ⏳ | Path-Y 🔄 | n/a | 🔄 gen |
| scifact | 🔄 ~4k gen (batch D, GPU) | ⏳ | ⏳ (build_v2 from test canon) | n/a | ⚠ data-poor |
| fiqa | 🔄 20k gen (batch D, GPU) | ⏳ | ⏳ | n/a | 🔄 gen |
| obliq_retrieval | ⏳ ~1797 cap | ⏳ | ⏳ | n/a | ⏳ data-poor |
| xabsence | ❌ 659-pair pool (LLM rebuild) | — | — | n/a | ❌ blocked |

## Provenance / flags
- nq TRAIN = `/scratch/users/prasann/nq_p10_20k/nq_train_k25-202_clean.jsonl` (19,967 ex, hard-ratio 0.097, CE-filtered "clean"). nq EVAL = p10/CE validation (hr 0.097), 2k/4k/8k clean; 16k/32k need uniform-k200 CE regen (GPU, queued-lower-pri).
- qdmatch_nq generated FROM the CE-clean nq source (inherits CE negs).
- Eval "Path-Y" = native per-rung generation @ seed 7; NOT nested-across-rungs (correct answers per rung) — document in result provenance.
- grouping eval = temporal held-out (--eval-year-min 2024), NOT nested — document in provenance.
- Staged shards so far: contradiction, grouping, hotpotqa, oolong, outlier, reorder, rerank (S2) + cycle, groups4, mathmatch, strmatch (synth). msmarco/grouping_labeled/nq/msmarco_rerank staging next.

## Remediation list (data-poor / deferred, do LAST)
- oolong 500-ex eval (currently 100/rung) — bump via more seed buckets if cheap.
- hotpotqa 32k (needs n290 eval canonical regen); rerank 32k (needs k~300 CE eval pool, A9).
- nq 16k/32k eval (uniform-k200 CE regen, GPU).
- groups4 16k rung (n900 answer-range assembly infeasible; O(N³) task, capped at 8k).
- scifact ~4k train (800-query BEIR cap); helmet_summ 17.5k (govreport train cap).
- obliq_retrieval ~1797 train (small pool); xabsence 659-pair pool (needs LLM paraphrase pool rebuild — blocked).
</content>
