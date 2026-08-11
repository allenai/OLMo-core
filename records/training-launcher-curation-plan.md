# Training-launcher curation plan: 161 -> 43 keep, 17 collapsible, 88 drop, 13 uncertain

Audit of every `.py`/`.sbatch` under `src/scripts/train/memexpress/` in the pre-migration repo
(`/accounts/projects/berkeleynlp/prasann/projects/OLMo-core`, branch `prasann/landmark`), done to
decide what actually ports into the fresh clone at `prasann/ctc`. Companion to
`clean-repo-migration-plan.md` §5 (the keep/drop ledger) and `clean-repo-target-structure.md` §5b
(`src/scripts/ctc/`) — this document is the launcher-specific version of that exercise, backed by a
file-by-file table instead of a family-level guess.

**Method.** Read `memexpress/README.md` and every family README first (they're accurate and saved
most of the reading). Then, for each of the 161 files, searched `records/*.md`, `results/**`,
`CLAUDE.md`, the three cluster docs, and the 96 files in the persistent Claude-memory directory
(`.claude/projects/.../memory/*.md` — the session-to-session notes that record what actually ran)
for the filename or its output checkpoint/run names. A hit in that search is the evidence column; no
hit does not always mean dead (older experiments are sometimes recorded only by run name), but it is
the strongest read-only signal available, and every verdict below cites what was found. Nothing was
modified, launched, or submitted.

## Headline numbers

| | Count |
|---|---|
| Total launchers audited | **161** |
| **KEEP** (port as-is) | **43** |
| **SUPERSEDED** (collapses into a kept sibling — the near-duplicate clusters in §3) | **17** |
| **DROP** (concluded probe, dead experiment, or superseded family) | **88** |
| **UNCERTAIN** (flagged, needs a yes/no from Prasann before deleting) | **13** |

43 is more than "well under 30" read literally, but roughly a third of it is single-purpose
`__init__.py` / shared-module plumbing for one package (`ctc_suite/ctc_suite/`, 7 files) that isn't
a separate "launcher" in the sense the task means. Counted as **decision points** instead of files —
one launcher family = one line — the actual keep set is **16**:

1. `train_ctc_suite.py` + its 3 launchers (local/lambda/Beaker) — the CTC-suite trainer
2. The `ctc_suite/ctc_suite/` eval package (`run_rung_eval.py` + support modules) — 1 decision, 7 files
3. 3 base converters (Qwen3.5, Llama-3.2-3B, OLMo-3-7B) — 3 decisions, 9 files incl. audits
4. `_docchunk_5task_32k_nocpt_common.py` + 3 kept variant launchers (dense-bs128, hier, landmark) + local twin
5. 2 `attn_explore` reference recipes (0.6B and 0.8B docchunk-mask-mix) + their sbatch scaffolding — 4 decisions
6. `local_4b`'s dense and docchunk local pairs — 2 decisions, 4 files
7. 3 CPT-base recipes (dense/landmark/compressive lr1.1e-4) — the only record of the checkpoints everything above still inits from
8. `probes/` (2 small utilities, kept as-is, no family)

Every number in this document is measured against the pre-migration repo as it stands today
(2026-08-11); nothing here required launching a job.

---

## 1. Recommended KEEP set (16 decision points, 43 files)

Covers the three things asked for: the CTC-suite training runs, the document-chunked/landmark SFT
runs, and the local-cluster path.

### CTC suite (the live effort — Figure 4, ~26 tasks, 2k-32k ladders)

| Keep | Why irreplaceable |
|---|---|
| `ctc_suite/train_ctc_suite.py` | The task-agnostic trainer every current run goes through — 13 references, `ctc-suite-scaling-plan.md`'s entire method section is written against its flags. |
| `ctc_suite/run_ctc_local.sbatch` | Primary Berkeley launcher (18 refs) — logs correctly to node-local `/data`, unlike most of the older sbatch tree (see §4). |
| `ctc_suite/run_ctc_lambda.sbatch` | Only path onto the air-gapped Lambda A100 pool (19 refs); CLAUDE.md documents its own `--gres` trap. |
| `ctc_suite/beaker_ctc_suite.py` | Only path onto Beaker for this trainer; `--priority` defaults to `urgent` (verified in source). |
| `ctc_suite/__init__.py` | Package init required by all of the above. |
| `ctc_suite/ctc_suite/run_rung_eval.py` (+ `.sbatch`, `__init__.py` x2, `vllm/__init__.py`, `results_io.py`, `plot_ctc_suite.py`) | The eval driver — **99 references**, the single most-cited file in the whole tree. This is one deliverable (the eval package), counted as 7 files. |
| `ctc_suite/convert_qwen35_base.py` + `convert_qwen35_bases.sbatch` | Produces the audited Qwen3.5 model-only bases every ctc_suite run reads from; named directly in `ctc-suite-scaling-plan.md`. |
| `ctc_suite/convert_llama_base.py` + `llama_configs.py` | Produced the base behind `results/ctc_suite_llama*`; paper-v2 item 9 ("more model families") is still open, this is the live extension axis. |
| `ctc_suite/convert_olmo3_base.sbatch` + `olmo3_configs.py` + `olmo3_marker_audit.py` + `olmo3_parity_check.py` | Produced the base behind `results/ctc_suite_olmo3_hpqa` (paper-v2 item 9: "done and clean"); the audit/parity scripts are the gate that must re-run on any future OLMo3 base rebuild. |

### Document-chunked / landmark SFT (Beaker)

| Keep | Why irreplaceable |
|---|---|
| `sft_docchunk/_docchunk_5task_32k_nocpt_common.py` | Shared builder every variant below imports; 8 references incl. two dedicated result records. |
| `sft_docchunk/Qwen3-4B-docchunk-5task-32k-nocpt-SFT.py` | **This is the exact script behind the headline numbers in `docchunk-4b-5task-ladder-results.md`** — the run name recorded there, `q4b-docchunk-5task-32k-nocpt-bs128`, matches this file's `flex_block_size=128`. (Its sibling `...-docchunk-dense-...py` is a byte-for-byte duplicate at a different, non-functional block size — see §3.) |
| `sft_docchunk/Qwen3-4B-docchunk-hier-5task-32k-nocpt-SFT.py` | `hierarchical_dilated` is the best-performing docchunk variant measured (`n100-clean-results`: 0.831 vs 0.441 pure chunked). |
| `sft_docchunk/Qwen3-4B-docchunk-landmark-5task-32k-nocpt-SFT.py` | The document-landmark arm — this is literally the "landmark SFT" half of the brief. |
| `sft_docchunk/Qwen3-4B-docchunk-5task-local.py` | The local torchrun twin, named explicitly in the family README — required local-cluster coverage for this family. |

### `attn_explore` — the mask-design reference recipe (Berkeley local, 0.6B/0.8B)

Kept because `ctc_suite/train_ctc_suite.py`'s own docstring says it "generalizes"
`attn_explore/Qwen3.5-0.8B-docchunk-mask-mix-contradiction-SFT-local.py` and the 0.6B script's
curriculum hard-fail logic — i.e. these are not legacy, they're the design document for the current
trainer, and CLAUDE.md's "curriculum mask-mixing is the default" rule traces back to them.

| Keep | Why |
|---|---|
| `attn_explore/Qwen3-0.6B-docchunk-mask-mix-contradiction-SFT-local.py` | Source of the curriculum hard-fail logic; 6 refs. |
| `attn_explore/Qwen3.5-0.8B-docchunk-mask-mix-contradiction-SFT-local.py` | Direct ancestor of `train_ctc_suite.py`; 3 refs. |
| `attn_explore/run_q06b_attn_explore_mooney.sbatch` | "The parametrized multi-variant launcher" per its own README; 6 refs. |
| `attn_explore/run_q35_08b_attn_explore.sbatch` | Beaker/local twin sbatch for the 0.8B script. |
| `attn_explore/run_q06b_dense_contra_n20_local_mooney.sbatch` + `Qwen3-0.6B-dense-contradiction-n20-SFT-local.py` | "The reference `local_env.sh` retrofit" per the family README — the template every other local sbatch in the tree copies (correct `/data` logging, node-local staging). |
| `attn_explore/eval_q06b_contra_n20_native.sbatch` + `eval_q4b_attn_explore_cubbins.sbatch` | The eval side of the same reference recipe. |
| `attn_explore/probe_train_memorization.py` | Small (140-line), general-purpose, no dead-experiment dependency. |

### Local-cluster path (Berkeley H200, no weka/Beaker)

| Keep | Why |
|---|---|
| `local_4b/Qwen3-4B-document-chunked-longctx-SFT-local.py` + `run_q4b_docchunk_contra.sbatch` | The 4B local document-chunked launcher — this *is* the local-cluster half of the docchunk/landmark coverage requirement. |
| `local_4b/Qwen3-4B-dense-cptmix-contra-local.py` + `run_q4b_cptmix_contra_local.sbatch` | Plain-dense local reference pair (the non-chunked baseline needed to compare against). |
| `local_4b/run_q4b_doc_oolong.sbatch` | Same shape as `run_q4b_docchunk_contra.sbatch`, pointed at oolong instead of contradiction — kept alongside it because oolong is the task document-chunking collapses hardest on (`docchunk-4b-5task-ladder-results.md`), so it's the one other task worth a dedicated local sanity check. |

### CPT bases (Beaker) — kept as the recipe of record, not for re-running routinely

| Keep | Why |
|---|---|
| `cpt/Qwen3-4B-base-dense-lr1.1e-4.py` | Produced `amandab/q4b-base-dense-lr1.1e-4/step2385`, the dense CPT base still used as init by `singletask-ladder-experiment` and `outlier-datascale-sweep`. |
| `cpt/Qwen3-4B-base-fast-landmark-lr1.1e-4.py` | Produced `amandab/q4b-base-fast-landmark-lr1p1e-4/step2385`, referenced the same way. |
| `cpt/Qwen3-4B-base-fast-compressive-landmark-dolma3longmino.py` | Produced the compressive CPT base; `single-task-ladder-experiment.md` states the compressive variant *requires* this exact base (landmark-token embeddings + grouped-softmax trained there — cannot substitute a fresh init). |

These three are the only surviving record of how those three still-in-use checkpoints were made. If
the checkpoints are ever lost or need extending to a new scale, this is the only place the recipe
exists. They are not needed for day-to-day re-runs.

### `probes/` (kept, no family)

`sanity_check_packing.py` and `scan_doc_lengths.py` — small, general-purpose diagnostics with no tie
to a concluded experiment. Cheap to carry, useful to keep.

---

## 2. DROP list, grouped by reason (88 files)

**Superseded family — pre-5task-mix generation (sft_longctx, 16 of 21 files).** The family's own
README says it outright: "Superseded by `sft_5task/` + `sft_docchunk/` for new experiments... kept
for parallel Beaker use." 4 of these (`*-noruler-4k*`, `*-noruler-5k*`) are additionally concluded
probes — `ruler-forgetting-experiment.md` answered the question in 2026-06-21 (CPT mixing helps,
15% too weak, 30% works) — **and** hardcode `beaker_launch_config.priority = "normal"`, a live
violation of the repo's always-urgent rule (see §4).

**Superseded family — the canonical 5-task mix without doc-chunking (sft_5task, 9 of 10 files).**
`sft_docchunk` applies the identical 5-task mix plus a chunk mask on top of the same recipe;
`clean-repo-target-structure.md`'s own recommended keep list (ctc_suite + sft_docchunk +
attn_explore) omits this family. The baseline numbers it would reproduce already live in
results-hub / as the "dense"/"comp" reference columns in `docchunk-4b-5task-ladder-results.md`.

**Superseded family — per-task ladder SFT, predates ctc_suite's generalization (singletask_ladder, 3
of 11 files as a flat DROP; 7 more as SUPERSEDED — see §3).** `Qwen3-4B-compressive-singletask-ladder-32k-SFT.py`
is labeled "Legacy / superseded (kept for reference)" in the family's own README. `run_q4b_singletask_ladder.sbatch`
is labeled "earlier local sbatch" in the same README. `Qwen3-4B-docchunk-oolong-sanity-8k-SFT.py` was
a one-off CoT-ablation probe whose finding (CoT rescues docchunk-oolong 2.7-3.1x) is already recorded
in `docchunk-4b-5task-ladder-results.md` — nothing left to re-run.

**Concluded diagnostic probe, negative result (goldgrad, 3 files + goldhop, 2 files).** Goldgrad's
measured speedup was 1.00x ("a probe, not an O(1) backward"), and the follow-up on leak-free data
(`n100-clean-results-2026-07-14.md`) shows the effect does not replicate at all. Goldhop's own
opening record says "proposal, awaiting refinement" and was run to completion
(`results/hopgold-n50-summary-ladder.md` exists) with no active follow-on found. Both are closed
lines of inquiry, fully written up.

**Superseded eval driver (evals/, 5 of 7 files).** `ctc_suite/ctc_suite/run_rung_eval.py` (99
references) is the eval driver every current effort uses; none of `eval_dense_base.sbatch`,
`eval_gen_native.sbatch`, `eval_ladder_32k.sbatch`, `eval_landmark_native.sbatch`,
`eval_landmark_topk.sbatch`, `run_eval_q4b_lc.sbatch` turned up anywhere in records/results/memory.

**One-off, already-completed patch jobs (ctc_suite, 4 files).** `retrain_oolong_full.sbatch` fixed a
missing arm — `results/ctc_suite/oolong/qwen3.5-4b_full` now exists, so its job is done.
`assemble_pilot2k_result.py` and `harvest_lambda_pilot.py` were pilot-wave (2026-07-18 to
2026-07-20) aggregators, superseded by `run_rung_eval.py` + `all_results.jsonl` once the suite moved
past the pilot stage. `eval_nq_ceconfirm.sbatch`'s finding (ce-filter alone is insufficient at 2.5k
examples, need the full 20k) is now baked into the standard nq pipeline rule.

**Concluded local exploration (local_4b, 4 files).** `run_q4b_contra_n250_{landmark,standard}.sbatch`
is an early n=250 probe superseded by the cleaner n=100/n=20 results
(`n100-clean-results-2026-07-14.md`). `run_q4b_cptmix_sweep.sbatch` and `run_q4b_fastlm_cpt40m.sbatch`
belong to the closed `contradiction-cptmix-ruler-experiment` (concluded 2026-06-25); no reference
beyond that experiment.

**Abandoned ablation, zero references (cpt/interleaved/, 6 files).** A data-interleaving-pattern
sweep (sparse:regular ratio, alternating, half-full split) with no hit anywhere in
records/results/memory — reads as an idea that was tried once and dropped.

**Smoke-test scaffolding (cpt/debug/, 13 files).** The family README calls these "smoke runs" for
single-node validation before the real CPT launchers went to Beaker. Zero references found; superseded
by the fact that the real launchers (also mostly DROP — see next item) now work.

**CPT grid — no reference found (cpt/ top-level, 17 of 23 files).** The dense/landmark/fast-landmark/
sparse-landmark x {Qwen3, Qwen3.5} x {8-node, lr-tagged, single-node} CPT grid. Three of the 23 are
kept (§1: the recipes behind checkpoints still in use); the other 17, including every Qwen3.5-tagged
CPT script, have no reference anywhere. If the Qwen3.5 CPT track is wanted again, none of these are
individually load-bearing — see the collapse cluster in §3.

**Pre-5task-mix predecessor (sft_docchunk, 1 file).** `Qwen3-4B-document-landmark-longctx-SFT.py`
predates the 5-task-mix `Qwen3-4B-docchunk-landmark-5task-32k-nocpt-SFT.py` it was superseded by.

**Beaker twin with no reference (attn_explore, 1 file).** `Qwen3-0.6B-fast-landmark-contradiction-n20-SFT.py`
(non-local) has no reference found; the Beaker path for this line of work moved into ctc_suite.

---

## 3. Near-duplicate clusters — the biggest win

These six clusters cover **93 of 161 files (58%)** and are the highest-leverage collapse targets:
each is one script's worth of actual logic wearing N filenames because the current repo encodes run
configuration in the filename instead of a flag. This is exactly the problem
`clean-repo-target-structure.md` names as the reason `configs/*.yaml` replaces `.py`-per-run.

| # | Cluster | Files absorbed | Shared shape | What actually varies |
|---|---|---|---|---|
| 1 | **`cpt/` CPT grid** (top-level 23 + `debug/` 13 + `interleaved/` 6) | **42** | One CPT training script (init from HF Qwen3/Qwen3.5, train on dolma3+longmino, save distcp) | attention variant (dense/landmark/fast-landmark/sparse-landmark/compressive-landmark), model family (Qwen3 vs Qwen3.5), node count (1 vs 8), LR, and — in `debug/` — whether it's a 1-node smoke run on wikitext instead of the real corpus. One `--attn-variant --model-family --nodes --lr [--smoke]` launcher replaces all 42; the 3 still-live checkpoints (§1) become 3 YAML configs, not 3 scripts. |
| 2 | **`sft_longctx/`** | **21** | One SFT-on-a-base script (dense/fast-landmark/sparse-landmark x mostly-identical trainer boilerplate) | attention variant, RULER-inclusion, CPT-mix fraction, packed-vs-unified-vs-base32k layout, 10task1k-vs-5task data. Already dead per §2, but if any of it is revived it collapses to 1 launcher the same way. |
| 3 | **`sft_5task/`** | **10** | One 5-task-mix SFT script | attention variant (dense/compressive/fast-landmark) x {32k-nocpt, 32k-nocpt-fixnq, cptmix-32k, cptmix-64k}. Same collapse shape as #2. |
| 4 | **`sft_docchunk/` variant scripts** | **7** (of which 2 are KEPT as the surviving representatives) | `_docchunk_5task_32k_nocpt_common.py`'s shared builder, already 90% of the way there | `cross_doc_mode` (chunked/hierarchical_dilated/landmark/random_doc), `dilation_cycle` (K=25 vs K=50), and — in the exact-duplicate case — `flex_block_size` (32, non-functional, vs 128). One `--variant` flag on the common builder finishes the collapse the family already started. |
| 5 | **`attn_explore/` per-variant n20 scripts** | **9** (of 11 total in this shape; 2 kept as the reference pair) | `run_q06b_attn_explore_mooney.sbatch` already *is* "the parametrized multi-variant launcher" per its own README | attention variant (dense/dilated/compressive/fast-landmark), local vs Beaker. The 4 standalone `.py` twins and 3 of the 4 standalone `.sbatch` twins are strictly redundant with the multi-variant launcher that already exists — finish deleting them, don't port them. |
| 6 | **`singletask_ladder/` `run_q4b_stl_*.sbatch`** | **4** | One train/eval sbatch reading from `/data` or `/scratch` | `--mode {eval, multirung_eval, traineval, validation}`. |

A smaller, lower-priority opportunity: the 4 base-converter bundles in `ctc_suite/` (Qwen3.5, Llama,
OLMo-3, Gemma-3 — 9 files total, all in §1's keep set) share a CLI/audit/parity-check shape but
genuinely differ in state-dict mapping per model family (tied embeddings, vision-tower stripping,
SWA pattern) — worth a shared harness with per-family plugins, not a flat merge.

---

## 4. Dangerous or environment-specific patterns worth fixing during the port

**22 sbatch files write slurm `--output` to the repo root on NFS, not node-local `/data`.** This is
exactly the trap CLAUDE.md warns about (`local-train-nfs-log-hang`: "slurm `--output` on NFS
deadlocks at step 0"). The current best-practice files (`ctc_suite/run_ctc_local.sbatch`,
`attn_explore/run_q06b_dense_contra_n20_local_mooney.sbatch`) correctly log to
`/data/prasann/joblogs/...`; these 22 still point at
`/accounts/projects/berkeleynlp/prasann/projects/OLMo-core/.<name>_%x_%j.log`:

```
attn_explore/run_q06b_fast_contra_n20_local_mooney.sbatch
attn_explore/run_q06b_comp_contra_n20_local_mooney.sbatch
local_4b/run_q4b_contra_n250_{landmark,standard}.sbatch
local_4b/run_q4b_cptmix_contra_local.sbatch         <- KEPT, fix on port
local_4b/run_q4b_docchunk_contra.sbatch             <- KEPT, fix on port
local_4b/run_q4b_doc_oolong.sbatch                  <- KEPT, fix on port
local_4b/run_q4b_lm_cptmix_5task.sbatch
local_4b/run_q4b_cptmix_sweep.sbatch
local_4b/run_q4b_fastlm_cpt40m.sbatch
evals/run_eval_q4b_lc.sbatch
evals/eval_{dense_base,gen_native,dense_local_vllm,ladder_32k,landmark_native,landmark_topk}.sbatch
singletask_ladder/run_q4b_stl_{traineval,eval,validation,multirung_eval}.sbatch
singletask_ladder/run_q4b_singletask_ladder.sbatch
```

Three of these (`run_q4b_cptmix_contra_local.sbatch`, `run_q4b_docchunk_contra.sbatch`,
`run_q4b_doc_oolong.sbatch`) are in the recommended KEEP set — **fix the log path when porting them**,
don't carry the trap forward.

**4 Beaker launchers hardcode `priority = "normal"`, violating the repo's always-urgent rule.** All
four are in `sft_longctx/` and all four are already DROP for other reasons (§2):
`Qwen3-4B-dense-noruler-{4k,4k-CPTmix,5k,5k-CPTmix}-SFT.py`. Flagging them anyway because if anyone
resurrects the ruler-forgetting-experiment lineage from git history, this is the line to change first
(`beaker_launch_config.priority = "normal"`, with a comment explaining the original reasoning: "this
is a cheap probe, not a headline run" — a rationale that doesn't override the repo rule).

**`ctc_suite/run_ctc_lambda.sbatch`'s header hardcodes `--gres=gpu:A100:8`.** Already a documented
trap (CLAUDE.md, lambda_cluster.md): direct `sbatch` execution (rather than going through the
self-submit path) eats the whole 8-GPU allocation. Kept (§1) because it's the only Lambda launcher,
but the trap should be called out in whatever doc replaces `CLAUDE.md`'s cluster section on the new
branch.

**Save-folder / auto-resume risk: not found as a code bug.** Grepped every launcher for a hardcoded
`save_folder` not keyed by `run_name`; found none — every script in the tree already parameterizes
its save folder by run name, so the `trainer-silent-autoresume-trap` here is an operational risk
(relaunching into an existing folder), not a code defect to fix on port.

---

## 5. Full table (all 161 files)

Columns: path (relative to `src/scripts/train/memexpress/`), line count, cluster type (Beaker /
Berkeley local / Lambda / smoke), family, verdict, and the evidence behind that verdict — a grep hit,
a README line, or a `records/`/memory-note reference, as instructed. Sorted by family in roughly the
order discussed above.

| Path (under `src/scripts/train/memexpress/`) | Lines | Cluster | Family | Verdict | Evidence |
|---|---|---|---|---|---|
| `ctc_suite/__init__.py` | 0 | n/a | ctc_suite | **KEEP** | package init, required by the above |
| `ctc_suite/beaker_ctc_suite.py` | 291 | Beaker | ctc_suite | **KEEP** | Beaker/gantry launcher for train_ctc_suite.py; 5 refs, default priority=urgent verified |
| `ctc_suite/convert_gemma3_base.py` | 272 | Berkeley local | ctc_suite | **UNCERTAIN** | Gemma-3 converter written but no results/ctc_suite_gemma* directory exists yet -- paper-v2 item 9 says 'at least one further family is still owed', plausibly this one; keep pending confirmation, drop if gemma is not the next family |
| `ctc_suite/convert_llama_base.py` | 271 | Berkeley local | ctc_suite | **KEEP** | produced the Llama-3.2-3B base behind results/ctc_suite_llama*; paper-v2 item 9 (more model families) still open |
| `ctc_suite/convert_olmo3_base.sbatch` | 59 | Berkeley local | ctc_suite | **KEEP** | produced the OLMo-3-7B base behind results/ctc_suite_olmo3_hpqa (paper-v2 item 9: 'done and clean') |
| `ctc_suite/convert_qwen35_base.py` | 101 | Beaker/local | ctc_suite | **KEEP** | produces the audited Qwen3.5 model-only bases every ctc_suite run reads; ctc-suite-scaling-plan.md cites it directly |
| `ctc_suite/convert_qwen35_bases.sbatch` | 38 | Berkeley local | ctc_suite | **KEEP** | sbatch wrapper for the above; ctc-suite-scaling-plan.md cites it directly |
| `ctc_suite/ctc_suite/__init__.py` | 0 | n/a | ctc_suite | **KEEP** | package init for the eval package; 18 refs (import surface) |
| `ctc_suite/ctc_suite/assemble_pilot2k_result.py` | 205 | n/a | ctc_suite | **DROP** | pilot-stage (2026-07-18) result aggregator; Stage 3 pilot is long done and superseded by run_rung_eval.py + all_results.jsonl |
| `ctc_suite/ctc_suite/eval_ctc_lambda.sbatch` | 70 | Lambda | ctc_suite | **UNCERTAIN** | no reference found in records/results/memory; likely an eval-side twin of run_ctc_lambda.sbatch -- confirm whether run_rung_eval.py now covers lambda eval before dropping |
| `ctc_suite/ctc_suite/eval_nq_ceconfirm.sbatch` | 36 | Berkeley local | ctc_suite | **DROP** | one-off nq CE-filter confirmation run (nq_debug_diagnosis.md); its finding (ce-filter alone insufficient at 2.5k, need 20k) is now folded into the standard nq pipeline rule |
| `ctc_suite/ctc_suite/harvest_lambda_pilot.py` | 120 | Lambda | ctc_suite | **DROP** | one-off harvester for the lambda pilot wave (SESSION-HANDOFF, 2026-07-20); that wave is over, harvesting is now routine via run_rung_eval.py |
| `ctc_suite/ctc_suite/plot_ctc_suite.py` | 238 | n/a | ctc_suite | **KEEP** | the Figure-4 plot generator named explicitly in ctc-suite-scaling-plan.md §8 |
| `ctc_suite/ctc_suite/results_io.py` | 312 | n/a | ctc_suite | **KEEP** | shared results-JSON schema writer/reader for the whole suite |
| `ctc_suite/ctc_suite/run_rung_eval.py` | 589 | Beaker+local+lambda | ctc_suite | **KEEP** | the live eval driver -- 99 references, by far the most-cited file in the whole tree |
| `ctc_suite/ctc_suite/run_rung_eval.sbatch` | 66 | Berkeley local | ctc_suite | **KEEP** | sbatch companion to run_rung_eval.py |
| `ctc_suite/ctc_suite/vllm/__init__.py` | 9 | n/a | ctc_suite | **KEEP** | vLLM eval subpackage init; 18 refs |
| `ctc_suite/llama_configs.py` | 139 | n/a | ctc_suite | **KEEP** | TransformerConfig factory for Llama-3.2-3B; feeds the live results/ctc_suite_llama* runs |
| `ctc_suite/olmo3_configs.py` | 166 | n/a | ctc_suite | **KEEP** | OLMo-3 SWA-aware TransformerConfig; feeds live olmo3 results |
| `ctc_suite/olmo3_marker_audit.py` | 149 | n/a | ctc_suite | **KEEP** | OLMo counterpart of fix_marker_embeddings.py gate; must run before any new OLMo3 base is trusted |
| `ctc_suite/olmo3_parity_check.py` | 114 | n/a | ctc_suite | **KEEP** | CE+top-1 parity gate for every OLMo3 HF->distcp conversion; needed whenever the base is rebuilt |
| `ctc_suite/olmo3_swa_ablation.py` | 110 | n/a | ctc_suite | **UNCERTAIN** | one-time architecture-safety measurement (does disabling SWA hurt the pretrained base); no downstream reference found beyond its own docstring -- keep if the OLMo3 arm is still being extended, else archive as a completed sanity check |
| `ctc_suite/retrain_oolong_full.sbatch` | 35 | Berkeley local | ctc_suite | **DROP** | one-off patch job to backfill the missing oolong dense arm; results/ctc_suite/oolong/qwen3.5-4b_full now exists, so the gap it fixed is closed -- superseded by its own successful run |
| `ctc_suite/run_ctc_lambda.sbatch` | 141 | Lambda | ctc_suite | **KEEP** | Lambda-cluster launcher; 19 refs; CLAUDE.md explicitly documents its --gres trap |
| `ctc_suite/run_ctc_local.sbatch` | 195 | Berkeley local | ctc_suite | **KEEP** | primary Berkeley launcher for train_ctc_suite.py; 18 refs |
| `ctc_suite/train_ctc_suite.py` | 1126 | Beaker+local+lambda | ctc_suite | **KEEP** | core task-agnostic trainer for the live Fig-4 effort; 13 refs incl. ctc-suite-scaling-plan.md, README |
| `sft_docchunk/Qwen3-4B-docchunk-5task-32k-nocpt-SFT.py` | 34 | Beaker | sft_docchunk | **KEEP** | the actual script behind the headline docchunk-4b-5task-ladder-results.md numbers (.68/.60/.51/.44 contradiction etc.) -- run name in that record is 'q4b-docchunk-5task-32k-nocpt-bs128', matching this file's flex_block_size=128 |
| `sft_docchunk/Qwen3-4B-docchunk-5task-local.py` | 294 | Berkeley local | sft_docchunk | **KEEP** | the local torchrun twin -- required for the local-cluster-path coverage; README calls it out by name |
| `sft_docchunk/Qwen3-4B-docchunk-compressive-5task-32k-nocpt-SFT.py` | 23 | Beaker | sft_docchunk | **SUPERSEDED** | thin variant of _docchunk_5task_32k_nocpt_common.py identical in shape to the dense/landmark/hier scripts above -- collapses into one --variant flag (see collapse cluster) |
| `sft_docchunk/Qwen3-4B-docchunk-dense-5task-32k-nocpt-SFT.py` | 24 | Beaker | sft_docchunk | **SUPERSEDED** | own docstring says 'Identical to Qwen3-4B-docchunk-5task-32k-nocpt-SFT.py' -- differs only in flex_block_size (32 vs 128, and 32 doesn't even lower correctly); exact-duplicate collapse target, keep only the bs128 version above |
| `sft_docchunk/Qwen3-4B-docchunk-hier-5task-32k-nocpt-SFT.py` | 24 | Beaker | sft_docchunk | **KEEP** | hierarchical_dilated arm, the best-performing docchunk variant per n100-clean-results (.831 vs .441 pure chunked); 1 direct ref |
| `sft_docchunk/Qwen3-4B-docchunk-hierK25-5task-32k-nocpt-SFT.py` | 40 | Beaker | sft_docchunk | **SUPERSEDED** | hier variant with dilation_cycle rotated (K=25 vs K=50) -- a hyperparameter of the hier launcher, not a separate script; collapse target |
| `sft_docchunk/Qwen3-4B-docchunk-landmark-5task-32k-nocpt-SFT.py` | 24 | Beaker | sft_docchunk | **KEEP** | the document-landmark arm -- this IS the 'landmark SFT' half of the required keep-set coverage; 1 direct ref |
| `sft_docchunk/Qwen3-4B-docchunk-randomdoc-5task-32k-nocpt-SFT.py` | 31 | Beaker | sft_docchunk | **SUPERSEDED** | ablation arm (random_doc keep-prob baseline) -- same shape as the others, collapse target |
| `sft_docchunk/Qwen3-4B-document-landmark-longctx-SFT.py` | 236 | Beaker | sft_docchunk | **DROP** | pre-5task-mix predecessor of Qwen3-4B-docchunk-landmark-5task-32k-nocpt-SFT.py; single-task-mix generation superseded by the 5task-mix family |
| `sft_docchunk/_docchunk_5task_32k_nocpt_common.py` | 446 | Beaker | sft_docchunk | **KEEP** | shared builder imported by every docchunk-5task variant; 8 refs incl. docchunk-4b-5task-ladder-results, docchunk-4b-h100-oom-fixes |
| `attn_explore/Qwen3-0.6B-compressive-contradiction-n20-SFT-local.py` | 219 | Berkeley local | attn_explore | **SUPERSEDED** | per-variant n20 twin now covered by run_q06b_attn_explore_mooney.sbatch's --variant sweep; collapse target |
| `attn_explore/Qwen3-0.6B-dense-contradiction-n20-SFT-local.py` | 238 | Berkeley local | attn_explore | **KEEP** | dense baseline paired with the reference sbatch above; 1 ref |
| `attn_explore/Qwen3-0.6B-dilated-contradiction-n20-SFT-local.py` | 215 | Berkeley local | attn_explore | **SUPERSEDED** | per-variant n20 twin, collapse target (same cluster as compressive/fast above) |
| `attn_explore/Qwen3-0.6B-docchunk-mask-mix-contradiction-SFT-local.py` | 550 | Berkeley local | attn_explore | **KEEP** | source of the curriculum hard-fail logic train_ctc_suite.py generalizes; 6 refs, CLAUDE.md's mask-mixing-default rule traces here |
| `attn_explore/Qwen3-0.6B-fast-landmark-contradiction-n20-SFT-local.py` | 241 | Berkeley local | attn_explore | **SUPERSEDED** | per-variant n20 twin, collapse target; 2 refs (qwen3-06b-fast-landmark-contra-n20 memory result .782 f1) |
| `attn_explore/Qwen3-0.6B-fast-landmark-contradiction-n20-SFT.py` | 227 | Beaker | attn_explore | **DROP** | non-local Beaker twin of the fast-landmark n20 script; no reference found, and the Beaker path for this line of work has moved to ctc_suite |
| `attn_explore/Qwen3.5-0.8B-docchunk-mask-mix-contradiction-SFT-local.py` | 355 | Berkeley local | attn_explore | **KEEP** | direct ancestor train_ctc_suite.py 'generalizes' per ctc_suite/README.md; 3 refs |
| `attn_explore/eval_q06b_contra_n20_native.sbatch` | 66 | Berkeley local | attn_explore | **KEEP** | the eval side of the family's reference recipe; 3 refs |
| `attn_explore/eval_q4b_attn_explore_cubbins.sbatch` | 118 | Berkeley local | attn_explore | **KEEP** | 4B eval twin; 6 refs |
| `attn_explore/probe_train_memorization.py` | 140 | n/a | attn_explore | **KEEP** | small (140-line) general-purpose memorization-probe utility, cheap to carry, 1 ref |
| `attn_explore/run_q06b_attn_explore_mooney.sbatch` | 172 | Berkeley local | attn_explore | **KEEP** | 'the parametrized multi-variant launcher' per its own README; 6 refs |
| `attn_explore/run_q06b_comp_contra_n20_local_mooney.sbatch` | 50 | Berkeley local | attn_explore | **SUPERSEDED** | per-variant sbatch twin of the multi-variant launcher, collapse target |
| `attn_explore/run_q06b_dense_contra_n20_local_mooney.sbatch` | 47 | Berkeley local | attn_explore | **KEEP** | 'the reference local_env.sh retrofit' per family README -- the template every other local sbatch in the repo copies |
| `attn_explore/run_q06b_dilated_contra_n20_local_mooney.sbatch` | 76 | Berkeley local | attn_explore | **SUPERSEDED** | per-variant sbatch twin, collapse target |
| `attn_explore/run_q06b_fast_contra_n20_local_mooney.sbatch` | 55 | Berkeley local | attn_explore | **SUPERSEDED** | per-variant sbatch twin, collapse target; 1 ref |
| `attn_explore/run_q35_08b_attn_explore.sbatch` | 104 | Berkeley local | attn_explore | **KEEP** | Beaker/local twin sbatch for the 0.8B docchunk-mask-mix script named in the family README |
| `local_4b/Qwen3-4B-dense-cptmix-contra-local.py` | 352 | Berkeley local | local_4b | **KEEP** | plain-dense local reference launcher (non-chunked baseline for the local path); 3 refs, contradiction-cptmix-ruler-experiment lineage |
| `local_4b/Qwen3-4B-document-chunked-longctx-SFT-local.py` | 379 | Berkeley local | local_4b | **KEEP** | the 4B local document-chunked launcher -- required local-cluster-path coverage for the docchunk/landmark family; 2 refs |
| `local_4b/Qwen3-4B-fast-landmark-cptmix-5task-local.py` | 418 | Berkeley local | local_4b | **UNCERTAIN** | 2 refs (landmark-cptmix-5task memory); local landmark-5task twin -- keep if the local landmark-5task path is still exercised, else redundant with local_4b/document-chunked + sft_docchunk's local twin |
| `local_4b/run_q4b_contra_n250_landmark.sbatch` | 115 | Berkeley local | local_4b | **DROP** | concluded n=250 contradiction probe (early landmark-vs-standard comparison, superseded by the n=100/n=20 clean results and by ctc_suite's contradiction ladder); no reference found |
| `local_4b/run_q4b_contra_n250_standard.sbatch` | 100 | Berkeley local | local_4b | **DROP** | standard-arm twin of the n250 probe above; no reference found |
| `local_4b/run_q4b_cptmix_contra_local.sbatch` | 72 | Berkeley local | local_4b | **KEEP** | sbatch companion to the dense local launcher above |
| `local_4b/run_q4b_cptmix_sweep.sbatch` | 102 | Berkeley local | local_4b | **DROP** | CPT-mix-fraction sweep utility from the contradiction-cptmix-ruler-experiment (concluded 2026-06-25); no reference beyond that closed experiment |
| `local_4b/run_q4b_doc_oolong.sbatch` | 107 | Berkeley local | local_4b | **KEEP** | docchunk oolong local sbatch, complements the contra one above (both feed oolong-document-chunked-dense-landmark memory) |
| `local_4b/run_q4b_docchunk_contra.sbatch` | 106 | Berkeley local | local_4b | **KEEP** | sbatch companion to the docchunk local launcher above |
| `local_4b/run_q4b_fastlm_cpt40m.sbatch` | 80 | Berkeley local | local_4b | **DROP** | one-off 40M-token CPT-fraction probe; no reference found anywhere |
| `local_4b/run_q4b_lm_cptmix_5task.sbatch` | 110 | Berkeley local | local_4b | **UNCERTAIN** | sbatch companion to the above; 1 ref |
| `sft_5task/Qwen3-4B-compressive-5task-32k-nocpt-SFT.py` | 327 | Beaker | sft_5task | **DROP** | superseded family: sft_docchunk applies the identical 5-task mix + chunk mask on top; clean-repo-target-structure.md's recommended keep list (ctc_suite+sft_docchunk+attn_explore) omits sft_5task; baseline numbers it would reproduce already live in results-hub / docchunk-4b-5task-ladder-results.md as 'dense'/'comp' reference columns |
| `sft_5task/Qwen3-4B-compressive-5task-32k-nocpt-fixnq-SFT.py` | 322 | Beaker | sft_5task | **DROP** | superseded family (see above) |
| `sft_5task/Qwen3-4B-compressive-cptmix-5task-32k-SFT.py` | 324 | Beaker | sft_5task | **DROP** | superseded family (see above) |
| `sft_5task/Qwen3-4B-dense-5task-32k-nocpt-SFT.py` | 312 | Beaker | sft_5task | **UNCERTAIN** | 1 ref (single-task-ladder-experiment.md names it as the adaptation base for the dense singletask-ladder variant) -- superseded as a standalone launcher, but its recipe was reused elsewhere; keep only if that lineage still matters |
| `sft_5task/Qwen3-4B-dense-5task-32k-nocpt-fixnq-SFT.py` | 308 | Beaker | sft_5task | **DROP** | superseded family (see above) |
| `sft_5task/Qwen3-4B-dense-cptmix-5task-32k-SFT.py` | 345 | Beaker | sft_5task | **DROP** | superseded family (see above) |
| `sft_5task/Qwen3-4B-dense-cptmix-5task-64k-SFT.py` | 338 | Beaker | sft_5task | **DROP** | superseded family (see above) |
| `sft_5task/Qwen3-4B-fast-landmark-5task-32k-nocpt-SFT.py` | 274 | Beaker | sft_5task | **DROP** | superseded family (see above) |
| `sft_5task/Qwen3-4B-fast-landmark-cptmix-5task-32k-SFT.py` | 334 | Beaker | sft_5task | **DROP** | superseded family (see above) |
| `sft_5task/Qwen3-4B-fast-landmark-cptmix-5task-64k-SFT.py` | 334 | Beaker | sft_5task | **DROP** | superseded family (see above) |
| `sft_longctx/Qwen3-4B-dense-10task1k-cpt40-64k-SFT.py` | 250 | Beaker | sft_longctx | **DROP** | pre-5task-mix generation; family README states explicitly 'superseded by sft_5task/ + sft_docchunk/ for new experiments' |
| `sft_longctx/Qwen3-4B-dense-10task1k-cpt40-8k-debug-SFT.py` | 233 | Beaker | sft_longctx | **DROP** | pre-5task-mix generation (see above) |
| `sft_longctx/Qwen3-4B-dense-SFT.py` | 232 | Beaker | sft_longctx | **DROP** | pre-5task-mix generation (see above) |
| `sft_longctx/Qwen3-4B-dense-longctx-SFT.py` | 257 | Beaker | sft_longctx | **DROP** | pre-5task-mix generation (see above) |
| `sft_longctx/Qwen3-4B-dense-longctx-base32k-SFT.py` | 260 | Beaker | sft_longctx | **DROP** | pre-5task-mix generation (see above) |
| `sft_longctx/Qwen3-4B-dense-noruler-4k-CPTmix-SFT.py` | 252 | Beaker | sft_longctx | **DROP** | concluded probe (ruler-forgetting-experiment.md, answered 2026-06-21: CPT mixing helps, 15% too weak, 30% better); ALSO hardcodes beaker_launch_config.priority='normal', violating the repo's always-urgent rule -- do not reuse as-is even for reference |
| `sft_longctx/Qwen3-4B-dense-noruler-4k-SFT.py` | 214 | Beaker | sft_longctx | **DROP** | concluded probe + priority='normal' violation (see above) |
| `sft_longctx/Qwen3-4B-dense-noruler-5k-CPTmix-SFT.py` | 252 | Beaker | sft_longctx | **DROP** | concluded probe + priority='normal' violation (see above) |
| `sft_longctx/Qwen3-4B-dense-noruler-5k-SFT.py` | 214 | Beaker | sft_longctx | **DROP** | concluded probe + priority='normal' violation (see above) |
| `sft_longctx/Qwen3-4B-dense-unified-SFT.py` | 212 | Beaker | sft_longctx | **UNCERTAIN** | cited by name in records/landmark-packing-cp-task.md as the reference point for the still-open CP+packing validation task (memory: 'code+CPU tests done', GPU validation pending) -- drop only once that task is closed or its validation harness stops needing a live example script |
| `sft_longctx/Qwen3-4B-fast-landmark-SFT.py` | 246 | Beaker | sft_longctx | **DROP** | pre-5task-mix generation (see above) |
| `sft_longctx/Qwen3-4B-fast-landmark-longctx-SFT.py` | 271 | Beaker | sft_longctx | **DROP** | pre-5task-mix generation (see above) |
| `sft_longctx/Qwen3-4B-fast-landmark-longctx-base32k-SFT.py` | 275 | Beaker | sft_longctx | **DROP** | pre-5task-mix generation (see above) |
| `sft_longctx/Qwen3-4B-fast-landmark-packed-SFT.py` | 252 | Beaker | sft_longctx | **UNCERTAIN** | CP+packing reference script (see dense-unified-SFT.py above) |
| `sft_longctx/Qwen3-4B-fast-landmark-unified-SFT.py` | 251 | Beaker | sft_longctx | **UNCERTAIN** | CP+packing reference script (see dense-unified-SFT.py above) |
| `sft_longctx/Qwen3-4B-landmark-SFT.py` | 247 | Beaker | sft_longctx | **DROP** | pre-5task-mix generation (see above) |
| `sft_longctx/Qwen3-4B-sparse-landmark-SFT.py` | 251 | Beaker | sft_longctx | **DROP** | pre-5task-mix generation (see above) |
| `sft_longctx/Qwen3-4B-sparse-landmark-longctx-SFT.py` | 281 | Beaker | sft_longctx | **DROP** | pre-5task-mix generation (see above) |
| `sft_longctx/Qwen3-4B-sparse-landmark-longctx-base32k-SFT.py` | 285 | Beaker | sft_longctx | **DROP** | pre-5task-mix generation (see above) |
| `sft_longctx/Qwen3-4B-sparse-landmark-packed-SFT.py` | 246 | Beaker | sft_longctx | **UNCERTAIN** | CP+packing reference script (see dense-unified-SFT.py above) |
| `sft_longctx/Qwen3-4B-sparse-landmark-unified-SFT.py` | 231 | Beaker | sft_longctx | **UNCERTAIN** | CP+packing reference script (see dense-unified-SFT.py above) |
| `singletask_ladder/Qwen3-4B-compressive-singletask-ladder-32k-SFT.py` | 204 | Beaker | singletask_ladder | **DROP** | family README itself labels this 'Legacy / superseded (kept for reference)' -- the compressive-only predecessor the 3-variant launcher generalizes |
| `singletask_ladder/Qwen3-4B-docchunk-oolong-sanity-8k-SFT.py` | 219 | Beaker | singletask_ladder | **DROP** | one-off dedicated-oolong CoT-ablation probe (docchunk-4b-5task-ladder-results.md's CoT experiment, concluded 2026-07-05); its finding (CoT rescues docchunk oolong 2.7-3.1x) is now recorded, no need to re-run |
| `singletask_ladder/Qwen3-4B-docchunk-singletask-ladder-10k-SFT.py` | 284 | Beaker | singletask_ladder | **SUPERSEDED** | docchunk_dense variant of the 3-variant launcher; 2 refs; same supersession as below |
| `singletask_ladder/Qwen3-4B-singletask-ladder-32k-10k-3variant-SFT.py` | 331 | Beaker | singletask_ladder | **SUPERSEDED** | per-task-ladder SFT superseded by ctc_suite's task-agnostic joint trainer (train_ctc_suite.py), which generalizes the same idea (per-task length ladders, mask variants) with curriculum mask-mix, provenance stamping, and a shared codepath across all ~26 tasks instead of 5 |
| `singletask_ladder/Qwen3-4B-singletask-ladder-SFT-local.py` | 272 | Berkeley local | singletask_ladder | **SUPERSEDED** | local torchrun twin of the 3-variant launcher; 3 refs; superseded along with the family by ctc_suite (which already has its own local path, run_ctc_local.sbatch) |
| `singletask_ladder/run_q4b_beaker_multirung_eval.py` | 180 | Beaker | singletask_ladder | **UNCERTAIN** | 10 refs -- the most-cited file in this family, and outlier-datascale-sweep.md records evals still owed as of 2026-07-21 using exactly this driver; keep until that eval debt is confirmed closed, otherwise DROP alongside the rest of the family |
| `singletask_ladder/run_q4b_singletask_ladder.sbatch` | 69 | Berkeley local | singletask_ladder | **DROP** | family README itself labels this 'earlier local sbatch', pre-dating the *_stl_* family below |
| `singletask_ladder/run_q4b_stl_eval.sbatch` | 109 | Berkeley local | singletask_ladder | **SUPERSEDED** | standalone eval variant of the traineval/multirung sbatches -- same family, collapses into one parameterized eval sbatch |
| `singletask_ladder/run_q4b_stl_multirung_eval.sbatch` | 126 | Berkeley local | singletask_ladder | **SUPERSEDED** | multirung variant of the same eval sbatch cluster |
| `singletask_ladder/run_q4b_stl_traineval.sbatch` | 87 | Berkeley local | singletask_ladder | **SUPERSEDED** | train+eval combined variant of the same cluster |
| `singletask_ladder/run_q4b_stl_validation.sbatch` | 65 | Berkeley local | singletask_ladder | **SUPERSEDED** | validation-only variant of the same cluster |
| `cpt/Qwen3-0.6B-dense-dolma3longmino_single_node.py` | 187 | Beaker | cpt | **DROP** | no reference found anywhere; redundant sibling in the dense/landmark/fast-landmark/sparse-landmark x {8node,lr-variant,single_node,Qwen3/3.5} CPT grid -- collapses into one parameterized CPT launcher (see collapse cluster) |
| `cpt/Qwen3-0.6B-landmark-dolma3longmino_single_node.py` | 200 | Beaker | cpt | **DROP** | no reference found; CPT grid (see above) |
| `cpt/Qwen3-4B-base-dense-8node.py` | 207 | Beaker | cpt | **DROP** | 8-node scale-up sibling of the lr-tagged base scripts; no reference found; same collapse cluster (a --nodes flag on the parameterized launcher) |
| `cpt/Qwen3-4B-base-dense-dolma3longmino.py` | 194 | Beaker | cpt | **DROP** | no reference found; CPT grid (see above) |
| `cpt/Qwen3-4B-base-dense-lr1.1e-4.py` | 207 | Beaker | cpt | **KEEP** | produced amandab/q4b-base-dense-lr1.1e-4/step2385, the dense CPT base still used as init for singletask-ladder-experiment and outlier-datascale-sweep |
| `cpt/Qwen3-4B-base-dense-lr9.6e-4.py` | 207 | Beaker | cpt | **DROP** | no reference found; CPT grid (see above) |
| `cpt/Qwen3-4B-base-fast-compressive-landmark-dolma3longmino.py` | 218 | Beaker | cpt | **KEEP** | produced the compressive CPT base (qwen4b-base-compressive-lr1.1e-4/step2385) singletask-ladder-experiment.md names as required init for the compressive variant (landmark-token emb + grouped-softmax trained there) |
| `cpt/Qwen3-4B-base-fast-landmark-8node.py` | 222 | Beaker | cpt | **DROP** | 8-node scale-up sibling; no reference found; same collapse cluster |
| `cpt/Qwen3-4B-base-fast-landmark-dolma3longmino.py` | 209 | Beaker | cpt | **DROP** | no reference found; CPT grid (see above) |
| `cpt/Qwen3-4B-base-fast-landmark-lr1.1e-4.py` | 222 | Beaker | cpt | **KEEP** | produced amandab/q4b-base-fast-landmark-lr1p1e-4/step2385, the landmark CPT base referenced the same way |
| `cpt/Qwen3-4B-base-landmark-dolma3longmino.py` | 206 | Beaker | cpt | **DROP** | no reference found; CPT grid (see above) |
| `cpt/Qwen3-4B-base-sparse-landmark-8node.py` | 235 | Beaker | cpt | **DROP** | 8-node scale-up sibling; no reference found; same collapse cluster |
| `cpt/Qwen3-4B-base-sparse-landmark-dolma3longmino.py` | 220 | Beaker | cpt | **DROP** | no reference found; CPT grid (see above) |
| `cpt/Qwen3-4B-dense-dolma3longmino.py` | 194 | Beaker | cpt | **DROP** | no reference found; CPT grid (see above) |
| `cpt/Qwen3-4B-fast-landmark-dolma3longmino.py` | 209 | Beaker | cpt | **DROP** | no reference found; CPT grid (see above) |
| `cpt/Qwen3-4B-fast-landmark_single_node.py` | 181 | Beaker | cpt | **DROP** | no reference found; CPT grid (see above) |
| `cpt/Qwen3-4B-landmark-dolma3longmino.py` | 206 | Beaker | cpt | **DROP** | no reference found; CPT grid (see above) |
| `cpt/Qwen3-4B-sparse-landmark-dolma3longmino.py` | 220 | Beaker | cpt | **DROP** | no reference found; CPT grid (see above) |
| `cpt/Qwen3-4B-sparse-landmark_single_node.py` | 186 | Beaker | cpt | **DROP** | no reference found; CPT grid (see above) |
| `cpt/Qwen3.5-4B-dense-dolma3longmino.py` | 212 | Beaker | cpt | **DROP** | no reference found; CPT grid (see above) |
| `cpt/Qwen3.5-4B-fast-landmark-dolma3longmino.py` | 241 | Beaker | cpt | **DROP** | no reference found; CPT grid (see above) |
| `cpt/Qwen3.5-4B-landmark-dolma3longmino.py` | 233 | Beaker | cpt | **DROP** | no reference found; CPT grid (see above) |
| `cpt/Qwen3.5-4B-sparse-landmark-dolma3longmino.py` | 240 | Beaker | cpt | **DROP** | no reference found; CPT grid (see above) |
| `cpt/debug/Qwen3-0.6B-landmark-longmino-4k-eager_single_node.py` | 232 | Berkeley local (smoke) | cpt/debug | **DROP** | one-node smoke-test scaffolding predating the CPT launchers going to Beaker; explicitly named 'smoke runs' in the family README; no reference found; collapse cluster |
| `cpt/debug/Qwen3-0.6B-landmark-longmino-4k_single_node.py` | 220 | Berkeley local (smoke) | cpt/debug | **DROP** | smoke-test scaffolding (see above) |
| `cpt/debug/Qwen3-0.6B-landmark-wikitext_single_node.py` | 234 | Berkeley local (smoke) | cpt/debug | **DROP** | smoke-test scaffolding (see above) |
| `cpt/debug/Qwen3-0.6B-landmark_single_node-no-compile.py` | 232 | Berkeley local (smoke) | cpt/debug | **DROP** | smoke-test scaffolding (see above) |
| `cpt/debug/Qwen3-0.6B-landmark_single_node.py` | 236 | Berkeley local (smoke) | cpt/debug | **DROP** | smoke-test scaffolding (see above) |
| `cpt/debug/Qwen3-0.6B-wikitext_single_node.py` | 191 | Berkeley local (smoke) | cpt/debug | **DROP** | smoke-test scaffolding (see above) |
| `cpt/debug/Qwen3-4B-landmark-longmino-4k_single_node.py` | 238 | Berkeley local (smoke) | cpt/debug | **DROP** | smoke-test scaffolding (see above) |
| `cpt/debug/Qwen3-4B-landmark.py` | 235 | Berkeley local (smoke) | cpt/debug | **DROP** | smoke-test scaffolding (see above) |
| `cpt/debug/Qwen3-4B-landmark_single_node.py` | 236 | Berkeley local (smoke) | cpt/debug | **DROP** | smoke-test scaffolding (see above) |
| `cpt/debug/Qwen3-4B-landmark_single_node_no_kernel.py` | 238 | Berkeley local (smoke) | cpt/debug | **DROP** | smoke-test scaffolding (see above) |
| `cpt/debug/Qwen3-4B-long-context.py` | 202 | Berkeley local (smoke) | cpt/debug | **DROP** | smoke-test scaffolding (see above) |
| `cpt/debug/Qwen3.5-4B-long-context.py` | 203 | Berkeley local (smoke) | cpt/debug | **DROP** | smoke-test scaffolding (see above) |
| `cpt/debug/Qwen3.5-4B-wikitext_single_node.py` | 191 | Berkeley local (smoke) | cpt/debug | **DROP** | smoke-test scaffolding (see above) |
| `cpt/interleaved/Qwen3-4B-interleaved-2sparse-1reg-dolma3longmino.py` | 233 | Beaker | cpt/interleaved | **DROP** | abandoned data-interleaving-pattern ablation (sparse:reg ratio / alternating / half-full split); no reference anywhere in records/results/memory; collapse cluster (one --interleave-pattern flag) |
| `cpt/interleaved/Qwen3-4B-interleaved-3sparse-1reg-dolma3longmino.py` | 234 | Beaker | cpt/interleaved | **DROP** | abandoned ablation (see above) |
| `cpt/interleaved/Qwen3-4B-interleaved-4sparse-1reg-dolma3longmino.py` | 236 | Beaker | cpt/interleaved | **DROP** | abandoned ablation (see above) |
| `cpt/interleaved/Qwen3-4B-interleaved-alternating-dolma3longmino.py` | 227 | Beaker | cpt/interleaved | **DROP** | abandoned ablation (see above) |
| `cpt/interleaved/Qwen3-4B-interleaved-first-half-full-dolma3longmino.py` | 230 | Beaker | cpt/interleaved | **DROP** | abandoned ablation (see above) |
| `cpt/interleaved/Qwen3-4B-interleaved-second-half-full-dolma3longmino.py` | 229 | Beaker | cpt/interleaved | **DROP** | abandoned ablation (see above) |
| `evals/eval_dense_base.sbatch` | 35 | Berkeley local | evals | **DROP** | superseded by ctc_suite/ctc_suite/run_rung_eval.py (99 refs), the now-standard eval driver; no reference found |
| `evals/eval_dense_local_vllm.sbatch` | 30 | Berkeley local | evals | **UNCERTAIN** | 1 ref; run_rung_eval.py's native backend is the current default and vLLM support there is still a TODO per clean-repo-target-structure.md -- this may be the only working local vLLM eval path until backends/vllm.py lands |
| `evals/eval_gen_native.sbatch` | 36 | Berkeley local | evals | **DROP** | superseded by run_rung_eval.py; no reference found |
| `evals/eval_ladder_32k.sbatch` | 76 | Berkeley local | evals | **DROP** | superseded by run_rung_eval.py's per-rung eval; no reference found |
| `evals/eval_landmark_native.sbatch` | 35 | Berkeley local | evals | **DROP** | superseded by run_rung_eval.py; no reference found |
| `evals/eval_landmark_topk.sbatch` | 36 | Berkeley local | evals | **DROP** | superseded by run_rung_eval.py; no reference found |
| `evals/run_eval_q4b_lc.sbatch` | 90 | Berkeley local | evals | **DROP** | superseded by run_rung_eval.py; no reference found |
| `goldgrad/Qwen3-0.6B-goldgrad-contradiction-n20-SFT-local.py` | 450 | Berkeley local | goldgrad | **DROP** | concluded negative-result probe: measured speedup was 1.00x ('a probe, not an O(1) backward'), and n100-clean-results-2026-07-14.md shows the effect does NOT replicate on leak-free data -- diagnosis complete, nothing left to re-run |
| `goldgrad/eval_q06b_goldgrad_contra.sbatch` | 111 | Berkeley local | goldgrad | **DROP** | concluded negative-result probe (see above) |
| `goldgrad/run_q06b_goldgrad_contra.sbatch` | 141 | Berkeley local | goldgrad | **DROP** | concluded negative-result probe (see above) |
| `goldhop/eval_q06b_hopgold_contra_n50.sbatch` | 135 | Berkeley local | goldhop | **DROP** | concluded probe (multihop-gold-routing-experiment.md 'proposal, awaiting refinement' -> results/hopgold-n50-summary-ladder.md exists, i.e. run to completion); no active follow-on found |
| `goldhop/run_q06b_hopgold_stage1.sbatch` | 192 | Berkeley local | goldhop | **DROP** | concluded probe (see above) |
| `probes/sanity_check_packing.py` | 322 | n/a | probes | **KEEP** | general-purpose packing-correctness checker, cheap (small file), 2 refs, no dependency on a specific dead experiment |
| `probes/scan_doc_lengths.py` | 80 | n/a | probes | **KEEP** | general-purpose shard doc-length stats tool, cheap, 2 refs |

---

## 6. Open items for Prasann (the 13 UNCERTAIN rows, consolidated)

1. **Gemma-3 (`ctc_suite/convert_gemma3_base.py`)** — is it the "further family owed" from paper-v2
   item 9, or dead? No results directory exists yet either way.
2. **OLMo3 SWA ablation (`ctc_suite/olmo3_swa_ablation.py`)** — is the OLMo3 arm still being extended?
3. **`ctc_suite/ctc_suite/eval_ctc_lambda.sbatch`** — does `run_rung_eval.py` already cover lambda
   eval, making this redundant?
4. **CP+packing reference scripts** (6 files in `sft_longctx/`) — is `landmark-cp-packing-task` (CLAUDE.md:
   "code+CPU tests done") still an open task needing these as GPU-validation examples, or closed?
5. **`local_4b/Qwen3-4B-fast-landmark-cptmix-5task-local.py` + its sbatch companion** — still exercised,
   or redundant with the docchunk/dense local pair already kept?
6. **`sft_5task/Qwen3-4B-dense-5task-32k-nocpt-SFT.py`** — keep only if the singletask-ladder lineage
   (itself mostly SUPERSEDED, §3) still matters.
7. **`singletask_ladder/run_q4b_beaker_multirung_eval.py`** — is the outlier-datascale-sweep eval debt
   (open as of 2026-07-21) closed? If yes, this drops with the rest of the family; if no, it may be the
   only working eval driver for that still-owed data.
8. **`evals/eval_dense_local_vllm.sbatch`** — confirm whether it's now redundant with `run_rung_eval.py`
   or the only working local vLLM eval path.

A wrongly-dropped live launcher is expensive to rediscover after the old repo is archived — these 8
items are cheap to resolve with a single question each before the corresponding files are excluded
from the port.
