# CTC dense-vs-chunked — session handoff (2026-07-24)

Handoff for a fresh agent continuing the CTC-suite (Qwen3.5-4B, full-attn vs document-chunked)
dense-vs-chunked comparison + the grid-review fixes. Overarching goal: **complete the comparison;
for any number that disagrees with the complexity hypothesis or looks low, verify the setting is
completely working.**

Grid: `results/ctc_suite/dense_vs_chunked_table.md`.
Artifact: https://claude.ai/code/artifact/15921b3d-c744-4ce1-8b93-3acb7d92c193
(file `<scratchpad>/ctc_grid.html`; data-driven `D` array; favicon 📊; update via same file path or `url=`).

---

# ⚠ UPDATE — 2026-07-24 evening: UNBLOCKED, queue MIGRATED TO LAMBDA

**Everything below this block about "THE BLOCKER", the queued Berkeley job IDs, and "Lambda is NOT
usable" is now STALE.** Read this section first.

What changed:

1. **The 9 Berkeley retrains (3353228-3353236) were all CANCELLED** to free the 8-GPU quota for
   `q35-4b-contra-128k-local` (job 3353237, 8×H200 sneetches, 24h wall). The Berkeley cluster is
   deliberately left alone.
2. **"Lambda is quota-full / NOT usable" was WRONG.** The 209G in `ctc_suite/ckpts` recorded as
   "others' ckpts" was **mine** — stale 0.8B stage-3 pilots + cancelled 4B partials. Deleted with
   user approval → quota **100% → 84.5% (1.10T/1.30T)**. (Quota accounting lags deletes by ~a minute;
   it still read 100% immediately after.)
3. **All 9 retrains now run on Lambda**, sequentially on `lambda-hyperplane03`, 8×A100 each,
   `preemptive_high`. Submitter: `debug/ctc_suite_lambda_migration/submit_queue_2026-07-24.sh`.
   Job IDs 109788-109796. Est. total ~30-35h.

   | job | run | task | variant | seq_len |
   |---|---|---|---|---|
   | 109788 | ctc-xabsence-full-4b-lam | xabsence | full | 4608 |
   | 109789 | ctc-xabsence-cmix-4b-lam | xabsence | chunked-mix | 4608 |
   | 109790 | obliq-mixedn-v2-twk40-4b-cmix-lam | retrieval | chunked-mix | 8192 |
   | 109791/2 | ctc-cycle-{full,cmix}-4b-fixed-lam | cycle | full / chunked-mix | 19968 |
   | 109793/4 | ctc-groups4-{full,cmix}-4b-fixed-lam | groups4 | full / chunked-mix | 24960 |
   | 109795/6 | ctc-grouping-{full,cmix}-4b-fixed-lam | grouping | full / chunked-mix | 36864 |

   Verified live: pilot 109788 trains cleanly, CE 0.846 → ~0.49, MFU ~18%, ~2,450 TPS/device.
   All 9 configs pre-validated with `train_ctc_suite.py --dry-run` on Lambda (family, marker set,
   seq_len vs `max_example_len`, mask-mix anneal `HARD-CHECK PASSED`).

4. **Checkpoints go to node-local `/tmp/prasann/ctc_suite/ckpts` on hyperplane03**, NOT NFS — so the
   1.30T quota stays flat. ⚠ **Eval MUST be node-pinned to `-w lambda-hyperplane03`.** Note `/data`
   is NOT writable on Lambda (root-owned, no per-user dir) — that attempt failed first; `/tmp` is
   2.2T, not quota'd, and verified to persist across jobs.
5. **Eval now works on Lambda** — it was never an air-gap limit. `run_ctc_lambda.sbatch` exported
   `HF_HOME=$LROOT/hf-cache`, **a path that never existed**; training didn't care (pre-tokenized
   shards) but eval loads a tokenizer and `HF_HUB_OFFLINE=1` makes a miss fatal. Fixed + verified
   (old → `OSError`; new → tokenizer loads, vocab 248,077).
6. New reference doc **`lambda_cluster.md`** (repo root, referenced from CLAUDE.md).

Open item re-scoped: **oolong "regenerate at scale" is bigger than it looks.** The correct-source
files (`oolong_test_synth_ctx*_splittrain.jsonl`) hold only **320 examples each = 2,240 total**, so
matching the deployed 21k shard needs genuinely NEW generation via
`generate_oolong_ladder_data.py` (CPU-only), *plus* a layout fix so the preamble stays FREE rather
than being wrapped as chunks — that layout mismatch is the original bug. Must also stay disjoint
from the v2 eval. Not started.

Still owed after the queue drains: eval each ckpt (both arms, node-pinned), obliq `-full` ladder
4k-32k, then update the grid + artifact.

---

## THE BLOCKER (why everything is stalled) — ⚠ SUPERSEDED, see UPDATE above

My 8-GPU per-user quota is fully consumed by **two OTHER-session jobs**:
`q35-4b-contra-256k-local` (job 3353224, 8×H200 sneetches) + `q3-iso64k` (job 3353222, 2×H200 horton).
The cluster HAS H200 capacity (cubbins/mcfuzz/mooney "mix"); the *per-user quota* is the wall. All CTC
retrains are PENDING behind it. **User decision pending: pause those jobs to unblock, or wait.**

## QUEUED retrains (pending on quota; auto-run as it frees). All submitted at gres=2.

| job | run-name | task | variant | data shard |
|---|---|---|---|---|
| 3353228 | ctc-cycle-full-4b-fixed | cycle | full | debug/ctc_fix_regen/cycle/shards_cycle_fixed |
| 3353229 | ctc-cycle-cmix-4b-fixed | cycle | chunked-mix | " |
| 3353230 | ctc-groups4-full-4b-fixed | groups4 | full | debug/ctc_fix_regen/groups4/shards_groups4_fixed |
| 3353231 | ctc-groups4-cmix-4b-fixed | groups4 | chunked-mix | " |
| 3353232 | ctc-grouping-full-4b-fixed | grouping | full | debug/ctc_fix_regen/grouping/shards_grouping_fixed |
| 3353233 | ctc-grouping-cmix-4b-fixed | grouping | chunked-mix | " |
| 3353234 | ctc-xabsence-full-4b | xabsence | full | debug/xabsence_fix/shards_xabsence_combined |
| 3353235 | ctc-xabsence-cmix-4b | xabsence | chunked-mix | " |
| 3353236 | obliq-mixedn-v2-twk40-4b-cmix | obliq_retrieval | chunked-mix | debug/obliq_synthetic/shards_mixedn_v2_twk40 |

Base for all: `BASE_SRC=/scratch/users/prasann/ctc_suite_lambda_stage/q35-4b-base-modelonly` (4B GDN hybrid).
Ckpts save to node-local `/data/prasann/ctc_suite/ckpts/<run>/` on whatever node they land — **node-pin
the eval to that host** (ckpts are NOT on shared fs). Preemption wipes node-local ckpts → may need re-run.

## DONE & banked (no compute needed)

1. **Chunk-leak audit** — `debug/ctc_vllm_validation/CHUNK_LEAK_AUDIT.md` + `validate_chunk_leak.py`.
   All chunked runs CLEAN (data docs isolated, train+eval, all rungs) EXCEPT oolong (train/eval wrapping
   mismatch — fixed, see below).
2. **Setting verification** — `records/ctc-setting-verification-2026-07-23.md`:
   - hpqa (qdmatch_hpqa): REAL, no bug (genuine O(M·N), sound grading). dense≈0.99 is the thesis.
   - cycle/groups4: rarity shortcut REAL (gold entity freq = **exactly 2.0** vs distractors 9.5→32,
     growing with N → the cycle is just the 3 rarest names). **NOTE: gold_doc_indices are 1-INDEXED.**
   - grouping: granularity-drift REAL (L0 level share 57%→0% across 2k→32k as OpenAlex fields run out).
   - contradiction: mask-mix anneal recipe is **CORRECT** (user verified anneal > no-anneal; an earlier
     "recipe bug" hypothesis was RETRACTED). Residual = dense-arm 32k digit artifact + missing 32k
     chunked rung. Do NOT change the anneal.
3. **scifact-dense fixed** (retrain): 0.963/0.948/0.926/0.899/0.879 (⚠ eval_size=300). Already in artifact.
4. **obliq ACCEPTED** at 2k=**0.620** (eval_size=486, below the 0.8 gate; verified genuine — data ceiling,
   4B≈0.8B). Ckpt: `/data/prasann/ctc_suite/ckpts/obliq-mixedn-v2-twk40-4b-full` on sneetches (VERIFY it
   survived preemption). Ladder evals (4k-32k) + cmix (3353236) still owed. Data: v2 mix with improved
   k40 twitter, shards at debug/obliq_synthetic/shards_mixedn_v2_twk40.

## Generator fixes — all 3 VALIDATED (code changed, data regenerated, gates passed)

- **cycle**: `src/corpus_reasoning/data/generate_cycle_data.py` build_example — cycle entities now share
  a global rank order (consecutive block), participate in background edges → freq-matched to distractors.
  Validated: gold≈distractor at every rung (was gold=2). Regen: `debug/ctc_fix_regen/cycle/{train.jsonl,
  eval_rungs/rung_{2048,4096,8192,16384}.jsonl}`; shards ready (20000 inst).
- **groups4**: `generate_groups4_data.py` sample_values — distractors may fall within tolerance of up to
  G-2 others (no spurious clique). Validated: 45-65% distractors now have a close neighbor (was 0%).
  Regen: `debug/ctc_fix_regen/groups4/{train.jsonl, eval_rungs/rung_{2048,4096,8192}.jsonl}`; shards
  ready (12000 inst, 3000 dropped >seq_len).
- **grouping**: `generate_arxiv_grouping_data.py` — new `--level-mix` (fixed per-level quota, default
  25/25/25/25), capacity-aware k, larger eval pool. Validated: level-mix flat 25/25/25/25 + mean cluster
  ~2.6-3.0 across ALL rungs (was drifting 2.27→1.16). Fetched eval pool `debug/ctc_fix_regen/grouping/
  openalex_eval2024_fetch.jsonl` (21.5k papers). Regen: `debug/ctc_fix_regen/grouping/{train.jsonl,
  eval_rungs/rung_*.jsonl}`; shards ready (20000 inst). grouping_labeled shares this eval dir.

## oolong fix (root-caused; result confounded)

Root cause: deployed `oolong_train` shard was built from the WRONG source
(`oolong_ladder_train_combined`, wrapped instruction/question/header as chunks) while eval comes from
the synth split. Correct source = `oolong_test_synth_ctx*_splittrain.jsonl`. Rebuilt:
`debug/oolong_fix/{oolong_synth_splittrain_combined.jsonl, shards_oolong_corrected}` (1920 inst, cot=plan).
Retrained -cmix (`ctc-oolong-cmix-4b-fixed`) + -full (`ctc-oolong-full-4b-fixed`) — ckpts on sneetches
/data (VERIFY survived preemption). **Result: fixed cmix chunked @2k = 0.534 vs deployed 0.628 —
CONFOUNDED** (correct-source data is only 1920 ex vs deployed 21000; can't attribute the drop to the
wrapping fix). oolong-full dense eval was preempted. **To do it right: regenerate correct-source oolong
at scale** (open decision).

## xabsence (new task — ~80% pre-built)

Eval harness fully wired already (run_rung_eval alias "N2", `_eval_absence` scorer, "Unmatched: [N]"
parsing). Pubmed-**abstracts** data exists. Combined train shards `debug/xabsence_fix/
shards_xabsence_combined` (6000 ex from p8/p18/p48, **maxlen only 4451** — abstracts are short, so
**16k/32k eval rungs are extrapolation — flag inline**). Retrains keep getting preempted (ckpt lost).
Eval rungs at `/scratch/users/prasann/ctc_suite_staged/eval_rungs/xabsence/rung_*.jsonl`.

## KEY INFRA GOTCHAS (learned this session)

- **`run_ctc_local.sbatch` header hardcodes `--gres=gpu:H200:8`** → NGPU=2 only sets torchrun nproc, NOT
  the allocation. A `bash run_ctc_local.sbatch` self-submit grabs 8 GPUs = whole quota, serializing +
  starving everything. FIX: submit via explicit `sbatch --gres=gpu:H200:2 --partition=jsteinhardt
  --qos=preemptive_high --account=site --time=06:00:00 --job-name=<RUN> --export=ALL,TASK=...,NGPU=2,...
  run_ctc_local.sbatch` (SLURM_JOB_ID set → skips self-submit → honors the --gres override). Use this.
- **New reusable eval launcher**: `debug/ctc_vllm_validation/eval_ctc_native.sbatch` — env knobs
  CKPT/TASK/VARIANT/ARM/SCALE/RUNG/EVAL_JSONL/NGPU/MAXLEN/OUTROOT. `--variant dense --arm full` for the
  dense arm, `--variant chunked --arm chunked-mix` for the chunked arm. Node-pin (`-w <host>`) to the
  training node. Native olmo_core path (run_rung_eval). The eval label prints "qwen3.5-0.8b" hardcoded —
  IGNORE the label; the arch comes from the ckpt's config.json (verify d_model=2560/n_layers=32 = 4B).
- **Lambda is NOT usable**: quota-full on both my projects (sewonm LROOT has 209G of others' ckpts;
  berkeleynlp project on lambda also full). No shared writable staging. Don't delete others' ckpts. ssh
  lambda works (→ lambda-headnode02); run_ctc_lambda.sbatch has SAVE_ROOT/WORK_ROOT overrides but the
  data-staging quota is the blocker. Stay local.
- run_ctc_local.sbatch also gained `MODEL_FAMILY` (auto from shard marker_set) and `MIX_START_P/MIX_END_P`
  (constant-p mask-mix — NOT needed, anneal is correct) env passthroughs this session.
- convert command template (CPU, corpus-reasoning-olmo env, PYTHONPATH=src):
  `python src/scripts/data/convert_unified_to_document_landmark.py --input-jsonl <f> --task <t> --emit
  dense --chunk-by document --cot-mode none --marker-set qwen3_5 --tokenizer Qwen/Qwen3.5-0.8B-Base
  --seq-len 40960 --out-dir <o>` (oolong uses `--chunk-by line --cot-mode plan`).
  ⚠ 2026-07-26: do NOT pass `--item-regex '||'` — a bare `||` is an empty-branch alternation that
  matches every line and wraps the preamble as chunks (the CHUNK_LEAK_AUDIT oolong leak). The
  converter default `r"\|\|"` is correct and the converter now rejects empty-matching regexes.

## NEXT STEPS (in order)

1. Unblock quota (user pauses competing jobs, or wait). Then the 9 queued retrains run.
2. As each completes, eval it with `eval_ctc_native.sbatch` node-pinned to its host — BOTH arms
   (dense on -full ckpt, chunked on -cmix ckpt) across each task's rungs:
   cycle 2k/4k/8k/16k · groups4 2k/4k/8k · grouping 2k/4k/8k/16k/32k · xabsence 2k/4k/8k/16k/32k
   (16k/32k=extrapolation) · obliq 2k-32k. Use the FIXED eval ladders in `debug/ctc_fix_regen/<t>/
   eval_rungs/` for cycle/groups4/grouping (NOT the stale staged ones).
3. Finish obliq: -full ladder evals (4k-32k, ckpt on sneetches) + -cmix (3353236) → fill its row.
4. Update `results/ctc_suite/dense_vs_chunked_table.md` + the artifact with all corrected numbers
   (show old→new for the fixed tasks so the fix effect is visible; flag xabsence 16k/32k extrapolation,
   scifact/obliq eval_size). Note oolong 0.534 confound.
5. (Optional, open decision) regenerate correct-source oolong at scale to de-confound its number.
6. Promote the validated regen eval ladders from `debug/ctc_fix_regen/` to the staged eval_rungs tree
   if you want them canonical (originals under /scratch/.../ctc_suite_staged/eval_rungs were left intact).

## OPEN DECISIONS FOR USER
1. Pause `q35-4b-contra-256k-local` (3353224) / `q3-iso64k` (3353222) to free quota? (unblocks everything)
2. oolong: regenerate correct-source data at scale (de-confound), or accept 0.534?

## Memory files written this session (in the auto-memory dir)
- `chunk-leak-audit-oolong-preamble.md` (+ MEMORY.md line)
- (`maskmix-anneal-to-zero-recipe-bug.md` was DELETED — hypothesis refuted by user)
