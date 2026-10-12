# The FINAL CTC suite — 22 tasks

**Frozen 2026-08-12 by prasann; amended 2026-08-13** to settle the open question below — PubMed
contradiction IS in the figure, as row 19, scored on the realistic/IID ladder. This is the task
roster for the CTC-suite figure (Qwen3.5-4B, dense vs document-chunked, 2k→32k token ladder).
Anything not on this list is out of the figure, whatever else exists on disk.

Supersedes the ~26–28-row roster in `records/ctc-suite-scaling-plan.md` §3. That plan stays as the
method/infra reference; **this file is the roster of record.**

## The 22 rows

| # | row | task dir | metric | class | notes |
|---|---|---|---|---|---|
| 1 | fiqa | `fiqa` | gold_id_f1 | O(N) | BEIR FiQA retrieval |
| 2 | nq | `nq` | gold_id_f1 | O(N) | CE-filtered p10 pool |
| 3 | hpqa | `hotpotqa` | gold_id_f1 | O(N) | |
| 4 | qdmatch fiqa | `qdmatch_fiqa` | pair_f1 | O(N²) | |
| 5 | qdmatch nq | `qdmatch_nq` | pair_f1 | O(N²) | |
| 6 | qdmatch hpqa | `qdmatch_hpqa` | pair_f1 | O(N²) | |
| 7 | outlier amazon | `outlier_amzn` | set_f1 | O(N) | **corrected 2026-08-13** — was listed O(NM) here; the attribute set (star rating / product category) is small and FIXED, so there is no category discovery. Only outlier scale-k (row 8) grows M with N |
| 8 | outlier scale-k | `outlier` | set_f1 | O(NM) | K ≈ n/9.5, M scales with N |
| 9 | outlier fix-k | `outlier_fixedM` | set_f1 | O(N) | K pinned at 3 — the control for row 8 |
| 10 | OOLONG | `oolong` | partial_credit | O(N) | |
| 11 | grouping | `grouping` | pairwise_f1 | O(NM) | OpenAlex |
| 12 | absence | `absence_gutenberg` | set_f1 | O(N²) | Gutenberg text-diff |
| 13 | xabsence | `xabsence` | set_f1 | O(N²) | exact-copy twins over 3-sentence chunks |
| 14 | rerank | `rerank` | ce_pos_recall | O(N) | **repriced 2026-08-14**: fraction of the CE-POSITIVE docs (median 3 / p90 5 per example) in the emitted top-10. mrr@10 saturated ~0.98 (single qrel); the interim CE-top-10 variant was rejected by measurement (~0.5 luck ceiling — ranks 4-10 are ordering noise). mrr@10/recall@10 still emitted as secondary; pre-reprice rows are not comparable |
| 15 | msmarco | `msmarco` | gold_id_f1 | O(N) | |
| 16 | re-order | `reorder` | kendall_tau | O(N²) | Gutenberg |
| 17 | obliq | `obliq_twitter` | gold_id_f1 | O(N) | the **twitter rebuild** — `obliq_retrieval` is VOID |
| 18 | niah contra | `niah` | gold_id_f1 | O(N) | NIAH-contradiction |
| 19 | contradiction (realistic) | `contradiction_iid` | set_f1 | O(N²) | PubMed multi-claim, **IID `realistic`-mode ladder** — see below |
| 20 | strmatch | `strmatch` | set_f1 | O(N²) | |
| 21 | textgroups | `textgroups` | textgroups_f1 | O(N³) | reclassified from O(NM) 2026-08-14 (prasann): triple discovery is a C(N,3) scan |
| 22 | scifact | `scifact` | gold_id_f1 | O(N) | BEIR SciFact |

Dropped from the older roster: cycle, groups4, mathmatch, grouping_labeled, helmet_qa,
helmet_summ, redundancy, absence-official, qdmatch-obliq, obliq_retrieval.

**Settled 2026-08-13 — PubMed `contradiction` IS in the figure, and the suite is 22 rows.** Row 18
is *NIAH*-contradiction (`niah`), a retrieval task; row 19 is the PubMed multi-claim task, the
project's flagship high-CTC example. They are different tasks and both stay.

⚠ **Row 19 must be scored on `contradiction_iid`, never on `contradiction_clean`.** Training is
`realistic`-mode gold pairs, and the `both`-mode ladder is a different generator whose gold pairs
are 38% near-duplicates — see [[contradiction-train-eval-mode-mismatch]] and
`records/contradiction-train-eval-non-iid.md`. The same checkpoints, nothing retrained:

| ladder | dense 2k→32k | chunked 2k→32k | gap at 32k |
|---|---|---|---|
| `contradiction_clean` (`both`-mode — **do not use**) | 0.843 → 0.559 | 0.402 → 0.191 | +0.368, *narrowing* with n |
| `contradiction_iid` (`realistic`, mode-matched) | **0.990 → 0.946** | **0.861 → 0.699** | **+0.247, widening** with n |

The mode swap is the whole result: dense does not collapse with corpus size (13.6× more documents
costs 0.043 f1), and the dense–chunked gap *widens* (0.128 → 0.247) where the mismatched ladder
showed it narrowing. Never mix the two ladders in one figure.

Its 2k column is the `rung_2560` / n=56 file, not 2048: n=44 sat below the training minimum of 52,
so the IID rebuild starts one step up. The other four rungs are n-identical to `contradiction_clean`
(92 / 187 / 379 / 762), which is what isolates the mode as the only change. The remap is explicit in
`build_suite_table.py` (`RUNG_ALIAS`), not silent.

## Coverage as of 2026-08-12 (row 19 added 2026-08-13)

Eval ladders (do the five rung files exist) vs numbers (is there a graded 4B result for each arm).
Numbers harvested by `debug/ctc_final_suite/harvest_grades.py`; merged table in
`debug/ctc_final_suite/suite_table.json`.

| row | ladder 2k–32k | dense cells | chunked cells | what's blocking |
|---|---|---|---|---|
| fiqa | ✗ **none** | 0/5 | 0/5 | `eval_rungs/fiqa/` is empty; the pilot's `rung_2048.jsonl` was 0 bytes. Train pool + BEIR test sets are ready on cubbins `ctc_suite_data/fiqa/`. **Build the ladder, then train + eval both arms.** |
| nq | ✓ (base + `ext32k`) | 5/5 | 5/5 | — |
| hpqa | ✓ (base + `ext32k`) | 5/5 | 5/5 | dense backfilled 2026-08-12 (the old `-full` ckpt was a corrupt S3 distcp) |
| qdmatch fiqa | ~ built, not staged | 0/5 | 0/5 | 5 rungs sit in cubbins `/data/prasann/qdmatch_fiqa`; not copied into `eval_rungs/`, no training run |
| qdmatch nq | ✓ | 0/5 | 0/5 | 4B: **both arms finished training 2026-08-12 (Beaker, exit 0)** after the truncated-shard repair; **no 4B eval submitted.** The 0.8b/2b model-scale evals *were* submitted and all four FAILED — see the export bug below |
| qdmatch hpqa | ✓ | 5/5 | 5/5 | rung labels run 0.74–0.82× of the stated tokens |
| outlier amazon | ✓ | 5/5 | 5/5 | rung labels run 1.12–1.24× |
| outlier scale-k | ✓ | 5/5 | 5/5 | — |
| outlier fix-k | ✓ (`eval_rungs_fixedM`) | 0/5 | 0/5 | **both arms finished training 2026-08-12 (Beaker, exit 0)**; **no eval has ever been submitted** — nothing named `fixedM` in sacct past the pilot + data build |
| OOLONG | ✓ | 0/5 | 0/5 | ⚠ **BOTH arms retraining on VESSL (2026-08-12)** off a rebuilt shard. All ten existing cells are VOID — see below |
| grouping | ✓ | 5/5 | 5/5 | regraded after the parser fix — old table's 0.439/0.358/0.186 numbers are dead |
| absence | ✗ 2k–16k only | 3/5 | 4/5 | 32k rung never built (`eval_rungs_ext32k/absence_gutenberg/` is empty); dense 16k also ungraded. ⚠ labels run 3.0–3.6× |
| xabsence | ✓ | 5/5† | 5/5† | † numbers exist only in the 2026-07-27 markdown — no grade JSON on any node. ⚠ dense score is half-trivial (A-then-B block layout) |
| rerank | ✓ (`tokenaccurate`) | 5/5 | 5/5 | rebuilt + fully regraded 2026-08-12 |
| msmarco | ✓ (`tokenaccurate`) | 5/5 | 5/5 | rebuilt + fully regraded 2026-08-12 |
| re-order | ✓ 2k–16k | 4/5 | 4/5 | 32k capped by rung policy — deliberate, not a gap |
| obliq | ✓ (`debug/obliq_twitter/eval_rungs`) | 5/5† | 5/5† | † table-only, as above. ⚠ eval_size=126 |
| niah contra | ✓ | 4.5/5 | 5/5 | dense 2k **conflicts**: 0.164 on disk (pre-fix run, 07-20) vs 0.988 in the table. Re-grade it. ⚠ labels run 0.51× |
| contradiction (realistic) | ✓ (`eval_rungs/contradiction_iid`, 2560/4k/8k/16k/32k) | 5/5 | 5/5 | — complete, parse_rate 1.00 throughout. ⚠ score only this ladder, not `contradiction_clean` |
| strmatch | ✓ | 4/5 | 5/5 | dense 32k ungraded. ⚠ labels run 0.60× |
| textgroups | ✓ | 5/5 | 5/5 | labels drift 0.89× → 1.55× across the ladder |
| scifact | ✓ | 5/5 | 5/5 | ⚠ eval_size=300; the old "dense ckpt collapsed" verdict is stale — the retrained `-full` scores 0.963→0.879 |

**Totals: 81/110 dense cells, 83/110 chunked, plus 10 superseded (oolong).** Five rows (fiqa,
qdmatch fiqa, qdmatch nq, outlier fix-k, oolong) hold 50 of the 56 non-current cells.

## Why oolong's ten cells are void

Both arms are retraining on VESSL (`ctcms-oolong-{full,cmix}-4b-vsl`, moved off jupiter at 16:44 on
2026-08-12) against a shard rebuilt by `debug/ctc_modelscale/finalize_oolong_ctc.sbatch`. Three
things changed at once, and each one alone would break comparability:

- **The leak is gone.** The July chunked row was trained on a shard built with a bare `'||'`
  `--item-regex`, which matched every line — see [[oolong-item-regex-leak-bug]]. The rebuild passes
  no `--item-regex` and takes the escaped default.
- **The shard is decontaminated** against the oolong eval rungs (query + first 200 chars signature).
  The rung files are deliberately *not* regenerated — they are the decontamination reference.
- **`query_position=after`**, per the standing directive for new runs. The eval therefore has to
  render `after` too; the July numbers were graded in `both` mode. The script says so in its own
  comment: numbers are not comparable to the July row.

So dense 0.232-flat-below-chunked was never worth diagnosing as a checkpoint problem — the whole row
is being replaced.

## Eval-launch reality check (updated 2026-08-13 ~06:00 UTC)

Neither owed eval is running — but the staging that has to precede them ran overnight and succeeded.

**Checkpoint relays weka → S3, all exit 0:**

| relay | Beaker id | result |
|---|---|---|
| `ctcms-fixedm-relay` | `01KZWKBFJKAJWYK1XPYXERF517` | outlier fixedM 4B **both arms** staged — 139 objects / 58 GB each, verified recursively |
| `ctcms-smallscale-relay` (+`-a682`) | `01KZWT1HETQMAA0YVR2NM73V9B`, `01KZWV3ACD4M2B4VCDQ63P8G4Q` | qdmatch_nq / reorder / contradiction / hotpotqa at 0.8b + 2b |

`relay_fixedM_to_s3.sh` documents why this goes back to Berkeley rather than evaluating in place on
Beaker: native and vLLM disagree by ~0.079 f1 (~2.8× SE) on this family, and the scale-K outlier
curve it has to sit beside was produced by vLLM. Same evaluator or the comparison isn't one.
(Its header says "model-only, ~9 GB each"; the sync actually moved 58 GB because `step2500` carries
the optimizer state — worth knowing for the Berkeley pull time, harmless otherwise.)

**Still missing: the qdmatch_nq 4B relay.** S3 `_transfer/` holds
`ctcms-qdmatch_nq-{full,cmix}-{08b,2b}` but no `-4b` prefix — and 4B is the row the suite needs.

- **outlier fix-k** — checkpoints now staged; **no eval job has ever been submitted**, at any scale.
- **qdmatch nq** — no *4B* eval submitted, and the 4B checkpoints are not yet on S3. The 0.8b/2b
  model-scale evals were submitted and all four failed (slurm 3443065, 3443066, 3443072, 3443073), together with five siblings in the same batch
  (contradiction-full-08b, reorder-full-08b, reorder-full/cmix-2b, hotpotqa-cmix-2b, 3443067–3443071).
- **Common cause: the HF-export step, not the eval.** `stage_and_eval_smallscale.sbatch` calls
  `export_olmo_to_hf.py --base-model /data/prasann/hf_models/Qwen3.5-{0.8B,2B}-Base`, which does
  `AutoConfig.from_pretrained(base).text_config`. The conda env's transformers is **4.57.6**, which
  has no `qwen3_5` → `KeyError: 'qwen3_5'` → `FATAL: export failed`, before a GPU is touched. The
  node-local overlay `/data/prasann/tf_qwen35_target` **is** transformers 5.10.2 and does ship
  `models/qwen3_5`, but the sbatch never puts it on `PYTHONPATH`. Fix that and the batch re-runs.
- Actually running right now: `ev-ctcms-contradiction-cmix-2b` (3442993), `rung-gate` (3442691),
  `gen-xabsence` (3443105).

## Ladder hygiene

`debug/ctc_modelscale/rung_token_audit.json` measures p50 realized tokens against the rung label.
Ratios far from 1.0 mean the x-axis is mislabeled for that row — see [[ctc-rung-labels-not-tokens]].
Worst offenders in this roster: absence 3.0–3.6×, niah 0.51×, strmatch 0.60×, qdmatch hpqa
0.74–0.82×. msmarco and rerank were the previous worst (0.53× at 16k) and have been **rebuilt
token-accurate and fully regraded**; the same treatment is what the four above need.

## Ordered to-do

1. Put `/data/prasann/tf_qwen35_target` on `PYTHONPATH` in `stage_and_eval_smallscale.sbatch`, then
   re-run the nine failed model-scale evals — their checkpoints are on S3 now.
2. Eval **outlier_fixedM at 4B** (both arms, 5 rungs) — staged and unblocked. 10 cells.
3. Relay **qdmatch_nq 4B** weka → S3, then eval both arms. 10 cells.
4. Build the fiqa eval ladder, then train + eval both arms.
5. Stage the qdmatch_fiqa ladder into `eval_rungs/`, train + eval both arms.
6. Re-grade niah dense 2k; grade strmatch dense 32k and absence dense 16k.
7. Build the absence 32k rung.
8. Re-run oolong end to end (both arms, `--query-position after`) once the VESSL pair lands.
9. Persist xabsence and obliq_twitter grade JSONs — right now those ten cells exist only as prose.
10. Rebuild absence / niah / strmatch / qdmatch_hpqa ladders token-accurate, as msmarco and rerank were.

Live artifact (numbers + error bars):
https://claude.ai/code/artifact/15921b3d-c744-4ce1-8b93-3acb7d92c193
