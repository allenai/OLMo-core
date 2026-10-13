---
experiment: ctc_data_fixes
title: CTC eval/data hygiene since mid-Aug — contradiction IID + FEVER rebuild, outlier IID (M-axis) rebuild, unsorted gold pairs, gold-sidecar index base
status: paused
updated: 2026-09-21
branch: prasann/landmark
launcher: debug/ctc_contra_iid_eval/gen_iid_eval.sbatch
records:
  - records/contradiction-train-eval-non-iid.md
  - records/contradiction-data-and-base-hygiene.md
  - records/paper-v2-todo-status.md
  - records/ctc-fast-suite-eval.md
results:
  - debug/ctc_contra_iid_eval/
  - debug/outlier_iid_rebuild/
  - debug/ctc_fast_suite/sort_gold_pairs.py
  - debug/ctc_modelscale/expand_ctc_rung.py
wandb_group:
next_step: Fix the stale "FEVER leak still OPEN for the CTC suite" lines in records/README.md and records/contradiction-data-and-base-hygiene.md §4b. Then re-score every pre-2026-09-21 contradiction number at ≥64k under canonicalised gold.
---

# ctc_data_fixes

## Question
Which shipped CTC train and eval files were silently wrong? What was fixed, and when? Which published
numbers does each defect still touch?

## Decisions
- 2026-07-29 (agent): found the FEVER/wiki filler leak. A mutable-dir glob (`contradiction_*_k3.jsonl`) pulled FEVER/wiki_mix fillers into PubMed contradiction evals. xlong 64k-2M was rebuilt PubMed-only. `_eval_bundle_eval500_v2_clean` became the default (0.00% leak).
- 2026-08-04/05 (agent): the CTC-suite contradiction ladder (92-99.6% FEVER/wiki fillers at 2k-32k) was rebuilt as `contradiction_clean` (0.00% leak) and every arm re-scored. The leak DEPRESSED dense 32k (0.335 → 0.559) rather than inflating it. Source: `records/paper-v2-todo-status.md` §1.
- 2026-08-11 (agent): train/eval mode mismatch. Train is `--mode realistic`, every eval was `--mode both`. `contradiction_iid` was regenerated with the training generator (Qwen2.5-7B-Instruct via vLLM, `--max-overlap 0.5`, PubMed filler pool).
- 2026-08-11 (agent): outlier M-axis mismatch. The train pool barely scaled M while the eval is scale-K. Added `--majority-mode articles` to `generate_wiki_outlier_data.py` and rebuilt 20k (`mooney:/data/prasann/outlier_iid/`). Data only, not retrained.
- 2026-08-12 (agent): the outlier xlong EVAL rungs had K pinned at 25 from n=451 to 14,419 (`self_nongold` expansion). They were rebuilt scale-K (`cb8e368b1`), and the xlong5 train pool was swapped scale-K with the original preserved (`weka_swap_outlier_train.sh`).
- 2026-08-13 (Prasann): the suite's contradiction row scores on `contradiction_iid` only, never `contradiction_clean`.
- 2026-09-08 (agent): `gold_chunks_from_gold_doc_indices` subtracted 1 for every task. Only contradiction is 1-indexed; nq/outlier/rerank are 0-based. Fixed with a per-task `base` on prasann/landmark (memory:gold-sidecar-index-base-bug).
- 2026-09-21 (agent): every `contradiction_iid` rung above 32k has ~50% unsorted gold pairs, which are unmatchable. The source was fixed in `expand_ctc_rung.py` (`32b5ae1cd`), and `sort_gold_pairs.py` repairs built files.
- 2026-09-21 (Prasann): fix the unsorted pairs in the GRADER, not the data. olmo-eval `5f165a65` canonicalises gold, so no HF re-upload is needed. A `UNSORTED_GOLD_SEEN` counter keeps the defect visible.
- 2026-09-22/23 (agent): setA's training data is calibrated from the eval's own rows (`build_ctc_sft.py --calibration`). The IID audit is 76/77 cells clean. textgroups is capped at 4k (shortcut).

## Current best numbers
Effect sizes of the fixes. These are not model results.

| arm | task / rung | metric | eval_size | ±SE | cot | source |
|---|---|---|---|---|---|---|
| `ctc-4b-contradiction-full`, eval both → iid | contradiction / n=762 (32k) | f1 0.559 → 0.946 | 500 | ±0.022 / ±0.010 | no-cot | records/contradiction-train-eval-non-iid.md (via memory) |
| same | contradiction / n=56/44 (2k) | f1 0.843 → 0.990 | 500 | ±0.016 / ±0.005 | no-cot | same |
| Qwen3.5-4B dense, contaminated → clean | contradiction / 32k | f1 0.335 → 0.559 | 500 | ±0.021 / ±0.022 | no-cot | records/paper-v2-todo-status.md §1 (via memory) |
| shipped `contradiction_iid` | 64k / 128k / 256k+ | unsorted gold pairs 744/1500, 764/1500, ~50% | — | — | — | records/ctc-fast-suite-eval.md |
| length-gen rung (cubbins) | `rung_65674.jsonl`, first 30 rows | 48/90 gold pairs unsorted | 30 rows sampled | — | — | checked 2026-10-12 (see `ctc_modelscale`) |
| outlier IID rebuild | K per rung | train 3/7/13/24 vs eval 3/7/13/25 | 20k train | — | — | records/contradiction-train-eval-non-iid.md |

## Next step
1. Correct the stale records. `records/README.md` (line 11) and `records/contradiction-data-and-base-hygiene.md` §4b (written 2026-07-29) still say "FEVER leak … still OPEN for the CTC suite ladder". The ladder was rebuilt 2026-08-04/05 and the row has scored on `contradiction_iid` since 2026-08-13.
2. Re-score contradiction numbers at ≥64k that were graded by an evaluator without gold canonicalisation. That covers the OLMo-core native and vLLM graders, and olmo-eval before `5f165a65`. The known instance is the CTC-LENGTHGEN contradiction cells (see `ctc_modelscale`).
3. Upstream the grader canonicalisation to the `ctc` package (vendored-copy rule).

## Open issues / traps
- **Outlier M-axis is still open for old checkpoints.** The IID rebuild shipped as data only. Every pre-setA outlier SFT (results-hub v2 5-task, CTC grid) trained on the scale-M-poor pool. setA's outlier is calibrated IID from eval rows, so new runs are fine. Models trained on the rebuilt pool need `max_new_tokens` ≥512, because the answer lists every majority topic.
- Older evaluators score unsorted contradiction gold SILENTLY at about half recall. Only the vendored spec's `_check_gold` refuses or canonicalises it. Validate any new pairwise-gold ladder through that guard. `qdmatch` pairs are ordered: never sort them.
- `dedup_rung_fillers.py`: expand-built rungs carry 0.07% (contradiction) to 0.5% (nq) duplicate fillers. contra_fever had 1.7-2.6% before the dedup pass.
- Never harvest fillers with a domain-agnostic glob. Pin a manifest and check `.manifest.json` or run `debug/xlong_5task/audit_contra_fever_leak.py` before quoting a long-rung contradiction number.
- Gold-index base is per task (memory:gold-doc-indices-per-task-base). The FLOP-scaling `kv17`/`kv33` nq/outlier arms trained with the wrong gold and are owed a rerun. That belongs to `flop_scaling`/`pooled_kv`.
- Related silent-wrong traps: memory:shard-src-positional-zip-bug, memory:oolong-item-regex-leak-bug, memory:xabsence-exact-vs-paraphrase-split, memory:ctc-rung-labels-not-tokens.

## Verified vs unverified
- **FEVER conflict resolved by dates.** `records/contradiction-data-and-base-hygiene.md` (mtime 2026-07-29) and the README line (added in `d65ff09df`, 2026-07-29, never updated since) both predate `records/paper-v2-todo-status.md` (2026-08-05). That record documents the `contradiction_clean` rebuild at 0.00% FEVER and the re-scored ladder. Memory `contra-fever-filler-leak` agrees, and itself flags §4b as stale. So the records are stale and the memory is right.
- Verified 2026-10-12: the record mtimes and git history above; the `contradiction_iid` generators (`gen_iid_eval.sbatch`, `build_v3_xlong_rungs.sbatch`) draw fillers from the PubMed generator's own pool; the unsorted-pair sample on the cubbins length-gen rung; `32b5ae1cd` exists; olmo-eval `5f165a65` is on the launcher branch.
- UNVERIFIED: `contradiction_iid` was not re-audited with `audit_contra_fever_leak.py`. It is FEVER-free by construction of its generator, not by a measured audit. The table's f1 numbers come from records and memory, not from re-reading grade JSONs.
- UNVERIFIED — confirm with Prasann: whether the public HF `contradiction_iid` xlong rungs should still be re-sorted and re-uploaded for collaborators using other graders. The grader fix covers olmo-eval only.
