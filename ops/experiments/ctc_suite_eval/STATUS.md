---
experiment: ctc_suite_eval
title: CTC-suite eval pipeline (infra) — olmo-eval launch_ctc_suite.py, 22 rows + 2 OOD, 2k-256k, Qwen3.5 checkpoints on Beaker
status: active
updated: 2026-10-13
branch: olmo-eval main (PR #400 merged) + PR #467 prasann/ctc-suite-launcher-on-main
launcher: scripts/ctc_suite/launch_ctc_suite.py
records:
  - records/ctc-fast-suite-eval.md
  - records/ctc-final-suite.md
  - records/ctc-suite-olmo-eval-handoff.md
results:
  - /accounts/projects/berkeleynlp/prasann/projects/OLMo-core/debug/ctc_fast_suite/results/bench_2k32k.json
  - /accounts/projects/berkeleynlp/prasann/projects/OLMo-core/debug/ctc_fast_suite/results/bench_xlong.json
  - /accounts/projects/berkeleynlp/prasann/projects/OLMo-core/debug/ctc_final_suite/suite_table.json
wandb_group:
next_step: Merge olmo-eval PR #467 (launcher + hashed prompt/marker/shard knobs + gold-pair canonicalisation, ported onto main), then launch from main; upload the two OOD ladders to enable --rows all.
---

# ctc_suite_eval

Note: `branch` and `launcher` refer to the **olmo-eval** repo
(`/accounts/projects/berkeleynlp/prasann/projects/olmo-eval`, tip `348bd0ff`, 2026-10-12). They are not
OLMo-core paths. The OLMo-core side is `prasann/landmark` (record, run-evals skill, chunked prefill
from `d28e458b2`).

## Question
How do we score any Qwen3.5 CTC checkpoint (dense, and later mask variants) on the frozen 22-row
suite plus the OOD rows, IID with the SFT prompt format? It should cover 2k-256k, finish in about an
hour on 8 single-GPU Beaker jobs, and use graders that are not silently wrong.

## Decisions
- 2026-08-12/13 (Prasann): the roster is frozen at 22 rows (`records/ctc-final-suite.md`). The contradiction row scores on `contradiction_iid` only, never on `contradiction_clean`, because `contradiction_clean` uses a different generator mode.
- 2026-08-14 (agent): the suite moves into olmo-eval (`prasann/ctc-suite`) with the spec vendored byte-faithful from the `ctc` package. Data comes from the public HF `PrasannSinghal/ctc-suite-eval`. eval_size is 500 through 128k and 125 at 256k and above.
- 2026-09-21 (Prasann): unsorted contradiction gold pairs are canonicalised in the grader (olmo-eval `5f165a65`) rather than by re-uploading the data.
- 2026-09-21 (agent): OOD rows `contra_fever` and `outlier_review` live in a separate `OOD_ROSTER` (`ctc:ood*`, olmo-eval `a150b926`). They are deliberately kept out of `ctc`/`ctc:figure`.
- 2026-09-21 (Prasann): scope is dense / full-attention arms through vLLM. Mask variants go through the native olmo_core provider at batch size 1.
- 2026-09-22/23 (agent): `launch_ctc_suite.py` is written (olmo-eval `e8e82967`…). It uses the native olmo_core provider with `CTC_SUITE_PROMPT_FORMAT=chat` (IID with setA) and `--doc-markers` for marker-trained checkpoints. It supersedes OLMo-core `launch_ctc_evals.py --backend olmo-eval`.
- 2026-09-28 (agent): keep the `Qwen/Qwen3.5-0.8B` tokenizer. The 4B chat template's generation prompt ends `<think>\n`, which is out of format for every CTC SFT checkpoint. A per-job `preflight.py` now refuses a bad template (`04894158`).
- 2026-10-12 (Prasann/agent): r256k defaults to YaRN x2, `max_model_len` 524288 and 131,072-token chunked prefill, in its own jobs (`8e63b917`). Without it, 43-100% of r256k rows were left-truncated. `--fit-window` is the alternative.

## Current best numbers
These are infra numbers: throughput and parity. Model results live on the per-experiment pages (e.g. `ctc_setA_sft`).

| arm | task / rung | metric | eval_size | ±SE | cot | source |
|---|---|---|---|---|---|---|
| stock Qwen3.5-4B, vLLM 0.25.1, 1xH100 | 7 tasks x 2k-32k | prefill fit 51,067 tok/s; decode 5,352 tok/s | 100/cell | rms resid 10.8 s | no-cot | debug/ctc_fast_suite/results/bench_2k32k.json (fit quoted from records/ctc-fast-suite-eval.md) |
| same | outlier 32k→64k→128k | prefill 48.3k→42.5k→33.9k tok/s | 100/cell ⚠ | n/a | no-cot | debug/ctc_fast_suite/results/bench_xlong.json (via record) |
| HF provider, hybridish 0.23B/0.29B (as stated in the record) | 8 tasks x 2k-32k | ~7.8k / ~5.0k prompt tok/s | 500 | n/a | no-cot | debug/hybridish_sft/sweep_ids.txt (via record) |
| olmo-eval native vs OLMo-core native driver | outlier @8k | f1 0.597 vs 0.587 | 500 | ±0.022 each | no-cot | memory olmo-eval-native-ctc-path (2026-09-23) |
| recommended policy "S+" | 200 per rung to 32k, 100 at 64k, 50 at 128k and 256k | 6.11 GPU-h, ~51 min on 8 shards | ⚠ <500 everywhere | ±0.032 @2k-32k, ±0.065 @256k | n/a | records/ctc-fast-suite-eval.md |

## Next step
Settle the branch question below (Prasann). Then push the vendored grader fixes upstream to the `ctc`
package (OLMo-core `prasann/ctc`, at `40a5c60d1` since 2026-10-01). Re-vendor once, so the launcher,
PR #400 and flopdrop_clean all grade with the same code.

## Open issues / traps
- **Branch conflict, flag for Prasann.** Gathered from the olmo-eval refs on 2026-10-12:
  - `prasann/ctc-suite-launcher` (`348bd0ff`, 2026-10-12) is the only branch holding `scripts/ctc_suite/launch_ctc_suite.py`. It contains `prasann/ctc-suite` (`a319dc86`, 09-08), `prasann/ctc-suite-grader-fixes` (`69729fad`, 09-09), `prasann/ctc-absence-low-ctc` (local, `37563d01`, 09-22) and `prasann/ctc-suite-train-iid` as ancestors. The run-evals skill, the record (2026-10-12) and memory all name it as current.
  - `origin/prasann/ctc-absence-low-ctc` = PR #400 (`8d7dec9b`, 2026-10-01; local worktree `olmo-eval-pr400`, branch `pr400-fixes`) carries 6 commits that are **NOT in the launcher branch**:
    - re-vendor at ctc `40a5c60d` (stopping `apply()` replays the decode loop);
    - obliq decode budget 64→512 (27/126 r32k rows were truncated);
    - a generation inside an unclosed `<think>` counts as a parse failure;
    - rerank skips rows with no CE positive;
    - the stored decode budget = the sent budget (previously every row stored 512 while sending 64);
    - HF dataset pinned by revision.
  - The launcher's vendored spec was last touched `de51be6c` (09-22). So launcher evals and PR-#400 evals can grade the same generations differently on obliq, rerank and reasoning outputs.
  - The hybridish sweep (2026-09-19) graded on `prasann/ctc-suite-grader-fixes` @`69729fa`.
  - OLMo-core `prasann/ctc-grader-fixes` (`c2b345fba`, 2026-09-09) is the OLMo-core-side grader fix ("three stop/parse rules"). Its worktree sits in a dead scratchpad (`2dba0efd…/scratchpad/ctc-wt`, marked prunable), and the commit is on no other branch.
  - None of the `prasann/ctc-*` olmo-eval branches is in `origin/main`.
- Jobs launched before olmo-eval `bdaa80ab` (2026-09-29) always exit 0. Read `=== rc=N` in the log (memory:qwen35-4b-chat-template-eval-mismatch).
- r256k cells from runs launched before `8e63b917` without explicit `--rope-yarn`/`--fit-window` are left-truncated. Do not use them. A one-shot forward above 262,144 tokens crashes (Triton illegal access); chunked prefill avoids it.
- Bad node jupiter-cs-aus-116 gave CUDA illegal-memory-access on long cells (2026-09-28). Steer around it with `--hostname`.
- The 128k MAX_LENGTH of 146,227 is too small (`fiqa:r128k` prompts are at least 146,228 tokens).
- 7 of 24 rows stop short of 64k. `absence`/`reorder` reach a real ceiling at 16k. `grouping`/`strmatch`/`textgroups`/`xabsence`/`scifact` are merely unbuilt past 32k (record, 2026-09-21).
- Item 4 of "Still open" in the record: staging new rung files (OOD ladders on `/scratch/users/prasann/ctc_ood_ladders/`) to weka or HF needs Prasann's HF token. Check the record again before launching OOD.
- Mask-variant arms have no vLLM path. The chunked-vLLM recipe is memory:chunked-eval-fast-config. Measure wall-clock per memory:eval-wallclock-sanity-check.

## Verified vs unverified
- Verified 2026-10-12 against git refs:
  - the olmo-eval branch tips, dates and ancestry above;
  - `launch_ctc_suite.py` exists only on the launcher branch;
  - the PR-#400-only commit list;
  - the launcher manifests under `olmo-eval/scripts/ctc_suite/launches/`. These record `olmo_eval_branch: prasann/ctc-suite-launcher` and a 2026-10-12 `setA-256k-default-yarn-SMOKE` with `yarn_rungs: r256k`.
- Verified: `debug/ctc_fast_suite/results/bench_{2k32k,xlong}.json` exist. The fitted rates were NOT recomputed. They are quoted from the record.
- RESOLVED 2026-10-13 (Prasann): PR #400 / olmo-eval `main` is canonical; the 25 launcher-only commits were ported onto main as PR #467 (`prasann/ctc-suite-launcher-on-main`). Superseded note follows: Evidence favours `prasann/ctc-suite-launcher` for launching, but PR #400 holds newer grader and hash fixes that the launcher lacks.
- UNVERIFIED: the "YaRN and `--fit-window` agree within noise on setA" claim. The result JSONs are on the weka results root and were not read.
- Conflict: the record (2026-09-21) gives the hybridish sweep as 0.23B/0.29B models. The hybridish branch README says the trained arms are 1.4B (4:1) and 2.1B (7:1). UNVERIFIED — confirm with Prasann which models the throughput numbers came from.
