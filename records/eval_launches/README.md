# Eval launch ledger

Gitignored working bookkeeping. One YAML per checkpoint; `pull-evals` flips jobs to `done`.

Last pull: **2026-08-13** — 126 rows pushed to results-hub (commit `d0a339d` on origin/main).

| date | run | step | passes | status after 2026-08-13 pull | ledger |
|---|---|---|---|---|---|
| 2026-08-12 | q35-4b-dense-xlong5-dolci25-256k | step560 | base (9 tasks) + xlong native/yarn2 (never launched) | **9 failed** from this ledger, but **full v3 ladder (all 9 tasks, base through 1M/yarn4) already in results-hub** — relaunched on H100 outside this ledger's tracking, verified 2026-08-31 | [2026-08-12_q35-4b-dense-xlong5-dolci25-256k_v3.yaml](2026-08-12_q35-4b-dense-xlong5-dolci25-256k_v3.yaml) |
| 2026-08-12 | q35-4b-fastcomplm-xlong5-dolci25-256k | step560 | base (9 tasks) + xlong native/yarn2 (never launched) | **9 failed** from this ledger, but **full v3 ladder already in results-hub** (correct tokenizer, base through 1M/yarn4) — verified 2026-08-31 | [2026-08-12_q35-4b-fastcomplm-xlong5-dolci25-256k_v3.yaml](2026-08-12_q35-4b-fastcomplm-xlong5-dolci25-256k_v3.yaml) |
| 2026-08-12 | q35-4b-fastcomplm-xlong5-dolci25-256k-ep1 | step634 | base (9 tasks) + xlong native/yarn2 (never launched) | **9 failed** from this ledger; results-hub has base + xlong native/yarn2 through 256k, but **512k/1M genuinely missing** (no yarn4 pass exists anywhere for this ep1 checkpoint) — verified 2026-08-31 | [2026-08-12_q35-4b-fastcomplm-xlong5-dolci25-256k-ep1_v3.yaml](2026-08-12_q35-4b-fastcomplm-xlong5-dolci25-256k-ep1_v3.yaml) |
| 2026-08-12 | q35-4b-dense-xlong5-qboth-dolci25-256k | step2240 | v2 base+xlong(native,yarn2) & v3 contra/outlier (21 jobs) | 16 failed (EVAL500 bug) / 5 superseded by the 08-13 relaunch | [2026-08-12_q35-4b-dense-xlong5-qboth-dolci25-256k_v2v3.yaml](2026-08-12_q35-4b-dense-xlong5-qboth-dolci25-256k_v2v3.yaml) |
| 2026-08-12 | q35-4b-fastlm-5task-dolci25-33344-datamatch | step10858 | v2 base+xlong(native,yarn2) & v3 contra/outlier (21 jobs) | 18 failed (EVAL500 bug / canceled) / 3 superseded | [2026-08-12_q35-4b-fastlm-5task-dolci25-33344-datamatch_v2v3.yaml](2026-08-12_q35-4b-fastlm-5task-dolci25-33344-datamatch_v2v3.yaml) |
| 2026-08-12 | q35-4b-fastlm-5task-dolci25-33344-tokenmatch | step10515 | v2 base+xlong(native,yarn2) & v3 contra/outlier (21 jobs) | 18 failed (EVAL500 bug / canceled) / 3 superseded | [2026-08-12_q35-4b-fastlm-5task-dolci25-33344-tokenmatch_v2v3.yaml](2026-08-12_q35-4b-fastlm-5task-dolci25-33344-tokenmatch_v2v3.yaml) |
| 2026-08-13 | q35-4b-dense-xlong5-qboth-dolci25-256k | step2240 | **RELAUNCH** v3-first, 19 jobs (supersedes the 2026-08-12 sweep: 48/63 died on the EVAL500 bug) | **19 done (54 rows)** | [2026-08-13_q35-4b-dense-xlong5-qboth-dolci25-256k_v3first.yaml](2026-08-13_q35-4b-dense-xlong5-qboth-dolci25-256k_v3first.yaml) |
| 2026-08-13 | q35-4b-fastlm-5task-dolci25-33344-datamatch | step10858 | **RELAUNCH** v3-first, 19 jobs (supersedes the 2026-08-12 sweep: 48/63 died on the EVAL500 bug) | **19/19 done (48 rows)** — the outlier@256k/512k straggler (finished exit=0 on 2026-08-14, sat un-pulled) pulled 2026-08-31, both f1=0.0 | [2026-08-13_q35-4b-fastlm-5task-dolci25-33344-datamatch_v3first.yaml](2026-08-13_q35-4b-fastlm-5task-dolci25-33344-datamatch_v3first.yaml) |
| 2026-08-13 | q35-4b-fastlm-5task-dolci25-33344-tokenmatch | step10515 | **RELAUNCH** v3-first, 19 jobs (supersedes the 2026-08-12 sweep: 48/63 died on the EVAL500 bug) | **19/19 done (48 rows)** — the outlier@256k/512k straggler (finished exit=0 on 2026-08-14, sat un-pulled) pulled 2026-08-31, both f1=0.0 | [2026-08-13_q35-4b-fastlm-5task-dolci25-33344-tokenmatch_v3first.yaml](2026-08-13_q35-4b-fastlm-5task-dolci25-33344-tokenmatch_v3first.yaml) |
| 2026-08-13 | hils-attention-7b | HF ckpt | **NEW: third-party model via `--backend hf`** — base ladder (9 tasks) × raw + chat (18 jobs) | 16 done; contra raw+chat **failed at 32k**, but 2k/8k/16k partial artifact already in results-hub (~0.000, flagged low-quality — parse degeneration) | [2026-08-13_hils-attention-7b.yaml](2026-08-13_hils-attention-7b.yaml) |
| 2026-08-13 | olmo3-1025-7b-base | HF ckpt | CONTROL for HiLS (its own base) — base ladder (9 tasks) × raw + chat (18 jobs) | 16 done; contra raw+chat **failed at 32k**, but 2k/8k/16k partial artifact already in results-hub (~0.000, flagged low-quality — parse degeneration) | [2026-08-13_olmo3-1025-7b-base.yaml](2026-08-13_olmo3-1025-7b-base.yaml) |
| 2026-08-14 | q35-4b-summ-causal/p50/decay-5task-packed | step1772 | summary-token smoke-2k, then v2 base after smoke gate | submitted | [causal](2026-08-14_q35-4b-summ-causal-5task-packed.yaml), [p50](2026-08-14_q35-4b-summ-p50-5task-packed.yaml), [decay](2026-08-14_q35-4b-summ-decay-5task-packed.yaml) |
| 2026-08-15 | q35-4b-summ-causal/p50/decay-5task-packed | step1772 | successful cache-fix gate + v2 base (all 9) + xlong native/YaRN2/4/8; ragged-prefill requeues on `bd1613f` | 87 original jobs + 37 targeted requeues; 3 gates succeeded | [causal](2026-08-15_q35-4b-summ-causal-5task-packed.yaml), [p50](2026-08-15_q35-4b-summ-p50-5task-packed.yaml), [decay](2026-08-15_q35-4b-summ-decay-5task-packed.yaml) |

| 2026-08-16 | q35-4b-summ-causal/p50/decay-5task-packed | step1772 | **RELAUNCH on the v3 ladder, serving the CAUSAL mask arm** — base (5 tasks) + xlong native (64k/128k) + YaRN2 (256k), 11 jobs/arm | nq/outlier/rerank/oolong **24/33 done** (8/arm); contra **9/33 failed** — v3 bundle's contra file was MISSING at eval time (empty result, harness bug); already relaunched+fixed and pulled into results-hub by another session (2026-08-25, slug `q35-summtok-{causal,decay,p50}-256K`, +OOD tasks fiqa/scifact/contra_fever/outlier_review) — cross-referenced 2026-08-31, values match exactly | [causal](2026-08-16_q35-4b-summ-causal-5task-packed_v3causal.yaml), [p50](2026-08-16_q35-4b-summ-p50-5task-packed_v3causal.yaml), [decay](2026-08-16_q35-4b-summ-decay-5task-packed_v3causal.yaml) |

## The 2026-08-15 summary-token sweep is void (2026-08-16)

All 113 jobs finished; only 27 produced a number, and **every one of those served the wrong mask**.
The mixture coin is drawn under `self.training`, so inference left `causal_example=None` and the
fully restricted mask applied to every example — including the arms that trained 100% causal
(`standard_mix_prob=1.0`; the decay curriculum ends at `mix_end_p=1.0`). Fixed in `6e3a4e309`:
`--summary-mask-mode`, defaulting to causal.

The other failure modes, all addressed in the relaunch:

- **60 jobs exited 0 with empty JSON** (`ladder keys present: []`) — the bundle has no 1M/2M files
  (every yarn4/yarn8 job) and no xlong rungs for rerank/oolong. Those passes are dropped, not rerun.
- **15 aborted on the `[maxlen]` guard** — the `<|summ|>` run lengthens prompts past the MAX_LENGTH
  table's dense margin (contra@16k built 59,487 tokens against a 40,960 cap). The runner now scales
  the cap for `VARIANT=summary`.
- **8 died in the block-sparse mask** at seq_len ~33–36k. Serving the causal arm skips that path.

**All 133 experiments deleted**, across all three arms, with their weka results removed — cleanup
jobs `01M05KD372H5SC2SWP11FZ5ZEM` (p50 + decay) and `01M05KNKK4Q1090HAA9JG2FCDT` (causal). The
ledgers are kept as the record of what was run and why it was void; every job reads `status:
deleted`. Nothing from this sweep reached results-hub.

## Open work after the 2026-08-13 pull

~~1. 10 jobs still running: datamatch/tokenmatch `xlong-yarn2`.~~ Resolved 2026-08-31 — pulled,
   both outlier@256k/512k = f1=0.0.
~~2. Relaunch the two fastcomplm 256k re-evals with `TOKENIZER=Qwen/Qwen3.5-0.8B`.~~ Resolved —
   already relaunched (by someone outside this ledger's tracking) and fully in results-hub as of
   2026-08-31.
~~3. Relaunch `q35-4b-dense-xlong5-dolci25-256k` step560 off saturn.~~ Resolved — same as above,
   full v3 ladder already in results-hub as of 2026-08-31.

**Remaining real gap**: `q35-4b-fastcomplm-xlong5-dolci25-256k-ep1` (step634) is missing 512k/1M
entirely — no yarn4 pass exists for it anywhere, unlike its step560 epoch-0 sibling which has full
coverage. Needs an actual launch if those rungs are wanted.
