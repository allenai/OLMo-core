# ds64 overnight loop — 2026-09-14 → 09-15

Autonomous iteration on soft-token training, run by the orchestrating session. Read this first after any
context compaction; append decisions to the log at the bottom. Campaign state: `records/ds64-handoff.md`
(§8 outlier header-real collapse, §9 shortcut-vs-leak arms, trap 10). Dashboard:
`python debug/ds64/collect_ds64.py && python debug/ds64/make_pareto_artifact.py` →
https://claude.ai/code/artifact/db31e725-e63b-48a7-b5a8-7af01b89038a

## Goals (user, 2026-09-14 evening)

1. **Maximally FLOP-optimal training** — Pareto dominance vs dense at matched `flop_meter/actual_pflops`.
2. **Trainable at much longer max sequence lengths.**
3. Do not over-invest in ideas that compromise either of those *too much*.
4. **Outlier: never accept parity — keep iterating** (user, 2026-09-15 ~10:45). Iterate through MANY ideas quickly.
5. **PRIMARY TARGET (user, 09-15 11:10, clarified): FROZEN-DENSE-MODEL eval parity for outlier** — a soft construction
   under which the frozen dense checkpoint's answer CE ≈ FULL (the result contradiction/oolong already have via header-real).
   Slot vectors are exhausted → hunt REAL-TOKEN subsets per doc (first-k, IDF top-k, first sentence; O(k) tokens/doc).
   Secondary: a TRAINED outlier checkpoint with eval-time CE parity** — answer CE under
   its own soft construction ≈ answer CE under full real text (same checkpoint) at 8k and 32k. Ladder f1 is secondary
   until this exists. Every fast8k arm reports CE_full / CE_soft / ΔCE first.
6. **Standing check for every idea:** eval-time CE parity of the TRAINED soft arm — its own soft construction vs FULL
   real text on the eval rungs (harness flag being added to debug/pooled_kv/outlier_probe/, see Log 10:45). Eval-side
   probes on a frozen dense model screen the *representation* only; the *readout* must be learned on outlier.

Hard reading of (1)+(2) for candidate recipes:
- steady-state recipe must be **low keep (≤ 1/6)**; keep 0.5 is a 2× arm and only a diagnostic.
- nothing dense-cost in steady state (mix curricula = diagnostics of the shortcut basin, not the recipe;
  a *short* dense-ish warm-up whose FLOPs are charged is acceptable if the low-keep phase then dominates).
- nothing that adds per-token or per-layer-per-token cost that scales with length; per-doc header
  (~6 tok/doc) is fine.
- layer-axis slot schedules: closed (records/layer-soft-probe.md, no headroom with header-real).
- compressive-landmark slots: closed (records/landmark-slot-probe.md, NO-GO).

## Decision rules (early signals)

- **CE-floor kill**: outlier train CE parked at the format floor (~0.48 at 16M/32M, k/n guess policy) by
  step ~10 of 28 → the arm is guessing; kill, do not wait for eval. Content-grounded runs (kvgb50) are
  well below by step 8.
- **Matched-FLOP delta**: at each eval, interpolate dense (log-FLOP) at the arm's actual_pflops; an arm
  below dense at its first budget and not closing at the second is dropped.
- **Complete rungs only** (`rungs_complete`), outlier scored on 4 rungs (64k broken), `steps ≥ 2`.
- Any number from <500 rows is flagged inline with its SE.

## Queue (in priority order; each gated on the signal above it)

A. **Outlier eval-side probe** (agent running): can a dense model read a pooled outlier from slot+header?
   Verdict A (readable) → the collapse is optimization → pursue B/C. Verdict B → slot too weak; the only
   allowed richer slot is 2 slots/doc (first/second-half means) — cheap, length-neutral; test eval-side first.
B. **Shortcut-breaking arms, full ladder** (running under orchestrator pid 3740274, 4 GPUs each):
   `xh2k50` (header + keep .5) 16M/32M; `xh2warm17` (header + keep 1/6, warm from kvgb50-16M) +16M/+32M;
   `xh2mix17` (header + keep 1/6, p_full 0.5→0 over first half) 16M/32M; `kvgbmix2` (no header, keep .5,
   same mix — the arm kvgbmix should have been) 16M/32M. Read train CE vs floor first (wandb group
   ds64-q35-4b), then 5-rung eval. FLOP accounting: warm arms = 190.3 PF + increment.
C. **fast2k harness** (agent building, `debug/ds64_fast2k/`): 1-GPU 2k-only outlier sweeps, <1 h per arm.
   Once validated (separates kvgb50 from xhdr17), use it for: keep-prob anneal 0.5→1/12 within gold-blind
   (cheaper than p_full rows), no-header→header two-phase, header + keep 1/12, 2-slots/doc.
D. Winner (if any) → confirm on full ladder 16M/32M/64M, then update handoff §1 outlier verdict.
E. If time remains: contradiction keep 1/36 + neighbour-runs (11× eval-side) as a trained arm — the
   most FLOP-aggressive recipe not yet trained at 64M; oolong `ohdr08`/`ohdr00` (headers only).

## Outlier idea queue (fast8k first, then ladder) — 2026-09-15
Screen at 8k (the failure rung); each must be keep-0-ish steady state, free slot (cent_cmean), header real, nothing
length-scaling:
- [running] `gold_pooled_random` p∈{1/6,1/3,1/2} (cpi17/33/50): gold ALWAYS pooled, random non-gold real → real-text
  exposure with zero copy credit. Two-phase cc00 (85%) → keep ½ (15%). cc00 + dense anchors at 8k.
- [pending diag] trained-cc00 eval-time probe: soft-construction vs full at 8k/16k → shift vs readout.
- next: reverse two-phase (keep ½ first, cc00 after); cc00 at 2 epochs (budget axis); slot at doc START vs centre for
  keep 0 (position prior on ids); `gold_pooled_random` + curriculum p 1/2→1/6 (anneal real exposure down, FLOPs charged);
  header = id only (drop `\n\nDocument [`) to shrink T2 at keep 0; 2 slots/doc only if the 8k readout is the limit.
- if the diag says the trained model reads slots well but real text at eval breaks it: consider reporting soft-construction
  INFERENCE as a separate (cheaper) product line — needs the user's call, since the campaign premise is full-attention eval.

## Log

- 19:22 xh2k50/xh2warm17 launched (agent). 19:48 mix fix committed 7ce879b40; xh2mix17/kvgbmix2 launched;
  orchestrator restarted pid 3740274 (trap 10: ARM_* frozen at startup → restart needed for new arms).
- Landmark-slot: NO-GO (b21d278b4). Layer probe: no headroom with header-real baseline; headless pass
  quarantined. xhdr*: k/n floor, cancelled, DEAD banner.
- 21:05 CE-floor read (16M): xh2k50 0.37 = content-grounded (header does NOT collapse at keep .5);
  xh2warm17 0.48 from step 1 = floor → warm start does not hold at keep 1/6. Points to slot-unreadable (B),
  not a basin. xh2warm17-32M had already finished 56/56 when cancelled (orchestrator marked DONE, eval 01M2HKDASESK80DVVVDT4NHDGA running — free data, left alone). Awaiting eval-side probe pooled-gold split.
- 22:10 Eval-side probe VERDICT B (records/outlier-slot-probe.md): pooled-gold recall = k/n floor in every
  construction; slot-swap undetectable; kvgb50 parity = real-gold rows only. Killed xh2mix17 ×2, kvgbmix2-32M
  (latched FAILED). Kept kvgbmix2-16M (single curriculum datapoint) and xh2k50-32M (header helps readout of
  REAL docs: 0.81 vs 0.73). Queue B/C (basin fixes) CLOSED. Next: richer-slot eval-side probe — content-token
  mean, centred/whitened mean, G=2/4 segment means, shallow k-layer block-local encoder mean (k=2,4).
  Side findings to propagate: answer-CE parity ≠ correct answers on outlier (check gen metric); latent RoPE
  OOB when GENERATING on compacted rows (rope.py:563-571). Jupiter did not schedule a 1-GPU urgent job in 1.5h;
  ceres+saturn did in 15 min → use "ai2/ceres-cirrascale,ai2/saturn-cirrascale" for small probes.
- 21:25 TRAP 11: the orchestrator holds state IN MEMORY and rewrites the json every cycle — editing the file
  while it runs is overwritten, and a cancelled job reads as rc=1 → relaunch (xh2mix17-16M was relaunched as
  01M2HMNVNVCEMSVSF75QQVGR2H and had to be cancelled again). Correct procedure: kill orchestrator BY PID →
  edit state (FAILED, retries 99) → cancel every live job of the run's CURRENT experiment → restart with
  DS64_GEN=3 DS64_NGPU=4 DS64_CLUSTER=jupiter,saturn. Restarted pid 3853397. Also: `pgrep -af orchestrate`
  matches the calling shell — read the pid from the python process line only.
- 22:40 fast2k VALIDATED (debug/ds64_fast2k/, handoff §10, wandb f2k-q35-4b): reproduces ds64 2k at 1/14 compute
  (kvgb50 0.882 / xhdr17 0.234=3/14; CE floor 0.39 separates by end of a 15-min train). xhdr50 0.895 = header
  harmless at keep .5; xhdr17-warm collapses → not a basin (consistent with verdict B). At 2k dense ≈ ceiling
  → fast2k is a KILL FILTER, not a matched-FLOP contest; 2 GPUs/arm; include ceres in cluster list.
  Use next: screen training arms of any slot the richer-slot probe lifts off the floor (keep 1/6 + header).
- 22:31 Queue E: oolong `ohdr08` (headers-only, keep 1/12, gold-blind) launched at 16M/32M ONLY, pinned via
  a new `ARM_BUDGETS["ohdr08"]` entry in `launch_ds64.py` (default oolong grid is 16M-128M; without the pin
  it would also fire 64M/128M). `ARM_MICRO["ohdr08"]` set to 2, not the pre-existing 4 -- conservative choice
  since ohdr08 has never run before and `ohdr17` (also micro 4) died to the 900s NCCL watchdog on oolong's
  worst-case padding (handoff §2 item 5); revisit once one ohdr08 run finishes clean. Confirmed no ohdr08 rows
  pre-existed in results.csv or orchestrator state before this. Dry-run resolved command (16M):
  `--task oolong --variant softtoken --global-batch 128 --micro-batch-instances 2 --num-gpus 4 --seq-len 65536
  --extra-args --st-gold-blind --st-keep-prob 0.0833 --st-header-stop-id 25 --st-header-stop-count 3
  --attn-backend flash_2` (2*4=8 divides 128; no --max-tokens). Added `ohdr08` to `soft_arms.json`'s oolong
  list; simulated the restart's A2 section first (script-level, not the orchestrator's own dry path) against
  the live state -- it showed exactly these 2 new launches and confirmed every `xh2mix17`/`kvgbmix2`/`xhdr*`
  entry stays latched FAILED/DONE, none resurrected. Killed the orchestrator by pid (3853397, read from the
  python process line of `pgrep -af`, not `pkill -f`), applied the edits, restarted
  (`DS64_GEN=3 DS64_NGPU=4 DS64_CLUSTER=jupiter,saturn`, new pid 3930595). Log confirms `runs` went 158→160
  (exactly +2) and the two launches:
  `ds64-oolong-ohdr08-b128f3-u16M` -> `01M2HRS5HNCKHT71FRC42XHW6S`,
  `ds64-oolong-ohdr08-b128f3-u32M` -> `01M2HRV1Z2NREG2SAM1M5H2HG0` -- both verified priority `urgent`,
  workspace `ai2/flex2`, cluster jupiter+saturn via `beaker experiment get --format json`, and logged in
  `debug/ds64/LAUNCH_LEDGER.tsv`.
- 09-15 00:40 tick: orchestrator alive (pid 3930595), 103/160 runs done, 0 pending, 7 evals pending (xh2k50 16M/32M,
  xh2warm17 16M/32M, kvgbmix2 16M, ohdr08 16M/32M). No new scored rows → dashboard not republished. Richer-slot
  probe agent hit the session rate limit (reset 00:10) with tables in hand → resumed to write
  records/outlier-richer-slot-probe.md. Oolong ohdr08 16M/32M launched 22:31 (01M2HRS5HNCKHT71FRC42XHW6S,
  01M2HRV1Z2NREG2SAM1M5H2HG0); both trained, evals running.
- 09-15 01:05 Richer-slot probe (records/outlier-richer-slot-probe.md): no eval-side construction is readable by the
  frozen dense model (all at k/n, swaps identical; enc2/enc4 +6–11% FLOPs buy nothing → rejected). BUT an oracle
  cosine readout shows the signal IS in the slot once common-word mass is removed: plain mean 0.39/0.075 (2k/8k,
  floors 0.225/0.053) → content-only+centred mean `cent_cmean` 0.73/0.24, FREE (no tokens, no per-layer cost).
  detach_soft_kv=True ⇒ projector never learns ⇒ fix must be in slot CONSTRUCTION. Trap: header-free docs are
  non-contiguous (marker+id, FREE header, body) — index within-doc ordinals, not pos−first[doc].
  → Dispatched: implement --st-slot-mode {mean,cmean,cent_cmean} in trainer; fast2k screen cc17/cm17/cc00/cc50 @4M
  vs floor 0.39 (kvgb50 0.264, xhdr17 0.41). If cc17 leaves the floor → full ladder cc17/cc00 16M/32M/64M (queue D).
- 09-15 00:55 tick: CORRECTION — the orchestrator was NOT stuck (status lines every 5 min through 00:53; I misread the
  clock). Restart was unnecessary but harmless: now pid 4078703 with DS64_CLUSTER=jupiter,saturn,CERES and python -u.
  The 7 pending evals had genuinely sat QUEUED 1–4 h on jupiter/saturn → cancelled so the orchestrator relaunches them
  on the 3-cluster list (ceres has storage:weka; fast2k evals scheduled there in ~25 min). No new scored rows; no
  kills; cent_cmean implementation + fast2k screen still running. Verify next tick that all 7 evals relaunched.
- 09-15 09:25 **Content-only slot SHIPPED + screened on fast2k (5 arms, 4.1 GPU-h, wandb f2k-q35-4b).**
  `--st-slot-mode {mean,cmean,cent_cmean}` now runs in the trainer (commit 3f407c63a, default `mean`
  = bit-identical; CPU test `src/test/nn/pooled_soft_token_slot_mode_test.py`). The stop set is built
  once at startup from token frequencies over the head of the training shard (measured: **510 ids
  from 1.51M tokens** = top-100 + markers + punctuation, sample `' the' ',' '.' ' of' '1' ' ' ' and'
  ... 'Document' ']:' <markers>`); the centroid is the **ROW** centroid (online, no extra pass,
  corpus-centroid alternative noted in the docstring). Two traps: `token_ids_part_*.npy` is a **raw
  headerless** array, not a `.npy` (`np.load` kills the run at model build -- 3 arms lost, fixed
  39f8d11ff); and the fp32 content-mean pass adds ~2x the row's bf16 embedding as a transient, which
  at 65536 x micro 2 is ~2.7 GB -- watch it when promoting.

  | arm | slot | keep | PF | xdense | CE@10 | CE fin | f1 2k (SE .013-.019) |
  |---|---|---|---|---|---|---|---|
  | dense 4M | -- | -- | 98.4 | 1.00 | 0.128 | 0.020 | 0.987 |
  | kvgb50 | mean | 1/2, no hdr | 56.3 | 0.57 | 0.377 | 0.264 | 0.882 |
  | xhdr50 | mean | 1/2 | 57.4 | 0.58 | 0.387 | 0.275 | 0.895 |
  | **cc50** | cent_cmean | 1/2 | 57.4 | 0.58 | 0.351 | **0.236** | **0.906** |
  | xhdr17 | mean | 1/6 | 27.3 | 0.28 | 0.435 | 0.408 | 0.234 FLOOR |
  | **cc17** | cent_cmean | 1/6 | 27.3 | 0.28 | 0.449 | 0.408 | **0.237** FLOOR |
  | **cm17** | cmean | 1/6 | 27.3 | 0.28 | 0.450 | 0.409 | **0.235** FLOOR |
  | **xhdr00** (control) | mean | 0 | 9.3 | 0.09 | 0.436 | 0.408 | 0.239 FLOOR |
  | **cc00** | cent_cmean | 0 | 9.3 | 0.09 | 0.429 | **0.348** | **0.482** |

  CE floor 0.390, uniform-guess f1 3/14 = 0.214, eval_size 500 each.

  **Verdict: NO at keep 1/6 -- cc17 0.237 / cm17 0.235 vs xhdr17 0.234, CE 0.408 in all three, i.e.
  the content-only slot changes nothing there. YES at keep 0, decisively: cc00 0.482 vs the
  same-FLOP xhdr00 control 0.239 (+0.243, >10 SE) with CE 0.348 vs 0.408, at 0.095x dense -- the
  cheapest arm in the whole screen.** The non-monotonicity is the finding: with 1/6 of bodies real
  the id-guess policy still gets partial credit and SGD takes it; with NO real body the guess is
  exactly 3/14 and the only gradient left comes from the slots, which a *readable* slot can supply.
  That is the first trained evidence that the richer-slot probe's "readout, not representation"
  reading is right -- the frozen-model readout failed on every candidate, but a trained reader finds
  `cent_cmean`. It also means the previous xhdr00 "hard leak" reading is wrong as stated: the leak is
  survivable once the slot carries topic. I did NOT run xhdr00 vs cc00 beyond 2k, and **fast2k
  cannot see length generalisation** (kv33: 0.890 at 2k, 0.018 at 32k) -- cc00 needs a ladder before
  any of this is a campaign result. cc50 0.906 vs xhdr50 0.895 is inside noise.
  Registered `cc00` (recommended) and `cc17` (not recommended, floor) in `debug/ds64/launch_ds64.py`
  ARM_EXTRA / ARM_MICRO 2 / ARM_BUDGETS 16M-64M, **working tree only, not committed and not
  launched**; `soft_arms.json` and the orchestrator untouched as asked.
  Jobs: train cc17 `01M2J1CK2TG2VYA5GX64354EZC`, cm17 `01M2J1DFSAT35R0AGYRKEGJBKB`, cc00
  `01M2J1EBYVKRREN264H3C15TZX`, cc50 `01M2J1F9DCYXJEB0FDPVRWPB6W`, xhdr00 `01M2J3J7K49M1WKDKFCP9BYTTC`;
  evals `01M2J3DDSRSVEWQXFY7KG0NDY5` / `01M2J3E9XAYHKSFNAFRCJ22WXD` / `01M2J2S43E50NYS7C2MW2Z21MT` /
  `01M2J46CVDEED2QK9DTHVHZSEF` / `01M2J543PV8NNRNJ3Z05SS2A8Y`.
- 09-15 02:30 cent_cmean SHIPPED (--st-slot-mode {mean,cmean,cent_cmean}; commits 3f407c63a..11f345b90; row centroid,
  stop-set = top-100 + punct + markers from the shard). fast2k screen (4M, 500 rows): cc17/cm17 stay at floor
  (0.237/0.235 — with 1/6 bodies real, id-guessing still earns partial credit); **cc00 0.482 vs xhdr00 control 0.239
  at 0.095× dense** (CE 0.348 vs 0.408) — first trained evidence the slot is readable once no real body remains;
  cc50 0.906 vs xhdr50 0.895 (noise). → PROMOTED cc00 to the full ladder 16M/32M/64M (orchestrator pid 4173299,
  cc00-u16M = 01M2J69GZ5E6JJBPH5RCHP21JP). Watch: fp32 content-mean pass adds ~2.7 GB transient at 65536×micro2.
  Length generalisation is the open question (kv33: 0.89@2k → 0.02@32k).
- 09-15 02:55 tick: orchestrator alive (4173552); runs 104/163 (cc00-32M/64M training), 8 evals pending (7 relaunched
  at 01:00 on 3 clusters + cc00-16M). No new scored rows → dashboard unchanged, not republished. No kills. cc00-16M
  ladder CE 1.04→0.46 @28 steps (floor ~0.48), 28 PF.
- 09-15 03:30 cc00 ladder all trained: 16M 28 PF CE→0.46; 32M 56 PF CE→0.42 (56 steps); 64M see below. All 3 evals
  queued (Beaker saturated; 9 evals pending in total). Verdict awaits the 5-rung evals.
- 09-15 03:35 cc00-64M: 114 PF, CE 1.03→0.50 (16)→0.36–0.39 (113 steps) — keeps descending with budget. Added 128M
  to cc00 ARM_BUDGETS and restarted orchestrator (3 clusters, -u) to launch cc00-u128M (~228 PF, still below dense-16M
  FLOPs → a matched Pareto point). 9 evals still queued.
- 09-15 05:05 TRAP 12 (root cause of the 4 h eval stall): DS64_EVAL_CLUSTER defaults to jupiter-only (orchestrate_ds64.py:42);
  every restart tonight set only DS64_CLUSTER, so all evals were pinned to jupiter's backfill queue while saturn had
  70 free GPUs. Restarted with DS64_EVAL_CLUSTER=jupiter,saturn,ceres; cancelled all queued evals → auto-relaunch.
  cc00-128M trained (01M2J9WCX1D2QCRD30D8R2GJQ7). ALWAYS pass DS64_EVAL_CLUSTER on restart.
- 09-15 05:10 cc00-128M: 233 PF, CE 1.04→0.46 (32)→0.28 (128)→0.20–0.23 (226 steps) — far below the 0.48 floor; the
  slot readout is being learned at scale. Matched comparison = dense-16M (~400 PF, mean 0.322). If its eval beats
  that → first outlier Pareto dominance. Next if so: cc00 at 2 epochs of the 128M shard (~465 PF) vs dense-32M.
  Eval relaunches pending the orchestrator's next cycle (verify constraints = 3 clusters).
- 09-15 06:50 EVALS LANDED (64k rung still pending on some). Oolong ohdr08: 16M 125 PF 0.851/0.633/0.569/0.508 vs dense-16M
  419 PF 0.853/0.579/0.528/0.528 → parity-or-better at ~3.3× (ohdr17 was 146 PF); 32M 245 PF ≈ dense-32M 811 PF. New
  oolong frontier point. Outlier: dense is 391 PF @8M (0.284), 780 PF @16M (0.322). cc00-128M 233 PF: 2k 0.80 (= dense-4M
  at ~195 PF) but 8k 0.17 vs 0.30–0.39 → 5-rung ≈0.21 vs dense≈0.26 at matched FLOPs: readable slot fixed 2k, but a model
  that never sees a real body does not generalise to 56+ real docs. xh2k50-32M 463 PF ≈ parity; kvgbmix2-16M 214 PF 0.254
  ≈ on the dense curve (+0.024 over kvgb50 for +12% FLOPs); xh2warm17 ≤ kvgb50. Everything outlier sits ON the dense
  curve at best. → Launched cc03 (cent_cmean, header, keep 1/36: real-body exposure with negligible guess credit) at
  16M/32M/64M/128M and cc08 (keep 1/12) at 16M/32M. Orchestrator restarted with DS64_EVAL_CLUSTER set.
- 09-15 06:55 tick: orchestrator alive (261497). cc00-128M complete 5 rungs: mean 0.207 (233 PF) vs dense ≈0.26 at
  matched FLOPs — below the curve, 2k parity only. xh2k50-32M 5 rungs 0.282 @463 PF = kvgb50-32M 0.282 @374 PF → header
  adds nothing at keep .5 and costs 24% FLOPs; xh2 family closed. ohdr08 64k rung still pending (2 evals). 6 training
  runs pending (cc03 ×4, cc08 ×2). No kills. Dashboard republished (84 scored).
- 09-15 07:00 oolong ohdr08 complete (5 rungs): 16M 0.614 @125 PF beats dense-16M 0.603 @419 PF → NEW DOMINANCE 3.3×
  (was ohdr17 2.9×); 32M 0.643 @245 PF = dense-32M 0.644 @811 PF (parity at 3.3×). Dashboard headline updated.
- 09-15 07:00-07:09 Lower-keep readable-slot arms added for oolong/nq (does `cent_cmean`
  let these two tasks go BELOW their current best keep without losing accuracy?), 16M/32M only,
  micro 2, pinned via new `ARM_BUDGETS`/`ARM_MICRO` entries in `launch_ds64.py` (working tree,
  not committed -- same as cc00/cc03/cc08 -- launch_ds64.py is invoked as a fresh subprocess per
  launch so this is not gated by trap 10):
    - oolong `occ08` = `ohdr08`'s exact flags (`--st-gold-blind --st-keep-prob 0.0833
      --st-header-stop-id 25 --st-header-stop-count 3`) + `--st-slot-mode cent_cmean
      --st-slot-tokenizer {WEKA}/hf_tokenizers/Qwen3.5-0.8B-Base` -- same-FLOP twin of ohdr08
      (16M 0.614@125PF / 32M 0.643@245PF, this log 07:00).
    - oolong `occ00` = same, `--st-keep-prob 0.0` (headers only, every line pooled) -- maximal
      compression, the oolong analogue of outlier's cc00 win.
    - nq `nqcc17` = `kv17`'s exact flags (`--st-keep-frac 0.1667 --st-keep-mode gold_plus_random`,
      no header -- nq renders no header) + the same slot flags. kv17 plain-slot control CONFIRMED
      in results.csv (already run, no relaunch needed): 16M 0.7068@115PF, 32M 0.7828@231PF,
      64M 0.8276@443PF, 128M 0.8384@916PF vs dense 0.8436@683/0.8652@1387/0.9068@2777/0.9152@5556.
    - nq `nqcc08` = kv08's keep ratio (1/12, never run for nq before) + same slot flags.
  Registered all four in `soft_arms.json` (oolong: +occ08,+occ00; nq: +nqcc17,+nqcc08). Dry-run
  (corpus-reasoning-olmo env) confirmed resolved `--extra-args` for occ08-16M and nqcc17-16M both
  carry `--st-slot-mode cent_cmean`, micro 2 x 8 GPU = 16 divides global-batch 128, no --max-tokens.
  **TRAP 10 hit live**: `orchestrate_ds64.py` imports `budgets_for`/`ARM_BUDGETS` ONCE at process
  start (`from launch_ds64 import ...`), so the pre-edit orchestrator (pid 261497, up since 06:43)
  had occ08 absent from its in-memory `ARM_BUDGETS` and fell back to the full default grid --
  between editing the file and killing the process it launched occ08 at ALL FOUR budgets
  (16M/32M/64M/128M: 01M2JNZRAK60DE4QSQ6FGGQV6Q / 01M2JP0M3V6C1Q80GKWVDEXQT8 /
  01M2JP1KASV0RG200M5PB0M4HP / 01M2JP2FRVG05XK2GRCKH76CQ1), violating the 16M/32M-only budget.
  Killed pid 261497 immediately (by pid, not pkill), cancelled the two stray jobs
  (01M2JP1K.../01M2JP2F..., both confirmed `canceled` via `beaker job cancel`), and latched their
  state-file entries `FAILED` / `retries=99` so a restart would not resurrect them. Simulated the
  restart (`DS64_GEN=3 DS64_NGPU=4`, fresh subprocess import so `ARM_BUDGETS` is current) against
  the live state: exactly 6 new launches (occ00 x2, nqcc17 x2, nqcc08 x2) -- occ08-16M/32M already
  running, occ08-64M/128M correctly absorbed as latched-FAILED, not relaunched. Restarted with BOTH
  cluster vars (`CL="ai2/jupiter-cirrascale-2,ai2/saturn-cirrascale,ai2/ceres-cirrascale";
  DS64_GEN=3 DS64_NGPU=4 DS64_CLUSTER="$CL" DS64_EVAL_CLUSTER="$CL"`), new pid 286326. Log confirmed
  the exact 6 predicted launches by 07:08:49
  (occ00: 01M2JP7HG1JY2AX50KMEHCHH7D/01M2JP8DMQ0PJSDP81CVZS8M3K; nqcc17:
  01M2JP9C32K5XCPS3QRMSKNQ4J/01M2JPA8SD7MNN6N2F79977TWP; nqcc08:
  01M2JPBB923T45QY8KYHKJM588/01M2JPC72S2SDEGNCP6MFSS94B) and status line `runs 107/180 done, 14
  pending` (174 after the stray occ08 pair + 6 = 180, matches). 8 arms training total: occ08 x2,
  occ00 x2, nqcc17 x2, nqcc08 x2 -- all logged in `debug/ds64/LAUNCH_LEDGER.tsv` by the launcher.
  Read next Log entry (or `results/ds64/results.csv`) for evals once these land.
- 09-15 08:25 cc03 (keep 1/36) and cc08 (keep 1/12) 2k = 0.23–0.26 at 16M AND 32M (chance 0.22) while cc00-32M = 0.53:
  any real bodies → id-guessing with partial credit (their CE descent is real-gold rows, not slot reading). Keep 0 is
  the only content-forcing regime; keep 0 cannot generalise to 8k+. Killed cc03-128M (latched FAILED); cc03-64M eval
  left to finish as confirmation. OUTLIER VERDICT (pending confirmation): on the dense curve at best. Readable slot
  solved 2k; the remaining gap is eval-time distribution shift (no real bodies ever seen).
  Early rungs: nqcc17-16M 2k 0.97; occ08-16M 2k 0.84 (ohdr08 0.85) — waiting for full rungs.
- 09-15 08:55 tick: orchestrator alive (369085); runs 118/180, 2 training (occ00/occ08-32M), 11 evals pending. PARTIAL
  rungs (2k/8k/16k): nq nqcc17-16M .97/.87/.77 vs kv17-16M .97/.85/.74 (same ~115 PF) — readable slot +0.02–0.03;
  nqcc08-16M .95/.79/.70 (keep 1/12). oolong occ00-16M (keep 0, every body pooled) .83/.63/.57 = ohdr08-16M .85/.63/.57
  ≥ dense-16M .85/.58/.53 — if 32k/64k hold, a new maximal-compression oolong point. cc03-64M 2k .65 / 8k .09 → no
  long-rung rescue, verdict unchanged. No kills; no launches; 86 scored rows unchanged → dashboard not republished.
- 09-15 10:10 nqcc17-16M complete: 0.751 @80 PF vs kv17-16M 0.707 @115 PF (+0.044, fewer PF?!) — FLOP-meter discrepancy
  between old and new soft arms at the same keep needs checking (pad accounting?) before any Pareto claim. Oolong
  occ00 (keep 0) 4 rungs = ohdr08 at 16M and 32M. Awaiting 64k rungs.
- 09-15 10:20 oolong occ00-16M complete: 0.605 @116 PF (headers real, EVERY body pooled, cent_cmean slot) beats dense-16M
  0.603 @419 PF → NEW DOMINANCE 3.6× (per-rung: +0.05 at 8k, +0.04 at 16k, −0.03 at 32k/64k). occ08-16M 0.610 @125 PF.
  The readable slot lets oolong go to keep 0 without loss; ohdr08 (plain slot) also holds at 1/12. Dashboard republished.
- 09-15 10:30 FLOP audit (debug/ds64/flop_audit/README.md): meter is correct and byte-identical across commits; the
  kv17 (115 PF) vs nqcc17 (80 PF) gap is a LAUNCH-CONFIG confound — kv17 ran 8 GPUs/micro 4 (09-09), nqcc17 4 GPUs/micro 2;
  microbatch_sort_pad_id sorts per-rank and compact_pooled_rows re-pads each micro-batch to its longest row, so the
  finer batching genuinely does less padding compute. ohdr08 vs occ08 (same config) are bit-identical PF → slot_mode has
  zero FLOP effect. Only kv17 is out of family; dense anchors unaffected. Rule: method-ablation pairs at identical
  gpus/micro; run flop_audit/recompute_flops.py before trusting a FLOP-matched soft-vs-soft comparison. Soft-vs-DENSE
  Pareto claims stand (dense is batching-independent); early 8-GPU soft arms are if anything overstated (conservative).
- 09-15 10:35 nq closed for the night: nqcc17-32M 0.785 @158 PF (32k .66, 64k .52) vs kv33-32M 0.849 @286 PF and
  dense-16M 0.844 @683 PF — the readable slot does not rescue keep 1/6 at long rungs (nq needs gold + ~1/3 real docs
  at 32k/64k); nqcc08 worse (0.648 @16M). kv33 stays the nq frontier (kv33-64M 0.878 @556 PF vs dense-16M 0.844 @683).
  Remaining evals: occ00/occ08-32M and nqcc08-32M 64k rungs.
- 09-15 10:45 User: never accept outlier parity; confirm eval-time loss parity is verified for all ideas; iterate fast.
  Dispatched (a) trained-cc00-128M eval-time probe (A full text vs B own soft construction vs C plain slot vs D gold-real
  oracle vs E bodies truncated to 8 tok) at 2k/8k/16k + reusable harness flag; (b) fast8k harness + new keep mode
  `gold_pooled_random` (gold always pooled, random non-gold real) + first sweep (cpi17/33/50, cc00, kvgb50, two-phase,
  dense ×3). Handoff-update agent still waiting on the last 64k rungs.
- 09-15 10:55 User: "do we have anything for outlier with eval-loss parity, e.g. at 32k?" Answer: NO — frozen-dense probes
  at 2k/8k show no construction at parity (gold-only: CE 0.194 vs 0.010, genF1 0.32 vs 0.98; keep ½: R@gold_real 0.73 vs
  0.98 — pooling the OTHER docs hurts even real ones). 32k unmeasured on the new harness; the 09-08 "outlier parity
  gold-only @32k" (24 rows, answer-CE) is WITHDRAWN pending re-measurement. Asked the trained-cc00 probe agent to add the
  32k rung + frozen-dense baselines (gold-only mean/cent_cmean, keep ½, keep 0+header) and state parity per rung.
- 09-15 11:00 User: focus on CE parity on a TRAINED checkpoint first. Redirected: probe agent ships a one-command
  `--trained-parity` check (CE_full vs CE_soft, same ckpt) and runs it on cc00-128M/32M at 8k/32k; fast8k agent makes
  ΔCE the primary column for every arm and adds cpi33 at two budgets (does ΔCE shrink with budget?).
- 09-15 11:10 User clarified: the target is FROZEN-model eval parity (as the other 3 tasks have). Dispatched the real-token
  subset hunt on dense-u64M at 2k/8k/32k: first{4..64} (+/- slot), idf{4,8,16}, idfspan, firstlast, sent1, swap control.
  Trained-cc00 probe narrowed to the --trained-parity check + cc00 conditions. Once a frozen-parity construction exists,
  the training-side equivalent (per-token keep mask, not per-doc) is the next implementation step for fast8k/ladder.
- 09-15 11:25 User pointer → old investigation (records/pooled-doc-kv-attention.md, 09-08 post-gold-fix): outlier frozen
  dense @32k (24 rows, answer CE) FULL 1.206 vs GOLD-ONLY real 1.201 = parity; gold + random real 1/3 → 1.398 (mixture
  effect: mixing real and pooled docs hurts); +log L bias hurt outlier (+0.8 nats). Caveats: parity at the FLOOR (FULL
  itself fails at 32k), and this week's 2k probe has gold-only at 0.194 vs 0.010 (genF1 0.32). Gold-only is the training
  shortcut, so it is a frozen-model REFERENCE, not a recipe. Asked the real-token hunt to include gold-only (mean and
  cent_cmean) at 2k/8k/32k with digit-CE + genF1 to settle it per rung. Also dispatched `--st-header-extra-tokens K`
  (first-K body tokens real) trainer flag + ck08/16/32 arm defs, so a first-k frozen-parity result can go to fast8k at once.
- 09-15 11:35 oolong 32M complete: occ00 0.643 @228 PF vs dense-32M 0.644 @811 PF → parity at 3.6× holds at 32M (keep 0,
  headers real, cent_cmean); occ08 0.644 @245; ohdr08 0.643 @245. All gen-4 training evals are in; 99 scored rows.
- 09-15 11:45 User ideas (eval-loss parity only, no training): (1) saliency-selected real tokens — oracle (input-gradient /
  attention-mass top-k per doc) vs a transferable feature rule (IDF, position, digit, first-sentence); (2) one-layer
  preview — layers 0..L−1 dense over the whole context, slot = per-doc mean of layer-L states (context-wide), remaining
  layers compacted (= layer probe fromL + layer_input_mean on outlier), plus layer-L attention scores as gradient-free
  token selector. Dispatched as debug/pooled_kv/outlier_probe/outlier_saliency_preview_probe.py on dense-u64M @2k/8k/32k.
  Now 6 agents live: real-token hunt, saliency/preview, trained-cc00 parity, fast8k+gold_pooled_random, first-k flag, handoff.
- 09-15 10:55 tick: orchestrator alive (369085; launched its end-of-campaign harvest job 01M2K30SSJE3QQTSNGGYR3CR47);
  99 scored rows unchanged → no republish; no kills. Handoff rewritten by agent (top summary, §1.0 per-task tables,
  §8a outlier mechanism, §11 slot mode, traps 1–14, §5 next steps); I retitled §8a from "CLOSED" to "hunt ACTIVE" per
  the user's directive and pointed it at this plan file. 6 agents live; no results yet from realtoken / saliency-preview /
  trained-parity / fast8k / first-k flag.
- 09-15 18:00 **Eval-time CE parity is now a one-command check of the outlier probe — run it on EVERY new
  soft checkpoint before believing its ladder number.** A ladder eval always feeds full real text; an arm
  trained at keep 0 never saw a real document body, so its ladder f1 confounds "the slot readout does not
  scale" with "eval-time distribution shift". `--trained-parity` scores the SAME checkpoint twice per rung
  (full real text vs its own training construction) and prints CE / CE-on-digits / genF1 both ways with the
  deltas, all rungs in one process:
  `python debug/pooled_kv/outlier_probe/outlier_slot_probe.py --trained-parity --ckpt-name ds64-outlier-cc00-b128f3-u128M --rungs 2k,8k,16k,32k --rows 240 --gen-rows 32`
  The construction comes from the run name's arm field via `ARMS` (mirrors `launch_ds64.py:ARM_EXTRA`); for an
  unregistered arm pass `--construction` with the trainer's flags, commas for spaces. The `cent_cmean` stop set
  is rebuilt from the arm's own TRAINING shard, verified bit-identical to the training job's (1244 ids from
  20001129 tokens), and `--trained-parity` keeps the checkpoint's projector (`max|w_out| = 0` => identity).
  Commits 3c053b656 / 6b48341e7. Writeup + results: `records/outlier-cc00-evaltime-probe.md`.
- 09-15 11:25 `--st-header-extra-tokens K` shipped (chunked_mask.py mark_doc_headers_free; slot excludes freed tokens
  automatically; 5 CPU tests; hunks landed in b69674dd4 via a git-index race with a concurrent commit, formatting fix
  83e387c7d; pushed tree verified). Arms ck08/ck16/ck32 defined in launch_ds64.py (uncommitted), NOT launched — gated on
  the frozen real-token hunt (which k, if any, gives frozen-model parity). Decoded K=8 row:
  'Document [1]: in his Gibson Motorsport-prepared Holden<SLOT>'.
- 09-15 11:40 Trained cc00-128M eval-time parity (⚠240 rows/144 gold): 2k FULL genF1 0.750 vs OWN-construction 0.778 (parity,
  reads slots: R@gold_pooled 0.78 vs 0.24 plain slot); 8k FULL 0.229 vs OWN 0.125 (digit-CE 0.63 vs 0.99) — soft is WORSE
  than full at 8k, no parity; 32k (job mid-run, 50/120 rows) FULL 0.000 vs OWN 0.042 — both fail (the trained model is far
  below dense at 32k anyway). Gold-real oracle D hurts the keep-0 model at every rung (distribution shift). Conclusion:
  cc00's slot reading does not scale with doc count (13 docs ok, 56 not). Frozen-model hunts (real-token, saliency, preview)
  are the live routes; realtoken 32k job queued, saliency probe launching.
- 09-15 11:50 User guidance: outlier needs HARD NEGATIVES preserved in the kept context (ambiguous docs); uniform per-doc k
  and random-doc keeps can starve them; the gradient probe should first be used as a DIAGNOSTIC (where does saliency mass
  go: gold vs non-gold vs headers; concentration over docs; token types; correlation with embedding ambiguity; right vs
  wrong rows). Actions: realtoken hunt += margin{M} (embedding-only ambiguity keep), margin+first-k, hardneg oracle, task
  generator structure + measured hard-negative rate; saliency probe += row-level adaptive budget, diagnostic first;
  fast8k agent: trim sweep, implement gold-blind `margin_keep` training mode (hold launch until frozen parity is shown).
- 09-15 11:55 User's preferred keep rule: cluster docs by topic (cent_cmean vectors), keep EVERY doc in the C=3–5 smallest
  categories REAL, pool the big categories (optionally one slot per pooled CATEGORY). Gold-blind, embedding-only, preserves
  outliers + hard negatives by construction. Sent to the frozen hunt as `smallcat{C}` (priority construction, report rule
  recall = fraction of gold in the kept set, compaction, ΔCE at 2k/8k/32k) and to the fast8k agent as training mode
  `smallcat_keep` (`--st-keep-smallcat C`; arms sc3/sc5 defined, launch held until frozen parity is shown).
- 09-15 12:00 User hypothesis on WHY gold-forcing collapsed: gold was the ONLY category kept WHOLE (all 3 outliers real)
  while other categories got 1–2 random real docs → "complete real category" = gold signature (also explains the frozen
  mixture effect). Refinement: keep gold, but every other kept doc's WHOLE category too. Actions: frozen probe += category-
  completeness diagnostic on the old gold+1/3 construction, `smallcat{C}+decoy{D}`, and gold-aware `goldcats{K}` (gold whole
  + K other whole categories); fast8k agent += training modes `smallcat_keep` (+decoy large cats) and `gold_plus_wholecats`
  (`--st-keep-cats K`), categories never partial; arms sc3/sc5/gw3/gw5 defined, launch held for frozen parity.
- 09-15 12:10 Task structure (records/outlier-realtoken-parity-probe.md §3b, read off the generators): a topic = one Wikipedia
  article; majority articles sampled uniformly (no topical coherence, NO deliberate hard negatives); the 3 gold docs are one
  further article; the outlier topic is by construction the STRICTLY SMALLEST category (every majority article ≥4 docs);
  #majority topics ≈ 3/7/13/25 at 2k/8k/16k/32k; ~147 tok/doc. ⇒ `smallcat` is the task's generative rule read backwards:
  rule recall (gold ∈ kept set) is the ceiling and the only failure mode is the clustering. Trained cc00-128M @32k final:
  FULL 0.000 / own construction 0.042 — both fail; cc00 route closed for long rungs.
- 09-15 19:15 **cc00 eval-time parity probe ANSWERED — the outlier gap is NOT eval-time distribution shift**
  (`records/outlier-cc00-evaltime-probe.md`, 5 x 1-GPU jobs, ~2 GPU-h). Scoring `cc00-128M` twice per rung
  (full real text vs its own training construction, same checkpoint): parity at **2k only**
  (dCEdig +0.157, dF1 +0.028 at 9.5x compaction), then the SOFT side is the worse one --
  dCEdig **+0.365 / +0.421 / +0.292** at 8k/16k/32k and genF1 **0.125 vs 0.229** at 8k. If the ladder loss
  were distribution shift the soft side would be strong where full text is weak; it is the opposite, and the
  gap WIDENS with document count. Controls: same construction with the PLAIN mean slot is at the k/n floor at
  every rung (cent_cmean is worth +0.535 genF1 at 2k, +0.056 at 8k, +0.020 at 16k -- the readable slot bought
  2k and nothing else); the gold-real ORACLE (gold bodies real, rest pooled) is WORSE than the pure-slot
  construction at every rung (0.007 vs 0.125 at 8k, below the floor) -- a keep-0-trained reader cannot
  integrate a real body next to slots, which closes the cheap exposure fix. `cc00-32M` has no parity at any
  rung (+0.768 at 2k): 4x budget shrinks the 2k gap 4.9x and only halves the 8k+ gaps, from a level where both
  inputs already sit on the floor. => Do NOT pursue exposure fixes or more cc00 budget; the outlier verdict
  ("on the dense curve at best") stands, and the only live axis is per-document slot CAPACITY, which the
  frozen-dense richer-slot probe already found hard (G=2/4 and enc2/enc4 rejected). Worth 20 GPU-min:
  run `--trained-parity` on `occ00` (oolong keep 0, same slot, currently a 3.6x Pareto win) to check that win
  is slot reading and not an eval-format artefact.
