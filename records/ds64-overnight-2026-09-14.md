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
