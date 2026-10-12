# ds64 FLOP-meter discrepancy audit (2026-09-15)

## TL;DR

The `flop_meter/actual_pflops` code is **unchanged** across every commit involved
(`flop_meter.py` is byte-identical at `dc2bdcd47`, `5d4b05da8`, `11f345b90`) and it is **not
miscounting**. The discrepancy is a **confound in how the two compared runs were launched**:
`ds64-nq-kv17-b128f3-u16M` ran on **8 GPUs / microbatch 4**, `ds64-nq-nqcc17-b128f3-u16M` ran on
**4 GPUs / microbatch 2**. Both settings change how many rows get grouped into one length-sorted
microbatch, and for soft-token/pooled-KV compaction arms (NOT dense arms) that changes how much
real padding compute the model executes. The flop meter faithfully measures that real difference.
`launch_ds64.py`'s own comment claims this can't happen ("Rows/step is unchanged, so accuracy and
FLOPs are unaffected [by gpu count]") — true for dense, **false for soft-token compaction arms**.

The oolong pair the task asked me to check as a second data point, `ohdr08` vs `occ08`, was
launched at **matching** 4 GPUs / microbatch 2 for both, and their `actual_pflops` come back
**bit-identical** (125.19810704603135 PF at 16M, 245.19052 PF at 32M) — direct, un-corrected proof
that `--st-slot-mode cent_cmean` itself has zero effect on measured FLOPs, and that the only thing
that moved kv17-vs-nqcc17's numbers is the gpus/microbatch mismatch.

## 1. Root cause

Chain of code, in order of execution:

1. **`research/token_routing/ds64/launch_ds64.py:116`** — `GPUS = int(os.environ.get("DS64_NGPU", "8"))`. The
   comment above it (line ~106-114, "when jupiter's free capacity is fragmented, an 8-GPU request
   can sit unplaceable for days while 4-GPU holes go begging (2026-09-11: ...)") documents that
   `DS64_NGPU=4` is a **capacity workaround**, started being used around 2026-09-11, and asserts
   *"Rows/step is unchanged, so accuracy and FLOPs are unaffected — only the gradient-accumulation
   depth and wall-clock change."* That assertion is what's wrong for soft-token arms.
2. **`research/token_routing/ds64/launch_ds64.py:130-138`** (`ARM_MICRO`) — per-arm microbatch row count, set
   independently per arm for memory reasons (e.g. the comment at line ~108-112 about `ohdr17`
   hitting a 900s NCCL watchdog SIGABRT at micro 4, so `ohdr08`/`occ08`/`occ00`/`nqcc17`/`nqcc08`
   were all conservatively pinned to micro 2). `kv17` (launched 2026-09-09, before that trap was
   found) is micro **4**.
3. **`src/olmo_core/train/train_module/transformer/train_module.py:395-401`** — when
   `microbatch_sort_pad_id` is set, each **rank's own local batch** (`global_batch / world_size`
   rows: 16 rows/rank at 8 GPUs, 32 rows/rank at 4 GPUs, since `global_batch=128` rows for every
   ds64 arm) is sorted by real (non-pad) length, **then** sliced into contiguous microbatches of
   `rank_microbatch_size // seq_len` rows (`ARM_MICRO`). A bigger per-rank pool (4 GPUs → 32 rows)
   sorted then cut into smaller groups (micro 2) produces much tighter length matches per group
   than a smaller pool (8 GPUs → 16 rows) cut into bigger groups (micro 4).
4. **`src/olmo_core/nn/pooled_soft_token.py:273-274`** — `compact_pooled_rows` compacts each row
   to its own real post-pooling length, then re-pads the WHOLE microbatch to
   `T2 = max(len(r) for r in rows_ids)` (`new_ids = torch.full((B, T2), pad_token_id, ...)`). This
   is a genuine, executed, quadratic-in-length repad **per microbatch**, so a microbatch with more
   rows / looser length grouping pays for more real (not fictitious) attention compute on padding.
5. **`src/olmo_core/train/callbacks/flop_meter.py:145-156`** — the compaction branch computes
   `out_len = comp["tokens_out"] / rows` and `actual = comp["tokens_out"] * scale *
   num_flops_per_token(out_len)`, where `tokens_out` is `input_ids.numel()` **after** the repad-to-
   `T2` in step 4 (`src/olmo_core/nn/transformer/model.py:1591`, `stats["tokens_out"] +=
   int(input_ids.numel())`, called right after `cb = compact_pooled_rows(...)`). So the meter is
   correctly integrating the real padded compute that steps 3-4 produced — it is not a measurement
   bug, it is measuring a real difference in what the two jobs actually ran.

Per-row compaction itself (which docs get pooled vs. kept) is **deterministic per row**, seeded by
content fingerprint (`src/olmo_core/nn/attention/pooled_doc_kv.py:611`,
`rng = _random.Random(f"{seed}:{fp}")`), independent of batching or slot_mode — confirmed by the
`dense_equivalent_pflops` (688.1119754279179 PF) and `tokens_processed` (14622462) being
**bit-identical** between kv17 and nqcc17 at 16M (both read straight off the all-reduced real
non-pad token count via `FlopMeterCallback.pre_step`, `flop_meter.py:51-61`, which is
world-size-invariant by construction). Only the compacted/padded term differs.

**Why dense anchors are immune**: dense arms never populate `model._soft_token_compaction`
(`flop_meter.py:145`, `comp is None`), so their `actual` falls back to `dense_flops = tokens *
dense_tok` at line 88-89, where `tokens` comes from `pre_step`'s all-reduced real-token count —
structurally independent of gpus/microbatch grouping. Confirmed: `actual_over_dense == 1.0` for
every `*-dense-*` row in `results/ds64/results.csv`, and `dense_pflops` for the confounded pair
(nq 16M: 688.11 PF for both kv17 and nqcc17) matches the dense arm's own number scaled to the same
token count.

## 2. Evidence pulled

Commits the five jobs actually ran (`Cloning source from 'allenai/OLMo-core' @ <sha>` in each
Beaker job log, `research/token_routing/ds64/flop_audit/*.log`):

| run | commit | date |
|---|---|---|
| `ds64-nq-kv17-b128f3-u16M` | `dc2bdcd47` | 2026-09-09 |
| `ds64-oolong-ohdr08-b128f3-u16M` | `5d4b05da8` | 2026-09-14 |
| `ds64-nq-nqcc17-b128f3-u16M` | `11f345b90` | 2026-09-15 |
| `ds64-oolong-occ08-b128f3-u16M` | `11f345b90` | 2026-09-15 |
| `ds64-oolong-occ00-b128f3-u16M` | `11f345b90` | 2026-09-15 |

`git diff` of `flop_meter.py` between all three commits is **empty**. The only relevant diffs
(`dc2bdcd47`→`5d4b05da8`, `5d4b05da8`→`11f345b90`) touch `enable_pooled_soft_tokens`'s
compression-mixing curriculum (`mix_start_p`/`mix_end_p`/`mix_total_calls`, unused here — every
arm below launches with `p_full=0.00`, confirmed in the `[pooled-kv] ... p_full=0.00` log lines)
and the new `slot_mode`/`slot_stop_ids`/`slot_centroid` machinery (`apply_slot_mode`, which only
rewrites the pooled slot's *feature vector*, not which tokens get compacted). Neither touches
`_soft_token_compaction`'s `tokens_in`/`tokens_out`/`rows` accounting
(`src/olmo_core/nn/transformer/model.py:1578-1592`), which is identical at all three commits.

`[ctc-suite] task=... world_size=...` line from each job log:

| run | world_size (gpus) | ARM_MICRO |
|---|---|---|
| `kv17-b128f3-u16M` | **8** | **4** |
| `nqcc17-b128f3-u16M` | **4** | **2** |
| `ohdr08-b128f3-u16M` | 4 | 2 |
| `occ08-b128f3-u16M` | 4 | 2 |
| `occ00-b128f3-u16M` | 4 | 2 |

`[flop-meter] wrote .../flops.json` summary line from each job log:

| run | actual_pflops | dense_equivalent_pflops | tokens_processed | steps |
|---|---|---|---|---|
| kv17-16M | 115.22454108 | 688.1119754279179 | 14622462 | 28 |
| nqcc17-16M | 79.93129827 | 688.1119754279179 | 14622462 | 28 |
| ohdr08-16M | 125.19810705 | 414.48937845476047 | 8807949 | 28 |
| occ08-16M | 125.19810705 | 414.48937845476047 | 8807949 | 28 |
| occ00-16M | 116.32524219 | 414.48937845476047 | 8807949 | 28 |

`dense_equivalent_pflops`/`tokens_processed`/`steps` match exactly within each task — proving
identical data/step count/real-token accounting — while `actual_pflops` (the compacted term) only
diverges for the gpus/micro-mismatched nq pair, and is bit-identical for the matched oolong pair.

## 3. Which arms are affected, and direction

**Only `ds64-nq-kv17-b128f3-*` (all budgets: `u16M`/`u32M`/`u64M`/`u128M`) is out of family** —
launched 2026-09-09 at `dc2bdcd47`, 8 GPUs, micro 4 — vs. its intended same-keep-fraction
comparison,
`ds64-nq-nqcc17-b128f3-*`, launched 2026-09-15 at `11f345b90`, 4 GPUs, micro 2.

- `kv17-16M`: raw **115.22 PF**, vs. `nqcc17-16M` raw **79.93 PF** — ratio **1.442x**.
- `kv17-32M`: raw **230.62 PF**, vs. `nqcc17-32M` raw **157.55 PF** — ratio **1.464x**.

`kv17`'s number is **not wrong for what kv17 itself ran** — it is a faithful measurement of the
real compute that specific 8-GPU/micro-4 job burned (more padding waste from looser length-sort
groups). It is **overstated relative to what the same method (`gold_plus_random`, keep 1/6) would
cost at the finer 4-GPU/micro-2 batching** that every later arm (including its own slot-mode
sibling `nqcc17`) uses. So reading "kv17 = 115 PF, nqcc17 = 80 PF, therefore `cent_cmean` buys a
1.44x FLOP reduction" is **wrong** — that delta is batching, not slot construction.

`ds64-nq-nqcc08-b128f3-*` (keep 1/12, 4 GPUs / micro 2) has **no plain-mean (`kv08`) counterpart
on nq at all** (the arm doesn't exist for nq — see `launch_ds64.py`'s own comment: *"nq has no
header arms... these are kv17/kv08's plain flags... at kv17's existing keep ratio... and one step
lower"*, i.e. `kv08` was never actually run for nq), so it isn't part of a mismatched pair, but
it also has no valid same-keep-fraction baseline to be compared against — it can only be read on
its own (against dense), not as a slot_mode ablation.

**Everything else audited is clean**: `ohdr08` vs. `occ08` (both 16M and 32M) and `ohdr08` vs.
`occ00` (different keep fraction, not a strict pair, but same 4g/2m launch) are all launched at
matching 4 GPUs / micro 2 — `occ08`'s `actual_pflops` is bit-identical to `ohdr08`'s at both
budgets. `research/token_routing/ds64/flop_audit/recompute_flops.py` confirms this is true campaign-wide: it is
the **only** confounded (arm, budget) group across all four ds64 tasks (contradiction, nq, outlier,
oolong) at the 4B scale — every other same-signature arm pair (outlier's `xhdr17`/`xh2warm17`,
`xhdr00`/`cc00`) is launched at matching gpus/micro and comes back with ratio 1.0000.

## 4. Dense anchors

**Unaffected**, both structurally (§1: `dense_flops` never touches `_soft_token_compaction`, uses
the all-reduced real-token count from `pre_step`, which is world-size-invariant) and empirically
(`actual_over_dense == 1.0` for every dense row in `results/ds64/results.csv`; `dense_pflops` for
kv17 and nqcc17 is bit-identical at 688.1119754279179 despite the gpus/micro mismatch). The
requested dense anchors:

| run | flops_meter (= dense_pflops, actual_over_dense=1.0) |
|---|---|
| `ds64-nq-dense-u16M` | 683.13867 PF |
| `ds64-oolong-dense-u16M` | 418.77209 PF |

## 5. Recommended fix for comparability

There is **no retroactive numeric correction available from the logs**: `flops.json` only persists
the final cumulative `actual_pflops`/`dense_pflops`, not the per-step `tokens_in`/`tokens_out`/
`rows` that would let you re-derive what kv17 would have measured at 4 GPUs/micro 2 (that depends
on the exact per-row post-compaction lengths and how the length-sort happened to bucket them —
order-statistics-dependent, not a simple scalar rescale). So:

1. **Comparability going forward**: always launch a method-ablation pair (e.g. slot_mode A vs. B)
   at **identical gpus and identical `ARM_MICRO`**. Fix `launch_ds64.py`'s `GPUS`/`ARM_MICRO`
   selection to be pinned per comparison group, not per individual arm's own memory headroom, OR
   at minimum flag in the launch ledger when a comparison group's gpus/micro don't match.
2. **What to do with the existing kv17 vs. nqcc17 numbers**: don't use them as a slot_mode
   comparison. `nqcc17`'s own 79.93/157.55 PF is a legitimate same-keep-fraction reading at the
   "efficient" (4g/2m) batching; there is no `kv17`-at-4g/2m number to compare it to without a
   rerun. The plain-mean-slot kv17 numbers (115.22/230.62 PF) remain valid **on their own** (e.g.
   against dense, or against kv33, both of which were also launched at 8g/4m) — just not against
   `nqcc17`/`nqcc08`.
3. **Mechanical audit tool** (does NOT modify `results.csv`, only reads it plus
   `launch_ds64.py`'s `ARM_EXTRA`/`ARM_MICRO`): `research/token_routing/ds64/flop_audit/recompute_flops.py`. It
   groups arms sharing an identical keep-signature (ARM_EXTRA with `--st-slot-mode`/
   `--st-slot-tokenizer` stripped) per (task, budget), flags any group whose members ran at
   different gpus/micro as CONFOUNDED with the observed ratio, and flags matched-gpus/micro groups
   as CLEAN with their ratio (sanity check — should be ~1.0000 whenever the compared arms are true
   slot-construction-only ablations). Run: `python research/token_routing/ds64/flop_audit/recompute_flops.py`.
   Current output: the nq kv17/nqcc17 pair (both budgets) is the only CONFOUNDED group in the
   whole 4B campaign; everything else (oolong ohdr08/occ08, outlier xhdr17/xh2warm17,
   xhdr00/cc00) is CLEAN at ratio 1.0000.

## Corrected numbers for the runs requested

| run | reported actual_pflops (PF) | status |
|---|---|---|
| `ds64-nq-kv17-b128f3-u16M` | 115.22454 | real, but NOT comparable to nqcc17 (8g/4m vs 4g/2m confound); no valid corrected number without a rerun at 4g/2m |
| `ds64-nq-nqcc17-b128f3-u16M` | 79.93130 | real and directly usable as the "keep 1/6, gold_plus_random, cent_cmean slot" cost at efficient (4g/2m) batching |
| `ds64-oolong-ohdr08-b128f3-u16M` | 125.19811 | real, clean (matches occ08 exactly — no correction needed) |
| `ds64-oolong-occ08-b128f3-u16M` | 125.19811 | real, clean, bit-identical to ohdr08 |
| `ds64-oolong-occ00-b128f3-u16M` | 116.32524 | real, clean (different keep_prob from ohdr08/occ08 by design, not a batching artifact) |
| `ds64-nq-dense-u16M` (anchor) | 683.13867 | real, unaffected by gpus/micro by construction |
| `ds64-oolong-dense-u16M` (anchor) | 418.77209 | real, unaffected by gpus/micro by construction |

(32M: kv17=230.61914, nqcc17=157.55022, ohdr08=245.19052, occ08=245.19052, occ00=228.23219 — same
pattern, same confound, same clean pairs.)
