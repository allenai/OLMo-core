"""
Launch the short-heavy 2k-64k data-scaling grid (records/ds64-scaling-plan.md): dense vs the best
soft-token construction per task, Qwen3.5-4B (SCALE=4b, default) or 27B (SCALE=27b), on the
marker shards debug/ds64/build_ds64_data_beaker.sh wrote to weka (ds64/shards/<task>_u<B>).

Arms (all soft arms: detached slots -- attention K/V AND GDN writes -- no bias, GDN intact forward):
  dense            packed seq 65536, 8 rows/step, flash_2
  hdr03/08/17/33   contradiction: `Claim N:` headers real, gold + 1/36 .. 1/3 random docs real (keep ablation)
  runs03/08        contradiction: gold + random picks kept as +-1 runs (no header)
  ohdr33/17/08     oolong: Date/User/Instance headers real, 1/3 .. 1/12 of lines real (gold-blind)
  xhdr33/17/00     outlier: `Document [N]:` headers real, 1/3 .. 0 of bodies real (gold-blind) -- DEAD, see below
  xh2k50/xh2warm17 outlier: header-real follow-ups that ask whether xhdr's collapse is an
                   OPTIMIZATION shortcut (the guess policy is reachable in a few steps once the ids
                   are visible) or a HARD leak (the pooled slot cannot carry topic at all).
                   xh2k50 = same header, keep 1/2 -- does the header ALWAYS collapse, or only at
                   low keep? (kvgb50 is its header-free twin, and it reaches dense parity.)
                   xh2warm17 = header + keep 1/6, WARM-STARTED from the finished kvgb50-16M
                   checkpoint -- if a content-grounded model does not fall back to guessing, the
                   collapse is a basin/optimization problem, not a property of the construction.
  xh2mix17/kvgbmix2 outlier: the compression-MIXING curriculum, which only started working in
                   commit 7ce879b40 (it was a silent no-op on every gold-blind arm, so the old
                   `kvgbmix` is a byte-identical re-run of `kvgb` -- do not cite it).
                   xh2mix17 = header + keep 1/6 + p_full 0.5 -> 0; kvgbmix2 = keep 1/2, no header.
  cc00/17/03/08    outlier: cc00's recipe (xhdr00 + --st-slot-mode cent_cmean) at keep 0/1-6/1-36/1-12
  ck08/16/32       outlier: cc00's exact recipe + --st-header-extra-tokens {8,16,32} -- the first K
                   real BODY tokens after the header stay real too (every doc, pooled or not), on
                   top of cc00's readable slot -- does a little real body TEXT (not just a better
                   slot) break the xhdr copy-the-visible-id shortcut? See xhdr_collapse_diagnosis.md.
  kv08/17/33       nq / outlier: gold + 1/12 .. 1/3 random docs real
Phase 1 = contradiction's full set + every task's dense; other soft arms via debug/ds64/soft_arms.json.
Soft arms train UNPACKED at seq 65536, 128 rows/step (~ the dense 524k tokens/step at the mix's
~4.5k mean length), micro-batch sized per arm (ARM_MICRO), torch backend unless DS64_SOFT_BACKEND.

    python debug/ds64/launch_ds64.py --tasks contradiction --budgets 32M --arms dense,hdr36 dry_run
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
import subprocess
import sys

REPO = "/accounts/projects/berkeleynlp/prasann/projects/OLMo-core"
LAUNCHER = f"{REPO}/src/scripts/train/memexpress/ctc_suite/beaker_ctc_suite.py"
WEKA = "/weka/oe-training-default/ai2-llm/checkpoints/prasanns"
SCALE = os.environ.get("DS64_SCALE", "4b")
BASE = {"4b": f"{WEKA}/ctc_suite/bases/q35-4b-base-markerfix/model_and_optim",
        "27b": f"{WEKA}/ctc_suite/bases/q35-27b-base-markerfix/model_and_optim"}[SCALE]
SHARDS = f"{WEKA}/ds64/shards"
LEDGER = f"{REPO}/debug/ds64/LAUNCH_LEDGER.tsv"
BUDGETS = ["16M", "32M", "64M", "128M"]
# contradiction's pair pool cannot build the 128M arm (compose skipped it): its grid stops at 64M
# Outlier runs a DEEPER ladder than the others: its gold-blind arms land at 156-190 PF, far below
# dense's cheapest measured point (780 PF), so every cheap-end comparison was extrapolated. 4M/8M
# ride the 16M shard via --max-tokens (see shard_for) -- a real budget with a complete LR schedule.
TASK_BUDGETS = {"contradiction": ["16M", "32M", "64M"],
                "outlier": ["4M", "8M", "16M", "32M", "64M", "128M"]}


# Per-ARM budget restrictions, intersected with the task's grid. Outlier's 4M/8M rungs exist for
# the DENSE anchors only: a soft arm does 128 rows x 65536 = 8.4M tokens/step, so a 4M or 8M budget
# silently collapses to a ONE-STEP run (no error) and lands far left of the dense curve, where it
# scores the largest apparent matched-FLOP win in the whole ladder -- pure artefact. See
# records/ds64-handoff.md section 1. 16M is the floor for any soft arm.
ARM_BUDGETS = {a: ["16M", "32M", "64M"] for a in ("xhdr33", "xhdr17", "xhdr00")}
# The xh2 follow-ups are decided by the 16M/32M points (28 / 56 steps): if the header
# collapse is an optimization shortcut it is already visible by step 8, where every xhdr
# CE curve flattened. 64M buys nothing the cheap points do not already show.
ARM_BUDGETS.update({a: ["16M", "32M"] for a in ("xh2k50", "xh2warm17")})
# Same reasoning for the two compression-MIXING arms (2026-09-14, gen-4c): the curriculum anneals
# p_full to 0 over the first half of training either way, so 16M/32M already show whether mixing
# moves the CE plateau.
ARM_BUDGETS.update({a: ["16M", "32M"] for a in ("xh2mix17", "kvgbmix2")})
# oolong ohdr08 (queue item E, records/ds64-overnight-2026-09-14.md): headers-only ablation at
# keep 1/12, pinned to 16M/32M only -- 64M/128M are not worth the spend until the cheap points
# show whether the low-keep header-only construction tracks ohdr33/ohdr17 or collapses.
ARM_BUDGETS.update({"ohdr08": ["16M", "32M"]})
# Content-only slot arms (records/outlier-richer-slot-probe.md + the 2026-09-15 fast2k screen).
# `cc00` is the one the screen supports: at keep 0 + header real, swapping the plain-mean slot for
# `cent_cmean` took the 2k f1 from 0.239 (xhdr00, the same-FLOP control) to 0.482 and the final CE
# from 0.408 (the 0.390 format floor) to 0.348, at 0.095x dense -- the cheapest arm in the set.
# `cc17` is registered for completeness and is NOT recommended: at keep 1/6 the same slot change
# does nothing (0.237 vs xhdr17's 0.234, CE 0.408 both) -- see the log in
# records/ds64-overnight-2026-09-14.md. 2k cannot see length generalisation either way.
ARM_BUDGETS.update({"cc00": ["16M", "32M", "64M", "128M"], "cc17": ["16M", "32M", "64M"],
                    "cc03": ["16M", "32M", "64M", "128M"], "cc08": ["16M", "32M"]})
# Overnight 2026-09-15: does the readable content-only slot (cent_cmean) let oolong/nq go to
# LOWER keep than their current best-known-working arms, without losing accuracy?
#   occ08/occ00: oolong occ08 is ohdr08's exact flags (headers real, keep 1/12, gold-blind) plus
#     --st-slot-mode cent_cmean -- a same-FLOP comparison to ohdr08, which just hit a new Pareto
#     dominance (3.3x, records/ds64-overnight-2026-09-14.md 09-15 07:00). occ00 is the maximal
#     -compression twin (keep 0.0, headers only, every body pooled) -- the oolong analogue of the
#     outlier cc00 win.
#   nqcc17/nqcc08: nq has no header arms (nq's ids are not in a rendered header the way
#     outlier's/oolong's are), so these are kv17/kv08's plain `gold_plus_random` flags plus
#     --st-slot-mode cent_cmean, at kv17's existing keep ratio (1/6, control = the already-run
#     plain-slot kv17 rows) and one step lower (1/12).
# All four pinned to 16M/32M only -- cheap points decide whether to extend the ladder.
ARM_BUDGETS.update({a: ["16M", "32M"] for a in ("occ08", "occ00", "nqcc17", "nqcc08")})
# ck08/ck16/ck32 (2026-09-15): outlier, xhdr00's exact gold-blind/header-real/cent_cmean recipe
# (== cc00) PLUS --st-header-extra-tokens {8,16,32} -- the first K real body tokens after the
# header, kept real for every doc including fully-pooled ones. Tests whether a few tokens of
# actual body content (not just the readable slot) is what it takes to break the xhdr shortcut
# (debug/ds64/xhdr_collapse_diagnosis.md): the header alone hands the model a copyable-but-
# unjustifiable id; real body text -- even a handful of tokens -- gives it something to read
# instead. Full ladder (cc00 is the strongest known same-family arm and got the full ladder too).
ARM_BUDGETS.update({a: ["16M", "32M", "64M", "128M"] for a in ("ck08", "ck16", "ck32")})
ARM_BUDGETS.update({"ck64": ["16M", "32M", "64M", "128M"]})
# rk08/rk16 (2026-09-15): the SAME budget as ck08/ck16, spent on DIFFERENT tokens --
# --st-keep-token-rule rule keeps the K highest-scoring body tokens per document (cheap token
# features: idf, relative position, first-sentence, capitalised, digit, piece length) instead of
# the first K. The eval-side probe says the choice is the whole effect at this budget: at K=8 on
# outlier, random selection scores CE 0.558 -- WORSE than keeping nothing (cc00 0.470) -- the
# first 8 score 0.263, and the rule scores 0.071, matching the gradient ORACLE's 0.072
# (records/outlier-saliency-preview-probe.md 5d). ck08/ck16 are therefore the exact controls.
# Same full ladder as ck*, so the pair is comparable rung for rung.
ARM_BUDGETS.update({a: ["16M", "32M", "64M", "128M"] for a in ("rk08", "rk16")})
# chk32/chk64 (contradiction) and nqk32/nqk64 (nq), 2026-09-15: does the first-k real-body-token
# recipe that gave outlier its ck32/ck64 Pareto dominance (16:40 log, ds64-overnight-2026-09-14.md)
# generalise to the other two tasks at maximal compression (keep 0.0, gold-blind)? Pinned to
# 16M/32M/64M -- the same three rungs ck32/ck64 needed to show the effect on outlier.
ARM_BUDGETS.update({a: ["16M", "32M", "64M"] for a in ("chk32", "chk64", "nqk32", "nqk64")})
# flk32/flk64 (outlier, 2026-09-15): cc00's recipe (gold-blind, keep 0.0, header-real, cent_cmean
# slot) + --st-keep-token-rule first_last --st-keep-token-k {32,64} instead of --st-header-extra-tokens
# -- first_last beats first (== ck32/ck64) on every rung of the frozen-model probe (b94958ac7).
ARM_BUDGETS.update({a: ["16M", "32M", "64M"] for a in ("flk32", "flk64")})
# slot-less twins of the two frontier arms (2026-09-22): same three rungs as their slot arms
ARM_BUDGETS.update({a: ["16M", "32M", "64M"] for a in ("chk32ns", "flk64ns")})


def budgets_for(task, arm=None):
    """Budgets to run for ``task`` (optionally narrowed to ``arm``'s own allowed set)."""
    grid = TASK_BUDGETS.get(task, BUDGETS)
    allowed = ARM_BUDGETS.get(arm)
    return [b for b in grid if b in allowed] if allowed else grid
TASKS = ["contradiction", "oolong", "nq", "outlier"]
# flash_2: on multi-row right-padded micro-batches the torch SDPA path is ~4x slower (local test
# 2026-09-08 22:50: micro 8, one step, torch 234 s vs flash 71 s incl. model load; equal at micro 2).
# Flash is exact for right-padded causal rows (real tokens never attend to the trailing PAD).
SOFT_BACKEND = os.environ.get("DS64_SOFT_BACKEND", "flash_2")
# per scale: nodes, GPUs/node, soft micro-batch, cluster
NODES = {"4b": 1, "27b": int(os.environ.get("DS64_NUM_NODES", "2"))}[SCALE]
# 8 by default. DS64_NGPU=4 asks for half a node: when jupiter's free capacity is fragmented, an
# 8-GPU request can sit unplaceable for days while 4-GPU holes go begging (2026-09-11: 2 nodes with
# 8 free vs 3 with >=4). Rows/step is unchanged, so accuracy and FLOPs are unaffected -- only the
# gradient-accumulation depth and wall-clock change. Keep (rows/step) % (micro * ngpu) == 0.
GPUS = int(os.environ.get("DS64_NGPU", "8"))
# Soft arms train UNPACKED rows; the short-heavy mix averages ~4.5k tokens/example, so the rows
# per step must match dense's ~524k tokens/step (8 packed 65536 rows) or the per-step overhead
# dominates wall-clock (first launch at 16 rows/step: 4.4x SLOWER than dense at 0.18x the FLOPs)
# and the soft arm gets 7x more optimizer steps than dense. 128 rows/step ~= 576k tokens/step.
SOFT_GB = int(os.environ.get("DS64_SOFT_GB", "128"))
# rows per micro-batch, sized to the arm's compaction so a micro-batch of 56k rows fits an 80GB GPU
# (rows/step must be divisible by micro x 8 GPUs -> micro in {1, 2, 4, 8, 16})
# ohdr17 is 2, not 4, on purpose: oolong pads an 8.9M-token shard to 238M (27x), and at micro 4 a
# single step ran past the process group's 900 s timeout, so the NCCL watchdog aborted the job
# (SIGABRT, no Python traceback). ohdr33 survives the same data at micro 2. See records/ds64-handoff.md §2.
# ohdr08 is conservatively 2, not 4, for the same reason -- it compacts more than ohdr17 so micro 4
# may well be fine, but it has never been run before (2026-09-14) and the ohdr17 failure mode is a
# silent SIGABRT with no traceback, so the untested case defaults to the known-safe micro.
ARM_MICRO = {"hdr03": 8, "runs03": 8, "kv08": 8, "hdr08": 4, "runs08": 4, "ohdr08": 2,
             "hdr17": 4, "kv17": 4, "ohdr17": 2, "hdr33": 2, "kv33": 2, "ohdr33": 2,
             "kvrb": 1, "kvmix": 2, "kvgb": 2, "kvgbmix": 2, "kvgb50": 1,
             "xhdr33": 2, "xhdr17": 2, "xhdr00": 2,
             "cc00": 2, "cc17": 2, "cc03": 2, "cc08": 2,
             "sc3": 2, "sc5": 2, "gw3": 2, "gw5": 2,
             "ck08": 2, "ck16": 2, "ck32": 2, "ck64": 2, "rk08": 2, "rk16": 2,
             "flk32": 2, "flk64": 2,
             "chk32ns": 2, "flk64ns": 2,
             "chk32": 2, "chk64": 1, "chk16": 2, "nqk32": 2, "nqk64": 2,
             "occ08": 2, "occ00": 2, "nqcc17": 2, "nqcc08": 2,
             "xh2k50": 2, "xh2warm17": 2,
             # micro 1, NOT 2, for the mixing arms: a row the curriculum promotes to UNCOMPRESSED
             # is its full original length, and length-sorted micro-batches put the 56k rows
             # together -- micro 2 would be 112k tokens in one micro-batch, ~1.7x the dense arm's
             # 65536, i.e. an OOM waiting to happen. kvgb50 (keep 1/2, no mixing) is already 1.
             "xh2mix17": 1, "kvgbmix2": 1,
             "hdr03gb": 8, "hdr03m1": 1, "hdr03prof": 8}
CLUSTER = os.environ.get("DS64_CLUSTER", {"4b": "ai2/jupiter-cirrascale-2", "27b": "ai2/titan-cirrascale"}[SCALE])

# keep-ratio suffix as in the old grid: 03 = 1/36, 08 = 1/12, 17 = 1/6, 33 = 1/3
_HDR1 = "--st-keep-mode gold_plus_random --st-header-stop-id 25 --st-header-stop-count 1"
ARM_EXTRA = {
    # contradiction: `Claim N:` headers real + gold + a fraction of random docs (keep-ratio ablation first, Prasann 2026-09-08)
    "hdr03": f"--st-keep-frac 0.0278 {_HDR1}", "hdr08": f"--st-keep-frac 0.0833 {_HDR1}",
    "hdr17": f"--st-keep-frac 0.1667 {_HDR1}", "hdr33": f"--st-keep-frac 0.3333 {_HDR1}",
    # contradiction: leak-free neighbour runs, no header
    "runs03": "--st-keep-frac 0.0278 --st-keep-mode gold_plus_random --st-neighbour-runs 1",
    "runs08": "--st-keep-frac 0.0833 --st-keep-mode gold_plus_random --st-neighbour-runs 1",
    # oolong: Date/User/Instance headers real, a fraction of lines real (gold-blind)
    "ohdr33": "--st-gold-blind --st-keep-prob 0.3333 --st-header-stop-id 25 --st-header-stop-count 3",
    "ohdr17": "--st-gold-blind --st-keep-prob 0.1667 --st-header-stop-id 25 --st-header-stop-count 3",
    "ohdr08": "--st-gold-blind --st-keep-prob 0.0833 --st-header-stop-id 25 --st-header-stop-count 3",
    # oolong iteration 2 (2026-09-15 overnight): exactly ohdr08's flags plus the content-only,
    # row-centred slot (--st-slot-mode cent_cmean, records/outlier-richer-slot-probe.md) -- does a
    # readable slot let oolong go LOWER than ohdr08's 3.3x dominance without losing accuracy?
    #   occ08: same keep (1/12) as ohdr08 -- a same-FLOP comparison to the plain-mean control.
    #   occ00: keep 0.0 -- headers only, every line pooled -- the maximal-compression oolong arm,
    #     mirroring the outlier finding that cent_cmean only pays off once no real body remains.
    "occ08": ("--st-gold-blind --st-keep-prob 0.0833 --st-header-stop-id 25 --st-header-stop-count 3 "
              f"--st-slot-mode cent_cmean --st-slot-tokenizer {WEKA}/hf_tokenizers/Qwen3.5-0.8B-Base"),
    "occ00": ("--st-gold-blind --st-keep-prob 0.0 --st-header-stop-id 25 --st-header-stop-count 3 "
              f"--st-slot-mode cent_cmean --st-slot-tokenizer {WEKA}/hf_tokenizers/Qwen3.5-0.8B-Base"),
    # outlier REPAIR arms (2026-09-09). kv17/kv33 collapse beyond 2k on outlier: --st-keep-frac
    # pins one compression ratio, and it OVERRIDES --st-n-random-range, which is the mechanism
    # pooled_doc_kv.py documents for exactly this failure -- "training at varied candidate breadths
    # teaches ranking that is scale-invariant, so the eval regime (every doc real) is not
    # out-of-distribution". keep-frac is right for a scale-invariant FLOP study and wrong for
    # train/eval transfer; these two arms buy the transfer back, by breadth and by curriculum.
    # The decisive one: oolong is the ONLY gold-blind arm and the only one that IMPROVES at long
    # context, while nq and outlier share byte-identical flags yet differ hugely -- so the shortcut
    # ("gold is always among the few real docs"), not the flag set, is what outlier cannot survive.
    # This is oolong's policy applied to outlier, with everything else held fixed.
    "kvgb": "--st-gold-blind --st-keep-prob 0.3333",
    # Outlier iteration 2 (2026-09-12), staged behind kvgb. Both keep gold-blind -- the shortcut is
    # settled -- and attack the REMAINING gap, which is the train/eval distribution jump (train sees
    # mostly pooled docs, eval sees every doc real).
    #   kvgbmix: the compression-mixing curriculum -- each row trains UNCOMPRESSED with prob p,
    #     annealed 0.5 -> 0 over the first half. Pooled analogue of the mask-mixing curriculum that
    #     is already the standing default elsewhere. Keeps keep 1/3, so the FLOP saving is intact:
    #     strictly the better bet if it works.
    #   kvgb50: raise keep to 1/2. Blunter -- buys accuracy by giving back compression (~0.4x vs
    #     0.23x at keep 1/3), so it only wins if accuracy rises faster than the dense curve does.
    "kvgbmix": ("--st-gold-blind --st-keep-prob 0.3333 "
                "--st-mix-start-p 0.5 --st-mix-end-p 0.0 --st-mix-anneal-frac 0.5"),
    "kvgb50": "--st-gold-blind --st-keep-prob 0.5",
    # Outlier iteration 3 (2026-09-14, Prasann: "do you ever keep a few header tokens per document
    # un-detached, like OOLONG?"). Header-real is where contradiction (hdr33) and oolong (ohdr33)
    # got their matched-FLOP wins and it was never tried on outlier in TRAINING. Gold-blind, since
    # the gold-forcing shortcut is settled (kv33 -> kvgb, ~70x).
    #
    # THE STOP ID IS 5491, NOT 25. Contradiction renders `Claim 269:` and oolong `Date: ... ||
    # User: ...`, whose colons tokenize as the standalone ':' (id 25). Outlier renders
    # `Document [7]:` and Qwen3.5 fuses the bracket and the colon into ONE token `']:'` = 5491, so
    # id 25 NEVER occurs in an outlier header: mark_doc_headers_free would find no stop and fall
    # back to its `dist <= cap` guard, keeping the first 32 tokens of EVERY document real (~25% of
    # a 130-token wiki100w passage -- a quarter of the compaction, silently). Decoded and measured
    # in debug/ds64/probe_outlier_header.py:
    #   <|box_start|> | '\n\n' | 'Document' | ' [' | '1' | ']:' | ' to' | ' finish' | ...
    # stop id 5491 / count 1 gives exactly `\n\nDocument [N]:` (~6 tokens/doc, 662 over 110 docs).
    #   xhdr33/xhdr17: headers real + 1/3, 1/6 of bodies real -- the ohdr33/ohdr17 ladder.
    #   xhdr00: keep_prob 0.0 -- headers real, EVERY body pooled. The cheapest point, and the one
    #     that asks the question directly: is a real document id enough to name the odd one out?
    #     keep_prob 0.0 is valid (resolve_keep_docs returns `u < keep_prob`, u >= 0 -> keep nothing);
    #     forward+backward smoke-tested on CPU at 0.0 / 1/6 / 1/3.
    # !! DEAD 2026-09-14 -- DO NOT RELAUNCH. All three arms train to the uniform-guess floor
    # (f1 = k/n exactly, at every rung, every keep_prob, every budget). Header-real makes EVERY
    # `Document [N]:` id real, including for fully-pooled docs, so a gold-blind row hands the
    # model a copyable-but-unjustifiable target and it converges to "sample 3 ids from the
    # visible list". Not a bug in mark_doc_headers_free (verified token-by-token on CPU).
    # NEVER pair --st-header-stop-id with --st-gold-blind when the answer is drawn FROM the
    # header (outlier, nq, rerank). See debug/ds64/xhdr_collapse_diagnosis.md.
    "xhdr33": "--st-gold-blind --st-keep-prob 0.3333 --st-header-stop-id 5491 --st-header-stop-count 1",
    "xhdr17": "--st-gold-blind --st-keep-prob 0.1667 --st-header-stop-id 5491 --st-header-stop-count 1",
    "xhdr00": "--st-gold-blind --st-keep-prob 0.0 --st-header-stop-id 5491 --st-header-stop-count 1",
    # Outlier iteration 5 (2026-09-15): same gold-blind header-real construction as xhdr*, but the
    # pooled slot is rebuilt from CONTENT tokens and centred (--st-slot-mode cent_cmean) instead of
    # being the raw mean input embedding, which is ~94% shared common-word mass. Screened on fast2k
    # at 4M: cc00 0.482 vs xhdr00 0.239 at identical FLOPs (CE 0.348 vs 0.408 against a 0.390
    # floor); cc17 0.237 vs xhdr17 0.234, i.e. no effect at keep 1/6.
    # --st-slot-tokenizer is pinned to the weka copy so the punctuation half of the slot stop set is
    # deterministic and costs no HF-hub call at job start (without it the trainer falls back to a
    # frequency-only stop set and logs that it did) -- same as the fast2k arms.
    "cc00": ("--st-gold-blind --st-keep-prob 0.0 --st-header-stop-id 5491 --st-header-stop-count 1 "
             f"--st-slot-mode cent_cmean --st-slot-tokenizer {WEKA}/hf_tokenizers/Qwen3.5-0.8B-Base"),
    "cc03": ("--st-gold-blind --st-keep-prob 0.0278 --st-header-stop-id 5491 --st-header-stop-count 1 "
             f"--st-slot-mode cent_cmean --st-slot-tokenizer {WEKA}/hf_tokenizers/Qwen3.5-0.8B-Base"),
    "cc08": ("--st-gold-blind --st-keep-prob 0.0833 --st-header-stop-id 5491 --st-header-stop-count 1 "
             f"--st-slot-mode cent_cmean --st-slot-tokenizer {WEKA}/hf_tokenizers/Qwen3.5-0.8B-Base"),
    "cc17": ("--st-gold-blind --st-keep-prob 0.1667 --st-header-stop-id 5491 --st-header-stop-count 1 "
             f"--st-slot-mode cent_cmean --st-slot-tokenizer {WEKA}/hf_tokenizers/Qwen3.5-0.8B-Base"),
    # WHOLE-CATEGORY keep policies (olmo_core.nn.attention.doc_categories, 2026-09-15). Why they
    # exist: gold_plus_random collapsed because forcing every gold document real made the gold
    # category the only COMPLETE category in the row, so "which category is entirely real?" answered
    # outlier without reading anything. Both of these keep whole categories only, and more than one,
    # so completeness names nothing. Categories come from clustering the documents' OWN cent_cmean
    # slot vectors, so the partition is computable from what a pooled document exposes.
    #   sc3/sc5  smallcat_keep, GOLD-BLIND: the C smallest categories kept whole + 1 decoy large one
    #   gw3/gw5  gold_plus_wholecats: the gold document's category whole + the K smallest non-gold
    #            categories, also whole
    # NOT LAUNCHED -- held until the frozen-model probe reports eval-loss parity for the rule. The
    # real-token fraction is data-dependent; the trainer prints it every 50 steps as [cat-keep].
    "sc3": ("--st-keep-mode smallcat_keep --st-keep-smallcat 3 --st-keep-decoy-cats 1 "
            "--st-header-stop-id 5491 --st-header-stop-count 1 "
            f"--st-slot-mode cent_cmean --st-slot-tokenizer {WEKA}/hf_tokenizers/Qwen3.5-0.8B-Base"),
    "sc5": ("--st-keep-mode smallcat_keep --st-keep-smallcat 5 --st-keep-decoy-cats 1 "
            "--st-header-stop-id 5491 --st-header-stop-count 1 "
            f"--st-slot-mode cent_cmean --st-slot-tokenizer {WEKA}/hf_tokenizers/Qwen3.5-0.8B-Base"),
    "gw3": ("--st-keep-mode gold_plus_wholecats --st-keep-cats 3 "
            "--st-header-stop-id 5491 --st-header-stop-count 1 "
            f"--st-slot-mode cent_cmean --st-slot-tokenizer {WEKA}/hf_tokenizers/Qwen3.5-0.8B-Base"),
    "gw5": ("--st-keep-mode gold_plus_wholecats --st-keep-cats 5 "
            "--st-header-stop-id 5491 --st-header-stop-count 1 "
            f"--st-slot-mode cent_cmean --st-slot-tokenizer {WEKA}/hf_tokenizers/Qwen3.5-0.8B-Base"),
    # Outlier iteration 6 (2026-09-15): cc00's exact recipe (gold-blind, keep 0.0, header real,
    # cent_cmean) PLUS --st-header-extra-tokens K -- the first K real BODY tokens after the header
    # stay real too, for every doc, pooled or not (src/olmo_core/nn/attention/chunked_mask.py
    # mark_doc_headers_free). cc00 already fixed the slot's readability (0.239 -> 0.482 at 4M); this
    # asks whether a few tokens of real body TEXT (not just a better slot) does more -- the xhdr
    # collapse (xhdr_collapse_diagnosis.md) is a model choosing "sample an id from the visible list"
    # over reading content, and a few real body tokens per doc give it something to actually read
    # instead of only a header-visible id. ck08/ck16/ck32 = K in {8, 16, 32}.
    "ck08": ("--st-gold-blind --st-keep-prob 0.0 --st-header-stop-id 5491 --st-header-stop-count 1 "
             "--st-header-extra-tokens 8 "
             f"--st-slot-mode cent_cmean --st-slot-tokenizer {WEKA}/hf_tokenizers/Qwen3.5-0.8B-Base"),
    "ck16": ("--st-gold-blind --st-keep-prob 0.0 --st-header-stop-id 5491 --st-header-stop-count 1 "
             "--st-header-extra-tokens 16 "
             f"--st-slot-mode cent_cmean --st-slot-tokenizer {WEKA}/hf_tokenizers/Qwen3.5-0.8B-Base"),
    "ck32": ("--st-gold-blind --st-keep-prob 0.0 --st-header-stop-id 5491 --st-header-stop-count 1 "
             "--st-header-extra-tokens 32 "
             f"--st-slot-mode cent_cmean --st-slot-tokenizer {WEKA}/hf_tokenizers/Qwen3.5-0.8B-Base"),
    "ck64": ("--st-gold-blind --st-keep-prob 0.0 --st-header-stop-id 5491 --st-header-stop-count 1 "
             "--st-header-extra-tokens 64 "
             f"--st-slot-mode cent_cmean --st-slot-tokenizer {WEKA}/hf_tokenizers/Qwen3.5-0.8B-Base"),
    # flk32/flk64 (outlier, 2026-09-15, commits 774ac6682/b94958ac7): cc00's exact recipe (gold-blind,
    # keep 0.0, header real, cent_cmean slot) but swap ck32/ck64's --st-header-extra-tokens (first K
    # body tokens) for --st-keep-token-rule first_last --st-keep-token-k K (first K//2 + last K-K//2
    # body tokens) -- the trainer refuses the two together (mutually exclusive: "first" IS
    # --st-header-extra-tokens). Motivation: the frozen-model probe (b94958ac7 commit message,
    # records/outlier-realtoken-parity-probe.md) shows first_last beats first at every rung the ck
    # ladder is built on (32k dCE +0.170 vs +0.186, genF1 0.800 vs 0.787) because a document's last
    # tokens carry its conclusion, which `first` never sees. Same K/budget grid as ck32/ck64 so the
    # pair is comparable rung for rung.
    "flk32": ("--st-gold-blind --st-keep-prob 0.0 --st-header-stop-id 5491 --st-header-stop-count 1 "
              "--st-keep-token-rule first_last --st-keep-token-k 32 "
              f"--st-slot-mode cent_cmean --st-slot-tokenizer {WEKA}/hf_tokenizers/Qwen3.5-0.8B-Base"),
    "flk64": ("--st-gold-blind --st-keep-prob 0.0 --st-header-stop-id 5491 --st-header-stop-count 1 "
              "--st-keep-token-rule first_last --st-keep-token-k 64 "
              f"--st-slot-mode cent_cmean --st-slot-tokenizer {WEKA}/hf_tokenizers/Qwen3.5-0.8B-Base"),
    # chk32/chk64 (contradiction) and nqk32/nqk64 (nq), 2026-09-15: apply ck32/ck64's recipe --
    # gold-blind, keep 0.0 (maximal compression), cent_cmean slot, + --st-header-extra-tokens K real
    # BODY tokens per doc -- to the other two tasks, to see whether it beats their current
    # matched-FLOP frontiers (contradiction hdr33 ~1.7x, nq kv33 ~2.4x, records/ds64-handoff.md).
    #
    # Contradiction renders `\n\nClaim N:` (stop id 25, ':', same as hdr33/hdr17/hdr08's _HDR1) --
    # unlike those arms, chk32/chk64 are GOLD-BLIND (no doc forced real), which is exactly the
    # combination mark_doc_headers_free's docstring warns against ONLY where the answer is drawn
    # FROM the header id (outlier/nq/rerank). Contradiction's answer is drawn from claim CONTENT
    # (which claim contradicts which), not from a header id alone, and this is the same rescue that
    # worked for outlier: xhdr00 (header + gold-blind + keep 0, NO extra tokens) collapsed to the
    # k/n floor, but ck32/ck64 (same flags + 32/64 real body tokens) reached Pareto dominance -- the
    # real body text gives the model something to read instead of only a copyable id. This is a
    # deliberate test of whether the same rescue generalises, not an oversight of the warning.
    "chk32": ("--st-gold-blind --st-keep-prob 0.0 --st-header-stop-id 25 --st-header-stop-count 1 "
              "--st-header-extra-tokens 32 "
              f"--st-slot-mode cent_cmean --st-slot-tokenizer {WEKA}/hf_tokenizers/Qwen3.5-0.8B-Base"),
    "chk16": ("--st-gold-blind --st-keep-prob 0.0 --st-header-stop-id 25 --st-header-stop-count 1 "
              "--st-header-extra-tokens 16 "
              f"--st-slot-mode cent_cmean --st-slot-tokenizer {WEKA}/hf_tokenizers/Qwen3.5-0.8B-Base"),
    "chk64": ("--st-gold-blind --st-keep-prob 0.0 --st-header-stop-id 25 --st-header-stop-count 1 "
              "--st-header-extra-tokens 64 "
              f"--st-slot-mode cent_cmean --st-slot-tokenizer {WEKA}/hf_tokenizers/Qwen3.5-0.8B-Base"),
    # SLOT-LESS twins (2026-09-22): the eval-side dev-loss grid found gold_fl20p8_noslot ==
    # gold_fl20p8 on every cell -- the slot token contributes nothing at eval time on a frozen
    # model. --st-drop-slots emits NO slot for a pooled document (its tokens vanish; kept tokens
    # keep their original positions), so there is no slot K/V or GDN write to detach either.
    # Same flags as the frontier arms otherwise, so the pair isolates the slot.
    "chk32ns": ("--st-gold-blind --st-keep-prob 0.0 --st-header-stop-id 25 --st-header-stop-count 1 "
                "--st-header-extra-tokens 32 --st-drop-slots "
                f"--st-slot-mode cent_cmean --st-slot-tokenizer {WEKA}/hf_tokenizers/Qwen3.5-0.8B-Base"),
    "flk64ns": ("--st-gold-blind --st-keep-prob 0.0 --st-header-stop-id 5491 --st-header-stop-count 1 "
                "--st-keep-token-rule first_last --st-keep-token-k 64 --st-drop-slots "
                f"--st-slot-mode cent_cmean --st-slot-tokenizer {WEKA}/hf_tokenizers/Qwen3.5-0.8B-Base"),
    # nq renders `Document (Title: {title}): {text}` with NO numeric/bracketed id
    # (src/scripts/data/convert_rag_tasks_to_sft.py, PASSAGE_TEMPLATE) -- unlike outlier's
    # `Document [N]:` and contradiction's `Claim N:`, there is no header token that names a
    # copyable id, so there is no stop_id to free and no id-guessing shortcut to rescue. Per
    # mark_doc_headers_free's docstring, `stop_id=None` treats the header as zero-length and
    # --st-header-extra-tokens counts from the document's own first token after <|doc_start|> --
    # so nqk32/nqk64 omit --st-header-stop-id/--st-header-stop-count entirely and keep only the
    # first 32/64 tokens of each doc's own text (title + body) real, same slot/keep-prob otherwise.
    "nqk32": ("--st-gold-blind --st-keep-prob 0.0 --st-header-extra-tokens 32 "
              f"--st-slot-mode cent_cmean --st-slot-tokenizer {WEKA}/hf_tokenizers/Qwen3.5-0.8B-Base"),
    "nqk64": ("--st-gold-blind --st-keep-prob 0.0 --st-header-extra-tokens 64 "
              f"--st-slot-mode cent_cmean --st-slot-tokenizer {WEKA}/hf_tokenizers/Qwen3.5-0.8B-Base"),
    # Outlier iteration 7 (2026-09-15): ck08/ck16 with the SAME budget spent on a different K.
    # --st-keep-token-rule rule scores every body token with a linear rule over features that need
    # no model forward (idf from the training shard, relative position in the body, first-sentence,
    # capitalised, digit, piece length) and keeps the top K PER DOCUMENT, at their ORIGINAL
    # positions -- so a pooled doc reads "Document [N]: <scattered real tokens><SLOT>" instead of
    # "Document [N]: <first K tokens><SLOT>". Ported from the eval-side probe's rule{k}
    # (records/outlier-saliency-preview-probe.md 5d), where at K=8 rand 0.558 > cc00 0.470 >
    # first 0.263 >> rule 0.071 ~= the gradient oracle 0.072: which tokens are kept is the effect.
    # ⚠ The weights are the documented PLACEHOLDER (olmo_core.nn.pooled_soft_token
    # .DEFAULT_KEEP_TOKEN_WEIGHTS -- signs from the measured within-document gradient profile of
    # probe record 5a(c)), NOT the probe's fitted ridge, which is not reachable from this repo.
    # Pass the fitted vector with --st-keep-token-weights '<json>' once it is harvested.
    "rk08": ("--st-gold-blind --st-keep-prob 0.0 --st-header-stop-id 5491 --st-header-stop-count 1 "
             "--st-keep-token-rule rule --st-keep-token-k 8 "
             f"--st-slot-mode cent_cmean --st-slot-tokenizer {WEKA}/hf_tokenizers/Qwen3.5-0.8B-Base"),
    "rk16": ("--st-gold-blind --st-keep-prob 0.0 --st-header-stop-id 5491 --st-header-stop-count 1 "
             "--st-keep-token-rule rule --st-keep-token-k 16 "
             f"--st-slot-mode cent_cmean --st-slot-tokenizer {WEKA}/hf_tokenizers/Qwen3.5-0.8B-Base"),
    # Outlier iteration 4 (2026-09-14, gen-4b). The xhdr verdict above says header-real + gold-blind
    # hands the model a copyable-but-unjustifiable target. That diagnosis is consistent with the
    # data but does NOT separate two stories:
    #   (a) HARD LEAK -- once every id is real text, the guess policy is the loss minimum and no
    #       amount of real body text or compute can beat it;
    #   (b) OPTIMIZATION SHORTCUT -- the guess policy is merely reachable in a handful of steps
    #       (every xhdr CE curve flattens by step 8), so SGD lands in it first and never explores
    #       the content-grounded basin. Without the header, a pooled doc's id is absent, copying is
    #       impossible, and kvgb50 was forced onto content -- reaching dense parity.
    # The mean-embedding slot should still carry TOPIC, which is all outlier needs, so (b) is live.
    # Two arms separate them; both keep the xhdr flags otherwise byte-identical.
    #   xh2k50: header + keep 1/2. Direct test of "does the header ALWAYS collapse, or only at low
    #     keep?" Its header-free twin kvgb50 (same keep, no header) is the parity arm, so a collapse
    #     here isolates the header and a descent isolates the keep ratio. Under (a) this must still
    #     sit on k/n; under (b) half the bodies real should be enough to find the content basin.
    #   xh2warm17: header + keep 1/6, WARM-STARTED from the finished kvgb50-b128f3-u16M checkpoint
    #     (ARM_BASE below) instead of the raw base. The model already reads documents when the
    #     header appears. Under (a) it abandons that and falls back to guessing; under (b) it keeps
    #     descending. Keep 1/6 is the harshest xhdr cell, so a descent there is decisive.
    #     FLOP ACCOUNTING: total budget = 16M (the kvgb50 warm start) + this arm's budget. The log's
    #     flop_meter charges only THIS run, and collect_ds64.py takes task/arm/budget from the
    #     orchestrator state (not from the run name), so nothing breaks -- but every matched-FLOP
    #     quote must add kvgb50-16M's 190.3 PF by hand. Its 16M rung also re-reads the same shard
    #     kvgb50 already saw (the arms are nested prefixes); that is a second epoch, not new data.
    "xh2k50": "--st-gold-blind --st-keep-prob 0.5 --st-header-stop-id 5491 --st-header-stop-count 1",
    "xh2warm17": ("--st-gold-blind --st-keep-prob 0.1667 "
                  "--st-header-stop-id 5491 --st-header-stop-count 1"),
    # Outlier iteration 5 (2026-09-14, gen-4c). THE CURRICULUM NOW ACTUALLY RUNS. Until commit
    # 7ce879b40 the --st-mix-* flags were only wired into make_fingerprint_keep_docs_fn, which
    # train_ctc_suite.py installs only when the run is NOT --st-gold-blind -- so every gold-blind
    # arm accepted them and ignored them, and `kvgbmix` above was a byte-identical re-run of
    # `kvgb` (records/ds64-handoff.md section 9). The trainer now applies p_full to the gold-blind
    # keep draw itself and REFUSES the flags anywhere they would be ignored.
    #   xh2mix17: the third xh2 arm section 9 could not launch -- header + keep 1/6, but with a
    #     fraction of rows trained UNCOMPRESSED, annealed 0.5 -> 0 over the first half. If xhdr's
    #     collapse is an optimization shortcut (story (b)), full rows are exactly the signal that
    #     keeps SGD out of the "sample 3 ids from the visible list" basin long enough to find the
    #     content-grounded one; if it is a hard leak (story (a)) the CE still plateaus at ~0.48.
    #   kvgbmix2: the control, and the arm `kvgbmix` was SUPPOSED to be -- gold-blind, NO header,
    #     same curriculum. Its twin kvgb50 (keep 1/2, no mixing) is the parity arm, so this
    #     isolates what the curriculum alone buys on an arm that already works.
    # Proof the curriculum is live: the job log prints `[pooled-kv] call#N gold-blind: ... p_full=`
    # lines. kvgbmix printed none -- that is the signature of the old silent no-op.
    "xh2mix17": ("--st-gold-blind --st-keep-prob 0.1667 "
                 "--st-header-stop-id 5491 --st-header-stop-count 1 "
                 "--st-mix-start-p 0.5 --st-mix-end-p 0.0 --st-mix-anneal-frac 0.5"),
    "kvgbmix2": ("--st-gold-blind --st-keep-prob 0.5 "
                 "--st-mix-start-p 0.5 --st-mix-end-p 0.0 --st-mix-anneal-frac 0.5"),
    "kvrb": "--st-n-random-range 128,512 --st-keep-mode gold_plus_random",          # v25 recipe
    "kvmix": ("--st-keep-frac 0.3333 --st-keep-mode gold_plus_random "
              "--st-mix-start-p 0.5 --st-mix-end-p 0.0 --st-mix-anneal-frac 0.5"),  # mixing curriculum
    # nq / outlier: gold + a fraction of random docs
    "kv08": "--st-keep-frac 0.0833 --st-keep-mode gold_plus_random",
    "kv17": "--st-keep-frac 0.1667 --st-keep-mode gold_plus_random",
    "kv33": "--st-keep-frac 0.3333 --st-keep-mode gold_plus_random",
    # nq iteration 2 (2026-09-15 overnight): kv17/kv08's exact flags plus the content-only,
    # row-centred slot (--st-slot-mode cent_cmean). nq has no rendered header (unlike outlier's
    # `Document [N]:` / oolong's `Date: ... || User: ...`), so there is no header-real twin here --
    # these are the plain gold_plus_random arms with the richer slot swapped in.
    #   nqcc17: same keep (1/6) as the already-run kv17 rows (16M 0.7068@115PF, 32M 0.7828@231PF,
    #     64M 0.8276@443PF, 128M 0.8384@916PF vs dense 0.8436@683/0.8652@1387/... -- the plain-slot
    #     control) -- same-FLOP comparison, does the readable slot close the gap to dense?
    #   nqcc08: keep 1/12 -- one step lower than any plain-slot nq arm on record, only worth it if
    #     nqcc17 shows the slot is buying something.
    "nqcc17": ("--st-keep-frac 0.1667 --st-keep-mode gold_plus_random "
               f"--st-slot-mode cent_cmean --st-slot-tokenizer {WEKA}/hf_tokenizers/Qwen3.5-0.8B-Base"),
    "nqcc08": ("--st-keep-frac 0.0833 --st-keep-mode gold_plus_random "
               f"--st-slot-mode cent_cmean --st-slot-tokenizer {WEKA}/hf_tokenizers/Qwen3.5-0.8B-Base"),
    # diagnostics (2026-09-08 23:30, why is a soft step ~65 s?): same as hdr03 but gold-blind (no
    # fingerprint keep hook) / one row per micro-step
    "hdr03gb": "--st-gold-blind --st-keep-prob 0.0278 --st-header-stop-id 25 --st-header-stop-count 1",
    "hdr03m1": f"--st-keep-frac 0.0278 {_HDR1}",
    "hdr03prof": f"--st-keep-frac 0.0278 {_HDR1} --torch-profile",  # prints the top ops (steps 3-4) in the job log
}
# Phase 1 (contradiction first): dense + the keep ablation. Other tasks: dense only until
# debug/ds64/soft_arms.json (read by the orchestrator every cycle) names their soft arms.
# Phase 2 (gen 3, after the 16M keep ablation: only keep >= 1/3 transfers; 1/6 for the trade-off
# curve): the wall-clock-faithful recipe (flash, 128 rows/step, length-sorted micro-batches).
TASK_ARMS = {"contradiction": ["dense", "hdr33", "hdr17"],
             "oolong": ["dense"], "nq": ["dense"], "outlier": ["dense"]}


def run_name(task, arm, budget):
    # soft arms carry rows/step + backend (gb16 first launch and the torch -b128 launch stay distinct)
    tag = "" if arm == "dense" else f"-b{SOFT_GB}{'f' if SOFT_BACKEND.startswith('flash') else ''}{os.environ.get('DS64_GEN', '')}"
    return f"ds64{'' if SCALE == '4b' else '-' + SCALE}-{task}-{arm}{tag}-u{budget}"


# Budgets below the smallest shard we built (16M) are trained ON the 16M shard, stopped by
# --max-tokens. The trainer treats that as a real budget -- warmup fraction and linear decay are
# relative to max_duration -- so it is a clean data-scaling point, unlike --max-steps which
# hard-stops mid-schedule. Legitimate because compose_uniform_arms.py emits NESTED PREFIXES: a 4M
# slice of the 16M arm carries the same short-heavy length mix.
SHARD_FLOOR = "16M"


def _tokens(b):
    return int(b.rstrip("Mm")) * 1_000_000


def shard_for(budget):
    """(shard budget to read, --max-tokens cap or None) for a requested training budget."""
    return (SHARD_FLOOR, _tokens(budget)) if _tokens(budget) < _tokens(SHARD_FLOOR) else (budget, None)


# WARM-START arms: train from a FINISHED ds64 run's model-only checkpoint instead of the raw base.
# train_ctc_suite.py writes `<save_folder>/model_and_optim` at the end of every run (and refuses a
# --base-checkpoint without a `.metadata` inside), and _tolerant_base_load restores every key
# including `pooled_projector.*`, so the projector is warm too. Only the 4B base layout is assumed.
ARM_BASE = {
    # kvgb50 = gold-blind, keep 1/2, NO header -- the only outlier soft arm at dense parity.
    # Starting xh2warm17 there asks whether a model that already reads document CONTENT abandons
    # that for the id-guessing policy the moment headers become real.
    "xh2warm17": f"{WEKA}/ctc_suite/ckpts/ds64-outlier-kvgb50-b128f3-u16M/model_and_optim",
}


def arm_args(task, arm, budget):
    shard, cap = shard_for(budget)
    data = f"{SHARDS}/{task}_u{shard}"
    cap_arg = f" --max-tokens {cap}" if cap else ""
    base = ARM_BASE.get(arm, BASE)
    if arm == "dense":
        return "full", data, ["--pack", "--seq-len", "65536", "--global-batch", "8", "--micro-batch-instances", "1", "--base-checkpoint", base], cap_arg.strip()
    if arm in ARM_EXTRA:
        micro = max(1, ARM_MICRO[arm] // (2 if SCALE == "27b" else 1))
        return "softtoken", data, ["--seq-len", "65536", "--global-batch", str(SOFT_GB), "--micro-batch-instances", str(micro), "--base-checkpoint", base], \
            f"{ARM_EXTRA[arm]} --attn-backend {SOFT_BACKEND}{cap_arg}"
    raise SystemExit(f"unknown arm {arm}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", default=",".join(TASKS))
    ap.add_argument("--budgets", default=",".join(BUDGETS))
    ap.add_argument("--arms", default="", help="comma list; default = the task's arms")
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--wandb-group", default=f"ds64-q35-{SCALE}")
    ap.add_argument("--skip", default="")
    ap.add_argument("mode", choices=["launch", "dry_run"])
    args = ap.parse_args()
    skip = set(x for x in args.skip.split(",") if x)
    rows = []
    for task in args.tasks.split(","):
        for arm in (args.arms.split(",") if args.arms else TASK_ARMS[task]):
            for budget in (args.budgets.split(",") if args.budgets != ",".join(BUDGETS) else budgets_for(task, arm)):
                variant, data, largs, extra = arm_args(task, arm, budget)
                name = run_name(task, arm, budget)
                if name in skip:
                    print(f"[skip] {name}"); continue
                cmd = [sys.executable, "-u", LAUNCHER, "--task", task, "--variant", variant,
                       "--model-family", "qwen3_5", "--model-scale", SCALE, "--data-root", data,
                       "--run-name", name, "--exact-run-name", "--num-nodes", str(NODES), "--num-gpus", str(GPUS),
                       "--epochs", "1", "--lr", str(args.lr), "--cluster", CLUSTER, "--wandb-group", args.wandb_group,
                       "--no-follow", "--no-compile"] + largs + (["--extra-args", extra] if extra else []) + [args.mode]
                print(" ".join(cmd), flush=True)
                res = subprocess.run(cmd, cwd=REPO, env=dict(os.environ, PYTHONPATH=f"{REPO}/src"), capture_output=True, text=True)
                out = res.stdout + res.stderr
                os.makedirs(f"{REPO}/debug/ds64/launch_logs", exist_ok=True)
                open(f"{REPO}/debug/ds64/launch_logs/{name}.{args.mode}.log", "w").write(out)
                ids = [l.split("id=")[1].split()[0] for l in out.splitlines() if "SUBMITTED id=" in l]
                tail = (ids[-1] + " " if ids else "") + "\n".join(out.strip().splitlines()[-2:]).replace("\t", " ")[:200]
                print(f"  -> rc={res.returncode}: {tail[:300]}", flush=True)
                rows.append((name, task, arm, budget, res.returncode, tail))
    if args.mode == "launch":
        with open(LEDGER, "a") as f:
            w = csv.writer(f, delimiter="\t")
            for name, task, arm, budget, rc, tail in rows:
                w.writerow(["ds64", SCALE, task, arm, budget, name, CLUSTER, dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
                            "LAUNCHED" if rc == 0 else "LAUNCH-FAILED", tail])


if __name__ == "__main__":
    main()
