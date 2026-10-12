"""
FLOP-audit for the ds64 campaign: detect (task, keep-config) pairs of soft-token arms that were
launched under DIFFERENT gpus/microbatch settings and are therefore NOT a valid matched-FLOP
comparison, even though their CLI keep-flags (--st-keep-frac/--st-keep-prob/--st-keep-mode/
--st-gold-blind/--st-header-stop-*) are identical.

Root cause (see research/token_routing/ds64/flop_audit/README.md for the full writeup): the trainer sorts each
RANK's local batch by length and splits it into microbatches of ARM_MICRO rows
(train_module.py:395-401); soft-token compaction then right-pads every row in a microbatch to
that microbatch's own longest post-compaction row (pooled_soft_token.py:273-274,
``T2 = max(len(r) for r in rows_ids)``). Fewer GPUs -> a bigger per-rank pool to sort -> tighter
length matches -> less repadding waste -> genuinely less real compute, which
FlopMeterCallback (flop_meter.py) faithfully measures. research/token_routing/ds64/launch_ds64.py's own comment
next to ``GPUS = int(os.environ.get("DS64_NGPU", "8"))`` asserts "Rows/step is unchanged, so
accuracy and FLOPs are unaffected [by gpu count] -- only the gradient-accumulation depth and
wall-clock change." That is true for DENSE arms (dense_pflops is computed from an all-reduced
REAL non-pad token count, independent of how rows are grouped into microbatches) but FALSE for
soft-token/pooled-KV compaction arms, where the per-microbatch repad-to-max makes real compute a
function of gpus x microbatch size.

This script does NOT modify results/ds64/results.csv. It only reads it (plus launch_ds64.py's
ARM_EXTRA/ARM_MICRO tables) and prints an audit table. It does not fabricate a "corrected PF" for
the confounded (coarser-batched) run in a mismatched pair -- that would require re-deriving each
row's post-compaction length, which is not retained anywhere in the logs or flops.json (only the
step-aggregated tokens_in/tokens_out/rows counters are kept, and even those are not persisted --
only the final cumulative actual_pflops/dense_pflops are). The one number this script WILL surface
as a legitimate proxy is: when slot_mode is proven FLOP-neutral (a matched-gpu/micro control pair
exists with ratio ~1.000), the finer-batched sibling's own actual_pflops is the best available
same-keep-config compute reading -- the coarser-batched run's number should not be read against it
as an ablation of slot_mode.

    python research/token_routing/ds64/flop_audit/recompute_flops.py
"""

import csv
import os
import re
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", ".."))
sys.path.insert(0, f"{REPO}/research/token_routing/ds64")
import launch_ds64 as L  # noqa: E402

RESULTS = f"{REPO}/results/ds64/results.csv"

SLOT_FLAG_RE = re.compile(r"--st-slot-mode \S+|--st-slot-tokenizer \S+")


def keep_signature(arm: str) -> str:
    """ARM_EXTRA string for ``arm`` with the slot-construction flags (--st-slot-mode/
    --st-slot-tokenizer) stripped, so two arms that differ ONLY in slot construction collapse to
    the same signature and get grouped for comparison."""
    extra = L.ARM_EXTRA.get(arm, "")
    sig = SLOT_FLAG_RE.sub("", extra)
    return " ".join(sig.split())  # normalize whitespace


def main():
    rows = list(csv.DictReader(open(RESULTS)))
    # Only soft-token/pooled-KV arms have a meaningful ARM_MICRO / gpus-dependent compaction path.
    soft_rows = [r for r in rows if r["arm"] in L.ARM_EXTRA and r.get("flops_meter")]

    # The orchestrator state can carry more than one launch attempt for the same (task, arm,
    # budget) -- e.g. a retry after a preemption -- which shows up as multiple `run` rows with the
    # same `arm`. Those are NOT a slot_mode comparison, just the same arm twice; dedupe to one row
    # per (task, budget, arm) so the audit below only ever compares DISTINCT arms. Later rows win
    # (results.csv is written in st["runs"] dict order, which reflects orchestrator insertion /
    # most-recent-relaunch order).
    dedup = {}
    for r in soft_rows:
        dedup[(r["task"], r["budget"], r["arm"])] = r
    soft_rows = list(dedup.values())

    groups = {}
    for r in soft_rows:
        sig = keep_signature(r["arm"])
        key = (r["task"], r["budget"], sig)
        groups.setdefault(key, []).append(r)
    # Only keep groups that actually span more than one DISTINCT arm -- a same-signature group of
    # one arm is nothing to compare.
    groups = {k: v for k, v in groups.items() if len({r["arm"] for r in v}) > 1}

    print(f"{'task':10} {'budget':6} {'arms (gpus/micro)':55} {'flops_meter (PF)':30} verdict")
    print("-" * 130)
    any_confounded = False
    for (task, budget, sig), grp in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        if len(grp) < 2:
            continue  # nothing to compare
        arms_desc = []
        pfs = []
        gpus_set = set()
        micro_set = set()
        for r in grp:
            arm = r["arm"]
            gpus = r.get("gpus") or "?"
            micro = L.ARM_MICRO.get(arm, "?")
            gpus_set.add(gpus)
            micro_set.add(micro)
            arms_desc.append(f"{arm}({gpus}g/{micro}m)")
            pfs.append((arm, float(r["flops_meter"])))
        matched = len(gpus_set) == 1 and len(micro_set) == 1
        pf_str = ", ".join(f"{a}={pf:.2f}" for a, pf in pfs)
        if matched:
            lo = min(pf for _, pf in pfs)
            hi = max(pf for _, pf in pfs)
            ratio = hi / lo if lo else float("nan")
            verdict = f"CLEAN (gpus/micro match; max/min ratio {ratio:.4f})"
        else:
            lo = min(pf for _, pf in pfs)
            hi = max(pf for _, pf in pfs)
            ratio = hi / lo if lo else float("nan")
            verdict = (
                f"CONFOUNDED by gpus/micro (ratio {ratio:.3f}x) -- "
                "do NOT read this as a slot_mode/method delta; needs a rerun at matched gpus+micro"
            )
            any_confounded = True
        print(f"{task:10} {budget:6} {', '.join(arms_desc):55} {pf_str:30} {verdict}")

    print()
    if any_confounded:
        print(
            "At least one pair above is confounded. Per the investigation "
            "(research/token_routing/ds64/flop_audit/README.md), the ONLY confounded pair in the current ds64 "
            "campaign is nq's kv17 (8 gpus / micro 4, launched 2026-09-09 under the pre-fragmentation "
            "default) vs nqcc17/nqcc08 (4 gpus / micro 2, launched 2026-09-15 under DS64_NGPU=4). "
            "The oolong ohdr08 vs occ08/occ00 pairs are CLEAN (both launched at 4 gpus / micro 2), "
            "which is why occ08's actual_pflops is bit-identical to ohdr08's at both 16M and 32M -- "
            "direct, uncorrected confirmation that slot_mode itself does not move the FLOP count."
        )


if __name__ == "__main__":
    main()
