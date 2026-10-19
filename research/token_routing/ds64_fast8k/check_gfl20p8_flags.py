"""CPU check: the fast8k `gfl20p8ns` arm trains the devloss grid's `gold_fl20p8_noslot` construction.

    PYTHONPATH=src:recipes/train/research_trainer:research/token_routing/devloss_grid \
        $PY research/token_routing/ds64_fast8k/check_gfl20p8_flags.py

Both sides drive the same ``Transformer._compact_pooled_soft_tokens`` path through the same config
keys (the trainer via ``recipes/train/variants.py`` -> ``enable_pooled_soft_tokens``, the grid via
``ctc_devloss_grid.configure_scheme``), so construction equality reduces to option equality:

1. ``train_ctc_suite.parse_args`` ACCEPTS the arm's flags (an 8-token prefix combined with
   ``first_last``) and still REJECTS ``--st-keep-token-rule first`` + ``--st-header-extra-tokens``.
2. The parsed options equal the grid scheme field by field: gold kept with no random extras,
   prefix 8, first_last at a 0.2 fraction, no slot, no header stop id (header cap 32 on both sides).
"""

import shlex
import sys

import ctc_devloss_grid as G
import train_ctc_suite as T

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import launch_fast8k as L  # noqa: E402

BASE = ["train_ctc_suite.py", "--task", "contradiction", "--data", "/nonexistent", "--variant", "softtoken"]


def parse(extra: str):
    sys.argv = BASE + shlex.split(extra)
    return T.parse_args()


def main() -> None:
    arm = L.ARM_EXTRA["gfl20p8ns"]
    o = parse(arm)
    s = G.SCHEMES["gold_fl20p8_noslot"]
    checks = {
        "gold kept (gold_plus_random)": (o.st_keep_mode == "gold_plus_random") == (s["keep"] == "gold"),
        "no random extra docs": o.st_keep_frac == s["frac"] == 0.0,
        "not gold-blind": not o.st_gold_blind,
        "8-token prefix": o.st_header_extra_tokens == s["extra"] == 8,
        "first_last rule": o.st_keep_token_rule == s["rule"] == "first_last",
        "20% fraction": o.st_keep_token_k == s["k"] == 0.2,
        "no slot": o.st_drop_slots is True and s["noslot"] is True,
        "no header stop id": o.st_header_stop_id is None,
    }
    for name, ok in checks.items():
        print(f"[{'ok' if ok else 'FAIL'}] {name}")
    try:
        parse("--st-keep-token-rule first --st-keep-token-k 8 --st-header-extra-tokens 8")
        print("[FAIL] `first` + prefix was accepted")
        bad = True
    except SystemExit:
        print("[ok] `first` + prefix still rejected")
        bad = False
    if bad or not all(checks.values()):
        raise SystemExit(1)
    print("PASS: gfl20p8ns == gold_fl20p8_noslot")


if __name__ == "__main__":
    main()
