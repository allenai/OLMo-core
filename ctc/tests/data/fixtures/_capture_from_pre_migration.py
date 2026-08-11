"""
Capture golden fixtures from the PRE-MIGRATION generators. Not part of the test run.

The fixtures this writes are the ground truth that ``test_synthetic_parity.py`` checks the ported
generators against, so they are captured from the old tree *before* the port is written -- a
reference derived from the new code would only prove the new code agrees with itself.

Each fixture drives the old ``build_example`` directly from a fresh ``random.Random(seed)``, rather
than running the old CLI. That isolates the part that must not change (how one example is
constructed) from the part the port deliberately does change (how train and eval splits draw from
the RNG -- see ``ctc.data.build``). A file-level diff would conflate the two and fail for the wrong
reason.

Run from the pre-migration checkout::

    cd /path/to/OLMo-core            # branch prasann/landmark, tag pre-migration-source
    PYTHONPATH=src python /path/to/newolmocore/OLMo-core/ctc/tests/data/fixtures/\\
        _capture_from_pre_migration.py --out /path/to/newolmocore/OLMo-core/ctc/tests/data/fixtures

Regenerating a fixture to make a failing test pass deletes the evidence instead of the bug. The
only legitimate reason to re-run this is adding a new task or a new configuration.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from argparse import Namespace
from pathlib import Path
from typing import Callable, Dict, List

#: task -> list of (config-name, kwargs for the old build_example). Chosen to cover both branches of
#: every knob that changes the RNG stream: numbers-only rendering, multi-group golds, the
#: ``separation`` margin, and a fixed vs. per-example-random feature.
CONFIGS: Dict[str, List[Dict]] = {
    "cycle": [
        {"_name": "n24_len3_k1", "num_docs": 24, "cycle_len": 3, "num_cycles": 1},
        {"_name": "n40_len4_k2", "num_docs": 40, "cycle_len": 4, "num_cycles": 2},
    ],
    "groups4": [
        {
            "_name": "n20_g4_k1_x5",
            "num_docs": 20,
            "num_groups": 1,
            "group_size": 4,
            "tolerance": 5,
            "ans_min": -500,
            "ans_max": 500,
            "numbers_only": False,
        },
        {
            "_name": "n16_g3_k2_x4_numsonly",
            "num_docs": 16,
            "num_groups": 2,
            "group_size": 3,
            "tolerance": 4,
            "ans_min": -500,
            "ans_max": 500,
            "numbers_only": True,
        },
    ],
    "mathmatch": [
        {
            "_name": "n20_k3_x2",
            "n_docs": 20,
            "n_pairs": 3,
            "tol": 2,
            "ans_min": -50,
            "ans_max": 50,
            "numbers_only": False,
        },
        {
            "_name": "n24_k2_x1_numsonly",
            "n_docs": 24,
            "n_pairs": 2,
            "tol": 1,
            "ans_min": -50,
            "ans_max": 50,
            "numbers_only": True,
        },
    ],
    "textgroups": [
        {
            "_name": "n15_g3_k2_t70_mixed",
            "num_docs": 15,
            "num_groups": 2,
            "group_size": 3,
            "target": 70,
            "tolerance": 0,
            "separation": 0,
            "feature": "mixed",
            "cmin": 4,
            "cmax": 40,
            "filler_max": None,
        },
        {
            # The 'mixed' case above happens to draw connector/connector/verbs, leaving the
            # adjective clause builder -- which draws its lexicon in a different order from the
            # others -- with no coverage. Pin it explicitly.
            "_name": "n12_g3_k1_t55_adjectives",
            "num_docs": 12,
            "num_groups": 1,
            "group_size": 3,
            "target": 55,
            "tolerance": 0,
            "separation": 0,
            "feature": "adjectives",
            "cmin": 4,
            "cmax": 40,
            "filler_max": None,
        },
        {
            "_name": "n12_g3_k1_t60_nouns_sep5",
            "num_docs": 12,
            "num_groups": 1,
            "group_size": 3,
            "target": 60,
            "tolerance": 0,
            "separation": 5,
            "feature": "nouns",
            "cmin": 4,
            "cmax": 40,
            "filler_max": None,
        },
    ],
}

SEED = 1234
N_EXAMPLES = 3


def _builders() -> Dict[str, Callable]:
    """
    :returns: task -> a ``(config, rng) -> example`` adapter over the old generator.

    :raises ImportError: If the pre-migration generators are not on the path.
    """
    import generate_cycle_data as old_cycle
    import generate_groups4_data as old_groups4
    import generate_mathmatch_data as old_mathmatch
    import generate_textgroups_data as old_textgroups

    return {
        "cycle": lambda c, rng: old_cycle.build_example(
            c["num_docs"], c["cycle_len"], c["num_cycles"], rng
        ),
        "groups4": lambda c, rng: old_groups4.build_example(Namespace(**c), rng),
        "textgroups": lambda c, rng: old_textgroups.build_example(Namespace(**c), rng),
        "mathmatch": lambda c, rng: old_mathmatch.build_example(
            c["n_docs"],
            c["n_pairs"],
            c["tol"],
            c["ans_min"],
            c["ans_max"],
            rng,
            numbers_only=c["numbers_only"],
        ),
    }


def main() -> int:
    """
    :returns: Process exit status.
    """
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, help="fixture directory to write into")
    ap.add_argument("--data-dir", default="src/corpus_reasoning/data", help="old generator dir")
    args = ap.parse_args()

    sys.path.insert(0, args.data_dir)
    builders = _builders()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for task, configs in CONFIGS.items():
        build = builders[task]
        cases = []
        for config in configs:
            payload = {k: v for k, v in config.items() if k != "_name"}
            rng = random.Random(SEED)
            cases.append(
                {
                    "name": config["_name"],
                    "config": payload,
                    "examples": [build(payload, rng) for _ in range(N_EXAMPLES)],
                }
            )
        path = out / f"{task}_golden.json"
        path.write_text(
            json.dumps({"seed": SEED, "source": "pre-migration-source", "cases": cases}, indent=1),
            encoding="utf-8",
        )
        print(f"{task}: {len(cases)} case(s) -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
