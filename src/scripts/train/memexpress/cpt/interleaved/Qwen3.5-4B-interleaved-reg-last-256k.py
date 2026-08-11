"""
Qwen3.5-4B CPT at 256k on the longmino-512k mix -- interleaved landmark arm ``reg-last``.

REGULAR landmark on the LAST full-attention layer only (global layer 31, also the model's
final layer); the other seven (3, 7, 11, 15, 19, 23, 27) use SPARSE landmark attention.

The positional mirror of ``reg-first``: one quadratic landmark layer, but at the *top* of the
stack, where it can gather full context over already-composed representations. Comparing the
two isolates depth placement at a fixed 1/7 budget.

All shared configuration -- landmark geometry, the CP<=4 / TP=1 / no-compile constraints, the data
mix, the budget, and how the per-layer pattern is built -- lives in
``_qwen35_interleaved_landmark_256k_common.py``. This file only selects the arm.

    PYTHONPATH=src python src/scripts/train/memexpress/cpt/interleaved/Qwen3.5-4B-interleaved-reg-last-256k.py \\
        dry_run q35-4b-il-reglast-256k ai2/jupiter-cirrascale-2

    PYTHONPATH=src python src/scripts/train/memexpress/cpt/interleaved/Qwen3.5-4B-interleaved-reg-last-256k.py \\
        launch q35-4b-il-reglast-256k ai2/jupiter-cirrascale-2 \\
        --launch.follow=false --launch.step_soft_timeout=null
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _qwen35_interleaved_landmark_256k_common import (  # noqa: E402
    build_qwen35_interleaved_experiment,
)

from olmo_core.internal.experiment import CliContext, ExperimentConfig, main  # noqa: E402

ARM = "reg-last"


def build_experiment_config(cli_context: CliContext) -> ExperimentConfig:
    """
    Build the ``reg-last`` arm's experiment config.

    :param cli_context: The CLI context supplied by :func:`olmo_core.internal.experiment.main`.

    :returns: The full experiment config.
    """
    return build_qwen35_interleaved_experiment(cli_context, arm=ARM)


if __name__ == "__main__":
    main(config_builder=build_experiment_config)
