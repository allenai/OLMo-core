"""
Qwen3.5-4B CPT at 256k on the longmino-512k mix -- interleaved landmark arm ``sparse-reg``.

ALTERNATING sparse:reg starting with SPARSE -- layers 3, 11, 19, 27 sparse and 7, 15, 23, 31
regular landmark. The model's final layer is regular landmark.

Half the full-attention layers pay the quadratic cost. Against ``reg-sparse`` (the same 4/4
split in the opposite phase) this isolates *phase* from *count*; against ``reg-first`` /
``reg-last`` it prices what the extra three regular layers buy.

All shared configuration -- landmark geometry, the CP<=4 / TP=1 / no-compile constraints, the data
mix, the budget, and how the per-layer pattern is built -- lives in
``_qwen35_interleaved_landmark_256k_common.py``. This file only selects the arm.

    PYTHONPATH=src python src/scripts/train/memexpress/cpt/interleaved/Qwen3.5-4B-interleaved-sparse-reg-256k.py \\
        dry_run q35-4b-il-sparsereg-256k ai2/jupiter-cirrascale-2

    PYTHONPATH=src python src/scripts/train/memexpress/cpt/interleaved/Qwen3.5-4B-interleaved-sparse-reg-256k.py \\
        launch q35-4b-il-sparsereg-256k ai2/jupiter-cirrascale-2 \\
        --launch.follow=false --launch.step_soft_timeout=null
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _qwen35_interleaved_landmark_256k_common import (  # noqa: E402
    build_qwen35_interleaved_experiment,
)

from olmo_core.internal.experiment import CliContext, ExperimentConfig, main  # noqa: E402

ARM = "sparse-reg"


def build_experiment_config(cli_context: CliContext) -> ExperimentConfig:
    """
    Build the ``sparse-reg`` arm's experiment config.

    :param cli_context: The CLI context supplied by :func:`olmo_core.internal.experiment.main`.

    :returns: The full experiment config.
    """
    return build_qwen35_interleaved_experiment(cli_context, arm=ARM)


if __name__ == "__main__":
    main(config_builder=build_experiment_config)
