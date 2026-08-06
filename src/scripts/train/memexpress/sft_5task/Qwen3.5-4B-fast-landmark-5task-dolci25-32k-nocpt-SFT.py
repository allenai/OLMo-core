"""
Qwen3.5-4B FAST LANDMARK attention on the full-attention layers -- 32k-scale SFT on 75% the 5 long-context tasks (contradiction, nq, oolong,
rerank, outlier) / 25% ``allenai/Dolci-Instruct-SFT``, no CPT text, from the
``q35-4b-fastlm`` CPT checkpoint at step2385.

All shared configuration lives in ``_qwen35_5task_dolci25_32k_nocpt_common.py`` -- see its module
docstring for the data provenance, the no-CP/GDN constraints, and the per-arm budgets. This file
only selects the arm.

    PYTHONPATH=src python src/scripts/train/memexpress/sft_5task/Qwen3.5-4B-fast-landmark-5task-dolci25-32k-nocpt-SFT.py \\
        dry_run q35-4b-fastlm-5task-dolci25-32k-nocpt ai2/jupiter
    PYTHONPATH=src python src/scripts/train/memexpress/sft_5task/Qwen3.5-4B-fast-landmark-5task-dolci25-32k-nocpt-SFT.py \\
        launch  q35-4b-fastlm-5task-dolci25-32k-nocpt ai2/jupiter --launch.num_nodes=1
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _qwen35_5task_dolci25_32k_nocpt_common import (  # noqa: E402
    build_qwen35_sft_experiment,
)

from olmo_core.internal.experiment import CliContext, ExperimentConfig, main  # noqa: E402

ARM = "fast-landmark"


def build_experiment_config(cli_context: CliContext) -> ExperimentConfig:
    """
    Build the fast-landmark arm's experiment config.

    :param cli_context: The CLI context supplied by :func:`olmo_core.internal.experiment.main`.

    :returns: The full experiment config.
    """
    return build_qwen35_sft_experiment(cli_context, arm=ARM)


if __name__ == "__main__":
    main(config_builder=build_experiment_config)
