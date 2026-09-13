"""Qwen3.5-4B eight-node CPT: compressive stage of the 5B + 5B curriculum.

See Qwen3.5-4B-staged-256k.md for launch and checkpoint handoff instructions.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _qwen35_sparse_to_compressive_256k_common import build_staged_experiment  # noqa: E402

from olmo_core.internal.experiment import CliContext, ExperimentConfig, main  # noqa: E402


def build_experiment_config(cli_context: CliContext) -> ExperimentConfig:
    return build_staged_experiment(cli_context, compressive=True)


if __name__ == "__main__":
    main(config_builder=build_experiment_config)
