"""Compressive vs its paired arm: contradiction-only, three epochs through 256K.

Readout and comparison caveats: _qwen35_contradiction_256k_common.py.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _qwen35_contradiction_256k_common import build_contradiction_experiment  # noqa: E402
from olmo_core.internal.experiment import main  # noqa: E402


def build_experiment_config(cli_context):
    """Select the compressive arm of the shared contradiction-only recipe."""
    return build_contradiction_experiment(cli_context, arm="compressive")


if __name__ == "__main__":
    main(config_builder=build_experiment_config)
