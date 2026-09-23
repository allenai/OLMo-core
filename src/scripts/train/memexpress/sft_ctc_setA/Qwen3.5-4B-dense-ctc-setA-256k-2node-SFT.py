"""Dense Qwen3.5-4B SFT on the CTC setA mix, window 256k at amandab's 2-node geometry. Recipe + caveats: _qwen35_setA_common.py."""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "sft_xlong256k"))  # amandab's 256k builder

from _qwen35_setA_common import build_setA_experiment  # noqa: E402

from olmo_core.internal.experiment import main  # noqa: E402


def build_experiment_config(cli_context):
    """Select the 256k-2node arm of the setA dense pair."""
    return build_setA_experiment(cli_context, arm="256k-2node")


if __name__ == "__main__":
    main(config_builder=build_experiment_config)
