"""Dense Qwen3-4B-Base SFT on the setA 32k arm's examples (window 65,536). See _qwen35_setA_common.py."""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)  # also holds amandab's 256k builder (_qwen35_xlong5_...)
# Before the builder is imported: it resolves the model, shards and arms at import time.
if os.environ.setdefault("SETA_MODEL", "qwen3-4b") != "qwen3-4b":
    raise SystemExit(f"SETA_MODEL={os.environ['SETA_MODEL']!r}: this script is the qwen3-4b arm")

from _qwen35_setA_common import build_setA_experiment, setA_train_hooks  # noqa: E402

from olmo_core.internal.experiment import main  # noqa: E402


def build_experiment_config(cli_context):
    """Select the 32k arm (the Qwen3.5 32k arm's example set) of the Qwen3-4B setA run."""
    return build_setA_experiment(cli_context, arm="32k")


if __name__ == "__main__":
    main(config_builder=build_experiment_config, train_hooks=setA_train_hooks())
