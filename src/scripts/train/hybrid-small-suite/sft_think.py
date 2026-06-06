"""
OLMo hybrid small suite — 275M, 810M, and 1.4B SFT Think runs.

Architecture: see arch.py (identical to pretraining/midtraining — checkpoints load cleanly).

Usage::

  python src/scripts/train/hybrid-small-suite/sft_think.py dry_run \\
      hybrid-small-sft-think-275M ai2/jupiter

  python src/scripts/train/hybrid-small-suite/sft_think.py launch \\
      hybrid-small-sft-think-275M ai2/titan \\
      --launch.num_nodes=1 --launch.priority=urgent --launch.budget=ai2/oe-other
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sft_common import SEQUENCE_LENGTH, run_sft

DATASET_PATH = (
    "/weka/oe-training-default/ai2-llm/jacobm/data/sft/rl-sft-32k/olmo-hybrid-sft-triple-tools"
)

SFT_THINK_CONFIGS = {
    "275m": dict(
        lr=4e-4,
        global_batch_size=64 * SEQUENCE_LENGTH,
        load_path="/weka/oe-training-default/ai2-llm/checkpoints/yashasbls/hybrid-small-long-context-v2-275m/step47684/",
    ),
    "810m": dict(
        lr=1e-4,
        global_batch_size=64 * SEQUENCE_LENGTH,
        load_path="/weka/oe-training-default/ai2-llm/checkpoints/yashasbls/hybrid-small-long-context-v2-810m/step23842/",
    ),
    "1.4b": dict(
        lr=1e-4,
        global_batch_size=64 * SEQUENCE_LENGTH,
        fused_linear_loss=True,
        load_path="/weka/oe-training-default/ai2-llm/checkpoints/yashasbls/hybrid-small-long-context-v2-1.4b/step23842/",
    ),
}

if __name__ == "__main__":
    run_sft(sft_configs=SFT_THINK_CONFIGS, dataset_path=DATASET_PATH, tags=["sft-think"])
