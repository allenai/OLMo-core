"""
Integration-test MoE midtraining stage.

This is the integration-architecture entrypoint for ``midtraining_ladder.py``.
It keeps the midtraining data/LR recipe shared with baseline midtraining, but
builds the source checkpoint architecture with ``integration_ladder.py`` so
wide/deep integration checkpoints can be loaded weight-only for midtraining.
"""

import sys
from pathlib import Path

REPO_SRC = Path(__file__).resolve().parents[5]
if str(REPO_SRC) not in sys.path:
    sys.path.insert(0, str(REPO_SRC))

LADDER_DIR = Path(__file__).resolve().parents[2]
if str(LADDER_DIR) not in sys.path:
    sys.path.insert(0, str(LADDER_DIR))

INTEGRATION_DIR = Path(__file__).resolve().parents[1] / "integration"
if str(INTEGRATION_DIR) not in sys.path:
    sys.path.insert(0, str(INTEGRATION_DIR))

import integration_ladder as integration_base  # noqa: E402
import midtraining_ladder as midtraining  # noqa: E402
from olmo_core.script_utils import main  # noqa: E402

# ``midtraining_ladder`` deliberately routes all architecture-specific behavior
# through this module-level object. Swap it before building the parser/config so
# ``--integration-config`` controls the model shape and W&B architecture tags.
midtraining.base = integration_base


if __name__ == "__main__":
    main(midtraining.build_config, parser=midtraining.get_parser())
