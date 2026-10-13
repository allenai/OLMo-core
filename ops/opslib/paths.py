"""Filesystem locations used by every ops module."""
import os
from pathlib import Path

OPS_ROOT = Path(__file__).resolve().parent.parent
# Live state lives OUTSIDE any code checkout (its own small git repo) so rebases never touch it;
# ops/state/ in the checkout is only the fallback when that directory does not exist.
DEFAULT_STATE_DIR = Path("/accounts/projects/berkeleynlp/prasann/ops_state")
STATE_DIR = Path(
    os.environ.get("OPS_STATE_DIR")
    or (DEFAULT_STATE_DIR if DEFAULT_STATE_DIR.is_dir() else OPS_ROOT / "state")
)
CACHE_DIR = STATE_DIR / "cache"
RUNS_PATH = STATE_DIR / "runs.jsonl"
RESULTS_PATH = STATE_DIR / "results.jsonl"
EXPERIMENTS_DIR = OPS_ROOT / "experiments"
# The main OLMo-core checkout (read-only from ops' point of view).
REPO_ROOT = Path(os.environ.get("OPS_REPO_ROOT", "/accounts/projects/berkeleynlp/prasann/projects/OLMo-core"))
