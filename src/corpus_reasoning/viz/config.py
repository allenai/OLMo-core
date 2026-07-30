"""
Central path/config for the corpus-reasoning visualization site.

All paths are overridable via environment variables so the pipeline runs the
same whether it is invoked from the standalone corpus-reasoning checkout or from
the OLMo-core submodule (``OLMo-core/corpus-reasoning``).

Environment overrides:
    CR_DATA_ROOT    Directory holding the task ``*.jsonl`` files.
                    Default: /scratch/users/prasann/corpus-reasoning/data
    OLMO_CORE_ROOT  Root of the OLMo-core checkout (for experiment configs).
                    Default: parent of the corpus-reasoning repo if that parent
                    looks like an OLMo-core checkout, else a known absolute path.
    VIZ_OUT_DIR     Where pipeline artifacts (JSON + HTML) are written.
                    Default: <repo>/viz/outputs
"""

import os

# Directory containing this file: <corpus-reasoning>/viz/
VIZ_DIR = os.path.dirname(os.path.abspath(__file__))
# The corpus-reasoning repo root (one level up from viz/).
CR_REPO_ROOT = os.path.dirname(VIZ_DIR)

# --- Task data (the .jsonl files; gitignored, live on scratch) ----------------
CR_DATA_ROOT = os.environ.get(
    "CR_DATA_ROOT", "/scratch/users/prasann/corpus-reasoning/data"
)


def _default_olmo_core_root() -> str:
    """Best-effort guess at the OLMo-core checkout that owns the experiment configs."""
    # When corpus-reasoning is a submodule, OLMo-core is its parent directory.
    parent = os.path.dirname(CR_REPO_ROOT)
    if os.path.isdir(os.path.join(parent, "src", "scripts", "train", "memexpress")):
        return parent
    # Fall back to the known dev checkout.
    fallback = "/accounts/projects/berkeleynlp/prasann/projects/OLMo-core"
    return fallback


OLMO_CORE_ROOT = os.environ.get("OLMO_CORE_ROOT", _default_olmo_core_root())

# Directory holding the SFT/CPT-mix training scripts whose docstrings/configs
# drive the "Experiments" section.
SFT_SCRIPTS_DIR = os.path.join(OLMO_CORE_ROOT, "src", "scripts", "train", "memexpress")

# --- Outputs ------------------------------------------------------------------
OUT_DIR = os.environ.get("VIZ_OUT_DIR", os.path.join(VIZ_DIR, "outputs"))
DATA_EXAMPLES_JSON = os.path.join(OUT_DIR, "data_examples.json")
EXPERIMENTS_JSON = os.path.join(OUT_DIR, "experiments.json")
SITE_HTML = os.path.join(OUT_DIR, "index.html")

# Checked-in demo (a committed snapshot of the rendered site, EMO-style).
DEMO_DIR = os.path.join(VIZ_DIR, "demo")
DEMO_HTML = os.path.join(DEMO_DIR, "index.html")


def ensure_out_dir() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
