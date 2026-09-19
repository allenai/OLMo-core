"""Reuse our existing Modal app without changing sandbox isolation or eval recipes.

App provenance: successful hero code experiment 01M22D1V352S52XWTM8GPHTVXM.
Each SWE-ReX deployment still creates its own sandbox, unique token, and runtime.
No app or unrelated sandbox is stopped/deleted, and no new app can be created.
"""

import runpy
import sys
from pathlib import Path

APP = "swerex-7ee77286d853"


def install(modal):
    """Bind SWE-ReX app lookups to a known app, failing closed if it is gone."""
    original = modal.App.lookup
    existing = original(APP, create_if_missing=False)

    def lookup(name, *args, **kwargs):
        if name == "swe-rex" or name.startswith("swerex-"):
            return existing
        return original(name, *args, **kwargs)

    modal.App.lookup = lookup
    print("HERO_CODE_EXISTING_MODAL_APP", APP, flush=True)


if __name__ == "__main__":
    import modal

    install(modal)
    script = Path(sys.argv[1]).resolve()
    sys.argv = sys.argv[1:]
    sys.path.insert(0, str(script.parent))
    runpy.run_path(str(script), run_name="__main__")
