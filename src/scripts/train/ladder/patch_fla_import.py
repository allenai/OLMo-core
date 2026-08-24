"""Make the installed fla package import-resilient next to transformers 5.x forks.

fla's ``__init__`` eagerly imports ``fla.layers`` and ``fla.models``, which import
transformers internals. Next to a transformers 5.0.0.dev0 build (e.g. the
``olmo-3.5-hybrid-state-bench`` fork needed to load converted StateBench checkpoints),
those imports raise an ``ImportError`` that escapes fla's optional-module handler
(which only swallows ``ModuleNotFoundError``), killing ``import fla`` entirely — even
though ``fla.ops`` and ``fla.modules``, the only parts olmo-core and the HF hybrid
modeling code use, don't depend on transformers at all.

This script patches the installed fla's ``_import_optional_public_module`` to treat any
import-time failure of those optional submodules as "unavailable". Run it once per
environment, after installing the transformers fork::

    python src/scripts/train/ladder/patch_fla_import.py

Idempotent: re-running on an already patched install is a no-op.
"""

import importlib.util
import sys
from pathlib import Path

OLD = """            return None
        raise
"""
NEW = """            return None
        raise
    except Exception:
        # Optional extension failed to import (e.g. an incompatible transformers
        # version); treat it as unavailable. Core fla.ops/fla.modules still work.
        return None
"""


def main() -> int:
    spec = importlib.util.find_spec("fla")
    if spec is None or spec.origin is None:
        raise SystemExit("fla is not installed in this environment")
    init = Path(spec.origin)
    source = init.read_text()

    if NEW in source:
        print(f"already patched: {init}")
        return 0
    if OLD not in source:
        raise SystemExit(
            f"{init} does not match the expected fla __init__ shape (fla version drift?); "
            "inspect and patch manually"
        )

    init.write_text(source.replace(OLD, NEW, 1))
    print(f"patched: {init}")

    # Verify the patch took: fla must now import even if fla.layers/fla.models are broken.
    import fla  # noqa: F401
    import fla.modules  # noqa: F401
    import fla.ops  # noqa: F401

    print("import fla / fla.modules / fla.ops OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
