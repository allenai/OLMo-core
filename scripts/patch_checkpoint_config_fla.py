"""
Rewrite stale ``olmo_core.nn.fla.*`` class references in a checkpoint's
``config.json`` to their current locations.

The ``olmo_core.nn.fla`` module has been removed; GDN classes now live under
``olmo_core.nn.attention.recurrent``. Older training checkpoints' configs still
embed fully-qualified class paths via the ``_CLASS_`` field, which breaks
``TransformerConfig.from_dict`` during HF conversion.

Usage::

    python patch_checkpoint_config_fla.py /path/to/checkpoint

    # Dry run (print what would change, don't write):
    python patch_checkpoint_config_fla.py /path/to/checkpoint --dry-run

    # Write to a different file instead of in-place:
    python patch_checkpoint_config_fla.py /path/to/checkpoint -o /tmp/config.json
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

# Old fully-qualified class name -> new fully-qualified class name.
# Extend as needed; unmapped ``olmo_core.nn.fla.*`` strings will be reported.
RENAMES: dict[str, str] = {
    "olmo_core.nn.fla.FLAConfig": "olmo_core.nn.attention.recurrent.GatedDeltaNetConfig",
    "olmo_core.nn.fla.FLA": "olmo_core.nn.attention.recurrent.GatedDeltaNet",
    "olmo_core.nn.fla.layer.FLAConfig": "olmo_core.nn.attention.recurrent.GatedDeltaNetConfig",
    "olmo_core.nn.fla.layer.FLA": "olmo_core.nn.attention.recurrent.GatedDeltaNet",
    "olmo_core.nn.fla.model.FLAConfig": "olmo_core.nn.attention.recurrent.GatedDeltaNetConfig",
    "olmo_core.nn.fla.model.FLA": "olmo_core.nn.attention.recurrent.GatedDeltaNet",
}


def rewrite(obj, changes: list[tuple[str, str, str]], unknown: list[tuple[str, str]], path: str = ""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            child = f"{path}.{k}" if path else k
            if isinstance(v, str) and v.startswith("olmo_core.nn.fla"):
                if v in RENAMES:
                    new = RENAMES[v]
                    changes.append((child, v, new))
                    obj[k] = new
                else:
                    unknown.append((child, v))
            else:
                rewrite(v, changes, unknown, child)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            rewrite(v, changes, unknown, f"{path}[{i}]")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint", help="Checkpoint dir or path to config.json")
    p.add_argument("-o", "--output", help="Write to this path instead of in-place")
    p.add_argument("--dry-run", action="store_true", help="Report changes without writing")
    p.add_argument("--no-backup", action="store_true", help="Skip writing config.json.bak when editing in place")
    args = p.parse_args()

    src = Path(args.checkpoint)
    if src.is_dir():
        src = src / "config.json"
    if not src.is_file():
        print(f"error: {src} not found", file=sys.stderr)
        return 2

    config = json.loads(src.read_text())

    changes: list[tuple[str, str, str]] = []
    unknown: list[tuple[str, str]] = []
    rewrite(config, changes, unknown)

    for where, old, new in changes:
        print(f"  {where}: {old}  ->  {new}")
    for where, old in unknown:
        print(f"  UNMAPPED {where}: {old}", file=sys.stderr)

    if not changes and not unknown:
        print("No olmo_core.nn.fla references found; nothing to do.")
        return 0
    if unknown:
        print(
            f"\n{len(unknown)} unmapped reference(s) above. Add them to RENAMES and re-run.",
            file=sys.stderr,
        )
        return 1
    if args.dry_run:
        print(f"\nDry run: {len(changes)} change(s) not written.")
        return 0

    dest = Path(args.output) if args.output else src
    if dest == src and not args.no_backup:
        shutil.copy2(src, src.with_suffix(src.suffix + ".bak"))
    dest.write_text(json.dumps(config, indent=2))
    print(f"\nWrote {len(changes)} change(s) to {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
