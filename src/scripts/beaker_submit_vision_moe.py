"""Submit a Vision-MoE Beaker spec only to the canonical project workspace.

This deliberately narrow wrapper is the only supported direct-spec submission path for
``configs/vision_moe``. It does not expose a workspace option or arbitrary Beaker arguments.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
from collections.abc import Sequence
from pathlib import Path

BEAKER_WORKSPACE = "ai2/molmofication"
OUTPUT_FORMATS = ("text", "json", "yaml", "csv")
_SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_SAFE_PATH_PART = re.compile(r"[A-Za-z0-9._ -]+")


def _get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spec", type=Path, help="Existing Beaker experiment YAML spec")
    parser.add_argument("-n", "--name", help="Optional Beaker experiment name")
    parser.add_argument("--format", choices=OUTPUT_FORMATS, help="Beaker CLI output format")
    parser.add_argument("-q", "--quiet", action="store_true", help="Enable Beaker quiet mode")
    return parser


def _validated_spec(parser: argparse.ArgumentParser, spec: Path) -> Path:
    if spec.suffix.lower() not in {".yaml", ".yml"}:
        parser.error(f"spec must have a .yaml or .yml suffix: {spec}")
    if any(part == ".." for part in spec.parts):
        parser.error(f"spec path traversal is not allowed: {spec}")
    for part in spec.parts:
        if part in {spec.anchor, "."}:
            continue
        if part.startswith("-") or _SAFE_PATH_PART.fullmatch(part) is None:
            parser.error(f"spec contains an unsafe path component: {part!r}")

    absolute_spec = Path(os.path.abspath(spec))
    try:
        resolved_spec = spec.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        parser.error(f"could not resolve spec {spec}: {error}")
    if absolute_spec != resolved_spec:
        parser.error(f"spec must not use symlinks: {spec}")
    if not resolved_spec.is_file():
        parser.error(f"spec does not exist or is not a file: {spec}")
    return resolved_spec


def _build_command(
    spec: Path,
    *,
    name: str | None,
    output_format: str | None,
    quiet: bool,
) -> list[str]:
    command = [
        "beaker",
        "experiment",
        "create",
        str(spec),
        f"--workspace={BEAKER_WORKSPACE}",
    ]
    if name is not None:
        command.append(f"--name={name}")
    if output_format is not None:
        command.append(f"--format={output_format}")
    if quiet:
        command.append("--quiet")
    return command


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments and submit the requested spec to the fixed project workspace."""
    parser = _get_parser()
    args = parser.parse_args(argv)
    if args.name is not None and _SAFE_NAME.fullmatch(args.name) is None:
        parser.error(
            "name must be 1-128 ASCII letters, digits, dots, underscores, or hyphens, "
            "and must start with a letter or digit"
        )
    spec = _validated_spec(parser, args.spec)

    subprocess.run(
        _build_command(
            spec,
            name=args.name,
            output_format=args.format,
            quiet=args.quiet,
        ),
        check=True,
        shell=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
