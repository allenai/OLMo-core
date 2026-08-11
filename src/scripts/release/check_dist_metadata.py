"""
Check built distributions for metadata that PyPI rejects at upload time.

``twine check`` only validates that the long description renders, so problems like a direct
URL requirement slip through the build and only surface as an opaque "400 Bad Request" from
``twine upload``, after the release tag has already been pushed.

Usage::

    python src/scripts/release/check_dist_metadata.py dist/*
"""

import sys
import tarfile
import zipfile
from pathlib import Path
from typing import List

from packaging.metadata import Metadata


def read_metadata(path: Path) -> str:
    """
    Read the core metadata out of a built distribution.

    :param path: A wheel or sdist.

    :returns: The contents of the distribution's ``METADATA``/``PKG-INFO``.

    :raises ValueError: If the file isn't a wheel or sdist, or the metadata is missing.
    """
    if path.suffix == ".whl":
        with zipfile.ZipFile(path) as zf:
            for name in zf.namelist():
                if name.endswith(".dist-info/METADATA"):
                    return zf.read(name).decode("utf-8")
        raise ValueError(f"No METADATA found in {path}")

    if path.name.endswith(".tar.gz"):
        with tarfile.open(path) as tf:
            for member in tf.getmembers():
                # The sdist's own PKG-INFO sits directly under the top-level directory,
                # unlike the copy nested in src/*.egg-info/.
                if member.name.count("/") == 1 and member.name.endswith("/PKG-INFO"):
                    handle = tf.extractfile(member)
                    assert handle is not None
                    return handle.read().decode("utf-8")
        raise ValueError(f"No PKG-INFO found in {path}")

    raise ValueError(f"Not a wheel or sdist: {path}")


def check(path: Path) -> List[str]:
    """
    Check a single distribution.

    :param path: A wheel or sdist.

    :returns: A list of problems, empty if the distribution is fine.
    """
    meta = Metadata.from_email(read_metadata(path), validate=True)
    return [
        f"Can't have direct dependency: {req}"
        for req in (meta.requires_dist or [])
        if req.url is not None
    ]


def main() -> None:
    paths = [Path(arg) for arg in sys.argv[1:]]
    if not paths:
        raise SystemExit("Usage: check_dist_metadata.py DIST [DIST ...]")

    failed = False
    for path in paths:
        problems = check(path)
        for problem in problems:
            print(f"{path.name}: {problem}", file=sys.stderr)
        failed = failed or bool(problems)

    if failed:
        raise SystemExit(
            "\nPyPI rejects these uploads. See "
            "https://packaging.python.org/specifications/core-metadata"
        )
    print(f"Checked {len(paths)} distribution(s): OK")


if __name__ == "__main__":
    main()
