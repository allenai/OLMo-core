"""
JSONL read/write. One implementation, with the flags pinned.

The pre-migration tree had five JSONL writers plus 29 inline ``f.write(json.dumps(x))`` sites, and
they disagreed: two crashed on a bare filename, two skipped the ``mkdir``, one passed
``ensure_ascii=False`` (making that task's escaping byte-incompatible with every other task's), and
**none passed ``encoding=``** -- so every read and write rode on the ambient locale and
"byte-identical" was a machine-dependent claim for any corpus with non-ASCII text.

Settled here, and not overridable: ``ensure_ascii=True`` (what 38 of the old files already
produced, so ported data stays comparable), explicit ``encoding="utf-8"``, and ``mkdir`` on write.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Union

__all__ = ["load_jsonl", "iter_jsonl", "save_jsonl", "save_json"]

PathLike = Union[str, Path]


def iter_jsonl(path: PathLike) -> Iterator[Dict[str, Any]]:
    """
    Stream a JSONL file one row at a time -- a 32k-rung ladder is several GB and the audits only
    ever hold one example.

    :param path: File to read.

    :yields: One decoded object per non-blank line.

    :raises json.JSONDecodeError: On a malformed line, with the line number in the message.
    """
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise json.JSONDecodeError(f"{path}:{lineno}: {e.msg}", e.doc, e.pos) from None


def load_jsonl(path: PathLike) -> List[Dict[str, Any]]:
    """
    :param path: File to read.

    :returns: One dict per non-blank line.
    """
    return list(iter_jsonl(path))


def save_jsonl(path: PathLike, rows: Iterable[Dict[str, Any]]) -> int:
    """
    Write rows as JSONL, creating the parent directory.

    :param path: Destination. A bare filename is fine.
    :param rows: Objects to write, consumed lazily so a generator streams.

    :returns: Number of rows written.
    """
    _parent(path).mkdir(parents=True, exist_ok=True)
    written = 0
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")
            written += 1
    return written


def save_json(path: PathLike, obj: Any) -> None:
    """
    Write one JSON object -- for manifests and audit reports, not rows.

    :param path: Destination.
    :param obj: Object to serialise.
    """
    _parent(path).mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=True, indent=2, sort_keys=True)
        f.write("\n")


def _parent(path: PathLike) -> Path:
    return Path(path).resolve().parent
