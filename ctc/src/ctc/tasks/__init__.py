"""
One package per task. Adding a task means adding a directory here, and nothing else.

Layout of a task package::

    ctc/tasks/contradiction/
        __init__.py     registers the spec -- the only file the loader touches
        spec.py         the EVAL contract: instruction, parse, score, rungs, gold index base
        generate.py     the DATA contract: source corpus -> task JSONL
        sources/        one module per corpus, when a task has more than one

``spec.py`` and ``generate.py`` are split because they answer to different masters and change for
different reasons. A spec change reprices every existing result; a generator change only affects
data built afterwards. Keeping them in one file made it easy to touch both while meaning to touch
one, and the format fingerprint could not tell you which had moved.

Anything shared by several tasks stays in :mod:`ctc.format` -- pair parsing alone is used by five
tasks, and five copies of it is precisely how the copies drifted apart. A file under ``tasks/``
should hold only what is true of that one task.

Discovery is explicit: :data:`TASK_MODULES` lists what gets imported. Scanning the directory
instead would mean a task silently disappearing from the suite because its module raised an
ImportError, which is the sort of thing that gets noticed a week later when a results table has one
fewer row than it should.
"""

from __future__ import annotations

import importlib
from typing import Dict, List

from ..format import registry

__all__ = ["TASK_MODULES", "load_all", "loaded", "import_errors"]

#: Task packages to import, in registration order. Add a task by adding its name here.
TASK_MODULES: tuple = (
    "contradiction",
    "redundancy",
    "strmatch",
    "mathmatch",
    "cycle",
    "groups4",
    "textgroups",
    "absence",
    "xabsence",
    "retrieval",
    "qa",
    "grouping_labeled",
    "reorder",
    "outlier",
    "rerank",
    "oolong",
    "summarization",
    "qdmatch",
)

_loaded: List[str] = []
_errors: Dict[str, BaseException] = {}


def load_all(*, strict: bool = True) -> List[str]:
    """
    Import every task package so their specs register themselves.

    Idempotent: importing a task twice would hit the registry's duplicate guard, so already-loaded
    modules are skipped.

    :param strict: Raise if a task package fails to import. Default on, because a task that
        silently vanishes from the registry produces a suite run with a missing row rather than an
        error -- and a missing row is easy not to notice.

    :returns: Names of the tasks now registered.

    :raises ImportError: If ``strict`` and any task package failed to import.
    """
    for name in TASK_MODULES:
        if name in _loaded:
            continue
        try:
            importlib.import_module(f"{__name__}.{name}")
        except BaseException as e:  # noqa: BLE001 -- recorded and re-raised below
            _errors[name] = e
            if strict:
                raise ImportError(f"task package {name!r} failed to import: {e}") from e
            continue
        _loaded.append(name)
    return registry.names()


def loaded() -> List[str]:
    """:returns: Task packages imported so far."""
    return list(_loaded)


def import_errors() -> Dict[str, BaseException]:
    """:returns: Task packages that failed to import, when loaded with ``strict=False``."""
    return dict(_errors)
