"""
The generator registry: how :mod:`ctc.data` finds a task's data builder.

A generator declares one thing -- how to construct **one example** from a seeded RNG. Everything
above that (how many, which rungs, train vs eval, deduplication, auditing) is shared and lives in
:mod:`ctc.data.build`, because those are exactly the decisions that must not be re-litigated
per task. The pre-migration tree let each generator own its own ``main()``, and they diverged:
five different train/eval splitters with different defaults, ``--eval-frac`` of 0.1 in one file and
0.2 in another, ``int(round(x))`` in one and bare ``int(x)`` in the next.

Discovery is lazy. ``ctc.tasks.<name>.generate`` is imported only when someone asks to build that
task, so installing the package for evaluation does not drag in corpus loaders, BM25 or a
cross-encoder -- ``pip install ./ctc`` promises no GPU and no compiler, and the generation side is
where that promise is easiest to break.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

__all__ = ["Generator", "get", "names"]

#: Tasks with a ported generator. Listed rather than discovered, for the same reason
#: :data:`ctc.tasks.TASK_MODULES` is: a task that vanishes from a suite build because its module
#: failed to import should be an error, not a quietly shorter results table.
GENERATORS: tuple = (
    "cycle",
    "groups4",
    "mathmatch",
    "textgroups",
)


@dataclass(frozen=True)
class Generator:
    """
    How to build one example of one task.

    :param task: Task name; must match a registered :class:`~ctc.format.registry.TaskSpec`.
    :param source: The ``source`` tag every emitted example carries.
    :param build_example: ``(rng, **config) -> example``. Must consume the RNG deterministically
        and must not read global state -- the ladder builder calls it with derived substreams and
        relies on the same ``(seed, config)`` giving the same example.
    :param defaults: Default construction parameters, overridden per build.
    :param scaling_param: The config key that sets corpus size, i.e. the one a rung ladder varies.
        ``"num_docs"`` for almost everything.
    :param shrink_safe: True when dropping non-gold documents from an example cannot create or
        destroy a gold answer. This is what allows the eval ladder to be *nested* -- one canonical
        set of examples built at the largest rung, with smaller rungs derived by removing
        distractors, so every rung grades the same underlying questions. False forces each rung to
        be generated independently, which costs cross-rung comparability.
    :param notes: Free text surfaced by ``ctc-data list``.
    """

    task: str
    source: str
    build_example: Callable[..., Dict[str, Any]]
    defaults: Dict[str, Any] = field(default_factory=dict)
    scaling_param: str = "num_docs"
    shrink_safe: bool = True
    notes: str = ""

    def config(self, **overrides: Any) -> Dict[str, Any]:
        """
        Merge overrides onto the defaults.

        :param overrides: Caller-supplied parameters. ``None`` values are ignored so a CLI can pass
            unset options through without clobbering a default.

        :returns: The resolved config.

        :raises TypeError: On a parameter this generator does not accept -- a typo in a build
            script would otherwise be silently ignored and produce data at the default size.
        """
        unknown = [k for k, v in overrides.items() if v is not None and k not in self.defaults]
        if unknown:
            raise TypeError(
                f"{self.task}: unknown parameter(s) {sorted(unknown)}; "
                f"accepts {sorted(self.defaults)}"
            )
        merged = dict(self.defaults)
        merged.update({k: v for k, v in overrides.items() if v is not None})
        return merged


def names() -> List[str]:
    """:returns: Tasks with a ported generator."""
    return list(GENERATORS)


def get(task: str) -> Generator:
    """
    Load one task's generator.

    :param task: Task name.

    :returns: Its :class:`Generator`.

    :raises KeyError: If the task has no ported generator.
    :raises AttributeError: If the module exists but exposes no ``GENERATOR``.
    """
    if task not in GENERATORS:
        raise KeyError(
            f"no generator ported for {task!r}; have {', '.join(GENERATORS)}. "
            "See records/data-generator-port-plan.md for the remaining batches."
        )
    module = importlib.import_module(f"ctc.tasks.{task}.generate")
    generator: Optional[Generator] = getattr(module, "GENERATOR", None)
    if generator is None:
        raise AttributeError(f"ctc.tasks.{task}.generate defines no GENERATOR")
    return generator
