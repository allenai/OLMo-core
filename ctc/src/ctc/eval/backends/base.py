"""
The backend protocol, and lazy discovery of which backends this install can actually use.

A backend does exactly one thing: turn prompts into text. It does **not** know what a task is, how
to parse an answer, or how to score one -- those live in :mod:`ctc.format` and :mod:`ctc.eval.tasks`
and are shared by every backend. That split is deliberate. When each driver carried its own parsing
and stop rules, the same bug had to be found three times: the ``</think>`` truncation fix landed in
the vLLM driver, the primed-bracket doc-id fix in the native one, and the grouping-JSON fix in a
third place. A backend that only produces text cannot host a grading bug.

Backends are discovered lazily. ``ctc`` installs with no heavy dependencies at all, so importing a
backend whose extra is missing is an expected condition, not an error -- it is reported as an
actionable install hint rather than an ImportError traceback from six frames down.
"""

from __future__ import annotations

import importlib
import importlib.util
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    runtime_checkable,
)

if TYPE_CHECKING:
    from ..stopping import StopCondition

__all__ = ["Backend", "BackendInfo", "available", "load", "describe"]


@runtime_checkable
class Backend(Protocol):
    """
    What every evaluation backend must provide.

    Implementations live in :mod:`ctc.eval.backends.hf`, ``.vllm`` and ``.native``.
    """

    def generate(
        self,
        prompts: Sequence[str],
        examples: Optional[Sequence[Mapping[str, Any]]] = None,
        *,
        stop: "StopCondition",
        task: Optional[str] = None,
    ) -> List[str]:
        """
        Generate a continuation for each prompt.

        Greedy decoding: every caller in this package compares scores across checkpoints, so
        sampling would add variance that swamps the differences being measured.

        Three obligations make the backends comparable, and a new backend that skips any of them
        will produce numbers that differ from the others for reasons that are not the model:

        1. **Build the prefill through** :mod:`ctc.eval.prefill`. Tokenizing the prompt string
           independently drops the document-marker scaffold, which is worth about -0.01 f1 --
           small enough to read as noise.
        2. **Let** :func:`ctc.eval.stopping.apply` **have the last word.** A backend's own stop
           mechanism is an early exit; each one cuts at a slightly different point, and only a
           host-side truncation over the final decoded string is identical across all three.
        3. **Preserve input order**, so generation *i* is graded against example *i*.

        :param prompts: Rendered prompts, one per example.
        :param examples: The unified-format examples the prompts came from. Required whenever the
            prefill is structural, which is the normal case.
        :param stop: The task's stop condition, carrying the decode budget as well.
        :param task: Task name.

        :returns: One continuation per prompt, prompt text excluded, truncated and think-stripped,
            in input order.
        """
        ...

    def count_tokens(self, text: str) -> int:
        """
        :param text: A prompt.

        :returns: Its length in tokens. Used for the runner's length audit, which is what catches
            prompts silently exceeding the budget and scoring a clean 0.0.
        """
        ...


@dataclass(frozen=True)
class BackendInfo:
    """
    A backend and whether this install can run it.

    :param name: ``--backend`` value.
    :param extra: The pip extra that provides it, e.g. ``vllm`` for ``pip install 'ctc[vllm]'``.
    :param probe_module: Import name checked to decide availability, without importing it.
    :param description: One line, shown by ``ctc-eval --list-backends``.
    """

    name: str
    extra: str
    probe_module: str
    description: str

    @property
    def installed(self) -> bool:
        """:returns: True if :attr:`probe_module` can be imported in this environment."""
        try:
            return importlib.util.find_spec(self.probe_module) is not None
        except (ImportError, ValueError):
            return False

    @property
    def install_hint(self) -> str:
        """:returns: The command a user should run to enable this backend."""
        return f"pip install 'ctc[{self.extra}]'"


#: Every backend, whether or not it is installed here. ``native`` is the only one that touches
#: olmo_core, and it is confined to :mod:`ctc.eval.backends.native` and
#: :mod:`ctc.eval.masking.native` -- no other module in this package may import olmo_core, which is
#: what keeps `ctc` installable without it.
BACKENDS: Dict[str, BackendInfo] = {
    "hf": BackendInfo(
        name="hf",
        extra="hf",
        probe_module="transformers",
        description="HuggingFace transformers. Widest model coverage, slowest.",
    ),
    "vllm": BackendInfo(
        name="vllm",
        extra="vllm",
        probe_module="vllm",
        description="vLLM. Fastest for prefill-heavy long-context grading.",
    ),
    "native": BackendInfo(
        name="native",
        extra="native",
        probe_module="olmo_core",
        description="olmo-core generation module. Grades a training checkpoint directly, no export.",
    ),
}


def available() -> List[str]:
    """:returns: Names of backends whose dependencies are present, in preference order."""
    return [name for name in ("native", "vllm", "hf") if BACKENDS[name].installed]


def describe() -> List[str]:
    """:returns: Human-readable lines for ``--list-backends``."""
    lines = []
    for name in ("native", "vllm", "hf"):
        info = BACKENDS[name]
        mark = "available" if info.installed else f"needs: {info.install_hint}"
        lines.append(f"  {name:<8} {info.description}\n           [{mark}]")
    return lines


def load(name: str, **kwargs) -> Backend:
    """
    Import and construct a backend by name.

    :param name: One of :data:`BACKENDS`.
    :param kwargs: Passed to the backend's constructor (checkpoint path, dtype, ...).

    :returns: A ready :class:`Backend`.

    :raises KeyError: If ``name`` is not a known backend.
    :raises ModuleNotFoundError: If the backend's extra is not installed. The message names the
        exact pip command rather than surfacing a missing-import traceback.
    """
    try:
        info = BACKENDS[name]
    except KeyError:
        raise KeyError(
            f"unknown backend {name!r}; choose from {', '.join(sorted(BACKENDS))}"
        ) from None

    if not info.installed:
        raise ModuleNotFoundError(
            f"backend {name!r} requires the '{info.extra}' extra, which is not installed.\n"
            f"  {info.install_hint}"
        )

    module = importlib.import_module(f"ctc.eval.backends.{name}")
    return module.build(**kwargs)  # type: ignore[no-any-return]
