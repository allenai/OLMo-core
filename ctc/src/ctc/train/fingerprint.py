"""
Stamp every checkpoint with the data format it was trained on.

This is the write half of the guard described in :mod:`ctc.format.fingerprint`. Without it the
read half can never fire: an unfingerprinted checkpoint grades with a warning and the warning is
the only thing standing between a format mismatch and a number in a results table. That warning is
not enough, and we have the receipt -- the contradiction @2k reproduction run printed exactly it,
and the unrecorded dimension (``query_position``) was precisely what differed, costing two rounds
of GPU time to find by bisection.

**Where the fingerprint comes from matters.** It is collected from the shard directories the run
actually reads (:data:`FormatFingerprintCallback.collect_from`), not declared alongside the launcher
config. A launcher's declaration is a claim about the data; the shards' own record is the data. When
they disagree, the launcher is the one that is wrong, and a guard built on the claim would certify
the mistake.

Usage in a training script::

    from ctc.train import FormatFingerprintCallback

    trainer_config.with_callback(
        "format_fingerprint",
        FormatFingerprintCallback(collect_from=[str(p) for p in shard_dirs]),
    )
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from olmo_core.aliases import PathOrStr
from olmo_core.distributed.utils import get_rank
from olmo_core.train.callbacks.callback import Callback

from ..format.fingerprint import (
    FINGERPRINT_FILENAME,
    FingerprintSet,
    FormatFingerprint,
    collect_fingerprints,
    conflicting_formats,
)

log = logging.getLogger(__name__)

__all__ = ["FormatFingerprintCallback"]


@dataclass
class FormatFingerprintCallback(Callback):
    """
    Write a :class:`~ctc.format.fingerprint.FingerprintSet` into every checkpoint directory.

    :param collect_from: Shard directories to read training-time fingerprints from. This is the
        preferred source -- see the module note on why the shards outrank the launcher.
    :param formats: Fingerprints supplied directly, for data whose shards have no record yet.
        Unioned with whatever ``collect_from`` yields.
    :param allow_missing: Permit a ``collect_from`` directory that has no fingerprint. Off by
        default: silently skipping one produces a checkpoint whose record is *incomplete*, which
        is worse than one with no record at all, because the eval-side guard will then confidently
        report a task as untrained.
    :param fname: Filename to write. Only override this to shadow a record you cannot delete.
    """

    collect_from: List[str] = field(default_factory=list)
    formats: List[FormatFingerprint] = field(default_factory=list)
    allow_missing: bool = False
    fname: str = FINGERPRINT_FILENAME

    _resolved: Optional[FingerprintSet] = None

    def pre_train(self) -> None:
        """
        Resolve and validate the set before the first step.

        Deliberately here rather than at the first checkpoint: a misconfigured ``collect_from``
        should cost seconds, not the hours until checkpoint one.

        :raises FileNotFoundError: If a ``collect_from`` directory has no fingerprint and
            ``allow_missing`` is not set.
        :raises ValueError: If nothing at all resolved.
        """
        self._resolved = self._resolve()
        log.info(
            "format fingerprint: %d format(s) over task(s) %s will be written to every checkpoint",
            len(self._resolved.formats),
            ", ".join(self._resolved.tasks),
        )

    def post_checkpoint_saved(self, path: PathOrStr) -> None:
        """
        :param path: The checkpoint directory just written. May be a remote URL, so the write goes
            through the trainer rather than the filesystem.
        """
        if get_rank() != 0:
            return
        if self._resolved is None:  # pre_train did not run (a resumed rank-0-only path)
            self._resolved = self._resolve()
        self.trainer.write_file(
            self.fname, json.dumps(self._resolved.to_dict(), indent=2, sort_keys=True), dir=path
        )

    # ── internals ───────────────────────────────────────────────────────────────────────────────

    def _resolve(self) -> FingerprintSet:
        """
        Collect the set, and say out loud anything that makes it less trustworthy.

        The collection itself lives in :func:`~ctc.format.fingerprint.collect_fingerprints`, shared
        with ``ctc-fingerprint collect`` -- a checkpoint stamped during training and one stamped
        afterwards must record the same thing.
        """
        resolved, skipped = collect_fingerprints(
            [Path(d) for d in self.collect_from],
            extra=self.formats,
            allow_missing=self.allow_missing,
        )
        if skipped:
            log.warning(
                "format fingerprint: %d shard dir(s) had no record and were skipped; the "
                "checkpoint's format record is INCOMPLETE: %s",
                len(skipped),
                ", ".join(skipped),
            )
        for task, fields in conflicting_formats(resolved).items():
            log.warning(
                "format fingerprint: task %r is recorded under several different formats, "
                "differing in %s. Intended for a curriculum; otherwise two shard builds have "
                "drifted apart.",
                task,
                ", ".join(fields) or "metadata only",
            )
        return resolved
