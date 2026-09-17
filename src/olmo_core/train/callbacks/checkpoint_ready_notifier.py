import hashlib
import json
import logging
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, ClassVar, Dict

from olmo_core.aliases import PathOrStr
from olmo_core.distributed.utils import get_rank
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.io import is_url

from ..checkpoint import Checkpointer
from .callback import Callback

log = logging.getLogger(__name__)

_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_STEP_PATTERN = re.compile(r"^step(\d+)$")


@dataclass
class CheckpointReadyNotifierCallback(Callback):
    """Publish durable filesystem events after checkpoints are finalized.

    This callback is intentionally independent of any uploader implementation. It does no
    networking and never waits for a remote service. On global rank zero it writes a small JSON
    event to :data:`inbox_dir` using atomic publication. A separate service can consume these
    events, while periodically scanning the registered checkpoint root as a correctness fallback.

    Event publication is best effort: failures are logged without failing training.
    """

    priority: ClassVar[int] = -10

    inbox_dir: str = ""
    """Shared filesystem directory where ready events should be published."""

    run_id: str = ""
    """Stable identifier for this training run."""

    lineage_id: str = ""
    """Stable identifier shared only by intentional continuations of one trajectory."""

    include_ephemeral: bool = False
    """Whether to publish events for checkpoints marked as ephemeral."""

    def __post_init__(self) -> None:
        if not self.inbox_dir:
            raise OLMoConfigurationError("'inbox_dir' is required")
        if is_url(self.inbox_dir):
            raise OLMoConfigurationError("'inbox_dir' must be on a local/shared filesystem")
        if not Path(self.inbox_dir).is_absolute():
            raise OLMoConfigurationError("'inbox_dir' must be an absolute path")
        for field_name, value in (("run_id", self.run_id), ("lineage_id", self.lineage_id)):
            if not _ID_PATTERN.fullmatch(value):
                raise OLMoConfigurationError(
                    f"'{field_name}' must match {_ID_PATTERN.pattern!r}, got {value!r}"
                )

    @staticmethod
    def _checkpoint_step(path: Path) -> int:
        match = _STEP_PATTERN.fullmatch(path.name)
        if match is None:
            raise ValueError(f"checkpoint directory name does not match 'step<N>': {path}")
        return int(match.group(1))

    @staticmethod
    def _checkpoint_metadata(path: Path) -> tuple[Dict[str, Any], str]:
        metadata_path = path / Checkpointer.METADATA_FNAME
        metadata_bytes = metadata_path.read_bytes()
        metadata = json.loads(metadata_bytes)
        if not isinstance(metadata, dict):
            raise ValueError(f"checkpoint metadata must be a JSON object: {metadata_path}")
        return metadata, hashlib.sha256(metadata_bytes).hexdigest()

    def _event_path(self, step: int) -> Path:
        return Path(self.inbox_dir) / self.run_id / f"step-{step:012d}.ready.json"

    @staticmethod
    def _same_event(existing: Dict[str, Any], new: Dict[str, Any]) -> bool:
        identity_fields = (
            "schema_version",
            "event",
            "run_id",
            "lineage_id",
            "step",
            "checkpoint_path",
            "checkpoint_format",
            "checkpoint_metadata_sha256",
        )
        return all(existing.get(field) == new.get(field) for field in identity_fields)

    @classmethod
    def _publish_atomically(cls, event_path: Path, event: Dict[str, Any]) -> bool:
        event_path.parent.mkdir(parents=True, exist_ok=True)
        if event_path.exists():
            existing = json.loads(event_path.read_text(encoding="utf-8"))
            if isinstance(existing, dict) and cls._same_event(existing, event):
                return False
            raise FileExistsError(f"conflicting checkpoint-ready event: {event_path}")

        payload = (json.dumps(event, indent=2, sort_keys=True) + "\n").encode()
        temp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=event_path.parent,
                prefix=f".{event_path.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temp_path = Path(handle.name)
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())

            try:
                # A hard link gives us atomic create-if-absent semantics. Readers can never see
                # a partially written final event, and concurrent duplicate writers cannot
                # replace one another.
                os.link(temp_path, event_path)
            except FileExistsError:
                existing = json.loads(event_path.read_text(encoding="utf-8"))
                if isinstance(existing, dict) and cls._same_event(existing, event):
                    return False
                raise FileExistsError(f"conflicting checkpoint-ready event: {event_path}")

            directory_fd = os.open(event_path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
            return True
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)

    def _notify(self, checkpoint_path: Path) -> None:
        if not Checkpointer.dir_is_checkpoint(checkpoint_path):
            raise ValueError(f"checkpoint is not finalized: {checkpoint_path}")

        step = self._checkpoint_step(checkpoint_path)
        metadata, metadata_digest = self._checkpoint_metadata(checkpoint_path)
        if not self.include_ephemeral and metadata.get("ephemeral") is True:
            log.debug("Skipping ephemeral checkpoint-ready event for '%s'", checkpoint_path)
            return

        event = {
            "schema_version": 1,
            "event": "checkpoint_ready",
            "run_id": self.run_id,
            "lineage_id": self.lineage_id,
            "step": step,
            "checkpoint_path": str(checkpoint_path.resolve()),
            "checkpoint_format": "olmo_core",
            "checkpoint_metadata_sha256": metadata_digest,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "beaker_experiment_id": os.environ.get("BEAKER_EXPERIMENT_ID"),
            "beaker_task_id": os.environ.get("BEAKER_TASK_ID"),
            "beaker_job_id": os.environ.get("BEAKER_JOB_ID"),
        }
        event_path = self._event_path(step)
        if self._publish_atomically(event_path, event):
            log.info("Published checkpoint-ready event to '%s'", event_path)
        else:
            log.debug("Checkpoint-ready event already exists at '%s'", event_path)

    def post_checkpoint_saved(self, path: PathOrStr) -> None:
        if get_rank() != 0:
            return
        if is_url(path):
            log.warning(
                "Could not publish checkpoint-ready event for '%s': checkpoint path must be "
                "on a local/shared filesystem",
                path,
            )
            return
        try:
            self._notify(Path(path))
        except Exception:
            # Notification is a low-latency hint. The uploader's periodic scan is the correctness
            # path, so notifier failures must never terminate training.
            log.warning("Could not publish checkpoint-ready event for '%s'", path, exc_info=True)
