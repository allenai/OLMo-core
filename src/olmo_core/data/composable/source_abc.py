from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import ClassVar, Iterable, Optional

import olmo_core.distributed.utils as dist_utils
import olmo_core.io as io
from olmo_core.aliases import PathOrStr
from olmo_core.exceptions import OLMoConfigurationError


class SourceABC(metaclass=ABCMeta):
    """
    Abstract base class for source types.

    :param work_dir: A common local working directory that can be used for caching files during
        preprocessing.
    :param label: An optional label for this source, useful for debugging and visualizing.
    """

    DISPLAY_ICON: ClassVar[str] = ""  # Nerd Font icon for visualizations

    def __init__(self, *, work_dir: PathOrStr, label: Optional[str] = None):
        if io.is_url(work_dir):
            raise OLMoConfigurationError(
                f"'work_dir' should be a local path, not a URL ('{work_dir}')."
            )
        work_dir = Path(io.normalize_path(work_dir))
        if work_dir.name == self.__class__.__name__:
            work_dir = work_dir.parent
        self._common_work_dir = work_dir
        self._fs_local_rank = dist_utils.get_fs_local_rank()
        self._rank = dist_utils.get_rank()
        self._label = label

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.fingerprint[:7]})"

    @property
    def common_work_dir(self) -> Path:
        """
        The common working directory, usually the parent of :data:`work_dir`.
        """
        return self._common_work_dir

    @property
    def work_dir(self) -> Path:
        """
        The class-specific local working directory that can be used by the source for caching files during
        preprocessing.
        """
        return self.common_work_dir / self.__class__.__name__

    @property
    def fs_local_rank(self) -> int:
        """
        The local rank of the current process with respect to filesystem access of the working
        directory.
        """
        return self._fs_local_rank

    @property
    def rank(self) -> int:
        """The global rank of the current process across the entire distributed job."""
        return self._rank

    @property
    def label(self) -> Optional[str]:
        """The label assigned to this source."""
        return self._label

    @property
    @abstractmethod
    def fingerprint(self) -> str:
        """A unique, deterministic string representing the ordered contents of the source."""
        raise NotImplementedError

    @property
    @abstractmethod
    def num_tokens(self) -> int:
        """The number of tokens available from this source."""
        raise NotImplementedError

    @abstractmethod
    def children(self) -> Iterable["SourceABC"]:
        """Get the child sources that make up this source, if any."""
        raise NotImplementedError

    @property
    def is_leaf(self) -> bool:
        """Check if this source is a leaf node (i.e. has no children)."""
        for _ in self.children():
            return False
        return True
