from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Optional, Sequence, TypedDict

from typing_extensions import NotRequired

import olmo_core.distributed.utils as dist_utils
import olmo_core.io as io
from olmo_core.aliases import PathOrStr
from olmo_core.config import Config
from olmo_core.exceptions import OLMoConfigurationError


class Instance(TypedDict):
    """
    An instance is just a dictionary that should include ``input_ids`` and optionally a
    corresponding ``label_mask``.
    """

    input_ids: Sequence[int]
    """The token IDs for this instance."""
    label_mask: NotRequired[Sequence[bool]]
    """An optional mask indicating which tokens should contribute to the loss."""


class InstanceSource(metaclass=ABCMeta):
    """
    An abstract base class for a source of instances, usually consumed by a :class:`ComposableDataLoader`.
    It essentially represents an array of instances, where each instance is a sequence of
    ``sequence_length`` tokens.

    :param work_dir: A local working directory that can be used for caching files during preprocessing.
    :param sequence_length: The length of each sequence (instance) to produce.
    :param max_sequence_length: For sources that support this. If you intend to increase the sequence
      length in the middle of an epoch, you should set this to the maximum sequence length that you'll
      train on to guarantee that you can restart the run with the same data order after changing sequence length.
      Care needs to be taken when implementing this in a subclass to ensure that the exact same tokens
      will be produced when `sequence_length` is changed but `max_sequence_length` is fixed.
    """

    def __init__(
        self,
        *,
        work_dir: PathOrStr,
        sequence_length: int,
        max_sequence_length: Optional[int] = None,
    ):
        if io.is_url(work_dir):
            raise OLMoConfigurationError(
                f"'work_dir' should be a local path, not a URL ('{work_dir}')."
            )
        assert sequence_length > 0
        if max_sequence_length is not None:
            assert max_sequence_length > 0
            if sequence_length > max_sequence_length:
                raise OLMoConfigurationError(
                    "'sequence_length' cannot be greater than 'max_sequence_length'."
                )
            if max_sequence_length % sequence_length != 0:
                raise OLMoConfigurationError(
                    "'max_sequence_length' must be a multiple of 'sequence_length'."
                )
        self._sequence_length = sequence_length
        self._max_sequence_length = max_sequence_length or sequence_length
        self._work_dir = Path(io.normalize_path(work_dir))
        if self._work_dir.name == self.__class__.__name__:
            self._work_dir = self._work_dir.parent
        self._fs_local_rank = dist_utils.get_fs_local_rank()
        self._rank = dist_utils.get_rank()

    @property
    def sequence_length(self) -> int:
        """The sequence length of each instance that this source will produce."""
        return self._sequence_length

    @property
    def max_sequence_length(self) -> int:
        """
        Typically the same as ``sequence_length`` though in some cases it can be greater, such
        as when the sequence length will be increased in the middle of an epoch.
        """
        return self._max_sequence_length

    @property
    def work_dir(self) -> Path:
        """
        A local working directly that can be used by the token source for caching files during
        preprocessing.
        """
        return self._work_dir / self.__class__.__name__

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
    @abstractmethod
    def fingerprint(self) -> str:
        """
        A unique, deterministic string representing the contents of the source.
        This is used by the data loader during restarts to validate that it can resume in the middle of an epoch
        by comparing the current fingerprint to the fingerprint stored in the checkpoint.

        So the fingerprint should take into account everything that impacts the complete sequence of tokens
        (i.e., what you'd get if you concatenated all instances) that the source produces,
        including the contents of the underlying data and any relevant parameters to this class,
        such as ``max_sequence_length``.
        However, it should not take into account parameters that are allowed to change on a restart.

        For example, the :class:`ConcatAndChunkInstanceSource` doesn't include ``sequence_length`` in its fingerprint
        because as long as ``sequence_length`` divides evenly into ``max_sequence_length`` it
        can vary while still producing the same complete sequence of tokens even though individual
        instances will change.
        """
        raise NotImplementedError

    @abstractmethod
    def __len__(self) -> int:
        """The number of instances available from this source."""
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, idx: int) -> Instance:
        """Get an instance by index."""
        raise NotImplementedError

    def __iter__(self) -> Generator[Instance, None, None]:
        """Iterate over all instances in the source."""
        for i in range(len(self)):
            yield self[i]

    def validate_index(self, idx: int) -> int:
        idx = int(idx)
        if idx < 0:
            idx = len(self) + idx
        if not (0 <= idx < len(self)):
            raise IndexError(
                f"Index {idx} is out of bounds for source {self} with {len(self):,d} instances."
            )
        return idx


@dataclass
class InstanceSourceConfig(Config):
    """A base config class for configuring and building an :class:`InstanceSource`."""

    @abstractmethod
    def build(self, work_dir: PathOrStr) -> InstanceSource:
        """Build the :class:`InstanceSource`."""
        raise NotImplementedError
