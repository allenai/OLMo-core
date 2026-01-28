from typing import Type, Union

import numpy as np

from olmo_core.config import StrEnum

NumpyUIntTypes = Union[Type[np.uint8], Type[np.uint16], Type[np.uint32], Type[np.uint64]]


class NumpyDatasetType(StrEnum):
    """
    An enumeration of the different :class:`NumpyDatasetBase` implementations.
    """

    fsl = "fsl"
    """
    Fixed sequenced length ➡️ :class:`NumpyFSLDataset`.
    """

    padded_fsl = "padded_fsl"
    """
    Padded fixed sequence length ➡️ :class:`NumpyPaddedFSLDataset`.
    """

    packed_fsl = "packed_fsl"
    """
    Packed fixed sequence length ➡️ :class:`NumpyPackedFSLDataset`.
    """

    interleaved_fsl = "interleaved_fsl"
    """
    Padded fixed sequence length with interleaved documents ➡️ :class:`NumpyInterleavedFSLDataset`.
    """

    vsl = "vsl"
    """
    Variable sequenced length ➡️ :class:`NumpyVSLDataset`.
    """

    shuffled_fsl = "shuffled_fsl"
    """
    Shuffled fixed sequence length ➡️ :class:`NumpyShuffledFSLDataset`.
    Documents are globally shuffled, optionally truncated, concatenated, and chunked.
    """


class LongDocStrategy(StrEnum):
    """
    Specifies how to handle documents that are longer than the max sequence length when packing.
    """

    truncate = "truncate"
    """
    Long docs are truncated and the excess tokens are discarded.
    """

    fragment = "fragment"
    """
    Long docs are split into smaller docs so that no tokens are discarded, but you end up with
    fragmented docs.
    """


class TruncateFrom(StrEnum):
    """
    Specifies which end of a document to truncate from when it exceeds the max window size.
    """

    start = "start"
    """
    Keep the first N tokens of the document (truncate from the end).
    """

    end = "end"
    """
    Keep the last N tokens of the document (truncate from the start).
    """


class NumpyDatasetDType(StrEnum):
    uint8 = "uint8"
    uint16 = "uint16"
    uint32 = "uint32"
    uint64 = "uint64"

    def as_np_dtype(
        self,
    ) -> Union[Type[np.uint8], Type[np.uint16], Type[np.uint32], Type[np.uint64]]:
        return getattr(np, str(self))
