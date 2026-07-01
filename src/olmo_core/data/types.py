from typing import Type, Union

import numpy as np

from olmo_core.config import StrEnum

NumpyUIntTypes = Union[Type[np.uint8], Type[np.uint16], Type[np.uint32], Type[np.uint64]]


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

    exclude = "exclude"
    """
    Long docs are dropped entirely (neither truncated nor fragmented). Useful when a truncated doc
    would lose the part that carries the training signal (e.g. an SFT answer at the end of the
    sequence), which would otherwise yield a fully-masked, NaN-loss instance.
    """


class NumpyDatasetDType(StrEnum):
    """
    Supported numpy unsigned integer data types for datasets.
    """

    uint8 = "uint8"
    uint16 = "uint16"
    uint32 = "uint32"
    uint64 = "uint64"

    def as_np_dtype(self) -> NumpyUIntTypes:
        """
        Convert the enum value to its corresponding numpy dtype.

        Returns:
            The numpy unsigned integer dtype corresponding to this enum value.
        """
        return getattr(np, str(self))
