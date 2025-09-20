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


class NumpyDatasetDType(StrEnum):
    uint8 = "uint8"
    uint16 = "uint16"
    uint32 = "uint32"
    uint64 = "uint64"

    def as_np_dtype(
        self,
    ) -> Union[Type[np.uint8], Type[np.uint16], Type[np.uint32], Type[np.uint64]]:
        return getattr(np, str(self))
