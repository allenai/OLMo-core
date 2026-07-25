import numpy as np

from olmo_core.config import StrEnum

NumpyUIntTypes = type[np.uint8] | type[np.uint16] | type[np.uint32] | type[np.uint64]


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
