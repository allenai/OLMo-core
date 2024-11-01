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

    vsl = "vsl"
    """
    Variable sequenced length ➡️ :class:`NumpyVSLDataset`.
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
