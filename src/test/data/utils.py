from os import PathLike
from pathlib import Path
from typing import Any, List, Tuple, Type, Union

import numpy as np

Mmaps = List[Tuple[Union[Path, PathLike[Any], str], Any]]


def mk_mmaps(
    tmp_path: Path,
    prefix: str,
    num_files: int,
    size: int,
    dtype: Union[Type[np.uint8], Type[np.uint16], Type[np.uint32], Type[np.uint64]] = np.uint32,
    eos: int = 0,
    seq_length: int = 4,
    seed: int = 42,
) -> Mmaps:
    mmaps: Mmaps = []
    for i in range(num_files):
        filepath = f"{tmp_path}/{prefix}_{i}.npy"
        np.random.seed(seed)
        data = np.random.randint(1, np.iinfo(dtype).max, size=size, dtype=dtype)
        data = np.append(
            np.insert(data, np.arange(seq_length + 1, len(data), seq_length), eos), eos
        )
        mm = np.memmap(filepath, mode="w+", dtype=dtype, shape=(len(data),))
        mm[:] = data
        mm.flush()
        mmaps.append((Path(filepath), data))

    return mmaps
