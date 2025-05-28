from pathlib import Path
from typing import Type, Union

import numpy as np

from olmo_core.data import NumpyDatasetBase, NumpyDatasetConfig, TokenizerConfig
from olmo_core.data.source_mixture import (
    SourceMixtureConfig,
    SourceMixtureDatasetConfig,
)
from olmo_core.data.types import NumpyDatasetDType

from .utils import mk_mmaps


def get_fsl_mixture(
    tmp_path: Path,
    dtype: Union[Type[np.uint8], Type[np.uint16], Type[np.uint32], Type[np.uint64]] = np.uint32,
    seed: int = 42,
    sequence_length: int = 4,
    num_tokens: int = 20 * 1000,
    eos: int = 0,
) -> NumpyDatasetBase:
    seed = 42
    mmap1 = mk_mmaps(
        tmp_path, "mmap1", 1, num_tokens * 2, dtype, eos=eos, seed=seed, seq_length=sequence_length
    )
    mmap2 = mk_mmaps(
        tmp_path, "mmap2", 1, num_tokens * 2, dtype, eos=eos, seed=seed, seq_length=sequence_length
    )

    tokenizer = TokenizerConfig(
        vocab_size=32_000,
        eos_token_id=eos,
        pad_token_id=-1,
    )

    mixture_config = SourceMixtureDatasetConfig(
        max_tokens=num_tokens,
        sequence_length=sequence_length,
        source_configs=[
            SourceMixtureConfig(
                source_name="mmap1",
                paths=[str(i[0]) for i in mmap1],
                target_ratio=0.8,
            ),
            SourceMixtureConfig(
                source_name="mmap2",
                paths=[str(i[0]) for i in mmap2],
                target_ratio=0.2,
            ),
        ],
        dtype=NumpyDatasetDType.uint16,
        processes=1,
        seed=seed,
    )

    ds = NumpyDatasetConfig(
        source_mixture_config=mixture_config,
        sequence_length=sequence_length,
        tokenizer=tokenizer,
        include_instance_metadata=False,
    ).build()
    ds.prepare()

    return ds
