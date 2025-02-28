from pathlib import Path
from typing import Type, Union

import numpy as np

from olmo_core.data import NumpyFSLDatasetMixture, TokenizerConfig
from olmo_core.data.source_mixture import (
    SourceMixtureConfig,
    SourceMixtureDatasetConfig,
)
from olmo_core.data.types import NumpyDatasetDType

from ..utils import mk_mmaps, Mmaps


def get_fsl_mixture(
    tmp_path: Path,
    dtype: Union[Type[np.uint8], Type[np.uint16], Type[np.uint32], Type[np.uint64]] = np.uint32,
    seed: int = 42,
    sequence_length: int = 4,
    num_tokens: int = 20 * 1000,
    eos: int = 0,
    vocab_size: int = 32_000,
    pad_token_id: int = -1,
) -> NumpyFSLDatasetMixture:
    seed = 42
    mmap1 = mk_mmaps(
        tmp_path, "mmap1", 1, num_tokens * 2, dtype, eos=eos, seed=seed, seq_length=sequence_length
    )
    mmap2 = mk_mmaps(
        tmp_path, "mmap2", 1, num_tokens * 2, dtype, eos=eos, seed=seed, seq_length=sequence_length
    )

    tokenizer = TokenizerConfig(
        vocab_size=vocab_size,
        eos_token_id=eos,
        pad_token_id=pad_token_id,
    )

    mixture_config = SourceMixtureDatasetConfig(
        render_tables=False,
        max_tokens=num_tokens,
        sequence_length=sequence_length,
        source_configs=[
            SourceMixtureConfig(
                source_name="mmap1",
                paths=[i[0] for i in mmap1],
                target_ratio=0.8,
            ),
            SourceMixtureConfig(
                source_name="mmap2",
                paths=[i[0] for i in mmap2],
                target_ratio=0.2,
            ),
        ],
        dtype=NumpyDatasetDType.uint16,
        processes=1,
        seed=seed,
    )

    mixture = mixture_config.build()
    return NumpyFSLDatasetMixture(
        *mixture.to_paths(),
        seed=mixture.seed,
        sequence_length=sequence_length,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        vocab_size=tokenizer.vocab_size,
        dtype=dtype,
        metadata=None,
        include_instance_metadata=None,
        generate_doc_lengths=False,
        path_offset_index=mixture.to_index(),
    )
