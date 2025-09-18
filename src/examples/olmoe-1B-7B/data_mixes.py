import os
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, List, Tuple

from olmo_core.data import NumpyDatasetConfig
from olmo_core.data.mixes import DataMixBase
from olmo_core.data.source_mixture import (
    SourceMixtureConfig,
    SourceMixtureDatasetConfig,
)
from olmo_core.data.types import NumpyDatasetDType

__all__ = ["CustomDataMix"]

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


class CustomDataMix(DataMixBase):
    """
    An enumeration of data mix names.
    """
    # test mixes
    test_mix = "test_mix"

    def build(self, base_dir: str, tokenizer: str) -> Tuple[List[str], List[str]]:
        """
        Construct the data mix.

        :param base_dir: Where the mix is stored, e.g. "s3://ai2-llm" or "/weka/oe-training-default/ai2-llm".
        :param tokenizer: The tokenizer identifier.

        :returns: A list of paths/URLs to the tokenized numpy data files in the mix and list
            of corresponding labels.
        """
        if not base_dir.endswith("/"):
            base_dir = base_dir + "/"

        tokenizer_id: str = tokenizer

        paths = []
        labels = []
        with _get_data_mix_path(self) as mix_path:
            with mix_path.open() as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    label, path = line.split(",")
                    path = path.replace("{TOKENIZER}", tokenizer_id)
                    paths.append(f"{base_dir}{path}")
                    labels.append(label)
        return paths, labels


@contextmanager
def _get_data_mix_path(name: str) -> Generator[Path, None, None]:
    import importlib_resources

    try:
        with importlib_resources.as_file(f"{CURRENT_DIR}/mixes/{os.path.basename(name)}.txt") as path:
            yield path
    finally:
        pass