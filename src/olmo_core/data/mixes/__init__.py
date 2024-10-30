import os
from abc import abstractmethod
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, List, Tuple

from olmo_core.config import StrEnum

from ..tokenizer import TokenizerName

__all__ = ["DataMixBase", "DataMix"]


class DataMixBase(StrEnum):
    """
    Base class for enumeration of data mixes.
    """

    @abstractmethod
    def build(self, base_dir: str, tokenizer: str) -> Tuple[List[str], List[str]]:
        """
        Construct the data mix.

        :param base_dir: Where the mix is stored, e.g. "s3://ai2-llm" or "/weka/oe-training-default/ai2-llm".
        :param tokenizer: The tokenizer identifier.

        :returns: A list of paths/URLs to the tokenized numpy data files in the mix and list
            of corresponding labels.
        """
        raise NotImplementedError


class DataMix(DataMixBase):
    """
    An enumeration of data mix names.
    """

    OLMoE_mix_0824 = "OLMoE-mix-0824"
    dolma17 = "dolma17"
    v3_small_ppl_validation = "v3-small-ppl-validation"

    def build(self, base_dir: str, tokenizer: str) -> Tuple[List[str], List[str]]:
        if not base_dir.endswith("/"):
            base_dir = base_dir + "/"

        tokenizer_id: str = tokenizer
        if self == DataMix.v3_small_ppl_validation:
            if tokenizer == TokenizerName.gpt_neox_olmo_dolma_v1_5:
                tokenizer_id = "gptneox20b"
            elif tokenizer == TokenizerName.dolma2:
                tokenizer_id = "dolma2-tokenizer"
        elif tokenizer == TokenizerName.gpt_neox_olmo_dolma_v1_5:
            tokenizer_id = "gpt-neox-olmo-dolma-v1_5"

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
        with importlib_resources.as_file(
            importlib_resources.files("olmo_core").joinpath(
                f"data/mixes/{os.path.basename(name)}.txt"
            )
        ) as path:
            yield path
    finally:
        pass
