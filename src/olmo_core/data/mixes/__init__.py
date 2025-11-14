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
    OLMo_mix_0625_150Bsample = "OLMo-mix-0625-150Bsample"
    OLMo_mix_0625_700Bsample = "OLMo-mix-0625-700Bsample"
    OLMo_mix_0625 = "OLMo-mix-0625"
    OLMo_mix_0625_official = "OLMo-mix-0625-official"
    OLMo_mix_0925 = "OLMo-mix-0925"

    @classmethod
    def _missing_(cls, value: object) -> "DataMix | None":
        """Handle alias lookups."""
        # Aliases mapping
        aliases = {
            "dolma3-0625-6T-mix": "OLMo-mix-0625",
            "dolma3-0925-6T-mix": "OLMo-mix-0925",
            "dolma3-0925-150B-mix": "OLMo-mix-0625-150Bsample",
        }

        # Check if the value is an alias
        if isinstance(value, str) and value in aliases:
            # Look up the real value and return the corresponding enum member
            real_value = aliases[value]
            for member in cls:
                if member.value == real_value:
                    return member
        return None

    def build(self, base_dir: str, tokenizer: str) -> Tuple[List[str], List[str]]:
        if not base_dir.endswith("/"):
            base_dir = base_dir + "/"

        tokenizer_id: str = tokenizer
        if self == DataMix.v3_small_ppl_validation:
            if tokenizer == TokenizerName.gpt_neox_olmo_dolma_v1_5:
                tokenizer_id = "gptneox20b"
            elif tokenizer == TokenizerName.dolma2:
                tokenizer_id = "dolma2-tokenizer"
        elif self == DataMix.OLMo_mix_0625:
            if tokenizer == TokenizerName.dolma2_sigdig:
                tokenizer_id = "dolma2-tokenizer-sigdig"
        elif self == DataMix.OLMo_mix_0625_official:
            if tokenizer == TokenizerName.dolma2:
                tokenizer_id = "allenai/dolma3-tokenizer"
        elif tokenizer == TokenizerName.gpt_neox_olmo_dolma_v1_5:
            tokenizer_id = "gpt-neox-olmo-dolma-v1_5"

        paths = []
        labels = []
        with _get_data_mix_path(self) as mix_path:
            with mix_path.open() as f:
                for line_num, line in enumerate(f):
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    label, path = line.split(",")
                    if "{TOKENIZER}" not in path:
                        raise ValueError(f"line {line_num + 1} in data mix '{self}' is invalid")
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
