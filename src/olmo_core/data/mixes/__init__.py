import os
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, List

from olmo_core.config import StrEnum

from ..tokenizer import TokenizerName


class DataMix(StrEnum):
    """
    An enumeration of data mix names.
    """

    OLMoE_mix_0824 = "OLMoE-mix-0824"
    dolma17 = "dolma17"

    def build(self, base_dir: str, tokenizer: TokenizerName) -> List[str]:
        """
        Construct the data mix.

        :param base_dir: Where the mix is stored, e.g. "s3://ai2-llm" or "/weka/oe-training-default/ai2-llm".
        :param tokenizer: The tokenizer identifier.

        :returns: A list of paths/URLs to the tokenized numpy data files in the mix.
        """
        if not base_dir.endswith("/"):
            base_dir = base_dir + "/"

        tokenizer_id: str = tokenizer
        if tokenizer == TokenizerName.gpt_neox_olmo_dolma_v1_5:
            tokenizer_id = "gpt-neox-olmo-dolma-v1_5"

        paths = []
        with self._get_data_mix_path() as mix_path:
            with mix_path.open() as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    line = line.replace("{TOKENIZER}", tokenizer_id)
                    paths.append(f"{base_dir}{line}")
        return paths

    @contextmanager
    def _get_data_mix_path(self) -> Generator[Path, None, None]:
        import importlib_resources

        try:
            with importlib_resources.as_file(
                importlib_resources.files("olmo_core").joinpath(
                    f"data/mixes/{os.path.basename(self)}.txt"
                )
            ) as path:
                yield path
        finally:
            pass
