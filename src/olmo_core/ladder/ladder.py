from abc import ABCMeta, abstractmethod

from olmo_core.config import StrEnum
from olmo_core.data import TokenizerConfig
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.optim import OptimConfig


class ModelSize(StrEnum):
    """
    An enumeration of the standard model sizes in the ladder.
    :class:`ModelLadder` implementations should produce models that match these sizes
    as close as possible, ignoring embeddings.
    """

    size_190M = "190M"
    """
    190M parameters.
    """
    size_370M = "370M"
    """
    370M parameters.
    """
    size_600M = "600M"
    """
    600M parameters.
    """
    size_760M = "760M"
    """
    760M parameters.
    """
    size_1B = "1B"
    """
    1B parameters.
    """
    size_3B = "3B"
    """
    3B parameters.
    """
    size_7B = "7B"
    """
    7B parameters.
    """
    size_13B = "13B"
    """
    13B parameters.
    """


class ModelLadder(metaclass=ABCMeta):
    """
    Base class for defining model ladder experiments.
    """

    def get_tokenizer_config(self) -> TokenizerConfig:
        """
        Get the tokenizer config to use throughput the ladder.
        """
        return TokenizerConfig.dolma2()

    @abstractmethod
    def get_model_config(
        self, size: ModelSize, sequence_length: int, tokenizer: TokenizerConfig
    ) -> TransformerConfig:
        """
        Get the model config for a given model size.

        :param size: The target model size.
        :param sequence_length: The sequence length to be used.
        """
        raise NotImplementedError

    @abstractmethod
    def get_optim_config(self, size: ModelSize, sequence_length: int) -> OptimConfig:
        """
        Get the optimizer config for a given model size.

        :param size: The target model size.
        :param sequence_length: The sequence length to be used.
        """
        raise NotImplementedError

    @abstractmethod
    def get_rank_microbatch_size(self, size: ModelSize, sequence_length: int, gpu_type: str) -> int:
        """
        Returns the micro-batch size in tokens per device that should be used for the given
        model size.

        :param size: The target model size.
        :param sequence_length: The sequence length to be used.
        :param gpu_type: The type of GPU.
        """
        raise NotImplementedError

    def get_global_batch_size(
        self, size: ModelSize, sequence_length: int, batch_size_divisor: int = 64
    ) -> int:
        """
        Get the global batch size in tokens for a given model size.
        """
        # Calculate batch size according to https://api.semanticscholar.org/CorpusID:270764838,
        # which assumes a sequence length of 2048. So adjust from there accordingly.
        assert sequence_length in {2048, 4096, 8192}
        seq_len_divisor = sequence_length // 2048

        num_params = self.get_model_config(
            size, sequence_length, self.get_tokenizer_config()
        ).num_non_embedding_params

        global_batch_size = 160 * (num_params / 108000000) ** (2 / 3)
        global_batch_size /= seq_len_divisor
        global_batch_size /= batch_size_divisor
        global_batch_size = round(global_batch_size)
        global_batch_size *= batch_size_divisor

        return sequence_length * global_batch_size

    def validate(self):
        """
        Validate the ladder configuration.

        :raises OLMoConfigurationError: If the ladder has any issues.
        """
        tokenizer = self.get_tokenizer_config()
        for size in ModelSize:
            target_size = int(size[:-1])
            if size.endswith("M"):
                target_size = target_size * 10**6
            elif size.endswith("B"):
                target_size = target_size * 10**9
            else:
                raise NotImplementedError(size)

            for sequence_length in (2048, 4096):
                model_config = self.get_model_config(size, sequence_length, tokenizer)

                # Make sure actual model size is close to target size.
                num_params = model_config.num_non_embedding_params
                if abs(num_params - target_size) / target_size > 0.05:
                    raise OLMoConfigurationError(
                        f"Model size of {num_params:,d} for sequence length {sequence_length} is "
                        f"too far from target size of {size}: {model_config}"
                    )

                self.get_optim_config(size, sequence_length)
                self.get_rank_microbatch_size(size, sequence_length, "H100")
                bz_tokens = self.get_global_batch_size(size, sequence_length)
                if bz_tokens % sequence_length != 0:
                    raise OLMoConfigurationError(
                        f"Batch size of {bz_tokens:,d} tokens for model size {size} "
                        f"must be divisible by the sequence length ({sequence_length})"
                    )
