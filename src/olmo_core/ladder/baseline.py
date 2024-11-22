from olmo_core.config import DType
from olmo_core.data import TokenizerConfig
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.nn.transformer import TransformerConfig, TransformerDataParallelConfig
from olmo_core.optim import AdamWConfig, OptimConfig, OptimGroupOverride

from .ladder import ModelLadder, ModelSize


class BaselineModelLadder(ModelLadder):
    """
    Baseline OLMo model ladder using the current recommended architecture.
    """

    MBZ_SIZES = {
        # TODO: may need to tune these
        # ===============================
        ModelSize.size_190M: 32 * 4096,
        ModelSize.size_370M: 32 * 4096,
        ModelSize.size_600M: 16 * 4096,
        ModelSize.size_760M: 16 * 4096,
        # ===============================
        ModelSize.size_1B: 8 * 4096,
        ModelSize.size_3B: 4 * 4096,
        ModelSize.size_7B: 2 * 4096,
        ModelSize.size_13B: 1 * 4096,
    }

    MODEL_OVERRIDES = {
        ModelSize.size_1B: dict(n_layers=16),  # need to scale down our actual 1B model
    }

    def get_model_config(
        self, size: ModelSize, sequence_length: int, tokenizer: TokenizerConfig
    ) -> TransformerConfig:
        del sequence_length
        return getattr(TransformerConfig, f"olmo_{size}")(
            vocab_size=tokenizer.padded_vocab_size(),
            compile=True,
            dp_config=TransformerDataParallelConfig(
                name=DataParallelType.fsdp, param_dtype=DType.bfloat16, reduce_dtype=DType.float32
            ),
            **self.MODEL_OVERRIDES.get(size, {}),
        )

    def get_optim_config(self, size: ModelSize, sequence_length: int) -> OptimConfig:
        # Calculate LR according to https://api.semanticscholar.org/CorpusID:270764838
        assert sequence_length in {2048, 4096}
        model_size = self.get_model_config(
            size, sequence_length, self.get_tokenizer_config()
        ).num_non_embedding_params
        lr = 0.0047 * (model_size / 108000000) ** (-1 / 3)
        if sequence_length == 4096:
            lr /= 4

        return AdamWConfig(
            lr=lr,
            weight_decay=0.1,
            betas=(0.9, 0.95),
            group_overrides=[
                OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0))
            ],
            fused=True,
        )

    def get_rank_microbatch_size(self, size: ModelSize, sequence_length: int, gpu_type: str) -> int:
        del sequence_length, gpu_type  # assuming we're running on 80GB GPUs
        return self.MBZ_SIZES[size]
