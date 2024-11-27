from dataclasses import dataclass
from typing import Any, ClassVar, Dict

from olmo_core.config import DType
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.internal.common import get_beaker_username, get_work_dir
from olmo_core.internal.model_ladder import main
from olmo_core.io import join_path
from olmo_core.model_ladder import ModelLadder, ModelSize
from olmo_core.nn.transformer import TransformerConfig, TransformerDataParallelConfig
from olmo_core.optim import AdamWConfig, OptimConfig, OptimGroupOverride


@dataclass
class BaselineModelLadder(ModelLadder):
    """
    Baseline OLMo model ladder using the current recommended architecture.
    """

    MBZ_SIZES: ClassVar[Dict[ModelSize, int]] = {
        # TODO: may need to tune these
        # ===============================
        ModelSize.size_190M: 16 * 4096,
        ModelSize.size_370M: 16 * 4096,
        ModelSize.size_600M: 16 * 4096,
        ModelSize.size_760M: 16 * 4096,
        # ===============================
        ModelSize.size_1B: 8 * 4096,
        ModelSize.size_3B: 4 * 4096,
        ModelSize.size_7B: 2 * 4096,
        ModelSize.size_13B: 1 * 4096,
    }

    MODEL_OVERRIDES: ClassVar[Dict[ModelSize, Dict[str, Any]]] = {
        ModelSize.size_1B: dict(n_layers=16),  # need to scale down our actual 1B model
    }

    def get_model_config(self, *, size: ModelSize) -> TransformerConfig:
        return getattr(TransformerConfig, f"olmo2_{size}")(
            vocab_size=self.tokenizer.padded_vocab_size(),
            init_seed=self.init_seed,
            compile=True,
            dp_config=TransformerDataParallelConfig(
                name=DataParallelType.fsdp, param_dtype=DType.bfloat16, reduce_dtype=DType.float32
            ),
            **self.MODEL_OVERRIDES.get(size, {}),
        )

    def get_optim_config(self, *, size: ModelSize) -> OptimConfig:
        # Calculate LR according to https://api.semanticscholar.org/CorpusID:270764838
        assert self.sequence_length in {2048, 4096}
        lr = 0.0047 * (size.num_params / 108000000) ** (-1 / 3)
        if self.sequence_length == 4096:
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

    def get_rank_microbatch_size(self, *, size: ModelSize, gpu_type: str) -> int:
        assert "h100" in gpu_type.lower()
        return self.MBZ_SIZES[size]


def build_ladder(root_dir: str) -> BaselineModelLadder:
    save_folder = str(join_path(root_dir, f"checkpoints/{get_beaker_username().lower()}/ladder"))
    return BaselineModelLadder(
        name="OLMo2",
        project="OLMo2-model-ladder",
        mix_base_dir=root_dir,
        work_dir=get_work_dir(root_dir),
        save_folder=save_folder,
    )


if __name__ == "__main__":
    main(build_ladder)
