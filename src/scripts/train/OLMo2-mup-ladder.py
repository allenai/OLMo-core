from dataclasses import dataclass
from typing import Any, ClassVar, Dict, List

from olmo_core.config import DType
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.float8 import Float8Config
from olmo_core.float8.ao import AOFloat8LinearConfig
from olmo_core.internal.common import get_beaker_username, get_work_dir
from olmo_core.internal.model_ladder import RunDuration, main
from olmo_core.io import join_path
from olmo_core.model_ladder import ModelLadder, ModelSize
from olmo_core.nn.attention import SlidingWindowAttentionConfig
from olmo_core.nn.mup import MuPConfig, MuPOptimizerType, MuPScalingStrategy
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.optim import OptimConfig, OptimGroupOverride
from olmo_core.optim.adamw import SkipStepAdamWConfig
from olmo_core.optim.scheduler import WSD
from olmo_core.train.callbacks.mup_coord_data import MuPCoordDataCallback
from olmo_core.train.config import TrainerConfig
from olmo_core.train.train_module import (
    TransformerDataParallelConfig,
    TransformerTrainModuleConfig,
)


@dataclass
class BaselineModelLadder(ModelLadder):
    """
    Baseline OLMo model ladder using the current recommended architecture.
    """

    SUPPORTED_MODEL_SIZES: ClassVar[List[ModelSize]] = [
        ModelSize.size_60M,
        ModelSize.size_370M,
        ModelSize.size_970M,
        ModelSize.size_7B,
    ]

    MBZ_SIZES: ClassVar[Dict[ModelSize, int]] = {
        # TODO: may need to tune these
        # ===============================
        ModelSize.size_60M: 16 * 4096,
        ModelSize.size_370M: 4 * 4096,
        # ===============================,
        ModelSize.size_970M: 8 * 4096,
        ModelSize.size_7B: 4 * 4096,
    }

    MODEL_OVERRIDES: ClassVar[Dict[ModelSize, Dict[str, Any]]] = {
        # ModelSize.size_970M: dict(n_layers=16),  # need to scale down our actual 1B model
    }

    def _get_model_config(self, *, size: ModelSize) -> TransformerConfig:
        if size not in self.SUPPORTED_MODEL_SIZES:
            raise OLMoConfigurationError(
                f"Size {size} not supported by this muP ladder (supported sizes: {self.SUPPORTED_MODEL_SIZES})."
            )

        model_config: TransformerConfig = getattr(TransformerConfig, f"olmo2_mup_{size}")(
            vocab_size=self.tokenizer.padded_vocab_size(),
            init_seed=self.init_seed,
            **self.MODEL_OVERRIDES.get(size, {}),
        )

        base_size = ModelSize.size_7B
        base_model_config = getattr(TransformerConfig, f"olmo2_mup_{base_size}")(
            vocab_size=self.tokenizer.padded_vocab_size(),
            init_seed=self.init_seed,
            **self.MODEL_OVERRIDES.get(base_size, {}),
        )
        mup_width_scalings = model_config.get_mup_width_scalings(base_model_config)
        mup_config = MuPConfig(
            MuPOptimizerType.adam_coupled_wd,
            scaling_strategy=MuPScalingStrategy.constant_inputs,
            width_scalings=mup_width_scalings,
        )

        # Need to reconstruct config to pass in muP config
        model_config = getattr(TransformerConfig, f"olmo2_mup_{size}")(
            vocab_size=self.tokenizer.padded_vocab_size(),
            init_seed=self.init_seed,
            mup=mup_config,
            **self.MODEL_OVERRIDES.get(size, {}),
        )

        model_config.block.attention.sliding_window = SlidingWindowAttentionConfig(
            force_full_attention_on_first_layer=False,
            force_full_attention_on_last_layer=True,
            # NOTE: 4097 instead of 4096 to reproduce with the off-by-one bug.
            pattern=[4097, 4097, 4097, -1],
        )
        model_config.block.attention.use_flash = True
        model_config.block.attention.use_head_qk_norm = True
        return model_config

    def get_optim_config(self) -> OptimConfig:
        # Calculate LR according to https://api.semanticscholar.org/CorpusID:270764838
        assert self.sequence_length in {2048, 4096, 8192}
        lr = 0.0047 * (self.model_size / 108000000) ** (-1 / 3)
        if self.sequence_length == 4096:
            lr /= 4
        elif self.sequence_length == 8192:
            lr /= 16

        return SkipStepAdamWConfig(
            lr=lr,
            weight_decay=0.1,
            betas=(0.9, 0.95),
            group_overrides=[
                OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0))
            ],
            compile=False,
            foreach=True,
            step_increment_bugfix=False,
        )

    def get_train_module_config(
        self, *, size: ModelSize, run_duration: RunDuration, gpu_type: str, dp_world_size: int
    ) -> TransformerTrainModuleConfig:
        config = super().get_train_module_config(
            size=size, run_duration=run_duration, gpu_type=gpu_type, dp_world_size=dp_world_size
        )
        config.compile_model = True
        config.dp_config = TransformerDataParallelConfig(
            name=DataParallelType.hsdp, param_dtype=DType.bfloat16, reduce_dtype=DType.float32
        )
        config.scheduler = WSD(
            warmup_steps=round(self.model_size / self.get_global_batch_size()), decay_fraction=0.25
        )
        config.float8_config = Float8Config(
            enabled=True,
            ao=AOFloat8LinearConfig(
                enable_fsdp_float8_all_gather=True,
                force_recompute_fp8_weight_in_bwd=True,
                round_scales_to_power_of_2=True,
            ),
        )
        config.z_loss_multiplier = 1e-5
        return config

    def get_trainer_config(
        self,
        *,
        size: ModelSize,
        run_duration: RunDuration,
        gpu_type: str,
        dp_world_size: int,
        cluster: str,
    ) -> TrainerConfig:
        config = super().get_trainer_config(
            size=size,
            run_duration=run_duration,
            gpu_type=gpu_type,
            dp_world_size=dp_world_size,
            cluster=cluster,
        )

        config = config.with_callback(
            "mup_coord_data",
            MuPCoordDataCallback(
                enabled=True,
                collection_step=10,
            ),
        )

        return config

    def get_global_batch_size(self) -> int:
        """
        Get the global batch size in tokens for a given model size.

        :param size: The target model size.
        """
        # Let's avoid global batch size making results harder to interpret, for now.
        return 4096 * 1024

    def get_rank_microbatch_size(self, *, size: ModelSize, gpu_type: str) -> int:
        if gpu_type.lower() in ("mps", "cpu"):
            return 4096
        else:
            # assert "h100" in gpu_type.lower()
            return self.MBZ_SIZES[size]


def build_ladder(root_dir: str) -> BaselineModelLadder:
    beaker_username = get_beaker_username()
    if beaker_username is not None:
        save_folder = str(join_path(root_dir, f"checkpoints/{beaker_username.lower()}/ladder"))
    else:
        save_folder = str(join_path(root_dir, "checkpoints/ladder"))
    return BaselineModelLadder(
        name="OLMo2",
        project="OLMo2-mup-ladder",
        mix_base_dir=root_dir,
        work_dir=get_work_dir(root_dir),
        save_folder=save_folder,
        sequence_length=8192,
        beaker_workspace="ai2/OLMo-mup",
        intra_document_masking=True,
    )


if __name__ == "__main__":
    main(build_ladder)
