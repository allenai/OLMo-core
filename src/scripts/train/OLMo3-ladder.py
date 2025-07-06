from dataclasses import dataclass
import math
from typing import Any, ClassVar, Dict
from datetime import datetime

from olmo_core.config import DType
from olmo_core.data import NumpyDatasetConfig
from olmo_core.data.numpy_dataset import InstanceFilterConfig, VSLCurriculumConfig, VSLCurriculumType
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.internal.common import get_beaker_username, get_work_dir, CLUSTER_TO_GPU_TYPE
from olmo_core.internal.model_ladder import RunDuration, main
from olmo_core.io import join_path
from olmo_core.model_ladder import ModelLadder, ModelSize
from olmo_core.optim.scheduler import WSD
from olmo_core.nn.mup import MuPConfig, MuPOptimizerType, MuPScalingStrategy
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.nn.attention import SlidingWindowAttentionConfig
from olmo_core.optim import OptimConfig, OptimGroupOverride, OptimGroupOverride, SchedulerUnits, SkipStepAdamWConfig
from olmo_core.train import TrainerConfig
from olmo_core.train.train_module import (
    TransformerDataParallelConfig,
    TransformerDataParallelWrappingStrategy,
    TransformerTrainModuleConfig,
)
from olmo_core.train.callbacks import (
    CheckpointerCallback,
    CometCallback,
    WandBCallback,
    ConfigSaverCallback,
)
from olmo_core.train.callbacks.mup_coord_data import MuPCoordDataCallback

SEQUENCE_LENGTH = 8192
SAVE_INTERVAL = 1_000
EVAL_INTERVAL = 100


def optimal_wsd_lr(D: float, G: float, T_0: int, T: int):
    """ WSD LR implementation from @shana based on https://arxiv.org/abs/2501.18965v1 """
    def _curlT_1_wsd(D: float, T_0: int, T: int) -> float:
        return (D * D) / (T + T_0)

    def _curlT_2_wsd(G: float, T_0: int, T: int) -> float:
        # Eq. 21
        term_1 = (G * G) / (6 * (T + T_0)) * (2 * T + 4 * T_0 - 1 + 1 / (T + 1 - T_0))
        omega_2 = (2 * T - 2 * T_0 + 3 / (T - T_0) + 3 * sum(1 / i for i in range(1, T - T_0))) / (T - T_0 + 1)
        # @shanea mathing (maybe wrong)
        # omega_1_part_1 = 3 * _lambda_2(T_0, T)
        omega_1_part_1 = 3 * sum(1 / i for i in range(T + T_0 - 2, T - T_0 + 1, -2))
        # omega_1_part_2_factor1 = (1 / (T + 1 - T_0)) - (T - T_0 + 1)
        omega_1_part_2_factor1 = - (T - T_0) / (T + 1 - T_0)
        omega_1_part_2_factor2 = (T_0 - 1) / ((T - T_0 + 2) * (T + T_0))
        omega_1_part_2 = omega_1_part_2_factor1 * omega_1_part_2_factor2
        omega_1 = omega_1_part_1 + omega_1_part_2
        term_2 = (G * G) * (1 / 3) * (omega_1 + omega_2)
        return term_1 + term_2
    
    assert T_0 <= T

    return math.sqrt(_curlT_1_wsd(D, T_0, T) / _curlT_2_wsd(G, T_0, T))


@dataclass
class BaselineWSDModelLadder(ModelLadder):
    """ WSD OLMo model ladder """

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
        ModelSize.size_1B: dict(n_layers=16, hidden_size_multiple_of=1024), # matches OLMo 3 1B

        # TODO: Do we need to customize hidden_size_multiple_of for other sizes?
    }

    INTRA_DOCUMENT_MASKING = False # We use SkipStepOptimizer for this problem.
    INCLUDE_INSTANCE_FILTER = True

    def _get_model_config(self, *, size: ModelSize) -> TransformerConfig:
        # Note: There are some olmo2_mup_{size}, but these are for @shanea's
        # mup ladder. We can just use the existing sizes out-of-the-box
        model_config: TransformerConfig = getattr(TransformerConfig, f"olmo2_{size}")(
            vocab_size=self.tokenizer.padded_vocab_size(),
            init_seed=self.init_seed,
            **self.MODEL_OVERRIDES.get(size, {}),
        )

        base_size = ModelSize.size_7B
        base_model_config = getattr(TransformerConfig, f"olmo2_{base_size}")(
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
        config = getattr(TransformerConfig, f"olmo2_mup_{size}")(
            vocab_size=self.tokenizer.padded_vocab_size(),
            init_seed=self.init_seed,
            mup=mup_config,
            **self.MODEL_OVERRIDES.get(size, {}),
        )

        # TOOD: Do I need to update this on 
        config.block.attention.sliding_window = SlidingWindowAttentionConfig(
            force_full_attention_on_first_layer=False,
            force_full_attention_on_last_layer=True,
            pattern=[4096, 4096, 4096, -1]
        )
        config.block.attention.use_flash = True
        config.block.attention.use_head_qk_norm = True

        return config
    
    def get_lr(self, total_toks: int) -> float:
        # TODO: What are these magic numbers?
        D = 0.099
        G = 0.1

        gbz_toks = self.get_global_batch_size()
        steps = total_toks / gbz_toks

        optimal_lr = optimal_wsd_lr(D, G, int(steps - 25_000), steps)

        return optimal_lr

    def get_optim_config(self, run_duration: RunDuration) -> OptimConfig:
        total_toks = self.get_duration(run_duration)

        return SkipStepAdamWConfig(
            lr=self.get_lr(total_toks),
            weight_decay=0.1, # Follows hero run and mUP ladder
            betas=(0.9, 0.95),
            group_overrides=[
                OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0))
            ],
            compile=False,
        )

    def get_train_module_config(
        # self, common: CommonComponents, dp_world_size: int
        self,
        *,
        size: ModelSize, 
        run_duration: RunDuration, 
        gpu_type: str, 
        dp_world_size: int,
        cluster: str,
        dataset: NumpyDatasetConfig
    ) -> TransformerTrainModuleConfig:
        rank_mbz = self.get_rank_mbz(size, gpu_type, dp_world_size)

        if "B200" in CLUSTER_TO_GPU_TYPE.get(cluster, "unknown"):
            rank_mbz *= 2

        # In the hero run, we're have a 100B decay stage for 6T toks, or 1.67% of the full run is decay
        total_tokens = self.get_duration(run_duration)
        ANNEAL_TOKENS = total_tokens * 0.0167 # decay for final 1.67% of training

        return TransformerTrainModuleConfig(
            rank_microbatch_size=rank_mbz,
            max_sequence_length=dataset.effective_sequence_length,
            optim=self.get_optim_config(run_duration),
            compile_model=True,
            dp_config=TransformerDataParallelConfig(
                name=DataParallelType.hsdp,
                param_dtype=DType.bfloat16,
                reduce_dtype=DType.float32,
                wrapping_strategy=TransformerDataParallelWrappingStrategy.blocks,
            ),
            z_loss_multiplier=1e-5,
            max_grad_norm=1.0,
            scheduler=WSD(
                units=SchedulerUnits.steps,
                # warmup=2000, # from hero run
                warmup=round(self.model_size / self.get_global_batch_size()), # from wsd ladder # TODO: how much warmup is right?
                decay=(
                    int(ANNEAL_TOKENS / self.get_global_batch_size())
                ),  # In the hero run, we use 4x global batch size due to batch warmup
                decay_fraction=None,
            ),
        )

    def get_rank_microbatch_size(self, *, size: ModelSize, gpu_type: str) -> int:
        if gpu_type.lower() in ("mps", "cpu"):
            return 4096
        
        return self.MBZ_SIZES[size]

    def get_trainer_config(
        self,
        *,
        size: ModelSize,
        run_duration: RunDuration,
        gpu_type: str,
        dp_world_size: int,
        cluster: str,
    ) -> TrainerConfig:

        full_name = f"{self.name}-{size}-{run_duration}"
        run_name = f"{full_name}-{datetime.now().astimezone().strftime('%Y%m%dT%H%M%S%z')}"

        config = (
            TrainerConfig(
                save_folder=self.get_save_folder(size, run_duration),
                metrics_collect_interval=10,
                cancel_check_interval=10,
                max_duration=self.get_duration(run_duration),
                save_overwrite=True,
            )
            .with_callback(
                "config_saver", 
                ConfigSaverCallback()
            )
            .with_callback(
                "checkpointer",
                CheckpointerCallback(
                    save_interval=SAVE_INTERVAL,
                    ephemeral_save_interval=250,
                    save_async=True,
                ),
            )
            .with_callback(
                "comet",
                CometCallback(
                    name=run_name,
                    workspace="ai2-llm",
                    project=self.project,
                    enabled=False,
                    cancel_check_interval=5,
                ),
            )
            .with_callback(
                "wandb",
                WandBCallback(
                    name=run_name,
                    group=full_name,
                    entity="ai2-llm",
                    project=self.project,
                    enabled=True,
                    cancel_check_interval=5,
                ),
            )
            .with_recommended_evals(
                tokenizer=self.tokenizer, 
                sequence_length=self.sequence_length, 
                cluster=cluster, 
                task_set="full", 
                eval_interval=EVAL_INTERVAL
            )
        )

        config = config.with_callback(
            "mup_coord_data",
            MuPCoordDataCallback(
                enabled=True,
                collection_step=10,
            ),
        )

        # We are not using batch size warmup for ladder runs
        # # batch size warmup 
        # config.callbacks["batchwup"] = BatchSizeSchedulerCallback(
        #     batch_sizes=[GLOBAL_BATCH_SIZE, GLOBAL_BATCH_SIZE * 2, GLOBAL_BATCH_SIZE * 4],
        #     schedule=[
        #         Duration.tokens(0),
        #         Duration.tokens(167_772_160_000),
        #         Duration.tokens(503_316_480_000),
        #     ],
        # )

        return config

    def get_dataset_config(self) -> NumpyDatasetConfig:
        # Taken from OLMo 3 config
        return NumpyDatasetConfig.from_data_mix(
            mix=self.data_mix, # DataMix.OLMoE_mix_0824
            tokenizer=self.tokenizer, # tokenizer_config
            mix_base_dir=self.mix_base_dir,
            sequence_length=self.sequence_length,
            max_target_sequence_length=max(8192, self.sequence_length),
            min_sequence_length=min(256, self.sequence_length),
            max_sequence_length=max(8192, self.sequence_length),
            vsl_curriculum=VSLCurriculumConfig(
                name=VSLCurriculumType.grow_p2, num_cycles=8, balanced=False
            ),
            work_dir=get_work_dir(self.mix_base_dir),
            generate_doc_lengths=self.INTRA_DOCUMENT_MASKING,
            instance_filter_config=None
            if not self.INCLUDE_INSTANCE_FILTER
            else InstanceFilterConfig(
                repetition_max_period=13,
                repetition_min_period=1,
                repetition_max_count=32,
            ),
        )


def build_ladder(root_dir: str) -> BaselineWSDModelLadder:
    beaker_username = get_beaker_username()
    if beaker_username is not None:
        save_folder = str(join_path(root_dir, f"checkpoints/{beaker_username.lower()}/ladder"))
    else:
        save_folder = str(join_path(root_dir, "checkpoints/ladder"))
    return BaselineWSDModelLadder(
        name="OLMo3",
        project="OLMo3-model-ladder",
        mix_base_dir=root_dir,
        work_dir=get_work_dir(root_dir),
        save_folder=save_folder,
        sequence_length=SEQUENCE_LENGTH
    )


if __name__ == "__main__":
    main(build_ladder)
