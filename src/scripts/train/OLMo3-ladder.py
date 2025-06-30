from dataclasses import dataclass
from typing import Any, ClassVar, Dict
from datetime import datetime

from olmo_core.config import DType
from olmo_core.data import NumpyDatasetConfig
from olmo_core.data.numpy_dataset import InstanceFilterConfig, VSLCurriculumConfig, VSLCurriculumType
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.internal.common import get_beaker_username, get_work_dir
from olmo_core.internal.common import CLUSTER_TO_GPU_TYPE
from olmo_core.internal.model_ladder import RunDuration, main
from olmo_core.io import join_path
from olmo_core.model_ladder import ModelLadder, ModelSize
from olmo_core.optim.scheduler import WSD
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.nn.attention import SlidingWindowAttentionConfig
from olmo_core.optim import OptimConfig, OptimGroupOverride
from olmo_core.optim import OptimGroupOverride, SchedulerUnits, SkipStepAdamWConfig
from olmo_core.train import Duration, TrainerConfig
from olmo_core.train.train_module import (
    TransformerDataParallelConfig,
    TransformerDataParallelWrappingStrategy,
    TransformerTrainModuleConfig,
)
from olmo_core.train.callbacks import (
    CheckpointerCallback,
    CometCallback,
    WandBCallback,
    ConfigSaverCallback
)


SEQUENCE_LENGTH = 8192
SAVE_INTERVAL = 1_000
EVAL_INTERVAL = 100


#### TODO: Fix all these (will this change ladder options?)
LR = 4.4e-5 * 2 # Based on 6T tokens with 100B anneal, don't forget to adjust when max duration or anneal length changes.


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
        config: TransformerConfig = getattr(TransformerConfig, f"olmo2_{size}")(
            vocab_size=self.tokenizer.padded_vocab_size(),
            init_seed=self.init_seed,
            **self.MODEL_OVERRIDES.get(size, {}),
        )

        config.block.attention.sliding_window = SlidingWindowAttentionConfig(
            force_full_attention_on_first_layer=False,
            force_full_attention_on_last_layer=True,
            pattern=[4096, 4096, 4096, -1]
        )
        config.block.attention.use_flash = True
        config.block.attention.use_head_qk_norm = True

        return config

    def get_optim_config(self) -> OptimConfig:
        return SkipStepAdamWConfig(
            lr=LR, # TODO: Where does this come from??
            weight_decay=0.033, # TODO: Why??
            betas=(0.9, 0.95),
            group_overrides=[
                OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0))
            ],
            compile=False,
        )

        # Old LR calculation::
        # Calculate LR according to https://api.semanticscholar.org/CorpusID:270764838
        assert self.sequence_length in {2048, 4096}
        lr = 0.0047 * (self.model_size / 108000000) ** (-1 / 3)
        if self.sequence_length == 4096:
            lr /= 4

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

        return TransformerTrainModuleConfig(
            rank_microbatch_size=rank_mbz,
            max_sequence_length=dataset.effective_sequence_length,
            optim=self.get_optim_config(),
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
                warmup=2000, # TODO: is this right?
                decay=0, # disable decay (we will launch separately)
                # decay=(
                #     int(ANNEAL_TOKENS / self.get_global_batch_size())
                # ),  # TODO (from OLMo 3 1B config): This isn't right because it doesn't take batchwup into account.
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
                sequence_length=SEQUENCE_LENGTH, 
                cluster=cluster, 
                task_set="full", 
                eval_interval=EVAL_INTERVAL
            )
        )

        # # batch size warmup # TODO: Figure out how to do this
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
