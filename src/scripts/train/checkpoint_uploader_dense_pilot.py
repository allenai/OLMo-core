"""Dense training pilot for the standalone checkpoint uploader.

The default run trains for 99,992,207,360 tokens: 11,920 steps with an
8,388,608-token global batch. It saves step 0 and every 596 steps, including
the final step, producing 21 full resumable checkpoints.

Example::

    python src/scripts/train/checkpoint_uploader_dense_pilot.py launch \
        checkpoint-uploader-dense-control-20260903-r1 ai2/holmes

A shorter deletion pilot uses the same code with an explicit duration override::

    python src/scripts/train/checkpoint_uploader_dense_pilot.py launch \
        checkpoint-uploader-dense-delete-20260903-r1 ai2/holmes \
        --trainer.max_duration.value=2384
"""

from __future__ import annotations

import math
from datetime import datetime

from olmo_core.config import DType
from olmo_core.data import (
    DataMix,
    InstanceFilterConfig,
    NumpyDataLoaderConfig,
    NumpyFSLDatasetConfig,
    TokenizerConfig,
)
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.float8 import Float8Config
from olmo_core.internal.common import build_launch_config
from olmo_core.internal.cookbook import configure_required_callbacks
from olmo_core.internal.experiment import CliContext, ExperimentConfig, main
from olmo_core.launch.beaker import BeakerEnvSecret, BeakerEnvVar, BeakerWekaBucket
from olmo_core.nn.attention import (
    AttentionBackendName,
    AttentionConfig,
    AttentionType,
    GateConfig,
    GatedDeltaNetConfig,
    GateGranularity,
)
from olmo_core.nn.feed_forward import ActivationFunction, FeedForwardConfig
from olmo_core.nn.layer_norm import LayerNormConfig, LayerNormType
from olmo_core.nn.lm_head import LMHeadConfig, LMLossImplementation
from olmo_core.nn.transformer import (
    TransformerBlockConfig,
    TransformerBlockType,
    TransformerConfig,
    TransformerDataParallelWrappingStrategy,
)
from olmo_core.optim import OptimGroupOverride, SkipStepAdamWConfig, WSD
from olmo_core.train import Duration, TrainerConfig
from olmo_core.train.callbacks import (
    CheckpointReadyNotifierCallback,
    CheckpointerCallback,
    CheckpointRemovalStrategy,
    WandBCallback,
)
from olmo_core.train.train_module import (
    TransformerDataParallelConfig,
    TransformerTrainModuleConfig,
)

SEQUENCE_LENGTH = 8192
GLOBAL_BATCH_SIZE = 8_388_608
RANK_MICROBATCH_SIZE = 16 * SEQUENCE_LENGTH
MAX_STEPS = 11_920
CHECKPOINT_INTERVAL = 596
LEARNING_RATE = 8e-4
B300_BEAKER_IMAGE = "akshitab/olmo-core-tch2110cu130-2026-07-03"

CHECKPOINT_MOUNT = "/weka/olmo-3p5-checkpoints"
CHECKPOINT_ROOT = f"{CHECKPOINT_MOUNT}/checkpoints"
UPLOADER_INBOX = f"{CHECKPOINT_MOUNT}/uploader/control/inbox"
DATA_ROOT = "s3://ai2-llm"
WORK_DIR = "/tmp/dolma3p5-dataset-cache"


def build_dense_275m(vocab_size: int, backend: AttentionBackendName) -> TransformerConfig:
    """Build the existing dense 275M ladder architecture."""
    d_model = 640
    hidden_size = 8 * d_model
    n_layers = 10
    n_heads = 8
    head_dim = 128
    layer_norm = LayerNormConfig(
        name=LayerNormType.rms,
        eps=1e-6,
        bias=False,
        dtype=DType.float32,
    )
    feed_forward = FeedForwardConfig(
        hidden_size=hidden_size,
        bias=False,
        dtype=DType.float32,
        activation=ActivationFunction.silu,
    )
    recurrent_block = TransformerBlockConfig(
        name=TransformerBlockType.peri_norm,
        sequence_mixer=GatedDeltaNetConfig(
            n_heads=n_heads,
            n_v_heads=n_heads,
            head_dim=head_dim,
            dtype=DType.float32,
        ),
        feed_forward=feed_forward,
        layer_norm=layer_norm,
    )
    attention_block = TransformerBlockConfig(
        name=TransformerBlockType.peri_norm,
        sequence_mixer=AttentionConfig(
            name=AttentionType.default,
            n_heads=n_heads,
            n_kv_heads=n_heads,
            head_dim=head_dim,
            bias=False,
            rope=None,
            gate=GateConfig(
                granularity=GateGranularity.elementwise,
                full_precision=True,
            ),
            qk_norm=layer_norm,
            use_head_qk_norm=True,
            backend=backend,
            dtype=DType.float32,
        ),
        feed_forward=feed_forward,
        layer_norm=layer_norm,
    )
    return TransformerConfig(
        d_model=d_model,
        vocab_size=vocab_size,
        n_layers=n_layers,
        block=recurrent_block,
        block_overrides={4: attention_block, 9: attention_block},
        lm_head=LMHeadConfig(
            loss_implementation=LMLossImplementation.default,
            layer_norm=layer_norm,
            bias=False,
            dtype=DType.float32,
        ),
        dtype=DType.float32,
        embed_scale=math.sqrt(d_model),
        embedding_norm=LayerNormConfig(
            name=LayerNormType.rms,
            eps=1e-6,
            bias=False,
        ),
    )


def attention_backend(cluster: str) -> AttentionBackendName:
    if cluster in {"ai2/holmes", "ai2/titan"}:
        return AttentionBackendName.flash_4
    if cluster in {"ai2/jupiter", "ai2/ceres"}:
        return AttentionBackendName.flash_3
    return AttentionBackendName.flash_2


def build_experiment_config(cli_context: CliContext) -> ExperimentConfig:
    run_id = cli_context.run_name
    lineage_id = run_id
    run_name_with_timestamp = f"{run_id}-{datetime.now().astimezone().strftime('%Y%m%dT%H%M%S%z')}"
    tokenizer = TokenizerConfig.dolma2()
    model = build_dense_275m(tokenizer.padded_vocab_size(), attention_backend(cli_context.cluster))

    launch = build_launch_config(
        name=run_id,
        cmd=cli_context.remote_cmd,
        cluster=cli_context.cluster,
        workspace="ai2/scaling-ladders",
        budget="ai2/oe-other",
        beaker_image=B300_BEAKER_IMAGE,
        num_nodes=1,
        nccl_debug=False,
        step_timeout=None,
        step_soft_timeout=None,
    )
    launch.weka_buckets = [BeakerWekaBucket("olmo-3p5-checkpoints", CHECKPOINT_MOUNT)]
    launch.shared_filesystem = True
    launch.priority = "urgent"
    launch.min_runtime = "8h"
    # Keep enough retries for a transient Beaker failure without allowing a
    # deterministic pilot/configuration error to occupy a node all night.
    launch.retries = 2
    launch.follow = False
    launch.env_secrets = [
        BeakerEnvSecret(
            name="BEAKER_TOKEN",
            secret="jacobm_BEAKER_TOKEN",
            required=True,
        ),
    ]
    # The jacobm AWS credentials in scaling-ladders use the standard
    # ``default`` profile, while OLMo-core's generic launcher defaults to
    # ``S3``. Match the profile layout used by these workspace secrets.
    launch.env_vars.append(BeakerEnvVar(name="S3_PROFILE", value="default"))
    launch.google_credentials_secret = None
    launch.aws_config_secret = "jacobm_AWS_CONFIG"
    launch.aws_credentials_secret = "jacobm_AWS_CREDENTIALS"

    train_module = TransformerTrainModuleConfig(
        rank_microbatch_size=RANK_MICROBATCH_SIZE,
        max_sequence_length=SEQUENCE_LENGTH,
        optim=SkipStepAdamWConfig(
            lr=LEARNING_RATE,
            weight_decay=0.1,
            betas=(0.9, 0.95),
            group_overrides=[
                OptimGroupOverride(params=["embeddings.weight"], opts={"weight_decay": 0.0})
            ],
        ),
        scheduler=WSD(warmup_fraction=0.1, decay_fraction=0.1),
        compile_model=True,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.hsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
            wrapping_strategy=TransformerDataParallelWrappingStrategy.full,
        ),
        float8_config=Float8Config(enabled=False),
        z_loss_multiplier=1e-5,
        max_grad_norm=1.0,
    )

    dataset = NumpyFSLDatasetConfig.from_data_mix(
        DataMix.Dolma3p5_14t,
        tokenizer=tokenizer,
        mix_base_dir=DATA_ROOT,
        work_dir=WORK_DIR,
        sequence_length=SEQUENCE_LENGTH,
        max_target_sequence_length=SEQUENCE_LENGTH,
        generate_doc_lengths=False,
        instance_filter_config=InstanceFilterConfig(
            repetition_max_period=13,
            repetition_min_period=1,
            repetition_max_count=32,
        ),
    )
    data_loader = NumpyDataLoaderConfig(
        global_batch_size=GLOBAL_BATCH_SIZE,
        seed=928_543_231,
        num_workers=8,
        prefetch_factor=8,
        num_threads=4,
    )

    trainer = (
        TrainerConfig(
            save_folder=f"{CHECKPOINT_ROOT}/{run_id}",
            work_dir=WORK_DIR,
            save_overwrite=True,
            metrics_collect_interval=10,
            cancel_check_interval=100,
            max_duration=Duration.steps(MAX_STEPS),
        )
        .with_callbacks(configure_required_callbacks(run_name_with_timestamp))
        .with_callback(
            "checkpointer",
            CheckpointerCallback(
                save_interval=CHECKPOINT_INTERVAL,
                ephemeral_save_interval=None,
                pre_train_checkpoint=True,
                save_async=True,
                remove=CheckpointRemovalStrategy.never,
                max_checkpoints=None,
            ),
        )
        .with_callback(
            "checkpoint_ready_notifier",
            CheckpointReadyNotifierCallback(
                inbox_dir=UPLOADER_INBOX,
                run_id=run_id,
                lineage_id=lineage_id,
            ),
        )
        .with_callback(
            "wandb",
            WandBCallback(
                name=run_name_with_timestamp,
                group=run_id,
                entity="ai2-llm",
                project="checkpoint-uploader-pilot",
                # Beaker stdout is the authoritative pilot log. Keeping W&B
                # disabled also makes checkpoint resume independent of W&B
                # callback initialization order while async checkpoint metrics
                # are drained during startup.
                enabled=False,
                cancel_check_interval=100,
                tags=["checkpoint-uploader", "dense-275m", "dolma3p5", "wsd"],
            ),
        )
    )

    config = ExperimentConfig(
        run_name=run_id,
        launch=launch,
        model=model,
        train_module=train_module,
        trainer=trainer,
        dataset=dataset,
        data_loader=data_loader,
        init_seed=1337,
    )
    return config.merge(cli_context.overrides)


if __name__ == "__main__":
    main(config_builder=build_experiment_config)
