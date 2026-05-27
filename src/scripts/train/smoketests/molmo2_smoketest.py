"""
End-to-end multimodal training smoke test.

Drives the full VLM stack through the trainer:

  PixMo-Cap (local /weka cache)
    → MultimodalPreprocessor (tokens + multi-crop patches + loss_masks)
    → MultimodalCollator
    → MultimodalDataLoader
    → MultimodalTransformerTrainModule (FSDP when ``--fsdp``)
    → Trainer

Verifies that loss decreases over ~20 steps. The model is tiny (1M-param
LM + 2-layer ViT) so it runs in seconds on a single GPU.

Examples
--------

Local single-GPU (no FSDP)::

    torchrun --standalone --nproc-per-node=1 \\
        src/scripts/train/smoketests/molmo2_smoketest.py local-1gpu \\
        --save-folder=/tmp/mm-smoke

Local multi-GPU dry-run (prints config only)::

    python src/scripts/train/smoketests/molmo2_smoketest.py local-1gpu --dry-run

Beaker 2-GPU launch::

    python -m olmo_core.launch.beaker \\
        --gpus=2 \\
        --workspace=ai2/oe-encoder \\
        --budget=ai2/oe-other \\
        --cluster=ai2/jupiter \\
        --priority=urgent \\
        --weka=oe-training-default \\
        --shared-filesystem \\
        --allow-dirty \\
        -- src/scripts/train/smoketests/molmo2_smoketest.py beaker-2gpu \\
            --save-folder=/weka/oe-training-default/jasonr/mm-smoke \\
            --fsdp
"""

import argparse
import logging
import os
import sys
from dataclasses import dataclass
from typing import List, Optional

import rich

from olmo_core.config import Config, DType
from olmo_core.data.multimodal import (
    CropMode,
    ImagePreprocessorConfig,
    MultiCropPreprocessorConfig,
    MultimodalCollatorConfig,
    MultimodalDataLoaderConfig,
    MultimodalPreprocessorConfig,
    MultimodalTokenizerConfig,
    PixmoCapDatasetConfig,
    SyntheticMultimodalDatasetConfig,
)
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.distributed.utils import get_rank, get_world_size
from olmo_core.nn.transformer.config import TransformerConfig
from olmo_core.nn.vision import (
    MultimodalTransformerConfig,
    VisionBackboneConfig,
    VisionBackboneType,
    VisionConnectorConfig,
)
from olmo_core.optim import AdamWConfig, CosWithWarmup
from olmo_core.train import (
    Duration,
    TrainerConfig,
    prepare_training_environment,
    teardown_training_environment,
)
from olmo_core.train.callbacks import (
    CheckpointerCallback,
    ConfigSaverCallback,
    GPUMemoryMonitorCallback,
)
from olmo_core.train.train_module.multimodal import (
    MultimodalTransformerTrainModuleConfig,
)
from olmo_core.train.train_module.transformer.config import (
    TransformerDataParallelConfig,
)
from olmo_core.utils import seed_all

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------


@dataclass
class ExperimentConfig(Config):
    model: MultimodalTransformerConfig
    train_module: MultimodalTransformerTrainModuleConfig
    data_loader: MultimodalDataLoaderConfig
    trainer: TrainerConfig
    use_pixmo_cap: bool = False
    """If ``True``, use :class:`PixmoCapDataset` for training (requires the
    local /weka cache). Otherwise use a synthetic dataset."""
    train_dataset_size: int = 2048
    """Number of synthetic training examples (one epoch)."""
    init_seed: int = 12536


# ---------------------------------------------------------------------------
# Config builder
# ---------------------------------------------------------------------------


def build_config(opts, overrides: List[str]) -> ExperimentConfig:
    save_folder = opts.save_folder or f"/tmp/{opts.run_name}"

    mm_tok = MultimodalTokenizerConfig.dolma2()

    # Model: 1M-param LM + tiny 2-layer ViT — fast to forward/backward, enough
    # to exercise every code path.
    lm_cfg = TransformerConfig.olmo2_1M(vocab_size=mm_tok.padded_vocab_size(128))
    vis_cfg = VisionBackboneConfig(
        name=VisionBackboneType.openai,
        image_default_input_size=(56, 56),
        image_patch_size=14,
        image_emb_dim=64,
        image_num_heads=4,
        image_num_key_value_heads=4,
        image_num_layers=2,
        image_head_dim=16,
        image_mlp_dim=128,
        image_num_pos=17,  # 1 CLS + 16 patches (4x4)
        image_norm_eps=1e-5,
    )
    conn_cfg = VisionConnectorConfig.from_vision_backbone(
        vis_cfg, output_dim=lm_cfg.d_model, mlp_hidden_size=64
    )
    model_config = MultimodalTransformerConfig(
        lm=lm_cfg,
        vision=vis_cfg,
        connector=conn_cfg,
        image_patch_token_id=mm_tok.image_patch_id,
    )

    # Data: multicrop=resize so we get a single crop per image (matches what
    # an end-to-end Molmo Stage 1 caption run looks like).
    multicrop_cfg = MultiCropPreprocessorConfig(
        base_image_input_size=(56, 56),
        crop_mode=CropMode.resize,
        pool_h=2,
        pool_w=2,
        image_preprocessor=ImagePreprocessorConfig(patch_size=14),
    )
    preprocessor_cfg = MultimodalPreprocessorConfig(
        tokenizer=mm_tok,
        multicrop=multicrop_cfg,
        max_sequence_length=opts.sequence_length,
    )
    data_loader_config = MultimodalDataLoaderConfig(
        preprocessor=preprocessor_cfg,
        collator=MultimodalCollatorConfig(tokenizer=mm_tok),
        global_batch_size=opts.global_batch_size,
        num_workers=opts.num_workers,
        seed=0,
        work_dir=os.path.join(save_folder, "data-work-dir"),
    )

    # Train module: FSDP iff the user asked for it (default off so single-GPU
    # local runs work without a distributed env).
    dp_config: Optional[TransformerDataParallelConfig] = None
    if opts.fsdp:
        dp_config = TransformerDataParallelConfig(
            name=DataParallelType.fsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
        )
    train_module_config = MultimodalTransformerTrainModuleConfig(
        rank_microbatch_size=opts.sequence_length,  # one instance per microbatch
        max_sequence_length=opts.sequence_length,
        optim=AdamWConfig(lr=1e-3, weight_decay=0.0),
        max_grad_norm=1.0,
        scheduler=CosWithWarmup(warmup_steps=5),
        dp_config=dp_config,
    )

    # Trainer: short by design.
    trainer_config = (
        TrainerConfig(
            save_folder=save_folder,
            save_overwrite=True,
            metrics_collect_interval=5,
            cancel_check_interval=5,
            max_duration=Duration.steps(opts.max_steps),
        )
        .with_callback("gpu_monitor", GPUMemoryMonitorCallback())
        .with_callback("config_saver", ConfigSaverCallback())
        .with_callback(
            "checkpointer",
            CheckpointerCallback(save_interval=opts.max_steps, save_async=False),
        )
    )

    cfg = ExperimentConfig(
        model=model_config,
        train_module=train_module_config,
        data_loader=data_loader_config,
        trainer=trainer_config,
        use_pixmo_cap=opts.pixmo_cap,
        train_dataset_size=opts.train_dataset_size,
        init_seed=12536,
    )
    cfg = cfg.merge(overrides)
    return cfg


# ---------------------------------------------------------------------------
# Train entry point
# ---------------------------------------------------------------------------


def train(config: ExperimentConfig):
    if get_rank() == 0:
        rich.print(config)

    seed_all(config.init_seed)

    mm_tok = config.data_loader.preprocessor.tokenizer
    hf_tok = mm_tok.load_hf_tokenizer()

    # Build the model on meta when using FSDP so init_weights can materialize.
    init_device = "meta" if config.train_module.dp_config is not None else "cpu"
    model = config.model.build(init_device=init_device)

    # Build train module — handles parallelization + materialize + optimizer.
    train_module = config.train_module.build(model)

    # Build the training source.
    if config.use_pixmo_cap:
        source = PixmoCapDatasetConfig(limit=config.train_dataset_size).build()
        log.info(f"Training on PixMo-Cap (up to {config.train_dataset_size} examples)")
    else:
        from olmo_core.data.multimodal import SyntheticMultimodalDataset

        source = SyntheticMultimodalDataset(
            SyntheticMultimodalDatasetConfig(
                n_examples=config.train_dataset_size, image_size=(56, 56), seed=1
            )
        )
        log.info(f"Training on synthetic data ({config.train_dataset_size} examples)")

    # Build training data loader (rank-aware).
    data_loader = config.data_loader.build(
        source,
        hf_tok,
        dp_world_size=get_world_size(train_module.dp_process_group),
        dp_rank=get_rank(train_module.dp_process_group),
    )

    # Build trainer.
    trainer = config.trainer.build(train_module, data_loader)

    # Stash the full config for checkpointer.
    config_dict = config.as_config_dict()
    from typing import cast as _cast

    _cast(ConfigSaverCallback, trainer.callbacks["config_saver"]).config = config_dict

    trainer.fit()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        prog=sys.argv[0],
        usage=f"python {sys.argv[0]} RUN_NAME [OPTIONS...] [OVERRIDES...]",
        description="Multimodal training smoke test.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("run_name", type=str, help="Name of the run (used in save folder).")
    parser.add_argument("--save-folder", type=str, default=None)
    parser.add_argument("--sequence-length", type=int, default=128, help="Max sequence length.")
    parser.add_argument(
        "--global-batch-size", type=int, default=2, help="Examples per batch (global)."
    )
    parser.add_argument(
        "--num-workers", type=int, default=0, help="DataLoader worker processes per rank."
    )
    parser.add_argument("--max-steps", type=int, default=20, help="Total training steps.")
    parser.add_argument(
        "--train-dataset-size",
        type=int,
        default=2048,
        help="Number of examples in the (synthetic or PixMo-Cap) training source.",
    )
    parser.add_argument(
        "--fsdp",
        action="store_true",
        help="Enable FSDP. Required for multi-GPU; defaults off for single-GPU.",
    )
    parser.add_argument(
        "--pixmo-cap",
        action="store_true",
        help="Train on local PixMo-Cap cache instead of synthetic data.",
    )
    parser.add_argument("--dry-run", action="store_true")
    opts, overrides = parser.parse_known_args()
    return opts, overrides


def main():
    opts, overrides = parse_args()
    config = build_config(opts, overrides)
    if opts.dry_run:
        rich.print(config)
        return
    prepare_training_environment()
    try:
        train(config)
    finally:
        teardown_training_environment()


if __name__ == "__main__":
    main()
